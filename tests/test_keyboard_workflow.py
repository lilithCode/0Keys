import argparse
from contextlib import ExitStack
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

import invisible_keyboard
import user_key_calibration
from helper_audio import TapEvent
from helper_classifier import UserKeyboardProfile, profile_context
from helper_fusion import FingerCandidate, FusionResult
from helper_keyboard import KeyboardCalibration, KeyboardLayout
from helper_vision import HandSnapshot


class KeyboardWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.calibration = KeyboardCalibration(((0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)))
        self.calibration.save(self.root / "calibration.json")
        (self.root / "model.task").touch()
        self.profile_path = self.root / "profile.json"
        self.now = 9.0
        self.keys = {key.name: key for key in KeyboardLayout().keys}

    def candidate(self, name="a", score=0.8):
        key = self.keys[name]
        x, y = key.center
        return FingerCandidate(1, "Left", 20, "pinky", key, score, x, y, 0.0, 0.0)

    def arguments(self, validate=False):
        return argparse.Namespace(
            model=str(self.root / "model.task"), calibration=str(self.root / "calibration.json"),
            profile=str(self.profile_path), camera=0, microphone=None,
            camera_offset_ms=70.0, no_mirror=False, width=640, height=480,
            samples_per_key=2, validation_rounds=1, key_set="common", keys=["a"],
            validate=validate, fresh=False, sensitivity="balanced",
            threshold_db=None, rise_db=None, crest_db=None, motion_threshold=None,
            no_session_log=True, log_dir=str(self.root / "logs"),
        )

    def run_app(self, module, args, controls, candidates, tap_frames=None):
        def read():
            self.now += 1.0
            return True, np.zeros((480, 640, 3), dtype=np.uint8)

        camera = MagicMock()
        camera.read.side_effect = read
        tracker = MagicMock()
        tracker.process.side_effect = lambda frame, timestamp: HandSnapshot(timestamp, ())
        microphone = MagicMock()
        microphone.drain.side_effect = lambda: (
            [SimpleNamespace(samples=np.zeros(256), first_sample_time=self.now - 0.4)]
            if tap_frames is None or int(self.now - 9) in tap_frames else []
        )
        detector = MagicMock()
        detector.calibrated = True
        detector.process_block.side_effect = lambda samples, timestamp: TapEvent(timestamp, 20.0, -10.0, -30.0)
        texts = []
        logger = MagicMock()
        with ExitStack() as stack:
            stack.enter_context(patch.object(module, "parse_args", return_value=args))
            stack.enter_context(patch.object(module.time, "perf_counter", side_effect=lambda: self.now))
            stack.enter_context(patch.object(module.cv2, "VideoCapture", return_value=camera))
            for name in ("namedWindow", "resizeWindow", "imshow", "destroyAllWindows"):
                stack.enter_context(patch.object(module.cv2, name))
            stack.enter_context(patch.object(module.cv2, "waitKey", side_effect=controls))
            stack.enter_context(patch.object(module, "MediaPipeHandTracker")).return_value.__enter__.return_value = tracker
            stack.enter_context(patch.object(module, "MicrophoneInput", return_value=microphone))
            stack.enter_context(patch.object(module, "AudioTapDetector", return_value=detector))
            stack.enter_context(patch.object(module, "draw_snapshot"))
            stack.enter_context(patch.object(module, "SessionLogger", return_value=logger))
            stack.enter_context(patch.object(
                module.TapFingerFusion, "select",
                side_effect=lambda tap, history: FusionResult(tap, candidates[0], "Key accepted", candidates),
            ))
            if module is invisible_keyboard:
                def panel(frame, composer, status, detail):
                    texts.append(composer.text)
                    return frame
                stack.enter_context(patch.object(module, "draw_text_panel", side_effect=panel))
            result = module.main()
        self.assertEqual(result, 0)
        camera.release.assert_called_once()
        return texts, logger

    def test_training_saves_only_confirmed_sample(self):
        self.run_app(user_key_calibration, self.arguments(), [ord(" "), ord("q")], (self.candidate(),))
        profile = UserKeyboardProfile.load(self.profile_path)
        self.assertEqual(profile.count("a"), 1)
        self.assertEqual(profile.samples[0].timestamp, 9.6)

    def test_rejected_training_review_never_writes_a_profile(self):
        self.run_app(user_key_calibration, self.arguments(), [ord("r"), ord("q")], (self.candidate(),))
        self.assertFalse(self.profile_path.exists())

    def test_validation_does_not_change_profile_and_counts_wrong_key(self):
        context = profile_context(self.calibration, 0, 70.0)
        from helper_classifier import PersonalKeyClassifier
        sample = PersonalKeyClassifier.sample_from_candidate("a", self.candidate(), 20.0, 1.0)
        UserKeyboardProfile([sample] * 5, context=context).save(self.profile_path)
        original = self.profile_path.read_bytes()
        _, logger = self.run_app(user_key_calibration, self.arguments(validate=True),
                                 [ord(" "), ord("q")], (self.candidate("p"),))
        self.assertEqual(original, self.profile_path.read_bytes())
        report = logger.write.call_args.kwargs
        self.assertEqual(report["attempted"], 1)
        self.assertEqual(report["correct"], 0)
        self.assertEqual(report["results"], [{"expected": "a", "predicted": "p"}])

    def test_ambiguous_tap_does_not_change_text(self):
        texts, _ = self.run_app(invisible_keyboard, self.arguments(), [ord("q")],
                                (self.candidate("a"), self.candidate("j")))
        self.assertEqual(texts, [""])

    def test_clear_cancels_pending_correction(self):
        texts, _ = self.run_app(invisible_keyboard, self.arguments(),
                                [ord("c"), ord("1"), ord("q")], (self.candidate(),), tap_frames={1})
        self.assertEqual(texts[:2], ["a", ""])
        self.assertFalse(self.profile_path.exists())

    def test_uncertain_choice_inserts_once_and_saves_actual_tap(self):
        texts, _ = self.run_app(invisible_keyboard, self.arguments(),
                                [ord("2"), ord("2"), ord("q")],
                                (self.candidate("a"), self.candidate("j")), tap_frames={1})
        self.assertEqual(texts, ["", "j", "j"])
        profile = UserKeyboardProfile.load(self.profile_path)
        self.assertEqual(len(profile.samples), 1)
        self.assertEqual(profile.samples[0].key_name, "j")
        self.assertEqual(profile.samples[0].timestamp, 9.6)

    def test_old_profile_is_backed_up_before_retraining(self):
        UserKeyboardProfile(version=1).save(self.profile_path)
        original = self.profile_path.read_bytes()
        self.run_app(user_key_calibration, self.arguments(), [ord(" "), ord("q")], (self.candidate(),))
        backups = list(self.root.glob("profile.backup_*.json"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), original)
        self.assertEqual(UserKeyboardProfile.load(self.profile_path).version, 2)


if __name__ == "__main__":
    unittest.main()

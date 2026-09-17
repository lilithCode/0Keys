import argparse
from dataclasses import replace
from contextlib import ExitStack
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

import press_keyboard
from helper_audio import TapEvent
from helper_press import PressProfile, stable_pose
from helper_vision import HandSnapshot
from test_helper_press import make_hand, profile_for, right_hand, right_profile


class PressWorkflowTests(unittest.TestCase):
    def run_app(self, directory, setup=False, controls=None, sound=True,
                hand_option="left", detected_hand="Left", panels=None, existing_left=False,
                voice_commands=False, voice_missing=False, phone=False):
        root = Path(directory)
        (root / "model.task").touch()
        profile_path = root / "press.json"
        both = hand_option == "both"
        identity = "usb:test-device:rear:r0" if phone else 0
        if not setup or existing_left:
            replace(profile_for(), camera=identity).save(profile_path)
        right_path = root / "right.json"
        if both and not setup:
            replace(right_profile(), camera=identity).save(right_path)
        args = argparse.Namespace(
            camera=0, microphone=None, hand=hand_option, model=str(root / "model.task"),
            profile=str(profile_path), calibrate=setup and not existing_left, no_mirror=False,
            camera_offset_ms=0.0, threshold_db=11.0,
            right_profile=str(right_path),
            phone=phone, rotation=0, hover_start=False,
            no_voice=not (voice_commands or voice_missing), voice_model=str(root / "voice"),
        )
        clock = [0.0]
        frame_index = [0]
        pose_index = [0]
        armed = [-1]
        voice_enabled = [False]
        voice = MagicMock(error="", feedback="Listening. Say ready, then pause")

        def listening(enabled):
            voice_enabled[0] = enabled

        def spoken_record():
            if voice_enabled[0] and armed[0] != pose_index[0]:
                armed[0] = pose_index[0]
                voice_enabled[0] = False
                return True
            return False

        voice.set_listening.side_effect = listening
        voice.pop_record.side_effect = spoken_record
        camera = MagicMock()
        camera.info = {"serial": "test-device", "model": "Test phone"}

        def read():
            clock[0] += 0.06
            camera.timestamp = clock[0]
            frame_index[0] += 1
            return True, np.zeros((480, 640, 3), np.uint8)

        def hand():
            if setup:
                side = ("Right" if existing_left or pose_index[0] >= 5 else "Left") if both else detected_hand
                order = ([(), (20,), (16,), (12,), (8,), ()] if detected_hand == "Left"
                         else [(), (8,), (12,), (16,), (20,), ()])
                if both:
                    order = ([(), (8,), (12,), (16,), (20,)] if side == "Right"
                             else [(), (20,), (16,), (12,), (8,)])
                    lifted = order[pose_index[0] % 5]
                    return right_hand(lifted) if side == "Right" else make_hand(lifted)
                return replace(make_hand(order[pose_index[0]]), handedness=detected_hand)
            return replace(make_hand((8,) if frame_index[0] in (4, 5) else ()), handedness=detected_hand)

        def capture_pose(hands):
            result = stable_pose(hands)
            pose_index[0] += 1
            return result

        def setup_control(_delay):
            if pose_index[0] == (10 if both and not existing_left else 5):
                return ord("q")
            if not voice_commands and armed[0] != pose_index[0]:
                armed[0] = pose_index[0]
                return ord(" ")
            if frame_index[0] > 500:
                raise AssertionError("Setup did not complete")
            return -1

        camera.read.side_effect = read
        tracker = MagicMock()
        tracker.process.side_effect = lambda frame, timestamp: HandSnapshot(
            timestamp, (hand(), right_hand()) if both and not setup else (hand(),))
        detector = MagicMock(calibrated=True)
        detector.process_block.side_effect = lambda samples, timestamp: (
            TapEvent(timestamp, 20.0, -10.0, -30.0) if sound and not setup and frame_index[0] == 6 else None
        )
        microphone = MagicMock(dropped_blocks=0)
        microphone.drain.side_effect = lambda: [SimpleNamespace(samples=np.zeros(256), first_sample_time=clock[0])]
        logger = MagicMock()
        with ExitStack() as stack:
            voice_factory = stack.enter_context(patch.object(press_keyboard, "VoiceRecordControl", return_value=voice))
            if voice_missing:
                voice_factory.side_effect = RuntimeError("Voice model missing")
            stack.enter_context(patch.object(press_keyboard, "parse_args", return_value=args))
            stack.enter_context(patch.object(press_keyboard.time, "perf_counter", side_effect=lambda: clock[0]))
            stack.enter_context(patch.object(press_keyboard.cv2, "VideoCapture", return_value=camera))
            stack.enter_context(patch.object(press_keyboard, "PhoneCamera", return_value=camera))
            for name in ("namedWindow", "resizeWindow", "imshow", "destroyAllWindows"):
                stack.enter_context(patch.object(press_keyboard.cv2, name))
            stack.enter_context(patch.object(press_keyboard.cv2, "waitKey",
                side_effect=controls if controls is not None else setup_control if setup else [-1] * 14 + [ord("q")]))
            stack.enter_context(patch.object(press_keyboard, "MediaPipeHandTracker")).return_value.__enter__.return_value = tracker
            stack.enter_context(patch.object(press_keyboard, "MicrophoneInput", return_value=microphone))
            stack.enter_context(patch.object(press_keyboard, "AudioTapDetector", return_value=detector))
            stack.enter_context(patch.object(press_keyboard, "SessionLogger", return_value=logger))
            stack.enter_context(patch.object(press_keyboard, "draw_snapshot"))
            stack.enter_context(patch.object(press_keyboard, "stable_pose", side_effect=capture_pose))
            if panels is not None:
                def panel(frame, title, instructions, status, typed, states, fps, audio_status=""):
                    panels.append((title, status, states))
                    return frame
                stack.enter_context(patch.object(press_keyboard, "draw_panel", side_effect=panel))
            result = press_keyboard.main()
        self.assertEqual(result, 0)
        camera.release.assert_called_once()
        if voice_commands:
            voice.close.assert_called_once()
            if not setup:
                voice.pop_record.assert_not_called()
                self.assertTrue(all(call.args == (False,) for call in voice.set_listening.call_args_list))
        return logger, profile_path

    def test_phone_sensor_loop_uses_saved_phone_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            logger, path = self.run_app(directory, phone=True)
            self.assertEqual(PressProfile.load(path).camera, "usb:test-device:rear:r0")
        events = [call.kwargs for call in logger.write.call_args_list if call.args[0] == "press"]
        self.assertEqual([event["key"] for event in events], ["f"])

    def test_voice_records_all_ten_poses_without_space(self):
        with tempfile.TemporaryDirectory() as directory:
            logger, path = self.run_app(directory, setup=True, hand_option="both", voice_commands=True)
            self.assertEqual(PressProfile.load(path).handedness, "Left")
            self.assertEqual(PressProfile.load(Path(directory) / "right.json").handedness, "Right")
        commands = [call for call in logger.write.call_args_list if call.args[0] == "press_setup_voice_record"]
        self.assertEqual(len(commands), 10)

    def test_voice_is_disabled_during_typing(self):
        with tempfile.TemporaryDirectory() as directory:
            self.run_app(directory, voice_commands=True)

    def test_missing_voice_model_keeps_space_setup_available(self):
        panels = []
        with tempfile.TemporaryDirectory() as directory:
            _, path = self.run_app(directory, setup=True, voice_missing=True, panels=panels)
            self.assertEqual(PressProfile.load(path).handedness, "Left")
        self.assertTrue(any("Voice unavailable" in states for _, _, states in panels))

    def test_all_five_setup_poses_save_a_usable_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            logger, path = self.run_app(directory, setup=True)
            profile = PressProfile.load(path)
        self.assertEqual([finger.key for finger in profile.fingers], ["a", "s", "d", "f"])
        self.assertIn("press_setup_saved", [call.args[0] for call in logger.write.call_args_list])

    def test_both_hands_setup_saves_separate_profiles(self):
        with tempfile.TemporaryDirectory() as directory:
            logger, path = self.run_app(directory, setup=True, hand_option="both")
            self.assertEqual(PressProfile.load(path).handedness, "Left")
            self.assertEqual(PressProfile.load(Path(directory) / "right.json").handedness, "Right")
        saved = [call.kwargs["handedness"] for call in logger.write.call_args_list
                 if call.args[0] == "press_setup_saved"]
        self.assertEqual(saved, ["Left", "Right"])

    def test_two_hand_mode_only_records_missing_right_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            logger, path = self.run_app(directory, setup=True, hand_option="both", existing_left=True)
            self.assertEqual(PressProfile.load(path), profile_for())
            self.assertEqual(PressProfile.load(Path(directory) / "right.json").handedness, "Right")
            self.assertEqual(list(Path(directory).glob("press.backup_*.json")), [])
        saved = [call.kwargs["handedness"] for call in logger.write.call_args_list
                 if call.args[0] == "press_setup_saved"]
        self.assertEqual(saved, ["Right"])

    def test_saved_two_hand_profiles_start_typing_with_eight_indicators(self):
        panels = []
        with tempfile.TemporaryDirectory() as directory:
            logger, _ = self.run_app(directory, hand_option="both", panels=panels)
        self.assertIn("Home row", panels[0][0])
        for key in "asdfjkl;":
            self.assertIn(key.upper() + ":", panels[0][2])
        events = [call.kwargs for call in logger.write.call_args_list if call.args[0] == "press"]
        self.assertEqual([event["key"] for event in events], ["f"])

    def test_complete_sensor_loop_emits_one_press_with_latency(self):
        with tempfile.TemporaryDirectory() as directory:
            logger, _ = self.run_app(directory)
        events = [call.kwargs for call in logger.write.call_args_list if call.args[0] == "press"]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["key"], "f")
        self.assertGreaterEqual(events[0]["latency_ms"], 0)

    def test_two_hand_accuracy_check_targets_each_key_five_times(self):
        with tempfile.TemporaryDirectory() as directory:
            logger, _ = self.run_app(directory, hand_option="both", controls=[ord("t"), ord("q")])
        started = [call.kwargs for call in logger.write.call_args_list
                   if call.args[0] == "press_accuracy_started"]
        self.assertEqual(len(started), 1)
        self.assertEqual(len(started[0]["targets"]), 40)
        for key in "asdfjkl;":
            self.assertEqual(started[0]["targets"].count(key), 5)

    def test_auto_setup_records_the_visible_right_hand(self):
        with tempfile.TemporaryDirectory() as directory:
            _, path = self.run_app(directory, setup=True, hand_option="auto", detected_hand="Right")
            profile = PressProfile.load(path)
        self.assertEqual(profile.handedness, "Right")
        self.assertEqual([finger.key for finger in profile.fingers], ["j", "k", "l", ";"])

    def test_wrong_hand_is_explained_in_recording_panel(self):
        panels = []
        with tempfile.TemporaryDirectory() as directory:
            _, path = self.run_app(directory, setup=True, detected_hand="Right", panels=panels,
                                   controls=[ord(" ")] + [-1] * 20 + [ord("q")])
            self.assertFalse(path.exists())
        self.assertTrue(any("Camera sees Right, setup wants Left" in status for _, status, _ in panels))
        self.assertTrue(any("Camera sees: Right" in states for _, _, states in panels))

    def test_h_switches_hand_and_resets_setup(self):
        panels = []
        with tempfile.TemporaryDirectory() as directory:
            self.run_app(directory, setup=True, detected_hand="Right", panels=panels,
                         controls=[ord("h"), ord("q")])
        self.assertIn("Right hand", panels[-1][0])
        self.assertIn("Selected: Right", panels[-1][2])

    def test_silent_cycle_emits_no_press(self):
        with tempfile.TemporaryDirectory() as directory:
            logger, _ = self.run_app(directory, sound=False)
        self.assertNotIn("press", [call.args[0] for call in logger.write.call_args_list])

    def test_paused_keyboard_emits_no_press(self):
        with tempfile.TemporaryDirectory() as directory:
            logger, _ = self.run_app(directory, controls=[ord(" ")] + [-1] * 13 + [ord("q")])
        self.assertNotIn("press", [call.args[0] for call in logger.write.call_args_list])

    def test_aborted_accuracy_check_is_labeled_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            logger, _ = self.run_app(directory, controls=[ord("t"), ord("q")])
        report = logger.write.call_args
        self.assertEqual(report.args[0], "press_accuracy_aborted")
        self.assertFalse(report.kwargs["complete"])
        self.assertEqual(report.kwargs["attempts"], [])

    def test_completed_accuracy_check_counts_all_missed_trials(self):
        with tempfile.TemporaryDirectory() as directory:
            logger, _ = self.run_app(directory, sound=False,
                controls=[ord("t")] + [-1] * 1700 + [ord("q")])
        reports = [call.kwargs for call in logger.write.call_args_list
                   if call.args[0] == "press_accuracy"]
        self.assertEqual(len(reports), 1)
        self.assertTrue(reports[0]["complete"])
        self.assertEqual(reports[0]["missed"], 20)
        self.assertEqual(reports[0]["correct"], 0)
        self.assertEqual(reports[0]["false_activations"], 0)
        self.assertIsNone(reports[0]["median_latency_ms"])


if __name__ == "__main__":
    unittest.main()

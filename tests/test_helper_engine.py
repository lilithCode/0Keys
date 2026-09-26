import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

import helper_engine
from helper_engine import EngineSettings, Session, TypingEngine
from helper_finger_press import FingerClick
from helper_keyboard import SpacedKeyboardLayout
from helper_vision import HandSnapshot
from test_helper_motion import snapshot

KEY_A = next(key for key in SpacedKeyboardLayout().keys if key.name == "a")
FRAME = np.zeros((360, 640, 3), np.uint8)


class FakeTracker:
    def __init__(self, hands=True):
        self.hands = hands
        self.closed = False

    def process(self, _frame, now):
        return snapshot(now) if self.hands else HandSnapshot(now, ())

    def close(self):
        self.closed = True


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        (root / "model.task").touch()
        layout = root / "layout.json"
        layout.write_text(json.dumps({"camera": 0, "rotation": 0, "mirrored": True,
                                      "points": [[0, 0], [1, 0], [1, 1], [0, 1]], "flip_rows": False}))
        self.settings = EngineSettings(model=str(root / "model.task"), calibration=str(layout),
                                       movement_profile=str(root / "movement.json"))
        patcher = patch.object(helper_engine, "FingerPressDetector")
        detector = patcher.start()
        self.addCleanup(patcher.stop)
        detector.return_value.update.side_effect = lambda snap: [FingerClick(KEY_A, "Left", 8, snap.timestamp)]
        detector.return_value.status = "ready"
        self.detector = detector.return_value

    def tearDown(self):
        self.directory.cleanup()

    def run_session(self, tracker, seconds, step=0.1):
        session = Session(self.settings)
        session.begin(0.0)
        results = []
        for index in range(1, int(seconds / step) + 1):
            results.append((index * step, session.process(FRAME, index * step, tracker)))
            if results[-1][1].finished:
                break
        return session, results

    def test_keys_are_typed_only_after_the_arming_delay(self):
        _, results = self.run_session(FakeTracker(), 3.0)
        typed = [now for now, result in results if result.keys]
        self.assertTrue(typed)
        self.assertGreaterEqual(min(typed), self.settings.arm_seconds - 1e-9)
        self.assertEqual({r.state for now, r in results if now < 1.4}, {"ready"})
        self.assertEqual(results[-1][1].state, "typing")
        self.detector.reset.assert_called()

    def test_start_sign_is_required_when_enabled(self):
        self.settings.start_sign = True
        _, results = self.run_session(FakeTracker(), 3.0)
        self.assertFalse(any(result.keys for _, result in results))
        self.assertEqual({result.state for _, result in results}, {"sign"})

    def test_no_hands_pauses_automatically(self):
        self.settings.auto_pause = 2.0
        _, results = self.run_session(FakeTracker(hands=False), 5.0)
        now, last = results[-1]
        self.assertTrue(last.finished)
        self.assertEqual(last.state, "off")
        self.assertAlmostEqual(now, 2.1)
        self.assertIn("no hands", last.message)

    def test_sensitivity_follows_the_settings_while_typing(self):
        session, _ = self.run_session(FakeTracker(), 0.2)
        self.settings.sensitivity = 9.0
        session.process(FRAME, 0.3, FakeTracker())
        self.assertEqual(self.detector.sensitivity, 2.5)

    def test_engine_thread_types_then_releases_the_camera(self):
        clock = [0.0]
        camera = MagicMock(fps=30.0)

        def read():
            clock[0] += 0.1
            return True, FRAME, clock[0]

        camera.read.side_effect = read
        tracker = FakeTracker()
        keys, states, previews = [], [], []
        typed = threading.Event()

        def on_key(key):
            keys.append(key)
            typed.set()

        self.settings.arm_seconds = 0.3
        engine = TypingEngine(self.settings, on_key, lambda state, message: states.append(state),
                              on_preview=previews.append, camera_factory=lambda _: camera,
                              tracker_factory=lambda model, mirrored: tracker, clock=lambda: 0.0)
        engine.start()
        self.assertTrue(typed.wait(5))
        engine.stop()
        self.assertFalse(engine.running)
        self.assertEqual(keys[0], KEY_A)
        self.assertEqual(states[0], "starting")
        self.assertIn("typing", states)
        self.assertEqual(states[-1], "off")
        self.assertTrue(previews and previews[0].shape[1] == TypingEngine.PREVIEW_WIDTH)
        camera.release.assert_called_once()
        self.assertTrue(tracker.closed)

    def test_camera_failure_is_reported_not_raised(self):
        states = []
        camera = MagicMock()
        camera.read.return_value = (False, None, 0.0)
        engine = TypingEngine(self.settings, lambda key: None, lambda state, message: states.append((state, message)),
                              camera_factory=lambda _: camera, tracker_factory=lambda model, mirrored: FakeTracker())
        engine.start()
        engine._thread.join(5)
        self.assertEqual(states[-1][0], "error")
        self.assertIn("camera", states[-1][1])
        camera.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()

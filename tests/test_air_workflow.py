import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

import air_keyboard
from helper_keyboard import SpacedKeyboardLayout
from helper_vision import HandSnapshot
from test_helper_air import pointing


class AirWorkflowTests(unittest.TestCase):
    def run_app(self, controls=None, pinch=True):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.task").touch()
            calibration = root / "layout.json"
            calibration.write_text(json.dumps({"camera": 0, "rotation": 0, "mirrored": True,
                                               "points": [[0, 0], [1, 0], [1, 1], [0, 1]], "flip_rows": False}))
            args = argparse.Namespace(camera=0, model=str(root / "model.task"), calibration=str(calibration),
                                      profile=str(root / "training.json"), rotation=0, no_mirror=False)
            clock = [0.0]
            frame_index = [0]
            camera = MagicMock()

            def read():
                clock[0] += 0.06
                frame_index[0] += 1
                return True, np.zeros((480, 640, 3), np.uint8)

            camera.read.side_effect = read
            tracker = MagicMock()
            layout = SpacedKeyboardLayout()
            tracker.process.side_effect = lambda frame, timestamp: HandSnapshot(
                timestamp, (pointing(layout, closed=pinch and frame_index[0] >= 6),))
            panels = []

            def panel(frame, composer, status, detail, fps):
                panels.append((composer.text, status, detail))
                return frame

            with ExitStack() as stack:
                stack.enter_context(patch.object(air_keyboard, "parse_args", return_value=args))
                stack.enter_context(patch.object(air_keyboard.time, "perf_counter", side_effect=lambda: clock[0]))
                stack.enter_context(patch.object(air_keyboard.cv2, "VideoCapture", return_value=camera))
                for name in ("namedWindow", "resizeWindow", "setMouseCallback", "imshow", "destroyAllWindows"):
                    stack.enter_context(patch.object(air_keyboard.cv2, name))
                stack.enter_context(patch.object(air_keyboard.cv2, "waitKey",
                                                side_effect=controls or [-1] * 15 + [ord("q")]))
                stack.enter_context(patch.object(air_keyboard, "MediaPipeHandTracker")).return_value.__enter__.return_value = tracker
                stack.enter_context(patch.object(air_keyboard, "draw_panel", side_effect=panel))
                stack.enter_context(patch.object(air_keyboard, "draw_layout"))
                stack.enter_context(patch.object(air_keyboard, "draw_snapshot"))
                self.assertEqual(air_keyboard.main(), 0)
            camera.release.assert_called_once()
            self.assertFalse((root / "training.json").exists())
            return panels

    def test_complete_webcam_loop_types_one_character_without_audio(self):
        self.assertEqual(self.run_app()[-1][0], "a")

    def test_hovering_does_not_type_or_train(self):
        self.assertTrue(all(text == "" for text, _, _ in self.run_app(pinch=False)))

    def test_paused_app_cannot_type(self):
        panels = self.run_app(controls=[ord(" ")] + [-1] * 14 + [ord("q")])
        self.assertEqual(panels[-1][0], "")

    def test_training_without_a_selected_key_does_not_type(self):
        panels = self.run_app(controls=[ord("t")] + [-1] * 14 + [ord("q")])
        self.assertEqual(panels[-1][0], "")
        self.assertIn("TRAINING", panels[-1][2])

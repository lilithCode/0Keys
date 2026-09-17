import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

import air_keyboard
import movement_keyboard
from helper_motion import MovementModel, NO_KEY
from helper_keyboard import SpacedKeyboardLayout, KeyboardCalibration
from helper_camera_view import CameraView
from helper_keyboard_ui import buttons, display_size
from helper_vision import HandSnapshot
from test_helper_motion import example, snapshot


def button_center(control, width=640, height=480):
    shown = display_size(width, height)
    left, top, right, bottom = next(rect for _, name, rect in buttons(*shown) if name == control)
    return (left + right) // 2, (top + bottom) // 2


class MovementWorkflowTests(unittest.TestCase):
    def run_app(self, controls=None, train=False, pretrained=False, trained_only=True, click_type=False,
                record=False, no_hands=False, click_rotate=False, saved_view=None, explicit_view=(0, False)):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.task").touch()
            calibration = root / "layout.json"
            calibration.write_text(json.dumps({"camera": 0, "rotation": 0, "mirrored": True,
                                               "points": [[0, 0], [1, 0], [1, 1], [0, 1]], "flip_rows": False,
                                               **(saved_view or {})}))
            args = argparse.Namespace(camera=0, model=str(root / "model.task"), calibration=str(calibration),
                                      profile=str(root / "pinch.json"), movement_profile=str(root / "movement.json"),
                                      rotation=explicit_view[0], no_mirror=explicit_view[1], mode="movement", trained_only=trained_only)
            if record:
                args.record_landmarks = str(root / "diagnostics" / "frames.jsonl")
            clock = [0.0]
            frame_index = [0]
            camera = MagicMock(fps=30.0, note="test exposure", dark=False)
            history = []
            callback = []
            self.frame_shapes = []
            if pretrained:
                geometry = KeyboardCalibration(((0, 0), (1, 0), (1, 1), (0, 1)))
                context = air_keyboard.context_for(args, CameraView.from_args(args, phone=False), geometry)
                context["gesture"] = "whole_hand_movement_v1"
                model = MovementModel(SpacedKeyboardLayout(), context)
                for _ in range(5):
                    model.add("a", example())
                for _ in range(3):
                    model.add(NO_KEY, example(amplitude=0))
                model.save(args.movement_profile)

            def read(timeout=2.0):
                clock[0] += 0.125
                frame_index[0] += 1
                shown = display_size(640, 480)
                if train and frame_index[0] == 2:
                    # Mouse positions are on the enlarged picture; A is at 15% across, half way down.
                    callback[0](1, int(0.15 * shown[0]), int(0.5 * shown[1]), None, None)
                if click_type and frame_index[0] == 3:
                    callback[0](1, *button_center("t"), None, None)
                if click_rotate and frame_index[0] == 3:
                    callback[0](1, *button_center("]"), None, None)
                return True, np.zeros((480, 640, 3), np.uint8), clock[0]

            camera.read.side_effect = read
            tracker = MagicMock()

            def process(frame, timestamp):
                self.frame_shapes.append(frame.shape)
                if no_hands:
                    return HandSnapshot(timestamp, ())
                start = 30 if train else 7
                index = frame_index[0] - start
                points = example().points[index] if 0 <= index < len(example().points) else None
                return snapshot(timestamp, points)

            tracker.process.side_effect = process

            def panel(frame, composer, status, detail, fps, movement=False):
                self.assertTrue(movement)
                history.append((composer.text, status, detail))
                return frame

            with ExitStack() as stack:
                stack.enter_context(patch.object(air_keyboard, "parse_args", return_value=args))
                stack.enter_context(patch.object(movement_keyboard.time, "perf_counter", side_effect=lambda: clock[0]))
                stack.enter_context(patch.object(movement_keyboard, "WebcamStream", return_value=camera))
                for name in ("namedWindow", "resizeWindow", "imshow", "destroyAllWindows"):
                    stack.enter_context(patch.object(movement_keyboard.cv2, name))
                stack.enter_context(patch.object(movement_keyboard.cv2, "setMouseCallback",
                                                side_effect=lambda name, cb: callback.append(cb)))
                stack.enter_context(patch.object(movement_keyboard.cv2, "waitKey",
                                                side_effect=controls or [-1] * 45 + [ord("q")]))
                factory = stack.enter_context(patch.object(movement_keyboard, "MediaPipeHandTracker"))
                factory.return_value.__enter__.return_value = tracker
                stack.enter_context(patch.object(air_keyboard, "draw_panel", side_effect=panel))
                stack.enter_context(patch.object(air_keyboard, "draw_layout"))
                stack.enter_context(patch.object(movement_keyboard, "draw_snapshot"))
                self.assertEqual(air_keyboard.main(), 0)
                self.tracker_mirrors = [call.kwargs["input_mirrored"] for call in factory.call_args_list]
            camera.release.assert_called_once()
            if record:
                lines = [json.loads(line) for line in Path(args.record_landmarks).read_text().splitlines()]
                self.assertIn("context", lines[0])
                self.assertAlmostEqual(lines[0]["aspect"], 640 / 480)
                self.assertEqual(len(lines) - 1, len(history))
                self.assertIn("landmarks", lines[1]["hands"][0])
                self.assertNotIn("text", lines[1])
            self.assertFalse((root / "pinch.json").exists())
            saved = json.loads(Path(args.movement_profile).read_text()) if Path(args.movement_profile).exists() else None
            self.saved_layout = json.loads(calibration.read_text())
            return history, saved

    def test_untrained_app_never_types_or_learns_automatically(self):
        panels, saved = self.run_app()
        self.assertTrue(all(text == "" for text, _, _ in panels))
        self.assertIsNone(saved)

    def test_mouse_target_and_countdown_save_without_pinch(self):
        panels, saved = self.run_app([ord("l")] + [-1] * 44 + [ord("q")], train=True)
        self.assertTrue(all(text == "" for text, _, _ in panels))
        self.assertIsNotNone(saved)
        self.assertEqual([s["label"] for s in saved["samples"]], ["a"])

    def test_pausing_cancels_training(self):
        panels, saved = self.run_app([ord("l")] + [-1] * 10 + [ord(" ")] + [-1] * 33 + [ord("q")], train=True)
        self.assertIsNone(saved)
        self.assertTrue(all(text == "" for text, _, _ in panels))

    def test_live_model_types_once_without_changing_training(self):
        panels, saved = self.run_app(pretrained=True)
        self.assertEqual(panels[-1][0], "a")
        self.assertEqual(len(saved["samples"]), 8)

    def test_basic_mode_can_type_without_any_saved_key_examples(self):
        panels, saved = self.run_app(trained_only=False)
        self.assertEqual(panels[-1][0], "a")
        self.assertIsNone(saved)

    def test_opt_in_landmark_capture_records_without_training_or_images(self):
        panels, saved = self.run_app(trained_only=False, record=True)
        self.assertEqual(panels[-1][0], "a")
        self.assertIsNone(saved)

    def test_t_always_enters_typing_and_never_toggles_training(self):
        panels, _ = self.run_app([ord("t"), ord("t")] + [-1] * 43 + [ord("q")], trained_only=False)
        self.assertEqual(panels[-1][0], "a")
        self.assertTrue(all("TYPE MODE" in status for _, status, _ in panels))

    def test_type_button_cancels_an_active_training_countdown(self):
        panels, saved = self.run_app([ord("l")] + [-1] * 44 + [ord("q")], train=True, click_type=True)
        self.assertIn("TYPE MODE", panels[-1][1])
        self.assertIsNone(saved)

    def test_empty_view_explains_where_to_point_the_camera(self):
        panels, _ = self.run_app(trained_only=False, no_hands=True)
        self.assertIn("No hands in the camera picture", panels[-1][1])
        self.assertIn("--phone", panels[-1][2])

    def test_minus_and_plus_change_sensitivity_live(self):
        panels, _ = self.run_app([ord("-"), ord("-"), ord("=")] + [-1] * 42 + [ord("q")], trained_only=False)
        self.assertIn("Sensitivity 0.85", panels[-1][2])
        self.assertEqual(panels[-1][0], "a")

    def test_rotate_during_press_pauses_cancels_motion_and_saves_view(self):
        panels, _ = self.run_app([-1] * 9 + [ord("]")] + [-1] * 20 + [ord("q")], trained_only=False)
        self.assertEqual(panels[-1][0], "")
        self.assertIn("PAUSED", panels[-1][1])
        self.assertIn("Portrait", panels[-1][1])
        self.assertEqual(self.frame_shapes[-1], (640, 480, 3))
        self.assertEqual(self.saved_layout["rotation"], 270)
        self.assertEqual(len(self.tracker_mirrors), 2)

    def test_mirror_restarts_tracker_and_does_not_overwrite_personal_examples(self):
        panels, saved = self.run_app([ord("m"), -1, ord("q")], pretrained=True)
        self.assertEqual(self.tracker_mirrors, [True, False])
        self.assertFalse(self.saved_layout["mirrored"])
        self.assertIn("Mirror off", panels[-1][1])
        self.assertIn("0 trained keys", panels[-1][2])
        self.assertEqual(len(saved["samples"]), 8)
        self.assertTrue(saved["context"]["mirrored"])

    def test_rotate_button_and_type_button_resume(self):
        panels, _ = self.run_app([-1] * 4 + [ord("t"), ord("q")], click_rotate=True)
        self.assertEqual(self.saved_layout["rotation"], 270)
        self.assertIn("TYPE MODE", panels[-1][1])

    def test_orientation_cannot_change_during_landmark_recording(self):
        panels, _ = self.run_app([ord("]"), ord("m"), ord("q")], record=True)
        self.assertEqual(self.saved_layout["rotation"], 0)
        self.assertEqual(self.tracker_mirrors, [True])
        self.assertIn("Finish the landmark recording", panels[-1][2])

    def test_saved_orientation_restored_on_startup(self):
        panels, _ = self.run_app([ord("q")], saved_view={"rotation": 90, "mirrored": False}, explicit_view=(None, None))
        self.assertEqual(self.frame_shapes[0], (640, 480, 3))
        self.assertEqual(self.tracker_mirrors, [False])
        self.assertIn("Portrait 90deg", panels[0][1])

    def test_rotation_cancels_active_training(self):
        panels, saved = self.run_app([ord("l")] + [-1] * 5 + [ord("]")] + [-1] * 38 + [ord("q")], train=True)
        self.assertIsNone(saved)
        self.assertIn("PAUSED", panels[-1][1])


if __name__ == "__main__":
    unittest.main()

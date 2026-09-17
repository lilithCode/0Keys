import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from helper_keyboard import SpacedKeyboardLayout
from helper_motion import Movement, MovementModel, NaturalPressModel, NO_KEY
from helper_keyboard_ui import PANEL_WIDTH, button_at, buttons, display_size, draw_workspace
from helper_fusion import TextComposer
from test_helper_motion import example, pose


class NaturalPressTests(unittest.TestCase):
    def setUp(self):
        self.layout = SpacedKeyboardLayout()
        self.personal = MovementModel(self.layout, {})
        self.model = NaturalPressModel(self.layout, self.personal)

    def test_all_fingers_on_both_hands_can_press_without_training(self):
        for hand in ("Left", "Right"):
            for tip, name in ((20, "a"), (16, "s"), (12, "d"), (8, "f"), (4, "space")):
                key = next(k for k in self.layout.keys if k.name == name)
                movement = example(finger=tip, coupled=None, hand=hand)
                offset = np.asarray(key.center) - movement.points[0][tip]
                for frame in movement.points:
                    frame[tip] += offset
                found, _ = self.model.predict(movement)
                self.assertIsNotNone(found, (hand, tip))
                self.assertEqual(found.name, name)
        self.assertEqual(self.personal.samples, [])

    def test_smaller_accompanying_motion_is_one_key(self):
        self.assertEqual(self.model.predict(example())[0].name, "a")

    def test_equal_fingers_rest_noise_and_hand_reposition_rejected(self):
        equal = example()
        for frame in equal.points:
            frame[16, 1] = frame[20, 1]
        reposition = example(amplitude=0)
        for index, frame in enumerate(reposition.points):
            frame += (0, (0, 0.2, 0.4, 0.65, 0.4, 0.2, 0, 0, 0)[index])
        noise = example(amplitude=0)
        noise.points[3][20, 1] += 0.65
        for movement in (equal, reposition, noise, example(amplitude=0)):
            self.assertIsNone(self.model.predict(movement)[0])

    def test_border_and_gap_do_not_snap_to_a_key(self):
        for x in (2.75, 1.84):
            movement = example()
            for frame in movement.points:
                frame[20, 0] = x
            self.assertIsNone(self.model.predict(movement)[0])

    def test_saved_no_key_examples_veto_basic_press(self):
        for _ in range(3):
            self.personal.add(NO_KEY, example())
        self.assertIsNone(self.model.predict(example())[0])

    def test_no_basic_fallback_for_a_trained_key_that_was_rejected(self):
        for _ in range(3):
            self.personal.add("a", example())
        self.assertIsNone(self.model.predict(example())[0])
        self.assertEqual(self.model.predict(example(finger=16, coupled=None))[0].name, "s")


class ReuseMovementTests(unittest.TestCase):
    def test_layout_edit_reuses_only_no_key_samples_without_overwriting_file(self):
        layout = SpacedKeyboardLayout()
        context = {"layout": "spaced_ansi_v1", "camera": 0, "rotation": 0, "mirrored": True,
                   "points": [[0, 0], [1, 0], [1, 1], [0, 1]], "flip_rows": False,
                   "gesture": "whole_hand_movement_v1"}
        model = MovementModel(layout, context)
        model.add("a", example())
        for _ in range(3):
            model.add(NO_KEY, example(amplitude=0))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "saved.json"
            model.save(path)
            original = path.read_bytes()
            changed = {**context, "flip_rows": True, "points": [[0.05, 0.1], [0.95, 0.1], [0.95, 0.9], [0.05, 0.9]]}
            loaded, _ = MovementModel.load_for_view(path, layout, changed)
            self.assertEqual(loaded.count(NO_KEY), 3)
            self.assertEqual(loaded.count("a"), 0)
            self.assertEqual(path.read_bytes(), original)
            self.assertTrue(np.isfinite(loaded.samples[0]["motion"]).all())
            for fields in ({"mirrored": False}, {"rotation": 180}, {"camera": 1}):
                with self.assertRaises(ValueError):
                    MovementModel.load_for_view(path, layout, {**context, **fields})


class WorkspaceTests(unittest.TestCase):
    def test_controls_have_distinct_actions(self):
        width, height = display_size(640, 360)
        controls = [control for _, control, _ in buttons(width, height)]
        self.assertEqual(sorted(controls), sorted("tl c[]mf"))
        for _, control, (left, top, right, bottom) in buttons(width, height):
            self.assertEqual(button_at((left + right) // 2, (top + bottom) // 2, width, height), control)
        # Clicks on the camera picture drag corners or pick keys, never buttons.
        self.assertIsNone(button_at(30, 30, width, height))
        self.assertIsNone(button_at(width // 2, height - 5, width, height))

    def test_output_has_its_own_visible_area(self):
        composer = TextComposer()
        empty = draw_workspace(np.zeros((360, 640, 3), np.uint8), composer, "TYPE MODE", "Basic detection", 16)
        composer.text = "hello keyboard\nsecond line"
        typed = draw_workspace(np.zeros((360, 640, 3), np.uint8), composer, "TYPE MODE", "Basic detection", 16)
        self.assertEqual(typed.shape, (640, 640 + PANEL_WIDTH, 3))
        self.assertTrue(np.any(empty[430:, 640:] != typed[430:, 640:]))
        # The text sits beside the camera picture instead of covering the hands.
        self.assertFalse(np.any(typed[:360, :640]))

    def test_camera_picture_is_enlarged_and_keeps_its_shape(self):
        # A 640 by 480 phone frame turned to portrait used to show as a narrow strip.
        width, height = display_size(480, 640)
        self.assertGreaterEqual(height, 800)
        self.assertAlmostEqual(width / height, 480 / 640, places=2)
        self.assertEqual(display_size(640, 360), (1280, 720))

    def test_caps_lock_is_visible(self):
        composer = TextComposer()
        off = draw_workspace(np.zeros((360, 640, 3), np.uint8), composer, "TYPE MODE", "", 16)
        composer.caps_lock = True
        on = draw_workspace(np.zeros((360, 640, 3), np.uint8), composer, "TYPE MODE", "", 16)
        self.assertTrue(np.any(off[430:470, 640:] != on[430:470, 640:]))


if __name__ == "__main__":
    unittest.main()

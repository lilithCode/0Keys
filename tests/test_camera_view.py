import argparse
import unittest

import numpy as np

from helper_camera_view import CameraView, add_view_arguments


class CameraViewTests(unittest.TestCase):
    def view(self, phone=True, arguments=()):
        parser = argparse.ArgumentParser()
        add_view_arguments(parser)
        return CameraView.from_args(parser.parse_args(arguments), phone=phone)

    def test_phone_defaults_to_upright_mirrored_landscape(self):
        self.assertEqual(self.view(), CameraView(180, True))

    def test_webcam_keeps_existing_mirrored_view(self):
        self.assertEqual(self.view(phone=False), CameraView(0, True))

    def test_explicit_view_options_override_defaults(self):
        self.assertEqual(self.view(arguments=("--rotation", "0", "--mirror")), CameraView(0, True))
        self.assertEqual(self.view(phone=False, arguments=("--no-mirror",)), CameraView(0, False))

    def test_upside_down_phone_frame_is_restored_without_reflection(self):
        upright = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
        phone_frame = np.rot90(upright, 2)
        np.testing.assert_array_equal(self.view(arguments=("--no-mirror",)).apply(phone_frame), upright)

    def test_default_phone_view_reflects_horizontal_positions(self):
        upright = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
        np.testing.assert_array_equal(self.view().apply(np.rot90(upright, 2)), np.fliplr(upright))

    def test_rotation_happens_before_optional_mirroring(self):
        frame = np.arange(6, dtype=np.uint8).reshape(2, 3)
        result = CameraView(90, True).apply(frame)
        np.testing.assert_array_equal(result, np.array([[0, 3], [1, 4], [2, 5]]))
        self.assertTrue(result.flags.c_contiguous)

    def test_live_rotation_follows_screen_direction_with_or_without_mirror(self):
        frame = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
        for mirrored in (False, True):
            for angle in (0, 90, 180, 270):
                view = CameraView(angle, mirrored)
                np.testing.assert_array_equal(view.rotate_preview().apply(frame), np.rot90(view.apply(frame), -1))
                np.testing.assert_array_equal(view.rotate_preview(False).apply(frame), np.rot90(view.apply(frame)))
                np.testing.assert_array_equal(view.toggle_mirror().apply(frame), np.fliplr(view.apply(frame)))
                rotated = view
                for _ in range(4):
                    rotated = rotated.rotate_preview()
                self.assertEqual(rotated, view)

    def test_saved_view_is_used_unless_overridden_explicitly(self):
        parser = argparse.ArgumentParser()
        add_view_arguments(parser)
        saved = {"rotation": 90, "mirrored": False}
        self.assertEqual(CameraView.from_args(parser.parse_args([]), True, saved), CameraView(90, False))
        self.assertEqual(CameraView.from_args(parser.parse_args(["--rotation", "0"]), True, saved), CameraView(0, False))
        self.assertEqual(CameraView.from_args(parser.parse_args(["--mirror"]), True, saved), CameraView(90, True))

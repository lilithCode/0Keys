import tempfile
import unittest
from pathlib import Path

from helper_keyboard import KeyboardCalibration, KeyboardLayout, SpacedKeyboardLayout


CAMERA_POINTS = (
    (0.15, 0.20),
    (0.85, 0.25),
    (0.90, 0.85),
    (0.10, 0.80),
)


class KeyboardLayoutTests(unittest.TestCase):
    def test_spaced_layout_has_real_gaps_and_staggered_rows(self):
        layout = SpacedKeyboardLayout()
        keys = {key.name: key for key in layout.keys}
        self.assertLess(keys["q"].x, keys["a"].x)
        self.assertLess(keys["a"].x, keys["z"].x)
        self.assertLess(keys["q"].x + keys["q"].width, keys["w"].x)
        self.assertLess(keys["q"].y + keys["q"].height, keys["a"].y)
        self.assertAlmostEqual(keys["space"].width, 6.25 - 0.16)
        self.assertIsNone(layout.key_at(2.5, 1.5))

    def test_maps_common_keys(self):
        layout = KeyboardLayout()
        self.assertEqual(layout.key_at(2.0, 1.5).label, "Q")
        self.assertEqual(layout.key_at(2.25, 2.5).label, "A")
        self.assertEqual(layout.key_at(7.5, 4.5).name, "space")
        self.assertIsNone(layout.key_at(15.1, 2.0))

    def test_layout_rows_fill_the_keyboard_width(self):
        layout = KeyboardLayout()
        for row in range(5):
            keys = [key for key in layout.keys if key.y == row]
            self.assertAlmostEqual(sum(key.width for key in keys), layout.width)


class KeyboardCalibrationTests(unittest.TestCase):
    def test_maps_camera_corners_to_keyboard_corners(self):
        calibration = KeyboardCalibration(CAMERA_POINTS)
        expected = ((0.0, 0.0), (15.0, 0.0), (15.0, 5.0), (0.0, 5.0))
        for source, target in zip(CAMERA_POINTS, expected):
            mapped = calibration.map_to_keyboard(*source)
            self.assertAlmostEqual(mapped[0], target[0], places=4)
            self.assertAlmostEqual(mapped[1], target[1], places=4)

    def test_round_trip_mapping(self):
        calibration = KeyboardCalibration(CAMERA_POINTS)
        image_point = calibration.map_to_image(6.25, 2.75)
        keyboard_point = calibration.map_to_keyboard(*image_point)
        self.assertAlmostEqual(keyboard_point[0], 6.25, places=5)
        self.assertAlmostEqual(keyboard_point[1], 2.75, places=5)

    def test_save_and_load(self):
        calibration = KeyboardCalibration(
            CAMERA_POINTS,
            mirrored=False,
            flip_rows=True,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration.json"
            calibration.save(path)
            loaded = KeyboardCalibration.load(path)
        self.assertFalse(loaded.mirrored)
        self.assertTrue(loaded.flip_rows)
        self.assertAlmostEqual(loaded.camera_points[2][0], CAMERA_POINTS[2][0])

    def test_flips_number_and_space_rows(self):
        normal = KeyboardCalibration(CAMERA_POINTS)
        flipped = KeyboardCalibration(CAMERA_POINTS, flip_rows=True)
        normal_number = normal.map_to_image(7.5, 0.5)
        normal_space = normal.map_to_image(7.5, 4.5)
        flipped_number = flipped.map_to_image(7.5, 0.5)
        flipped_space = flipped.map_to_image(7.5, 4.5)
        self.assertLess(normal_number[1], normal_space[1])
        self.assertGreater(flipped_number[1], flipped_space[1])

    def test_rejects_crossed_points(self):
        crossed = (CAMERA_POINTS[0], CAMERA_POINTS[2], CAMERA_POINTS[1], CAMERA_POINTS[3])
        with self.assertRaises(ValueError):
            KeyboardCalibration(crossed)

    def test_rejects_vertically_reversed_points(self):
        reversed_points = (
            CAMERA_POINTS[3],
            CAMERA_POINTS[2],
            CAMERA_POINTS[1],
            CAMERA_POINTS[0],
        )
        with self.assertRaisesRegex(ValueError, "upper corners"):
            KeyboardCalibration(reversed_points)


if __name__ == "__main__":
    unittest.main()

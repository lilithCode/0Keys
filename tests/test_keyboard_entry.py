import unittest
from pathlib import Path

from air_keyboard import context_for
from helper_camera_view import CameraView
from helper_keyboard import KeyboardCalibration
from keyboard import parse_args


class KeyboardEntryTests(unittest.TestCase):
    def test_phone_and_webcam_have_separate_default_files_and_context(self):
        webcam, phone = parse_args([]), parse_args(["--phone"])
        self.assertNotEqual(webcam.calibration, phone.calibration)
        self.assertNotEqual(webcam.movement_profile, phone.movement_profile)
        self.assertEqual(Path(phone.calibration).name, "phone_air_keyboard_layout.json")
        layout = KeyboardCalibration(((0, 0), (1, 0), (1, 1), (0, 1)))
        view = CameraView(0, True)
        self.assertNotEqual(context_for(webcam, view, layout)["camera"],
                            context_for(phone, view, layout)["camera"])

    def test_explicit_paths_are_preserved(self):
        args = parse_args(["--phone", "--calibration", "my-layout.json", "--movement-profile", "my-movement.json"])
        self.assertEqual(args.calibration, "my-layout.json")
        self.assertEqual(args.movement_profile, "my-movement.json")

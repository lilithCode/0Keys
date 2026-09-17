import unittest

from helper_audio import TapEvent
from helper_fusion import TapFingerFusion, TextComposer
from helper_keyboard import KeyboardCalibration, KeyboardLayout
from helper_vision import HandHistory, HandSnapshot, Landmark, TrackedHand


CALIBRATION_POINTS = ((0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9))


def hand_at(index_y: float, middle_y: float = 0.44) -> TrackedHand:
    landmarks = [Landmark(0.5, 0.44, 0.0) for _ in range(21)]
    landmarks[0] = Landmark(0.5, 0.30, 0.0)
    landmarks[8] = Landmark(0.5, index_y, 0.0)
    landmarks[12] = Landmark(0.56, middle_y, 0.0)
    return TrackedHand(1, "Right", tuple(landmarks))


def tap_event(timestamp: float = 1.0) -> TapEvent:
    return TapEvent(timestamp, 20.0, -10.0, -30.0)


class TapFingerFusionTests(unittest.TestCase):
    def setUp(self):
        self.layout = KeyboardLayout()
        self.calibration = KeyboardCalibration(CALIBRATION_POINTS)

    def test_selects_finger_with_down_and_up_motion(self):
        history = HandHistory()
        history.append(HandSnapshot(0.88, (hand_at(0.42),)))
        history.append(HandSnapshot(1.00, (hand_at(0.52),)))
        history.append(HandSnapshot(1.12, (hand_at(0.43),)))
        fusion = TapFingerFusion(self.layout, self.calibration, camera_offset_seconds=0.0)

        result = fusion.select(tap_event(), history)

        self.assertTrue(result.accepted)
        self.assertEqual(result.candidate.finger_name, "index")
        self.assertEqual(result.candidate.key.label, "H")

    def test_rejects_tap_when_fingers_are_still(self):
        history = HandHistory()
        for timestamp in (0.88, 1.00, 1.12):
            history.append(HandSnapshot(timestamp, (hand_at(0.44),)))
        fusion = TapFingerFusion(self.layout, self.calibration, camera_offset_seconds=0.0)

        result = fusion.select(tap_event(), history)

        self.assertFalse(result.accepted)
        self.assertIn("motion", result.reason.lower())

    def test_missing_post_tap_frame_cannot_reuse_contact_frame(self):
        history = HandHistory()
        history.append(HandSnapshot(0.88, (hand_at(0.42),)))
        history.append(HandSnapshot(1.0, (hand_at(0.52),)))
        fusion = TapFingerFusion(self.layout, self.calibration, camera_offset_seconds=0.0)
        result = fusion.select(tap_event(), history)
        self.assertFalse(result.accepted)
        self.assertEqual(result.candidates, ())

    def test_rejects_equal_finger_motion(self):
        history = HandHistory()
        history.append(HandSnapshot(0.88, (hand_at(0.42, 0.42),)))
        history.append(HandSnapshot(1.00, (hand_at(0.52, 0.52),)))
        history.append(HandSnapshot(1.12, (hand_at(0.43, 0.43),)))
        fusion = TapFingerFusion(self.layout, self.calibration, camera_offset_seconds=0.0)

        result = fusion.select(tap_event(), history)

        self.assertFalse(result.accepted)
        self.assertIn("ambiguous", result.reason.lower())


class TextComposerTests(unittest.TestCase):
    def setUp(self):
        self.layout = KeyboardLayout()
        self.keys = {key.name: key for key in self.layout.keys}
        self.composer = TextComposer()

    def test_writes_letters_space_and_backspace(self):
        self.composer.apply(self.keys["h"])
        self.composer.apply(self.keys["space"])
        self.composer.apply(self.keys["i"])
        self.composer.apply(self.keys["backspace"])
        self.assertEqual(self.composer.text, "h ")

    def test_caps_lock_and_shift(self):
        self.composer.apply(self.keys["caps_lock"])
        self.composer.apply(self.keys["a"])
        self.composer.apply(self.keys["left_shift"])
        self.composer.apply(self.keys["b"])
        self.assertEqual(self.composer.text, "Ab")

    def test_shifted_number(self):
        self.composer.apply(self.keys["left_shift"])
        self.composer.apply(self.keys["1"])
        self.assertEqual(self.composer.text, "!")


if __name__ == "__main__":
    unittest.main()

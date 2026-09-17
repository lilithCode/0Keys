from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from helper_air import AirClickDetector, AirKeyModel
from helper_keyboard import KeyboardCalibration, SpacedKeyboardLayout
from helper_vision import HandSnapshot, Landmark, TrackedHand


def pointing(layout, name="a", closed=False, hand_id=1, handedness="Left"):
    key = next(key for key in layout.keys if key.name == name)
    x, y = key.center[0] / 15, key.center[1] / 5
    points = [Landmark(x, y + 0.10, 0) for _ in range(21)]
    points[0] = Landmark(x, y + 0.12, 0)
    points[5] = Landmark(x - 0.04, y + 0.08, 0)
    points[17] = Landmark(x + 0.06, y + 0.08, 0)
    points[8] = Landmark(x, y, 0)
    points[4] = Landmark(x + (0.012 if closed else 0.09), y, 0)
    return TrackedHand(hand_id, handedness, tuple(points))


class AirModelTests(unittest.TestCase):
    def setUp(self):
        self.layout = SpacedKeyboardLayout()
        self.model = AirKeyModel(self.layout, {"camera": 0})

    def test_letter_centers_and_space_predict_their_keys(self):
        for key in self.layout.keys:
            if key.value not in ("CONTROL", "ALT"):
                prediction = self.model.predict(key.center, "Left")
                self.assertIsNotNone(prediction.key, key.name)
                self.assertEqual(prediction.key.name, key.name)

    def test_gaps_are_not_filled_by_prediction(self):
        for point in ((2.5, 1.5), (2.25, 2.0), (-1, 0), (0, 0)):
            self.assertIsNone(self.model.predict(point, "Left").key)

    def test_confirmed_examples_round_trip_and_wrong_context_rejected(self):
        key = next(key for key in self.layout.keys if key.name == "a")
        for _ in range(5):
            self.model.add("a", key.center, "Left")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            self.model.save(path)
            restored = AirKeyModel.load(path, self.layout, self.model.context)
            self.assertEqual(len(restored.samples), 5)
            self.assertEqual(restored.predict(key.center, "Left").key.name, "a")
            with self.assertRaises(ValueError):
                AirKeyModel.load(path, self.layout, {"camera": 1})

    def test_remote_and_unsupported_training_is_rejected(self):
        for name, point in (("a", (14, 4)), ("left_control", (0.5, 4.5)), ("unknown", (0, 0))):
            with self.assertRaises(ValueError):
                self.model.add(name, point, "Left")

    def test_confirmed_personal_offset_improves_its_relative_score(self):
        key = next(key for key in self.layout.keys if key.name == "a")
        point = (key.center[0] + 0.18, key.center[1])
        before = self.model.predict(point, "Left").confidence
        for _ in range(3):
            self.model.add("a", point, "Left")
        after = self.model.predict(point, "Left")
        self.assertEqual(after.key.name, "a")
        self.assertGreater(after.confidence, before)


class AirClickTests(unittest.TestCase):
    def setUp(self):
        self.layout = SpacedKeyboardLayout()
        calibration = KeyboardCalibration(((0, 0), (1, 0), (1, 1), (0, 1)))
        self.model = AirKeyModel(self.layout, {})
        self.detector = AirClickDetector(self.layout, calibration, self.model)
        self.events = []

    def feed(self, timestamp, closed=False, name="a", hands=None):
        hands = (pointing(self.layout, name, closed),) if hands is None else hands
        self.events.extend(self.detector.update(HandSnapshot(timestamp, hands)))

    def aim(self):
        for time in (0, 0.06, 0.12, 0.18, 0.24):
            self.feed(time)

    def test_hover_never_types(self):
        for i in range(60):
            self.feed(i * 0.06)
        self.assertEqual(self.events, [])

    def test_closed_hand_on_entry_never_types(self):
        for i in range(30):
            self.feed(i * 0.06, closed=True)
        self.assertEqual(self.events, [])

    def test_deliberate_pinch_types_once_without_audio_or_table_contact(self):
        self.aim()
        for i in range(5, 30):
            self.feed(i * 0.06, closed=True)
        self.assertEqual([event.key.name for event in self.events], ["a"])

    def test_single_frame_pinch_noise_does_not_type(self):
        self.aim()
        self.feed(0.30, closed=True)
        self.feed(0.36)
        self.assertEqual(self.events, [])

    def test_tracking_loss_cancels_armed_target(self):
        self.aim()
        self.feed(0.30, hands=())
        self.feed(0.36, closed=True)
        self.feed(0.42, closed=True)
        self.assertEqual(self.events, [])

    def test_camera_gap_cancels_armed_target(self):
        self.aim()
        self.feed(0.60, closed=True)
        self.feed(0.66, closed=True)
        self.assertEqual(self.events, [])

    def test_changing_key_requires_aiming_again(self):
        self.aim()
        self.feed(0.30, name="f")
        self.feed(0.36, name="f", closed=True)
        self.feed(0.42, name="f", closed=True)
        self.assertEqual(self.events, [])

    def test_both_hands_pinching_in_same_frame_is_rejected(self):
        for i in range(7):
            closed = i >= 5
            self.feed(i * 0.06, hands=(pointing(self.layout, "a", closed),
                      pointing(self.layout, "j", closed, hand_id=2, handedness="Right")))
        self.assertEqual(self.events, [])

    def test_release_and_new_aim_allow_second_click(self):
        self.aim()
        self.feed(0.30, closed=True)
        self.feed(0.36, closed=True)
        for i in range(7, 16):
            self.feed(i * 0.06)
        self.feed(0.96, closed=True)
        self.feed(1.02, closed=True)
        self.assertEqual([event.key.name for event in self.events], ["a", "a"])

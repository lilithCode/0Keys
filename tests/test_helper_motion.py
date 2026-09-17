import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from helper_keyboard import KeyboardCalibration, SpacedKeyboardLayout
from helper_motion import GuidedTraining, Movement, MovementModel, MovementRecorder, NO_KEY
from helper_vision import HandSnapshot, Landmark, TrackedHand


def pose():
    points = np.tile([4.0, 3.4], (21, 1))
    points[5] = [5.0, 3.0]
    points[17] = [2.0, 3.0]
    for index, x in ((20, 2.25), (16, 3.25), (12, 4.25), (8, 5.25), (4, 6.0)):
        points[index] = [x, 2.5]
    return points


def example(finger=20, coupled=16, amplitude=0.65, hand="Left"):
    points = []
    for progress in (0, 0.35, 0.8, 1, 0.8, 0.35, 0, 0, 0):
        frame = pose()
        frame[finger, 1] += amplitude * progress
        if coupled is not None:
            frame[coupled, 1] += amplitude * 0.55 * progress
        points.append(frame)
    return Movement(hand, list(np.arange(len(points)) * 0.125), points)


def snapshot(timestamp, points=None, hand_id=1, hand="Left"):
    points = pose() if points is None else points
    tracked = TrackedHand(hand_id, hand, tuple(Landmark(x / 15, y / 5, 0) for x, y in points))
    return HandSnapshot(timestamp, (tracked,))


class MovementModelTests(unittest.TestCase):
    def setUp(self):
        self.layout = SpacedKeyboardLayout()
        self.model = MovementModel(self.layout, {"view": "test"})

    def train(self):
        for amplitude in (0.60, 0.65, 0.70, 0.62, 0.68):
            self.model.add("a", example(amplitude=amplitude))
            self.model.add("s", example(finger=16, coupled=12, amplitude=amplitude))
        for _ in range(3):
            self.model.add(NO_KEY, example(amplitude=0))

    def test_coupled_ring_movement_produces_only_pinky_key(self):
        self.train()
        key, _ = self.model.predict(example(amplitude=0.66))
        self.assertEqual(key.name, "a")
        key, _ = self.model.predict(example(finger=16, coupled=12, amplitude=0.66))
        self.assertEqual(key.name, "s")

    def test_untrained_hover_wrong_hand_and_unknown_motion_rejected(self):
        self.assertIsNone(self.model.predict(example())[0])
        self.train()
        for movement in (example(amplitude=0), example(hand="Right"), example(finger=8, coupled=4)):
            self.assertIsNone(self.model.predict(movement)[0])

    def test_needs_three_examples_of_key_and_no_key_for_same_hand(self):
        for _ in range(2):
            self.model.add("a", example())
        for _ in range(3):
            self.model.add(NO_KEY, example(amplitude=0))
        self.assertIsNone(self.model.predict(example())[0])
        self.model.add("a", example())
        self.assertEqual(self.model.predict(example())[0].name, "a")

    def test_identical_no_key_movement_overrides_key(self):
        self.train()
        for _ in range(3):
            self.model.add(NO_KEY, example())
        self.assertIsNone(self.model.predict(example())[0])

    def test_wrong_target_and_tiny_press_do_not_poison_training(self):
        for name, movement in (("j", example()), ("a", example(amplitude=0.01)), ("left_control", example())):
            with self.assertRaises(ValueError):
                self.model.add(name, movement)
        self.assertEqual(self.model.samples, [])

    def test_ambiguous_labels_rejected(self):
        self.train()
        for _ in range(5):
            self.model.add("s", example())
        self.assertIsNone(self.model.predict(example())[0])

    def test_single_frame_noise_and_missing_return_rejected(self):
        noise = example(amplitude=0)
        noise.points[3][20, 1] += 0.6
        with self.assertRaises(ValueError):
            self.model.add("a", noise)
        held = example()
        held.points[-1][20, 1] += 0.6
        with self.assertRaises(ValueError):
            self.model.add("a", held)

    def test_save_load_context_and_corruption(self):
        self.train()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            self.model.save(path)
            loaded = MovementModel.load(path, self.layout, self.model.context)
            self.assertEqual(loaded.predict(example())[0].name, "a")
            with self.assertRaises(ValueError):
                MovementModel.load(path, self.layout, {})
            data = json.loads(path.read_text())
            data["samples"][0]["motion"] = [[float("nan")]]
            path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                MovementModel.load(path, self.layout, self.model.context)


class MovementRecorderTests(unittest.TestCase):
    def setUp(self):
        self.calibration = KeyboardCalibration(((0, 0), (1, 0), (1, 1), (0, 1)))
        self.recorder = MovementRecorder(self.calibration)

    def resting(self):
        for i in range(5):
            self.assertEqual(self.recorder.update(snapshot(i * 0.125)), [])

    def test_whole_movement_is_one_event_at_eight_fps(self):
        self.resting()
        events = []
        for i, points in enumerate(example().points):
            events.extend(self.recorder.update(snapshot(0.625 + i * 0.125, points)))
        self.assertEqual(len(events), 1)
        model = MovementModel(SpacedKeyboardLayout(), {})
        model.add("a", events[0])

    def test_resting_and_holding_do_not_emit(self):
        self.resting()
        held = example().points[3]
        for i in range(30):
            self.assertEqual(self.recorder.update(snapshot(0.625 + i * 0.125, held)), [])

    def test_lost_tracking_and_time_gap_cancel(self):
        self.resting()
        self.recorder.update(snapshot(0.625, example().points[3]))
        self.recorder.update(HandSnapshot(0.75, ()))
        for i in range(5):
            self.assertEqual(self.recorder.update(snapshot(0.875 + i * 0.125)), [])
        self.recorder.update(snapshot(1.5, example().points[3]))
        self.assertEqual(self.recorder.update(snapshot(3.0)), [])

    def test_guided_batch_records_without_pinch_or_confirmation(self):
        model = MovementModel(SpacedKeyboardLayout(), {})
        training = GuidedTraining("a", 0, repetitions=1)
        for i in range(29):
            training.update(snapshot(i * 0.125), self.recorder, model)
        for i, points in enumerate(example().points):
            training.update(snapshot(3.625 + i * 0.125, points), self.recorder, model)
        self.assertTrue(training.done)
        self.assertEqual(model.count("a"), 1)

    def test_countdown_prepares_rest_pose_but_does_not_save_early_motion(self):
        model = MovementModel(SpacedKeyboardLayout(), {})
        training = GuidedTraining("a", 0, repetitions=1)
        for i in range(24):
            points = example().points[i - 5] if 5 <= i < 14 else pose()
            training.update(snapshot(i * 0.125, points), self.recorder, model)
        self.assertEqual(model.count("a"), 0)
        for i, points in enumerate(example().points[1:]):
            training.update(snapshot(3 + i * 0.125, points), self.recorder, model)
        self.assertTrue(training.done)
        self.assertEqual(model.count("a"), 1)

    def test_negative_recording_counts_both_hands_and_does_not_type(self):
        model = MovementModel(SpacedKeyboardLayout(), {})
        training = GuidedTraining(NO_KEY, 0, repetitions=1)
        for i in range(43):
            left = snapshot(i * 0.125).hands[0]
            right = snapshot(i * 0.125, hand="Right", hand_id=2).hands[0]
            training.update(HandSnapshot(i * 0.125, (left, right)), self.recorder, model)
        self.assertTrue(training.done)
        self.assertEqual(model.count(NO_KEY, "Left"), 1)
        self.assertEqual(model.count(NO_KEY, "Right"), 1)


if __name__ == "__main__":
    unittest.main()

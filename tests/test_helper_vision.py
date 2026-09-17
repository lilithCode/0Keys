import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from helper_vision import (
    DetectedHand,
    HandHistory,
    HandIdentityAssigner,
    HandSnapshot,
    Landmark,
    MediaPipeHandTracker,
)


class HandLabelTests(unittest.TestCase):
    def process_result(self, raw_label, mirrored):
        tracker = MediaPipeHandTracker.__new__(MediaPipeHandTracker)
        tracker.input_mirrored = mirrored
        tracker._mp = SimpleNamespace(Image=MagicMock(), ImageFormat=SimpleNamespace(SRGB=1))
        tracker._landmarker = MagicMock()
        points = [SimpleNamespace(x=0.2, y=0.3, z=-0.01) for _ in range(21)]
        tracker._landmarker.detect_for_video.return_value = SimpleNamespace(
            hand_landmarks=[points],
            handedness=[[SimpleNamespace(category_name=raw_label, display_name=raw_label)]],
        )
        tracker._identity = HandIdentityAssigner()
        tracker._last_mediapipe_timestamp = -1
        return tracker.process(np.zeros((10, 10, 3), dtype=np.uint8), 1.0).hands[0]

    def test_mirrored_labels_are_corrected_without_moving_landmarks(self):
        hand = self.process_result("Right", True)
        self.assertEqual(hand.handedness, "Left")
        self.assertEqual(hand.landmarks[8], Landmark(0.2, 0.3, -0.01))
        self.assertEqual(self.process_result("Left", True).handedness, "Right")

    def test_unmirrored_labels_are_unchanged(self):
        for label in ("Left", "Right", "Unknown"):
            self.assertEqual(self.process_result(label, False).handedness, label)

    def test_unknown_hand_is_not_assigned_a_side(self):
        self.assertEqual(self.process_result("Unknown", True).handedness, "Unknown")

    def test_same_physical_hand_has_same_label_in_both_views(self):
        self.assertEqual(self.process_result("Right", False).handedness,
                         self.process_result("Left", True).handedness)


def detected_hand(x: float, y: float, handedness: str) -> DetectedHand:
    landmarks = tuple(Landmark(x + index * 0.001, y, 0.0) for index in range(21))
    return DetectedHand(handedness=handedness, landmarks=landmarks)


class HandIdentityTests(unittest.TestCase):
    def test_identity_survives_detection_order_change(self):
        assigner = HandIdentityAssigner()
        first = assigner.assign(
            [detected_hand(0.2, 0.5, "Left"), detected_hand(0.8, 0.5, "Right")],
            1.0,
        )
        second = assigner.assign(
            [detected_hand(0.79, 0.5, "Right"), detected_hand(0.21, 0.5, "Left")],
            1.03,
        )

        first_ids = {hand.handedness: hand.hand_id for hand in first}
        second_ids = {hand.handedness: hand.hand_id for hand in second}
        self.assertEqual(first_ids, second_ids)


class HandHistoryTests(unittest.TestCase):
    def test_nearest_snapshot_and_tip_velocity(self):
        assigner = HandIdentityAssigner()
        history = HandHistory()
        hands_1 = assigner.assign([detected_hand(0.2, 0.4, "Left")], 10.0)
        hands_2 = assigner.assign([detected_hand(0.2, 0.5, "Left")], 10.1)
        snapshot_1 = HandSnapshot(timestamp=10.0, hands=hands_1)
        snapshot_2 = HandSnapshot(timestamp=10.1, hands=hands_2)
        history.append(snapshot_1)
        history.append(snapshot_2)

        self.assertEqual(history.nearest(10.09), snapshot_2)
        velocity = history.tip_velocity(snapshot_2, hands_2[0].hand_id, 8, 0.1)
        self.assertIsNotNone(velocity)
        assert velocity is not None
        self.assertAlmostEqual(velocity.y, 1.0)

    def test_rejects_out_of_order_timestamps(self):
        history = HandHistory()
        history.append(HandSnapshot(timestamp=2.0, hands=()))
        with self.assertRaises(ValueError):
            history.append(HandSnapshot(timestamp=1.9, hands=()))


if __name__ == "__main__":
    unittest.main()

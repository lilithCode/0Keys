import unittest

from helper_vision import (
    DetectedHand,
    HandHistory,
    HandIdentityAssigner,
    HandSnapshot,
    Landmark,
)


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

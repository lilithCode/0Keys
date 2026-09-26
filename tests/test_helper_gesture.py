import json
from pathlib import Path
import unittest

from helper_gesture import GAP_SECONDS, HOLD_SECONDS, StartSign, is_start_sign
from helper_vision import Landmark, TrackedHand

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "start_sign_landmarks.json").read_text())


def fixture_hands(prefix):
    samples = [(name, sample) for name, sample in FIXTURE["samples"].items() if name.startswith(prefix)]
    return [(name, sample["aspect"], TrackedHand(1, "Left", tuple(Landmark(*point) for point in hand)))
            for name, sample in samples for hand in sample["hands"]]


class StartSignTests(unittest.TestCase):
    def test_raised_index_finger_is_found_at_every_angle_and_mirror(self):
        hands = fixture_hands("pointing_up/")
        self.assertEqual(len(hands), 8)
        for name, aspect, hand in hands:
            self.assertTrue(is_start_sign(hand, aspect), name)

    def test_look_alike_hand_shapes_do_not_start_typing(self):
        for prefix in ("thumbs_up/", "thumbs_down/", "victory/", "open_hands/"):
            hands = fixture_hands(prefix)
            self.assertGreaterEqual(len(hands), 8)
            for name, aspect, hand in hands:
                self.assertFalse(is_start_sign(hand, aspect), name)

    def test_sign_must_be_held_and_starts_once(self):
        _, aspect, sign = fixture_hands("pointing_up/rot0")[0]
        _, _, other = fixture_hands("victory/rot0")[0]
        detector = StartSign()
        now, triggers = 0.0, 0
        for _ in range(4):
            now += 0.1
            triggers += detector.update([sign], now, aspect)
        now += GAP_SECONDS + 0.1
        detector.update([other], now, aspect)
        self.assertEqual(detector.progress(now), 0.0)
        for _ in range(int(HOLD_SECONDS / 0.1) + 2):
            now += 0.1
            triggers += detector.update([sign], now, aspect)
        self.assertEqual(triggers, 1)
        self.assertAlmostEqual(detector.seen, now)

    def test_one_missed_frame_does_not_restart_the_hold(self):
        _, aspect, sign = fixture_hands("pointing_up/rot90")[0]
        detector = StartSign()
        results = [detector.update([sign] if step != 3 else [], step * 0.1, aspect) for step in range(8)]
        self.assertEqual(results.index(True), 6)


if __name__ == "__main__":
    unittest.main()

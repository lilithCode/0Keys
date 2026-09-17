from dataclasses import replace
import tempfile
import json
from pathlib import Path
import unittest

from helper_audio import TapEvent
from helper_press import AccuracyCheck, KEYS, PressDetector, PressProfile, TwoHandPressDetector, stable_pose
from helper_vision import HandSnapshot, Landmark, TrackedHand


def make_hand(lifted=(), direction=-1, shift=0.0, hand_id=1):
    points = [Landmark(0.5 + shift, 0.5, 0.0) for _ in range(21)]
    points[0] = Landmark(0.5 + shift, 0.7, 0.0)
    for tip, x in ((8, 0.4), (12, 0.5), (16, 0.6), (20, 0.7)):
        points[tip - 3] = Landmark(x + shift, 0.5, 0.0)
        points[tip] = Landmark(x + shift, 0.3 + (direction * 0.05 if tip in lifted else 0), 0.0)
    return TrackedHand(hand_id, "Left", tuple(points))


def profile_for(direction=-1):
    rest = stable_pose([make_hand()] * 8)
    lifts = {tip: stable_pose([make_hand((tip,), direction)] * 8) for tip in KEYS["Left"]}
    return PressProfile.from_poses("Left", 0, True, rest, lifts)


def right_hand(lifted=()):
    return replace(make_hand(lifted, shift=0.2), handedness="Right", hand_id=2)


def right_profile():
    rest = stable_pose([right_hand()] * 8)
    lifts = {tip: stable_pose([right_hand((tip,))] * 8) for tip in KEYS["Right"]}
    return PressProfile.from_poses("Right", 0, True, rest, lifts)


class HoverStartTests(unittest.TestCase):
    def setUp(self):
        self.engine = PressDetector(profile_for(), camera_offset=0, allow_hover=True)
        self.events = []

    def feed(self, time, lifted, sound=False):
        taps = [TapEvent(time, 20, -10, -30)] if sound else []
        self.events.extend(self.engine.update(HandSnapshot(time, (make_hand(lifted),)), taps))

    def test_one_finger_taps_while_others_keep_hovering(self):
        for index in range(30):
            self.feed(index * 0.06, (8, 12, 16, 20))
        self.feed(1.80, (12, 16, 20), sound=True)
        for index in range(31, 40):
            self.feed(index * 0.06, (12, 16, 20))
        self.assertEqual([event.key for event in self.events], ["f"])

    def test_noise_while_hovering_never_types(self):
        for index in range(40):
            self.feed(index * 0.06, (8, 12, 16, 20), sound=True)
        self.assertEqual(self.events, [])

    def test_placing_all_hovering_fingers_down_is_rejected(self):
        for index in range(5):
            self.feed(index * 0.06, (8, 12, 16, 20))
        self.feed(0.30, (), sound=True)
        for index in range(6, 15):
            self.feed(index * 0.06, ())
        self.assertEqual(self.events, [])

    def test_silent_hover_return_remains_rejected(self):
        for index in range(5):
            self.feed(index * 0.06, (8,))
        for index in range(5, 15):
            self.feed(index * 0.06, ())
        self.assertEqual(self.events, [])


class TwoHandPressTests(unittest.TestCase):
    def setUp(self):
        self.engine = TwoHandPressDetector({"Left": profile_for(), "Right": right_profile()})
        self.events = []

    def feed(self, timestamp, left=(), right=(), sound=False):
        taps = [TapEvent(timestamp, 20.0, -10.0, -30.0)] if sound else []
        snapshot = HandSnapshot(timestamp, (make_hand(left), right_hand(right)))
        self.events.extend(self.engine.update(snapshot, taps))

    def rest(self):
        for timestamp in (0.0, 0.06, 0.12):
            self.feed(timestamp)

    def test_alternating_hands_type_their_own_keys(self):
        self.rest()
        for base, side in ((0.0, "left"), (0.6, "right")):
            self.feed(base + 0.18, **{side: (8,)})
            self.feed(base + 0.24, **{side: (8,)})
            self.feed(base + 0.30, sound=True)
            for offset in (0.36, 0.42, 0.48, 0.54, 0.60):
                self.feed(base + offset)
        self.assertEqual([event.key for event in self.events], ["f", "j"])

    def test_simultaneous_hands_cannot_claim_the_same_sound(self):
        self.rest()
        self.feed(0.18, left=(8,), right=(8,))
        self.feed(0.24, left=(8,), right=(8,))
        self.feed(0.30, sound=True)
        for timestamp in (0.36, 0.42, 0.48, 0.54):
            self.feed(timestamp)
        self.assertEqual(self.events, [])

    def test_ambiguous_left_hand_cannot_make_right_hand_win(self):
        self.rest()
        self.feed(0.18, left=(8, 12), right=(8,))
        self.feed(0.24, left=(8, 12), right=(8,))
        self.feed(0.30, sound=True)
        for timestamp in (0.36, 0.42, 0.48, 0.54):
            self.feed(timestamp)
        self.assertEqual(self.events, [])

    def test_resting_both_hands_with_noise_types_nothing(self):
        for index in range(25):
            self.feed(index * 0.06, sound=True)
        self.assertEqual(self.events, [])

    def test_reset_cancels_both_lifts(self):
        self.rest()
        self.feed(0.18, left=(8,), right=(8,))
        self.feed(0.24, left=(8,), right=(8,))
        self.engine.reset()
        self.feed(0.30, sound=True)
        for timestamp in (0.36, 0.42, 0.48, 0.54):
            self.feed(timestamp)
        self.assertEqual(self.events, [])

    def test_mismatched_camera_profiles_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "same camera view"):
            TwoHandPressDetector({"Left": profile_for(), "Right": replace(right_profile(), camera=1)})


class PressDetectorTests(unittest.TestCase):
    def setUp(self):
        self.profile = profile_for()
        self.engine = PressDetector(self.profile, camera_offset=0.0)
        self.events = []

    def feed(self, timestamp, lifted=(), sound=False, hand=None):
        snapshot = HandSnapshot(timestamp, (hand or make_hand(lifted),))
        taps = [TapEvent(timestamp, 20.0, -10.0, -30.0)] if sound else []
        self.events.extend(self.engine.update(snapshot, taps))

    def rest(self):
        for timestamp in (0.0, 0.06, 0.12):
            self.feed(timestamp)

    def finish(self):
        for timestamp in (0.36, 0.42, 0.48, 0.54, 0.60, 0.66):
            self.feed(timestamp)

    def test_resting_and_first_hand_placement_never_type(self):
        for index in range(30):
            self.feed(index * 0.06, sound=True)
        self.assertEqual(self.events, [])

    def test_lift_then_return_emits_one_key_and_no_repeat_while_down(self):
        self.rest()
        self.feed(0.18, (20,))
        self.feed(0.24, (20,))
        self.feed(0.30, sound=True)
        self.finish()
        for index in range(12, 30):
            self.feed(index * 0.06, sound=True)
        self.assertEqual([event.key for event in self.events], ["a"])

    def test_second_press_requires_a_new_lift(self):
        self.rest()
        for base in (0.0, 0.60):
            self.feed(base + 0.18, (8,))
            self.feed(base + 0.24, (8,))
            self.feed(base + 0.30, sound=True)
            for offset in (0.36, 0.42, 0.48, 0.54, 0.60):
                self.feed(base + offset)
        self.assertEqual([event.key for event in self.events], ["f", "f"])

    def test_silent_return_does_not_type(self):
        self.rest()
        self.feed(0.18, (16,))
        self.feed(0.24, (16,))
        self.feed(0.30)
        self.finish()
        self.assertEqual(self.events, [])

    def test_sound_while_hovering_is_not_a_press(self):
        self.rest()
        self.feed(0.18, (20,), sound=True)
        for timestamp in (0.24, 0.30, 0.36, 0.42, 0.48, 0.54, 0.60):
            self.feed(timestamp, (20,))
        self.assertEqual(self.events, [])

    def test_a_single_noisy_lift_frame_cannot_arm(self):
        self.rest()
        self.feed(0.18, (20,))
        self.feed(0.24, sound=True)
        self.feed(0.30)
        self.finish()
        self.assertEqual(self.events, [])

    def test_simultaneous_returns_are_rejected(self):
        self.rest()
        self.feed(0.18, (20, 16))
        self.feed(0.24, (20, 16))
        self.feed(0.30, sound=True)
        self.finish()
        self.assertEqual(self.events, [])

    def test_tracking_loss_cancels_a_lift(self):
        self.rest()
        self.feed(0.18, (20,))
        self.feed(0.24, (20,))
        self.engine.update(HandSnapshot(0.27, ()), [])
        self.feed(0.30, sound=True)
        self.finish()
        self.assertEqual(self.events, [])

    def test_hand_identity_change_cancels_a_lift(self):
        self.rest()
        self.feed(0.18, (20,))
        self.feed(0.24, (20,))
        self.feed(0.30, sound=True, hand=make_hand(hand_id=2))
        self.finish()
        self.assertEqual(self.events, [])

    def test_hand_translation_with_sound_is_not_a_press(self):
        self.rest()
        self.feed(0.18, hand=make_hand(shift=0.12), sound=True)
        self.feed(0.24)
        self.feed(0.30)
        self.finish()
        self.assertEqual(self.events, [])

    def test_sliding_fingertips_does_not_type(self):
        self.rest()
        points = list(make_hand().landmarks)
        for tip in KEYS["Left"]:
            points[tip] = replace(points[tip], x=points[tip].x + 0.05)
        self.feed(0.18, hand=TrackedHand(1, "Left", tuple(points)))
        self.feed(0.24, sound=True)
        self.feed(0.30)
        self.finish()
        self.assertEqual(self.events, [])

    def test_camera_gap_cancels_pending_press(self):
        self.rest()
        self.feed(0.18, (20,))
        self.feed(0.24, (20,))
        self.feed(0.60, sound=True)
        for timestamp in (0.66, 0.72, 0.78, 0.84):
            self.feed(timestamp)
        self.assertEqual(self.events, [])

    def test_lift_direction_is_learned_instead_of_assuming_screen_down(self):
        self.engine = PressDetector(profile_for(direction=1), camera_offset=0.0)
        self.rest()
        self.feed(0.18, hand=make_hand((20,), direction=1))
        self.feed(0.24, hand=make_hand((20,), direction=1))
        self.feed(0.30, sound=True)
        self.finish()
        self.assertEqual([event.key for event in self.events], ["a"])

    def test_delayed_audio_can_match_a_recent_return(self):
        self.rest()
        self.feed(0.18, (20,))
        self.feed(0.24, (20,))
        self.feed(0.30)
        event = TapEvent(0.30, 20.0, -10.0, -30.0)
        self.events.extend(self.engine.update(HandSnapshot(0.36, (make_hand(),)), [event]))
        for timestamp in (0.42, 0.48, 0.54):
            self.feed(timestamp)
        self.assertEqual([item.key for item in self.events], ["a"])


class PressProfileTests(unittest.TestCase):
    def test_old_hand_label_profile_requires_new_setup(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            profile_for().save(path)
            data = json.loads(path.read_text())
            data["version"] = 1
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "Hand labels have changed"):
                PressProfile.load(path)

    def test_round_trip(self):
        original = profile_for()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            original.save(path)
            loaded = PressProfile.load(path)
        self.assertEqual(original, loaded)

    def test_invisible_lift_is_rejected(self):
        pose = stable_pose([make_hand()] * 8)
        with self.assertRaisesRegex(ValueError, "too small"):
            PressProfile.from_poses("Left", 0, True, pose, {tip: pose for tip in KEYS["Left"]})

    def test_motion_during_recording_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "moved"):
            stable_pose([make_hand()] * 4 + [make_hand(shift=0.1)] * 4)


class AccuracyCheckTests(unittest.TestCase):
    def test_missed_wrong_extra_and_idle_keys_are_counted(self):
        check = AccuracyCheck(["a", "s", "d", "f"], 0.0)
        check.events = [(1, "a"), (11, "a"), (19, "a"), (23, "f"), (24, "f")]
        report = check.report(36)
        self.assertTrue(report["complete"])
        self.assertEqual(report["correct"], 1)
        self.assertEqual(report["missed"], 1)
        self.assertEqual(report["extra_keys"], 1)
        self.assertEqual(report["false_activations"], 1)

    def test_aborted_test_does_not_count_unseen_prompts_as_misses(self):
        check = AccuracyCheck(["a", "s", "d", "f"], 0.0)
        report = check.report(15)
        self.assertFalse(report["complete"])
        self.assertEqual(len(report["attempts"]), 1)
        self.assertEqual(report["missed"], 1)


if __name__ == "__main__":
    unittest.main()

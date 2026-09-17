import unittest

from helper_classifier import UserKeyboardProfile
from helper_keyboard import KeyboardLayout
from user_key_calibration import build_schedule, training_keys
from helper_training import TrainingSession, training_candidate
from helper_classifier import KeyTrainingSample
from helper_fusion import FingerCandidate, FusionResult
from helper_audio import TapEvent


class PersonalCalibrationTests(unittest.TestCase):
    def test_common_training_set_contains_letters_and_main_actions(self):
        names = training_keys(KeyboardLayout(), "common")

        self.assertEqual(len(names), len(set(names)))
        self.assertTrue(set("abcdefghijklmnopqrstuvwxyz").issubset(names))
        self.assertTrue({"space", "backspace", "enter"}.issubset(names))

    def test_schedule_resumes_from_saved_counts(self):
        profile = UserKeyboardProfile()
        classifier_keys = training_keys(KeyboardLayout(), "common")
        first_key = classifier_keys[0]
        from helper_classifier import KeyTrainingSample

        profile.add(
            KeyTrainingSample(
                first_key,
                1.0,
                1.0,
                0.0,
                0.0,
                "Left",
                8,
                0.5,
                12.0,
                1.0,
            )
        )

        schedule = build_schedule(profile, classifier_keys, 2)

        self.assertEqual(schedule.count(first_key), 1)
        self.assertEqual(schedule.count(classifier_keys[1]), 2)

    def test_schedule_keeps_repetitions_on_same_key(self):
        self.assertEqual(build_schedule(UserKeyboardProfile(), ["a", "s"], 3),
                         ["a", "a", "a", "s", "s", "s"])


def training_sample(key="a", timestamp=10.0):
    return KeyTrainingSample(key, 2.25, 2.5, 0.0, 0.0, "Left", 20, 0.5, 12.0, timestamp)


class TrainingSessionTests(unittest.TestCase):
    def test_burst_cannot_save_or_advance_without_confirmation(self):
        profile = UserKeyboardProfile()
        session = TrainingSession(profile, ["a", "s"])
        self.assertTrue(session.stage(training_sample()))
        self.assertFalse(session.stage(training_sample(timestamp=10.2)))
        self.assertEqual(profile.samples, [])
        self.assertEqual(session.target, "a")
        self.assertTrue(session.confirm(11.0))
        self.assertFalse(session.confirm(11.0))
        self.assertFalse(session.stage(training_sample("s", 11.1)))
        self.assertTrue(session.stage(training_sample("s", 12.0)))
        session.confirm(13.0)
        self.assertIsNone(session.target)
        self.assertFalse(session.stage(training_sample("s", 20.0)))
        self.assertEqual(len(profile.samples), 2)

    def test_retry_discards_review_and_blocks_old_audio(self):
        session = TrainingSession(UserKeyboardProfile(), ["a"])
        session.stage(training_sample())
        session.retry(11.0)
        self.assertEqual(session.profile.samples, [])
        self.assertFalse(session.can_capture(10.9))
        self.assertTrue(session.can_capture(12.0))

    def test_undo_after_skip_restores_actual_sample_target(self):
        session = TrainingSession(UserKeyboardProfile(), ["a", "s", "d"])
        session.stage(training_sample())
        session.confirm(11.0)
        session.skip(12.0)
        self.assertEqual(session.target, "d")
        self.assertTrue(session.undo(13.0))
        self.assertEqual(session.target, "a")
        self.assertEqual(session.profile.samples, [])

    def test_far_or_stationary_finger_is_not_a_training_sample(self):
        keys = {key.name: key for key in KeyboardLayout().keys}
        candidate = FingerCandidate(1, "Left", 20, "pinky", keys["p"], 0.8,
                                    11.0, 1.5, 0.0, 0.0)
        tap = TapEvent(10.0, 12.0, -10.0, -22.0)
        result = FusionResult(tap, candidate, "Key accepted", (candidate,))
        chosen, _ = training_candidate(result, keys["a"])
        self.assertIsNone(chosen)

    def test_target_guidance_can_select_nearby_moving_finger(self):
        keys = {key.name: key for key in KeyboardLayout().keys}
        far = FingerCandidate(1, "Left", 8, "index", keys["f"], 1.0, 4.5, 2.5, 0.0, 0.0)
        near = FingerCandidate(1, "Left", 20, "pinky", keys["a"], 0.8, 2.25, 2.5, 0.0, 0.0)
        tap = TapEvent(10.0, 12.0, -10.0, -22.0)
        chosen, _ = training_candidate(FusionResult(tap, far, "Key accepted", (far, near)), keys["a"])
        self.assertEqual(chosen, near)


if __name__ == "__main__":
    unittest.main()

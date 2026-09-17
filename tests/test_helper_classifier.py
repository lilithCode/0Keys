import tempfile
import unittest
import json
from dataclasses import replace
from pathlib import Path

from helper_classifier import (
    KeyTrainingSample,
    PersonalKeyClassifier,
    UserKeyboardProfile,
)
from helper_fusion import FingerCandidate
from helper_keyboard import KeyboardCalibration, KeyboardLayout
from helper_classifier import profile_context


def candidate_for(
    layout: KeyboardLayout,
    key_name: str,
    keyboard_x: float,
    keyboard_y: float,
    finger_index: int = 8,
) -> FingerCandidate:
    keys = {key.name: key for key in layout.keys}
    return FingerCandidate(
        hand_id=1,
        handedness="Right",
        finger_index=finger_index,
        finger_name="index",
        key=keys[key_name],
        score=0.8,
        keyboard_x=keyboard_x,
        keyboard_y=keyboard_y,
        relative_x=keyboard_x - 5.0,
        relative_y=keyboard_y - 3.5,
    )


def sample_for(
    key_name: str,
    keyboard_x: float,
    keyboard_y: float,
    finger_index: int = 8,
) -> KeyTrainingSample:
    return KeyTrainingSample(
        key_name=key_name,
        keyboard_x=keyboard_x,
        keyboard_y=keyboard_y,
        relative_x=keyboard_x - 5.0,
        relative_y=keyboard_y - 3.5,
        handedness="Right",
        finger_index=finger_index,
        motion_score=0.8,
        audio_strength_db=18.0,
        timestamp=10.0,
    )


class UserKeyboardProfileTests(unittest.TestCase):
    def test_saves_and_loads_samples(self):
        profile = UserKeyboardProfile([sample_for("f", 4.4, 2.5)])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            profile.save(path)
            loaded = UserKeyboardProfile.load(path)

        self.assertEqual(loaded.count("f"), 1)
        self.assertAlmostEqual(loaded.samples[0].keyboard_x, 4.4)

    def test_rejects_nonfinite_profile_values(self):
        profile = UserKeyboardProfile([sample_for("f", float("nan"), 2.5)])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            profile.save(path)
            with self.assertRaisesRegex(ValueError, "Invalid sample"):
                UserKeyboardProfile.load(path)

    def test_legacy_profile_requires_retraining(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(json.dumps({"version": 1, "samples": []}))
            loaded = UserKeyboardProfile.load(path)
        self.assertFalse(loaded.compatible_with({}))

    def test_changed_calibration_or_timing_requires_retraining(self):
        calibration = KeyboardCalibration(((0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)))
        context = profile_context(calibration, 0, 70.0)
        profile = UserKeyboardProfile(context=context)
        self.assertTrue(profile.compatible_with(context))
        self.assertFalse(profile.compatible_with(profile_context(calibration, 0, 0.0)))
        calibration.flip_rows = True
        self.assertFalse(profile.compatible_with(profile_context(calibration, 0, 70.0)))


class PersonalKeyClassifierTests(unittest.TestCase):
    def setUp(self):
        self.layout = KeyboardLayout()

    def test_personal_sample_can_correct_geometric_key(self):
        profile = UserKeyboardProfile(
            [
                sample_for("f", 6.05, 2.5),
                sample_for("f", 6.10, 2.5),
                sample_for("f", 6.15, 2.5),
            ]
        )
        classifier = PersonalKeyClassifier(self.layout, profile)
        candidate = candidate_for(self.layout, "g", 6.10, 2.5)

        predictions = classifier.predict(candidate)

        self.assertEqual(predictions[0].key.name, "f")

    def test_finger_identity_affects_overlapping_samples(self):
        profile = UserKeyboardProfile(
            [
                sample_for("f", 5.75, 2.5, finger_index=8),
                sample_for("g", 5.75, 2.5, finger_index=12),
            ] * 3
        )
        classifier = PersonalKeyClassifier(self.layout, profile)
        candidate = candidate_for(
            self.layout,
            "g",
            5.75,
            2.5,
            finger_index=12,
        )

        predictions = classifier.predict(candidate)

        self.assertEqual(predictions[0].key.name, "g")

    def test_weights_are_normalized_before_truncating_choices(self):
        classifier = PersonalKeyClassifier(self.layout, UserKeyboardProfile())
        candidate = candidate_for(self.layout, "h", 6.25, 2.5)

        predictions = classifier.predict(candidate)

        self.assertEqual(len(predictions), 3)
        all_predictions = classifier.predict(candidate, limit=100)
        self.assertAlmostEqual(sum(item.probability for item in all_predictions), 1.0)
        self.assertEqual(predictions, all_predictions[:3])

    def test_compares_multiple_moving_fingers(self):
        profile = UserKeyboardProfile(
            [
                sample_for("a", 2.2, 2.5, finger_index=20),
                sample_for("s", 3.2, 2.5, finger_index=16),
            ] * 3
        )
        classifier = PersonalKeyClassifier(self.layout, profile)
        stronger_wrong_finger = candidate_for(
            self.layout,
            "s",
            3.2,
            2.5,
            finger_index=8,
        )
        weaker_matching_finger = candidate_for(
            self.layout,
            "a",
            2.2,
            2.5,
            finger_index=20,
        )
        stronger_wrong_finger = replace(stronger_wrong_finger, score=1.0)
        weaker_matching_finger = replace(weaker_matching_finger, score=0.8)

        predictions = classifier.predict_candidates(
            (stronger_wrong_finger, weaker_matching_finger)
        )

        self.assertEqual(predictions[0].key.name, "a")
        self.assertEqual(predictions[0].candidate.finger_index, 20)

    def test_remote_mislabeled_samples_cannot_steal_a_key(self):
        profile = UserKeyboardProfile([sample_for("p", 2.25, 2.5)] * 20)
        classifier = PersonalKeyClassifier(self.layout, profile)
        predictions = classifier.predict(candidate_for(self.layout, "a", 2.25, 2.5))
        self.assertEqual(predictions[0].key.name, "a")
        self.assertNotIn("p", [prediction.key.name for prediction in predictions])

    def test_one_correction_does_not_override_geometric_mapping(self):
        profile = UserKeyboardProfile([sample_for("f", 6.25, 2.5)])
        classifier = PersonalKeyClassifier(self.layout, profile)
        predictions = classifier.predict(candidate_for(self.layout, "g", 6.25, 2.5))
        self.assertEqual(predictions[0].key.name, "g")

    def test_equal_finger_evidence_abstains(self):
        classifier = PersonalKeyClassifier(self.layout, UserKeyboardProfile())
        predictions = classifier.predict_candidates((
            candidate_for(self.layout, "a", 2.25, 2.5),
            candidate_for(self.layout, "j", 8.25, 2.5),
        ))
        self.assertFalse(classifier.accepts(predictions))

    def test_clear_center_is_accepted_but_boundary_is_not(self):
        classifier = PersonalKeyClassifier(self.layout, UserKeyboardProfile())
        self.assertTrue(classifier.accepts(classifier.predict(
            candidate_for(self.layout, "a", 2.25, 2.5)
        )))
        self.assertFalse(classifier.accepts(classifier.predict(
            candidate_for(self.layout, "s", 2.75, 2.5)
        )))

    def test_extreme_outlier_does_not_move_learned_center(self):
        samples = [sample_for("f", x, 2.5) for x in (6.02, 6.06, 6.10, 6.14, 4.15)]
        classifier = PersonalKeyClassifier(self.layout, UserKeyboardProfile(samples))
        self.assertEqual(classifier.predict(
            candidate_for(self.layout, "g", 6.08, 2.5)
        )[0].key.name, "f")


if __name__ == "__main__":
    unittest.main()

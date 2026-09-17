from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import numpy as np

from helper_fusion import FingerCandidate
from helper_keyboard import KeyRegion, KeyboardCalibration, KeyboardLayout


def profile_context(calibration: KeyboardCalibration, camera: int, offset_ms: float) -> dict:
    return {
        "camera_points": calibration.camera_points.tolist(),
        "mirrored": calibration.mirrored,
        "flip_rows": calibration.flip_rows,
        "camera": camera,
        "camera_offset_ms": offset_ms,
        "handedness_convention": "physical_v1",
    }


def near_key(key: KeyRegion, x: float, y: float) -> bool:
    return (
        key.x - 0.65 <= x <= key.x + key.width + 0.65
        and key.y - 0.45 <= y <= key.y + key.height + 0.45
    )


@dataclass(frozen=True)
class KeyTrainingSample:
    key_name: str
    keyboard_x: float
    keyboard_y: float
    relative_x: float
    relative_y: float
    handedness: str
    finger_index: int
    motion_score: float
    audio_strength_db: float
    timestamp: float


@dataclass(frozen=True)
class KeyPrediction:
    key: KeyRegion
    candidate: FingerCandidate
    probability: float
    score: float


class UserKeyboardProfile:
    def __init__(
        self, samples: list[KeyTrainingSample] | None = None,
        context: dict | None = None, version: int = 2,
    ) -> None:
        self.samples = list(samples) if samples is not None else []
        self.context = context
        self.version = version

    def add(self, sample: KeyTrainingSample) -> None:
        self.samples.append(sample)

    def samples_for(self, key_name: str) -> list[KeyTrainingSample]:
        return [sample for sample in self.samples if sample.key_name == key_name]

    def count(self, key_name: str) -> int:
        return sum(sample.key_name == key_name for sample in self.samples)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": self.version,
            "context": self.context,
            "samples": [asdict(sample) for sample in self.samples],
        }
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        temporary_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        temporary_path.replace(path)

    @classmethod
    def load(cls, path: str | Path) -> "UserKeyboardProfile":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("version") not in (1, 2):
            raise ValueError("Unsupported keyboard profile version")
        samples = [KeyTrainingSample(**sample) for sample in data.get("samples", [])]
        keys = {key.name for key in KeyboardLayout().keys}
        for sample in samples:
            numbers = (
                sample.keyboard_x, sample.keyboard_y, sample.relative_x,
                sample.relative_y, sample.motion_score, sample.audio_strength_db,
                sample.timestamp,
            )
            if (
                sample.key_name not in keys
                or sample.finger_index not in (4, 8, 12, 16, 20)
                or sample.handedness not in ("Left", "Right")
                or not all(math.isfinite(value) for value in numbers)
            ):
                raise ValueError("Invalid sample in keyboard profile")
        return cls(samples, data.get("context"), data["version"])

    def compatible_with(self, context: dict) -> bool:
        return self.version == 2 and self.context == context


class PersonalKeyClassifier:
    def __init__(self, layout: KeyboardLayout, profile: UserKeyboardProfile) -> None:
        self.layout = layout
        self.profile = profile

    @staticmethod
    def _spread(values: np.ndarray, minimum: float) -> float:
        if values.size < 2:
            return minimum
        deviation = float(np.median(np.abs(values - np.median(values)))) * 1.4826
        return min(max(deviation, minimum), minimum * 2)

    def _trained_score(
        self,
        key: KeyRegion,
        samples: list[KeyTrainingSample],
        candidate: FingerCandidate,
    ) -> float:
        absolute_x = np.asarray([sample.keyboard_x for sample in samples])
        absolute_y = np.asarray([sample.keyboard_y for sample in samples])
        relative_x = np.asarray([sample.relative_x for sample in samples])
        relative_y = np.asarray([sample.relative_y for sample in samples])

        absolute_distance = (
            (candidate.keyboard_x - float(np.median(absolute_x)))
            / self._spread(absolute_x, max(0.25, key.width * 0.20))
        ) ** 2 + (
            (candidate.keyboard_y - float(np.median(absolute_y)))
            / self._spread(absolute_y, 0.22)
        ) ** 2
        relative_distance = (
            (candidate.relative_x - float(np.median(relative_x)))
            / self._spread(relative_x, 0.45)
        ) ** 2 + (
            (candidate.relative_y - float(np.median(relative_y)))
            / self._spread(relative_y, 0.35)
        ) ** 2

        return -0.65 * absolute_distance - 0.15 * min(relative_distance, 9.0)

    def _geometric_score(self, key: KeyRegion, candidate: FingerCandidate) -> float:
        center_x, center_y = key.center
        x_distance = (candidate.keyboard_x - center_x) / max(key.width / 2, 0.5)
        y_distance = (candidate.keyboard_y - center_y) / 0.5
        score = -0.8 * (x_distance * x_distance + y_distance * y_distance)
        return score

    def predict(
        self, candidate: FingerCandidate, limit: int = 3
    ) -> tuple[KeyPrediction, ...]:
        return self.predict_candidates((candidate,), limit)

    def predict_candidates(
        self,
        candidates: tuple[FingerCandidate, ...],
        limit: int = 3,
    ) -> tuple[KeyPrediction, ...]:
        candidates = tuple(
            candidate for candidate in candidates
            if candidate.score > 0 and all(math.isfinite(value) for value in (
                candidate.score, candidate.keyboard_x, candidate.keyboard_y,
                candidate.relative_x, candidate.relative_y,
            ))
        )
        if not candidates:
            return ()

        maximum_motion = max(candidate.score for candidate in candidates)
        scored = []
        for key in self.layout.keys:
            samples = [
                sample for sample in self.profile.samples_for(key.name)
                if self.profile.version == 2
                and near_key(key, sample.keyboard_x, sample.keyboard_y)
            ]
            best_candidate = candidates[0]
            best_score = -math.inf
            for candidate in candidates:
                if not near_key(key, candidate.keyboard_x, candidate.keyboard_y):
                    continue
                matching = [
                    sample for sample in samples
                    if sample.handedness == candidate.handedness
                    and sample.finger_index == candidate.finger_index
                ]
                geometric = self._geometric_score(key, candidate)
                key_score = geometric
                if len(matching) >= 3:
                    key_score = self._trained_score(key, matching, candidate)
                elif len(samples) >= 3:
                    key_score -= 1.0
                motion_ratio = candidate.score / max(maximum_motion, 1e-9)
                score = key_score + 2.0 * math.log(max(motion_ratio, 1e-9))
                if score > best_score:
                    best_score = score
                    best_candidate = candidate
            if math.isfinite(best_score):
                scored.append((key, best_candidate, best_score))

        scored.sort(key=lambda item: item[2], reverse=True)
        if not scored:
            return ()
        selected = scored[: max(1, limit)]
        maximum = selected[0][2]
        weights = [math.exp(score - maximum) for _, _, score in scored]
        total = sum(weights)
        return tuple(
            KeyPrediction(
                key=key,
                candidate=candidate,
                probability=weight / total,
                score=score,
            )
            for (key, candidate, score), weight in zip(selected, weights)
        )

    @staticmethod
    def accepts(predictions: tuple[KeyPrediction, ...]) -> bool:
        if not predictions or predictions[0].score < -4.0:
            return False
        margin = (
            predictions[0].score - predictions[1].score
            if len(predictions) > 1 else math.inf
        )
        return predictions[0].probability >= 0.55 and margin >= 0.7

    @staticmethod
    def sample_from_candidate(
        key_name: str,
        candidate: FingerCandidate,
        audio_strength_db: float,
        timestamp: float,
    ) -> KeyTrainingSample:
        return KeyTrainingSample(
            key_name=key_name,
            keyboard_x=candidate.keyboard_x,
            keyboard_y=candidate.keyboard_y,
            relative_x=candidate.relative_x,
            relative_y=candidate.relative_y,
            handedness=candidate.handedness,
            finger_index=candidate.finger_index,
            motion_score=candidate.score,
            audio_strength_db=audio_strength_db,
            timestamp=timestamp,
        )

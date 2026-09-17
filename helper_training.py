from __future__ import annotations

from dataclasses import dataclass

from helper_classifier import KeyTrainingSample, UserKeyboardProfile, near_key
from helper_fusion import FingerCandidate, FusionResult
from helper_keyboard import KeyRegion


def training_candidate(
    result: FusionResult, target: KeyRegion, minimum_motion: float = 0.04,
) -> tuple[FingerCandidate | None, str]:
    if not result.candidates:
        return None, result.reason
    best_motion = max(candidate.score for candidate in result.candidates)
    nearby = [
        candidate for candidate in result.candidates
        if candidate.score >= minimum_motion
        and candidate.score >= best_motion * 0.5
        and near_key(target, candidate.keyboard_x, candidate.keyboard_y)
    ]
    nearby.sort(key=lambda candidate: candidate.score, reverse=True)
    if not nearby:
        return None, "No clear tap near the target. Check the overlay and tap again."
    if len(nearby) > 1 and nearby[0].score - nearby[1].score < 0.02:
        return None, "Two fingers moved near the target. Try one finger."
    return nearby[0], "Check the marked finger. Space saves. R retries."


@dataclass
class TrainingSession:
    profile: UserKeyboardProfile
    schedule: list[str]
    index: int = 0
    ready_at: float = 0.0
    pending: KeyTrainingSample | None = None

    def __post_init__(self) -> None:
        self.accepted: list[tuple[int, KeyTrainingSample]] = []

    @property
    def target(self) -> str | None:
        return self.schedule[self.index] if self.index < len(self.schedule) else None

    def can_capture(self, timestamp: float) -> bool:
        return self.target is not None and self.pending is None and timestamp >= self.ready_at

    def stage(self, sample: KeyTrainingSample) -> bool:
        if not self.can_capture(sample.timestamp) or sample.key_name != self.target:
            return False
        self.pending = sample
        return True

    def confirm(self, now: float) -> bool:
        if self.pending is None:
            return False
        self.profile.add(self.pending)
        self.accepted.append((self.index, self.pending))
        self.index += 1
        self.retry(now)
        return True

    def retry(self, now: float) -> None:
        self.pending = None
        self.ready_at = now + 0.9

    def skip(self, now: float) -> None:
        if self.target is not None:
            self.index += 1
        self.retry(now)

    def undo(self, now: float) -> bool:
        if not self.accepted:
            return False
        self.index, sample = self.accepted.pop()
        for index in range(len(self.profile.samples) - 1, -1, -1):
            if self.profile.samples[index] is sample:
                self.profile.samples.pop(index)
                break
        self.retry(now)
        return True

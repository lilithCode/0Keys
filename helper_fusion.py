from __future__ import annotations

from dataclasses import dataclass
import math

from helper_audio import TapEvent
from helper_keyboard import KeyRegion, KeyboardCalibration, KeyboardLayout
from helper_vision import FINGERTIPS, HandHistory, HandSnapshot, Landmark, TrackedHand


SHIFTED_VALUES = {
    "`": "~",
    "1": "!",
    "2": "@",
    "3": "#",
    "4": "$",
    "5": "%",
    "6": "^",
    "7": "&",
    "8": "*",
    "9": "(",
    "0": ")",
    "-": "_",
    "=": "+",
    "[": "{",
    "]": "}",
    "\\": "|",
    ";": ":",
    "'": '"',
    ",": "<",
    ".": ">",
    "/": "?",
}

FINGER_MOTION_GAIN = {
    4: 1.10,
    8: 1.00,
    12: 1.00,
    16: 1.15,
    20: 1.35,
}


@dataclass(frozen=True)
class FingerCandidate:
    hand_id: int
    handedness: str
    finger_index: int
    finger_name: str
    key: KeyRegion
    score: float
    keyboard_x: float
    keyboard_y: float
    relative_x: float
    relative_y: float


@dataclass(frozen=True)
class FusionResult:
    tap: TapEvent
    candidate: FingerCandidate | None
    reason: str
    candidates: tuple[FingerCandidate, ...] = ()

    @property
    def accepted(self) -> bool:
        return self.candidate is not None


class TapFingerFusion:
    def __init__(
        self,
        layout: KeyboardLayout,
        calibration: KeyboardCalibration,
        camera_offset_seconds: float = 0.07,
        post_motion_seconds: float = 0.12,
        minimum_motion_score: float = 0.04,
        minimum_score_margin: float = 0.015,
    ) -> None:
        self.layout = layout
        self.calibration = calibration
        self.camera_offset_seconds = camera_offset_seconds
        self.post_motion_seconds = post_motion_seconds
        self.minimum_motion_score = minimum_motion_score
        self.minimum_score_margin = minimum_score_margin

    def plausible_candidates(self, result: FusionResult) -> tuple[FingerCandidate, ...]:
        if not result.candidates:
            return ()
        best_motion = max(candidate.score for candidate in result.candidates)
        return tuple(
            candidate for candidate in result.candidates
            if math.isfinite(candidate.score)
            and candidate.score >= self.minimum_motion_score
            and candidate.score >= best_motion * 0.5
        )

    @staticmethod
    def _hand(snapshot: HandSnapshot, hand_id: int) -> TrackedHand | None:
        return next((hand for hand in snapshot.hands if hand.hand_id == hand_id), None)

    @staticmethod
    def _relative_velocity(
        tip_velocity: Landmark | None, wrist_velocity: Landmark | None
    ) -> Landmark | None:
        if tip_velocity is None or wrist_velocity is None:
            return None
        return Landmark(
            tip_velocity.x - wrist_velocity.x,
            tip_velocity.y - wrist_velocity.y,
            tip_velocity.z - wrist_velocity.z,
        )

    def select(self, tap: TapEvent, history: HandHistory) -> FusionResult:
        contact_time = tap.timestamp + self.camera_offset_seconds
        contact = history.nearest(contact_time, max_difference=0.12)
        after = history.nearest(
            contact_time + self.post_motion_seconds,
            max_difference=0.12,
        )
        if contact is None or after is None:
            return FusionResult(tap, None, "No synchronized camera frame")
        if after.timestamp <= contact.timestamp:
            return FusionResult(tap, None, "No post-tap camera frame")
        if not contact.hands:
            return FusionResult(tap, None, "No hand was visible")

        candidates = []
        for hand in contact.hands:
            after_hand = self._hand(after, hand.hand_id)
            if after_hand is None:
                continue
            wrist = hand.landmarks[0]
            wrist_keyboard = self.calibration.map_to_keyboard(wrist.x, wrist.y)
            wrist_before = history.tip_velocity(contact, hand.hand_id, 0, 0.12)
            wrist_after = history.tip_velocity(after, hand.hand_id, 0, self.post_motion_seconds)

            for tip_index, tip_name in FINGERTIPS.items():
                tip = hand.landmarks[tip_index]
                keyboard_point = self.calibration.map_to_keyboard(tip.x, tip.y)
                key = self.layout.key_at(*keyboard_point)
                if key is None:
                    continue

                velocity_before = self._relative_velocity(
                    history.tip_velocity(contact, hand.hand_id, tip_index, 0.12),
                    wrist_before,
                )
                velocity_after = self._relative_velocity(
                    history.tip_velocity(
                        after,
                        hand.hand_id,
                        tip_index,
                        self.post_motion_seconds,
                    ),
                    wrist_after,
                )
                if velocity_before is None or velocity_after is None:
                    continue

                downward = max(0.0, velocity_before.y)
                upward = max(0.0, -velocity_after.y)
                depth_motion = 0.15 * (
                    abs(velocity_before.z) + abs(velocity_after.z)
                )
                score = (
                    downward + 0.8 * upward + depth_motion
                ) * FINGER_MOTION_GAIN[tip_index]
                candidates.append(
                    FingerCandidate(
                        hand_id=hand.hand_id,
                        handedness=hand.handedness,
                        finger_index=tip_index,
                        finger_name=tip_name,
                        key=key,
                        score=score,
                        keyboard_x=keyboard_point[0],
                        keyboard_y=keyboard_point[1],
                        relative_x=keyboard_point[0] - wrist_keyboard[0],
                        relative_y=keyboard_point[1] - wrist_keyboard[1],
                    )
                )

        if not candidates:
            return FusionResult(tap, None, "No fingertip was inside the keyboard")

        candidates.sort(key=lambda item: item.score, reverse=True)
        best = candidates[0]
        ranked = tuple(candidates)
        if not math.isfinite(best.score) or best.score < self.minimum_motion_score:
            return FusionResult(tap, None, "Fingertip motion was too small", ranked)
        if len(candidates) > 1:
            margin = best.score - candidates[1].score
            if margin < self.minimum_score_margin:
                return FusionResult(tap, None, "Fingertip motion was ambiguous", ranked)
        return FusionResult(tap, best, "Key accepted", ranked)


class TextComposer:
    def __init__(self, maximum_characters: int = 2_000) -> None:
        self.maximum_characters = maximum_characters
        self.text = ""
        self.caps_lock = False
        self.shift = False

    def clear(self) -> None:
        self.text = ""
        self.caps_lock = False
        self.shift = False

    def apply(self, key: KeyRegion) -> str:
        value = key.value
        if value == "BACKSPACE":
            self.text = self.text[:-1]
        elif value == "ENTER":
            self.text += "\n"
        elif value == "TAB":
            self.text += "    "
        elif value == "CAPS_LOCK":
            self.caps_lock = not self.caps_lock
        elif value == "SHIFT":
            self.shift = not self.shift
        elif value not in {"CONTROL", "ALT"}:
            character = value
            if len(character) == 1 and character.isalpha():
                if self.caps_lock != self.shift:
                    character = character.upper()
            elif self.shift:
                character = SHIFTED_VALUES.get(character, character)
            self.text += character
            self.shift = False

        if len(self.text) > self.maximum_characters:
            self.text = self.text[-self.maximum_characters :]
        return self.text

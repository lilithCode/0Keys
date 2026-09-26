from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
from collections import deque
import math

import numpy as np


FINGERTIPS = {
    4: "thumb",
    8: "index",
    12: "middle",
    16: "ring",
    20: "pinky",
}

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


@dataclass(frozen=True)
class Landmark:
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class DetectedHand:
    handedness: str
    landmarks: tuple[Landmark, ...]


@dataclass(frozen=True)
class TrackedHand:
    hand_id: int
    handedness: str
    landmarks: tuple[Landmark, ...]


@dataclass(frozen=True)
class HandSnapshot:
    timestamp: float
    hands: tuple[TrackedHand, ...]


@dataclass
class _Track:
    hand_id: int
    handedness: str
    wrist: Landmark
    last_seen: float


class HandIdentityAssigner:
    def __init__(self, maximum_wrist_distance: float = 0.35, timeout: float = 0.75):
        self.maximum_wrist_distance = maximum_wrist_distance
        self.timeout = timeout
        self._tracks: dict[int, _Track] = {}
        self._next_id = 1

    @staticmethod
    def _distance(a: Landmark, b: Landmark) -> float:
        return math.hypot(a.x - b.x, a.y - b.y)

    def assign(
        self, detections: Sequence[DetectedHand], timestamp: float
    ) -> tuple[TrackedHand, ...]:
        self._tracks = {
            track_id: track
            for track_id, track in self._tracks.items()
            if timestamp - track.last_seen <= self.timeout
        }

        pairs: list[tuple[float, float, int, int]] = []
        for detection_index, detection in enumerate(detections):
            wrist = detection.landmarks[0]
            for track_id, track in self._tracks.items():
                distance = self._distance(wrist, track.wrist)
                handedness_penalty = (
                    0.0 if detection.handedness == track.handedness else 0.20
                )
                pairs.append(
                    (distance + handedness_penalty, distance, detection_index, track_id)
                )

        assignments: dict[int, int] = {}
        used_tracks: set[int] = set()
        for _, distance, detection_index, track_id in sorted(pairs):
            if distance > self.maximum_wrist_distance:
                continue
            if detection_index in assignments or track_id in used_tracks:
                continue
            assignments[detection_index] = track_id
            used_tracks.add(track_id)

        tracked_hands: list[TrackedHand] = []
        for detection_index, detection in enumerate(detections):
            hand_id = assignments.get(detection_index)
            if hand_id is None:
                hand_id = self._next_id
                self._next_id += 1

            wrist = detection.landmarks[0]
            self._tracks[hand_id] = _Track(
                hand_id=hand_id,
                handedness=detection.handedness,
                wrist=wrist,
                last_seen=timestamp,
            )
            tracked_hands.append(
                TrackedHand(
                    hand_id=hand_id,
                    handedness=detection.handedness,
                    landmarks=detection.landmarks,
                )
            )
        return tuple(tracked_hands)


class HandHistory:
    def __init__(self, history_seconds: float = 0.75, maximum_frames: int = 120):
        self.history_seconds = history_seconds
        self._frames: deque[HandSnapshot] = deque(maxlen=maximum_frames)

    def append(self, snapshot: HandSnapshot) -> None:
        if self._frames and snapshot.timestamp < self._frames[-1].timestamp:
            raise ValueError("Hand snapshot timestamps must be monotonic")
        self._frames.append(snapshot)
        cutoff = snapshot.timestamp - self.history_seconds
        while self._frames and self._frames[0].timestamp < cutoff:
            self._frames.popleft()

    def nearest(self, timestamp: float, max_difference: float = 0.12) -> HandSnapshot | None:
        if not self._frames:
            return None
        snapshot = min(self._frames, key=lambda item: abs(item.timestamp - timestamp))
        if abs(snapshot.timestamp - timestamp) > max_difference:
            return None
        return snapshot

    @staticmethod
    def _find_hand(snapshot: HandSnapshot, hand_id: int) -> TrackedHand | None:
        return next((hand for hand in snapshot.hands if hand.hand_id == hand_id), None)

    def tip_velocity(
        self,
        snapshot: HandSnapshot,
        hand_id: int,
        tip_index: int,
        lookback_seconds: float = 0.08,
    ) -> Landmark | None:
        current_hand = self._find_hand(snapshot, hand_id)
        if current_hand is None or tip_index >= len(current_hand.landmarks):
            return None

        candidates = [
            frame
            for frame in self._frames
            if frame.timestamp < snapshot.timestamp
            and self._find_hand(frame, hand_id) is not None
        ]
        if not candidates:
            return None
        target_time = snapshot.timestamp - lookback_seconds
        previous = min(candidates, key=lambda item: abs(item.timestamp - target_time))
        delta_time = snapshot.timestamp - previous.timestamp
        if delta_time <= 0.01 or delta_time > 0.25:
            return None

        previous_hand = self._find_hand(previous, hand_id)
        assert previous_hand is not None
        current = current_hand.landmarks[tip_index]
        old = previous_hand.landmarks[tip_index]
        return Landmark(
            x=(current.x - old.x) / delta_time,
            y=(current.y - old.y) / delta_time,
            z=(current.z - old.z) / delta_time,
        )


class MediaPipeHandTracker:
    def __init__(
        self,
        model_path: str | Path,
        number_of_hands: int = 2,
        detection_confidence: float = 0.55,
        presence_confidence: float = 0.55,
        tracking_confidence: float = 0.55,
        input_mirrored: bool = False,
    ) -> None:
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(
                f"MediaPipe model not found: {model_path}"
            )

        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError(
                "MediaPipe is not installed. Install requirements.txt."
            ) from exc

        self._mp = mp
        self.input_mirrored = input_mirrored
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=number_of_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=presence_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
        self._identity = HandIdentityAssigner()
        self._last_mediapipe_timestamp = -1

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "MediaPipeHandTracker":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def process(self, bgr_frame: np.ndarray, capture_time: float) -> HandSnapshot:
        rgb_frame = np.ascontiguousarray(bgr_frame[:, :, ::-1])
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=rgb_frame,
        )
        timestamp_ms = max(
            self._last_mediapipe_timestamp + 1,
            int(capture_time * 1_000),
        )
        self._last_mediapipe_timestamp = timestamp_ms
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        detections: list[DetectedHand] = []
        for index, hand_landmarks in enumerate(result.hand_landmarks):
            handedness = "Unknown"
            if index < len(result.handedness) and result.handedness[index]:
                category = result.handedness[index][0]
                handedness = category.category_name or category.display_name or "Unknown"
            if self.input_mirrored:
                handedness = {"Left": "Right", "Right": "Left"}.get(handedness, handedness)
            landmarks = tuple(
                Landmark(x=float(point.x), y=float(point.y), z=float(point.z))
                for point in hand_landmarks
            )
            if len(landmarks) == 21:
                detections.append(
                    DetectedHand(handedness=handedness, landmarks=landmarks)
                )

        hands = self._identity.assign(detections, capture_time)
        return HandSnapshot(timestamp=capture_time, hands=hands)

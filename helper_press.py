from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path

import numpy as np

from helper_audio import TapEvent
from helper_vision import HandSnapshot, TrackedHand


FINGER_BASES = {8: 5, 12: 9, 16: 13, 20: 17}
KEYS = {"Left": {20: "a", 16: "s", 12: "d", 8: "f"},
        "Right": {8: "j", 12: "k", 16: "l", 20: ";"}}


def hand_measurement(hand: TrackedHand) -> dict:
    points = np.asarray([(p.x, p.y, p.z) for p in hand.landmarks], dtype=float)
    if points.shape != (21, 3) or not np.isfinite(points).all():
        raise ValueError("Hand landmarks are missing or invalid")
    scale = float(np.linalg.norm(points[9, :2] - points[0, :2]))
    if scale < 0.04:
        raise ValueError("Move your hand closer to the camera")
    features = {}
    for tip, base in FINGER_BASES.items():
        feature = (points[tip] - points[base]) / scale
        feature[2] *= 0.35
        features[tip] = feature
    return {"scale": scale, "anchors": points[[0, 5, 17], :2],
            "tips": {tip: points[tip, :2] for tip in FINGER_BASES}, "features": features}


def stable_pose(hands: list[TrackedHand]) -> dict:
    if len(hands) < 8:
        raise ValueError("Hold still for eight camera frames")
    if len({(hand.hand_id, hand.handedness) for hand in hands}) != 1:
        raise ValueError("Keep the same hand visible")
    readings = [hand_measurement(hand) for hand in hands]
    scale = float(np.median([item["scale"] for item in readings]))
    anchors = np.asarray([item["anchors"] for item in readings])
    center = np.median(anchors, axis=0)
    if float(np.max(np.linalg.norm(anchors - center, axis=2))) > scale * 0.12:
        raise ValueError("Your hand moved. Hold the pose still and try again")
    features = {}
    tips = {}
    for tip in FINGER_BASES:
        values = np.asarray([item["features"][tip] for item in readings])
        median = np.median(values, axis=0)
        if float(np.max(np.linalg.norm(values - median, axis=1))) > 0.10:
            raise ValueError("Finger movement was too noisy. Hold still and try again")
        features[tip] = median
        tips[tip] = np.median([item["tips"][tip] for item in readings], axis=0)
    return {"scale": scale, "anchors": center, "features": features, "tips": tips}


@dataclass
class FingerProfile:
    key: str
    tip: int
    rest: list[float]
    lift: list[float]
    position: list[float]


@dataclass
class PressProfile:
    handedness: str
    camera: int | str
    mirrored: bool
    scale: float
    anchors: list[list[float]]
    fingers: list[FingerProfile]

    @classmethod
    def from_poses(cls, handedness: str, camera: int | str, mirrored: bool,
                   rest: dict, lifts: dict[int, dict]) -> "PressProfile":
        fingers = []
        for tip, key in KEYS[handedness].items():
            pose = lifts[tip]
            if np.max(np.linalg.norm(pose["anchors"] - rest["anchors"], axis=1)) > rest["scale"] * 0.25:
                raise ValueError("Your palm moved during setup. Start again with R")
            axis = pose["features"][tip] - rest["features"][tip]
            if float(np.linalg.norm(axis)) < 0.12:
                raise ValueError(f"Lift for {key.upper()} was too small to see. Try a clearer lift")
            fingers.append(FingerProfile(key, tip, rest["features"][tip].tolist(),
                                         axis.tolist(), rest["tips"][tip].tolist()))
        return cls(handedness, camera, mirrored, rest["scale"], rest["anchors"].tolist(), fingers)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"version": 2, **asdict(self)}, indent=2) + "\n")
        temporary.replace(path)

    @classmethod
    def load(cls, path: str | Path) -> "PressProfile":
        data = json.loads(Path(path).read_text())
        if data.pop("version", None) != 2:
            raise ValueError("Hand labels have changed. Repeat setup to record the correct hand")
        data["fingers"] = [FingerProfile(**item) for item in data["fingers"]]
        profile = cls(**data)
        if profile.handedness not in KEYS or not math.isfinite(profile.scale) or profile.scale < 0.04:
            raise ValueError("Invalid hand in press profile")
        if np.asarray(profile.anchors).shape != (3, 2) or not np.isfinite(profile.anchors).all():
            raise ValueError("Invalid hand anchors")
        if {finger.tip: finger.key for finger in profile.fingers} != KEYS[profile.handedness] or len(profile.fingers) != 4:
            raise ValueError("Press profile needs four distinct fingers")
        for finger in profile.fingers:
            for values, length in ((finger.rest, 3), (finger.lift, 3), (finger.position, 2)):
                if np.asarray(values).shape != (length,) or not np.isfinite(values).all():
                    raise ValueError("Invalid finger in press profile")
            if np.linalg.norm(finger.lift) < 0.12:
                raise ValueError("Finger lift is too small")
        return profile


@dataclass
class FingerState:
    phase: str = "find rest"
    rest_since: float | None = None
    lift_since: float | None = None
    progress: float = 0.0


@dataclass(frozen=True)
class Contact:
    tip: int
    key: str
    timestamp: float


@dataclass(frozen=True)
class PressEvent:
    key: str
    tip: int
    audio_time: float
    contact_time: float


class PressDetector:
    def __init__(self, profile: PressProfile, camera_offset: float = 0.07, allow_hover: bool = False):
        self.profile = profile
        self.camera_offset = camera_offset
        self.allow_hover = allow_hover
        self.states = {finger.tip: FingerState() for finger in profile.fingers}
        self.contacts: list[Contact] = []
        self.audio: list[TapEvent] = []
        self.last_time: float | None = None
        self.hand_id: int | None = None
        self.status = "Rest your hand at the marked positions"
        self.match_window = 0.10
        self.settle_seconds = 0.03
        self.last_matches: dict[float, int] = {}

    def reset(self, message: str = "Rest your hand at the marked positions") -> None:
        self.states = {finger.tip: FingerState() for finger in self.profile.fingers}
        self.contacts.clear()
        self.audio.clear()
        self.last_time = None
        self.hand_id = None
        self.status = message
        self.last_matches.clear()

    def update(self, snapshot: HandSnapshot, taps: list[TapEvent]) -> list[PressEvent]:
        self.last_matches.clear()
        now = snapshot.timestamp
        if self.last_time is not None and now <= self.last_time:
            raise ValueError("Camera timestamps must increase")
        if self.last_time is not None and now - self.last_time > 0.25:
            self.reset("Camera gap. Rest your fingers to rearm")
        self.last_time = now
        hands = [hand for hand in snapshot.hands if hand.handedness == self.profile.handedness]
        if len(hands) != 1:
            self.reset("Selected hand is not clearly visible")
            return []
        hand = hands[0]
        if self.hand_id is not None and hand.hand_id != self.hand_id:
            self.reset("Hand tracking changed. Rest to rearm")
        self.hand_id = hand.hand_id
        try:
            measured = hand_measurement(hand)
        except ValueError as exc:
            self.reset(str(exc))
            return []
        drift = np.max(np.linalg.norm(measured["anchors"] - np.asarray(self.profile.anchors), axis=1))
        if drift > self.profile.scale * 0.25 or not 0.8 <= measured["scale"] / self.profile.scale <= 1.2:
            self.reset("Hand moved from home. Return to the markers or press R")
            return []

        self.status = ("Hovering is safe. Tap one finger back at its marker" if self.allow_hover
                       else "Resting fingers are safe. Lift then tap once")
        for finger in self.profile.fingers:
            state = self.states[finger.tip]
            axis = np.asarray(finger.lift)
            delta = measured["features"][finger.tip] - np.asarray(finger.rest)
            progress = float(np.dot(delta, axis) / np.dot(axis, axis))
            sideways = float(np.linalg.norm(delta - progress * axis))
            state.progress = progress
            at_home = np.linalg.norm(measured["tips"][finger.tip] - finger.position) <= self.profile.scale * 0.28
            resting = abs(progress) <= 0.25 and sideways <= 0.09 and at_home
            lifted = progress >= 0.65 and sideways <= 0.12
            if sideways > 0.18 or progress < -0.5 or progress > 2.0:
                self.states[finger.tip] = FingerState()
                continue
            if state.phase == "find rest":
                if self.allow_hover and lifted:
                    state.rest_since = None
                    if state.lift_since is None:
                        state.lift_since = now
                    elif now - state.lift_since >= 0.12:
                        state.phase = "lifted"
                    continue
                state.lift_since = None
                if resting:
                    if state.rest_since is None:
                        state.rest_since = now
                    elif now - state.rest_since >= 0.1:
                        state.phase = "resting"
                else:
                    state.rest_since = None
            elif state.phase == "resting":
                if lifted:
                    state.phase = "lifting"
                    state.lift_since = now
                elif not resting and abs(progress) < 0.3:
                    self.states[finger.tip] = FingerState()
            elif state.phase == "lifting":
                if lifted and now - state.lift_since >= 0.04:
                    state.phase = "lifted"
                elif resting or now - state.lift_since > 0.4:
                    self.states[finger.tip] = FingerState()
            elif state.phase == "lifted":
                if not self.allow_hover and now - state.lift_since > 1.5:
                    self.states[finger.tip] = FingerState()
                elif resting:
                    self.contacts.append(Contact(finger.tip, finger.key, now))
                    self.states[finger.tip] = FingerState(rest_since=now)

        self.audio.extend(tap for tap in taps if math.isfinite(tap.timestamp))
        events = []
        remaining = []
        for tap in self.audio:
            target_time = tap.timestamp + self.camera_offset
            if now < target_time + self.match_window + self.settle_seconds:
                remaining.append(tap)
                continue
            matches = [contact for contact in self.contacts
                       if abs(contact.timestamp - target_time) <= self.match_window]
            self.last_matches[tap.timestamp] = len(matches)
            if len(matches) == 1:
                contact = matches[0]
                events.append(PressEvent(contact.key, contact.tip, tap.timestamp, contact.timestamp))
                self.contacts.remove(contact)
                self.status = f"Pressed {contact.key.upper()}"
            elif len(matches) > 1:
                for contact in matches:
                    self.contacts.remove(contact)
                self.status = "Two fingers returned together. Tap one finger at a time"
            else:
                self.status = "Sound ignored: no matching lift and return"
        self.audio = remaining
        oldest = now - self.match_window * 2 - self.settle_seconds - abs(self.camera_offset)
        expired = [contact for contact in self.contacts if contact.timestamp < oldest]
        if expired:
            self.status = "Finger returned but no tap sound matched"
        self.contacts = [contact for contact in self.contacts if contact.timestamp >= oldest]
        return events


class TwoHandPressDetector:
    def __init__(self, profiles: dict[str, PressProfile], camera_offset: float = 0.0,
                 allow_hover: bool = False):
        if set(profiles) != {"Left", "Right"} or any(
            profile.handedness != side for side, profile in profiles.items()
        ):
            raise ValueError("Two-hand typing needs a profile for each physical hand")
        if len({(profile.camera, profile.mirrored) for profile in profiles.values()}) != 1:
            raise ValueError("Both profiles must use the same camera view")
        self.engines = {side: PressDetector(profile, camera_offset, allow_hover) for side, profile in profiles.items()}
        self.status = "Rest both hands at their markers"

    def reset(self, message: str = "Rest both hands at their markers") -> None:
        for engine in self.engines.values():
            engine.reset(message)
        self.status = message

    def update(self, snapshot: HandSnapshot, taps: list[TapEvent]) -> list[PressEvent]:
        candidates = []
        claims = {}
        for engine in self.engines.values():
            candidates.extend(engine.update(snapshot, taps))
            for timestamp, count in engine.last_matches.items():
                claims[timestamp] = claims.get(timestamp, 0) + count
        events = [event for event in candidates if claims.get(event.audio_time) == 1]
        if any(count > 1 for count in claims.values()):
            self.status = "Multiple fingers matched one sound. Tap one finger at a time"
        elif events:
            self.status = "Pressed " + " ".join(event.key.upper() for event in events)
        else:
            self.status = " | ".join(f"{side}: {engine.status}" for side, engine in self.engines.items())
        return sorted(events, key=lambda event: event.audio_time)


@dataclass
class AccuracyCheck:
    keys: list[str]
    started: float
    events: list[tuple[float, str]] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return 20.0 + len(self.keys) * 4.0

    def phase(self, now: float) -> tuple[str, str | None]:
        elapsed = now - self.started
        if elapsed < 10:
            return "Rest your fingers. Do not type", None
        index = int((elapsed - 10) // 4)
        if index >= len(self.keys):
            return "Rest and gently slide your hand. Do not type", None
        if (elapsed - 10) % 4 < 0.8:
            return "Get ready", self.keys[index]
        return "Tap once", self.keys[index]

    def report(self, now: float) -> dict:
        elapsed = max(0.0, now - self.started)
        attempts = []
        false_activations = 0
        for timestamp, _ in self.events:
            relative = timestamp - self.started
            if relative < 10 or relative >= 10 + 4 * len(self.keys) or (relative - 10) % 4 < 0.8:
                false_activations += 1
        for index, expected in enumerate(self.keys):
            start = self.started + 10 + 4 * index + 0.8
            end = self.started + 10 + 4 * (index + 1)
            if now < end:
                break
            actual = [key for timestamp, key in self.events if start <= timestamp < end]
            attempts.append({"expected": expected, "actual": actual,
                             "correct": actual == [expected]})
        return {"complete": elapsed >= self.duration, "attempts": attempts,
                "correct": sum(item["correct"] for item in attempts),
                "missed": sum(not item["actual"] for item in attempts),
                "extra_keys": sum(max(0, len(item["actual"]) - 1) for item in attempts),
                "false_activations": false_activations,
                "elapsed_seconds": min(elapsed, self.duration)}

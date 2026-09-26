from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np

from helper_keyboard import KeyRegion, KeyboardCalibration, SpacedKeyboardLayout
from helper_vision import HandSnapshot


@dataclass(frozen=True)
class AirPrediction:
    key: KeyRegion | None
    confidence: float
    reason: str


class AirKeyModel:
    def __init__(self, layout, context, samples=None):
        self.layout = layout
        self.context = context
        self.samples = list(samples or [])

    def add(self, key_name, point, handedness):
        key = next((key for key in self.layout.keys if key.name == key_name), None)
        if key is None or key.value in ("CONTROL", "ALT") or handedness not in ("Left", "Right"):
            raise ValueError("Choose a supported key and a visible hand")
        x, y = point
        if not all(math.isfinite(v) for v in point) or not (
            key.x <= x < key.x + key.width and key.y <= y < key.y + key.height
        ):
            raise ValueError("Example is outside the requested key. Aim inside its keycap")
        self.samples.append({"key": key_name, "x": float(x), "y": float(y), "hand": handedness})

    def predict(self, point, handedness):
        if not all(math.isfinite(v) for v in point):
            return AirPrediction(None, 0, "Invalid pointer")
        cap = self.layout.key_at(*point)
        if cap is None or cap.value in ("CONTROL", "ALT"):
            return AirPrediction(None, 0, "Gap or inactive key")
        scores = []
        for key in self.layout.keys:
            if key.value in ("CONTROL", "ALT"):
                continue
            center = np.asarray(key.center)
            spread = np.asarray([max(key.width * 0.32, 0.23), 0.25])
            examples = [item for item in self.samples if item["key"] == key.name and item["hand"] == handedness]
            if len(examples) >= 3:
                values = np.asarray([[item["x"], item["y"]] for item in examples])
                learned = np.median(values, axis=0)
                center += np.clip(learned - center, -0.18, 0.18)
                spread = np.clip(1.4826 * np.median(np.abs(values - learned), axis=0),
                                 [0.20, 0.18], spread)
            score = -0.5 * float(np.sum(((np.asarray(point) - center) / spread) ** 2))
            scores.append((score, key))
        scores.sort(key=lambda item: item[0], reverse=True)
        best, key = scores[0]
        weights = [math.exp(score - best) for score, _ in scores]
        confidence = 1 / sum(weights)
        margin = best - scores[1][0]
        if key.name != cap.name or confidence < 0.60 or margin < 0.8:
            return AirPrediction(None, confidence, "Uncertain. Aim nearer the key center")
        return AirPrediction(key, confidence, "Learned position model" if len(self.samples) else "Untrained position model")

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"version": 1, "context": self.context, "samples": self.samples}, indent=2) + "\n")
        temporary.replace(path)

    @classmethod
    def load(cls, path, layout, context):
        data = json.loads(Path(path).read_text())
        if not isinstance(data, dict) or data.get("version") != 1 or data.get("context") != context:
            raise ValueError("Saved air training uses another layout or camera view")
        model = cls(layout, context)
        for sample in data["samples"]:
            model.add(sample["key"], (sample["x"], sample["y"]), sample["hand"])
        return model


@dataclass(frozen=True)
class AirClick:
    key: KeyRegion
    handedness: str
    point: tuple[float, float]
    timestamp: float
    confidence: float


@dataclass
class PinchState:
    phase: str = "open fingers"
    key: KeyRegion | None = None
    point: tuple[float, float] | None = None
    anchor: tuple[float, float] | None = None
    confidence: float = 0
    since: float = 0
    closing_since: float | None = None
    last_open: float = 0
    release_since: float | None = None


class AirClickDetector:
    def __init__(self, layout: SpacedKeyboardLayout, calibration: KeyboardCalibration, model: AirKeyModel):
        self.layout = layout
        self.calibration = calibration
        self.model = model
        self.states = {}
        self.last_time = None
        self.status = "Point with index, then pinch thumb and index"

    def reset(self):
        self.states.clear()
        self.last_time = None

    def update(self, snapshot: HandSnapshot):
        now = snapshot.timestamp
        if self.last_time is not None and now <= self.last_time:
            raise ValueError("Camera timestamps must increase")
        if self.last_time is not None and now - self.last_time > 0.25:
            self.reset()
        self.last_time = now
        current = {(h.hand_id, h.handedness) for h in snapshot.hands}
        self.states = {identity: state for identity, state in self.states.items() if identity in current}
        clicks = []
        self.status = "Hovering does not type"
        for hand in snapshot.hands:
            identity = (hand.hand_id, hand.handedness)
            if (hand.handedness not in ("Left", "Right") or len(hand.landmarks) != 21
                    or sum(h.handedness == hand.handedness for h in snapshot.hands) != 1):
                self.states.pop(identity, None)
                continue
            points = np.asarray([(p.x, p.y) for p in hand.landmarks])
            scale = float(np.linalg.norm(points[5] - points[17]))
            if not np.isfinite(points).all() or scale < 0.04:
                self.states.pop(identity, None)
                continue
            ratio = float(np.linalg.norm(points[4] - points[8]) / scale)
            pointer = self.calibration.map_to_keyboard(*points[8])
            anchor = tuple(self.calibration.map_to_keyboard(*points[0]))
            state = self.states.setdefault(identity, PinchState())
            if ratio >= 0.65:
                prediction = self.model.predict(pointer, hand.handedness)
                state.closing_since = None
                if state.phase == "held":
                    if state.release_since is None:
                        state.release_since = now
                    if now - state.release_since < 0.08:
                        continue
                    state = PinchState()
                    self.states[identity] = state
                key = prediction.key
                if key is None:
                    self.states[identity] = PinchState()
                    self.status = prediction.reason
                    continue
                if state.key is None or state.key.name != key.name or state.point is None or np.linalg.norm(np.asarray(pointer) - state.point) > 0.25:
                    state = PinchState(phase="aiming", key=key, point=pointer, anchor=anchor, since=now)
                    self.states[identity] = state
                state.confidence = prediction.confidence
                state.last_open = now
                if now - state.since >= 0.18:
                    state.phase = "ready"
                continue
            if state.phase == "held":
                state.release_since = None
                continue
            if state.phase != "ready":
                self.states[identity] = PinchState()
                continue
            if (now - state.last_open > 0.45
                    or np.linalg.norm(np.asarray(anchor) - state.anchor) > 0.45
                    or np.linalg.norm(np.asarray(pointer) - state.point) > 0.9):
                self.states[identity] = PinchState()
                continue
            if ratio <= 0.30:
                if state.closing_since is None:
                    state.closing_since = now
                elif now - state.closing_since >= 0.04:
                    clicks.append(AirClick(state.key, hand.handedness, state.point, now, state.confidence))
                    state.phase = "held"
                    state.release_since = None
            else:
                state.closing_since = None
        if len(clicks) > 1:
            self.status = "Both hands clicked together. Try one at a time"
            return []
        if clicks:
            self.status = "Clicked " + clicks[0].key.label
        return clicks

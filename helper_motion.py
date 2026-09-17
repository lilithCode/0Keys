from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np


TIPS = (4, 8, 12, 16, 20)
NO_KEY = "no_key"


@dataclass
class Movement:
    hand: str
    times: list
    points: list

    def features(self):
        points = np.asarray(self.points, dtype=float)
        times = np.asarray(self.times, dtype=float)
        if (points.shape != (len(times), 21, 2) or len(times) < 4
                or not np.isfinite(points).all() or not np.isfinite(times).all()
                or np.any(np.diff(times) <= 0) or times[-1] - times[0] < 0.2
                or np.max(np.diff(times)) > 0.35 or self.hand not in ("Left", "Right")):
            raise ValueError("Incomplete movement. Keep the hand visible and try again")
        scale = float(np.linalg.norm(points[0, 5] - points[0, 17]))
        if scale < 0.3:
            raise ValueError("Hand is too small in the keyboard view")
        grid = np.linspace(times[0], times[-1], 16)
        sampled = np.stack([np.interp(grid, times, column) for column in points.reshape(len(times), -1).T], axis=1)
        sampled = sampled.reshape(16, 21, 2)
        motion = (sampled - points[0]) / scale
        return motion, points[0], scale


@dataclass
class MotionState:
    frames: list = field(default_factory=list)
    anchor: np.ndarray | None = None
    active: bool = False
    returned: float | None = None


class MovementRecorder:
    def __init__(self, calibration):
        self.calibration = calibration
        self.states = {}
        self.last_time = None
        self.status = "Hold naturally, then press and release one key"

    def reset(self):
        self.states.clear()
        self.last_time = None

    def hand_points(self, hand):
        if len(hand.landmarks) != 21:
            return None
        points = np.asarray([self.calibration.map_to_keyboard(p.x, p.y) for p in hand.landmarks])
        if not np.isfinite(points).all() or np.linalg.norm(points[5] - points[17]) < 0.3:
            return None
        return points

    def update(self, snapshot):
        now = snapshot.timestamp
        if self.last_time is not None and (now <= self.last_time or now - self.last_time > 0.35):
            self.reset()
        self.last_time = now
        identities = {(h.hand_id, h.handedness) for h in snapshot.hands}
        self.states = {k: v for k, v in self.states.items() if k in identities}
        completed = []
        self.status = "Hold naturally, then press and release one key"
        for hand in snapshot.hands:
            identity = (hand.hand_id, hand.handedness)
            if hand.handedness not in ("Left", "Right") or sum(h.handedness == hand.handedness for h in snapshot.hands) != 1:
                self.states.pop(identity, None)
                continue
            points = self.hand_points(hand)
            if points is None:
                self.states.pop(identity, None)
                continue
            state = self.states.setdefault(identity, MotionState())
            state.frames.append((now, points))
            if state.anchor is None:
                state.frames = [f for f in state.frames if now - f[0] <= 0.5]
                scale = np.linalg.norm(points[5] - points[17])
                recent = np.asarray([f[1] for f in state.frames])
                jitter = np.max(np.linalg.norm(recent - recent[0], axis=2)) / scale
                if jitter > 0.06:
                    state.frames = [(now, points)]
                elif len(recent) >= 3 and now - state.frames[0][0] >= 0.24:
                    state.anchor = np.median(recent, axis=0)
                continue
            scale = np.linalg.norm(state.anchor[5] - state.anchor[17])
            distance = np.max(np.linalg.norm(points - state.anchor, axis=1)) / scale
            if not state.active:
                if distance > 0.09:
                    state.active = True
                    previous_time = state.frames[-2][0]
                    state.frames = [(previous_time, state.anchor.copy()), (now, points)]
                else:
                    state.frames = state.frames[-2:]
                continue
            self.status = "Movement seen. Release back to your starting pose"
            if now - state.frames[0][0] > 2.5:
                self.states[identity] = MotionState()
                self.status = "Movement too long. Return to a relaxed pose and try again"
                continue
            if distance <= 0.07:
                if state.returned is None:
                    state.returned = now
                if now - state.returned >= 0.22 and len(state.frames) >= 4:
                    completed.append(Movement(hand.handedness, [f[0] for f in state.frames], [f[1] for f in state.frames]))
                    self.states[identity] = MotionState()
            else:
                state.returned = None
        return completed


class MovementModel:
    def __init__(self, layout, context):
        self.layout = layout
        self.context = context
        self.samples = []

    def count(self, label, hand=None):
        return sum(s["label"] == label and (hand is None or s["hand"] == hand) for s in self.samples)

    def matches_no_key(self, movement):
        motion, start, scale = movement.features()
        distances = sorted(self.distance(motion, start, scale, sample) for sample in self.samples
                           if sample["label"] == NO_KEY and sample["hand"] == movement.hand)
        return len(distances) >= 3 and float(np.mean(distances[:3])) < 0.055

    def add(self, label, movement):
        motion, start, scale = movement.features()
        if label != NO_KEY:
            key = next((k for k in self.layout.keys if k.name == label and k.value not in ("CONTROL", "ALT")), None)
            if key is None:
                raise ValueError("Select an enabled key")
            raw = np.asarray(movement.points)
            travel = np.max(np.linalg.norm(raw - raw[0], axis=2), axis=0) / scale
            if np.max(travel) < 0.12:
                raise ValueError("Movement too small to learn. Make one visible press and release")
            excursion = np.max(np.linalg.norm(raw - raw[0], axis=2), axis=1) / scale
            if np.count_nonzero(excursion > 0.09) < 2:
                raise ValueError("Only one moving frame seen. Press slower so the camera can follow")
            if np.max(np.linalg.norm(raw[-1] - raw[0], axis=1)) / scale > 0.09:
                raise ValueError("Finish by returning to your starting pose")
            # Other fingers can join in, but a moving fingertip must reach this key.
            near = any(travel[tip] >= 0.09 and np.any(
                (raw[:, tip, 0] >= key.x - 0.2) & (raw[:, tip, 0] <= key.x + key.width + 0.2)
                & (raw[:, tip, 1] >= key.y - 0.2) & (raw[:, tip, 1] <= key.y + key.height + 0.2)
            ) for tip in TIPS)
            if not near:
                raise ValueError("Move the finger you use for this key near its highlighted cap")
        self.samples.append({"label": label, "hand": movement.hand, "motion": motion.tolist(),
                             "start": start.tolist(), "scale": scale})

    @staticmethod
    def distance(motion, start, scale, sample):
        # Match the whole hand as one pattern, not five separate clicks.
        shape = np.asarray(sample["motion"])
        delta = np.linalg.norm(motion - shape, axis=2)
        movement_error = float(np.mean(np.max(delta, axis=1)))
        placement = float(np.mean(np.linalg.norm(start - np.asarray(sample["start"]), axis=1)))
        return movement_error + 0.25 * placement / max(scale, sample["scale"])

    def predict(self, movement):
        motion, start, scale = movement.features()
        if self.count(NO_KEY, movement.hand) < 3:
            return None, "Record no-key examples for this hand with N first"
        groups = {}
        for sample in self.samples:
            if sample["hand"] == movement.hand:
                groups.setdefault(sample["label"], []).append(self.distance(motion, start, scale, sample))
        scores = sorted((float(np.mean(sorted(values)[:3])), label)
                        for label, values in groups.items() if len(values) >= 3)
        if not scores or scores[0][1] == NO_KEY:
            return None, "Ignored: matches no-key movement"
        best, label = scores[0]
        runner = scores[1][0] if len(scores) > 1 else 0.0
        if best > 0.12 or runner - best < 0.035:
            return None, "Uncertain movement ignored. Add varied examples for nearby keys"
        try:
            validation = MovementModel(self.layout, self.context)
            validation.add(label, movement)
        except ValueError as exc:
            return None, f"Ignored: {exc}"
        return next(k for k in self.layout.keys if k.name == label), "Matched your recorded hand movement"

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"version": 1, "context": self.context, "samples": self.samples}) + "\n")
        temporary.replace(path)

    @classmethod
    def load(cls, path, layout, context):
        data = json.loads(Path(path).read_text())
        if not isinstance(data, dict) or data.get("version") != 1 or data.get("context") != context:
            raise ValueError("Movement profile uses another layout or camera view")
        model = cls(layout, context)
        labels = {k.name for k in layout.keys if k.value not in ("CONTROL", "ALT")} | {NO_KEY}
        for item in data["samples"]:
            if not isinstance(item, dict) or item.get("label") not in labels or item.get("hand") not in ("Left", "Right"):
                raise ValueError("Invalid movement sample")
            motion, start = np.asarray(item["motion"], dtype=float), np.asarray(item["start"], dtype=float)
            scale = float(item["scale"])
            if (motion.shape != (16, 21, 2) or start.shape != (21, 2)
                    or not np.isfinite(motion).all() or not np.isfinite(start).all()
                    or not np.isfinite(scale) or scale < 0.3):
                raise ValueError("Invalid movement sample")
            model.samples.append(item)
        return model

    @classmethod
    def load_for_view(cls, path, layout, context):
        from helper_keyboard import KeyboardCalibration

        data = json.loads(Path(path).read_text())
        old_context = data.get("context", {}) if isinstance(data, dict) else {}
        if old_context == context:
            return cls.load(path, layout, context), "Saved movement examples loaded"
        view_fields = ("layout", "camera", "rotation", "mirrored", "gesture")
        if any(old_context.get(k) != context.get(k) for k in view_fields):
            raise ValueError("Saved movements use another camera view. Original file kept")
        old = cls.load(path, layout, old_context)
        source = KeyboardCalibration(old_context["points"], old_context["mirrored"], old_context["flip_rows"])
        target = KeyboardCalibration(context["points"], context["mirrored"], context["flip_rows"])
        model = cls(layout, context)
        for sample in old.samples:
            # A moved keycap invalidates key labels, but not no-key recordings.
            if sample["label"] != NO_KEY:
                continue
            raw = np.asarray(sample["start"]) + np.asarray(sample["motion"]) * sample["scale"]
            points = np.asarray([[target.map_to_keyboard(*source.map_to_image(*p)) for p in frame] for frame in raw])
            model.add(NO_KEY, Movement(sample["hand"], list(np.linspace(0, 1.5, 16)), points))
        return model, f"Layout changed: reused {len(model.samples)} no-key examples only. Original file kept"


class NaturalPressModel:
    def __init__(self, layout, personal):
        self.layout = layout
        self.personal = personal

    def predict(self, movement):
        movement.features()
        raw = np.asarray(movement.points)
        scale = float(np.linalg.norm(raw[0, 5] - raw[0, 17]))
        if self.personal.matches_no_key(movement):
            return None, "Ignored: resembles your saved no-key movements"
        personal_reason = "No trained key matched"
        if any(self.personal.count(k.name, movement.hand) >= 3 for k in self.layout.keys):
            key, reason = self.personal.predict(movement)
            if key is not None:
                return key, "Personal movement match"
            personal_reason = reason
        bases = (2, 5, 9, 13, 17)
        relative = raw[:, TIPS, :] - raw[:, bases, :]
        displacements = np.linalg.norm(relative - relative[0], axis=2) / scale
        amplitudes = np.max(displacements, axis=0)
        order = np.argsort(amplitudes)[::-1]
        best, second = int(order[0]), int(order[1])
        if amplitudes[best] < 0.12:
            return None, "Ignored: no clear finger press"
        if amplitudes[second] > amplitudes[best] * 0.70 or amplitudes[best] - amplitudes[second] < 0.045:
            return None, "Ignored: more than one finger moved strongly"
        palm_motion = np.max(np.linalg.norm(raw[:, [0, 5, 9, 13, 17]] - raw[0, [0, 5, 9, 13, 17]], axis=2)) / scale
        if palm_motion > 0.16:
            return None, "Ignored: hand repositioning rather than a finger press"
        active = displacements[:, best] > 0.09
        if np.count_nonzero(active) < 2:
            return None, "Press slower: only one moving camera frame"
        if np.count_nonzero(np.diff(np.r_[False, active, False].astype(int)) == 1) != 1:
            return None, "Ignored: repeated or irregular finger movement"
        if np.max(np.linalg.norm(raw[-1] - raw[0], axis=1)) / scale > 0.09:
            return None, "Release the press before typing the next key"
        tip = TIPS[best]
        key = self.layout.key_at(*raw[0, tip])
        if key is None or key.value in ("CONTROL", "ALT"):
            return None, "Start with your fingertip inside a key, not a gap"
        if self.personal.count(key.name, movement.hand) >= 3:
            return None, personal_reason
        x, y = raw[0, tip]
        margin = 0.07
        if not (key.x + margin <= x <= key.x + key.width - margin
                and key.y + margin <= y <= key.y + key.height - margin):
            return None, "Aim nearer the key center before pressing"
        return key, "Basic press detection, not a trained accuracy estimate"


class GuidedTraining:
    def __init__(self, label, now, repetitions=5):
        self.label = label
        self.repetitions = repetitions
        self.saved = 0
        self.hand = None
        self.ready_at = now + 3
        self.negative_frames = {}
        self.message = ""

    @property
    def done(self):
        return self.saved >= self.repetitions

    def prompt(self, now):
        if self.done:
            return "Training saved. Select another key or press T to type"
        if now < self.ready_at:
            return f"Get ready in {max(1, int(np.ceil(self.ready_at - now)))}. {self.message}"
        action = "Move naturally without pressing a key" if self.label == NO_KEY else "Press the highlighted key ONCE, then return to your starting pose"
        return f"Example {self.saved + 1} of {self.repetitions}: {action}"

    def update(self, snapshot, recorder, model):
        now = snapshot.timestamp
        if self.done:
            recorder.reset()
            return False
        if now < self.ready_at:
            # Find the resting pose before the prompt, without saving any movement.
            recorder.update(snapshot)
            if any(state.active for state in recorder.states.values()):
                recorder.reset()
            return False
        examples = recorder.update(snapshot)
        if self.label == NO_KEY:
            identities = {(h.hand_id, h.handedness) for h in snapshot.hands}
            self.negative_frames = {k: v for k, v in self.negative_frames.items() if k in identities}
            for hand in snapshot.hands:
                points = recorder.hand_points(hand)
                if points is None or sum(h.handedness == hand.handedness for h in snapshot.hands) != 1:
                    self.negative_frames.pop((hand.hand_id, hand.handedness), None)
                    continue
                frames = self.negative_frames.setdefault((hand.hand_id, hand.handedness), [])
                if frames and now - frames[-1][0] > 0.35:
                    frames.clear()
                frames.append((now, points))
            examples = [Movement(side, [f[0] for f in frames], [f[1] for f in frames])
                        for (_, side), frames in self.negative_frames.items()
                        if len(frames) >= 4 and now - frames[0][0] >= 2]
        if len(examples) > 1 and self.label != NO_KEY:
            self.message = "Two hands pressed together. Repeat one key only"
            examples = []
        accepted = False
        for example in examples:
            if self.hand is not None and self.label != NO_KEY and example.hand != self.hand:
                self.message = "Use the same hand for this batch"
                continue
            try:
                model.add(self.label, example)
            except ValueError as exc:
                self.message = str(exc)
            else:
                self.hand = example.hand
                accepted = True
        if accepted:
            self.saved += 1
            self.ready_at = now + 3
            self.message = "Saved. Relax before the next example"
            self.negative_frames.clear()
            recorder.reset()
        elif now > self.ready_at + 6:
            self.ready_at = now + 3
            self.message = "No usable example yet. Keep the whole hand visible and try slower"
            self.negative_frames.clear()
            recorder.reset()
        return accepted

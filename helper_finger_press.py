from dataclasses import dataclass, field
import math

import numpy as np

from helper_motion import Movement, TIPS

BASES = (2, 5, 9, 13, 17)
START = 0.07
PEAK = 0.10
RELEASE = 0.05
REST_SECONDS = 0.5
MIN_PALM = 0.05
COUPLED_SECONDS = 0.14
DEPTH_WEIGHT = 0.6
NOISE_SECONDS = 1.0
NOISE_START = 1.8
NOISE_PEAK = 3.0
NOISE_RELEASE = 2.0
NOISE_STILL = 3.0
MAX_TAP_SECONDS = 0.9
REBOUND = 0.45
REBOUND_FRAMES = 2


def typing_hands(hands, calibration, aspect=16 / 9):
    def palm(hand):
        a, b = hand.landmarks[5], hand.landmarks[17]
        return math.hypot((a.x - b.x) * aspect, a.y - b.y)

    largest = max((palm(hand) for hand in hands), default=0.0)
    kept, ignored = [], []
    for hand in hands:
        tips = [calibration.map_to_keyboard(hand.landmarks[tip].x, hand.landmarks[tip].y) for tip in TIPS[1:]]
        beyond = all(np.isfinite(y) and y < -0.5 for _, y in tips)
        (ignored if beyond or palm(hand) < 0.5 * largest else kept).append(hand)
    return kept, ignored


@dataclass
class FingerState:
    baseline: np.ndarray | None = None
    warmup: list = field(default_factory=list)
    start: float | None = None
    peak: float = 0.0
    moving_frames: int = 0
    returned: int = 0
    key: object = None
    blocked: bool = False
    last_click: float = -10.0
    frames: list = field(default_factory=list)
    previous: tuple | None = None
    aim: list = field(default_factory=list)
    contact: list = field(default_factory=list)
    releasing: bool = False
    reference: np.ndarray | None = None


@dataclass(frozen=True)
class FingerClick:
    key: object
    handedness: str
    tip: int
    timestamp: float


class FingerPressDetector:
    def __init__(self, layout, calibration, personal, aspect=16 / 9, sensitivity=1.0):
        self.layout = layout
        self.calibration = calibration
        self.personal = personal
        self.aspect = aspect
        self.sensitivity = sensitivity
        self.states = {}
        self.sides = {}
        self.last_time = None
        self.hover = {}
        self.levels = {}
        self.phases = {}
        self.noise = {}
        self.status = "Place fingers over the keys, then make a press and release"
        self.events = 0
        self.last_status = ""
        self.status_until = 0.0

    def reset(self):
        self.states.clear()
        self.hover.clear()
        self.levels.clear()
        self.phases.clear()
        self.last_time = None
        self.status_until = 0.0

    def thresholds(self, noise=0.0):
        scale = 1.0 / max(self.sensitivity, 1e-3)
        return (max(START, NOISE_START * noise) * scale, max(PEAK, NOISE_PEAK * noise) * scale,
                max(RELEASE, NOISE_RELEASE * noise) * scale)

    def track_noise(self, name, position, tip, elapsed, resting):
        last, level = self.noise.get(name, (None, 0.0))
        if last is not None and resting and elapsed > 0:
            step = min(self.motion(position - last, tip), START)
            level += (step - level) * (1.0 - math.exp(-elapsed / NOISE_SECONDS))
        self.noise[name] = (position.copy(), level)
        return level

    @staticmethod
    def motion(delta, tip):
        weights = np.asarray([1.0 if tip == 4 else 0.0, 1.0, DEPTH_WEIGHT])
        return float(np.linalg.norm(delta * weights))

    def contact_key(self, samples, tip):
        extension, points = max(samples, key=lambda item: item[0])
        candidate = self.layout.key_at(*points[tip])
        near = [points[tip] for length, points in samples if extension - length <= 0.06]
        keys = [self.layout.key_at(*point) for point in near]
        if candidate is None or keys.count(candidate) < 2:
            return None
        if any(self.layout.key_at(*points[tip]) != candidate for length, points in samples
               if extension - length <= 0.015):
            return None
        return candidate

    @staticmethod
    def finger_positions(hand, aspect):
        image = np.asarray([(p.x * aspect, p.y, p.z * aspect) for p in hand.landmarks])
        if not np.isfinite(image).all():
            return None
        palm = image[5] - image[17]
        scale = float(np.linalg.norm(palm))
        if scale < MIN_PALM:
            return None
        across = palm / scale
        forward = np.mean(image[[5, 9, 13, 17]], axis=0) - image[0]
        forward -= np.dot(forward, across) * across
        length = np.linalg.norm(forward)
        if length < scale * 0.05:
            return None
        forward /= length
        normal = np.cross(across, forward)
        basis = np.asarray([across, forward, normal])
        return (image[list(TIPS)] - image[list(BASES)]) @ basis.T / scale

    def assign_sides(self, hands):
        if len(hands) == 2:
            ordered = sorted(hands, key=lambda h: np.mean([h.landmarks[i].x for i in (0, 5, 9, 13, 17)]))
            names = ("Left", "Right") if self.calibration.mirrored else ("Right", "Left")
            for hand, name in zip(ordered, names):
                self.sides[hand.hand_id] = name
        for hand in hands:
            if hand.hand_id not in self.sides:
                if hand.handedness in ("Left", "Right"):
                    self.sides[hand.hand_id] = hand.handedness
                else:
                    left_half = np.mean([p.x for p in hand.landmarks]) < 0.5
                    self.sides[hand.hand_id] = "Left" if left_half == self.calibration.mirrored else "Right"
        if len(self.sides) > 16:
            present = {h.hand_id for h in hands}
            self.sides = {k: v for k, v in self.sides.items() if k in present}

    def update(self, snapshot):
        now = snapshot.timestamp
        if self.last_time is not None and (now <= self.last_time or now - self.last_time > 0.4):
            self.reset()
        elapsed = 0.0 if self.last_time is None else now - self.last_time
        self.last_time = now
        follow = 1.0 - math.exp(-elapsed / REST_SECONDS)
        hands = [h for h in snapshot.hands if len(h.landmarks) == 21]
        self.assign_sides(hands)
        present = {h.hand_id for h in hands}
        self.states = {hand_id: states for hand_id, states in self.states.items() if hand_id in present}
        self.noise = {name: value for name, value in self.noise.items() if name[0] in present}
        self.hover, self.levels, self.phases = {}, {}, {}
        ready = "Ready. Press with a finger and release that finger"
        self.status = ready
        clicks = []
        for hand in hands:
            side = self.sides[hand.hand_id]
            local = self.finger_positions(hand, self.aspect)
            raw = np.asarray([self.calibration.map_to_keyboard(p.x, p.y) for p in hand.landmarks])
            if local is None or not np.isfinite(raw).all():
                self.states.pop(hand.hand_id, None)
                continue
            states = self.states.setdefault(hand.hand_id, [FingerState() for _ in TIPS])
            completed = []
            for index, (tip, position, state) in enumerate(zip(TIPS, local, states)):
                name = (hand.hand_id, tip)
                resting = state.start is None and not state.releasing and not state.blocked
                noise = self.track_noise(name, position, tip, elapsed, resting)
                start_level, peak_level, release_level = self.thresholds(noise)
                still = max(0.04, NOISE_STILL * noise)
                steady = max(0.025, NOISE_STILL * noise)
                self.hover[name] = self.layout.key_at(*raw[tip])
                if state.reference is None:
                    state.reference = raw.copy()
                if (state.baseline is not None and state.start is None and state.previous is not None
                        and not state.releasing and not state.blocked):
                    palm_indices = [0, 5, 9, 13, 17]
                    shift = np.median(raw[palm_indices] - state.previous[1][palm_indices], axis=0)
                    state.reference += shift
                    state.aim = [(length, points + shift) for length, points in state.aim]
                state.aim.append((float(position[1]), raw.copy()))
                state.aim = state.aim[-3:]
                if state.baseline is None:
                    state.warmup.append((now, position.copy()))
                    if np.max(np.linalg.norm(np.asarray([p for _, p in state.warmup]) - position, axis=1)) > still:
                        state.warmup = [(now, position.copy())]
                    if len(state.warmup) >= 3:
                        state.baseline = np.median([p for _, p in state.warmup[-3:]], axis=0)
                        state.reference = raw.copy()
                    state.previous = (now, raw.copy())
                    self.levels[name], self.phases[name] = 0.0, "warmup"
                    continue
                delta = position - state.baseline
                distance = self.motion(delta, tip)
                self.levels[name] = distance / peak_level
                if state.releasing:
                    self.phases[name] = "release"
                    state.warmup.append((now, position.copy()))
                    state.warmup = state.warmup[-3:]
                    settled = (len(state.warmup) >= 3 and now - state.warmup[0][0] >= 0.06
                               and np.max(np.linalg.norm(np.asarray([p for _, p in state.warmup]) - position, axis=1)) < steady)
                    if distance < release_level or settled:
                        state.releasing = False
                        state.baseline = position.copy()
                        state.previous = (now, raw.copy())
                        state.reference = raw.copy()
                        state.warmup = []
                    continue
                if state.blocked:
                    self.phases[name] = "blocked"
                    state.warmup.append((now, position.copy()))
                    state.warmup = [f for f in state.warmup if now - f[0] <= 0.35]
                    settled = (len(state.warmup) >= 3 and now - state.warmup[0][0] >= 0.24
                               and np.max(np.linalg.norm(np.asarray([p for _, p in state.warmup]) - position, axis=1)) < steady)
                    if distance < release_level:
                        states[index] = FingerState(last_click=state.last_click)
                    elif settled:
                        states[index] = FingerState(baseline=position.copy(), previous=(now, raw.copy()),
                                                    last_click=state.last_click)
                    continue
                if state.start is None:
                    self.phases[name] = "rest"
                    if tip != 4 and abs(delta[0]) > max(start_level, distance * 1.5):
                        state.baseline = position.copy()
                        state.previous = (now, raw.copy())
                        state.reference = raw.copy()
                        state.aim = state.aim[-1:]
                        self.phases[name] = "aiming"
                        continue
                    if distance >= start_level and now - state.last_click >= 0.09:
                        self.phases[name] = "moving"
                        state.start = now
                        state.peak = distance
                        state.moving_frames = 1
                        state.returned = 0
                        state.frames = [state.previous, (now, raw.copy())]
                        state.contact = [(float(state.baseline[1]), state.reference.copy())] + list(state.aim)
                    else:
                        state.baseline = state.baseline + (position - state.baseline) * follow
                        state.reference += (raw - state.reference) * follow
                        state.previous = (now, raw.copy())
                    continue
                self.phases[name] = "moving"
                state.frames.append((now, raw.copy()))
                state.contact.append((float(position[1]), raw.copy()))
                state.warmup.append((now, position.copy()))
                state.warmup = [f for f in state.warmup if now - f[0] <= 0.35]
                if distance > state.peak:
                    state.peak = distance
                if distance > start_level:
                    state.moving_frames += 1
                released = state.peak - distance >= max(release_level, state.peak * REBOUND)
                state.returned = state.returned + 1 if released else 0
                if (state.returned >= REBOUND_FRAMES and state.peak >= peak_level and state.moving_frames >= 2
                        and now - state.start >= 0.06):
                    state.key = self.contact_key(state.contact, tip)
                    completed.append(index)
                elif distance < release_level and state.moving_frames < 2:
                    states[index] = FingerState(baseline=position.copy(), previous=(now, raw.copy()),
                                                last_click=state.last_click, aim=list(state.aim))
                elif (now - state.start > 0.35 and len(state.warmup) >= 3
                      and now - state.warmup[0][0] >= 0.20
                      and np.max(np.linalg.norm(np.asarray([p for _, p in state.warmup]) - position, axis=1)) < steady
                      and self.layout.key_at(*raw[tip]) != self.layout.key_at(*state.frames[0][1][tip])):
                    states[index] = FingerState(baseline=position.copy(), previous=(now, raw.copy()),
                                                last_click=state.last_click, reference=raw.copy())
                    self.phases[name] = "aiming"
                elif now - state.start > MAX_TAP_SECONDS:
                    states[index] = FingerState(baseline=position.copy(), previous=(now, raw.copy()),
                                                last_click=state.last_click, reference=raw.copy())
                    self.phases[name] = "aiming"
            peers = [(other.start, other.peak, other.blocked) for other in states]
            for index in sorted(completed, key=lambda i: states[i].peak, reverse=True):
                state = states[index]
                if state.blocked or state.start is None:
                    continue
                overlapping = [other for j, other in enumerate(states) if j != index and other.start is not None
                               and not other.blocked and abs(other.start - state.start) <= COUPLED_SECONDS]
                other_peak = max((peak for j, (start, peak, blocked) in enumerate(peers)
                                  if j != index and start is not None and not blocked
                                  and abs(start - state.start) <= COUPLED_SECONDS), default=0)
                if other_peak > state.peak * 0.85:
                    self.status = "Unclear which finger pressed. Try one clear press"
                    state.blocked = True
                    state.warmup = []
                    continue
                key = state.key
                reason = None
                if key is None or key.value in ("CONTROL", "ALT"):
                    reason = "Tap position unclear, in a gap, or inactive. Aim inside one key"
                elif TIPS[index] == 4 and key.value != " ":
                    reason = "Thumb press ignored away from Space"
                elif len(state.frames) >= 4 and self.personal.count(key.name, side) >= 3:
                    movement = Movement(side, [f[0] for f in state.frames], [f[1] for f in state.frames])
                    try:
                        learned, reason = self.personal.predict(movement)
                        if learned is not None and learned.name == key.name:
                            reason = None
                    except ValueError as exc:
                        reason = str(exc)
                if reason is None:
                    clicks.append(FingerClick(key, side, TIPS[index], now))
                    self.status = f"Pressed {key.label} with {side} finger"
                    self.events += 1
                    state.last_click = now
                    for other in overlapping:
                        if other.peak < state.peak * 0.85:
                            other.blocked = True
                            other.warmup = []
                else:
                    self.status = reason
                states[index] = FingerState(baseline=state.baseline.copy(), previous=(now, raw.copy()),
                                            last_click=state.last_click, aim=list(state.aim), releasing=True)
        if self.status != ready:
            self.last_status, self.status_until = self.status, now + 1.0
        elif now < self.status_until:
            self.status = self.last_status
        return clicks

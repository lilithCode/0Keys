export const KEYBOARD_WIDTH = 15;
export const KEYBOARD_HEIGHT = 5;
export const TIPS = [4, 8, 12, 16, 20];
const BASES = [2, 5, 9, 13, 17];
const PALM = [0, 5, 9, 13, 17];

const START = 0.07;
const PEAK = 0.10;
const RELEASE = 0.05;
const REST_SECONDS = 0.5;
const MIN_PALM = 0.05;
const COUPLED_SECONDS = 0.14;
const DEPTH_WEIGHT = 0.6;
const NOISE_SECONDS = 1.0;
const NOISE_START = 1.8;
const NOISE_PEAK = 3.0;
const NOISE_RELEASE = 2.0;
const NOISE_STILL = 3.0;
const MAX_TAP_SECONDS = 0.9;
const REBOUND = 0.45;
const REBOUND_FRAMES = 2;

export const DEFAULT_POINTS = [[0.05, 0.28], [0.95, 0.28], [0.95, 0.72], [0.05, 0.72]];

const SHIFTED = {
  "`": "~", "1": "!", "2": "@", "3": "#", "4": "$", "5": "%", "6": "^", "7": "&", "8": "*",
  "9": "(", "0": ")", "-": "_", "=": "+", "[": "{", "]": "}", "\\": "|", ";": ":", "'": '"',
  ",": "<", ".": ">", "/": "?",
};

function row(y, specs) {
  let x = 0;
  return specs.map(([name, label, value, width]) => {
    const key = { name, label, value, x, y, width, height: 1 };
    x += width;
    return key;
  });
}

function letters(text) {
  return [...text].map((c) => [c.toLowerCase(), c, c.toLowerCase(), 1]);
}

export function buildLayout(gapX = 0.16, gapY = 0.22) {
  const top = row(0, [["grave", "`", "`", 1], ...[..."1234567890"].map((c) => [c, c, c, 1]),
    ["minus", "-", "-", 1], ["equals", "=", "=", 1], ["backspace", "Back", "BACKSPACE", 2]]);
  const middle = [
    ...row(1, [["tab", "Tab", "TAB", 1.5], ...letters("QWERTYUIOP"), ["left_bracket", "[", "[", 1],
      ["right_bracket", "]", "]", 1], ["backslash", "\\", "\\", 1.5]]),
    ...row(2, [["caps_lock", "Caps", "CAPS_LOCK", 1.75], ...letters("ASDFGHJKL"), ["semicolon", ";", ";", 1],
      ["apostrophe", "'", "'", 1], ["enter", "Enter", "ENTER", 2.25]]),
    ...row(3, [["left_shift", "Shift", "SHIFT", 2.25], ...letters("ZXCVBNM"), ["comma", ",", ",", 1],
      ["period", ".", ".", 1], ["question_mark", "/", "/", 1], ["right_shift", "Shift", "SHIFT", 2.75]]),
  ];
  const bottom = row(4, [["left_control", "Ctrl", "CONTROL", 1.25], ["left_meta", "Win", "CONTROL", 1.25],
    ["left_alt", "Alt", "ALT", 1.25], ["space", "Space", " ", 6.25], ["right_alt", "Alt", "ALT", 1.25],
    ["right_meta", "Win", "CONTROL", 1.25], ["menu", "Menu", "CONTROL", 1.25], ["right_control", "Ctrl", "CONTROL", 1.25]]);
  const keys = [...top, ...middle, ...bottom].map((k) => ({
    ...k, x: k.x + gapX / 2, y: k.y + gapY / 2, width: k.width - gapX, height: k.height - gapY,
  }));
  return {
    keys,
    keyAt(x, y) {
      if (!(x >= 0 && x < KEYBOARD_WIDTH && y >= 0 && y < KEYBOARD_HEIGHT)) return null;
      return keys.find((k) => k.x <= x && x < k.x + k.width && k.y <= y && y < k.y + k.height) || null;
    },
  };
}

export function corners(key) {
  return [[key.x, key.y], [key.x + key.width, key.y], [key.x + key.width, key.y + key.height], [key.x, key.y + key.height]];
}

function solve(matrix, vector) {
  const n = vector.length;
  const a = matrix.map((r, i) => [...r, vector[i]]);
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let r = col + 1; r < n; r++) if (Math.abs(a[r][col]) > Math.abs(a[pivot][col])) pivot = r;
    [a[col], a[pivot]] = [a[pivot], a[col]];
    if (Math.abs(a[col][col]) < 1e-12) throw new Error("Calibration points are degenerate");
    for (let r = 0; r < n; r++) {
      if (r === col) continue;
      const f = a[r][col] / a[col][col];
      for (let c = col; c <= n; c++) a[r][c] -= f * a[col][c];
    }
  }
  return a.map((r, i) => r[n] / r[i]);
}

function perspective(src, dst) {
  const m = [];
  const v = [];
  for (let i = 0; i < 4; i++) {
    const [x, y] = src[i];
    const [u, w] = dst[i];
    m.push([x, y, 1, 0, 0, 0, -x * u, -y * u]);
    v.push(u);
    m.push([0, 0, 0, x, y, 1, -x * w, -y * w]);
    v.push(w);
  }
  const h = solve(m, v);
  return [[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1]];
}

function transform(h, x, y) {
  const w = h[2][0] * x + h[2][1] * y + h[2][2];
  if (Math.abs(w) < 1e-9) return [NaN, NaN];
  return [(h[0][0] * x + h[0][1] * y + h[0][2]) / w, (h[1][0] * x + h[1][1] * y + h[1][2]) / w];
}

function cross(o, a, b) {
  return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
}

export function calibrationProblem(points) {
  if (points.length !== 4 || points.flat().some((v) => !Number.isFinite(v) || v < 0 || v > 1)) {
    return "Keep all four corners inside the picture";
  }
  const turns = points.map((p, i) => cross(p, points[(i + 1) % 4], points[(i + 2) % 4]));
  if (!(turns.every((t) => t > 0) || turns.every((t) => t < 0))) return "The corners must form a four-sided shape";
  const area = Math.abs(turns.reduce((s, _, i) => s + (points[i][0] * points[(i + 1) % 4][1] - points[(i + 1) % 4][0] * points[i][1]), 0)) / 2;
  if (area < 0.02) return "The keyboard is too small";
  if ((points[0][1] + points[1][1]) / 2 >= (points[2][1] + points[3][1]) / 2) return "Top corners must stay above the bottom ones";
  if ((points[0][0] + points[3][0]) / 2 >= (points[1][0] + points[2][0]) / 2) return "Left corners must stay on the left";
  return null;
}

export class Calibration {
  constructor(points, mirrored = true, flipRows = false) {
    const problem = calibrationProblem(points);
    if (problem) throw new Error(problem);
    let target = [[0, 0], [KEYBOARD_WIDTH, 0], [KEYBOARD_WIDTH, KEYBOARD_HEIGHT], [0, KEYBOARD_HEIGHT]];
    if (flipRows) target = [target[3], target[2], target[1], target[0]];
    this.points = points.map((p) => [...p]);
    this.mirrored = mirrored;
    this.flipRows = flipRows;
    this.toKeyboard = perspective(points, target);
    this.toImage = perspective(target, points);
  }

  mapToKeyboard(x, y) {
    return transform(this.toKeyboard, x, y);
  }

  mapToImage(x, y) {
    return transform(this.toImage, x, y);
  }
}

export class HandIdentity {
  constructor(maximumDistance = 0.35, timeout = 0.75) {
    this.maximumDistance = maximumDistance;
    this.timeout = timeout;
    this.tracks = new Map();
    this.nextId = 1;
  }

  assign(detections, now) {
    for (const [id, track] of this.tracks) if (now - track.seen > this.timeout) this.tracks.delete(id);
    const pairs = [];
    detections.forEach((d, i) => {
      for (const [id, track] of this.tracks) {
        const distance = Math.hypot(d.landmarks[0].x - track.wrist.x, d.landmarks[0].y - track.wrist.y);
        pairs.push([distance + (d.handedness === track.handedness ? 0 : 0.2), distance, i, id]);
      }
    });
    pairs.sort((a, b) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2] || a[3] - b[3]);
    const assigned = new Map();
    const used = new Set();
    for (const [, distance, i, id] of pairs) {
      if (distance > this.maximumDistance || assigned.has(i) || used.has(id)) continue;
      assigned.set(i, id);
      used.add(id);
    }
    return detections.map((d, i) => {
      const id = assigned.has(i) ? assigned.get(i) : this.nextId++;
      this.tracks.set(id, { handedness: d.handedness, wrist: d.landmarks[0], seen: now });
      return { handId: id, handedness: d.handedness, landmarks: d.landmarks };
    });
  }
}

function palmWidth(hand, aspect) {
  const a = hand.landmarks[5];
  const b = hand.landmarks[17];
  return Math.hypot((a.x - b.x) * aspect, a.y - b.y);
}

export function typingHands(hands, calibration, aspect) {
  const largest = Math.max(0, ...hands.map((h) => palmWidth(h, aspect)));
  const kept = [];
  const ignored = [];
  for (const hand of hands) {
    const tips = TIPS.slice(1).map((t) => calibration.mapToKeyboard(hand.landmarks[t].x, hand.landmarks[t].y));
    const beyond = tips.every(([, y]) => Number.isFinite(y) && y < -0.5);
    (beyond || palmWidth(hand, aspect) < 0.5 * largest ? ignored : kept).push(hand);
  }
  return { kept, ignored };
}

const sub = (a, b) => a.map((v, i) => v - b[i]);
const add = (a, b) => a.map((v, i) => v + b[i]);
const scale = (a, s) => a.map((v) => v * s);
const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
const norm = (a) => Math.sqrt(dot(a, a));
const copyPoints = (points) => points.map((p) => [...p]);

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

function spread(frames, position) {
  return Math.max(...frames.map(([, p]) => norm(sub(p, position))));
}

function motion(delta, tip) {
  return norm([delta[0] * (tip === 4 ? 1 : 0), delta[1], delta[2] * DEPTH_WEIGHT]);
}

function fingerPositions(hand, aspect) {
  const image = hand.landmarks.map((p) => [p.x * aspect, p.y, p.z * aspect]);
  if (image.flat().some((v) => !Number.isFinite(v))) return null;
  const palm = sub(image[5], image[17]);
  const width = norm(palm);
  if (width < MIN_PALM) return null;
  const across = scale(palm, 1 / width);
  let forward = scale([5, 9, 13, 17].reduce((s, i) => add(s, image[i]), [0, 0, 0]), 0.25);
  forward = sub(forward, image[0]);
  forward = sub(forward, scale(across, dot(forward, across)));
  const length = norm(forward);
  if (length < width * 0.05) return null;
  forward = scale(forward, 1 / length);
  const normal = [across[1] * forward[2] - across[2] * forward[1], across[2] * forward[0] - across[0] * forward[2],
    across[0] * forward[1] - across[1] * forward[0]];
  return TIPS.map((tip, i) => {
    const d = sub(image[tip], image[BASES[i]]);
    return [dot(d, across) / width, dot(d, forward) / width, dot(d, normal) / width];
  });
}

function fingerState(values = {}) {
  return {
    baseline: null, warmup: [], start: null, peak: 0, movingFrames: 0, returned: 0, key: null, blocked: false,
    lastClick: -10, frames: [], previous: null, aim: [], contact: [], releasing: false, reference: null, ...values,
  };
}

export class FingerPressDetector {
  constructor(layout, calibration, aspect = 16 / 9, sensitivity = 1) {
    this.layout = layout;
    this.calibration = calibration;
    this.aspect = aspect;
    this.sensitivity = sensitivity;
    this.states = new Map();
    this.sides = new Map();
    this.noise = new Map();
    this.lastTime = null;
    this.hover = new Map();
    this.levels = new Map();
    this.phases = new Map();
    this.status = "Place fingers over the keys, then make a press and release";
    this.events = 0;
    this.lastStatus = "";
    this.statusUntil = 0;
  }

  reset() {
    this.states.clear();
    this.hover.clear();
    this.levels.clear();
    this.phases.clear();
    this.lastTime = null;
    this.statusUntil = 0;
  }

  thresholds(noise) {
    const s = 1 / Math.max(this.sensitivity, 1e-3);
    return [Math.max(START, NOISE_START * noise) * s, Math.max(PEAK, NOISE_PEAK * noise) * s,
      Math.max(RELEASE, NOISE_RELEASE * noise) * s];
  }

  trackNoise(name, position, tip, elapsed, resting) {
    let [last, level] = this.noise.get(name) || [null, 0];
    if (last !== null && resting && elapsed > 0) {
      const step = Math.min(motion(sub(position, last), tip), START);
      level += (step - level) * (1 - Math.exp(-elapsed / NOISE_SECONDS));
    }
    this.noise.set(name, [[...position], level]);
    return level;
  }

  contactKey(samples, tip) {
    const [extension, points] = samples.reduce((best, s) => (s[0] > best[0] ? s : best));
    const candidate = this.layout.keyAt(...points[tip]);
    const near = samples.filter(([length]) => extension - length <= 0.06).map(([, p]) => this.layout.keyAt(...p[tip]));
    if (!candidate || near.filter((k) => k === candidate).length < 2) return null;
    if (samples.some(([length, p]) => extension - length <= 0.015 && this.layout.keyAt(...p[tip]) !== candidate)) return null;
    return candidate;
  }

  assignSides(hands) {
    if (hands.length === 2) {
      const centre = (h) => PALM.reduce((s, i) => s + h.landmarks[i].x, 0) / PALM.length;
      const ordered = [...hands].sort((a, b) => centre(a) - centre(b));
      const names = this.calibration.mirrored ? ["Left", "Right"] : ["Right", "Left"];
      ordered.forEach((h, i) => this.sides.set(h.handId, names[i]));
    }
    for (const hand of hands) {
      if (this.sides.has(hand.handId)) continue;
      if (hand.handedness === "Left" || hand.handedness === "Right") {
        this.sides.set(hand.handId, hand.handedness);
      } else {
        const leftHalf = hand.landmarks.reduce((s, p) => s + p.x, 0) / hand.landmarks.length < 0.5;
        this.sides.set(hand.handId, leftHalf === this.calibration.mirrored ? "Left" : "Right");
      }
    }
    if (this.sides.size > 16) {
      const present = new Set(hands.map((h) => h.handId));
      for (const id of [...this.sides.keys()]) if (!present.has(id)) this.sides.delete(id);
    }
  }

  update(snapshot) {
    const now = snapshot.timestamp;
    if (this.lastTime !== null && (now <= this.lastTime || now - this.lastTime > 0.4)) this.reset();
    const elapsed = this.lastTime === null ? 0 : now - this.lastTime;
    this.lastTime = now;
    const follow = 1 - Math.exp(-elapsed / REST_SECONDS);
    const hands = snapshot.hands.filter((h) => h.landmarks.length === 21);
    this.assignSides(hands);
    const present = new Set(hands.map((h) => h.handId));
    for (const id of [...this.states.keys()]) if (!present.has(id)) this.states.delete(id);
    for (const name of [...this.noise.keys()]) if (!present.has(Number(name.split(":")[0]))) this.noise.delete(name);
    this.hover = new Map();
    this.levels = new Map();
    this.phases = new Map();
    const ready = "Ready. Press with a finger and release that finger";
    this.status = ready;
    const clicks = [];
    for (const hand of hands) {
      const side = this.sides.get(hand.handId);
      const local = fingerPositions(hand, this.aspect);
      const raw = hand.landmarks.map((p) => this.calibration.mapToKeyboard(p.x, p.y));
      if (local === null || raw.flat().some((v) => !Number.isFinite(v))) {
        this.states.delete(hand.handId);
        continue;
      }
      if (!this.states.has(hand.handId)) this.states.set(hand.handId, TIPS.map(() => fingerState()));
      const states = this.states.get(hand.handId);
      const completed = [];
      TIPS.forEach((tip, index) => {
        const position = local[index];
        const state = states[index];
        const name = `${hand.handId}:${tip}`;
        const resting = state.start === null && !state.releasing && !state.blocked;
        const noise = this.trackNoise(name, position, tip, elapsed, resting);
        const [startLevel, peakLevel, releaseLevel] = this.thresholds(noise);
        const still = Math.max(0.04, NOISE_STILL * noise);
        const steady = Math.max(0.025, NOISE_STILL * noise);
        this.hover.set(name, this.layout.keyAt(...raw[tip]));
        if (state.reference === null) state.reference = copyPoints(raw);
        if (state.baseline !== null && state.start === null && state.previous !== null && !state.releasing && !state.blocked) {
          const shift = [0, 1].map((axis) => median(PALM.map((i) => raw[i][axis] - state.previous[1][i][axis])));
          state.reference = state.reference.map((p) => add(p, shift));
          state.aim = state.aim.map(([length, points]) => [length, points.map((p) => add(p, shift))]);
        }
        state.aim.push([position[1], copyPoints(raw)]);
        state.aim = state.aim.slice(-3);
        if (state.baseline === null) {
          state.warmup.push([now, [...position]]);
          if (spread(state.warmup, position) > still) state.warmup = [[now, [...position]]];
          if (state.warmup.length >= 3) {
            const recent = state.warmup.slice(-3).map(([, p]) => p);
            state.baseline = [0, 1, 2].map((axis) => median(recent.map((p) => p[axis])));
            state.reference = copyPoints(raw);
          }
          state.previous = [now, copyPoints(raw)];
          this.levels.set(name, 0);
          this.phases.set(name, "warmup");
          return;
        }
        const delta = sub(position, state.baseline);
        const distance = motion(delta, tip);
        this.levels.set(name, distance / peakLevel);
        if (state.releasing) {
          this.phases.set(name, "release");
          state.warmup.push([now, [...position]]);
          state.warmup = state.warmup.slice(-3);
          const settled = state.warmup.length >= 3 && now - state.warmup[0][0] >= 0.06 && spread(state.warmup, position) < steady;
          if (distance < releaseLevel || settled) {
            state.releasing = false;
            state.baseline = [...position];
            state.previous = [now, copyPoints(raw)];
            state.reference = copyPoints(raw);
            state.warmup = [];
          }
          return;
        }
        if (state.blocked) {
          this.phases.set(name, "blocked");
          state.warmup.push([now, [...position]]);
          state.warmup = state.warmup.filter(([t]) => now - t <= 0.35);
          const settled = state.warmup.length >= 3 && now - state.warmup[0][0] >= 0.24 && spread(state.warmup, position) < steady;
          if (distance < releaseLevel) {
            states[index] = fingerState({ lastClick: state.lastClick });
          } else if (settled) {
            states[index] = fingerState({ baseline: [...position], previous: [now, copyPoints(raw)], lastClick: state.lastClick });
          }
          return;
        }
        if (state.start === null) {
          this.phases.set(name, "rest");
          if (tip !== 4 && Math.abs(delta[0]) > Math.max(startLevel, distance * 1.5)) {
            state.baseline = [...position];
            state.previous = [now, copyPoints(raw)];
            state.reference = copyPoints(raw);
            state.aim = state.aim.slice(-1);
            this.phases.set(name, "aiming");
            return;
          }
          if (distance >= startLevel && now - state.lastClick >= 0.09) {
            this.phases.set(name, "moving");
            state.start = now;
            state.peak = distance;
            state.movingFrames = 1;
            state.returned = 0;
            state.frames = [state.previous, [now, copyPoints(raw)]];
            state.contact = [[state.baseline[1], copyPoints(state.reference)], ...state.aim];
          } else {
            state.baseline = add(state.baseline, scale(sub(position, state.baseline), follow));
            state.reference = state.reference.map((p, i) => add(p, scale(sub(raw[i], p), follow)));
            state.previous = [now, copyPoints(raw)];
          }
          return;
        }
        this.phases.set(name, "moving");
        state.frames.push([now, copyPoints(raw)]);
        state.contact.push([position[1], copyPoints(raw)]);
        state.warmup.push([now, [...position]]);
        state.warmup = state.warmup.filter(([t]) => now - t <= 0.35);
        if (distance > state.peak) state.peak = distance;
        if (distance > startLevel) state.movingFrames += 1;
        const released = state.peak - distance >= Math.max(releaseLevel, state.peak * REBOUND);
        state.returned = released ? state.returned + 1 : 0;
        if (state.returned >= REBOUND_FRAMES && state.peak >= peakLevel && state.movingFrames >= 2 && now - state.start >= 0.06) {
          state.key = this.contactKey(state.contact, tip);
          completed.push(index);
        } else if (distance < releaseLevel && state.movingFrames < 2) {
          states[index] = fingerState({ baseline: [...position], previous: [now, copyPoints(raw)], lastClick: state.lastClick, aim: [...state.aim] });
        } else if (now - state.start > 0.35 && state.warmup.length >= 3 && now - state.warmup[0][0] >= 0.20
          && spread(state.warmup, position) < steady
          && this.layout.keyAt(...raw[tip]) !== this.layout.keyAt(...state.frames[0][1][tip])) {
          states[index] = fingerState({ baseline: [...position], previous: [now, copyPoints(raw)], lastClick: state.lastClick, reference: copyPoints(raw) });
          this.phases.set(name, "aiming");
        } else if (now - state.start > MAX_TAP_SECONDS) {
          states[index] = fingerState({ baseline: [...position], previous: [now, copyPoints(raw)], lastClick: state.lastClick, reference: copyPoints(raw) });
          this.phases.set(name, "aiming");
        }
      });
      const peers = states.map((s) => [s.start, s.peak, s.blocked]);
      for (const index of [...completed].sort((a, b) => states[b].peak - states[a].peak)) {
        const state = states[index];
        if (state.blocked || state.start === null) continue;
        const overlapping = states.filter((o, j) => j !== index && o.start !== null && !o.blocked && Math.abs(o.start - state.start) <= COUPLED_SECONDS);
        const otherPeak = Math.max(0, ...peers.filter(([start, , blocked], j) => j !== index && start !== null && !blocked
          && Math.abs(start - state.start) <= COUPLED_SECONDS).map(([, peak]) => peak));
        if (otherPeak > state.peak * 0.85) {
          this.status = "Unclear which finger pressed. Try one clear press";
          state.blocked = true;
          state.warmup = [];
          continue;
        }
        const key = state.key;
        let reason = null;
        if (key === null || key.value === "CONTROL" || key.value === "ALT") {
          reason = "Tap position unclear, in a gap, or inactive. Aim inside one key";
        } else if (TIPS[index] === 4 && key.value !== " ") {
          reason = "Thumb press ignored away from Space";
        }
        if (reason === null) {
          clicks.push({ key, handedness: side, tip: TIPS[index], timestamp: now });
          this.status = `Pressed ${key.label} with ${side} finger`;
          this.events += 1;
          state.lastClick = now;
          for (const other of overlapping) {
            if (other.peak < state.peak * 0.85) {
              other.blocked = true;
              other.warmup = [];
            }
          }
        } else {
          this.status = reason;
        }
        states[index] = fingerState({ baseline: [...state.baseline], previous: [now, copyPoints(raw)], lastClick: state.lastClick, aim: [...state.aim], releasing: true });
      }
    }
    if (this.status !== ready) {
      this.lastStatus = this.status;
      this.statusUntil = now + 1;
    } else if (now < this.statusUntil) {
      this.status = this.lastStatus;
    }
    return clicks;
  }
}

export class KeyTranslator {
  constructor() {
    this.capsLock = false;
    this.shift = false;
  }

  apply(text, value) {
    if (value === "BACKSPACE") return text.slice(0, -1);
    if (value === "ENTER") return text + "\n";
    if (value === "TAB") return text + "    ";
    if (value === "CAPS_LOCK") {
      this.capsLock = !this.capsLock;
      return text;
    }
    if (value === "SHIFT") {
      this.shift = !this.shift;
      return text;
    }
    if (value === "CONTROL" || value === "ALT") return text;
    let character = value;
    if (/^[a-z]$/.test(character)) {
      if (this.capsLock !== this.shift) character = character.toUpperCase();
    } else if (this.shift) {
      character = SHIFTED[character] || character;
    }
    this.shift = false;
    return text + character;
  }
}

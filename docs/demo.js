import {
  Calibration, DEFAULT_POINTS, FingerPressDetector, HandIdentity, KeyTranslator, TIPS,
  buildLayout, calibrationProblem, corners, typingHands,
} from "./engine.js";

const VISION = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1";
const MODEL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const ARM_SECONDS = 1.5;
const MASK_SECONDS = 1.5;
const STORE = "0keys-demo";
const CONNECTIONS = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11],
  [11, 12], [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [0, 17], [17, 18], [18, 19], [19, 20]];
const PHASES = { rest: "#e9e4d8", warmup: "#9aa3c0", moving: "#ffb547", blocked: "#ff8a80", aiming: "#8fb4ff", release: "#7fe0c2" };

const $ = (id) => document.getElementById(id);
const canvas = $("view");
const context = canvas.getContext("2d");
const video = document.createElement("video");
video.muted = true;
video.playsInline = true;
const scratch = document.createElement("canvas");
const scratchContext = scratch.getContext("2d", { willReadFrequently: false });

const layout = buildLayout();
const saved = load();
let points = saved.points && !calibrationProblem(saved.points) ? saved.points : DEFAULT_POINTS.map((p) => [...p]);
let flipRows = Boolean(saved.flipRows);
let sensitivity = Number.isFinite(saved.sensitivity) ? saved.sensitivity : 1;
let calibration = new Calibration(points, true, flipRows);
let detector = new FingerPressDetector(layout, calibration, 16 / 9, sensitivity);
const identity = new HandIdentity();
const translator = new KeyTranslator();

let landmarker = null;
let stream = null;
let mode = "off";
let armedAt = 0;
let text = "";
let hands = [];
let masks = [];
let lastVideoTime = -1;
let dragging = null;
let flash = null;
let handsSeen = 0;
let noticeUntil = 0;
let notice = "";
const frameTimes = [];

function load() {
  try {
    return JSON.parse(localStorage.getItem(STORE)) || {};
  } catch {
    return {};
  }
}

function save() {
  try {
    localStorage.setItem(STORE, JSON.stringify({ points, flipRows, sensitivity }));
  } catch {
  }
}

function rebuild() {
  calibration = new Calibration(points, true, flipRows);
  detector = new FingerPressDetector(layout, calibration, detector.aspect, sensitivity);
  save();
}

function setState(state, label, detail) {
  $("light").dataset.state = state;
  $("state-label").textContent = label;
  if (detail !== undefined) $("detail").textContent = detail;
}

function renderText() {
  $("typed").textContent = text;
  $("tape").scrollTop = $("tape").scrollHeight;
  const on = [translator.capsLock && "CAPS LOCK ON", translator.shift && "SHIFT ON"].filter(Boolean);
  $("modifiers").textContent = on.join("  ·  ");
}

function setMode(next) {
  mode = next;
  const button = $("toggle");
  button.disabled = next === "off";
  button.textContent = next === "ready" || next === "typing" ? "Stop typing" : "Start typing";
  button.classList.toggle("stop", next === "ready" || next === "typing");
  if (next === "ready") {
    armedAt = performance.now() / 1000 + ARM_SECONDS;
    detector.reset();
    setState("ready", "Get ready", "Put your hands on the table and hold them still.");
  } else if (next === "idle") {
    detector.reset();
    setState("idle", "Not typing", "Drag the round corners so the keys sit under your fingertips, then start typing.");
  }
}

async function loadTracker() {
  const { FilesetResolver, HandLandmarker } = await import(`${VISION}/vision_bundle.mjs`);
  const files = await FilesetResolver.forVisionTasks(`${VISION}/wasm`);
  const options = (delegate) => ({
    baseOptions: { modelAssetPath: MODEL, delegate },
    runningMode: "VIDEO",
    numHands: 2,
    minHandDetectionConfidence: 0.55,
    minHandPresenceConfidence: 0.55,
    minTrackingConfidence: 0.55,
  });
  try {
    return await HandLandmarker.createFromOptions(files, options("GPU"));
  } catch {
    return HandLandmarker.createFromOptions(files, options("CPU"));
  }
}

async function openCamera(deviceId) {
  if (stream) stream.getTracks().forEach((track) => track.stop());
  stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: deviceId ? { deviceId: { exact: deviceId }, width: { ideal: 1280 }, height: { ideal: 720 } }
      : { width: { ideal: 1280 }, height: { ideal: 720 } },
  });
  video.srcObject = stream;
  await video.play();
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  scratch.width = video.videoWidth;
  scratch.height = video.videoHeight;
  canvas.parentElement.style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`;
  detector.aspect = video.videoWidth / video.videoHeight;
  lastVideoTime = -1;
}

async function listCameras() {
  const select = $("camera");
  const current = stream?.getVideoTracks()[0]?.getSettings().deviceId;
  const cameras = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput");
  select.replaceChildren(...cameras.map((camera, index) => {
    const option = document.createElement("option");
    option.value = camera.deviceId;
    option.textContent = camera.label || `Camera ${index + 1}`;
    option.selected = camera.deviceId === current;
    return option;
  }));
}

async function start() {
  const note = $("load-note");
  const button = $("start");
  button.disabled = true;
  note.classList.remove("error");
  if (!navigator.mediaDevices?.getUserMedia) {
    note.textContent = "This browser cannot use a camera here. Open the page over https in Chrome, Edge, Firefox or Safari.";
    note.classList.add("error");
    button.disabled = false;
    return;
  }
  try {
    note.textContent = "Asking for the camera…";
    await openCamera();
    note.textContent = "Loading the hand tracker (about 8 MB, once)…";
    landmarker = landmarker || await loadTracker();
  } catch (error) {
    note.classList.add("error");
    note.textContent = describe(error);
    button.disabled = false;
    return;
  }
  $("cover").hidden = true;
  setMode("idle");
  listCameras().catch(() => {});
  requestAnimationFrame(loop);
}

function describe(error) {
  if (error?.name === "NotAllowedError") return "The camera is blocked. Allow it from the icon in the address bar, then press Turn on camera again.";
  if (error?.name === "NotFoundError") return "No camera was found. Connect a webcam and try again.";
  if (error?.name === "NotReadableError") return "Another app is using the camera. Close it and try again.";
  return "The hand tracker did not load. Check your internet connection and reload the page.";
}

function trackerInput(now) {
  masks = masks.filter((m) => m.until > now);
  if (!masks.length) return video;
  scratchContext.drawImage(video, 0, 0);
  scratchContext.fillStyle = "rgb(128,128,128)";
  for (const m of masks) {
    const x0 = (1 - m.x1) * scratch.width;
    scratchContext.fillRect(x0, m.y0 * scratch.height, (m.x1 - m.x0) * scratch.width, (m.y1 - m.y0) * scratch.height);
  }
  return scratch;
}

function maskIgnored(ignored, kept, now) {
  for (const hand of ignored) {
    const xs = hand.landmarks.map((p) => p.x);
    const ys = hand.landmarks.map((p) => p.y);
    const padX = 0.3 * (Math.max(...xs) - Math.min(...xs)) + 8 / canvas.width;
    const padY = 0.3 * (Math.max(...ys) - Math.min(...ys)) + 8 / canvas.height;
    const box = { x0: Math.max(0, Math.min(...xs) - padX), y0: Math.max(0, Math.min(...ys) - padY),
      x1: Math.min(1, Math.max(...xs) + padX), y1: Math.min(1, Math.max(...ys) + padY), until: now + MASK_SECONDS };
    const overlaps = kept.some((other) => {
      const ox = other.landmarks.map((p) => p.x);
      const oy = other.landmarks.map((p) => p.y);
      return box.x0 < Math.max(...ox) && Math.min(...ox) < box.x1 && box.y0 < Math.max(...oy) && Math.min(...oy) < box.y1;
    });
    if (!overlaps) masks.push(box);
  }
}

function loop() {
  if (!stream) return;
  if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const stamp = performance.now();
    const now = stamp / 1000;
    const result = landmarker.detectForVideo(trackerInput(now), stamp);
    const detections = result.landmarks.map((landmarks, i) => ({
      handedness: result.handedness?.[i]?.[0]?.categoryName === "Left" ? "Right" : "Left",
      landmarks: landmarks.map((p) => ({ x: 1 - p.x, y: p.y, z: p.z })),
    }));
    const { kept, ignored } = typingHands(identity.assign(detections, now), calibration, detector.aspect);
    maskIgnored(ignored, kept, now);
    hands = kept;
    if (hands.length) handsSeen = now;
    frameTimes.push(now);
    while (frameTimes.length > 30) frameTimes.shift();
    step(now);
  }
  draw();
  requestAnimationFrame(loop);
}

function step(now) {
  if (mode === "ready" && now >= armedAt) {
    mode = "typing";
    setState("typing", "Typing");
  }
  if (mode !== "typing" || dragging !== null) {
    detector.reset();
  } else {
    for (const click of detector.update({ timestamp: now, hands })) {
      text = translator.apply(text, click.key.value);
      flash = { name: click.key.name, until: now + 0.5 };
      renderText();
    }
  }
  if (mode === "typing") {
    const fps = frameTimes.length > 1 ? (frameTimes.length - 1) / (frameTimes.at(-1) - frameTimes[0]) : 0;
    let detail = detector.status;
    if (!hands.length) {
      detail = now - handsSeen > 2.5
        ? "No hands in the picture. Point the camera down at your desk so both hands fit."
        : "No hands in the picture.";
    }
    if (now < noticeUntil) detail = notice;
    $("detail").textContent = fps && fps < 12 ? `${detail} · ${fps.toFixed(0)} FPS: tap slowly` : detail;
  }
}

function toCanvas([x, y]) {
  return [x * canvas.width, y * canvas.height];
}

function polygon(points) {
  context.beginPath();
  points.forEach(([x, y], i) => (i ? context.lineTo(x, y) : context.moveTo(x, y)));
  context.closePath();
}

function draw() {
  const w = canvas.width;
  const h = canvas.height;
  const now = performance.now() / 1000;
  const unit = w / 1280;
  context.save();
  context.translate(w, 0);
  context.scale(-1, 1);
  context.drawImage(video, 0, 0, w, h);
  context.restore();

  const hovered = new Map();
  const typing = mode === "typing";
  if (typing) {
    for (const [name, key] of detector.hover) {
      const phase = detector.phases.get(name);
      if (key && (!name.endsWith(":4") || phase === "moving")) hovered.set(key.name, PHASES[phase] || PHASES.rest);
    }
  }
  context.lineJoin = "round";
  for (const key of layout.keys) {
    const shape = corners(key).map(([x, y]) => toCanvas(calibration.mapToImage(x, y)));
    const inactive = key.value === "CONTROL" || key.value === "ALT";
    const pressed = flash && flash.name === key.name && flash.until > now;
    polygon(shape);
    context.fillStyle = pressed ? "rgba(255,181,71,0.75)" : "rgba(20,24,44,0.42)";
    context.fill();
    context.lineWidth = (hovered.has(key.name) ? 3 : 1.2) * unit;
    context.strokeStyle = hovered.get(key.name) || (inactive ? "rgba(233,228,216,0.18)" : "rgba(233,228,216,0.55)");
    context.stroke();
    const [cx, cy] = toCanvas(calibration.mapToImage(key.x + key.width / 2, key.y + key.height / 2));
    const height = Math.hypot(shape[3][0] - shape[0][0], shape[3][1] - shape[0][1]);
    context.font = `600 ${Math.max(9, Math.min(22 * unit, height * 0.42))}px "JetBrains Mono", monospace`;
    context.textAlign = "center";
    context.textBaseline = "middle";
    context.fillStyle = pressed ? "#1c2238" : inactive ? "rgba(233,228,216,0.3)" : "rgba(233,228,216,0.9)";
    context.fillText(key.label, cx, cy);
  }

  for (const hand of hands) {
    context.strokeStyle = "rgba(127,224,194,0.8)";
    context.lineWidth = 2 * unit;
    for (const [a, b] of CONNECTIONS) {
      context.beginPath();
      context.moveTo(...toCanvas([hand.landmarks[a].x, hand.landmarks[a].y]));
      context.lineTo(...toCanvas([hand.landmarks[b].x, hand.landmarks[b].y]));
      context.stroke();
    }
    for (const tip of TIPS) {
      const [x, y] = toCanvas([hand.landmarks[tip].x, hand.landmarks[tip].y]);
      const name = `${hand.handId}:${tip}`;
      const color = PHASES[detector.phases.get(name)] || PHASES.rest;
      context.beginPath();
      context.arc(x, y, 5 * unit, 0, Math.PI * 2);
      context.fillStyle = color;
      context.fill();
      if (!typing) continue;
      const key = detector.hover.get(name);
      const level = Math.min(detector.levels.get(name) || 0, 1);
      const lx = Math.min(w - 60 * unit, x + 10 * unit);
      const ly = Math.max(20 * unit, y - 12 * unit);
      context.font = `600 ${15 * unit}px "JetBrains Mono", monospace`;
      context.textAlign = "left";
      context.fillStyle = color;
      context.fillText(key ? key.label : "gap", lx, ly);
      context.fillStyle = "rgba(20,24,44,0.8)";
      context.fillRect(lx, ly + 9 * unit, 36 * unit, 5 * unit);
      context.fillStyle = color;
      context.fillRect(lx, ly + 9 * unit, 36 * unit * level, 5 * unit);
    }
  }

  if (!typing) {
    points.forEach((point, i) => {
      const [x, y] = toCanvas(point);
      context.beginPath();
      context.arc(x, y, (dragging === i ? 13 : 10) * unit, 0, Math.PI * 2);
      context.fillStyle = "#1c2238";
      context.fill();
      context.lineWidth = 3 * unit;
      context.strokeStyle = "#ffb547";
      context.stroke();
    });
  }
  if (mode === "ready") {
    const progress = 1 - Math.max(0, armedAt - now) / ARM_SECONDS;
    context.fillStyle = "rgba(20,24,44,0.85)";
    context.fillRect(0, 0, w, 8 * unit);
    context.fillStyle = "#ffb547";
    context.fillRect(0, 0, w * progress, 8 * unit);
  }
}

function pointerPosition(event) {
  const rect = canvas.getBoundingClientRect();
  return { x: (event.clientX - rect.left) / rect.width, y: (event.clientY - rect.top) / rect.height, rect };
}

canvas.addEventListener("pointerdown", (event) => {
  if (!stream || mode === "typing" || mode === "ready") return;
  const { x, y, rect } = pointerPosition(event);
  const distances = points.map(([px, py]) => Math.hypot((px - x) * rect.width, (py - y) * rect.height));
  const nearest = distances.indexOf(Math.min(...distances));
  if (distances[nearest] > 32) return;
  dragging = nearest;
  canvas.setPointerCapture(event.pointerId);
});

canvas.addEventListener("pointermove", (event) => {
  const { x, y } = pointerPosition(event);
  if (dragging === null) {
    const rect = canvas.getBoundingClientRect();
    const near = stream && mode === "idle" && points.some(([px, py]) => Math.hypot((px - x) * rect.width, (py - y) * rect.height) < 32);
    canvas.style.cursor = near ? "grab" : "default";
    return;
  }
  const moved = points.map((p, i) => (i === dragging ? [Math.min(1, Math.max(0, x)), Math.min(1, Math.max(0, y))] : p));
  const problem = calibrationProblem(moved);
  if (problem) {
    $("detail").textContent = problem;
    return;
  }
  points = moved;
  calibration = new Calibration(points, true, flipRows);
  canvas.style.cursor = "grabbing";
});

function endDrag() {
  if (dragging === null) return;
  dragging = null;
  canvas.style.cursor = "default";
  rebuild();
  setMode(mode);
}

canvas.addEventListener("pointerup", endDrag);
canvas.addEventListener("pointercancel", endDrag);

$("start").addEventListener("click", start);
$("toggle").addEventListener("click", () => setMode(mode === "ready" || mode === "typing" ? "idle" : "ready"));
$("clear").addEventListener("click", () => {
  text = "";
  translator.capsLock = false;
  translator.shift = false;
  renderText();
});
$("copy").addEventListener("click", async () => {
  const button = $("copy");
  try {
    await navigator.clipboard.writeText(text);
    button.textContent = "Copied";
  } catch {
    button.textContent = "Copy failed";
  }
  setTimeout(() => (button.textContent = "Copy"), 1400);
});
$("flip").addEventListener("click", () => {
  flipRows = !flipRows;
  rebuild();
});
$("reset").addEventListener("click", () => {
  points = DEFAULT_POINTS.map((p) => [...p]);
  flipRows = false;
  rebuild();
});
$("sensitivity").value = sensitivity;
$("sensitivity-value").textContent = sensitivity.toFixed(2);
$("sensitivity").addEventListener("input", (event) => {
  sensitivity = Number(event.target.value);
  detector.sensitivity = sensitivity;
  $("sensitivity-value").textContent = sensitivity.toFixed(2);
  save();
});
$("camera").addEventListener("change", async (event) => {
  try {
    await openCamera(event.target.value);
    setMode(mode === "off" ? "off" : "idle");
  } catch (error) {
    notice = describe(error);
    noticeUntil = performance.now() / 1000 + 4;
    $("detail").textContent = notice;
  }
});

renderText();

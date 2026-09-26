from argparse import Namespace
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import threading
import time

import cv2

from helper_finger_press import FingerPressDetector
from helper_gesture import SETTLE_SECONDS, StartSign
from helper_keyboard import SpacedKeyboardLayout
from helper_session import FalseHandMasker, load_layout, load_movement_model
from helper_vision import HandHistory

ROOT = Path(__file__).resolve().parent


@dataclass
class EngineSettings:
    phone: bool = False
    camera: int = 0
    resolution: str = "640x360"
    exposure: str = "auto"
    sensitivity: float = 1.0
    start_sign: bool = False
    arm_seconds: float = 1.5
    auto_pause: float = 20.0
    model: str = str(ROOT / "models/hand_landmarker.task")
    calibration: str | None = None
    movement_profile: str | None = None

    def args(self):
        prefix = "phone_" if self.phone else ""
        return Namespace(
            phone=self.phone, camera=self.camera, rotation=None, no_mirror=None, model=self.model,
            calibration=self.calibration or str(ROOT / f"config/{prefix}air_keyboard_layout.json"),
            movement_profile=self.movement_profile or str(ROOT / f"config/{prefix}air_movement_model.json"))


def open_camera(settings):
    from helper_webcam import PhoneStream, WebcamStream
    if settings.phone:
        return PhoneStream()
    width, height = (settings.resolution or "640x360").lower().split("x")
    return WebcamStream(settings.camera, width=int(width), height=int(height), exposure=settings.exposure)


def open_tracker(model, mirrored):
    from helper_vision import MediaPipeHandTracker
    return MediaPipeHandTracker(model, input_mirrored=mirrored)


class TypingEngine:
    PREVIEW_WIDTH = 440
    PREVIEW_SECONDS = 0.1
    STATUS_SECONDS = 0.4

    def __init__(self, settings, on_key, on_status, on_preview=None,
                 camera_factory=open_camera, tracker_factory=open_tracker, clock=time.perf_counter):
        self.settings = settings
        self.on_key = on_key
        self.on_status = on_status
        self.on_preview = on_preview
        self.camera_factory = camera_factory
        self.tracker_factory = tracker_factory
        self.clock = clock
        self._stop = threading.Event()
        self._thread = None
        self.state = "off"

    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive()

    def start(self):
        if self.running:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="0keys-engine", daemon=True)
        self._thread.start()

    def stop(self, wait=True):
        self._stop.set()
        if wait and self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=5)

    def _report(self, state, message):
        self.state = state
        self.on_status(state, message)

    def _run(self):
        camera = tracker = None
        reason = "Typing is off"
        try:
            self._report("starting", "Opening the camera...")
            if not Path(self.settings.model).is_file():
                raise RuntimeError("Hand model missing: restart 0Keys to download it")
            session = Session(self.settings)
            camera = self.camera_factory(self.settings)
            tracker = self.tracker_factory(self.settings.model, session.view.mirrored)
            session.begin(self.clock())
            while not self._stop.is_set():
                ok, frame, now = camera.read()
                if not ok:
                    raise RuntimeError("The camera stopped sending pictures")
                result = session.process(frame, now, tracker, getattr(camera, "fps", 0.0))
                for key in result.keys:
                    self.on_key(key)
                if result.state != self.state or result.report:
                    self._report(result.state, result.message)
                if result.preview and self.on_preview is not None:
                    self.on_preview(session.preview())
                if result.finished:
                    reason = result.message
                    break
        except Exception as exc:
            reason = None
            self._report("error", str(exc) or type(exc).__name__)
        finally:
            for release in (getattr(tracker, "close", None), getattr(camera, "release", None)):
                try:
                    if release is not None:
                        release()
                except Exception:
                    pass
            if reason is not None:
                self._report("off", reason)


@dataclass
class StepResult:
    state: str
    message: str
    keys: list
    report: bool = False
    preview: bool = False
    finished: bool = False


class Session:
    def __init__(self, settings):
        self.settings = settings
        args = settings.args()
        from air_keyboard import context_for
        self.layout = SpacedKeyboardLayout()
        self.view, self.calibration, self.layout_notice = load_layout(args, settings.phone)
        context = context_for(args, self.view, self.calibration)
        context["gesture"] = "whole_hand_movement_v1"
        model, _ = load_movement_model(args.movement_profile, self.layout, context)
        self.detector = FingerPressDetector(self.layout, self.calibration, model, sensitivity=self.sensitivity())
        self.masker = FalseHandMasker()
        self.history = HandHistory()
        self.start_sign = StartSign()
        self.frame_times = deque(maxlen=30)
        self.frame = None
        self.snapshot = None
        self.last_key = None

    def sensitivity(self):
        return min(max(float(self.settings.sensitivity or 1.0), 0.4), 2.5)

    def begin(self, now):
        self.started = now
        self.hands_seen = now
        self.waiting_for_sign = bool(self.settings.start_sign)
        self.armed_at = now + self.settings.arm_seconds
        self.last_status = -10.0
        self.last_preview = -10.0

    def process(self, frame, now, tracker, camera_fps=0.0):
        frame = self.frame = self.view.apply(frame)
        size = (frame.shape[1], frame.shape[0])
        self.detector.aspect = size[0] / size[1]
        self.detector.sensitivity = self.sensitivity()
        snapshot = tracker.process(self.masker.prepare(frame, now), now)
        snapshot, _ = self.masker.filter(snapshot, self.calibration, self.detector.aspect, size, now)
        self.snapshot = snapshot
        self.history.append(snapshot)
        self.frame_times.append(now)
        if snapshot.hands:
            self.hands_seen = now
        keys = []

        auto_pause = self.settings.auto_pause
        if auto_pause and now - self.hands_seen > auto_pause:
            return StepResult("off", f"Paused automatically: no hands seen for {auto_pause:.0f} s", keys, finished=True)

        if self.waiting_for_sign:
            if self.start_sign.update(snapshot.hands, now, self.detector.aspect):
                self.waiting_for_sign = False
                self.armed_at = now + SETTLE_SECONDS
            state = "sign"
        else:
            state = "ready" if now < self.armed_at else "typing"
        if state == "typing":
            for press in self.detector.update(snapshot):
                keys.append(press.key)
                self.last_key = (press.key, now)
        else:
            self.detector.reset()

        report = now - self.last_status >= TypingEngine.STATUS_SECONDS or bool(keys)
        if report:
            self.last_status = now
        preview = now - self.last_preview >= TypingEngine.PREVIEW_SECONDS
        if preview:
            self.last_preview = now
        return StepResult(state, self.message(state, now, camera_fps), keys, report, preview)

    def message(self, state, now, camera_fps):
        fps = (len(self.frame_times) - 1) / max(self.frame_times[-1] - self.frame_times[0], 1e-3)
        if state == "sign":
            text = "Hold up the start sign: index finger up, other fingers folded"
        elif state == "ready":
            text = "Get ready: put your hands on the table"
        else:
            text = self.detector.status
        if not self.snapshot.hands:
            text = "No hands in the camera picture"
            if now - self.hands_seen > 2.5:
                text += (" - point the phone down at your hands" if self.settings.phone
                         else " - point the camera down at the table")
        return f"{text} | {fps:.0f} FPS"

    def preview(self):
        from air_keyboard import draw_layout
        from hand_tracking import draw_snapshot
        width = TypingEngine.PREVIEW_WIDTH
        height = int(self.frame.shape[0] * width / self.frame.shape[1])
        small = cv2.resize(self.frame, (width, height), interpolation=cv2.INTER_AREA)
        highlighted = set()
        if self.last_key is not None and self.snapshot is not None and self.snapshot.timestamp - self.last_key[1] < 0.5:
            highlighted = {self.last_key[0].name}
        draw_layout(small, self.layout, self.calibration, highlighted)
        if self.snapshot is not None:
            draw_snapshot(small, self.snapshot, self.history)
        return small

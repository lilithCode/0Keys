from collections import deque
import sys
import threading
import time

import cv2
import numpy as np

V4L2_MANUAL = 1


class WebcamStream:
    def __init__(self, index, width=640, height=360, fps=30, exposure="auto",
                 target_brightness=115.0, min_exposure=5, max_exposure=280,
                 capture=None, clock=time.perf_counter):
        if capture is None:
            backend = cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY
            capture = cv2.VideoCapture(index, backend)
        self.capture = capture
        self.clock = clock
        self.target_brightness = target_brightness
        self.min_exposure = min_exposure
        self.max_exposure = max_exposure
        self.exposure = None
        self.adaptive = False
        self.brightness = None
        self.note = "camera auto exposure"
        self._restore = None
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._frame = None
        self._time = None
        self._count = 0
        self._returned = 0
        self._error = None
        self._arrivals = deque(maxlen=31)
        if not self.capture.isOpened():
            self.capture.release()
            raise RuntimeError("Could not open laptop webcam. Close other camera programs")
        self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.capture.set(cv2.CAP_PROP_FPS, fps)
        self._setup_exposure(exposure)
        self._thread = threading.Thread(target=self._run, name="webcam", daemon=True)
        self._thread.start()

    def set_exposure(self, mode):
        with self._condition:
            self._restore_exposure()
            self.adaptive = False
            self.exposure = None
            self.note = "camera auto exposure"
            self._setup_exposure(mode)
        return self.note

    def _restore_exposure(self):
        if self._restore is None:
            return
        auto, exposure = self._restore
        self.capture.set(cv2.CAP_PROP_EXPOSURE, exposure)
        self.capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, auto)
        self._restore = None

    def _backend(self):
        try:
            return self.capture.getBackendName()
        except (AttributeError, cv2.error):
            return "unknown"

    def _setup_exposure(self, mode):
        mode = str(mode).lower()
        if mode == "auto":
            return
        if self._backend() != "V4L2":
            self.note = f"camera auto exposure ({self._backend()} backend)"
            return
        original = (self.capture.get(cv2.CAP_PROP_AUTO_EXPOSURE), self.capture.get(cv2.CAP_PROP_EXPOSURE))
        self.capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, V4L2_MANUAL)
        if round(self.capture.get(cv2.CAP_PROP_AUTO_EXPOSURE)) != V4L2_MANUAL:
            self.note = "camera auto exposure (manual exposure unsupported)"
            return
        self._restore = original
        self.adaptive = mode == "fast"
        start = 60.0 if self.adaptive else float(mode)
        self.exposure = min(max(start, self.min_exposure), 5000.0)
        self.capture.set(cv2.CAP_PROP_EXPOSURE, round(self.exposure))
        self._describe()

    def _describe(self):
        kind = "fast exposure" if self.adaptive else "fixed exposure"
        self.note = f"{kind} {self.exposure / 10:.1f} ms"

    def _adjust(self, frame):
        self.brightness = float(np.asarray(frame)[::6, ::6].mean())
        ratio = self.target_brightness / max(self.brightness, 1.0)
        if 0.85 <= ratio <= 1.18:
            return
        value = self.exposure * min(max(ratio, 0.7), 1.4)
        value = min(max(value, self.min_exposure), self.max_exposure)
        if abs(value - self.exposure) >= 1:
            self.exposure = value
            self.capture.set(cv2.CAP_PROP_EXPOSURE, round(value))
            self._describe()

    def _run(self):
        failures = 0
        while not self._stop.is_set():
            ok, frame = self.capture.read()
            now = self.clock()
            if not ok or frame is None:
                failures += 1
                if failures >= 30:
                    with self._condition:
                        self._error = "Webcam stopped returning frames"
                        self._condition.notify_all()
                    return
                time.sleep(0.01)
                continue
            failures = 0
            with self._condition:
                self._frame, self._time = frame, now
                self._count += 1
                self._arrivals.append(now)
                self._condition.notify_all()
            if self._count % 8 == 0:
                if self.adaptive:
                    self._adjust(frame)
                else:
                    self.brightness = float(np.asarray(frame)[::6, ::6].mean())

    @property
    def fps(self):
        with self._condition:
            if len(self._arrivals) < 2:
                return 0.0
            return (len(self._arrivals) - 1) / max(self._arrivals[-1] - self._arrivals[0], 1e-3)

    @property
    def dark(self):
        return self.brightness is not None and self.brightness < 45 and (
            not self.adaptive or self.exposure >= self.max_exposure)

    def read(self, timeout=2.0):
        with self._condition:
            self._condition.wait_for(
                lambda: self._count > self._returned or self._error or self._stop.is_set(), timeout)
            if self._count <= self._returned:
                if self._error:
                    raise RuntimeError(self._error)
                return False, None, None
            self._returned = self._count
            return True, self._frame, self._time

    def release(self):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2)
        self._restore_exposure()
        self.capture.release()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()


class PhoneStream:
    def __init__(self, clock=time.perf_counter, camera=None):
        if camera is None:
            from helper_phone import PhoneCamera

            camera = PhoneCamera()
        self.camera = camera
        self.clock = clock
        self.brightness = None
        self.exposure = None
        self.adaptive = False
        model = getattr(camera, "info", {}).get("model", "phone")
        self.note = f"{model} rear camera over USB"
        self._arrivals = deque(maxlen=31)
        self._count = 0

    def set_exposure(self, mode):
        return self.note + ", exposure is controlled by the phone"

    @property
    def fps(self):
        if len(self._arrivals) < 2:
            return 0.0
        return (len(self._arrivals) - 1) / max(self._arrivals[-1] - self._arrivals[0], 1e-3)

    @property
    def dark(self):
        return self.brightness is not None and self.brightness < 45

    def read(self, timeout=2.0):
        ok, frame = self.camera.read()
        if not ok:
            raise RuntimeError(self.camera.error or "Phone video stopped")
        stamp = self.camera.timestamp or self.clock()
        self._arrivals.append(stamp)
        self._count += 1
        if self._count % 8 == 0:
            self.brightness = float(np.asarray(frame)[::6, ::6].mean())
        return True, frame, stamp

    def release(self):
        self.camera.release()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()

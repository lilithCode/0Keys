import threading
import time
import unittest

import cv2
import numpy as np

from helper_webcam import PhoneStream, WebcamStream


class FakeCapture:
    """Behaves like a V4L2 webcam whose brightness follows the exposure time."""

    def __init__(self, backend="V4L2", manual_supported=True, gain=2.5, fail_after=None):
        self.backend = backend
        self.manual_supported = manual_supported
        self.gain = gain
        self.fail_after = fail_after
        self.props = {cv2.CAP_PROP_AUTO_EXPOSURE: 3.0, cv2.CAP_PROP_EXPOSURE: 156.0}
        self.calls = []
        self.frames = 0
        self.released = False
        self.lock = threading.Lock()

    def isOpened(self):
        return True

    def getBackendName(self):
        return self.backend

    def get(self, prop):
        return self.props.get(prop, 0.0)

    def set(self, prop, value):
        with self.lock:
            self.calls.append((prop, value))
            if prop == cv2.CAP_PROP_AUTO_EXPOSURE and value == 1 and not self.manual_supported:
                return False
            if prop == cv2.CAP_PROP_EXPOSURE and self.props[cv2.CAP_PROP_AUTO_EXPOSURE] != 1:
                return False  # Exposure time is inactive in auto mode.
            self.props[prop] = float(value)
        return True

    def read(self):
        time.sleep(0.002)
        with self.lock:
            self.frames += 1
            if self.fail_after is not None and self.frames > self.fail_after:
                return False, None
            manual = self.props[cv2.CAP_PROP_AUTO_EXPOSURE] == 1
            level = min(255, self.gain * self.props[cv2.CAP_PROP_EXPOSURE]) if manual else 150
            return True, np.full((36, 64, 3), level, np.uint8)

    def release(self):
        self.released = True


class WebcamStreamTests(unittest.TestCase):
    def test_fast_exposure_converges_and_is_restored(self):
        fake = FakeCapture()
        stream = WebcamStream(0, capture=fake, exposure="fast")
        deadline = time.time() + 3
        while time.time() < deadline and abs((stream.brightness or 0) - 115) > 20:
            time.sleep(0.01)
        self.assertTrue(stream.adaptive)
        self.assertLess(abs(stream.brightness - 115), 20)
        self.assertLessEqual(stream.exposure, 280)
        self.assertIn("fast exposure", stream.note)
        stream.release()
        self.assertTrue(fake.released)
        self.assertEqual(fake.props[cv2.CAP_PROP_AUTO_EXPOSURE], 3.0)
        self.assertEqual(fake.props[cv2.CAP_PROP_EXPOSURE], 156.0)

    def test_exposure_is_capped_to_keep_the_frame_rate(self):
        fake = FakeCapture(gain=0.1)
        with WebcamStream(0, capture=fake, exposure="fast") as stream:
            deadline = time.time() + 3
            while time.time() < deadline and stream.exposure < 280:
                time.sleep(0.01)
            self.assertEqual(stream.exposure, 280)
            self.assertTrue(stream.dark)

    def test_auto_mode_and_other_backends_leave_exposure_alone(self):
        for fake, mode in ((FakeCapture(), "auto"), (FakeCapture(backend="MSMF"), "fast"),
                           (FakeCapture(manual_supported=False), "fast")):
            with WebcamStream(0, capture=fake, exposure=mode) as stream:
                self.assertFalse(stream.adaptive)
                self.assertIn("auto exposure", stream.note)
            self.assertEqual(fake.props[cv2.CAP_PROP_AUTO_EXPOSURE], 3.0)
            self.assertNotIn((cv2.CAP_PROP_BUFFERSIZE, 1), fake.calls)

    def test_fixed_exposure_value_is_not_adjusted(self):
        fake = FakeCapture()
        with WebcamStream(0, capture=fake, exposure="45") as stream:
            for _ in range(20):
                self.assertTrue(stream.read()[0])
            self.assertEqual(stream.exposure, 45)
            self.assertEqual(fake.props[cv2.CAP_PROP_EXPOSURE], 45)

    def test_each_read_returns_a_newer_frame_with_its_arrival_time(self):
        with WebcamStream(0, capture=FakeCapture(), exposure="auto") as stream:
            times = []
            for _ in range(5):
                ok, frame, stamp = stream.read()
                self.assertTrue(ok)
                self.assertEqual(frame.shape, (36, 64, 3))
                times.append(stamp)
                time.sleep(0.01)
            self.assertTrue(all(b > a for a, b in zip(times, times[1:])))
            self.assertGreater(stream.fps, 0)

    def test_camera_failure_is_reported(self):
        with WebcamStream(0, capture=FakeCapture(fail_after=2), exposure="auto") as stream:
            stream.read()
            with self.assertRaises(RuntimeError):
                for _ in range(5):
                    stream.read(timeout=2)



class CameraChoiceTests(unittest.TestCase):
    def test_the_camera_keeps_its_own_exposure_by_default(self):
        # A forced short exposure is very noisy on webcams without gain control.
        fake = FakeCapture()
        with WebcamStream(0, capture=fake) as stream:
            self.assertFalse(stream.adaptive)
            self.assertIsNone(stream.exposure)
            self.assertEqual(stream.note, "camera auto exposure")
            for _ in range(12):
                stream.read()
            self.assertIsNotNone(stream.brightness)
        self.assertEqual(fake.props[cv2.CAP_PROP_AUTO_EXPOSURE], 3.0)
        self.assertNotIn(cv2.CAP_PROP_BUFFERSIZE, [prop for prop, _ in fake.calls])

    def test_exposure_mode_can_be_switched_while_running(self):
        fake = FakeCapture()
        with WebcamStream(0, capture=fake) as stream:
            self.assertIn("fast exposure", stream.set_exposure("fast"))
            self.assertTrue(stream.adaptive)
            self.assertEqual(fake.props[cv2.CAP_PROP_AUTO_EXPOSURE], 1.0)
            self.assertEqual(stream.set_exposure("auto"), "camera auto exposure")
            self.assertFalse(stream.adaptive)
            self.assertEqual(fake.props[cv2.CAP_PROP_AUTO_EXPOSURE], 3.0)
            self.assertEqual(fake.props[cv2.CAP_PROP_EXPOSURE], 156.0)
            self.assertTrue(stream.read()[0])


class FakePhone:
    info = {"model": "Redmi Note 10 Pro"}
    error = ""

    def __init__(self):
        self.timestamp = 0.0
        self.released = False

    def read(self):
        self.timestamp += 1 / 30
        return True, np.full((480, 640, 3), 120, np.uint8)

    def release(self):
        self.released = True


class PhoneStreamTests(unittest.TestCase):
    def test_phone_frames_carry_their_capture_time(self):
        phone = FakePhone()
        with PhoneStream(camera=phone) as stream:
            stamps = [stream.read()[2] for _ in range(10)]
            self.assertTrue(all(b > a for a, b in zip(stamps, stamps[1:])))
            self.assertIn("Redmi Note 10 Pro", stream.note)
            self.assertAlmostEqual(stream.fps, 30, delta=1)
            self.assertFalse(stream.dark)
        self.assertTrue(phone.released)

    def test_phone_failure_is_reported(self):
        phone = FakePhone()
        phone.read = lambda: (False, None)
        phone.error = "Phone video timed out. Unlock the phone and check USB"
        with PhoneStream(camera=phone) as stream:
            with self.assertRaises(RuntimeError):
                stream.read()

if __name__ == "__main__":
    unittest.main()

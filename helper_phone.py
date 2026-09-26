from __future__ import annotations

from fractions import Fraction
from pathlib import Path
import re
import secrets
import shutil
import socket
import struct
import subprocess
import tempfile
import threading
import time


def run_command(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"Could not run {command[0]}: {exc}") from exc
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "Phone command failed")
    return result.stdout.strip()


def check_phone():
    for name in ("adb", "scrcpy"):
        if not shutil.which(name):
            raise RuntimeError("Install USB tools: sudo pacman -S --needed android-tools scrcpy")
    state = run_command(["adb", "-d", "get-state"])
    if state != "device":
        raise RuntimeError("Unlock the phone and allow USB debugging for this laptop")
    sdk = run_command(["adb", "-d", "shell", "getprop", "ro.build.version.sdk"])
    if not sdk.isdigit() or int(sdk) < 31:
        raise RuntimeError("Direct phone camera needs Android 12 or newer. Check the phone's Android version")
    model = run_command(["adb", "-d", "shell", "getprop", "ro.product.model"])
    serial = run_command(["adb", "-d", "get-serialno"])
    match = re.search(r"scrcpy (\d+\.\d+(?:\.\d+)?)", run_command(["scrcpy", "--version"]))
    if not match:
        raise RuntimeError("Could not identify the installed scrcpy version")
    if int(match.group(1).split(".")[0]) not in (3, 4):
        raise RuntimeError("This camera backend supports scrcpy 3 and 4. Check the installed version")
    server = Path("/usr/share/scrcpy/scrcpy-server")
    if not server.is_file():
        raise RuntimeError("Missing scrcpy server at /usr/share/scrcpy/scrcpy-server. Reinstall the scrcpy package")
    return {"model": model, "serial": serial, "sdk": int(sdk),
            "version": match.group(1), "server": server}


def read_exact(connection, size):
    data = bytearray()
    while len(data) < size:
        chunk = connection.recv(size - len(data))
        if not chunk:
            raise RuntimeError("Phone video disconnected")
        data.extend(chunk)
    return bytes(data)


def read_packet(connection, modern=False):
    flags, size = struct.unpack(">QI", read_exact(connection, 12))
    if not 0 < size <= 8 * 1024 * 1024:
        raise RuntimeError("Invalid phone video packet. Check the scrcpy version")
    config_bit = 62 if modern else 63
    return bool(flags & (1 << config_bit)), flags & ((1 << (config_bit - 1)) - 1), read_exact(connection, size)


class PhoneCamera:
    def __init__(self):
        self.info = check_phone()
        try:
            import av
        except ImportError as exc:
            raise RuntimeError("Install the video decoder: python -m pip install -r requirements.txt") from exc
        self._av = av
        self._modern = int(self.info["version"].split(".")[0]) >= 4
        self._socket = None
        self._process = None
        self._thread = None
        self._port = None
        self._remote = None
        self._log = tempfile.TemporaryFile(mode="w+b")
        self._stop = threading.Event()
        self._condition = threading.Condition()
        self._latest = None
        self._sequence = 0
        self._consumed = 0
        self.timestamp = 0.0
        self.received_frames = 0
        self.error = ""
        self._adb = ["adb", "-s", self.info["serial"]]
        try:
            token = secrets.token_hex(4)
            scid = int(token, 16) & 0x7fffffff
            self._remote = f"/data/local/tmp/0keys-camera-{token}.jar"
            run_command([*self._adb, "push", str(self.info["server"]), self._remote])
            self._port = int(run_command([*self._adb, "forward", "tcp:0", f"localabstract:scrcpy_{scid:08x}"]))
            command = [*self._adb, "shell", f"CLASSPATH={self._remote}", "app_process", "/",
                       "com.genymobile.scrcpy.Server", self.info["version"], f"scid={scid:08x}",
                       "video_source=camera", "camera_facing=back", "camera_size=640x480",
                       "camera_fps=30", "video_codec=h264", "video_bit_rate=4000000",
                       "audio=false", "control=false", "tunnel_forward=true", "cleanup=false",
                       "send_device_meta=false", "send_dummy_byte=false",
                       "send_stream_meta=false" if self._modern else "send_codec_meta=false"]
            self._process = subprocess.Popen(command, stdout=self._log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                if self._process.poll() is not None:
                    raise RuntimeError("Phone camera could not start: " + self._log_text())
                try:
                    connection = socket.create_connection(("127.0.0.1", self._port), timeout=0.5)
                    connection.settimeout(5)
                    first = connection.recv(1, socket.MSG_PEEK)
                    if first:
                        self._socket = connection
                        break
                    connection.close()
                except OSError:
                    if 'connection' in locals():
                        connection.close()
                time.sleep(0.1)
            if self._socket is None:
                raise RuntimeError("No phone video. Unlock the phone and close other camera apps. " + self._log_text())
            self._thread = threading.Thread(target=self._decode, name="phone-camera", daemon=True)
            self._thread.start()
        except Exception:
            self.release()
            raise

    def _log_text(self):
        self._log.seek(0)
        return self._log.read().decode(errors="replace")[-1500:]

    def _decode(self):
        config = b""
        origin = None
        previous = None
        try:
            decoder = self._av.CodecContext.create("h264", "r")
            decoder.thread_count = 1
            while not self._stop.is_set():
                is_config, pts, payload = read_packet(self._socket, self._modern)
                if is_config:
                    config = payload
                    continue
                packet = self._av.Packet(config + payload)
                config = b""
                packet.pts = pts
                packet.time_base = Fraction(1, 1_000_000)
                for frame in decoder.decode(packet):
                    seconds = float(frame.pts * frame.time_base)
                    if previous is not None and seconds <= previous:
                        raise RuntimeError("Phone timestamps changed. Restart camera setup")
                    previous = seconds
                    arrived = time.perf_counter()
                    if origin is None:
                        origin = arrived - seconds
                    image = frame.to_ndarray(format="bgr24")
                    with self._condition:
                        self._latest = (origin + seconds, image)
                        self._sequence += 1
                        self.received_frames += 1
                        self._condition.notify_all()
        except Exception as exc:
            if not self._stop.is_set():
                with self._condition:
                    self.error = str(exc)
                    self._condition.notify_all()

    def isOpened(self):
        return not self._stop.is_set() and not self.error

    def read(self):
        with self._condition:
            available = self._condition.wait_for(
                lambda: self._sequence > self._consumed or self.error or self._stop.is_set(), timeout=6)
            if not available or self.error or self._stop.is_set():
                if not available:
                    self.error = "Phone video timed out. Unlock the phone and check USB"
                return False, None
            self._consumed = self._sequence
            self.timestamp, frame = self._latest
            if time.perf_counter() - self.timestamp > 0.3:
                self.error = "Phone video is delayed. Reconnect USB before typing"
                return False, None
            return True, frame

    def release(self):
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        if self._socket:
            try:
                self._socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self._socket.close()
            self._socket = None
        if self._thread:
            self._thread.join(timeout=2)
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=2)
        for arguments in (["forward", "--remove", f"tcp:{self._port}"] if self._port else [],
                          ["shell", "rm", "-f", self._remote] if self._remote else []):
            if arguments:
                try:
                    run_command([*self._adb, *arguments])
                except RuntimeError:
                    pass
        self._port = self._remote = None
        self._log.close()

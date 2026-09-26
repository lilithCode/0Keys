import json
from multiprocessing import AuthenticationError
from multiprocessing.connection import Client, Listener
import os
from pathlib import Path
import secrets
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading

from helper_output import session_kind

ROOT = Path(__file__).resolve().parent
DEFAULT_HOTKEY = "<ctrl>+<alt>+k"
COMMANDS = ("toggle", "on", "off", "show", "quit", "ping")
WINDOWS_PORT = 47931
HYPRLAND_MODS = {"ctrl": "CTRL", "alt": "ALT", "shift": "SHIFT", "cmd": "SUPER", "super": "SUPER"}


class AlreadyRunning(RuntimeError):
    pass


def channel_address(platform=None):
    platform = sys.platform if platform is None else platform
    if platform == "win32":
        return ("127.0.0.1", WINDOWS_PORT)
    base = os.environ.get("XDG_RUNTIME_DIR") or tempfile.gettempdir()
    return os.path.join(base, f"0keys-{os.getuid()}.sock")


def channel_key(path=None):
    path = Path(path) if path is not None else ROOT / "config" / ".control_key"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(secrets.token_bytes(32))
    return path.read_bytes()


def send_command(command, address=None, key=None):
    address = channel_address() if address is None else address
    try:
        with Client(address, authkey=channel_key() if key is None else key) as connection:
            connection.send(command)
            return connection.recv()
    except (OSError, EOFError, AuthenticationError):
        return None


class CommandServer:
    def __init__(self, handler, address=None, key=None):
        self.handler = handler
        self.address = channel_address() if address is None else address
        self.key = channel_key() if key is None else key
        self.listener = None
        self.closed = threading.Event()

    def start(self):
        if send_command("ping", self.address, self.key) is not None:
            raise AlreadyRunning("0Keys is already running")
        if isinstance(self.address, str) and os.path.exists(self.address):
            os.unlink(self.address)
        self.listener = Listener(self.address, authkey=self.key)
        threading.Thread(target=self._serve, name="0keys-commands", daemon=True).start()

    def _serve(self):
        while True:
            try:
                connection = self.listener.accept()
            except Exception:
                if self.closed.is_set():
                    return
                continue
            with connection:
                if self.closed.is_set():
                    return
                try:
                    command = connection.recv()
                    if command not in COMMANDS:
                        reply = f"Unknown command. Use one of: {', '.join(COMMANDS[:-1])}"
                    else:
                        reply = "pong" if command == "ping" else self.handler(command)
                    connection.send(reply)
                except (OSError, EOFError):
                    pass

    def close(self):
        if self.listener is None or self.closed.is_set():
            return
        self.closed.set()
        send_command("ping", self.address, self.key)
        self.listener.close()
        if isinstance(self.address, str) and os.path.exists(self.address):
            os.unlink(self.address)


def describe_hotkey(hotkey):
    parts = [part.strip("<>") for part in hotkey.split("+")]
    return "+".join(part.upper() if len(part) == 1 or part[1:].isdigit() else part.capitalize() for part in parts)


def hyprland_bind(hotkey):
    parts = [part.strip("<>").lower() for part in hotkey.split("+")]
    mods = [HYPRLAND_MODS[part] for part in parts[:-1] if part in HYPRLAND_MODS]
    if len(mods) != len(parts) - 1 or not parts[-1]:
        raise ValueError(f"Unsupported hotkey for Hyprland: {hotkey}")
    return " ".join(mods), parts[-1].upper()


def toggle_command():
    return f"{shlex.quote(sys.executable)} {shlex.quote(str(ROOT / 'app.py'))} toggle"


class HotkeyManager:
    def __init__(self, hotkey, callback, env=None, which=shutil.which, run=subprocess.run):
        self.hotkey = hotkey
        self.callback = callback
        self.env = os.environ if env is None else env
        self.which = which
        self.run = run
        self.listener = None
        self.hyprland = None
        self.label = describe_hotkey(hotkey)

    def start(self):
        kind = session_kind(self.env)
        if kind == "wayland":
            if self.env.get("HYPRLAND_INSTANCE_SIGNATURE") and self.which("hyprctl"):
                return self._start_hyprland()
            return False, (f"This desktop does not let apps watch the keyboard. Add a custom shortcut "
                           f"({self.label}) in your system keyboard settings that runs: {toggle_command()}")
        try:
            from pynput.keyboard import GlobalHotKeys
            self.listener = GlobalHotKeys({self.hotkey: self.callback})
            self.listener.start()
        except Exception as exc:
            return False, f"Hotkey unavailable ({exc}). Use the ON/OFF button or run: {toggle_command()}"
        note = " (allow 0Keys under Privacy > Accessibility and Input Monitoring)" if kind == "macos" else ""
        return True, f"Press {self.label} in any app to start or stop typing{note}"

    def _hyprctl(self, *args):
        try:
            result = self.run(["hyprctl", *args], capture_output=True, text=True, timeout=3)
        except (OSError, subprocess.TimeoutExpired):
            return None
        return result.stdout if result.returncode == 0 else None

    def _start_hyprland(self):
        try:
            mods, key = hyprland_bind(self.hotkey)
        except ValueError as exc:
            return False, str(exc)
        self.hyprland = (mods, key)
        if not self.bound() and self._hyprctl("keyword", "bind", f"{mods}, {key}, exec, {toggle_command()}") is None:
            self.hyprland = None
            return False, "Could not add the Hyprland shortcut. Use the ON/OFF button"
        return True, (f"Press {self.label} in any app to start or stop typing. To keep it after a Hyprland "
                      f"config reload, add to hyprland.conf: bind = {mods}, {key}, exec, {toggle_command()}")

    def bound(self):
        output = self._hyprctl("binds", "-j")
        if not output:
            return False
        try:
            return any("app.py" in bind.get("arg", "") and bind.get("arg", "").endswith(" toggle")
                       for bind in json.loads(output))
        except (ValueError, AttributeError):
            return False

    def refresh(self):
        if self.hyprland is not None and not self.bound():
            mods, key = self.hyprland
            self._hyprctl("keyword", "bind", f"{mods}, {key}, exec, {toggle_command()}")

    def stop(self):
        if self.listener is not None:
            self.listener.stop()
            self.listener = None
        if self.hyprland is not None:
            mods, key = self.hyprland
            self._hyprctl("keyword", "unbind", f"{mods}, {key}")
            self.hyprland = None

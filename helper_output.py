import os
import shutil
import subprocess
import sys

from helper_fusion import SHIFTED_VALUES

SPECIAL_KEYS = {"BACKSPACE": "BackSpace", "ENTER": "Return", "TAB": "Tab"}
YDOTOOL_CODES = {"BackSpace": 14, "Return": 28, "Tab": 15}


class OutputError(RuntimeError):
    pass


class KeyTranslator:
    def __init__(self):
        self.caps_lock = False
        self.shift = False

    def reset(self):
        self.caps_lock = False
        self.shift = False

    def translate(self, value):
        if value in SPECIAL_KEYS:
            return "key", SPECIAL_KEYS[value]
        if value == "CAPS_LOCK":
            self.caps_lock = not self.caps_lock
            return None
        if value == "SHIFT":
            self.shift = not self.shift
            return None
        if value in ("CONTROL", "ALT"):
            return None
        character = value
        if len(character) == 1 and character.isalpha():
            if self.caps_lock != self.shift:
                character = character.upper()
        elif self.shift:
            character = SHIFTED_VALUES.get(character, character)
        self.shift = False
        return "text", character


class CommandSender:
    name = ""
    tool = ""

    def run(self, command):
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=3)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise OutputError(f"{self.tool} failed: {exc}") from exc
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip().splitlines()
            raise OutputError(f"{self.tool} failed: {detail[-1] if detail else result.returncode}")

    def send(self, kind, value):
        self.run(self.command(kind, value))


class WtypeSender(CommandSender):
    name, tool = "wtype (Wayland)", "wtype"

    def command(self, kind, value):
        return ["wtype", "-k", value] if kind == "key" else ["wtype", "--", value]


class XdotoolSender(CommandSender):
    name, tool = "xdotool (X11)", "xdotool"

    def command(self, kind, value):
        if kind == "key":
            return ["xdotool", "key", "--clearmodifiers", value]
        return ["xdotool", "type", "--clearmodifiers", "--", value]


class YdotoolSender(CommandSender):
    name, tool = "ydotool (Wayland)", "ydotool"

    def command(self, kind, value):
        if kind == "key":
            code = YDOTOOL_CODES[value]
            return ["ydotool", "key", f"{code}:1", f"{code}:0"]
        return ["ydotool", "type", "--", value]


class PynputSender:
    name = "pynput"

    def __init__(self):
        from pynput.keyboard import Controller, Key
        self.controller = Controller()
        self.keys = {"BackSpace": Key.backspace, "Return": Key.enter, "Tab": Key.tab}

    def send(self, kind, value):
        try:
            if kind == "key":
                self.controller.tap(self.keys[value])
            else:
                self.controller.type(value)
        except Exception as exc:
            raise OutputError(f"pynput failed: {exc}") from exc


class CallbackSender:
    name = "this window"

    def __init__(self, callback):
        self.callback = callback

    def send(self, kind, value):
        self.callback(kind, value)


COMMAND_SENDERS = {"wtype": WtypeSender, "xdotool": XdotoolSender, "ydotool": YdotoolSender}


def session_kind(env=None, platform=None):
    env = os.environ if env is None else env
    platform = sys.platform if platform is None else platform
    if platform == "win32":
        return "windows"
    if platform == "darwin":
        return "macos"
    if env.get("WAYLAND_DISPLAY") or env.get("XDG_SESSION_TYPE", "").lower() == "wayland":
        return "wayland"
    return "x11"


def candidate_tools(env=None, platform=None):
    env = os.environ if env is None else env
    kind = session_kind(env, platform)
    if kind in ("windows", "macos"):
        return ["pynput"]
    if kind == "x11":
        return ["xdotool", "pynput"]
    desktop = env.get("XDG_CURRENT_DESKTOP", "").lower()
    return ["ydotool", "wtype"] if "gnome" in desktop else ["wtype", "ydotool"]


def install_hint(tool, which=shutil.which):
    if tool == "pynput":
        return "python -m pip install -r requirements.txt"
    for manager, command in (("pacman", "sudo pacman -S {}"), ("apt", "sudo apt install {}"),
                             ("dnf", "sudo dnf install {}"), ("zypper", "sudo zypper install {}"),
                             ("brew", "brew install {}")):
        if which(manager):
            return command.format(tool)
    return f"install {tool} with your package manager"


def choose_sender(preference="auto", env=None, platform=None, which=shutil.which):
    tools = candidate_tools(env, platform) if preference in (None, "", "auto") else [preference]
    for tool in tools:
        if tool == "pynput":
            try:
                sender = PynputSender()
            except Exception:
                continue
            return sender, f"Typing with {sender.name}"
        if tool in COMMAND_SENDERS and which(tool):
            sender = COMMAND_SENDERS[tool]()
            note = " (ydotoold must be running)" if tool == "ydotool" else ""
            return sender, f"Typing with {sender.name}{note}"
    wanted = tools[0]
    return None, f"Cannot type into other apps yet. Install {wanted}: {install_hint(wanted, which)}"

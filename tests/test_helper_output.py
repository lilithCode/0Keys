import subprocess
import unittest
from unittest.mock import MagicMock, patch

import helper_output
from helper_output import (KeyTranslator, OutputError, WtypeSender, XdotoolSender, YdotoolSender,
                           candidate_tools, choose_sender, install_hint, session_kind)


def which_only(*tools):
    return lambda name: f"/usr/bin/{name}" if name in tools else None


class KeyTranslatorTests(unittest.TestCase):
    def test_letters_symbols_and_special_keys(self):
        translator = KeyTranslator()
        self.assertEqual(translator.translate("a"), ("text", "a"))
        self.assertEqual(translator.translate(" "), ("text", " "))
        self.assertEqual(translator.translate("BACKSPACE"), ("key", "BackSpace"))
        self.assertEqual(translator.translate("ENTER"), ("key", "Return"))
        self.assertEqual(translator.translate("TAB"), ("key", "Tab"))
        self.assertIsNone(translator.translate("CONTROL"))
        self.assertIsNone(translator.translate("ALT"))

    def test_shift_applies_to_one_key_and_caps_lock_stays(self):
        translator = KeyTranslator()
        self.assertIsNone(translator.translate("SHIFT"))
        self.assertEqual(translator.translate("h"), ("text", "H"))
        self.assertEqual(translator.translate("i"), ("text", "i"))
        translator.translate("SHIFT")
        self.assertEqual(translator.translate("1"), ("text", "!"))
        translator.translate("CAPS_LOCK")
        self.assertEqual(translator.translate("a"), ("text", "A"))
        self.assertEqual(translator.translate("b"), ("text", "B"))
        self.assertEqual(translator.translate("2"), ("text", "2"))
        translator.translate("SHIFT")
        self.assertEqual(translator.translate("c"), ("text", "c"))
        translator.reset()
        self.assertEqual(translator.translate("d"), ("text", "d"))


class BackendChoiceTests(unittest.TestCase):
    def test_session_kinds(self):
        self.assertEqual(session_kind({}, "win32"), "windows")
        self.assertEqual(session_kind({}, "darwin"), "macos")
        self.assertEqual(session_kind({"WAYLAND_DISPLAY": "wayland-1"}, "linux"), "wayland")
        self.assertEqual(session_kind({"XDG_SESSION_TYPE": "wayland"}, "linux"), "wayland")
        self.assertEqual(session_kind({"DISPLAY": ":0"}, "linux"), "x11")

    def test_tools_are_ordered_for_each_desktop(self):
        hyprland = {"WAYLAND_DISPLAY": "w", "XDG_CURRENT_DESKTOP": "Hyprland"}
        gnome = {"WAYLAND_DISPLAY": "w", "XDG_CURRENT_DESKTOP": "ubuntu:GNOME"}
        self.assertEqual(candidate_tools(hyprland, "linux"), ["wtype", "ydotool"])
        self.assertEqual(candidate_tools(gnome, "linux"), ["ydotool", "wtype"])
        self.assertEqual(candidate_tools({"DISPLAY": ":0"}, "linux"), ["xdotool", "pynput"])
        self.assertEqual(candidate_tools({}, "win32"), ["pynput"])
        self.assertEqual(candidate_tools({}, "darwin"), ["pynput"])

    def test_first_installed_tool_is_used(self):
        env = {"WAYLAND_DISPLAY": "w", "XDG_CURRENT_DESKTOP": "Hyprland"}
        sender, message = choose_sender("auto", env, "linux", which_only("wtype", "ydotool"))
        self.assertIsInstance(sender, WtypeSender)
        sender, message = choose_sender("auto", env, "linux", which_only("ydotool"))
        self.assertIsInstance(sender, YdotoolSender)
        self.assertIn("ydotoold", message)

    def test_missing_tool_names_the_install_command(self):
        env = {"WAYLAND_DISPLAY": "w", "XDG_CURRENT_DESKTOP": "Hyprland"}
        sender, message = choose_sender("auto", env, "linux", which_only("pacman"))
        self.assertIsNone(sender)
        self.assertIn("sudo pacman -S wtype", message)
        self.assertEqual(install_hint("wtype", which_only("apt")), "sudo apt install wtype")

    def test_pynput_used_on_x11_without_xdotool(self):
        with patch.object(helper_output, "PynputSender") as pynput:
            pynput.return_value.name = "pynput"
            sender, _ = choose_sender("auto", {"DISPLAY": ":0"}, "linux", which_only())
        self.assertIs(sender, pynput.return_value)


class CommandTests(unittest.TestCase):
    def test_commands_for_each_tool(self):
        self.assertEqual(WtypeSender().command("text", "-a"), ["wtype", "--", "-a"])
        self.assertEqual(WtypeSender().command("key", "BackSpace"), ["wtype", "-k", "BackSpace"])
        self.assertEqual(XdotoolSender().command("text", "x"), ["xdotool", "type", "--clearmodifiers", "--", "x"])
        self.assertEqual(XdotoolSender().command("key", "Return"), ["xdotool", "key", "--clearmodifiers", "Return"])
        self.assertEqual(YdotoolSender().command("key", "Return"), ["ydotool", "key", "28:1", "28:0"])
        self.assertEqual(YdotoolSender().command("text", "q"), ["ydotool", "type", "--", "q"])

    def test_send_runs_the_command_and_reports_failures(self):
        with patch.object(helper_output.subprocess, "run",
                          return_value=MagicMock(returncode=0, stdout="", stderr="")) as run:
            WtypeSender().send("text", "a")
        self.assertEqual(run.call_args.args[0], ["wtype", "--", "a"])
        failed = MagicMock(returncode=1, stdout="", stderr="Compositor does not support the virtual keyboard\n")
        with patch.object(helper_output.subprocess, "run", return_value=failed):
            with self.assertRaisesRegex(OutputError, "virtual keyboard"):
                WtypeSender().send("text", "a")
        with patch.object(helper_output.subprocess, "run", side_effect=subprocess.TimeoutExpired("wtype", 3)):
            with self.assertRaises(OutputError):
                WtypeSender().send("text", "a")


if __name__ == "__main__":
    unittest.main()

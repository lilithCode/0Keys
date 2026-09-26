import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import MagicMock

from helper_hotkey import (AlreadyRunning, CommandServer, HotkeyManager, channel_key, describe_hotkey,
                           hyprland_bind, send_command)


@unittest.skipIf(sys.platform == "win32", "Unix socket channel")
class CommandChannelTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        self.address = str(root / "0keys.sock")
        self.key = channel_key(root / "key")

    def tearDown(self):
        self.directory.cleanup()

    def test_key_file_is_private_and_reused(self):
        path = Path(self.directory.name) / "key"
        self.assertEqual(os.stat(path).st_mode & 0o777, 0o600)
        self.assertEqual(channel_key(path), self.key)

    def test_commands_reach_the_running_app(self):
        received = []
        seen = threading.Event()

        def handler(command):
            received.append(command)
            seen.set()
            return f"did {command}"

        server = CommandServer(handler, self.address, self.key)
        server.start()
        try:
            self.assertEqual(send_command("toggle", self.address, self.key), "did toggle")
            self.assertTrue(seen.wait(2))
            self.assertEqual(received, ["toggle"])
            self.assertIn("Unknown", send_command("format disk", self.address, self.key))
            self.assertIsNone(send_command("toggle", self.address, b"wrong key"))
            with self.assertRaises(AlreadyRunning):
                CommandServer(handler, self.address, self.key).start()
        finally:
            server.close()
        self.assertIsNone(send_command("toggle", self.address, self.key))
        self.assertFalse(os.path.exists(self.address))

    def test_stale_socket_from_a_crash_is_replaced(self):
        Path(self.address).touch()
        server = CommandServer(lambda command: "ok", self.address, self.key)
        server.start()
        try:
            self.assertEqual(send_command("on", self.address, self.key), "ok")
        finally:
            server.close()


class HotkeyTests(unittest.TestCase):
    def test_names_and_hyprland_binds(self):
        self.assertEqual(describe_hotkey("<ctrl>+<alt>+k"), "Ctrl+Alt+K")
        self.assertEqual(describe_hotkey("<f9>"), "F9")
        self.assertEqual(hyprland_bind("<ctrl>+<alt>+k"), ("CTRL ALT", "K"))
        self.assertEqual(hyprland_bind("<cmd>+k"), ("SUPER", "K"))
        with self.assertRaises(ValueError):
            hyprland_bind("<hyper>+k")

    def test_hyprland_bind_is_added_restored_and_removed(self):
        calls = []
        binds = []

        def run(command, **_):
            calls.append(command[1:])
            if command[1] == "binds":
                return MagicMock(returncode=0, stdout=json.dumps(binds))
            return MagicMock(returncode=0, stdout="ok")

        env = {"WAYLAND_DISPLAY": "w", "HYPRLAND_INSTANCE_SIGNATURE": "abc"}
        manager = HotkeyManager("<ctrl>+<alt>+k", lambda: None, env=env, which=lambda _: "/usr/bin/hyprctl", run=run)
        working, message = manager.start()
        self.assertTrue(working)
        self.assertIn("hyprland.conf", message)
        bind = next(call for call in calls if call[:2] == ["keyword", "bind"])
        self.assertTrue(bind[2].startswith("CTRL ALT, K, exec, "))
        self.assertTrue(bind[2].endswith("app.py toggle") or bind[2].endswith("app.py' toggle"))

        binds.append({"arg": bind[2].split("exec, ", 1)[1]})
        calls.clear()
        manager.refresh()
        self.assertEqual(calls, [["binds", "-j"]])
        binds.clear()
        manager.refresh()
        self.assertEqual(calls[-1][:2], ["keyword", "bind"])

        manager.stop()
        self.assertEqual(calls[-1], ["keyword", "unbind", "CTRL ALT, K"])

    def test_other_wayland_desktops_get_instructions(self):
        manager = HotkeyManager("<ctrl>+<alt>+k", lambda: None, env={"WAYLAND_DISPLAY": "w"}, which=lambda _: None)
        working, message = manager.start()
        self.assertFalse(working)
        self.assertIn("toggle", message)
        self.assertIn("Ctrl+Alt+K", message)


if __name__ == "__main__":
    unittest.main()

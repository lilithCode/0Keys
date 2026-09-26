import json
from pathlib import Path
import tempfile
import tkinter as tk
import unittest
from unittest.mock import MagicMock, patch

import app
import helper_hotkey
import helper_output
from helper_keyboard import SpacedKeyboardLayout

KEYS = {key.name: key for key in SpacedKeyboardLayout().keys}


class SettingsTests(unittest.TestCase):
    def test_defaults_saved_values_and_bad_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            path.write_text(json.dumps({"camera": 2, "sensitivity": 1, "start_sign": "yes", "unknown": 1}))
            settings = app.AppSettings(path)
            self.assertEqual(settings["camera"], 2)
            self.assertEqual(settings["sensitivity"], 1)
            self.assertFalse(settings["start_sign"])
            settings["target"] = "window"
            self.assertEqual(json.loads(path.read_text())["target"], "window")
            engine = app.AppSettings(path).engine()
            self.assertEqual((engine.camera, engine.sensitivity, engine.auto_pause), (2, 1.0, 20.0))

    def test_text_readout(self):
        text = app.apply_to_text("", "text", "h")
        text = app.apply_to_text(text, "text", "i")
        self.assertEqual(app.apply_to_text(text, "key", "BackSpace"), "h")
        self.assertEqual(app.apply_to_text(text, "key", "Return"), "hi\n")


def display_available():
    try:
        tk.Tk().destroy()
        return True
    except tk.TclError:
        return False


@unittest.skipUnless(display_available(), "needs a display for Tk")
class PanelTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.sender = MagicMock(name="sender")
        patches = [patch.object(helper_hotkey, "CommandServer"),
                   patch.object(helper_hotkey, "HotkeyManager"),
                   patch.object(helper_output, "choose_sender", return_value=(self.sender, "Typing with test")),
                   patch.object(app.ControlPanel, "_ensure_model")]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)
        helper_hotkey.HotkeyManager.return_value.start.return_value = (True, "Press Ctrl+Alt+K")
        self.root = tk.Tk()
        self.settings = app.AppSettings(Path(self.directory.name) / "settings.json")
        self.panel = app.ControlPanel(self.root, self.settings)

    def tearDown(self):
        self.panel.quit()
        self.directory.cleanup()

    def type_keys(self, *names, focused):
        with patch.object(self.root, "focus_displayof", return_value=self.root if focused else None):
            for name in names:
                self.panel._typed(KEYS[name])

    def test_keys_go_to_the_focused_app_with_shift_resolved(self):
        self.type_keys("left_shift", "h", "i", "backspace", focused=False)
        self.panel.outgoing.join()
        self.assertEqual([c.args for c in self.sender.send.call_args_list],
                         [("text", "H"), ("text", "i"), ("key", "BackSpace")])
        self.assertIn("H", self.panel.recent_label.cget("text"))

    def test_keys_stay_in_the_panel_while_it_has_focus(self):
        self.type_keys("o", "k", "k", "backspace", focused=True)
        self.panel.outgoing.join()
        self.sender.send.assert_not_called()
        self.assertEqual(self.panel.test_box.get("1.0", "end-1c"), "ok")

    def test_this_window_only_target(self):
        self.panel.target.set("window")
        self.type_keys("a", focused=False)
        self.panel.outgoing.join()
        self.sender.send.assert_not_called()
        self.assertEqual(self.panel.test_box.get("1.0", "end-1c"), "a")

    def test_toggle_command_starts_and_stops_the_engine(self):
        with patch("helper_engine.TypingEngine") as engine_class:
            engine = engine_class.return_value
            engine.running = False
            with patch.object(Path, "is_file", return_value=True), patch.object(app, "notify"):
                self.panel._handle_command("toggle")
                engine.start.assert_called_once()
                engine.running = True
                self.panel._handle_command("toggle")
                engine.stop.assert_called_once_with(wait=False)

    def test_status_from_an_old_engine_is_ignored(self):
        self.panel.engine_id = 2
        self.panel.events.put(("status", 1, "typing", "old"))
        self.panel.events.put(("status", 2, "ready", "Get ready"))
        self.panel._poll()
        self.assertEqual(self.panel.state_label.cget("text"), "GET READY")


if __name__ == "__main__":
    unittest.main()

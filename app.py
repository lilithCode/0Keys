import argparse
import base64
import json
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import ttk
import urllib.request

ROOT = Path(__file__).resolve().parent
SETTINGS_PATH = ROOT / "config" / "app_settings.json"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")

COLORS = {"bg": "#16181e", "panel": "#1f222b", "field": "#2a2e39", "text": "#e8eaf0", "muted": "#9aa1b2",
          "accent": "#46e6aa", "warn": "#ffb454", "error": "#ff6b6b", "off": "#5c6373", "info": "#6cb6ff"}
STATES = {"off": ("OFF", "off"), "starting": ("STARTING", "info"), "ready": ("GET READY", "warn"),
          "sign": ("SHOW START SIGN", "warn"), "typing": ("TYPING", "accent"), "error": ("ERROR", "error")}


class AppSettings:
    DEFAULTS = {"phone": False, "camera": 0, "sensitivity": 1.0, "start_sign": False, "auto_pause": 20,
                "target": "apps", "output": "auto", "hotkey": "<ctrl>+<alt>+k", "preview": True}

    def __init__(self, path=SETTINGS_PATH):
        self.path = Path(path)
        self.values = dict(self.DEFAULTS)
        try:
            saved = json.loads(self.path.read_text())
            self.values.update({k: v for k, v in saved.items() if k in self.DEFAULTS
                                and isinstance(v, type(self.DEFAULTS[k]) if k != "sensitivity" else (int, float))})
        except (OSError, ValueError, AttributeError):
            pass

    def __getitem__(self, name):
        return self.values[name]

    def __setitem__(self, name, value):
        self.values[name] = value
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(json.dumps(self.values, indent=2) + "\n")
            temporary.replace(self.path)
        except OSError:
            pass

    def engine(self):
        from helper_engine import EngineSettings
        return EngineSettings(phone=self["phone"], camera=int(self["camera"]), sensitivity=float(self["sensitivity"]),
                              start_sign=self["start_sign"], auto_pause=float(self["auto_pause"]))


def notify(title, message):
    try:
        if sys.platform.startswith("linux") and shutil.which("notify-send"):
            subprocess.Popen(["notify-send", "-a", "0Keys", "-t", "2500", title, message],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif sys.platform == "darwin":
            script = f"display notification {json.dumps(message)} with title {json.dumps(title)}"
            subprocess.Popen(["osascript", "-e", script], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        pass


def apply_to_text(text, kind, value):
    if kind == "key":
        return text[:-1] if value == "BackSpace" else text + ("\n" if value == "Return" else "\t")
    return text + value


class ControlPanel:
    def __init__(self, root, settings, initial_command=None, minimized=False):
        from helper_hotkey import CommandServer, HotkeyManager, describe_hotkey
        from helper_output import KeyTranslator, choose_sender

        self.root = root
        self.settings = settings
        self.events = queue.Queue()
        self.outgoing = queue.Queue()
        self.translator = KeyTranslator()
        self.engine = None
        self.engine_id = 0
        self.setup_process = None
        self.preview_data = None
        self.preview_lock = threading.Lock()
        self.photo = None
        self.recent = ""
        self.closing = False

        self.sender, self.output_note = choose_sender(settings["output"])
        self.server = CommandServer(lambda command: self._queue_command(command))
        self.server.start()
        self.hotkeys = HotkeyManager(settings["hotkey"], lambda: self._queue_command("toggle"))
        _, self.hotkey_note = self.hotkeys.start()
        self.hotkey_label = describe_hotkey(settings["hotkey"])

        self._build()
        threading.Thread(target=self._send_keys, name="0keys-output", daemon=True).start()
        self._ensure_model()
        self.root.after(20, self._poll)
        self.root.after(30_000, self._refresh_hotkey)
        if initial_command:
            self._queue_command(initial_command)
        if minimized:
            self.root.iconify()

    def _build(self):
        root = self.root
        root.title("0Keys")
        root.configure(bg=COLORS["bg"])
        root.minsize(480, 400)
        root.protocol("WM_DELETE_WINDOW", root.iconify)
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure(".", background=COLORS["bg"], foreground=COLORS["text"], fieldbackground=COLORS["field"],
                        bordercolor=COLORS["field"], lightcolor=COLORS["field"], darkcolor=COLORS["field"],
                        troughcolor=COLORS["field"], font=("TkDefaultFont", 10))
        style.configure("Panel.TFrame", background=COLORS["panel"])
        style.configure("Panel.TLabel", background=COLORS["panel"])
        style.configure("Panel.TCheckbutton", background=COLORS["panel"])
        style.configure("Panel.TRadiobutton", background=COLORS["panel"])
        style.configure("Muted.TLabel", foreground=COLORS["muted"], background=COLORS["bg"])
        style.configure("Title.TLabel", font=("TkDefaultFont", 18, "bold"))
        style.configure("State.TLabel", font=("TkDefaultFont", 12, "bold"))
        style.configure("TButton", background=COLORS["field"], foreground=COLORS["text"], padding=(12, 6))
        style.map("TButton", background=[("active", "#353a48"), ("disabled", COLORS["panel"])],
                  foreground=[("disabled", COLORS["muted"])])
        style.configure("Big.TButton", font=("TkDefaultFont", 12, "bold"), padding=(16, 10),
                        background=COLORS["accent"], foreground="#0d1a14")
        style.map("Big.TButton", background=[("active", "#6ff0c0"), ("disabled", COLORS["field"])])
        style.configure("Stop.TButton", font=("TkDefaultFont", 12, "bold"), padding=(16, 10),
                        background=COLORS["error"], foreground="#1f0d0d")
        style.map("Stop.TButton", background=[("active", "#ff8f8f")])
        style.map("TCheckbutton", background=[("active", COLORS["panel"])])
        style.map("TRadiobutton", background=[("active", COLORS["panel"])])
        style.map("TCombobox", fieldbackground=[("readonly", COLORS["field"])], foreground=[("readonly", COLORS["text"])])
        root.option_add("*TCombobox*Listbox.background", COLORS["field"])
        root.option_add("*TCombobox*Listbox.foreground", COLORS["text"])

        outer = ttk.Frame(root, padding=16)
        outer.pack(fill="both", expand=True)
        wrap = 440

        header = ttk.Frame(outer)
        header.pack(fill="x")
        ttk.Label(header, text="0Keys", style="Title.TLabel").pack(side="left")
        self.light = tk.Canvas(header, width=18, height=18, bg=COLORS["bg"], highlightthickness=0)
        self.light.pack(side="right", padx=(8, 0))
        self.dot = self.light.create_oval(2, 2, 16, 16, fill=COLORS["off"], outline="")
        self.state_label = ttk.Label(header, text="OFF", style="State.TLabel")
        self.state_label.pack(side="right")

        self.toggle_button = ttk.Button(outer, style="Big.TButton", command=self.toggle)
        self.toggle_button.pack(fill="x", pady=(14, 6))
        self.detail = ttk.Label(outer, style="Muted.TLabel", wraplength=wrap, justify="left")
        self.detail.pack(fill="x")

        self.preview = tk.Label(outer, bg=COLORS["panel"], fg=COLORS["muted"], height=6,
                                text="The camera picture appears here while typing is on")
        self.preview.pack(fill="x", pady=(10, 0))

        typed = ttk.Frame(outer, style="Panel.TFrame", padding=10)
        typed.pack(fill="x", pady=(10, 0))
        ttk.Label(typed, text="Test area - keys land here while this window has focus",
                  style="Panel.TLabel", foreground=COLORS["muted"]).pack(anchor="w")
        self.test_box = tk.Text(typed, height=3, bg=COLORS["field"], fg=COLORS["text"], insertbackground=COLORS["text"],
                                relief="flat", wrap="word", font=("TkFixedFont", 11))
        self.test_box.pack(fill="x", pady=(6, 0))
        self.recent_label = ttk.Label(typed, style="Panel.TLabel", foreground=COLORS["muted"], text="Last typed: -")
        self.recent_label.pack(anchor="w", pady=(6, 0))

        options = ttk.Frame(outer, style="Panel.TFrame", padding=10)
        options.pack(fill="x", pady=(10, 0))
        options.columnconfigure(1, weight=1)

        ttk.Label(options, text="Camera", style="Panel.TLabel").grid(row=0, column=0, sticky="w", pady=3)
        self.cameras = [f"Webcam {index}" for index in range(4)]
        if sys.platform.startswith("linux") and shutil.which("adb") and shutil.which("scrcpy"):
            self.cameras.append("Phone over USB")
        self.camera_choice = tk.StringVar(value="Phone over USB" if self.settings["phone"] and "Phone over USB" in self.cameras
                                          else f"Webcam {self.settings['camera']}")
        camera = ttk.Combobox(options, textvariable=self.camera_choice, values=self.cameras, state="readonly", width=18)
        camera.grid(row=0, column=1, sticky="w", pady=3)
        camera.bind("<<ComboboxSelected>>", lambda _: self._camera_changed())

        ttk.Label(options, text="Type into", style="Panel.TLabel").grid(row=1, column=0, sticky="w", pady=3)
        self.target = tk.StringVar(value=self.settings["target"])
        targets = ttk.Frame(options, style="Panel.TFrame")
        targets.grid(row=1, column=1, sticky="w")
        for value, text in (("apps", "The focused app"), ("window", "This window only")):
            ttk.Radiobutton(targets, text=text, value=value, variable=self.target, style="Panel.TRadiobutton",
                            command=lambda: self._set("target", self.target.get())).pack(side="left", padx=(0, 10))

        ttk.Label(options, text="Sensitivity", style="Panel.TLabel").grid(row=2, column=0, sticky="w", pady=3)
        self.sensitivity = tk.DoubleVar(value=self.settings["sensitivity"])
        scale_row = ttk.Frame(options, style="Panel.TFrame")
        scale_row.grid(row=2, column=1, sticky="ew")
        ttk.Scale(scale_row, from_=0.4, to=2.5, variable=self.sensitivity, length=200,
                  command=lambda _: self._sensitivity_changed()).pack(side="left")
        self.sensitivity_label = ttk.Label(scale_row, style="Panel.TLabel", width=5)
        self.sensitivity_label.pack(side="left", padx=8)

        ttk.Label(options, text="Auto-pause", style="Panel.TLabel").grid(row=3, column=0, sticky="w", pady=3)
        pause_row = ttk.Frame(options, style="Panel.TFrame")
        pause_row.grid(row=3, column=1, sticky="w")
        self.auto_pause = tk.StringVar(value=str(self.settings["auto_pause"]))
        spin = ttk.Spinbox(pause_row, from_=0, to=600, increment=5, width=5, textvariable=self.auto_pause,
                           command=self._auto_pause_changed)
        spin.pack(side="left")
        spin.bind("<FocusOut>", lambda _: self._auto_pause_changed())
        ttk.Label(pause_row, text="seconds without hands (0 = never)", style="Panel.TLabel",
                  foreground=COLORS["muted"]).pack(side="left", padx=6)

        self.start_sign = tk.BooleanVar(value=self.settings["start_sign"])
        ttk.Checkbutton(options, text="Also require the start sign (index finger up) after the hotkey",
                        variable=self.start_sign, style="Panel.TCheckbutton",
                        command=lambda: self._set("start_sign", self.start_sign.get())
                        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=3)
        self.show_preview = tk.BooleanVar(value=self.settings["preview"])
        ttk.Checkbutton(options, text="Show the camera picture", variable=self.show_preview,
                        style="Panel.TCheckbutton", command=self._preview_changed
                        ).grid(row=5, column=0, columnspan=2, sticky="w", pady=3)

        notes = ttk.Frame(outer)
        notes.pack(fill="x", pady=(10, 0))
        self.output_label = ttk.Label(notes, style="Muted.TLabel", wraplength=wrap, justify="left")
        self.output_label.pack(fill="x")
        self.hotkey_text = ttk.Label(notes, style="Muted.TLabel", wraplength=wrap, justify="left",
                                     text=self.hotkey_note)
        self.hotkey_text.pack(fill="x", pady=(4, 0))

        buttons = ttk.Frame(outer)
        buttons.pack(fill="x", pady=(12, 0))
        self.setup_button = ttk.Button(buttons, text="Set up keyboard / train...", command=self.open_setup)
        self.setup_button.pack(side="left")
        ttk.Button(buttons, text="Quit 0Keys", command=self.quit).pack(side="right")

        self._sensitivity_changed(save=False)
        self._show_output_note()
        self._show_state("off", self._idle_message())

    def _idle_message(self):
        if not Path(self.settings.engine().args().calibration).exists():
            return "First time? Click \"Set up keyboard\" to place the keyboard under your hands."
        return f"Click into any app, press {self.hotkey_label}, then rest your hands on the table and tap."

    def _show_output_note(self):
        if self.target.get() == "window":
            text = "Keys go to the test area above only."
        else:
            text = self.output_note
        color = COLORS["error"] if self.sender is None and self.target.get() == "apps" else COLORS["muted"]
        self.output_label.configure(text=text, foreground=color)

    def _show_state(self, state, message):
        label, color = STATES.get(state, STATES["off"])
        self.state_label.configure(text=label, foreground=COLORS[color])
        self.light.itemconfigure(self.dot, fill=COLORS[color])
        on = state in ("starting", "ready", "sign", "typing")
        self.toggle_button.configure(text=f"{'Stop' if on else 'Start'} typing   ({self.hotkey_label})",
                                     style="Stop.TButton" if on else "Big.TButton",
                                     state="disabled" if self.setup_process is not None else "normal")
        self.detail.configure(text=message, foreground=COLORS["error"] if state == "error" else COLORS["muted"])
        if not on:
            self.preview.configure(image="", text="The camera picture appears here while typing is on", height=6)
            self.photo = None

    def _set(self, name, value):
        self.settings[name] = value
        if name == "target":
            self._show_output_note()

    def _camera_changed(self):
        choice = self.camera_choice.get()
        phone = choice == "Phone over USB"
        self.settings["phone"] = phone
        if not phone:
            self.settings["camera"] = int(choice.split()[-1])
        if self.engine is not None and self.engine.running:
            self.stop()
        self._show_state("off", self._idle_message())

    def _sensitivity_changed(self, save=True):
        value = round(float(self.sensitivity.get()), 2)
        self.sensitivity_label.configure(text=f"{value:.2f}")
        if save:
            self.settings["sensitivity"] = value
            if self.engine is not None:
                self.engine.settings.sensitivity = value

    def _auto_pause_changed(self):
        try:
            value = max(0, min(600, int(float(self.auto_pause.get()))))
        except ValueError:
            value = self.settings["auto_pause"]
        self.auto_pause.set(str(value))
        self.settings["auto_pause"] = value

    def _preview_changed(self):
        self.settings["preview"] = self.show_preview.get()
        if not self.show_preview.get():
            self.preview.configure(image="", text="Camera picture hidden", height=2)
            self.photo = None

    def toggle(self):
        if self.engine is not None and self.engine.running:
            self.stop()
        else:
            self.start()

    def start(self):
        from helper_engine import TypingEngine
        if self.setup_process is not None:
            return
        if not Path(self.settings.engine().model).is_file():
            self._show_state("error", "The hand model is still downloading. Try again in a moment.")
            return
        if self.engine is not None and self.engine.running:
            self.engine.stop()
        self.engine_id += 1
        engine_id = self.engine_id
        self.translator.reset()
        self.engine = TypingEngine(
            self.settings.engine(),
            on_key=lambda key: self.events.put(("key", engine_id, key)),
            on_status=lambda state, message: self.events.put(("status", engine_id, state, message)),
            on_preview=lambda image: self._store_preview(engine_id, image))
        self.engine.start()
        target = "this window" if self.target.get() == "window" else "the focused app"
        notify("0Keys is on", f"Typing into {target}. {self.hotkey_label} stops.")

    def stop(self):
        if self.engine is not None:
            self.engine.stop(wait=False)
            notify("0Keys is off", f"{self.hotkey_label} starts typing again.")

    def _store_preview(self, engine_id, image):
        if not self.settings["preview"]:
            return
        import cv2
        ok, png = cv2.imencode(".png", image)
        if ok:
            with self.preview_lock:
                self.preview_data = (engine_id, base64.b64encode(png.tobytes()))

    def _typed(self, key):
        action = self.translator.translate(key.value)
        if action is None:
            return
        kind, value = action
        self.recent = apply_to_text(self.recent, kind, value)[-40:]
        shown = self.recent.replace("\n", "⏎").replace("\t", "⇥")
        self.recent_label.configure(text=f"Last typed: {shown}")
        if self.target.get() == "window" or self.root.focus_displayof() is not None or self.sender is None:
            if kind == "key" and value == "BackSpace":
                self.test_box.delete("end-2c", "end-1c")
            else:
                self.test_box.insert("end", "\n" if value == "Return" else "\t" if value == "Tab" else value)
            self.test_box.see("end")
        else:
            self.outgoing.put(action)

    def _send_keys(self):
        from helper_output import OutputError
        while True:
            kind, value = self.outgoing.get()
            try:
                self.sender.send(kind, value)
            except OutputError as exc:
                self.events.put(("output_error", str(exc)))
            finally:
                self.outgoing.task_done()

    def open_setup(self):
        if self.setup_process is not None:
            return
        if self.engine is not None and self.engine.running:
            self.engine.stop()
        command = [sys.executable, str(ROOT / "keyboard.py"), "--no-start-sign",
                   "--sensitivity", str(self.settings["sensitivity"])]
        command += ["--phone"] if self.settings["phone"] else ["--camera", str(self.settings["camera"])]
        try:
            self.setup_process = subprocess.Popen(command, cwd=ROOT)
        except OSError as exc:
            self._show_state("error", f"Could not open the setup window: {exc}")
            return
        self.setup_button.configure(state="disabled")
        self._show_state("off", "Setup window open: drag the green corners so the keyboard sits under your "
                                "fingertips, then close it (Q) to come back here.")
        self.root.after(500, self._watch_setup)

    def _watch_setup(self):
        if self.setup_process.poll() is None:
            self.root.after(500, self._watch_setup)
            return
        self.setup_process = None
        self.setup_button.configure(state="normal")
        self._show_state("off", "Layout saved. " + self._idle_message())

    def _queue_command(self, command):
        self.events.put(("command", command))
        return f"0Keys: {command}"

    def _handle_command(self, command):
        if command == "toggle":
            self.toggle()
        elif command == "on" and not (self.engine is not None and self.engine.running):
            self.start()
        elif command == "off":
            self.stop()
        elif command == "show":
            self.root.deiconify()
            self.root.lift()
        elif command == "quit":
            self.quit()

    def _poll(self):
        if self.closing:
            return
        while True:
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            kind = event[0]
            if kind == "command":
                self._handle_command(event[1])
            elif kind == "status" and event[1] == self.engine_id:
                _, _, state, message = event
                if state == "off" and message == "Typing is off":
                    message = "Typing is off. " + self._idle_message()
                self._show_state(state, message)
                if state == "off" and message.startswith("Paused automatically"):
                    notify("0Keys paused", message)
                elif state == "error":
                    notify("0Keys stopped", message)
            elif kind == "key" and event[1] == self.engine_id:
                self._typed(event[2])
            elif kind == "output_error":
                self.output_label.configure(text=event[1], foreground=COLORS["error"])
            elif kind == "model":
                self._show_state(*event[1:])
        with self.preview_lock:
            data, self.preview_data = self.preview_data, None
        if data is not None and data[0] == self.engine_id and self.engine is not None and self.engine.running \
                and self.show_preview.get():
            self.photo = tk.PhotoImage(data=data[1])
            self.preview.configure(image=self.photo, text="", height=self.photo.height())
        self.root.after(20, self._poll)

    def _refresh_hotkey(self):
        if self.closing:
            return
        threading.Thread(target=self.hotkeys.refresh, daemon=True).start()
        self.root.after(30_000, self._refresh_hotkey)

    def _ensure_model(self):
        path = Path(self.settings.engine().model)
        if path.is_file():
            return

        def download():
            self.events.put(("model", "starting", "Downloading the hand tracking model (about 8 MB), once..."))
            temporary = path.with_suffix(".part")
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(MODEL_URL, temporary)
                temporary.replace(path)
                self.events.put(("model", "off", "Hand model ready. " + self._idle_message()))
            except OSError as exc:
                self.events.put(("model", "error", f"Could not download the hand model: {exc}. "
                                                   "Check the internet connection and restart 0Keys."))

        threading.Thread(target=download, daemon=True).start()

    def quit(self):
        if self.closing:
            return
        self.closing = True
        if self.engine is not None:
            self.engine.stop()
        if self.setup_process is not None and self.setup_process.poll() is None:
            self.setup_process.terminate()
        self.hotkeys.stop()
        self.server.close()
        self.root.destroy()


def main(argv=None):
    from helper_hotkey import AlreadyRunning, COMMANDS, send_command
    parser = argparse.ArgumentParser(description="0Keys: type into any app by tapping the table")
    parser.add_argument("command", nargs="?", choices=[c for c in COMMANDS if c != "ping"],
                        help="Control the running app. Starts it first when needed, except for off and quit")
    parser.add_argument("--minimized", action="store_true", help="Start minimised, for example at login")
    args = parser.parse_args(argv)

    reply = send_command(args.command or "show")
    if reply is not None:
        print(reply)
        return 0
    if args.command in ("off", "quit"):
        print("0Keys is not running")
        return 1
    root = tk.Tk()
    try:
        panel = ControlPanel(root, AppSettings(), initial_command=args.command if args.command != "show" else None,
                             minimized=args.minimized)
    except AlreadyRunning:
        root.destroy()
        print("0Keys is already running")
        return 0
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            panel.quit()
        except tk.TclError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

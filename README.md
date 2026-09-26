# 0Keys

**Type on your table instead of a keyboard.**

0Keys watches your hands through a camera. You tap your fingers on the table, as if a
keyboard were there, and the letters show up in whatever app you're using: a browser,
an editor, a chat window, anything.

**Try it in your browser first:** https://lilithcode.github.io/0Keys. There's nothing to install, and the video
never leaves your computer.

The desktop app runs quietly in the background. When you want to type with your hands, press
**Ctrl + Alt + K**. When you're done, press it again.

> **Heads up:** this is an experimental project. It works best with slow, deliberate
> taps. It won't keep up with fast touch typing yet, so try it on something where a
> typo doesn't matter before you rely on it.

---

## What you need

- A computer running **Linux, Windows or macOS**
- **Python 3.10, 3.11 or 3.12**, 3.12 if you can choose
- A **camera that can see your hands on the table**. This matters more than anything
  else (see [Where to put the camera](#where-to-put-the-camera)). Any of these works:
  - an Android phone propped above the table, the best option
  - a USB webcam clipped above the desk, pointing down
  - your laptop webcam, with the screen tilted forward
- Decent lighting

---

## Installing

### Linux and macOS

Open a terminal in the project folder and run:

```bash
./install.sh
```

The script sets up everything: the Python packages, the hand-tracking model, and a
**0Keys** entry in your app menu. To have 0Keys start whenever you log in, run
`./install.sh --autostart` instead.

### Windows

Right-click `install.ps1` and choose **Run with PowerShell**, or run:

```powershell
powershell -ExecutionPolicy Bypass -File install.ps1
```

This adds **0Keys** to your Start menu. Add `-Autostart` to start it at login.

### One extra step for typing into other apps

0Keys needs a small helper program to send keystrokes to other apps. On Windows and
macOS it's built in. On Linux, install the one for your desktop:

| Your desktop | Install this |
|---|---|
| Hyprland, Sway or another wlroots desktop | `wtype`, for example `sudo pacman -S wtype` or `sudo apt install wtype` |
| GNOME or KDE on Wayland | `ydotool`, and keep the `ydotoold` service running |
| Anything on X11 | `xdotool`, which is nice to have (0Keys can also manage without it) |

Not sure what you're missing? Just start 0Keys. The window tells you exactly which
command to run.

---

## First-time setup (about 2 minutes)

### 1. Start 0Keys

Open it from your app menu, or run `./0keys` on Linux and macOS (or `0keys.bat` on
Windows). A small dark window called **0Keys** appears. This is the control panel.

### 2. Where to put the camera

The camera has to look **down at your hands on the table**. A laptop webcam at its
normal angle looks at your face instead, and 0Keys will just keep saying *"No hands in
the camera picture."*

Make sure your **whole hands, including your wrists and palms,** fit in the picture.
If only your fingers are visible, the tracker keeps losing them.

If you're using an Android phone, see
[Using your phone as the camera](#using-your-phone-as-the-camera). Then pick
**Phone over USB** in the panel's *Camera* box.

### 3. Place the keyboard

Click **Set up keyboard / train...** in the panel. A bigger window opens that shows
the camera picture with a keyboard drawn over it.

- **Drag the four green corners** until the keys sit right under your fingertips.
- If the space bar ends up on the wrong side, press **F** to flip the rows.
- If the picture is sideways or mirrored, use **[** and **]** to rotate it, or **M**
  to mirror it.
- Try a few taps. The letters appear in that window's text box.
- Press **Q** to close it when you're happy. Your layout is saved automatically.

You only need to do this again if you move the camera.

---

## Everyday use

1. **Click into the app** you want to type in.
2. Press **Ctrl + Alt + K**.
3. Put your hands on the table. The light in the panel goes **amber (GET READY)** for
   a moment while you get into position, then **green (TYPING)**.
4. **Type:** rest your fingers for a second, then move a finger over a key, touch the
   table and lift. That's one keypress.
5. Press **Ctrl + Alt + K** again when you're done. The camera switches off.

A few things worth knowing:

- **Shift and Caps Lock** on the virtual keyboard work as toggles. Tap Shift once,
  and the next letter comes out capitalised.
- **Your thumbs** only type the space bar.
- If you **walk away**, 0Keys notices after 20 seconds with no hands in view and
  switches itself off. You can change the time in the panel, and 0 turns this off.
- If the **0Keys window itself is focused**, your taps go to the little test box in
  the panel instead of being typed somewhere. That's handy for practising.
- **Closing the window** only minimises it, so 0Keys keeps running. Click
  **Quit 0Keys** to close it for real.

You can also control 0Keys from a terminal:

```bash
./0keys toggle    # start or stop typing
./0keys on        # start typing
./0keys off       # stop typing
./0keys show      # bring the window to the front
./0keys quit      # close 0Keys
```

### About the Ctrl + Alt + K shortcut

- **Windows, macOS, and Linux on X11:** it just works. On macOS, the first time,
  allow Python under *System Settings → Privacy & Security → Accessibility* and
  *Input Monitoring*.
- **Hyprland:** it just works too, because 0Keys adds the shortcut for you while it's
  running. To make it permanent, copy the line shown in the panel into
  `hyprland.conf`. To start 0Keys at login, add
  `exec-once = /path/to/0Keys/0keys --minimized`.
- **GNOME, KDE and other Wayland desktops:** Wayland doesn't let apps listen for
  shortcuts, so add it yourself. Open your system's keyboard shortcut settings and
  create a custom shortcut for **Ctrl + Alt + K** that runs
  `/path/to/0Keys/0keys toggle`.

---

## Tips for better typing

- **Tap slowly and clearly,** especially at first. Touch, lift, pause a moment.
- **Rest before you start.** 0Keys learns what "still" looks like for each finger in
  the first second.
- **Watch the little labels in the setup window.** Each fingertip shows the key it's
  over, plus a small bar that fills as the finger presses down. A full bar means a
  press big enough to count.
- **Keys getting missed?** Move the **Sensitivity** slider up. In the setup window,
  press **+**.
- **Letters appearing when you didn't tap?** Move the slider down. In the setup
  window, press **-**.
- **Keep the background tidy.** Boxes and clutter at the far edge of the picture can
  sometimes be mistaken for a hand.
- **More light helps a lot.** In a bright room you can press **E** in the setup window
  to switch to a faster camera exposure (smoother, up to 30 FPS). In a dim room, keep
  the default exposure, because the fast one gets too noisy to track.

---

## The setup window in detail

The setup window is the one opened by **Set up keyboard / train...**, or directly
with `python keyboard.py`. It also has a built-in text box, so you can use it on its
own. These are its keys:

| Key | What it does |
|---|---|
| **T** | Start typing (in its own text box) |
| **Space** | Pause |
| **+ / -** | Raise or lower sensitivity |
| **[ / ]** | Rotate the camera picture left or right |
| **M** | Mirror the picture |
| **F** | Flip the keyboard rows |
| **E** | Switch camera exposure (normal / fast) |
| **L** | Learn mode: click a key and give it 5 example taps |
| **N** | Record 3 examples of "not a key" (resting, fidgeting) |
| **R** | Redo the current batch of learning examples |
| **C** | Clear the text |
| **Q** or **Esc** | Close |

When this window opens on its own, it waits for a **start sign**: hold one hand up
with your index finger pointing straight up and the other fingers curled in, for
about half a second. A green bar fills across the top, and then it starts typing.
Pressing **T** works too. The control panel doesn't need the sign, because the
Ctrl + Alt + K shortcut does that job. You can turn it on there as an extra check if
you like.

### Teaching it your own taps (optional)

0Keys works out of the box with no training. If a few keys keep going wrong, you can
teach it how *you* tap them:

1. Press **L** in the setup window, then click the key you want to teach.
2. Follow the countdown and tap that key five times.
3. Press **N** once and just rest or fidget normally. This teaches it what *isn't* a
   tap.
4. Press **T** to go back to typing.

Your examples are saved in the `config/` folder and loaded automatically next time.

---

## Using your phone as the camera

A phone propped above the table is the best camera for 0Keys, because it can see both
hands and wrists from above. This works on **Linux** with an Android phone (Android 12
or newer) connected by USB.

1. Install the phone tools. On Arch Linux:
   ```bash
   sudo pacman -S --needed android-tools scrcpy
   ```
2. On the phone, turn on **Developer options**. Usually you tap *Build number* seven
   times under *About phone*; on Xiaomi, tap *MIUI version*. Then turn on
   **USB debugging**.
3. Plug the phone in, unlock it, and accept the "Allow USB debugging?" prompt. If the
   prompt doesn't show up, try reconnecting the cable.
4. Prop the phone up so the back camera looks down at the table.
5. In the 0Keys panel, choose **Phone over USB** as the camera. From a terminal, use
   `python keyboard.py --phone` instead.

The phone's picture never gets saved anywhere. When 0Keys closes, it cleans up after
itself on the phone. You don't need wireless debugging or anything more invasive.

---

## Something not working?

| What you see | What to do |
|---|---|
| *"No hands in the camera picture"* | The camera isn't looking at the table. Point it down so whole hands fit in the picture. |
| Nothing gets typed into my app | Check the panel for a red message, which usually means a helper like `wtype` is missing. Also make sure you clicked into your app *before* pressing the shortcut. |
| Letters appear in the 0Keys window instead | The 0Keys window had focus. Click into your app first. |
| Ctrl + Alt + K does nothing | On GNOME, KDE and similar desktops, add the shortcut yourself (see [the shortcut section](#about-the-ctrl--alt--k-shortcut)). You can always use the Start button instead. |
| Keys are missed | Tap more slowly, raise sensitivity, and add light. |
| Random letters while resting | Lower sensitivity and keep the camera still. |
| Keys land one row off | Open **Set up keyboard** and adjust the green corners. |
| The panel shows ERROR about the camera | Another app may be using the camera, or pick a different webcam number in the panel. |
| 0Keys stopped by itself | It pauses after 20 seconds without seeing hands. Press the shortcut again. |

---

## How it works (the short version)

1. **Hand tracking.** Google's MediaPipe finds 21 points on each hand, about 16 to 30
   times a second.
2. **Finding the tap.** For each finger, 0Keys measures how far the fingertip bends
   and stretches compared to its own knuckle. Moving your whole hand doesn't count,
   only the finger itself. A tap is a quick press followed by a lift, and it has to
   be clearly bigger than the normal jitter of that finger at rest.
3. **Picking the key.** The four green corners map the camera picture onto a
   keyboard, correcting for the camera's angle. The key under the fingertip at the
   moment of the press is the one that gets typed.
4. **Typing it.** The key is sent to your focused app through `wtype`, `ydotool`,
   `xdotool` or `pynput`, depending on your system.

It doesn't use a microphone, it doesn't guess words, and it never records video. It
only sees what the camera sees, so it can't tell exactly when your finger touches the
table. That's why slow, clear taps work best.

---

## For developers

### Running from source

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python app.py                      # the control panel
python keyboard.py                 # the setup/typing window on its own
```

The hand model downloads itself the first time `app.py` starts. You can also fetch it
by hand into `models/hand_landmarker.task` from
[Google's model page](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task).

Useful options for `keyboard.py`:

| Option | Meaning |
|---|---|
| `--phone` | Use the Android phone camera |
| `--camera 1` | Use a different webcam |
| `--sensitivity 1.3` | Start with a different sensitivity (0.4 to 2.5) |
| `--exposure fast` | Short exposure for bright rooms |
| `--resolution 1280x720` | More detail, lower frame rate |
| `--rotation 180`, `--mirror`, `--no-mirror` | Override the saved camera orientation |
| `--no-start-sign` | Start typing immediately |
| `--trained-only` | Only type keys you've trained |
| `--record-landmarks FILE` | Record hand positions for debugging (no video) |

### Measuring accuracy

Record yourself typing a sentence you picked in advance, then replay the recording
against that sentence:

```bash
python keyboard.py --phone --record-landmarks logs/test_1.jsonl
python scripts/replay_keyboard.py --snapshots logs/test_1.jsonl \
  --profile config/phone_air_movement_model.json --expected "hey how are you"
```

The replay prints what the detector typed and counts correct, wrong, extra and
missed letters. It's a quick way to check whether a code change actually helped.
Recordings contain only hand coordinates: no images, no sound, and not the text you
typed.

### Tests

```bash
python -m unittest discover -s tests
```

The tests use made-up hand movements and fake cameras, so they check the logic, not
how well it works with real hands.

### Where things are

```text
app.py                   the control panel: hotkey, background typing
0keys, 0keys.bat         launchers
install.sh, install.ps1  installers
keyboard.py              the setup/typing window
movement_keyboard.py     main loop of the setup window
helper_engine.py         camera → hand tracker → tap detector, with no window
helper_output.py         sends keys to other apps
helper_hotkey.py         the Ctrl+Alt+K shortcut and the `0keys toggle` commands
helper_finger_press.py   tap detection
helper_gesture.py        the start sign
helper_motion.py         learning your own taps
helper_session.py        loading the layout, ignoring false hands
helper_keyboard*.py      keyboard layout and drawing
helper_vision.py         MediaPipe hand tracking
helper_webcam.py         webcam and phone camera
helper_phone.py          talks to the phone via scrcpy
air_keyboard.py          an older "point and pinch" mode, plus shared drawing code
scripts/                 the replay tool
config/                  your saved layout, training and settings
models/                  the hand-tracking model
tests/                   automated tests
```

There's also an older experimental mode where you point with your index finger
and pinch to click. Run it with `python air_keyboard.py --mode pinch` if you're
curious.

---

## License

No license has been chosen yet.

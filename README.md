# 0Keys

## Run the keyboard

Close the old window and restart with:

```bash
source .venv/bin/activate
python keyboard.py
```

For the overhead Android phone connected by USB:

```bash
python keyboard.py --phone
```

The phone now automatically uses `config/phone_air_keyboard_layout.json` and
`config/phone_air_movement_model.json`. Webcam files stay separate. Point the
rear lens at the table, keep both wrists visible, and drag the four green
corners to place the keyboard beneath your fingertips. Use F if the spacebar
is on the wrong side. Explicit `--calibration` and `--movement-profile` paths
still override the defaults.

You can orient the live camera picture from the keyboard window:

- Press `[` or click **ROTATE LEFT** for a 90-degree turn to the left.
- Press `]` or click **ROTATE RIGHT** for a 90-degree turn to the right.
- Press `M` or click **MIRROR** to reverse left and right.
- Press `F` or click **FLIP ROWS** to reverse just the keyboard rows.

Quarter turns switch between portrait and landscape. Rotation follows the
direction on screen even when mirroring is enabled. The status shows the
current orientation and mirror setting. Rotating or mirroring pauses typing
and cancels any press or training in progress. Adjust the green corners if
needed, then click **TYPE** or press `T` to resume; existing text stays intact.
The view is saved with the layout and restored at startup. Explicit
`--rotation`, `--mirror`, or `--no-mirror` options override the saved setting.
Finish an active landmark recording before changing the view, so a recording
always uses a single coordinate system.

### The camera must see your hands

Nothing can be tracked unless your hands are inside the camera picture. A laptop
webcam at a normal screen angle looks at your face, not at the table, so the
keyboard shows no hands at all. Choose one of these:

1. **Phone above the table, recommended.** Connect the phone over USB as in the
   section below and run `python keyboard.py --phone`. The phone looks down at
   your hands while the laptop screen stays at a comfortable angle. This is the
   only setup where whole hands, including wrists, stay in view while you work.
2. **A separate USB webcam** clipped above the table, pointing down.
3. **Tilt the laptop screen forward** until your hands fill the picture. This
   works for trying things out, but the screen is then hard to read.

The panel says `No hands in the camera picture` when the view is empty, and
after a couple of seconds it says where to point the camera. Fix the aim before
anything else; sensitivity and training cannot compensate for an empty picture.

This is the main launch file. It starts in TYPE MODE and shows the spaced keyboard
and your hands. The camera picture is enlarged to about 860 pixels high, keeping
its shape, so a portrait phone view is no longer a narrow strip. The buttons,
status and a tall text box sit in a panel beside the picture instead of covering
your hands. The panel shows CAPS LOCK ON or SHIFT ON in orange while either is
active. This is an on-screen overlay, not a physical projection. No microphone is used. The camera is the laptop webcam,
or the phone with `--phone`.

Aim the camera so that whole hands, palms included, are inside the picture.
Fingers alone can be tracked once tracking has started, but it only starts
again from a visible palm, so a view that cuts the hands off at the knuckles
keeps losing them. Rest briefly to establish a relaxed finger pose, then move
above the intended key, touch the table, and lift. Any finger can be used;
thumbs type only Space. The camera estimates a tap from the motion; it does not
measure actual table contact.
No pinch is needed. Hovering alone does not type. Drag the green corners to
position the keyboard. F reverses the row order.

Each fingertip outlines the key under it and shows its label and a small bar,
so you can see exactly where a tap will land before you make it. The bar fills as that
finger extends or bends relative to its palm; sideways movement alone does not
fill it. A full bar is a press-sized movement. The label turns orange while a
press is in progress and green during release. If presses are missed
while the bar only half fills, press + to raise sensitivity. If keys appear
while you rest, press - to lower it. `--sensitivity 1.3` sets it at startup.

The buttons work with the mouse as well as the keyboard:

- T always enters TYPE MODE. Pressing T again stays in typing, never training.
- L opens optional LEARN mode. Click a key to record five countdown-guided examples.
- N optionally records three no-key examples of resting and ordinary finger motion.
- Space pauses. T resumes typing and cancels unfinished training.
- C clears text. Q exits. R retries the current learning batch.
- - and + (or =) lower and raise press sensitivity between 0.4 and 2.5.
- [ and ] rotate the live picture left/right; M mirrors it. These also have buttons.

### Camera speed and picture quality

The keyboard used to run at about 8 FPS on this laptop because a one-frame
OpenCV buffer made the V4L2 driver skip every other frame. The camera is now
read on a background thread with normal buffering, always using the newest
frame, which doubles the rate to about 16 FPS with no loss of picture quality.

The camera's own exposure is the default, because it gives the cleanest
picture. This webcam has no gain control, so forcing a short exposure in a dim
room keeps the sensor gain high and produces heavy speckle noise: measured
frame-to-frame noise rose from 2.1 to 13.0, and hand tracking failed on it.
A short exposure is still worth using in a bright room, where it reaches 30 FPS
and removes motion blur from fast fingers. Press E to switch between them while
the keyboard runs and watch which one tracks your hands, or start with
`--exposure fast`. A fixed value such as `--exposure 50` (units of 0.1 ms) also
works. The panel shows the processed rate, the camera rate and the exposure.

Any exposure change is undone when the keyboard closes. If the program is
killed first, run `v4l2-ctl -c auto_exposure=3` to restore it. `--resolution
1280x720` captures more detail at a lower frame rate. In a very dark room the
panel asks for more light.

Training is optional in the default mode. It is no longer necessary to record
every key before anything can type. Blue outlines mark a learning target; yellow
briefly marks an emitted key. Existing examples are loaded automatically, and
typing never changes your saved training. `air_keyboard.py` also opens this mode.

## What detects a press

MediaPipe provides the pretrained hand tracker. Without personal key examples,
the app uses a rule-based detector, not imaginary AI training. It compares each
fingertip's extension and estimated depth relative to its finger base in a 3D
palm frame. Estimated depth has reduced weight because it can be noisy.
Sideways spreading is treated as aiming. Motion from a rigid translation or
rotation of the whole hand cancels in this frame.

The key is estimated from the most extended part of the finger movement,
including the resting pose for a lift-first tap. For example, extending from
A toward Q and lifting can select Q instead of locking A before the reach.
The target needs support from multiple samples; gaps and ambiguous boundaries
are rejected. This is a heuristic: a reach that resembles a tap can still type,
and a curled finger reaching to a nearer row may need to settle above that key
first. It does not infer intended letters from the words you want to type.

A sufficient rebound emits one key without requiring the finger to return
fully to its original position. A tap has to be quick: the finger must come back
within 0.9 seconds, and stay back for two camera frames. A finger that shifts
while hovering and stays in its new place just rests there and types nothing.
In an earlier recording, a hovering hand that had not tapped typed J and L:
its fingertip had shifted slightly over the same key and stayed marked as
moving, until a single frame of tracking wobble looked like a lift.

Hand tracking wobbles by different amounts with light, distance and hand pose,
so each finger's frame-to-frame wobble is measured while it rests. A press must
move clearly more than that wobble: at least 1.8 times it to start, 3 times it at
its deepest point, and 2 times it on the way back, and never less than the fixed
minimums. A still camera and steady light therefore let smaller taps register.
In a simulation with random wobble and pose shifts, false keys from shifting
fingers fell from about 19 to 2 per 40 shifts. At small and moderate wobble
taps were detected as often as before; with very large wobble about one tap in
five was missed, where the old detector typed wrong keys instead. A simulation
is not a measurement with real hands. Release must finish or settle before rearming,
so one motion cannot repeatedly type. Closely timed accompanying finger motion
still requires a dominant finger; staggered movements more than 140 ms apart
can be processed independently. Smaller accompanying movements may remain
displaced without blocking the pressing finger.

Finger motion is measured relative to palm width with the camera aspect ratio
preserved, so dragging
the keyboard corners into a wide or squashed shape does not change how large a
press must be. The resting place follows slow drift over about half a second,
regardless of frame rate. Hands are identified by tracker and screen position.
MediaPipe's own left and right labels can flip when it sees the backs of the
hands with the palms out of view, and both hands used to be ignored whenever
they received the same label. Saved no-key recordings are no longer used by
basic detection, because they could reject small real presses; they still take
part when a key has its own trained examples.

Labels beside each fingertip show its current key or `gap`. The app says when no
hands are in the picture, and when palms are outside it, which prevents hand
detection from recovering after it is lost.
Rest briefly at first, then press and release. A press still needs at least two
moving camera frames to distinguish it from a single-frame tracking glitch.

Compatible personal examples add a local whole-hand movement classifier. Three
key examples and three no-key examples for the same hand are needed for a personal
key match. In `--trained-only` mode, saved no-key recordings can reject movements
even without any key examples. A trained key's rejection is not overridden by basic detection. Use
`python keyboard.py --trained-only` to disable basic detection completely.

This remains experimental, not a physical-keyboard replacement. It cannot directly measure pressure or reliable finger
height from one webcam, and similar-looking intentional and accidental movements
can still be confused. There is no guarantee that all unfamiliar motions will be
rejected. At low FPS, use slow presses with a short rest between them. This
version waits for release before typing. Some staggered presses can overlap,
but simultaneous fingers and fast touch typing remain unreliable. Measure
missed keys, wrong keys and accidental output on fresh
attempts before relying on it. Do not judge accuracy from replaying training data.

The webcam now requests MJPEG at 640 by 360, which the laptop camera supports,
instead of the unsupported 640 by 480 request. Requesting 30 FPS does not guarantee
30 FPS in practice; watch the measured frame rate. Lighting and tracking still matter.

Webcam movement examples are stored in `config/air_movement_model.json`; phone
examples use `config/phone_air_movement_model.json`. They contain
landmark features, not video or audio. Existing profiles are backed up before
replacement. Camera or orientation changes prevent incompatible examples from
loading. If only corners or row direction change, no-key features can be mapped
to the new coordinates in memory, but old key labels are not reused. The old pinch
and tap profiles are not erased or incorrectly treated as movement recordings.
Only the app's text box receives output, not other applications.

For a reproducible local bug report, run `python keyboard.py --record-landmarks
logs/typing_attempt.jsonl` with a new filename. This records landmark coordinates
and camera context only, not images, sound or typed text. It does not train the
model. Replay with `python scripts/replay_keyboard.py --snapshots
logs/typing_attempt.jsonl`. A screen recording containing keyboard overlays may
not work for fresh hand detection, so zero detected hands is not a typing result.
Position the keyboard before recording diagnostics; layout editing is locked
during the recording so its saved coordinates stay meaningful.

For raw reference video, extraction preserves the original image proportions:

```bash
python scripts/replay_keyboard.py --extract --video /path/to/typing.mp4 \
  --snapshots logs/reference.jsonl --rotation 180 --mirror
```

Choose rotation/mirroring to match the view being analyzed; `--crop X Y WIDTH
HEIGHT` is optional. Use `--calibration` for a layout aligned with that video.
Extracted video gets its own camera identity so live-camera training is not
silently reused. Replay uses the recorded aspect ratio (legacy logs without it
assume 16:9). Counts of detected events are not an accuracy score without
labeled intended keys and tap times.

## Optional older mode: index pointing and pinch

Use `python air_keyboard.py --mode pinch` to run the previous interaction below.
The V accuracy check described in this section belongs to pinch mode only.

Run the older pinch mode with:

```bash
source .venv/bin/activate
python air_keyboard.py --mode pinch
```

It uses the laptop webcam, not the phone or microphone. Position the webcam to
see your hands from above. The window overlays a spaced, staggered QWERTY keyboard
on the video. Number, letter, and space rows have separate keycaps with inactive
gaps, normal row offsets, and a 6.25-unit spacebar. This is a camera overlay,
not a projection onto the actual table or a measurement in centimeters.

Keep your hands above the table. Point at a key using an index fingertip, hold
the aim briefly until the hand status says `ready`, then bring that thumb and
index together for a short pinch. Separate them again before the next key.
Either hand can be used. This is a two-pointer air keyboard, not ten-finger
touch typing or a finger-pressure detector. Hovering alone never types.

The key is locked from the stable position before the pinch, so the movement
used to click does not immediately select a neighboring key. Holding a pinch
does not repeat. Tracking loss, long frame gaps, uncertain predictions, inactive
gaps, and two clicks in the same frame are rejected. Ordinary movements that
closely imitate the deliberate gesture can still cause an accidental click.

Drag the four green corners with the mouse to resize the keyboard. The corners
must stay in order and form a convex quadrilateral. F reverses the row direction
if the spacebar is on the wrong side. The webcam preview is mirrored by default;
`--no-mirror` disables this, and `--rotation 180` turns the image upside down.
Layout and camera changes invalidate personal position training.

The camera uses MediaPipe's pretrained neural hand tracker. A separate local
position classifier ranks keys using their geometry and learns bounded position
adjustments from confirmed examples. Before training, this classifier uses only
geometry. Its relative confidence is not a measured accuracy percentage. It
does not use a language model to invent characters or fill the gaps between keys.

To personalize it:

1. Press T, then use the mouse to select a key on the overlay.
2. Aim at that key with an index finger and pinch once.
3. Press Y to confirm the example or N to discard it.
4. Collect five confirmed examples per key and per hand you plan to use.
5. Press T again to return to typing.

Learning starts for a key and hand after three confirmed examples. Examples
outside the selected keycap are rejected. Automatic typing never trains the
model. Layout is stored in `config/air_keyboard_layout.json` and examples in
`config/air_key_model.json`, separately from all earlier tap and phone profiles.
Existing files are backed up before their first replacement in a session.

Space pauses, C clears text, and Q exits. Virtual Shift, Caps, Backspace, Enter,
Tab, punctuation, and Space edit the text inside this window. Dim Ctrl, Alt, Win,
and Menu keys are inactive because this mode does not send system shortcuts or
type into other applications. No microphone, speech commands, or tap sound is
used in this mode.

Press V for an approximately 84-second guided check with sixteen prompted clicks
and idle checks. Missed prompts, wrong keys, extra keys, and accidental activations
are counted. Q aborts and marks the report incomplete. Results are printed and
saved in `logs/air_accuracy_*.json`; no video or audio is saved. Tests do not change
the learned examples. Repeat across sessions before judging live reliability.

The older programs below remain available for comparison; they have not been
deleted or silently switched to air gestures.

0Keys is an invisible keyboard that uses a laptop webcam and microphone. The
user types by tapping their fingers on a table without using a physical
keyboard.

The microphone detects the exact moment of a table tap. The webcam tracks both
hands and estimates which fingertip moved toward the table. A calibrated QWERTY
layout will then convert the fingertip position into a character.

The current accuracy work starts with the four-key press keyboard below. The
full QWERTY programs remain available, but their everyday typing accuracy has
not been established.

## Start here: four-key press keyboard

For a phone camera, use the USB setup below first. Laptop and phone calibration
files are kept separate.

After installation and the hand model download, run:

```bash
python press_keyboard.py
```

Setup automatically selects a single visible hand using the corrected physical hand label.
For Left, pinky is A, ring is S, middle is D, and index is F. For Right, index is
J, middle is K, ring is L, and pinky is semicolon.
Keys are assigned to fingers, so there is no QWERTY position classifier or
language guessing. It displays these four characters in its own window. It does
not yet type into other applications.

The first run guides you through five poses:

1. Rest all four fingers naturally on the table and say "record".
2. Pause speaking, wait for the 1.5-second countdown, then hold still while eight camera frames are recorded.
3. For each following prompt, lift only the named finger, keeping your palm at
   its home position. Say "record" and hold the lifted pose to record it.
4. After the four lifted poses, return your fingers to the displayed markers.

The camera must see the fingers clearly. Setup rejects a lift that is too small
to distinguish or a pose recorded while your hand moves. You can press Space
to retry a pose, or say "record" again. R restarts setup. The helper learns the visible direction
of each lift rather than assuming that down in the image means table contact.
This is a motion reference, not a pressure measurement.

The laptop webcam and phone previews are mirrored by default.
The shared tracker corrects Left and Right
after mirrored inference, so the labels refer to your physical hands. Fingertip
coordinates stay in preview coordinates, keeping the overlay aligned. With
`--no-mirror`, the tracker leaves the model's hand labels unchanged. This applies
to every camera program. Profiles created before this correction require fresh
hand training; the previous file is backed up when the replacement is saved.

If recording waits for the wrong hand, check `Camera sees` and `Selected` in the
bottom panel. Press H to switch the selected side and restart setup, then Space
to record. Keep your wrist and all fingers inside the camera image. Waiting for
a missing or mismatched hand stops after ten seconds; Space retries. At eight
tracking FPS, eight frames take roughly a second after the preparation countdown.
Use `--hand left` or `--hand right` to require a specific camera label instead of
automatic selection. H also switches hands when a saved profile is active; it
selects the other physical hand rather than changing the label convention.

To type, lift a finger clearly, then tap it back at its marked resting position.
Start slowly: the lift must be seen in at least two processed frames. A stable
rest comes first, and each new press requires a new lift. Staying down cannot
repeat a key. A matched microphone onset is required, so a silent return will
be missed. Sounds while fingers rest or hover do not activate a key. Two fingers
returning together are rejected. Tracking loss and movement of the palm away
from home cancel pending presses.

The window shows each finger's current state and the processed camera frame
rate. Press physical Space to pause or resume, C to clear the text, R to redo
setup, and Q to quit. Control key sounds are suppressed briefly. Setup is saved
to `config/press_profile.json`; replacing it first creates a timestamped backup.
If the camera or your hand position changes, repeat setup with R.

For the right hand instead:

```bash
python press_keyboard.py --hand right
```

That maps index, middle, ring, and pinky to J, K, L, and semicolon. It records a
separate setup when the saved hand does not match. The previous profile is
backed up before being replaced.

## USB phone camera on Arch Linux

The phone backend uses the rear Android camera through scrcpy over USB. It needs
Android 12 or newer, USB debugging authorization, and these packages:

```bash
sudo pacman -S --needed android-tools scrcpy
source .venv/bin/activate
python -m pip install -r requirements.txt
python phone_camera.py --check
python phone_camera.py
```

On Xiaomi, enable Developer options by tapping MIUI version seven times under
About phone. Enable USB debugging under Additional settings, Developer options.
Unlock the phone and accept the debugging prompt for your own laptop. If it does
not appear, reconnect the data cable. Do not enable wireless debugging or unlock
the bootloader. USB debugging can be turned off when you finish.

The preview uses the rear camera at 640 by 480 and requests 30 FPS. Support the
phone securely, looking diagonally down at the table with both hands and wrists
visible. Keep the laptop open. Close the preview with Q before starting typing:

```bash
python press_keyboard.py --phone --hand both --hover-start
```

For the current horizontal Redmi placement, both commands default to a 180-degree
rotation with mirroring on, as requested for the facing-camera view. Use
`--no-mirror` if you prefer the unreflected view. The orientation is corrected before hand tracking,
and labels refer to your physical hands, not their positions on screen. The phone
preview shows those labels so you can check them before recording setup.
If you turn the phone the other way, use `--rotation 0` in both commands. Quarter
turns, `--mirror`, and `--no-mirror` are also supported. Always use the same options
for preview and typing. Avoid another camera app applying an additional flip.
Moving or rotating the phone requires fresh
calibration; R restarts setup. Say "record" for each requested pose as before.
Profiles saved with different rotation or mirroring are not reused. They remain
on disk until replacement setup is saved, at which point they are backed up.

Phone setup saves to `config/phone_press_left.json` and
`config/phone_press_right.json`. Camera identity and rotation are checked before
reusing a profile, and replacements are backed up. The eight key markers are
based on the resting positions recorded during setup. They do not move after
every tap, and this mode is not a full QWERTY layout yet.

`--hover-start` allows each finger to become ready from a stable raised pose,
without first touching down. One finger can tap while others stay raised. A tap
still needs a return to its calibrated marker and matching microphone sound.
Holding a finger down does not repeat keys. Simultaneous ambiguous contacts are
rejected. This optional behavior is not pressure sensing or silent air typing.

The project reads scrcpy's video packets directly and decodes them in a background
thread with PyAV. No virtual-camera kernel module is needed. Only the newest
decoded frame is used by tracking. Phone audio and remote input control are
disabled; the laptop microphone still supplies tap audio and setup voice control.
Camera frames are not saved to disk. Closing the app removes its temporary phone
server file and its own ADB forwarding rule.

To measure delivery without opening a window:

```bash
python phone_camera.py --probe
```

This reads sixty frames and reports received video FPS, not processed hand FPS
or true end-to-end latency. Phone timestamps preserve frame spacing and are
anchored to laptop arrival time; the remaining audio-to-video delay still needs
measurement. `--camera-offset-ms` adjusts audio matching. A stable camera feed
does not establish tapping accuracy. Press T in the keyboard for the live test.
This camera provides no direct measurement of finger height above the table.

The implementation follows the upstream [scrcpy camera support](https://github.com/Genymobile/scrcpy/blob/master/doc/camera.md)
and [server protocol](https://github.com/Genymobile/scrcpy/blob/master/doc/develop.md).
Client and server versions must match, so keep the Arch scrcpy package intact.

## Hands-free setup with your voice

During pose setup, say **ready**, **record**, or **start recording**, then pause speaking and keep the requested
pose still. The app waits 1.5 seconds before collecting eight camera frames.
Say it once per pose. You can keep both hands on the table throughout setup.
Voice control starts only after the initial quiet microphone calibration and
is disabled during capture, countdowns, and typing. Physical Space remains a
fallback, with its original one-second preparation time.

Voice recognition uses the same microphone stream as tap detection, with a
background worker. It runs locally with [Vosk](https://alphacephei.com/vosk/install)
and the free Apache-licensed [small English model](https://alphacephei.com/vosk/models).
No account, paid API, or uploaded audio is needed. The app does not save speech
recordings or transcripts. It logs only that a setup recording was requested.

Install the dependency and download the model once if not already present:

```bash
python -m pip install -r requirements.txt
mkdir -p models
curl -L --fail -o models/vosk-small-en.zip https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip -n models/vosk-small-en.zip -d models
python press_keyboard.py --hand both
```

The download is about 40 MB. Runtime uses the extracted local directory and
does not download models automatically. Use `--voice-model PATH` for another
compatible model, or `--no-voice` for Space-only setup. If loading fails, the
terminal explains why and the app keeps Space available.

Say "ready" clearly toward the laptop microphone, then leave a short silence so
recognition can finish. Partial phrases and low-confidence results do not start
recording. Recognition can still miss speech or mishear background voices, so
watch for the countdown and use a quiet room. A rejected pose still needs to be
retried. Automated tests check the control flow, not recognition of your accent
or microphone. This command does not pause typing or start an accuracy test.

The setup panel now displays microphone level in dBFS and voice status, including
recognition misses and audio gaps. If the level stays near -120 while you speak,
check input selection and mute settings. A quiet input warning alone does not
prove the microphone is broken. The phone supplies video only, not voice audio.
Use `--microphone pipewire` on Arch to select the PipeWire input route explicitly.
If `Camera sees: none`, raise the phone above table height and tilt it down until
fingers and wrists are visible. Reflections and fingers hiding one another can
prevent tracking; flipping the image alone cannot fix that.

## Next step: both hands and eight keys

To use A S D F with your left hand and J K L ; with your right hand, run:

```bash
source .venv/bin/activate
python press_keyboard.py --hand both
```

A compatible left-hand setup is reused from `config/press_profile.json`. The
app asks for the five right-hand poses and saves them separately in
`config/press_right_profile.json`. If neither profile is available, it guides
you through the left hand first, then the right. Keep the camera still and put
each hand where it will rest during typing. Say "record" to record
each requested pose. H is disabled in this mode because setup selects each hand
in turn.

Once setup finishes, the window shows eight markers and each finger's state.
Rest both hands at their markers, then lift and tap one finger at a time.
Start with F, J, D, K, S, L, A, and semicolon, checking the displayed text.
If fingers on both hands match the same sound, the app rejects that tap instead
of choosing a hand or emitting two keys. This mode still requires an audible
tap and visible motion. It has no thumb spacebar, full QWERTY, or output into
other applications yet.

Press T for forty trials, five per key, plus idle checks. This takes about three
minutes. Press R to redo both hand setups, or start with `--hand both --calibrate`.
Existing files are backed up before replacements are saved. If only one hand
needs setup on startup, the other compatible saved profile is left unchanged.

The automated checks cover separate profiles, alternating hands, and ambiguous
cross-hand taps with simulated sensors. They do not establish live accuracy.
At the roughly eight tracking FPS seen in the demo, start slowly enough for
each lift to appear in at least two frames before returning to the table.

## Measure the press keyboard result

Press T in the press keyboard window. The single-hand guided check takes about
100 seconds; both hands take about three minutes:

- Rest your fingers for ten seconds without typing.
- Follow twenty prompts for one hand or forty for both, five per key. Each has a preparation period followed
  by a `Tap once` prompt. Make one deliberate tap in each trial.
- Rest and gently slide your hand during the final ten seconds without typing.

Each completed trial counts automatically. Not tapping during a requested trial
counts as a miss, so follow every prompt. Wrong keys and duplicate outputs count
as failures, even if the expected letter also appears. Keys during rest or
preparation count as accidental activations. Q aborts the test and marks the
report incomplete. Tests never adjust the saved motion profile.

The result is printed in the terminal, shown in the window, and saved in `logs`.
The log includes each expected and emitted key, camera frame rate, missing hand
frames, dropped audio blocks, and measured output delay. During the test it also
records hand landmark coordinates, finger states, and detected audio timestamps
for diagnosing errors. It does not save camera images or raw microphone audio.
The test sequence and landmarks are local data, so inspect the logs before
sharing them.

Try multiple sessions before trusting an accuracy percentage. Twenty prompted
taps and twenty seconds of idle testing are a small initial check, not proof of
general typing reliability. A real-world result requires low missed-tap counts
as well as few wrong keys and accidental activations. The automated tests use
synthetic hand motion and mocked sensors; live accuracy must be measured here.

This press detector is a conservative rule-based baseline. It does not detect
silent pressure changes while a finger remains stationary on the table. A
coincident unrelated noise and a valid lift-and-return movement can still be
mistaken for a tap. The recorded test sequences will help identify where a
future motion classifier adds value.



## Requirements

- Linux, Windows, or macOS
- Python 3.12 recommended
- Built in or USB webcam
- Built in or USB microphone
- A table surface visible to the webcam
- Good lighting

My current development laptop uses an AMD Ryzen 5 7000 series

## Installation

Clone or open the project folder, then create a Python environment:

```bash
pyenv local 3.12.11
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows activation command:

```powershell
.venv\Scripts\activate
```

Download the free MediaPipe hand landmark model:

```bash
mkdir -p models
curl -L --fail \
  -o models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

The model is stored locally 

## Tap detection

List available microphones:

```bash
python tap_detection.py --list-devices
```

Run with the default microphone:

```bash
python tap_detection.py
```

Run with a selected microphone:

```bash
python tap_detection.py --device 3
```

Remain quiet during the initial calibration. After the ready message appears,
tap the table normally.

If taps are missed, lower the threshold:

```bash
python tap_detection.py --threshold-db 10
```

If normal room noise produces false taps, raise the threshold:

```bash
python tap_detection.py --threshold-db 18
```

## Hand tracking

Run the camera tracker:

```bash
python hand_tracking.py
```

Try another camera index if camera zero is unavailable:

```bash
python hand_tracking.py --camera 1
```


Press `Q` or `Escape` to close the camera window.

## Keyboard calibration

Run the calibration program:

```bash
python keyboard_calibration.py
```

Click four table positions in this order:

1. Upper left in the camera image
2. Upper right in the camera image
3. Lower right in the camera image
4. Lower left in the camera image

The selected shape becomes the virtual keyboard area. The calibration is saved
to `config/keyboard_calibration.json`. Move your fingertips through the area to
check that the expected keys are highlighted. Press `R` to recalibrate and
press `Q` or `Escape` to exit.

The yellow corner points can be dragged after calibration. Drag them outward to
make the keyboard cover a larger table area. A normal keyboard area should be
about three times wider than it is deep on the physical table.

If the space row and number row appear on opposite sides, press `F`. This flips
the keyboard rows while preserving the selected boundary. The orientation is
saved automatically.

## Invisible keyboard

After microphone testing and keyboard calibration are complete, run the combined
application:

```bash
python invisible_keyboard.py
```

The microphone first measures the room noise. When the status changes to
`Ready to type`, tap the virtual keys while keeping both hands visible. Accepted
keys appear in the text panel below the camera image.

Use a selected microphone or tune the synchronization if needed:

```bash
python invisible_keyboard.py --microphone 3 --camera-offset-ms 70
```

Press `C` to clear the text. Press `Q` or `Escape` to exit.

Camera windows open at a larger size and can be resized. The typing and personal
training windows request 1100 by 900 pixels. Hyprland may adjust this to fit your
current tile. Camera capture resolution stays the same, so enlarging the window
does not increase hand tracking work.

The combined application uses balanced sensitivity by default. For very quiet
ring or pinky taps, use:

```bash
python invisible_keyboard.py --sensitivity very-high
```

If background sounds or small hand movements create unwanted keys, use:

```bash
python invisible_keyboard.py --sensitivity normal
```

If the status says `Typed Back`, backspace was accepted. It removes the previous
character, so the text stays empty when there is nothing to remove.

## Personal key training

The camera mapping alone assumes that everyone reaches each key in the same
way. Personal training records where your fingertip really appears, which hand
and finger you use, and how that finger moves for every prompted key.

Train the letters and main action keys first:

```bash
python user_key_calibration.py
```

Wait for microphone calibration and the `TAP` heading, then tap the highlighted
key once using your usual finger. The trainer freezes the camera image near the
tap time and marks the recorded contact position in red. Check that it selected
the finger you used and that the position is near the target. Press physical
`Space` to save it, or `R` to discard it and retry. Extra taps during this review
are ignored. Wait for `GET READY` to change back to `TAP` before tapping again.

The normal session collects five confirmed examples on the same key before
moving on. It covers every letter, Space, Backspace, and Enter, for 145 samples.
Press `U` to remove the previous saved sample, `N` to skip a prompt, and `Q` to
leave. Only confirmed samples are saved. Running the command again resumes the
unfinished profile.

Start with a smaller group to check your setup:

```bash
python user_key_calibration.py --keys a s d f
```

If the highlighted target does not match where you intend to tap, adjust the
table boundary in `keyboard_calibration.py` first. Training rejects samples far
from the target. Use a consistent finger for each key, and check the detected
finger name before confirming.

For every key in the full layout, run:

```bash
python user_key_calibration.py --key-set all
```

To start a fresh profile with a backup of the existing file, run:

```bash
python user_key_calibration.py --fresh
```

Training data is stored locally in `config/user_keyboard_profile.json`. It is
loaded automatically by `invisible_keyboard.py`. Profiles from the old automatic
trainer are ignored during typing. Starting the new trainer automatically backs
up the old file beside it and starts a new profile. Changing the saved boundary,
row orientation, camera index, or timing offset also requires retraining. Keep
the laptop and your seating position steady between training and typing.

After a tap with usable camera evidence, the typing window shows up to three
nearby key choices. Uncertain predictions leave the text unchanged.
If the first result is wrong, immediately press `1`, `2`, or `3` on the laptop
keyboard to select the correct displayed choice. This replaces the last typed
key, or inserts your choice if that tap was uncertain. Choices expire after five
seconds or the next prediction. Corrections are added to a compatible personal
profile, once per tap. A single correction does not establish a learned key
position: at least three nearby examples using the same hand and finger are
needed. Old incompatible profiles are never overwritten by corrections.

The classifier combines fingertip position, position relative to the wrist,
hand, finger, and tap motion. It uses medians to limit the effect of outliers,
restricts predictions to nearby keys, and checks the gap between competing
choices before typing. Audio strength is stored for diagnostics, but is not a
learned key feature. Internal prediction weights are relative model scores,
not measured accuracy percentages. No paid API or extra hardware is required.

## Check accuracy after training

Test the small group with fresh taps:

```bash
python user_key_calibration.py --validate --keys a s d f
```

For the full common key set, run:

```bash
python user_key_calibration.py --validate
```

Validation prompts each selected key three times, in separate rounds. It uses
the typing classifier without telling it which key is expected. Physical
`Space` counts the attempt, including wrong predictions. If a tap makes no
response, press `M` to count it as missed. Use `R` only for an accidental sound,
not to remove a typing error. `N` skips an untested prompt. These exclusions are
reported separately.

Validation never adds samples or changes the training profile. Press `Q` to
print the results and the incorrect key pairs. A report is saved in `logs` on
exit with correct, rejected, skipped, discarded, and remaining counts. Accuracy
includes rejected attempts in its denominator. Taps the microphone misses must
be marked with `M` to be counted. This is a guided key check, not a measurement
of continuous typing speed or accuracy.

For keys that still fail, collect additional confirmed samples:

```bash
python user_key_calibration.py --keys a s --samples-per-key 10
```

That command raises the target to ten total examples for each selected key.
Then validate again with fresh taps. Automated tests check the software logic;
your live validation measures performance with your camera and table.

Each run also writes a local JSON Lines record in `logs`. It contains rejected
taps, predicted alternatives, confidence values, and corrections. Disable this
with:

```bash
python invisible_keyboard.py --no-session-log
```

## Tests

Run all automated tests:

```bash
python -m unittest discover -s tests -v
```

The tests currently verify:

- A synthetic table impact creates one tap event
- Steady background noise does not create tap events
- Hand identifiers survive detection order changes
- Camera snapshots can be matched by timestamp
- Fingertip velocity is calculated correctly
- Invalid timestamp order is rejected
- Keyboard rows fill the calibrated area
- Camera coordinates map to the correct QWERTY keys
- Calibration data can be saved and loaded
- Audio taps select the fingertip with matching down and up motion
- Ambiguous fingertip movement is rejected
- Letters, modifiers, and special keys update the text correctly
- Personal training profiles can be saved and resumed
- Learned positions can override a neighboring geometric key
- Finger identity helps separate keys with overlapping positions
- Prediction confidence and session logs use valid data
- Queued taps cannot advance training without confirmation
- Retrying and undoing do not leave mislabeled samples
- Distant mislabeled samples cannot take over unrelated keys
- Uncertain predictions leave typed text unchanged
- Validation leaves the profile untouched and records wrong predictions

## Project structure

```text
0Keys
|-- tap_detection.py
|-- press_keyboard.py
|-- helper_press.py
|-- helper_audio.py
|-- hand_tracking.py
|-- helper_vision.py
|-- keyboard_calibration.py
|-- helper_keyboard.py
|-- invisible_keyboard.py
|-- helper_fusion.py
|-- user_key_calibration.py
|-- helper_classifier.py
|-- helper_training.py
|-- session_logger.py
|-- requirements.txt
|-- models
|   `-- hand_landmarker.task
|-- tests
|   |-- test_helper_audio.py
|   |-- test_helper_vision.py
|   |-- test_helper_keyboard.py
|   |-- test_helper_fusion.py
|   |-- test_helper_classifier.py
|   |-- test_personal_calibration.py
|   |-- test_keyboard_workflow.py
|   |-- test_helper_press.py
|   |-- test_press_workflow.py
|   `-- test_session_logger.py
`-- Theroy_practice
    |-- M1-chirp.py
    |-- M2-FFT.py
    |-- M3-STFT.py
    `-- M4-Filtering.py
```

The runnable programs test individual sensors, calibrate the table and personal
typing style, and run the complete invisible keyboard. The helper files contain
reusable processing logic.

## Signal processing concepts

The microphone uses a sample rate of 48 kHz. This gives a Nyquist frequency of
24 kHz. Audio is processed in blocks of 256 samples, which represents about
5.33 milliseconds per block.

The audio helper keeps frequencies from approximately 600 Hz to 8 kHz. It
measures peak level, average energy, crest level, sudden energy rise, and the
current background noise level. A tap must satisfy several conditions before it
is accepted.

The vision helper tracks normalized landmark coordinates. Camera `x` and `y`
values normally range from zero to one. Fingertip motion is estimated by
comparing positions from nearby timestamped frames.

## Development plan

1. Build and validate real time audio tap detection.
2. Track both hands and store timestamped fingertip motion.
3. Calibrate a virtual QWERTY area and map positions to keys.
4. Synchronize audio and camera events and display typed text.
5. Tune accuracy, add personal calibration, and prepare the final demo.

## Practical limitations

A normal webcam cannot measure exact distance from a fingertip to the table.
The final system will estimate contact by combining microphone onset timing with
fingertip direction, speed, position, and relative depth.

Best results require a stable laptop position, visible hands, good lighting,
and a short calibration when the program starts. Simultaneous taps are harder
to separate with one microphone, so modifier keys may use toggle behavior.

## License

No project license has been selected yet.

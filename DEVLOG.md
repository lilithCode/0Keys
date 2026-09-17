# Development log

## Vision tracking milestone

Today I completed the vision foundation for 0Keys. I added real time two hand
tracking with MediaPipe, tracked all 21 landmarks on each hand, highlighted the
five fingertips, kept hand IDs stable, estimated fingertip motion, and stored
camera frames with timestamps for later audio matching. I also reorganized the
project into clear helper and runnable modules, added automated tests, and
documented the complete setup in the README.

## Keyboard calibration milestone

Today I turned the camera view into a usable virtual keyboard area. I built a
four corner calibration flow, used a homography to correct camera perspective,
defined a complete QWERTY layout, and added a live overlay that highlights keys
under the fingertips. The calibration is saved locally, loads automatically,
and is covered by tests for coordinate mapping, layout geometry, and file
storage.

## Personal accuracy milestone

Today I improved 0Keys so it learns how I personally type instead of relying
only on fixed keyboard boxes. I added guided training for each key, a local
classifier that combines fingertip position, wrist relative position, hand,
finger, and motion. It can compare several moving fingers and shows three likely
key choices during typing. When a result is wrong I can correct it with 1, 2,
or 3, and the system saves that example so future predictions improve. I also
added local session logs and automated tests for profile storage, prediction,
training resume, and correction data.

## Training reliability fixes

Today I found why personal training was making 0Keys guess strange keys. In my
saved profile, 74 of 87 samples were far from their labeled key. I changed the
trainer to review each tap, show the recorded finger position, and wait for my
confirmation before saving. It now collects five examples per key and blocks
extra taps during review. I also added checks for distant samples, more robust
position estimates, and rejection of uncertain predictions. A separate
validation mode measures fresh attempts without changing training data. The
software checks pass, but I still need to retrain and measure accuracy on my
actual setup.

## Four-key press detection

Today I built a focused four-key version of 0Keys to work on intentional presses.
It records my resting hand and the lifted pose of each finger, then requires a
visible lift and return with matching tap audio before typing once. Resting,
holding a finger down, losing tracking, and moving the hand away from home do
not activate keys in the tested scenarios. I added a timed accuracy check that
counts missed taps automatically and checks for accidental keys while resting.
It also logs motion, frame rate, audio gaps, and output delay. Synthetic and
simulated app tests verify the logic; the next step is measuring my real taps.

## Mirrored hand labels

Today I fixed the reversed hand names caused by using the mirrored camera image
without correcting the model's labels. I checked the installed model with a
reference image and corrected handedness in the shared tracker. The preview and
fingertip coordinates still line up, while hand selection and key assignments now
use physical Left and Right. Older hand profiles require fresh setup so reversed
labels are not reused.

## Both hands on the home row

Today I extended the deliberate tap keyboard to eight keys using both hands.
It keeps separate hand setups and reuses my existing left setup while I record
the right. Each finger still needs a visible lift and return with matching sound.
When both hands could explain the same tap, it rejects the event instead of
guessing. I added a forty-trial check and automated tests for setup and cross-hand
ambiguity. Next I need to run that check with my own taps to measure the result.

## Hands-free pose recording

Today I added a spoken record command so I can keep both hands in position
during setup. It uses a free offline speech model through the existing microphone
and gives me a short countdown before capturing each pose. Voice commands stop
during recording and typing, and delayed commands are discarded. Space still
works as a fallback. I added tests for the hands-free setup flow and still need
to check how well it hears my voice on the actual laptop.

## USB phone camera and hovering fingers

Today I added a rear phone camera input over USB so I can keep my laptop open
and place the camera where it can see the table. The phone has its own preview,
rotation controls, and separate hand setup files. I also added an optional mode
that lets fingers start raised while another finger taps. Sound confirmation and
ambiguous tap rejection remain active. Automated tests cover the new logic;
the new camera angle still needs a live typing accuracy check.

I connected the Redmi Note 10 Pro running Android 13 and verified sixty live
rear-camera frames at about 30 FPS over USB. I also fixed compatibility with
scrcpy 4's changed packet format. The initial view was dark, so positioning the
phone with a clear view of the table comes before recording the new hand poses.

## Phone orientation correction

Today I corrected the horizontal phone view by rotating it 180 degrees and
turning off the automatic mirror. Preview and typing now share the same image
transform before tracking. The preview also shows physical hand labels so I can
check Left and Right before setup. Old orientation profiles are not reused.

## Mirrored phone view and voice feedback

Today I switched the upright phone view back to a horizontal mirror as requested,
while keeping physical hand labels separate from screen positions. I added ready
and start recording as voice commands and made microphone level, recognition
misses, and audio gaps visible in the setup panel. Setup now also explains when
no hands are detected instead of only saying it is ready.

## Webcam air keyboard

Today I added a laptop webcam mode with separate, staggered QWERTY keycaps and
space between rows. I can aim with an index finger and pinch to click without
touching the table or using tap audio. The click locks the aimed key before the
pinch moves my fingers. MediaPipe tracks my hands, and a local position model can
learn from examples I explicitly confirm. I also added a guided click accuracy
check. This is a two-pointer air keyboard, so it still needs live testing and
does not yet behave like ten-finger physical typing.

## Guided movement training

Today I replaced the default pinch training with a countdown and five natural
press examples per key. I can use my pinky for A without freezing the other
fingers. The local model compares the whole hand movement as one pattern, and I
can record no-key movements to teach it what to ignore. Examples save without
confirmation buttons, and I can retry a batch. Automated checks cover coupled
finger motion and rejected movements, but live typing accuracy still needs work,
especially with the laptop camera running at about eight frames per second.

## One launch file and clear typing controls

Today I added keyboard.py as the main launcher and made T mean Type every time,
with L and a separate button for optional learning. I found that my movement
profile had 45 no-key examples and no key presses, which explained why nothing
could type. Basic press detection now works without per-key training and uses
compatible no-key recordings to help reject unwanted motion. I also added a clear
text box below the keys and switched to a supported webcam resolution. A short
capture check measured about 16 FPS, but real typing accuracy is still unverified.

## Fixing the whole-hand release blocker

The screen recording showed repeated waiting for the whole hand to return, even
while an individual finger was pressing. I changed the default detector to track
release per finger, compensate for palm motion, and tolerate smaller accompanying
finger movements. Fingertips now show their hovered keys, and there is an optional
landmark log for reproducible testing without storing camera images. Regression
tests cover the old stalled-hand case. Re-tracking the heavily overlaid screen
recording found no hands, so it cannot establish live typing accuracy.

## Why the recording typed nothing, and a 30 FPS camera

Today I recovered the hand landmarks from the screen recording itself by reading
the coloured skeleton the app drew on each of its 473 frames. Replaying them
through the detector that was running in the video produced no presses at all,
matching the empty text box: it waited for every hand point to return to the
starting pose, which never happened. The newer finger detector did fire, but it
had problems of its own. It ignored both hands whenever MediaPipe gave them the
same left or right label, and that label flipped on the same hand during the
recording. It measured motion in keyboard units, which the squashed overlay
made about a quarter less sensitive vertically, the direction a press moves in
this view. Resting thumbs produced Space presses, and saved no-key recordings
could veto small real presses once enough frames were available.

The camera measured 8.3 FPS, exactly as in the video. A one-frame buffer made
the driver skip every other frame, and automatic exposure halved the rate again.
A background reader with normal buffering and an adjusted short exposure now
gives 30 processed FPS on this laptop, with 2.7 ms frames instead of long blurry
ones. The detector now measures motion in camera pixels, keeps hands apart by
tracker and position, lets thumbs type only Space, and follows drift by time
rather than frame count. Fingertip bars and live - and + sensitivity make
missed or extra presses visible and adjustable. New regression tests fail on
the previous detector and pass now. Live typing accuracy still needs measuring
with real presses; the recording's hands had their wrists above the picture,
which also made tracking unsteady.

## Camera aim, and picture quality over frame rate

A screenshot showed the keyboard tracking nothing, with a very noisy picture.
Two separate mistakes. First, the short exposure I had made the default: this
webcam exposes no gain control, so a 3 ms frame in a dim room keeps the high
low-light gain and amplifies sensor noise. Measured frame-to-frame noise was
13.0 against 2.1 on the camera's own exposure, and tracking could not read it.
The camera's exposure is the default again, E switches modes while running, and
the short exposure stays available for bright rooms.

Second, and the real reason nothing was tracked: I captured 407 frames over 25
seconds and looked at one. The webcam was pointing at the room and my face. At
a normal screen angle a laptop camera cannot see hands on the table, so there
was nothing to detect. The keyboard now says so instead of waiting quietly, and
the main launcher accepts --phone, using the rear phone camera over USB that the
press keyboard already used. A phone above the table sees whole hands, wrists
included, while the screen stays readable. The buffer fix alone still doubles
the laptop webcam to 16 FPS at full picture quality.

## Wrong keys from still fingers, and a bigger picture

A screen recording of me typing showed a narrow strip of camera picture with
the text box drawn over my hands, and text such as `heoj jt wt QQTL`. Tracking
could not be recovered from the recording, so I stepped through the app's own
frames around each typed letter. At one point J appeared while only my left hand
moved: the right index finger had not moved, but its label had been orange,
meaning in motion, for many frames. Its hover position had shifted slightly
over the same key, and the detector only let a finger settle when it ended over
a different key. It waited up to 1.8 seconds, and one frame of tracking wobble
back toward rest counted as a lift. A resting pinky also pressed Caps, which
turned the following letters into capitals without any visible sign.

A tap now has to come back within 0.9 seconds and stay back for two frames.
Held movement becomes the new rest. Each finger's resting wobble is measured,
and a press must stand clearly above it. A new test reproduces the shifted,
wobbling finger and failed on the old detector; another checks that real taps
through the same wobble still type once. In a simulation over a range of wobble
sizes, false keys from shifting fingers fell about tenfold. Taps were detected
as often as before except with very large wobble, where some were missed. I chose the thresholds with a small sweep of that
simulation, so they still need checking with real typing.

The camera picture is now enlarged beside a control panel, keys under the
fingertips are outlined, and Caps Lock and Shift are shown when on. Drawing the
larger picture takes less time than before, 1.1 ms against 6.6 ms, because only
the text box is blended now. The next step is recording landmarks while typing
a known sentence, so accuracy can be measured instead of estimated.

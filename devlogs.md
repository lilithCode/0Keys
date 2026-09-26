# 0Keys Devlogs

## Devlog 1: Going back to school on signals

Before writing any keyboard code I spent a few days just learning how sound works on a computer. My plan was to hear taps on the table with the mic, so I needed to know what I was actually dealing with. I played around with chirps, FFTs and filters until the Nyquist theorem finally clicked. At 48 kHz I can capture anything up to 24 kHz, which is way more than a finger tap needs. Honestly it felt slow not building anything "real", but I'm glad I did it first.

## Devlog 2: Hearing a tap

Got the microphone detecting table taps today! I chopped the audio into tiny 5 ms blocks and filtered out everything below 600 Hz, because my fan and the room hum live down there. The hard part was the difference between a tap and someone closing a door. I ended up checking several things at once, like how sudden the energy jump is and how "spiky" the sound is. Steady background noise now gets ignored, which felt like a small win.

## Devlog 3: The camera sees my hands

MediaPipe is kind of magic. I plugged it in and suddenly had 21 points on each hand, live, from a normal webcam. The annoying part was that hands kept swapping IDs whenever one left the frame for a second. So I wrote a little tracker that remembers which hand is which by where the wrist was. I also started saving timestamps with every frame so I can match them to the mic later.

## Devlog 4: Drawing a keyboard on the table

Now there's a keyboard! You click four corners on the camera picture and a full QWERTY layout gets stretched between them. Since the camera looks at the table at an angle, a plain rectangle looked totally wrong. I used a homography to fix the perspective, which I had to read up on for a while. Keys light up when a fingertip hovers over them, and the layout saves itself so I don't redo it every time.

## Devlog 5: My training data was garbage

I built personal training so the app could learn how *I* type, and at first it just guessed weird keys. I finally opened the saved profile and found that 74 of my 87 samples weren't even near the key they were labelled as. Oops. So now the trainer freezes the frame after each tap, shows me where it thinks my finger was, and only saves if I confirm. It's slower, but at least the data is honest now.

## Devlog 6: Starting small with four keys

Full QWERTY was too much to debug at once, so I stepped back to just A, S, D and F. Each finger gets its own key, and a press only counts if the camera sees the finger lift and come back down *and* the mic hears the tap at the same moment. Resting or just holding a finger down doesn't type anything. I also added a timed test that prompts me to tap and counts what it missed. Shrinking the problem made it so much easier to see what was going wrong.

## Devlog 7: Left is right and right is left

Spent way too long on this one. My left hand kept getting set up as the right hand. It turns out that when you mirror the camera picture, MediaPipe's left/right labels flip too, and I wasn't correcting them. I tested it with a reference photo to be sure, then fixed it in the shared tracker so every program gets the real hand names. Old profiles had the labels backwards, so those need to be recorded again.

## Devlog 8: Talking to my keyboard

Recording hand poses was awkward because I had to take a hand off the table to press Space. So I added voice commands: I just say "record" and it counts down and captures the pose. It all runs offline with a small speech model, with nothing uploaded anywhere. I made sure the voice part switches off while I'm actually typing, so random talking can't mess anything up.

## Devlog 9: Borrowing my phone's camera

The laptop webcam looks at my face, not my hands, which is a pretty big problem for a hands-on-table keyboard. So I hooked up my Redmi phone over USB and I'm using its back camera instead. I read the video straight out of scrcpy, which avoids needing any weird virtual camera drivers. Then scrcpy 4 changed its packet format and broke everything for an afternoon, but I got it working at around 30 FPS. The phone propped above the table sees both hands perfectly.

## Devlog 10: Trying a totally different idea

Tapping was fighting me, so I tried something else just to compare: point at a key with your index finger and pinch to click. No table and no microphone. The tricky bit was that pinching moves your finger, so it would click the key *next to* the one you aimed at. I fixed that by locking the key from where your finger was just before the pinch. It works, but it's a two-finger air keyboard, not real typing, so I kept it as a side mode.

## Devlog 11: Why wasn't anything typing?

I finally made one main launcher, `keyboard.py`, with simple controls where T always means "type". Then I figured out why nothing had been typing for days. My saved profile had 45 examples of "not a key" and literally zero examples of actual key presses, lol. So the detector now works without any training at all, and training is just an optional extra. That one felt good to find.

## Devlog 12: The camera was the bug all along

The keyboard was running at about 8 FPS, which is way too slow to catch a quick tap. The cause was a one-frame buffer setting that made the camera driver skip every other frame. I moved camera reading to a background thread and got a free 2x speedup. I also tried forcing a short exposure for even more FPS, but in my dim room the picture turned into pure noise and tracking died, so I switched back. And the funniest part: when I looked at the frames, the webcam was pointing at my face the whole time. The app now actually tells you that.

## Devlog 13: Ghost letters from still fingers

In a recording I noticed a J getting typed while my right hand wasn't even moving. Stepping through frame by frame, I saw the finger had shifted a tiny bit and the detector kept thinking it was "mid-press" for almost two seconds. Then one frame of tracking jitter looked like a lift, and boom, J. Now a tap has to come back within 0.9 seconds, and each finger's normal wobble is measured so a press has to clearly beat it. I also made the camera picture way bigger and moved the text box off my hands.

## Devlog 14: A fake hand in the background

Typed a whole sentence and got ` j qp[iqjto p` back. Great. It turned out the tracker had latched onto some blurry thing in the background and decided it was a hand. Since it only tracks two hands, my actual left hand got ignored completely. Now tiny "hands" or ones sitting way past the keyboard get ignored, and I grey out that spot for a moment so the tracker lets go. I also made the replay tool score a recording against the sentence I meant to type, so I can measure progress instead of guessing.

## Devlog 15: Point up to start

I wanted a way to start typing without touching the laptop, so I added a hand sign: index finger up, everything else folded. The tracker couldn't read my drawing of the sign, so I tested on MediaPipe's own sample photos instead. That covered thumbs up, peace signs and open hands, all rotated and mirrored, to make sure only the right one triggers. You hold it for about half a second and a green bar fills up. After that you get a short moment to put your hands down so moving into place doesn't type anything.

## Devlog 16: It types everywhere now!!

This is the big one. Until today 0Keys could only type into its own little text box, which is kind of useless in real life. Now there's a small control panel that stays running, and pressing Ctrl+Alt+K in *any* app starts typing into that app. Wayland made this painful because apps aren't allowed to listen for global shortcuts. My workaround on Hyprland is to have the app add the shortcut to the compositor itself, and on other systems it uses the normal hotkey APIs. I also added safety stuff: a short "get ready" delay, auto-pause when your hands leave, and the camera only turns on while you're typing.

## Devlog 17: Spring cleaning

After months of experiments the project folder was a mess. There were old four-key keyboards, the voice setup, the mic tap detector and my practice scripts from week one. None of it is used by the actual app anymore, so I deleted it all (git still has it if I ever miss it). I also rewrote the README from scratch, because the old one read like a lab notebook. Now it just says how to install it, set it up and use it. It felt weirdly emotional deleting the tap detector, since that's where the whole thing started.

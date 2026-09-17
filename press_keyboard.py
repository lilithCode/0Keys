from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime
from pathlib import Path
import random
import shutil
import sys
import textwrap
import time

import cv2
import numpy as np
import sounddevice as sd

from hand_tracking import draw_snapshot
from helper_audio import AudioTapDetector, MicrophoneInput
from helper_press import AccuracyCheck, KEYS, PressDetector, PressProfile, TwoHandPressDetector, stable_pose
from helper_vision import FINGERTIPS, HandHistory, MediaPipeHandTracker
from helper_voice import VoiceRecordControl
from helper_phone import PhoneCamera
from helper_camera_view import CameraView, add_view_arguments
from session_logger import SessionLogger


def parse_args():
    parser = argparse.ArgumentParser(description="Home-row keyboard with deliberate lift and tap detection")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--phone", action="store_true", help="Use the rear Android camera over USB")
    add_view_arguments(parser)
    parser.add_argument("--hover-start", action="store_true",
                        help="Allow a stable raised finger to arm without touching the table first")
    parser.add_argument("--microphone")
    parser.add_argument("--hand", choices=("auto", "left", "right", "both"), default="auto",
                        help="Select one visible hand automatically, a specific hand, or both hands")
    parser.add_argument("--model", default="models/hand_landmarker.task")
    parser.add_argument("--profile")
    parser.add_argument("--right-profile",
                        help="Right-hand setup file when using both hands")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--camera-offset-ms", type=float, default=0.0)
    parser.add_argument("--threshold-db", type=float, default=11.0)
    parser.add_argument("--voice-model", default="models/vosk-model-small-en-us-0.15",
                        help="Local Vosk model for the spoken record command")
    parser.add_argument("--no-voice", action="store_true",
                        help="Use physical Space instead of voice during setup")
    return parser.parse_args()


def draw_panel(frame, title, instructions, status, typed, states, fps, audio_status=""):
    height, width = frame.shape[:2]
    canvas = np.full((height + 285, width, 3), (24, 24, 28), np.uint8)
    canvas[:height] = frame
    lines = [title, instructions, status, states, audio_status, f"Tracking {fps:.1f} FPS | {typed[-65:]}"]
    y = height + 26
    for index, text in enumerate(lines):
        for line in textwrap.wrap(text, width=75)[:2]:
            cv2.putText(canvas, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.46, (80, 230, 160) if index == 0 else (230, 230, 235),
                        1, cv2.LINE_AA)
            y += 22
    return canvas


def main() -> int:
    args = parse_args()
    auto_hand = args.hand == "auto"
    both_hands = args.hand == "both"
    handedness = "Left" if auto_hand or both_hands else args.hand.title()
    view = CameraView.from_args(args, phone=args.phone)
    args.rotation = view.rotation
    mirrored = view.mirrored
    args.profile = args.profile or ("config/phone_press_left.json" if args.phone else "config/press_profile.json")
    args.right_profile = args.right_profile or ("config/phone_press_right.json" if args.phone else "config/press_right_profile.json")
    path = Path(args.profile)
    profiles = {}
    paths = {"Left": path}
    if both_hands:
        paths["Right"] = Path(args.right_profile)
        if paths["Left"].resolve() == paths["Right"].resolve():
            print("Left and right profiles must use different files", file=sys.stderr)
            return 1
    if not Path(args.model).is_file():
        print("Hand model is missing. Follow the model download in README.md.", file=sys.stderr)
        return 1
    if not np.isfinite(args.camera_offset_ms) or abs(args.camera_offset_ms) > 200:
        print("Camera offset must be between -200 and 200 milliseconds", file=sys.stderr)
        return 1
    try:
        camera = PhoneCamera() if args.phone else cv2.VideoCapture(args.camera)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Phone camera: {exc}", file=sys.stderr)
        return 1
    if not camera.isOpened():
        camera.release()
        print("Could not open the camera", file=sys.stderr)
        return 1
    if args.phone:
        args.camera = f"usb:{camera.info['serial']}:rear:r{args.rotation}"
        print(f"Using {camera.info['model']} rear camera over USB. Keep the phone fixed.")
    else:
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 30)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if args.rotation:
            args.camera = f"webcam:{args.camera}:r{args.rotation}"
    profile = None
    if both_hands:
        if not args.calibrate:
            for side, profile_path in paths.items():
                if not profile_path.exists():
                    continue
                try:
                    loaded = PressProfile.load(profile_path)
                    if (loaded.camera, loaded.mirrored, loaded.handedness) == (args.camera, mirrored, side):
                        profiles[side] = loaded
                except (OSError, ValueError, TypeError, KeyError) as exc:
                    print(f"Repeat {side.lower()} setup: {exc}")
        handedness = next((side for side in paths if side not in profiles), "Left")
        path = paths[handedness]
    elif path.exists() and not args.calibrate:
        try:
            loaded = PressProfile.load(path)
            if (loaded.camera, loaded.mirrored) == (args.camera, mirrored) and (
                auto_hand or loaded.handedness == handedness
            ):
                profile = loaded
                handedness = loaded.handedness
        except (OSError, ValueError, TypeError, KeyError) as exc:
            print(f"Setup needs to be repeated: {exc}")
    microphone_device = args.microphone
    if microphone_device is not None and microphone_device.isdigit():
        microphone_device = int(microphone_device)
    audio = AudioTapDetector(threshold_db=args.threshold_db,
                             minimum_rise_db=3.5, minimum_crest_db=3.0)
    microphone = MicrophoneInput(device=microphone_device)
    history = HandHistory()
    engine = (TwoHandPressDetector(profiles, args.camera_offset_ms / 1000, args.hover_start)
              if both_hands and len(profiles) == 2
              else PressDetector(profile, args.camera_offset_ms / 1000, args.hover_start) if profile else None)
    pose_order = [None, *KEYS[handedness]]
    pose_index = 0
    poses = {}
    pose_frames = []
    collecting_at = None
    paused = False
    suppressed_until = 0.0
    typed = ""
    status = "Keep quiet while the microphone calibrates"
    frame_times = deque(maxlen=30)
    check = None
    logger = None
    voice = None
    voice_drops = 0
    last_drops = 0
    last_summary = ""
    mic_level_db = -120.0
    trial_fps = []
    trial_latencies = []
    trial_missing_frames = 0
    trial_audio_drops = 0
    window = "0Keys home-row press keyboard" if both_hands else "0Keys four-key press keyboard"
    try:
        if not args.no_voice:
            try:
                voice = VoiceRecordControl(args.voice_model)
            except RuntimeError as exc:
                print(f"{exc}. Space still records setup poses.", file=sys.stderr)
        logger = SessionLogger("logs")
        print(f"Results: {logger.path}")
        cv2.namedWindow(window, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(window, 1100, 900)
        with MediaPipeHandTracker(args.model, input_mirrored=mirrored) as tracker, microphone:
            while True:
                ok, frame = camera.read()
                captured = camera.timestamp if args.phone else time.perf_counter()
                if not ok:
                    print(camera.error if args.phone else "Camera stopped returning frames", file=sys.stderr)
                    return 1
                frame = view.apply(frame)
                snapshot = tracker.process(frame, captured)
                history.append(snapshot)
                frame_times.append(captured)
                fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0]) if len(frame_times) > 1 else 0
                taps = []
                if voice:
                    voice.set_listening(engine is None and collecting_at is None
                                        and audio.calibrated and captured >= suppressed_until)
                    if microphone.dropped_blocks != voice_drops:
                        voice.reset()
                        voice_drops = microphone.dropped_blocks
                blocks = microphone.drain()
                if blocks:
                    samples = np.concatenate([block.samples for block in blocks])
                    mic_level_db = float(20 * np.log10(max(float(np.sqrt(np.mean(samples ** 2))), 1e-6)))
                if voice:
                    voice.submit(blocks)
                for block in blocks:
                    tap = audio.process_block(block.samples, block.first_sample_time)
                    if tap and tap.timestamp >= suppressed_until:
                        taps.append(tap)
                now = time.perf_counter()
                if (engine is None and auto_hand and pose_index == 0 and not pose_frames
                        and len(snapshot.hands) == 1 and snapshot.hands[0].handedness in KEYS):
                    handedness = snapshot.hands[0].handedness
                    pose_order = [None, *KEYS[handedness]]
                selected = [hand for hand in snapshot.hands if hand.handedness == handedness]
                if both_hands and engine is not None:
                    selected = [hand for hand in snapshot.hands if hand.handedness in KEYS]
                auto_ambiguous = engine is None and auto_hand and pose_index == 0 and len(snapshot.hands) > 1
                if auto_ambiguous:
                    selected = []
                states = ""

                if engine is None:
                    voice_ready = voice is not None and not voice.error
                    record_hint = 'Say "ready" or "record", or press Space' if voice_ready else "Press Space"
                    if voice and voice.pop_record():
                        collecting_at = now + 1.5
                        pose_frames.clear()
                        suppressed_until = now + 1.5
                        logger.write("press_setup_voice_record", handedness=handedness,
                                     pose_index=pose_index)
                    tip = pose_order[pose_index]
                    target = "rest all four fingers on the table" if tip is None else f"lift only your {FINGERTIPS[tip]} for {KEYS[handedness][tip].upper()}"
                    title = f"Setup {pose_index + 1} of 5: {handedness} hand"
                    instructions = f"{target.capitalize()}. {record_hint}. R restarts. Q quits."
                    if not both_hands:
                        instructions += " H switches hand."
                    seen = ", ".join(hand.handedness for hand in snapshot.hands) or "none"
                    states = f"Camera sees: {seen}. Selected: {handedness}. Keys: {' '.join(KEYS[handedness].values()).upper()}"
                    if not args.no_voice and not voice_ready:
                        states += " Voice unavailable. See terminal and README."
                    if not audio.calibrated:
                        status = f"Microphone calibration {audio.calibration_progress * 100:.0f}%. Keep quiet."
                    elif collecting_at is not None:
                        if now < collecting_at:
                            status = f"Recording in {collecting_at - now:.1f}s. Keep your palm still."
                        elif len(selected) != 1:
                            pose_frames.clear()
                            if auto_ambiguous:
                                status = "Show just the hand you want to record, or press H to select a side."
                            elif not snapshot.hands:
                                status = "No hand detected. Keep the wrist and all fingers in view."
                            elif not selected:
                                status = f"Camera sees {seen}, setup wants {handedness}."
                                if not both_hands:
                                    status += " Press H to switch."
                            else:
                                status = f"Multiple hands labeled {handedness}. Leave just one in view."
                            if now - collecting_at > 10.0:
                                collecting_at = None
                                status = "Recording stopped. " + status + f" {record_hint} to retry."
                        else:
                            if pose_frames and (selected[0].hand_id != pose_frames[-1].hand_id or captured - previous_pose_time > 0.25):
                                pose_frames.clear()
                            pose_frames.append(selected[0])
                            previous_pose_time = captured
                            status = f"Hold still. Recording {len(pose_frames)} of 8 frames"
                            if len(pose_frames) == 8:
                                try:
                                    pose = stable_pose(pose_frames)
                                    if tip is not None:
                                        rest = poses[None]
                                        if np.max(np.linalg.norm(pose["anchors"] - rest["anchors"], axis=1)) > rest["scale"] * 0.25:
                                            raise ValueError("Palm moved. Return to your resting position and retry")
                                        if np.linalg.norm(pose["features"][tip] - rest["features"][tip]) < 0.12:
                                            raise ValueError("Lift was too small to see. Lift a little more and retry")
                                    poses[tip] = pose
                                    pose_index += 1
                                    status = f"Pose saved. Set up the next pose. {record_hint}."
                                    if pose_index == len(pose_order):
                                        profile = PressProfile.from_poses(handedness, args.camera, mirrored, poses[None], poses)
                                        if path.exists():
                                            backup = path.with_name(path.stem + ".backup_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".json")
                                            shutil.copy2(path, backup)
                                        profile.save(path)
                                        if both_hands:
                                            profiles[handedness] = profile
                                            if len(profiles) == 2:
                                                engine = TwoHandPressDetector(profiles, args.camera_offset_ms / 1000, args.hover_start)
                                            else:
                                                handedness = next(side for side in paths if side not in profiles)
                                                path = paths[handedness]
                                                pose_order = [None, *KEYS[handedness]]
                                                pose_index = 0
                                                poses.clear()
                                                status = f"Now set up your {handedness.lower()} hand. {record_hint}."
                                        else:
                                            engine = PressDetector(profile, args.camera_offset_ms / 1000, args.hover_start)
                                        suppressed_until = now + 0.8
                                        logger.write("press_setup_saved", handedness=profile.handedness)
                                except ValueError as exc:
                                    status = str(exc) + f". {record_hint} to retry."
                                    pose_index = min(pose_index, len(pose_order) - 1)
                                collecting_at = None
                                pose_frames.clear()
                    elif status.startswith(("Microphone calibration", "Keep quiet")):
                        status = f"Ready. Hold the requested pose. {record_hint}."
                    if audio.calibrated and collecting_at is None and not snapshot.hands:
                        status = "No hands detected. Raise and tilt the phone to show fingers and wrists."
                else:
                    active_profiles = list(profiles.values()) if both_hands else [profile]
                    title = ("Home row: A S D F + J K L ;" if both_hands
                             else "Four-key keyboard: " + " ".join(KEYS[handedness].values()).upper())
                    instructions = "Lift then tap. Space pauses. T tests. R sets up. C clears. Q quits."
                    if not both_hands:
                        instructions += " H switches hand."
                    if args.hover_start:
                        title += " | Hover start"
                    if microphone.dropped_blocks != last_drops:
                        engine.reset("Audio dropped. Rest your fingers before continuing")
                        logger.write("audio_gap", dropped_blocks=microphone.dropped_blocks)
                        last_drops = microphone.dropped_blocks
                        taps = []
                    if not audio.calibrated or paused or captured < suppressed_until:
                        engine.reset()
                        status = (last_summary or "Paused. Space resumes") if paused else "Wait, then rest your fingers at the markers"
                    else:
                        events = engine.update(snapshot, taps)
                        status = engine.status
                        for event in events:
                            typed = (typed + event.key)[-2000:]
                            latency = (now - event.audio_time) * 1000
                            logger.write("press", key=event.key, tip=event.tip,
                                         audio_time=event.audio_time, contact_time=event.contact_time,
                                         latency_ms=latency, tracking_fps=fps)
                            if check:
                                check.events.append((event.audio_time, event.key))
                                trial_latencies.append(latency)
                    if check:
                        trial_fps.append(fps)
                        trial_missing_frames += any(
                            sum(hand.handedness == item.handedness for hand in selected) != 1
                            for item in active_profiles
                        )
                        logger.write(
                            "press_test_frame", camera_time=captured,
                            prompt=check.phase(now), status=status,
                            landmarks=[[(p.x, p.y, p.z) for p in hand.landmarks] for hand in selected],
                            states={f"{item.handedness}:{tip}": {"phase": state.phase, "progress": state.progress}
                                    for item in active_profiles
                                    for tip, state in (engine.engines[item.handedness] if both_hands else engine).states.items()},
                            audio_timestamps=[tap.timestamp for tap in taps],
                        )
                        title, expected = check.phase(now)
                        if expected:
                            title += ": " + expected.upper()
                        instructions = "Follow the prompt. One tap per trial. Misses count automatically. Q aborts."
                        if now >= check.started + check.duration + 0.5:
                            report = check.report(now)
                            report.update(
                                median_tracking_fps=float(np.median(trial_fps)),
                                missing_hand_frames=trial_missing_frames,
                                audio_dropped_blocks=microphone.dropped_blocks - trial_audio_drops,
                                median_latency_ms=float(np.median(trial_latencies)) if trial_latencies else None,
                                p95_latency_ms=float(np.percentile(trial_latencies, 95)) if trial_latencies else None,
                            )
                            logger.write("press_accuracy", **report)
                            total = len(report["attempts"])
                            print(json_report(report))
                            status = f"Test: {report['correct']} of {total} correct, {report['missed']} missed, {report['false_activations']} accidental"
                            last_summary = status + ". Space resumes."
                            check = None
                            paused = True
                    state_labels = []
                    for item in active_profiles:
                        hand_engine = engine.engines[item.handedness] if both_hands else engine
                        for finger in item.fingers:
                            phase = hand_engine.states[finger.tip].phase
                            state_labels.append(f"{finger.key.upper()}: {phase}")
                            point = (int(finger.position[0] * frame.shape[1]), int(finger.position[1] * frame.shape[0]))
                            color = (40, 230, 150) if phase == "resting" else (40, 190, 255)
                            cv2.circle(frame, point, 15, color, 2)
                            cv2.putText(frame, finger.key.upper(), (point[0] - 5, point[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    states = " | ".join(state_labels)

                draw_snapshot(frame, snapshot, history)
                voice_status = (voice.error or voice.feedback) if voice else "Voice unavailable" if not args.no_voice else "Voice disabled"
                audio_status = f"Mic {mic_level_db:.0f} dBFS. {voice_status}."
                if engine is None and mic_level_db < -60 and audio.calibrated and collecting_at is None:
                    audio_status += " Input quiet. Speak toward the laptop."
                canvas = draw_panel(frame, title, instructions, status, typed, states, fps, audio_status)
                cv2.imshow(window, canvas)
                key = cv2.waitKey(1) & 0xFF
                now = time.perf_counter()
                if key in (ord("q"), 27):
                    return 0
                if check:
                    continue
                if key == ord("r") or (key == ord("h") and not both_hands):
                    if voice:
                        voice.set_listening(False)
                    if both_hands:
                        profiles.clear()
                        handedness = "Left"
                        path = paths[handedness]
                        pose_order = [None, *KEYS[handedness]]
                    if key == ord("h"):
                        handedness = "Right" if handedness == "Left" else "Left"
                        auto_hand = False
                        pose_order = [None, *KEYS[handedness]]
                    engine = None
                    pose_index = 0
                    collecting_at = None
                    poses.clear()
                    pose_frames.clear()
                    paused = False
                    last_summary = ""
                    status = "Start with your fingers resting naturally"
                    suppressed_until = now + 0.8
                elif key == ord(" "):
                    if voice:
                        voice.set_listening(False)
                    suppressed_until = now + 0.8
                    if engine:
                        paused = not paused
                        last_summary = ""
                        engine.reset()
                    elif audio.calibrated:
                        collecting_at = now + 1.0
                        pose_frames.clear()
                elif key == ord("c") and engine:
                    typed = ""
                    suppressed_until = now + 0.8
                    engine.reset()
                elif key == ord("t") and engine and audio.calibrated:
                    targets = ([key for side in ("Left", "Right") for key in KEYS[side].values()]
                               if both_hands else list(KEYS[handedness].values())) * 5
                    random.Random(7).shuffle(targets)
                    check = AccuracyCheck(targets, now + 1.0)
                    trial_fps.clear()
                    trial_latencies.clear()
                    trial_missing_frames = 0
                    trial_audio_drops = microphone.dropped_blocks
                    engine.reset()
                    paused = False
                    suppressed_until = now + 1.0
                    logger.write("press_accuracy_started", targets=targets)
    except KeyboardInterrupt:
        return 0
    except (OSError, RuntimeError, ValueError, sd.PortAudioError) as exc:
        print(f"Press keyboard failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if voice:
            voice.close()
        camera.release()
        cv2.destroyAllWindows()
        if check and logger:
            report = check.report(time.perf_counter())
            report["complete"] = False
            logger.write("press_accuracy_aborted", **report)


def json_report(report):
    total = len(report["attempts"])
    return (f"Press test: {report['correct']} of {total} correct; "
            f"{report['missed']} missed; {report['extra_keys']} extra; "
            f"{report['false_activations']} accidental during rest or preparation. "
            "See the session log for each trial.")


if __name__ == "__main__":
    raise SystemExit(main())

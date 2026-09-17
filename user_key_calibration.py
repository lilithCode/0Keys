from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime
from pathlib import Path
import shutil
import sys
import textwrap
import time

import cv2
import numpy as np
import sounddevice as sd

from hand_tracking import draw_snapshot
from helper_audio import AudioTapDetector, MicrophoneInput, TapEvent
from helper_classifier import PersonalKeyClassifier, UserKeyboardProfile, profile_context
from helper_fusion import TapFingerFusion
from helper_keyboard import KeyboardCalibration, KeyboardLayout
from helper_training import TrainingSession, training_candidate
from helper_vision import HandHistory, MediaPipeHandTracker
from keyboard_calibration import draw_keyboard, highlighted_keys
from session_logger import SessionLogger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a personal keyboard profile")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--microphone")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--model", default="models/hand_landmarker.task")
    parser.add_argument("--calibration", default="config/keyboard_calibration.json")
    parser.add_argument("--profile", default="config/user_keyboard_profile.json")
    parser.add_argument("--key-set", choices=("common", "all"), default="common")
    parser.add_argument("--keys", nargs="+", help="Practice specific key names, for example a s backspace")
    parser.add_argument("--samples-per-key", type=int, default=5)
    parser.add_argument("--validate", action="store_true", help="Measure accuracy without training")
    parser.add_argument("--validation-rounds", type=int, default=3)
    parser.add_argument("--camera-offset-ms", type=float, default=70.0)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--no-mirror", action="store_true")
    return parser.parse_args()


def normalize_device(value: str | None) -> int | str | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return value


def training_keys(layout: KeyboardLayout, key_set: str) -> list[str]:
    if key_set == "common":
        return list("asdfjklqwertyuiopzxcvbnmgh") + [
            "space",
            "backspace",
            "enter",
        ]
    return [key.name for key in layout.keys]


def build_schedule(
    profile: UserKeyboardProfile,
    key_names: list[str],
    samples_per_key: int,
) -> list[str]:
    schedule = []
    for key_name in key_names:
        schedule.extend([key_name] * max(0, samples_per_key - profile.count(key_name)))
    return schedule


def draw_training_panel(
    frame,
    target_label: str,
    progress: str,
    status: str,
    audio_ready: bool,
    reviewing: bool = False,
    waiting: bool = False,
):
    height, width = frame.shape[:2]
    panel_height = 150
    canvas = np.full((height + panel_height, width, 3), (24, 24, 28), dtype=np.uint8)
    canvas[:height] = frame
    heading = "TAP" if audio_ready else "WAIT FOR AUDIO CALIBRATION"
    if reviewing:
        heading = "CHECK"
    elif audio_ready and waiting:
        heading = "GET READY"
    if target_label == "COMPLETE":
        heading = "DONE"
    cv2.putText(
        canvas,
        f"{heading}: {target_label}",
        (14, height + 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.90,
        (40, 230, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        progress,
        (14, height + 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    lines = textwrap.wrap(status, width=78)[:2]
    for index, line in enumerate(lines):
        cv2.putText(
            canvas,
            line,
            (14, height + 101 + index * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (180, 200, 210),
            1,
            cv2.LINE_AA,
        )
    return canvas


def main() -> int:
    args = parse_args()
    if args.samples_per_key < 1 or args.validation_rounds < 1:
        print("Sample and validation counts must be at least one", file=sys.stderr)
        return 1

    model_path = Path(args.model)
    calibration_path = Path(args.calibration)
    profile_path = Path(args.profile)
    mirrored = not args.no_mirror
    if not model_path.is_file() or not calibration_path.is_file():
        print("The model and table calibration are required", file=sys.stderr)
        return 1

    try:
        calibration = KeyboardCalibration.load(calibration_path)
    except (ValueError, KeyError, OSError) as exc:
        print(f"Could not load table calibration: {exc}", file=sys.stderr)
        return 1
    if calibration.mirrored != mirrored:
        print("Calibration mirror setting does not match this run", file=sys.stderr)
        return 1

    context = profile_context(calibration, args.camera, args.camera_offset_ms)
    profile = UserKeyboardProfile(context=context)
    replace_profile = args.fresh
    if args.validate and args.fresh:
        print("Validation cannot be combined with fresh training", file=sys.stderr)
        return 1
    if profile_path.is_file() and not args.fresh:
        try:
            loaded = UserKeyboardProfile.load(profile_path)
        except (ValueError, TypeError, KeyError, OSError) as exc:
            print(f"Could not load personal profile: {exc}", file=sys.stderr)
            return 1
        if loaded.compatible_with(context) and not args.fresh:
            profile = loaded
        else:
            replace_profile = True
    if args.validate and (not profile.samples or replace_profile):
        print("Train a new profile with this camera calibration before validation.", file=sys.stderr)
        return 1

    layout = KeyboardLayout()
    keys_by_name = {key.name: key for key in layout.keys}
    key_names = list(dict.fromkeys(args.keys)) if args.keys else training_keys(layout, args.key_set)
    if any(name not in keys_by_name for name in key_names):
        print("Unknown key name. Use lowercase letters or names such as space and backspace.", file=sys.stderr)
        return 1
    schedule = (
        key_names * args.validation_rounds if args.validate
        else build_schedule(profile, key_names, args.samples_per_key)
    )
    total_needed = len(schedule)
    session = TrainingSession(profile, schedule)
    validation_results = []
    review_frame = None
    review_predictions = ()
    frames = deque(maxlen=30)
    skipped = 0
    discarded = 0
    validation_log = None

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        print(f"Could not open camera {args.camera}", file=sys.stderr)
        return 1
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    history = HandHistory(history_seconds=1.0, maximum_frames=150)
    detector = AudioTapDetector(
        threshold_db=11.0,
        minimum_rise_db=3.5,
        minimum_crest_db=3.0,
    )
    microphone = MicrophoneInput(device=normalize_device(args.microphone))
    fusion = TapFingerFusion(
        layout,
        calibration,
        camera_offset_seconds=args.camera_offset_ms / 1_000.0,
        minimum_motion_score=0.04,
    )
    classifier = PersonalKeyClassifier(layout, profile)
    pending_taps: list[TapEvent] = []
    status = "Keep quiet while the microphone calibrates"
    window_name = "0Keys personal key training"

    try:
        if args.validate:
            validation_log = SessionLogger("logs")
            print(f"Validation report: {validation_log.path}")
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(window_name, 1100, 900)
        with MediaPipeHandTracker(model_path, input_mirrored=mirrored) as tracker, microphone:
            if replace_profile and profile_path.is_file():
                backup = profile_path.with_name(
                    profile_path.stem + ".backup_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".json"
                )
                shutil.copy2(profile_path, backup)
                print(f"Previous profile preserved at {backup}")
                profile.save(profile_path)
            while True:
                ok, frame = camera.read()
                capture_time = time.perf_counter()
                if not ok:
                    print("Camera stopped returning frames", file=sys.stderr)
                    return 1
                if mirrored:
                    frame = cv2.flip(frame, 1)
                frames.append((capture_time, frame.copy()))

                snapshot = tracker.process(frame, capture_time)
                history.append(snapshot)
                for block in microphone.drain():
                    tap = detector.process_block(block.samples, block.first_sample_time)
                    if tap is not None and session.can_capture(tap.timestamp):
                        pending_taps.append(tap)

                ready_delay = fusion.camera_offset_seconds + fusion.post_motion_seconds + 0.08
                waiting = []
                for tap in pending_taps:
                    if not session.can_capture(tap.timestamp):
                        continue
                    if capture_time < tap.timestamp + ready_delay:
                        waiting.append(tap)
                        continue
                    result = fusion.select(tap, history)
                    expected_name = session.target
                    target_key = keys_by_name[expected_name]
                    candidate, status = training_candidate(result, target_key)
                    if args.validate:
                        candidates = fusion.plausible_candidates(result)
                        review_predictions = classifier.predict_candidates(candidates)
                        candidate = result.candidates[0] if result.candidates else None
                        status = "Space counts this attempt. R discards an accidental sound."
                    if candidate is None:
                        if args.validate:
                            validation_results.append((expected_name, None))
                            session.skip(time.perf_counter())
                            status = "Attempt rejected: " + result.reason
                        continue
                    sample = classifier.sample_from_candidate(
                        expected_name,
                        candidate,
                        tap.strength_db,
                        tap.timestamp,
                    )
                    if session.stage(sample):
                        contact_time = tap.timestamp + fusion.camera_offset_seconds
                        review_frame = min(frames, key=lambda item: abs(item[0] - contact_time))[1].copy()
                        draw_keyboard(review_frame, layout, calibration, {expected_name})
                        x, y = calibration.map_to_image(candidate.keyboard_x, candidate.keyboard_y)
                        cv2.circle(
                            review_frame,
                            (int(x * frame.shape[1]), int(y * frame.shape[0])),
                            15, (0, 0, 255), 3,
                        )
                        status = (
                            f"{candidate.handedness} {candidate.finger_name}. " + status
                        )
                        waiting.clear()
                        break
                pending_taps = waiting

                draw_snapshot(frame, snapshot, history)
                active_keys = highlighted_keys(snapshot, layout, calibration)
                if session.target is not None:
                    target_name = session.target
                    target_key = keys_by_name[target_name]
                    active_keys.add(target_name)
                    target_label = target_key.label
                    progress = f"Sample {session.index + 1} of {total_needed}. U undoes. N skips. Q quits."
                else:
                    target_label = "COMPLETE"
                    progress = f"Saved {len(profile.samples)} personal samples. Press Q to finish."
                if args.validate:
                    correct = sum(expected == actual for expected, actual in validation_results)
                    rejected = sum(actual is None for _, actual in validation_results)
                    progress = f"{correct} of {len(validation_results)} correct. {rejected} rejected. M missed. Q quits."
                elif session.pending is None and time.perf_counter() < session.ready_at:
                    status = "Get ready for the target. Lift your finger, then tap once."
                elif detector.calibrated and session.pending is None and status.startswith(
                    ("Keep quiet", "Get ready", "Sample saved")
                ):
                    status = "Tap the target once using your usual finger."
                draw_keyboard(frame, layout, calibration, active_keys)
                if session.pending is not None:
                    frame = review_frame
                    if args.validate:
                        guess = (
                            review_predictions[0].key.label
                            if classifier.accepts(review_predictions) else "REJECTED"
                        )
                        status = f"Predicted {guess}. Space counts. R retries."
                canvas = draw_training_panel(
                    frame,
                    target_label,
                    progress,
                    status,
                    detector.calibrated,
                    session.pending is not None,
                    time.perf_counter() < session.ready_at,
                )
                cv2.imshow(window_name, canvas)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    if args.validate:
                        print(f"Validation: {correct} correct of {len(validation_results)}, {rejected} rejected")
                        for expected, actual in validation_results:
                            if expected != actual:
                                print(f"  {expected}: {actual or 'rejected'}")
                    return 0
                now = time.perf_counter()
                if key == ord(" ") and session.pending is not None:
                    if args.validate:
                        actual = review_predictions[0].key.name if classifier.accepts(review_predictions) else None
                        validation_results.append((session.target, actual))
                        session.skip(now)
                    else:
                        session.confirm(now)
                        profile.save(profile_path)
                    pending_taps.clear()
                    status = "Sample saved. Get ready."
                if key == ord("r"):
                    if args.validate and session.pending is not None:
                        discarded += 1
                    session.retry(now)
                    pending_taps.clear()
                if key == ord("n") and session.target is not None:
                    status = f"Skipped {target_label}"
                    skipped += 1
                    session.skip(now)
                    pending_taps.clear()
                if key == ord("m") and args.validate and session.target is not None:
                    validation_results.append((session.target, None))
                    session.skip(now)
                    pending_taps.clear()
                    status = "Missed tap counted. Get ready."
                if key == ord("u") and not args.validate:
                    if session.undo(now):
                        profile.save(profile_path)
                        pending_taps.clear()
                        status = "Removed the previous training sample"
    except KeyboardInterrupt:
        return 0
    except (sd.PortAudioError, OSError, RuntimeError) as exc:
        print(f"Training failed: {exc}", file=sys.stderr)
        return 1
    finally:
        camera.release()
        cv2.destroyAllWindows()
        if validation_log is not None:
            correct = sum(expected == actual for expected, actual in validation_results)
            rejected = sum(actual is None for _, actual in validation_results)
            validation_log.write(
                "validation_summary",
                attempted=len(validation_results), correct=correct,
                rejected=rejected, skipped=skipped,
                discarded=discarded,
                remaining=total_needed - session.index,
                accuracy=correct / len(validation_results) if validation_results else None,
                results=[{"expected": expected, "predicted": actual}
                         for expected, actual in validation_results],
            )


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import textwrap
import time

import cv2
import numpy as np
import sounddevice as sd

from hand_tracking import draw_snapshot
from helper_audio import AudioTapDetector, MicrophoneInput, TapEvent
from helper_classifier import PersonalKeyClassifier, UserKeyboardProfile, profile_context
from helper_fusion import FusionResult, TapFingerFusion, TextComposer
from helper_keyboard import KeyboardCalibration, KeyboardLayout
from helper_vision import HandHistory, MediaPipeHandTracker
from keyboard_calibration import draw_keyboard, highlighted_keys
from session_logger import SessionLogger


SENSITIVITY_PRESETS = {
    "normal": (14.0, 5.0, 4.0, 0.05),
    "balanced": (11.0, 3.5, 3.0, 0.04),
    "high": (9.0, 2.5, 2.5, 0.03),
    "very-high": (6.0, 1.5, 1.5, 0.02),
}

CORRECTION_WINDOW_SECONDS = 5.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the invisible keyboard")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--microphone")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--model", default="models/hand_landmarker.task")
    parser.add_argument("--calibration", default="config/keyboard_calibration.json")
    parser.add_argument("--profile", default="config/user_keyboard_profile.json")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--no-session-log", action="store_true")
    parser.add_argument(
        "--sensitivity",
        choices=tuple(SENSITIVITY_PRESETS),
        default="balanced",
    )
    parser.add_argument("--threshold-db", type=float)
    parser.add_argument("--rise-db", type=float)
    parser.add_argument("--crest-db", type=float)
    parser.add_argument("--camera-offset-ms", type=float, default=70.0)
    parser.add_argument("--motion-threshold", type=float)
    parser.add_argument("--no-mirror", action="store_true")
    return parser.parse_args()


def normalize_device(value: str | None) -> int | str | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return value


def draw_text_panel(frame, composer: TextComposer, status: str, detail: str):
    height, width = frame.shape[:2]
    panel_height = 175
    canvas = np.full((height + panel_height, width, 3), (24, 24, 28), dtype=np.uint8)
    canvas[:height] = frame
    cv2.putText(
        canvas,
        status,
        (14, height + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (80, 230, 150),
        2,
        cv2.LINE_AA,
    )
    detail_lines = textwrap.wrap(detail, width=82)[:2]
    for index, line in enumerate(detail_lines):
        cv2.putText(
            canvas,
            line,
            (14, height + 53 + index * 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.41,
            (190, 190, 200),
            1,
            cv2.LINE_AA,
        )

    visible_text = composer.text[-180:].replace("\t", "    ")
    lines = []
    for paragraph in visible_text.split("\n")[-3:]:
        lines.extend(textwrap.wrap(paragraph, width=68) or [""])
    for index, line in enumerate(lines[-3:]):
        cv2.putText(
            canvas,
            line,
            (14, height + 101 + index * 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )
    return canvas


def describe_result(result: FusionResult) -> str:
    if result.candidate is None:
        return f"Tap ignored: {result.reason}"
    candidate = result.candidate
    return (
        f"Typed {candidate.key.label} with {candidate.handedness.lower()} "
        f"{candidate.finger_name}, score {candidate.score:.2f}"
    )


def main() -> int:
    args = parse_args()
    model_path = Path(args.model)
    calibration_path = Path(args.calibration)
    profile_path = Path(args.profile)
    mirrored = not args.no_mirror

    if not model_path.is_file():
        print(f"Model is missing: {model_path}", file=sys.stderr)
        return 1
    if not calibration_path.is_file():
        print("Calibration is missing. Run keyboard_calibration.py first.", file=sys.stderr)
        return 1

    try:
        calibration = KeyboardCalibration.load(calibration_path)
    except (ValueError, KeyError, OSError) as exc:
        print(f"Could not load calibration: {exc}", file=sys.stderr)
        return 1
    if calibration.mirrored != mirrored:
        print("Calibration mirror setting does not match this run", file=sys.stderr)
        return 1

    context = profile_context(calibration, args.camera, args.camera_offset_ms)
    learning_enabled = True
    profile_status = "No personal training"
    if profile_path.is_file():
        try:
            profile = UserKeyboardProfile.load(profile_path)
        except (ValueError, TypeError, KeyError, OSError) as exc:
            print(f"Could not load personal profile: {exc}", file=sys.stderr)
            return 1
        if not profile.compatible_with(context):
            print("Old or mismatched training ignored. Run user_key_calibration.py to retrain.")
            profile = UserKeyboardProfile(context=context)
            learning_enabled = False
            profile_status = "Retraining needed"
        else:
            profile_status = f"{len(profile.samples)} personal samples"
    else:
        profile = UserKeyboardProfile(context=context)

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        print(f"Could not open camera {args.camera}", file=sys.stderr)
        return 1
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    layout = KeyboardLayout()
    classifier = PersonalKeyClassifier(layout, profile)
    history = HandHistory(history_seconds=1.0, maximum_frames=150)
    preset = SENSITIVITY_PRESETS[args.sensitivity]
    threshold_db = args.threshold_db if args.threshold_db is not None else preset[0]
    rise_db = args.rise_db if args.rise_db is not None else preset[1]
    crest_db = args.crest_db if args.crest_db is not None else preset[2]
    motion_threshold = (
        args.motion_threshold if args.motion_threshold is not None else preset[3]
    )
    detector = AudioTapDetector(
        threshold_db=threshold_db,
        minimum_rise_db=rise_db,
        minimum_crest_db=crest_db,
    )
    microphone = MicrophoneInput(device=normalize_device(args.microphone))
    fusion = TapFingerFusion(
        layout,
        calibration,
        camera_offset_seconds=args.camera_offset_ms / 1_000.0,
        minimum_motion_score=motion_threshold,
    )
    composer = TextComposer()
    logger = None if args.no_session_log else SessionLogger(args.log_dir)
    pending_taps: list[TapEvent] = []
    last_result = "Waiting for microphone calibration"
    last_key_name = None
    last_key_time = 0.0
    last_predictions = ()
    last_tap = None
    last_composer_state = None
    controls_until = 0.0
    window_name = "0Keys invisible keyboard"

    if logger is not None:
        print(f"Session log: {logger.path}")
        logger.write(
            "session_started",
            sensitivity=args.sensitivity,
            profile_samples=len(profile.samples),
            camera_offset_ms=args.camera_offset_ms,
        )

    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(window_name, 1100, 900)
        with MediaPipeHandTracker(model_path, input_mirrored=mirrored) as tracker, microphone:
            while True:
                ok, frame = camera.read()
                capture_time = time.perf_counter()
                if not ok:
                    print("Camera stopped returning frames", file=sys.stderr)
                    return 1
                if mirrored:
                    frame = cv2.flip(frame, 1)

                snapshot = tracker.process(frame, capture_time)
                history.append(snapshot)

                for block in microphone.drain():
                    tap = detector.process_block(block.samples, block.first_sample_time)
                    if tap is not None and tap.timestamp >= controls_until:
                        pending_taps.append(tap)

                ready_delay = fusion.camera_offset_seconds + fusion.post_motion_seconds + 0.08
                waiting = []
                for tap in pending_taps:
                    if tap.timestamp < controls_until:
                        continue
                    if capture_time < tap.timestamp + ready_delay:
                        waiting.append(tap)
                        continue
                    result = fusion.select(tap, history)
                    last_result = describe_result(result)
                    candidate_pool = fusion.plausible_candidates(result)
                    if logger is not None and not candidate_pool:
                        logger.write(
                            "tap_rejected",
                            reason=result.reason,
                            audio_strength_db=tap.strength_db,
                        )
                    if candidate_pool:
                        last_predictions = classifier.predict_candidates(candidate_pool)
                        if not last_predictions:
                            last_result = "Tap ignored: no nearby key"
                            last_tap = None
                            last_composer_state = None
                            continue
                        prediction = last_predictions[0]
                        predicted_key = prediction.key
                        predicted_candidate = prediction.candidate
                        previous_text = composer.text
                        last_composer_state = (
                            composer.text,
                            composer.caps_lock,
                            composer.shift,
                        )
                        accepted = classifier.accepts(last_predictions)
                        if accepted:
                            composer.apply(predicted_key)
                        if not accepted:
                            last_result = "Uncertain tap. Choose 1, 2 or 3, or tap again."
                        elif (
                            predicted_key.value == "BACKSPACE"
                            and not previous_text
                        ):
                            last_result = "Typed Back, text was already empty"
                        else:
                            last_result = (
                                f"Typed {predicted_key.label} with "
                                f"{predicted_candidate.handedness.lower()} "
                                f"{predicted_candidate.finger_name}"
                            )
                        last_tap = tap
                        last_key_name = predicted_key.name if accepted else None
                        last_key_time = capture_time
                        if logger is not None:
                            logger.write(
                                "key_predicted" if accepted else "key_uncertain",
                                tap_timestamp=tap.timestamp,
                                audio_strength_db=tap.strength_db,
                                fusion_reason=result.reason,
                                candidate_count=len(candidate_pool),
                                hand=predicted_candidate.handedness,
                                finger=predicted_candidate.finger_name,
                                motion_score=predicted_candidate.score,
                                keyboard_x=predicted_candidate.keyboard_x,
                                keyboard_y=predicted_candidate.keyboard_y,
                                predictions=[
                                    {
                                        "key": prediction.key.name,
                                        "probability": prediction.probability,
                                    }
                                    for prediction in last_predictions
                                ],
                            )
                pending_taps = waiting

                draw_snapshot(frame, snapshot, history)
                active_keys = highlighted_keys(snapshot, layout, calibration)
                if last_key_name is not None and capture_time - last_key_time < 0.60:
                    active_keys.add(last_key_name)
                draw_keyboard(frame, layout, calibration, active_keys)

                if detector.calibrated:
                    status = f"Ready, {args.sensitivity}, {profile_status}"
                else:
                    progress = int(detector.calibration_progress * 100)
                    status = f"Calibrating microphone: {progress}%"
                mode = "Caps on" if composer.caps_lock else "Caps off"
                if composer.shift:
                    mode += ", Shift waiting"
                if (
                    last_predictions
                    and capture_time - last_key_time > CORRECTION_WINDOW_SECONDS
                ):
                    last_predictions = ()
                    last_tap = None
                    last_composer_state = None
                choices = ""
                if last_predictions:
                    choices = " | Choices: " + " ".join(
                        f"{index + 1} {prediction.key.label}"
                        for index, prediction in enumerate(last_predictions)
                    )
                detail = f"{last_result}{choices} | {mode} | C clears | Q quits"
                canvas = draw_text_panel(frame, composer, status, detail)
                cv2.imshow(window_name, canvas)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    return 0
                if key == ord("c"):
                    controls_until = time.perf_counter() + 0.5
                    pending_taps.clear()
                    composer.clear()
                    last_result = "Text cleared"
                    last_predictions = ()
                    last_tap = None
                    last_composer_state = None
                if (
                    key in (ord("1"), ord("2"), ord("3"))
                    and last_tap is not None
                    and last_composer_state is not None
                ):
                    choice_index = key - ord("1")
                    if choice_index < len(last_predictions):
                        corrected = last_predictions[choice_index].key
                        corrected_candidate = last_predictions[choice_index].candidate
                        composer.text = last_composer_state[0]
                        composer.caps_lock = last_composer_state[1]
                        composer.shift = last_composer_state[2]
                        composer.apply(corrected)
                        sample = classifier.sample_from_candidate(
                            corrected.name,
                            corrected_candidate,
                            last_tap.strength_db,
                            last_tap.timestamp,
                        )
                        if learning_enabled:
                            profile.add(sample)
                            profile.save(profile_path)
                        last_key_name = corrected.name
                        last_key_time = capture_time
                        last_result = f"Corrected to {corrected.label}"
                        controls_until = time.perf_counter() + 0.5
                        pending_taps.clear()
                        if logger is not None:
                            logger.write(
                                "key_corrected",
                                selected_key=corrected.name,
                                choice=choice_index + 1,
                                tap_timestamp=last_tap.timestamp,
                            )
                        last_predictions = ()
                        last_tap = None
                        last_composer_state = None
    except KeyboardInterrupt:
        return 0
    except (sd.PortAudioError, OSError, RuntimeError) as exc:
        print(f"Application failed: {exc}", file=sys.stderr)
        return 1
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys
import textwrap
import time

import cv2
import numpy as np

from hand_tracking import draw_snapshot
from helper_air import AirClickDetector, AirKeyModel
from helper_camera_view import CameraView, add_view_arguments
from helper_fusion import TextComposer
from helper_keyboard import KeyboardCalibration, SpacedKeyboardLayout
from helper_vision import HandHistory, MediaPipeHandTracker
from helper_press import AccuracyCheck


def parse_args():
    parser = argparse.ArgumentParser(description="Laptop webcam air keyboard with spaced QWERTY keys")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", default="models/hand_landmarker.task")
    parser.add_argument("--calibration", default="config/air_keyboard_layout.json")
    parser.add_argument("--profile", default="config/air_key_model.json")
    parser.add_argument("--mode", choices=("movement", "pinch"), default="movement")
    parser.add_argument("--movement-profile", default="config/air_movement_model.json")
    parser.add_argument("--trained-only", action="store_true", help="Movement mode: require personal key examples")
    add_view_arguments(parser)
    return parser.parse_args()


def context_for(args, view, calibration):
    return {"layout": "spaced_ansi_v1", "camera": "phone:back" if getattr(args, "phone", False) else args.camera, "rotation": view.rotation,
            "mirrored": view.mirrored, "points": calibration.camera_points.tolist(),
            "flip_rows": calibration.flip_rows, "gesture": "index_pinch_v1"}


def backup(path):
    if path.exists():
        target = path.with_name(path.stem + ".backup_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + path.suffix)
        shutil.copy2(path, target)


def draw_layout(frame, layout, calibration, highlighted):
    height, width = frame.shape[:2]
    layer = frame.copy()
    polygons = []
    for key in layout.keys:
        polygon = np.asarray([(int(x * width), int(y * height))
                              for x, y in (calibration.map_to_image(*p) for p in key.corners)], np.int32)
        active = key.name in highlighted
        disabled = key.value in ("CONTROL", "ALT")
        cv2.fillConvexPoly(layer, polygon, (40, 190, 240) if active else (24, 28, 35))
        polygons.append((key, polygon, active, disabled))
    cv2.addWeighted(layer, 0.30, frame, 0.70, 0, frame)
    for key, polygon, active, disabled in polygons:
        color = (80, 90, 105) if disabled else (40, 235, 255) if active else (225, 230, 240)
        cv2.polylines(frame, [polygon], True, color, 2 if active else 1, cv2.LINE_AA)
        x, y = calibration.map_to_image(*key.center)
        scale = 0.30 if len(key.label) > 1 else 0.43
        (tw, th), _ = cv2.getTextSize(key.label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        cv2.putText(frame, key.label, (int(x * width) - tw // 2, int(y * height) + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
    for point in calibration.camera_points:
        cv2.circle(frame, (int(point[0] * width), int(point[1] * height)), 7, (70, 230, 170), -1)


def draw_panel(frame, composer, status, detail, fps, movement=False):
    if movement:
        from helper_keyboard_ui import draw_workspace
        return draw_workspace(frame, composer, status, detail, fps)
    height, width = frame.shape[:2]
    canvas = np.full((height + 245, width, 3), (22, 24, 30), np.uint8)
    canvas[:height] = frame
    lines = ["AIR KEYBOARD | Index points. Thumb and index pinch to click.",
             "Drag corners. Space pauses. F flips rows. T trains. V tests. C clears. Q quits.",
             status, detail,
             f"Tracking {fps:.1f} FPS | Shift {'ON' if composer.shift else 'off'} | Caps {'ON' if composer.caps_lock else 'off'}",
             composer.text[-150:].replace("\n", " | ")]
    y = height + 23
    for index, text in enumerate(lines):
        for line in textwrap.wrap(text, width=82)[:2]:
            cv2.putText(canvas, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.43,
                        (70, 230, 170) if index == 0 else (230, 230, 235), 1, cv2.LINE_AA)
            y += 21
    return canvas


def main():
    args = parse_args()
    if getattr(args, "mode", "pinch") == "movement":
        from movement_keyboard import run
        return run(args)
    view = CameraView.from_args(args, phone=False)
    if not Path(args.model).is_file():
        print("Hand model missing. Follow README installation.", file=sys.stderr)
        return 1
    layout = SpacedKeyboardLayout()
    calibration_path = Path(args.calibration)
    profile_path = Path(args.profile)
    calibration = KeyboardCalibration(((0.05, 0.28), (0.95, 0.28), (0.95, 0.72), (0.05, 0.72)),
                                      mirrored=view.mirrored)
    status = "Hold hands above the keyboard. Aim at a key center, then pinch."
    if calibration_path.exists():
        try:
            data = json.loads(calibration_path.read_text())
            if (data["camera"], data["rotation"], data["mirrored"]) != (args.camera, view.rotation, view.mirrored):
                raise ValueError("Saved layout uses a different camera view")
            calibration = KeyboardCalibration(data["points"], view.mirrored, data["flip_rows"])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            status = f"Using default layout: {exc}"
    model = AirKeyModel(layout, context_for(args, view, calibration))
    if profile_path.exists():
        try:
            model = AirKeyModel.load(profile_path, layout, model.context)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            status = str(exc) + ". Old training has not been changed."
    detector = AirClickDetector(layout, calibration, model)
    composer = TextComposer()
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        camera.release()
        print("Could not open laptop webcam. Close other camera programs.", file=sys.stderr)
        return 1
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_FPS, 30)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    window = "0Keys air keyboard"
    size = [640, 480]
    dragging = None
    paused = False
    training = False
    target = None
    pending = None
    suppress_until = 0.0
    layout_backed_up = False
    profile_backed_up = False
    check = None

    def save_check(complete):
        nonlocal check
        report = check.report(time.perf_counter())
        report["complete"] = complete
        report["mode"] = "air_index_pinch"
        output = Path("logs") / ("air_accuracy_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".json")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Air test: {report['correct']} correct, {report['missed']} missed, "
              f"{report['false_activations']} accidental. Results: {output}")
        check = None
        return report

    def changed_layout():
        nonlocal model, detector, pending, status, layout_backed_up
        context = context_for(args, view, calibration)
        calibration_path.parent.mkdir(parents=True, exist_ok=True)
        if not layout_backed_up:
            backup(calibration_path)
            layout_backed_up = True
        temporary = calibration_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(context, indent=2) + "\n")
        temporary.replace(calibration_path)
        model = AirKeyModel(layout, context)
        detector = AirClickDetector(layout, calibration, model)
        pending = None
        status = "Layout saved. Position training reset for this placement."

    def mouse(event, x, y, _flags, _data):
        nonlocal dragging, calibration, suppress_until, target, pending, status
        if check is not None:
            return
        if event == cv2.EVENT_LBUTTONUP and dragging is not None:
            dragging = None
            changed_layout()
            suppress_until = time.perf_counter() + 0.5
            return
        if y >= size[1] or x < 0 or y < 0:
            return
        point = (min(x / size[0], 1.0), min(y / size[1], 1.0))
        if event == cv2.EVENT_LBUTTONDOWN:
            distances = [np.linalg.norm((np.asarray(p) - point) * size) for p in calibration.camera_points]
            if min(distances) < 24:
                dragging = int(np.argmin(distances))
                detector.reset()
                pending = None
            elif training:
                key = layout.key_at(*calibration.map_to_keyboard(*point))
                if key and key.value not in ("CONTROL", "ALT"):
                    target = key
                    pending = None
                    detector.reset()
                    status = f"Training {key.label}. Aim at it and pinch once."
        elif event == cv2.EVENT_MOUSEMOVE and dragging is not None:
            points = calibration.camera_points.tolist()
            points[dragging] = point
            try:
                calibration = KeyboardCalibration(points, view.mirrored, calibration.flip_rows)
            except ValueError as exc:
                status = str(exc)

    frame_times = deque(maxlen=30)
    history = HandHistory()
    try:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(window, 1200, 1000)
        cv2.setMouseCallback(window, mouse)
        with MediaPipeHandTracker(args.model, input_mirrored=view.mirrored) as tracker:
            while True:
                ok, frame = camera.read()
                now = time.perf_counter()
                if not ok:
                    raise RuntimeError("Webcam stopped returning frames")
                frame = view.apply(frame)
                size[:] = [frame.shape[1], frame.shape[0]]
                snapshot = tracker.process(frame, now)
                history.append(snapshot)
                frame_times.append(now)
                fps = (len(frame_times) - 1) / max(frame_times[-1] - frame_times[0], 0.001)
                if paused or dragging is not None or pending is not None or now < suppress_until or (training and target is None):
                    detector.reset()
                    clicks = []
                else:
                    clicks = detector.update(snapshot)
                for click in clicks:
                    if training:
                        try:
                            trial = AirKeyModel(layout, model.context)
                            trial.add(target.name, click.point, click.handedness)
                            pending = click
                            status = f"Save example for {target.label}? Y confirms. N discards."
                        except ValueError as exc:
                            status = str(exc)
                    else:
                        composer.apply(click.key)
                        if check:
                            check.events.append((click.timestamp, click.key.name))
                        status = f"Typed {click.key.label}. Relative model confidence {click.confidence:.0%}"
                highlighted = {state.key.name for state in detector.states.values() if state.key is not None}
                if training and target:
                    highlighted.add(target.name)
                draw_layout(frame, layout, calibration, highlighted)
                draw_snapshot(frame, snapshot, history)
                details = []
                for (hand_id, side), state in detector.states.items():
                    details.append(f"{side}: {state.key.label if state.key else '-'} {state.phase}")
                detail = " | ".join(details) or "Show hands with thumbs, index fingers, and wrists visible"
                if training:
                    count = sum(item["key"] == target.name for item in model.samples) if target else 0
                    detail = (f"TRAINING {target.label}: {count} confirmed examples. Aim for five."
                              if target else "TRAINING: Click the desired key with the mouse first.")
                elif paused:
                    detail = "PAUSED. Space resumes. No microphone or table contact is used."
                elif not model.samples:
                    detail += " | No personal examples yet"
                if not snapshot.hands:
                    status = "No hands detected. Adjust the webcam angle and lighting."
                elif status.startswith("No hands detected"):
                    status = "Aim at a key center. Pinch only when its state says ready."
                if check:
                    prompt, expected = check.phase(now)
                    status = prompt.replace("Rest your fingers", "Hold your hands still").replace("Tap once", "Pinch once")
                    if expected:
                        status += ": " + expected.upper()
                    detail = "GUIDED TEST. One pinch per prompt. Q aborts. Misses count automatically."
                    if now >= check.started + check.duration:
                        report = save_check(True)
                        status = f"Test: {report['correct']} correct, {report['missed']} missed, {report['false_activations']} accidental."
                        paused = True
                canvas = draw_panel(frame, composer, status, detail, fps)
                cv2.imshow(window, canvas)
                key = cv2.waitKey(1) & 0xff
                if key in (ord("q"), 27):
                    return 0
                if check:
                    continue
                if key in (ord(" "), ord("t"), ord("c"), ord("f"), ord("y"), ord("n")):
                    detector.reset()
                    suppress_until = time.perf_counter() + 0.5
                if key == ord(" "):
                    paused = not paused
                    pending = None
                elif key == ord("c"):
                    composer.clear()
                elif key == ord("t"):
                    training = not training
                    pending = None
                    target = None
                    status = "Training enabled. Select a key with the mouse." if training else "Typing resumed."
                elif key == ord("v") and not training:
                    check = AccuracyCheck(["a", "s", "d", "f", "j", "k", "l", "space"] * 2,
                                          time.perf_counter() + 1)
                    paused = False
                    detector.reset()
                    suppress_until = time.perf_counter() + 1
                elif key == ord("f"):
                    calibration = KeyboardCalibration(calibration.camera_points, view.mirrored, not calibration.flip_rows)
                    changed_layout()
                elif key == ord("y") and pending is not None and target is not None:
                    model.add(target.name, pending.point, pending.handedness)
                    if not profile_backed_up:
                        backup(profile_path)
                        profile_backed_up = True
                    model.save(profile_path)
                    status = f"Saved {target.label}. Repeat or select another key. T returns to typing."
                    pending = None
                elif key == ord("n") and pending is not None:
                    pending = None
                    status = "Example discarded. Aim again and pinch."
    except KeyboardInterrupt:
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Air keyboard: {exc}", file=sys.stderr)
        return 1
    finally:
        camera.release()
        cv2.destroyAllWindows()
        if check:
            save_check(False)


if __name__ == "__main__":
    raise SystemExit(main())

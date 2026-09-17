from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from hand_tracking import draw_snapshot
from helper_keyboard import KeyboardCalibration, KeyboardLayout
from helper_vision import FINGERTIPS, HandHistory, MediaPipeHandTracker


CORNER_NAMES = ("top left", "top right", "bottom right", "bottom left")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate the virtual keyboard")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--model", default="models/hand_landmarker.task")
    parser.add_argument("--calibration", default="config/keyboard_calibration.json")
    parser.add_argument("--no-mirror", action="store_true")
    return parser.parse_args()


def to_pixel(point: tuple[float, float], width: int, height: int) -> tuple[int, int]:
    return int(point[0] * width), int(point[1] * height)


def draw_keyboard(
    frame,
    layout: KeyboardLayout,
    calibration: KeyboardCalibration,
    highlighted: set[str],
) -> None:
    height, width = frame.shape[:2]
    fill_layer = frame.copy()

    for key in layout.keys:
        polygon = np.asarray(
            [to_pixel(calibration.map_to_image(*point), width, height) for point in key.corners],
            dtype=np.int32,
        )
        if key.name in highlighted:
            cv2.fillConvexPoly(fill_layer, polygon, (40, 210, 255))

    cv2.addWeighted(fill_layer, 0.30, frame, 0.70, 0.0, frame)

    for key in layout.keys:
        polygon = np.asarray(
            [to_pixel(calibration.map_to_image(*point), width, height) for point in key.corners],
            dtype=np.int32,
        )
        color = (40, 230, 255) if key.name in highlighted else (220, 220, 220)
        cv2.polylines(frame, [polygon], True, color, 1, cv2.LINE_AA)
        center = to_pixel(calibration.map_to_image(*key.center), width, height)
        scale = 0.28 if len(key.label) > 2 else 0.38
        cv2.putText(
            frame,
            key.label,
            (center[0] - int(5 * len(key.label)), center[1] + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            1,
            cv2.LINE_AA,
        )


def highlighted_keys(snapshot, layout, calibration) -> set[str]:
    highlighted = set()
    for hand in snapshot.hands:
        for tip_index in FINGERTIPS:
            tip = hand.landmarks[tip_index]
            keyboard_x, keyboard_y = calibration.map_to_keyboard(tip.x, tip.y)
            key = layout.key_at(keyboard_x, keyboard_y)
            if key is not None:
                highlighted.add(key.name)
    return highlighted


def main() -> int:
    args = parse_args()
    mirrored = not args.no_mirror
    model_path = Path(args.model)
    calibration_path = Path(args.calibration)
    layout = KeyboardLayout()
    clicked_points: list[tuple[float, float]] = []
    calibration = None
    status_message = ""
    dragging_corner = None

    if not model_path.is_file():
        print(f"Model is missing: {model_path}", file=sys.stderr)
        return 1

    if calibration_path.is_file():
        try:
            loaded = KeyboardCalibration.load(calibration_path)
            if loaded.mirrored == mirrored:
                calibration = loaded
                status_message = "Loaded saved calibration"
            else:
                status_message = "Saved calibration uses another mirror setting"
        except (ValueError, KeyError, OSError) as exc:
            status_message = f"Could not load calibration: {exc}"

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        print(f"Could not open camera {args.camera}", file=sys.stderr)
        return 1
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    window_name = "0Keys keyboard calibration"
    latest_frame_size = [args.width, args.height]

    def mouse_callback(event, x, y, _flags, _data) -> None:
        nonlocal calibration, status_message, dragging_corner
        width, height = latest_frame_size
        point = (x / width, y / height)

        if calibration is not None:
            if event == cv2.EVENT_LBUTTONDOWN:
                distances = [
                    (x - corner[0] * width) ** 2 + (y - corner[1] * height) ** 2
                    for corner in calibration.camera_points
                ]
                nearest = int(np.argmin(distances))
                if distances[nearest] <= 35 ** 2:
                    dragging_corner = nearest
            elif event == cv2.EVENT_MOUSEMOVE and dragging_corner is not None:
                points = calibration.camera_points.tolist()
                points[dragging_corner] = point
                try:
                    calibration = KeyboardCalibration(
                        points,
                        mirrored=mirrored,
                        flip_rows=calibration.flip_rows,
                    )
                    status_message = "Release the corner to save"
                except ValueError:
                    pass
            elif event == cv2.EVENT_LBUTTONUP and dragging_corner is not None:
                calibration.save(calibration_path)
                dragging_corner = None
                status_message = f"Calibration saved to {calibration_path}"
            return

        if event != cv2.EVENT_LBUTTONDOWN:
            return
        clicked_points.append(point)
        if len(clicked_points) == 4:
            try:
                calibration = KeyboardCalibration(clicked_points, mirrored=mirrored)
                calibration.save(calibration_path)
                status_message = f"Calibration saved to {calibration_path}"
            except ValueError as exc:
                clicked_points.clear()
                status_message = f"Try again: {exc}"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(window_name, 1100, 825)
    cv2.setMouseCallback(window_name, mouse_callback)
    history = HandHistory()
    frame_times: deque[float] = deque(maxlen=30)

    try:
        with MediaPipeHandTracker(model_path, input_mirrored=mirrored) as tracker:
            while True:
                ok, frame = camera.read()
                capture_time = time.perf_counter()
                if not ok:
                    print("Camera stopped returning frames", file=sys.stderr)
                    return 1
                if mirrored:
                    frame = cv2.flip(frame, 1)

                latest_frame_size[:] = [frame.shape[1], frame.shape[0]]
                snapshot = tracker.process(frame, capture_time)
                history.append(snapshot)
                draw_snapshot(frame, snapshot, history)

                if calibration is not None:
                    active_keys = highlighted_keys(snapshot, layout, calibration)
                    draw_keyboard(frame, layout, calibration, active_keys)
                    for index, point in enumerate(calibration.camera_points):
                        pixel = to_pixel(tuple(point), frame.shape[1], frame.shape[0])
                        cv2.circle(frame, pixel, 9, (30, 230, 255), -1, cv2.LINE_AA)
                        cv2.putText(
                            frame,
                            str(index + 1),
                            (pixel[0] + 10, pixel[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (30, 230, 255),
                            2,
                            cv2.LINE_AA,
                        )
                    instruction = "Drag corners to resize. F flips rows. R starts again."
                else:
                    for index, point in enumerate(clicked_points):
                        pixel = to_pixel(point, frame.shape[1], frame.shape[0])
                        cv2.circle(frame, pixel, 7, (40, 230, 255), -1, cv2.LINE_AA)
                        cv2.putText(
                            frame,
                            str(index + 1),
                            (pixel[0] + 8, pixel[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (40, 230, 255),
                            2,
                            cv2.LINE_AA,
                        )
                    target = CORNER_NAMES[len(clicked_points)]
                    instruction = f"Click the {target} corner of your keyboard area"

                frame_times.append(capture_time)
                fps = 0.0
                if len(frame_times) > 1:
                    fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 70), (20, 20, 20), -1)
                cv2.putText(
                    frame,
                    f"Calibration | Hands: {len(snapshot.hands)} | FPS: {fps:.1f}",
                    (12, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (240, 240, 240),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    instruction,
                    (12, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.46,
                    (220, 220, 220),
                    1,
                    cv2.LINE_AA,
                )
                if status_message:
                    cv2.putText(
                        frame,
                        status_message,
                        (12, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (60, 220, 255),
                        1,
                        cv2.LINE_AA,
                    )

                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    return 0
                if key == ord("r"):
                    calibration = None
                    clicked_points.clear()
                    status_message = "Calibration reset"
                if key == ord("f") and calibration is not None:
                    calibration = KeyboardCalibration(
                        calibration.camera_points,
                        mirrored=mirrored,
                        flip_rows=not calibration.flip_rows,
                    )
                    calibration.save(calibration_path)
                    state = "flipped" if calibration.flip_rows else "normal"
                    status_message = f"Keyboard rows are now {state}"
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())

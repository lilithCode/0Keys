from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
import sys
import time

import cv2

from helper_vision import FINGERTIPS, HAND_CONNECTIONS, HandHistory, MediaPipeHandTracker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track hands and fingertips")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--model", default="models/hand_landmarker.task", help="MediaPipe model path"
    )
    parser.add_argument("--no-mirror", action="store_true")
    return parser.parse_args()


def draw_snapshot(frame, snapshot, history: HandHistory) -> None:
    height, width = frame.shape[:2]
    colors = ((80, 230, 120), (255, 170, 60), (230, 100, 230), (80, 210, 255))

    for hand in snapshot.hands:
        color = colors[(hand.hand_id - 1) % len(colors)]
        pixels = [
            (int(point.x * width), int(point.y * height))
            for point in hand.landmarks
        ]
        for start, end in HAND_CONNECTIONS:
            cv2.line(frame, pixels[start], pixels[end], color, 2, cv2.LINE_AA)
        for index, pixel in enumerate(pixels):
            radius = 7 if index in FINGERTIPS else 3
            cv2.circle(frame, pixel, radius, color, -1, cv2.LINE_AA)

        wrist_x, wrist_y = pixels[0]
        cv2.putText(
            frame,
            f"{hand.handedness} #{hand.hand_id}",
            (wrist_x + 8, wrist_y + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

        # A downward arrow shows movement toward the table.
        for tip_index in FINGERTIPS:
            velocity = history.tip_velocity(snapshot, hand.hand_id, tip_index)
            if velocity is None:
                continue
            tip_x, tip_y = pixels[tip_index]
            scale = 0.035
            arrow_end = (
                int(tip_x + velocity.x * width * scale),
                int(tip_y + velocity.y * height * scale),
            )
            cv2.arrowedLine(
                frame, (tip_x, tip_y), arrow_end, (40, 40, 255), 2, cv2.LINE_AA,
                tipLength=0.35,
            )


def main() -> int:
    args = parse_args()
    model_path = Path(args.model)
    if not model_path.is_file():
        print(f"Model is missing: {model_path}", file=sys.stderr)
        print("Download the hand landmarker model before running.", file=sys.stderr)
        return 1

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        print(
            f"Could not open camera {args.camera}. Try --camera 1 or close other camera apps.",
            file=sys.stderr,
        )
        return 1

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    camera.set(cv2.CAP_PROP_FPS, args.fps)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    history = HandHistory()
    frame_times: deque[float] = deque(maxlen=30)

    try:
        cv2.namedWindow("0Keys hand tracking", cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow("0Keys hand tracking", 1100, 825)
        with MediaPipeHandTracker(model_path=model_path, input_mirrored=not args.no_mirror) as tracker:
            while True:
                ok, frame = camera.read()
                capture_time = time.perf_counter()
                if not ok:
                    print("Camera stopped returning frames.", file=sys.stderr)
                    return 1
                if not args.no_mirror:
                    frame = cv2.flip(frame, 1)

                snapshot = tracker.process(frame, capture_time)
                history.append(snapshot)
                draw_snapshot(frame, snapshot, history)

                frame_times.append(capture_time)
                fps = 0.0
                if len(frame_times) > 1:
                    fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 64), (20, 20, 20), -1)
                cv2.putText(
                    frame,
                    f"Hands: {len(snapshot.hands)} | FPS: {fps:.1f}",
                    (14, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (240, 240, 240), 2, cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    "Keep both hands visible. Red arrows show motion. Q or Escape quits.",
                    (14, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (200, 200, 200), 1, cv2.LINE_AA,
                )
                cv2.imshow("0Keys hand tracking", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    return 0
    except (FileNotFoundError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())

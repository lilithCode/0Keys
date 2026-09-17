import argparse
import sys

import cv2

from helper_phone import PhoneCamera, check_phone
from helper_camera_view import CameraView, add_view_arguments
from helper_vision import HandHistory, MediaPipeHandTracker
from hand_tracking import draw_snapshot


def main():
    parser = argparse.ArgumentParser(description="Check and position the USB phone camera")
    parser.add_argument("--check", action="store_true", help="Check USB authorization and Android version")
    parser.add_argument("--probe", action="store_true", help="Read 60 frames without opening a window")
    add_view_arguments(parser)
    parser.add_argument("--model", default="models/hand_landmarker.task")
    args = parser.parse_args()
    camera = None
    tracker = None
    view = CameraView.from_args(args, phone=True)
    try:
        if args.check:
            info = check_phone()
            print(f"Ready: {info['model']}, Android API {info['sdk']}, scrcpy {info['version']}")
            print("Now run: python phone_camera.py")
            return 0
        camera = PhoneCamera()
        if not args.probe:
            tracker = MediaPipeHandTracker(args.model, input_mirrored=view.mirrored)
            history = HandHistory()
            cv2.namedWindow("0Keys phone camera", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("0Keys phone camera", 1100, 850)
        first = None
        frames = 0
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError(camera.error or "Phone stopped sending frames")
            frame = view.apply(frame)
            frames += 1
            if first is None:
                first = camera.timestamp
            fps = (frames - 1) / max(camera.timestamp - first, 0.001)
            if args.probe:
                if frames == 60:
                    print(f"USB video OK: {frame.shape[1]}x{frame.shape[0]}, {fps:.1f} received FPS")
                    print("This measures video delivery, not hand-tracking speed or true camera latency.")
                    return 0
                continue
            snapshot = tracker.process(frame, camera.timestamp)
            history.append(snapshot)
            draw_snapshot(frame, snapshot, history)
            lines = ["Rear camera over USB. Q closes this preview.",
                     "Keep both hands and wrists visible. Aim diagonally at the table.",
                     f"Rotation {view.rotation}. Mirror {'on' if view.mirrored else 'off'}. Tracking {fps:.1f} FPS.",
                     "Labels name your physical hands, not their side of the image."]
            for index, line in enumerate(lines):
                cv2.putText(frame, line, (10, 23 + index * 22), cv2.FONT_HERSHEY_SIMPLEX,
                            0.46, (70, 240, 150), 1, cv2.LINE_AA)
            cv2.imshow("0Keys phone camera", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                return 0
    except KeyboardInterrupt:
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Phone camera: {exc}", file=sys.stderr)
        return 1
    finally:
        if tracker:
            tracker.close()
        if camera:
            camera.release()
        if not args.check and not args.probe:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())

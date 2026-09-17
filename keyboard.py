import argparse
from pathlib import Path

from helper_camera_view import add_view_arguments
from movement_keyboard import run


def parse_args(argv=None):
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="0Keys: touch the table and lift to type with an overhead camera")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", default=str(root / "models/hand_landmarker.task"))
    parser.add_argument("--calibration", help="Layout file; phone and webcam use separate defaults")
    parser.add_argument("--movement-profile", help="Personal movement file; separate for phone and webcam")
    parser.add_argument("--trained-only", action="store_true", help="Disable basic detection for untrained hands")
    parser.add_argument("--record-landmarks", help="Optional new JSONL file for local replay, without camera images or audio")
    parser.add_argument("--phone", action="store_true",
                        help="Use the rear Android camera over USB, propped above the table")
    parser.add_argument("--resolution", default="640x360", help="Webcam capture size, such as 1280x720")
    parser.add_argument("--exposure", default="auto",
                        help="auto (default): the camera's own exposure, cleanest picture; fast: short exposure "
                             "for 30 FPS in a bright room; or a fixed V4L2 value such as 50. E switches live")
    parser.add_argument("--sensitivity", type=float, default=1.0,
                        help="Press sensitivity from 0.4 to 2.5. Higher detects smaller presses. Adjust live with - and +")
    add_view_arguments(parser)
    args = parser.parse_args(argv)
    prefix = "phone_" if args.phone else ""
    if args.calibration is None:
        args.calibration = str(root / f"config/{prefix}air_keyboard_layout.json")
    if args.movement_profile is None:
        args.movement_profile = str(root / f"config/{prefix}air_movement_model.json")
    return args


def main():
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())

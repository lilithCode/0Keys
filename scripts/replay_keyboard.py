import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import sys

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helper_keyboard import KeyboardCalibration, SpacedKeyboardLayout
from helper_finger_press import FingerPressDetector
from helper_camera_view import CameraView
from helper_motion import MovementModel, MovementRecorder, NaturalPressModel
from helper_vision import MediaPipeHandTracker, HandSnapshot, TrackedHand, Landmark


def snapshots_from_json(path):
    text = Path(path).read_text()
    data = json.loads(text) if text.lstrip().startswith("[") else [json.loads(line) for line in text.splitlines()]
    data = [frame for frame in data if "timestamp" in frame]
    return [HandSnapshot(frame["timestamp"], tuple(TrackedHand(hand["hand_id"], hand["handedness"],
            tuple(Landmark(**point) for point in hand["landmarks"])) for hand in frame["hands"]))
            for frame in data]


def main():
    parser = argparse.ArgumentParser(description="Offline keyboard diagnostics, without saving training")
    parser.add_argument("--video")
    parser.add_argument("--crop", nargs=4, type=int, metavar=("X", "Y", "WIDTH", "HEIGHT"))
    parser.add_argument("--snapshots", required=True)
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--rotation", type=int, choices=(0, 90, 180, 270), default=0,
                        help="Clockwise rotation when extracting a video")
    parser.add_argument("--mirror", action="store_true", help="Mirror the video before tracking")
    parser.add_argument("--calibration", default="config/air_keyboard_layout.json")
    parser.add_argument("--profile", default="config/air_movement_model.json")
    parser.add_argument("--model", default="models/hand_landmarker.task")
    args = parser.parse_args()
    data = json.loads(Path(args.calibration).read_text())
    aspect = 16 / 9  # Legacy recordings did not save their image dimensions.
    if not args.extract:
        first = Path(args.snapshots).read_text().splitlines()[0]
        if not first.lstrip().startswith("["):
            header = json.loads(first)
            if "context" in header:
                data = header["context"]
                aspect = header.get("aspect", aspect)
    if args.extract:
        if not args.video or args.fps <= 0:
            parser.error("Extraction needs a video and positive FPS")
        if Path(args.snapshots).exists():
            parser.error("Choose a new snapshots path to avoid overwriting a replay")
        cap = cv2.VideoCapture(args.video)
        rate = cap.get(cv2.CAP_PROP_FPS)
        if not cap.isOpened() or rate <= 0:
            parser.error("Cannot open recording")
        crop_rect = args.crop
        view = CameraView(args.rotation, args.mirror)
        data = {**data, "camera": "video:" + Path(args.video).name,
                "rotation": view.rotation, "mirrored": view.mirrored}
        frames = []
        try:
            with MediaPipeHandTracker(args.model, input_mirrored=view.mirrored) as tracker:
                index, next_time = 0, 0.0
                while cap.grab():
                    timestamp = index / rate
                    index += 1
                    if timestamp < next_time:
                        continue
                    next_time += 1 / args.fps
                    ok, frame = cap.retrieve()
                    if not ok:
                        raise ValueError("Cannot read video frame")
                    x, y, width, height = crop_rect or (0, 0, frame.shape[1], frame.shape[0])
                    if not ok or x < 0 or y < 0 or width <= 0 or height <= 0 or y + height > frame.shape[0] or x + width > frame.shape[1]:
                        raise ValueError("Invalid video frame or crop")
                    crop = view.apply(frame[y:y + height, x:x + width])
                    aspect = crop.shape[1] / crop.shape[0]
                    frames.append(asdict(tracker.process(crop, timestamp)))
        finally:
            cap.release()
        with Path(args.snapshots).open("x") as output:
            output.write(json.dumps({"context": data, "aspect": aspect}) + "\n")
            for frame in frames:
                output.write(json.dumps(frame) + "\n")
    frames = snapshots_from_json(args.snapshots)
    calibration = KeyboardCalibration(data["points"], data["mirrored"], data["flip_rows"])
    layout = SpacedKeyboardLayout()
    data["gesture"] = "whole_hand_movement_v1"
    try:
        personal, profile_note = MovementModel.load_for_view(args.profile, layout, data)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        personal = MovementModel(layout, data)
        profile_note = f"Basic detection only: {exc}"
    old = MovementRecorder(calibration)
    model = NaturalPressModel(layout, personal)
    new = FingerPressDetector(layout, calibration, personal, aspect=aspect)
    new_events = []
    counts, events, states = Counter(), [], Counter()
    for snapshot in frames:
        new_events.extend((round(snapshot.timestamp, 2), click.key.name) for click in new.update(snapshot))
        for movement in old.update(snapshot):
            key, reason = model.predict(movement)
            counts[reason] += 1
            if key:
                events.append((round(snapshot.timestamp, 2), key.name))
        states[old.status] += 1
    report = {"frames": len(frames), "frames_with_hands": sum(bool(f.hands) for f in frames),
              "valid_tracking_replay": any(f.hands for f in frames), "new_events": new_events,
              "old_reasons": counts, "old_events": events, "old_states": states,
              "aspect": aspect, "profile": profile_note,
              "note": "Detected events are not accuracy: intended keys and tap times are not labeled. "
                      "Screen recordings may also contain overlays that interfere with tracking."}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

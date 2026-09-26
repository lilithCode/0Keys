import json
from pathlib import Path

from helper_camera_view import CameraView
from helper_finger_press import typing_hands
from helper_keyboard import KeyboardCalibration
from helper_motion import MovementModel
from helper_vision import HandSnapshot

DEFAULT_POINTS = ((0.05, 0.28), (0.95, 0.28), (0.95, 0.72), (0.05, 0.72))
MASK_SECONDS = 1.5


def load_layout(args, phone):
    view = CameraView.from_args(args, phone=phone)
    calibration_path = Path(args.calibration)
    calibration = None
    notice = None
    if calibration_path.exists():
        try:
            data = json.loads(calibration_path.read_text())
            camera_id = "phone:back" if phone else args.camera
            if data["camera"] != camera_id:
                raise ValueError("Layout belongs to another camera")
            view = CameraView.from_args(args, phone=phone, saved=data)
            if (data["camera"], data["rotation"], data["mirrored"]) != (camera_id, view.rotation, view.mirrored):
                raise ValueError("Layout belongs to another camera view")
            calibration = KeyboardCalibration(data["points"], view.mirrored, data["flip_rows"])
        except (OSError, ValueError, TypeError, KeyError) as exc:
            notice = f"Default layout: {exc}"
    if calibration is None:
        calibration = KeyboardCalibration(DEFAULT_POINTS, view.mirrored)
    return view, calibration, notice


def load_movement_model(profile_path, layout, context):
    model = MovementModel(layout, context)
    notice = None
    if Path(profile_path).exists():
        try:
            model, notice = MovementModel.load_for_view(profile_path, layout, context)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            notice = f"Old movement profile not loaded: {exc}"
    return model, notice


class FalseHandMasker:
    def __init__(self):
        self.masks = []

    def clear(self):
        self.masks.clear()

    def prepare(self, frame, now):
        self.masks = [mask for mask in self.masks if mask[4] > now]
        if not self.masks:
            return frame
        tracked = frame.copy()
        for x0, y0, x1, y1, _ in self.masks:
            tracked[y0:y1, x0:x1] = 128
        return tracked

    def filter(self, snapshot, calibration, aspect, size, now):
        kept, ignored = typing_hands(snapshot.hands, calibration, aspect)
        for hand in ignored:
            xs = [p.x * size[0] for p in hand.landmarks]
            ys = [p.y * size[1] for p in hand.landmarks]
            pad_x, pad_y = 0.3 * (max(xs) - min(xs)) + 8, 0.3 * (max(ys) - min(ys)) + 8
            box = (max(0, int(min(xs) - pad_x)), max(0, int(min(ys) - pad_y)),
                   min(size[0], int(max(xs) + pad_x)), min(size[1], int(max(ys) + pad_y)))
            overlaps = any(box[0] < max(p.x for p in other.landmarks) * size[0]
                           and min(p.x for p in other.landmarks) * size[0] < box[2]
                           and box[1] < max(p.y for p in other.landmarks) * size[1]
                           and min(p.y for p in other.landmarks) * size[1] < box[3] for other in kept)
            if not overlaps:
                self.masks.append((*box, now + MASK_SECONDS))
        return HandSnapshot(snapshot.timestamp, tuple(kept)), bool(ignored)

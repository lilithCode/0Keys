from dataclasses import dataclass

import numpy as np


def add_view_arguments(parser):
    parser.add_argument("--rotation", type=int, choices=(0, 90, 180, 270),
                        help="Clockwise rotation. Keyboard restores its saved view; otherwise phone defaults to 180, webcam to 0")
    mirror = parser.add_mutually_exclusive_group()
    mirror.add_argument("--mirror", dest="no_mirror", action="store_false")
    mirror.add_argument("--no-mirror", dest="no_mirror", action="store_true")
    parser.set_defaults(no_mirror=None)


@dataclass(frozen=True)
class CameraView:
    rotation: int
    mirrored: bool

    def __post_init__(self):
        if self.rotation not in (0, 90, 180, 270) or type(self.mirrored) is not bool:
            raise ValueError("Camera view needs a quarter-turn rotation and a mirror setting")

    def rotate_preview(self, clockwise=True):
        # Mirroring reverses the apparent direction of a raw-frame rotation.
        step = 90 if clockwise else -90
        if self.mirrored:
            step = -step
        return CameraView((self.rotation + step) % 360, self.mirrored)

    def toggle_mirror(self):
        return CameraView(self.rotation, not self.mirrored)

    @classmethod
    def from_args(cls, args, phone, saved=None):
        default = cls(saved["rotation"], saved["mirrored"]) if saved is not None else cls(180 if phone else 0, True)
        rotation = args.rotation if args.rotation is not None else default.rotation
        mirrored = not args.no_mirror if args.no_mirror is not None else default.mirrored
        return cls(rotation, mirrored)

    def apply(self, frame):
        # Fix orientation before tracking so the overlay stays aligned.
        frame = np.rot90(frame, -(self.rotation // 90))
        if self.mirrored:
            frame = np.fliplr(frame)
        return np.ascontiguousarray(frame)

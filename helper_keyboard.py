from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


KEYBOARD_WIDTH = 15.0
KEYBOARD_HEIGHT = 5.0


@dataclass(frozen=True)
class KeyRegion:
    name: str
    label: str
    value: str
    x: float
    y: float
    width: float
    height: float = 1.0

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.width / 2, self.y + self.height / 2

    @property
    def corners(self) -> tuple[tuple[float, float], ...]:
        return (
            (self.x, self.y),
            (self.x + self.width, self.y),
            (self.x + self.width, self.y + self.height),
            (self.x, self.y + self.height),
        )


class KeyboardLayout:
    def __init__(self) -> None:
        self.width = KEYBOARD_WIDTH
        self.height = KEYBOARD_HEIGHT
        self.keys = tuple(self._build_keys())

    @staticmethod
    def _row(
        y: float, specs: Sequence[tuple[str, str, str, float]]
    ) -> list[KeyRegion]:
        keys = []
        x = 0.0
        for name, label, value, width in specs:
            keys.append(KeyRegion(name, label, value, x, y, width))
            x += width
        return keys

    def _build_keys(self) -> list[KeyRegion]:
        rows = []
        rows.extend(
            self._row(
                0.0,
                [(key, key, key, 1.0) for key in "1234567890"]
                + [
                    ("minus", "-", "-", 1.0),
                    ("equals", "=", "=", 1.0),
                    ("backspace", "Back", "BACKSPACE", 3.0),
                ],
            )
        )
        rows.extend(
            self._row(
                1.0,
                [("tab", "Tab", "TAB", 1.5)]
                + [(key.lower(), key, key.lower(), 1.0) for key in "QWERTYUIOP"]
                + [
                    ("left_bracket", "[", "[", 1.0),
                    ("right_bracket", "]", "]", 1.0),
                    ("backslash", "\\", "\\", 1.5),
                ],
            )
        )
        rows.extend(
            self._row(
                2.0,
                [("caps_lock", "Caps", "CAPS_LOCK", 1.75)]
                + [(key.lower(), key, key.lower(), 1.0) for key in "ASDFGHJKL"]
                + [
                    ("semicolon", ";", ";", 1.0),
                    ("apostrophe", "'", "'", 1.0),
                    ("enter", "Enter", "ENTER", 2.25),
                ],
            )
        )
        rows.extend(
            self._row(
                3.0,
                [("left_shift", "Shift", "SHIFT", 2.25)]
                + [(key.lower(), key, key.lower(), 1.0) for key in "ZXCVBNM"]
                + [
                    ("comma", ",", ",", 1.0),
                    ("period", ".", ".", 1.0),
                    ("question_mark", "/", "/", 1.0),
                    ("right_shift", "Shift", "SHIFT", 2.75),
                ],
            )
        )
        rows.extend(
            self._row(
                4.0,
                [
                    ("left_control", "Ctrl", "CONTROL", 1.5),
                    ("left_alt", "Alt", "ALT", 1.5),
                    ("space", "Space", " ", 9.0),
                    ("right_alt", "Alt", "ALT", 1.5),
                    ("right_control", "Ctrl", "CONTROL", 1.5),
                ],
            )
        )
        return rows

    def key_at(self, x: float, y: float) -> KeyRegion | None:
        if not 0.0 <= x < self.width or not 0.0 <= y < self.height:
            return None
        for key in self.keys:
            if key.x <= x < key.x + key.width and key.y <= y < key.y + key.height:
                return key
        return None


class SpacedKeyboardLayout(KeyboardLayout):
    def __init__(self, gap_x: float = 0.16, gap_y: float = 0.22):
        if not 0 <= gap_x < 0.5 or not 0 <= gap_y < 0.5:
            raise ValueError("Key gaps must be between zero and half a key")
        super().__init__()
        middle = [key for key in self.keys if 0 < key.y < 4]
        top = self._row(0, [("grave", "`", "`", 1)]
                        + [(key, key, key, 1) for key in "1234567890"]
                        + [("minus", "-", "-", 1), ("equals", "=", "=", 1),
                           ("backspace", "Back", "BACKSPACE", 2)])
        bottom = self._row(4, [("left_control", "Ctrl", "CONTROL", 1.25),
                              ("left_meta", "Win", "CONTROL", 1.25),
                              ("left_alt", "Alt", "ALT", 1.25),
                              ("space", "Space", " ", 6.25),
                              ("right_alt", "Alt", "ALT", 1.25),
                              ("right_meta", "Win", "CONTROL", 1.25),
                              ("menu", "Menu", "CONTROL", 1.25),
                              ("right_control", "Ctrl", "CONTROL", 1.25)])
        self.keys = tuple(replace(key, x=key.x + gap_x / 2, y=key.y + gap_y / 2,
                                  width=key.width - gap_x, height=key.height - gap_y)
                          for key in top + middle + bottom)


class KeyboardCalibration:
    def __init__(
        self,
        camera_points: Sequence[Sequence[float]],
        mirrored: bool = True,
        flip_rows: bool = False,
    ) -> None:
        points = np.asarray(camera_points, dtype=np.float32)
        if points.shape != (4, 2):
            raise ValueError("Calibration needs four camera points")
        if not np.all(np.isfinite(points)):
            raise ValueError("Calibration points must be finite")
        if np.any(points < 0.0) or np.any(points > 1.0):
            raise ValueError("Calibration points must be normalized")
        if not cv2.isContourConvex(points.reshape(-1, 1, 2)):
            raise ValueError("Calibration points must form a convex shape")
        if abs(float(cv2.contourArea(points))) < 0.02:
            raise ValueError("Calibration area is too small")
        top_y = float((points[0][1] + points[1][1]) / 2)
        bottom_y = float((points[2][1] + points[3][1]) / 2)
        left_x = float((points[0][0] + points[3][0]) / 2)
        right_x = float((points[1][0] + points[2][0]) / 2)
        if top_y >= bottom_y:
            raise ValueError("Click the upper corners before the lower corners")
        if left_x >= right_x:
            raise ValueError("Click the left corners on the left side")

        keyboard_points = np.asarray(
            [
                (0.0, 0.0),
                (KEYBOARD_WIDTH, 0.0),
                (KEYBOARD_WIDTH, KEYBOARD_HEIGHT),
                (0.0, KEYBOARD_HEIGHT),
            ],
            dtype=np.float32,
        )
        if flip_rows:
            keyboard_points = keyboard_points[[3, 2, 1, 0]]
        self.camera_points = points
        self.mirrored = mirrored
        self.flip_rows = flip_rows
        self.image_to_keyboard = cv2.getPerspectiveTransform(points, keyboard_points)
        self.keyboard_to_image = cv2.getPerspectiveTransform(keyboard_points, points)

    @staticmethod
    def _transform(matrix: np.ndarray, x: float, y: float) -> tuple[float, float]:
        result = matrix @ np.asarray([x, y, 1.0], dtype=np.float64)
        if abs(float(result[2])) < 1e-9:
            raise ValueError("Point cannot be transformed")
        return float(result[0] / result[2]), float(result[1] / result[2])

    def map_to_keyboard(self, image_x: float, image_y: float) -> tuple[float, float]:
        return self._transform(self.image_to_keyboard, image_x, image_y)

    def map_to_image(self, keyboard_x: float, keyboard_y: float) -> tuple[float, float]:
        return self._transform(self.keyboard_to_image, keyboard_x, keyboard_y)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1,
            "mirrored": self.mirrored,
            "flip_rows": self.flip_rows,
            "camera_points": self.camera_points.tolist(),
        }
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "KeyboardCalibration":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("version") != 1:
            raise ValueError("Unsupported calibration version")
        return cls(
            data["camera_points"],
            mirrored=bool(data.get("mirrored", True)),
            flip_rows=bool(data.get("flip_rows", False)),
        )

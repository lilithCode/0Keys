import textwrap

import cv2
import numpy as np

# The camera picture is enlarged on screen, and the controls and text sit in a
# panel beside it. A portrait phone view used to be a narrow strip, with the
# text box drawn over the hands.
PANEL_WIDTH = 420
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 860
MIN_HEIGHT = 640


def display_size(width, height):
    """Size of the camera picture on screen, keeping its shape."""
    scale = min(DISPLAY_WIDTH / width, DISPLAY_HEIGHT / height)
    return round(width * scale), round(height * scale)


def buttons(width, height):
    left = width + 10
    step = (PANEL_WIDTH - 20) // 2
    return [(name, control, (left + (index % 2) * step, 10 + (index // 2) * 46,
                             left + (index % 2 + 1) * step - 6, 50 + (index // 2) * 46))
            for index, (name, control) in enumerate((("TYPE [T]", "t"), ("LEARN [L]", "l"),
                                                    ("PAUSE", " "), ("CLEAR", "c"),
                                                    ("ROTATE LEFT [", "["), ("ROTATE RIGHT ]", "]"),
                                                    ("MIRROR [M]", "m"), ("FLIP ROWS [F]", "f")))]


def button_at(x, y, width, height):
    return next((control for _, control, (left, top, right, bottom) in buttons(width, height)
                 if left <= x <= right and top <= y <= bottom), None)


def text_box(canvas, text, rect):
    left, top, right, bottom = rect
    # Blend only the box itself; copying the whole enlarged canvas was slow.
    area = canvas[top:bottom + 1, left:right + 1]
    layer = area.copy()
    cv2.rectangle(layer, (0, 0), (layer.shape[1], layer.shape[0]), (15, 20, 28), -1)
    cv2.addWeighted(layer, 0.88, area, 0.12, 0, area)
    cv2.rectangle(canvas, (left, top), (right, bottom), (70, 200, 155), 1)
    cv2.putText(canvas, "YOUR TEXT", (left + 10, top + 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (70, 230, 170), 1, cv2.LINE_AA)
    lines = []
    current = ""
    for char in text[-1000:]:
        if char == "\n":
            lines.append(current)
            current = ""
            continue
        if cv2.getTextSize(current + char, cv2.FONT_HERSHEY_SIMPLEX, 0.60, 1)[0][0] > right - left - 24:
            lines.append(current)
            current = ""
        current += char
    lines.append(current)
    maximum = max(1, (bottom - top - 30) // 26)
    if not text:
        lines = ["Your typed text appears here"]
    for index, line in enumerate(lines[-maximum:]):
        cv2.putText(canvas, line, (left + 10, top + 46 + index * 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.60, (235, 240, 245) if text else (150, 160, 175), 1, cv2.LINE_AA)


def draw_workspace(frame, composer, status, detail, fps):
    height, width = frame.shape[:2]
    canvas = np.zeros((max(height, MIN_HEIGHT), width + PANEL_WIDTH, 3), np.uint8)
    # OpenCV fills colour far faster than NumPy broadcasting on a large canvas.
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], canvas.shape[0]), (22, 24, 30), -1)
    canvas[:height, :width] = frame
    left = width + 10
    for label, _, (x0, y0, x1, y1) in buttons(width, height):
        cv2.rectangle(canvas, (x0, y0), (x1, y1), (60, 75, 85), -1)
        label_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
        scale = 0.5 * min(1, (x1 - x0 - 16) / max(label_width, 1))
        cv2.putText(canvas, label, (x0 + 8, y0 + 26), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (240, 240, 240), 1, cv2.LINE_AA)
    y = 222
    for text, color, rows in ((status, (70, 230, 170), 3), (detail, (230, 230, 235), 5)):
        for line in textwrap.wrap(text, width=52)[:rows]:
            cv2.putText(canvas, line, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            y += 20
        y += 8
    rate = fps if isinstance(fps, str) else f"{fps:.1f} FPS"
    for line in textwrap.wrap(f"{rate} | -/+ sensitivity | F flip rows | Q quit", width=58)[:2]:
        cv2.putText(canvas, line, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 190, 200), 1, cv2.LINE_AA)
        y += 18
    top = max(y + 10, 430)
    bottom = canvas.shape[0] - 10
    text_box(canvas, composer.text, (left, top, width + PANEL_WIDTH - 10, bottom))
    # An accidental Caps press silently turned later letters into capitals.
    flags = [name for name, on in (("CAPS LOCK", composer.caps_lock), ("SHIFT", composer.shift)) if on]
    if flags:
        label = " + ".join(flags) + " ON"
        label_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0]
        cv2.putText(canvas, label, (width + PANEL_WIDTH - 20 - label_width, top + 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (60, 170, 255), 1, cv2.LINE_AA)
    return canvas

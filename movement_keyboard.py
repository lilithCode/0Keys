from collections import deque
from contextlib import ExitStack
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

from hand_tracking import draw_snapshot
from helper_fusion import TextComposer
from helper_gesture import SETTLE_SECONDS, StartSign
from helper_finger_press import FingerPressDetector
from helper_keyboard import KeyboardCalibration, SpacedKeyboardLayout
from helper_keyboard_ui import button_at, display_size
from helper_motion import GuidedTraining, MovementModel, MovementRecorder, NaturalPressModel, NO_KEY
from helper_session import FalseHandMasker, load_layout, load_movement_model
from helper_vision import HandHistory, MediaPipeHandTracker
from helper_webcam import PhoneStream, WebcamStream

PHASE_COLORS = {"rest": (50, 240, 250), "warmup": (150, 160, 170), "moving": (60, 170, 255),
                "blocked": (120, 120, 235), "aiming": (200, 180, 70), "release": (80, 230, 130)}


def run(args):
    from air_keyboard import backup, context_for, draw_layout, draw_panel

    if not Path(args.model).is_file():
        print("Hand model missing. Follow README installation.", file=sys.stderr)
        return 1
    layout = SpacedKeyboardLayout()
    phone = bool(getattr(args, "phone", False))
    calibration_path = Path(args.calibration)
    profile_path = Path(args.movement_profile)
    view, calibration, notice = load_layout(args, phone)
    notice = notice or "Rest briefly, reach to a key, touch the table, then lift your finger"

    def context():
        data = context_for(args, view, calibration)
        data["gesture"] = "whole_hand_movement_v1"
        return data

    model, loaded = load_movement_model(profile_path, layout, context())
    notice = loaded or notice
    recorder = MovementRecorder(calibration)
    sensitivity = min(max(float(getattr(args, "sensitivity", 1.0) or 1.0), 0.4), 2.5)
    press_detector = FingerPressDetector(layout, calibration, model, sensitivity=sensitivity)
    predictor = model if getattr(args, "trained_only", False) else NaturalPressModel(layout, model)
    composer = TextComposer()
    history = HandHistory()
    training = None
    selecting = False
    paused = bool(getattr(args, "start_sign", False))
    start_sign = StartSign()
    settling = False
    dragging = None
    size = [640, 480]
    shown = [640, 480]
    window_sized = False
    frame_times = deque(maxlen=30)
    backed_up = set()
    batch_start = len(model.samples)
    last_key = None
    last_click = -10.0
    diagnostic = None
    exposure_modes = ["auto", "fast"]
    exposure_mode = str(getattr(args, "exposure", "auto") or "auto").lower()
    if exposure_mode not in exposure_modes:
        exposure_modes.insert(0, exposure_mode)
    hands_seen = None
    view_changed = False
    masker = FalseHandMasker()

    def reset_input():
        recorder.reset()
        press_detector.reset()

    def preserve(path):
        if path not in backed_up:
            backup(path)
            backed_up.add(path)

    def save_model():
        preserve(profile_path)
        model.save(profile_path)

    def changed_layout():
        nonlocal model, recorder, predictor, press_detector, training, notice, batch_start
        preserve(calibration_path)
        calibration_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = calibration_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(context(), indent=2) + "\n")
        temporary.replace(calibration_path)
        model = MovementModel(layout, context())
        if profile_path.exists():
            try:
                model, notice = MovementModel.load_for_view(profile_path, layout, context())
            except (OSError, ValueError, KeyError, TypeError):
                pass
        predictor = model if getattr(args, "trained_only", False) else NaturalPressModel(layout, model)
        recorder = MovementRecorder(calibration)
        press_detector = FingerPressDetector(layout, calibration, model, sensitivity=sensitivity)
        training = None
        batch_start = len(model.samples)
        notice = "Layout saved. Basic detection ready; old key labels are not moved to new keys"

    def begin(label):
        nonlocal training, selecting, paused, batch_start
        batch_start = len(model.samples)
        training = GuidedTraining(label, time.perf_counter(), repetitions=3 if label == NO_KEY else 5)
        selecting = True
        paused = False
        reset_input()

    def action(control):
        nonlocal selecting, training, paused, calibration, notice, last_key, sensitivity, exposure_mode
        nonlocal view, view_changed, dragging, history, hands_seen
        if control in ("[", "]", "m"):
            if diagnostic is not None:
                notice = "Finish the landmark recording before rotating or mirroring the view"
                return
            view = view.toggle_mirror() if control == "m" else view.rotate_preview(clockwise=control == "]")
            calibration = KeyboardCalibration(calibration.camera_points, view.mirrored, calibration.flip_rows)
            dragging = None
            selecting = False
            paused = True
            history = HandHistory()
            hands_seen = None
            last_key = None
            frame_times.clear()
            changed_layout()
            view_changed = True
            masker.clear()
            notice = "View saved. Adjust the green corners if needed, then click TYPE or press T"
        elif control == "e" and camera is not None:
            exposure_mode = exposure_modes[(exposure_modes.index(exposure_mode) + 1) % len(exposure_modes)]
            notice = "Camera: " + camera.set_exposure(exposure_mode)
        elif control in ("-", "=", "+"):
            sensitivity = min(max(sensitivity * (0.85 if control == "-" else 1 / 0.85), 0.4), 2.5)
            press_detector.sensitivity = sensitivity
            notice = f"Sensitivity {sensitivity:.2f}. Higher catches smaller presses, lower ignores more motion"
        elif control == "t":
            selecting = False
            training = None
            paused = False
            notice = "Type mode. T always returns here; L opens optional learning"
            reset_input()
        elif control == "l":
            selecting = True
            training = None
            paused = False
            reset_input()
        elif control == "n":
            begin(NO_KEY)
        elif control == "r" and training is not None:
            label = training.label
            model.samples = model.samples[:batch_start]
            save_model()
            begin(label)
        elif control == " ":
            paused = not paused
            training = None
            reset_input()
        elif control == "c":
            composer.clear()
            last_key = None
            reset_input()
        elif control == "f" and (training is None or training.done):
            if diagnostic is not None:
                notice = "Finish the landmark recording before changing the layout"
                return
            calibration = KeyboardCalibration(calibration.camera_points, view.mirrored, not calibration.flip_rows)
            changed_layout()

    def key_polygon(key):
        return np.asarray([(int(x * shown[0]), int(y * shown[1]))
                           for x, y in (calibration.map_to_image(*p) for p in key.corners)], np.int32)

    def mouse(event, x, y, _flags, _data):
        nonlocal dragging, calibration, notice
        if event == cv2.EVENT_LBUTTONDOWN:
            control = button_at(x, y, shown[0], shown[1])
            if control is not None:
                action(control)
                return
        if training is not None and not training.done:
            return
        if event == cv2.EVENT_LBUTTONUP and dragging is not None:
            dragging = None
            changed_layout()
            return
        if not 0 <= y < shown[1] or not 0 <= x < shown[0]:
            return
        point = (x / shown[0], y / shown[1])
        if event == cv2.EVENT_LBUTTONDOWN:
            distances = [np.linalg.norm((np.asarray(p) - point) * shown) for p in calibration.camera_points]
            if min(distances) < 28:
                if diagnostic is not None:
                    notice = "Finish the landmark recording before changing the layout"
                    return
                dragging = int(np.argmin(distances))
                reset_input()
            elif selecting:
                target = layout.key_at(*calibration.map_to_keyboard(*point))
                if target is not None and target.value not in ("CONTROL", "ALT"):
                    begin(target.name)
        elif event == cv2.EVENT_MOUSEMOVE and dragging is not None:
            points = calibration.camera_points.tolist()
            points[dragging] = point
            try:
                calibration = KeyboardCalibration(points, view.mirrored, calibration.flip_rows)
            except ValueError as exc:
                notice = str(exc)

    camera = None
    window = "0Keys movement keyboard"
    try:
        if getattr(args, "record_landmarks", None):
            diagnostic_path = Path(args.record_landmarks)
            diagnostic_path.parent.mkdir(parents=True, exist_ok=True)
            diagnostic = diagnostic_path.open("x", encoding="utf-8")
        if phone:
            camera = PhoneStream()
            print(f"Using {camera.note}. Keep the phone fixed above the table.")
        else:
            width, height = (getattr(args, "resolution", None) or "640x360").lower().split("x")
            camera = WebcamStream(args.camera, width=int(width), height=int(height), exposure=exposure_mode)
        cv2.namedWindow(window, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.setMouseCallback(window, mouse)
        with ExitStack() as tracker_context:
            tracker = tracker_context.enter_context(MediaPipeHandTracker(args.model, input_mirrored=view.mirrored))
            while True:
                ok, frame, now = camera.read()
                if not ok:
                    raise RuntimeError("Webcam stopped returning frames")
                if view_changed:
                    tracker_context.close()
                    tracker = tracker_context.enter_context(MediaPipeHandTracker(args.model, input_mirrored=view.mirrored))
                    view_changed = False
                frame = view.apply(frame)
                size[:] = [frame.shape[1], frame.shape[0]]
                press_detector.aspect = size[0] / size[1]
                if diagnostic is not None and diagnostic.tell() == 0:
                    diagnostic.write(json.dumps({"context": context(), "aspect": press_detector.aspect}) + "\n")
                snapshot = tracker.process(masker.prepare(frame, now), now)
                if diagnostic is not None:
                    diagnostic.write(json.dumps(asdict(snapshot)) + "\n")
                snapshot, ignored = masker.filter(snapshot, calibration, press_detector.aspect, size, now)
                if ignored:
                    notice = "Ignored a false hand outside the keyboard so both real hands can be tracked"
                history.append(snapshot)
                if start_sign.update(snapshot.hands, now, press_detector.aspect) and paused and dragging is None:
                    paused, selecting, training, settling = False, False, None, True
                    notice = "Start sign seen. Lower your hand onto the keys and type"
                settling = settling and start_sign.seen is not None and now - start_sign.seen < SETTLE_SECONDS
                frame_times.append(now)
                fps = (len(frame_times) - 1) / max(frame_times[-1] - frame_times[0], 0.001)
                if paused or dragging is not None or settling:
                    reset_input()
                elif training is not None and not training.done:
                    if training.update(snapshot, recorder, model):
                        save_model()
                        notice = "Example saved automatically. No confirmation key needed"
                elif selecting:
                    reset_input()
                elif not getattr(args, "trained_only", False):
                    presses = press_detector.update(snapshot)
                    for press in presses:
                        composer.apply(press.key)
                        last_key, last_click = press.key, now
                        notice = f"Pressed {press.key.label}. {press_detector.events} presses accepted"
                else:
                    movements = recorder.update(snapshot)
                    matches = []
                    for movement in movements:
                        try:
                            key, notice = predictor.predict(movement)
                        except ValueError as exc:
                            notice = str(exc)
                            continue
                        if key is not None:
                            matches.append(key)
                    if len(matches) == 1 and now - last_click >= 0.45:
                        composer.apply(matches[0])
                        last_key, last_click = matches[0], now
                    elif len(matches) > 1:
                        notice = "Two matching hand movements at once. Ignored for safety"
                shown[:] = display_size(size[0], size[1])
                display = cv2.resize(frame, tuple(shown), interpolation=cv2.INTER_LINEAR)
                draw_layout(display, layout, calibration, {last_key.name} if last_key and now - last_click < 0.5 else set())
                if selecting and training is not None and training.label != NO_KEY:
                    target = next(k for k in layout.keys if k.name == training.label)
                    cv2.polylines(display, [key_polygon(target)], True, (255, 140, 50), 3, cv2.LINE_AA)
                typing = not selecting and not paused and not getattr(args, "trained_only", False)
                if typing:
                    for (hand_id, tip), key in press_detector.hover.items():
                        phase = press_detector.phases.get((hand_id, tip))
                        if key is not None and (tip != 4 or phase == "moving"):
                            cv2.polylines(display, [key_polygon(key)], True,
                                          PHASE_COLORS.get(phase, PHASE_COLORS["rest"]), 2, cv2.LINE_AA)
                draw_snapshot(display, snapshot, history)
                if paused or settling:
                    progress = 1.0 if settling else start_sign.progress(now)
                    banner = ("Lower your hand onto the keys..." if settling else
                              "Hold up the start sign to type: index finger up, other fingers folded")
                    cv2.rectangle(display, (0, 0), (shown[0], 48), (22, 24, 30), -1)
                    cv2.rectangle(display, (0, 42), (int(shown[0] * progress), 48), (70, 230, 170), -1)
                    scale = min(0.62, 0.62 * (shown[0] - 20) / cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 1)[0][0])
                    cv2.putText(display, banner, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, scale, (240, 240, 240), 1, cv2.LINE_AA)
                if typing:
                    for hand in snapshot.hands:
                        for tip in (4, 8, 12, 16, 20):
                            name = (hand.hand_id, tip)
                            key = press_detector.hover.get(name)
                            color = PHASE_COLORS.get(press_detector.phases.get(name), PHASE_COLORS["rest"])
                            point = hand.landmarks[tip]
                            x = max(5, min(shown[0] - 60, int(point.x * shown[0]) + 10))
                            y = max(20, min(shown[1] - 8, int(point.y * shown[1]) - 10))
                            cv2.putText(display, key.label if key else "gap", (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.55, color, 1, cv2.LINE_AA)
                            level = min(press_detector.levels.get(name, 0.0), 1.0)
                            cv2.rectangle(display, (x, y + 5), (x + 32, y + 9), (40, 45, 55), -1)
                            cv2.rectangle(display, (x, y + 5), (x + int(32 * level), y + 9), color, -1)
                if paused:
                    status = "PAUSED | Show the start sign, click TYPE or press T to type"
                elif selecting:
                    status = "LEARN | " + (training.prompt(now) if training else "Click a key to train it. T returns to typing")
                else:
                    status = "TYPE MODE | " + (recorder.status if getattr(args, "trained_only", False) else press_detector.status)
                orientation = "Landscape" if size[0] >= size[1] else "Portrait"
                status = f"{orientation} {view.rotation}deg / Mirror {'on' if view.mirrored else 'off'} | " + status
                if selecting and training:
                    detail = f"Training {training.label.upper()}: {training.saved} saved. R retries this batch. T stops."
                    if training.message and now >= training.ready_at and not training.done:
                        detail = training.message + " | " + detail
                else:
                    trained = sum(any(model.count(k.name, h) >= 3 and model.count(NO_KEY, h) >= 3
                                      for h in ("Left", "Right")) for k in layout.keys)
                    detail = (f"{trained} trained keys | {notice}" if getattr(args, "trained_only", False)
                              else f"Sensitivity {sensitivity:.2f} (- / +) | {trained} trained keys | {notice}")
                if snapshot.hands:
                    hands_seen = now
                if not snapshot.hands:
                    status += " | No hands in the camera picture"
                    if hands_seen is None or now - hands_seen > 2.5:
                        detail = (("Point the phone down at your hands and keep it still | " if phone else
                                   "The laptop camera is probably facing you, not your hands. Tilt the screen "
                                   "down until your hands fill the picture, or run keyboard.py --phone | ")
                                  + detail)
                elif any(not 0.01 <= hand.landmarks[0].y <= 0.99 or not 0.01 <= hand.landmarks[0].x <= 0.99
                         for hand in snapshot.hands):
                    detail = ("Palms outside the picture: tracking recovers faster when whole hands show | "
                              + detail)
                elif camera.dark:
                    detail = "Too dark for fast exposure: add light, or start with --exposure auto | " + detail
                elif len(frame_times) > 5 and fps < 12:
                    detail += " | LOW FPS: use slow presses, fast typing may be missed"
                fps_note = f"{fps:.1f} FPS, camera {camera.fps:.0f} FPS, {camera.note} (E changes)"
                canvas = draw_panel(display, composer, status, detail, fps_note, movement=True)
                if not window_sized:
                    fit = min(1.0, 1800 / canvas.shape[1], 940 / canvas.shape[0])
                    cv2.resizeWindow(window, int(canvas.shape[1] * fit), int(canvas.shape[0] * fit))
                    window_sized = True
                cv2.imshow(window, canvas)
                control = cv2.waitKey(1) & 0xff
                if control in (ord("q"), 27):
                    return 0
                action(chr(control))
    except KeyboardInterrupt:
        return 0
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        print(f"Movement keyboard: {exc}", file=sys.stderr)
        return 1
    finally:
        if diagnostic is not None:
            diagnostic.close()
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()

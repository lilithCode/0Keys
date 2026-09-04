from __future__ import annotations

import argparse
import queue
import sys
import time

import numpy as np
import sounddevice as sd

from helper_audio import AudioTapDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect table taps from a microphone")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--device", help="Input device number or exact device name")
    parser.add_argument("--sample-rate", type=int, default=48_000)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--calibration-seconds", type=float, default=2.5)
    parser.add_argument("--threshold-db", type=float, default=14.0)
    return parser.parse_args()


def normalize_device(value: str | None) -> int | str | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return value


def main() -> int:
    args = parse_args()
    if args.list_devices:
        try:
            print(sd.query_devices())
            return 0
        except (sd.PortAudioError, OSError) as exc:
            print(f"Could not query audio devices: {exc}", file=sys.stderr)
            return 1

    detector = AudioTapDetector(
        sample_rate=args.sample_rate,
        calibration_seconds=args.calibration_seconds,
        threshold_db=args.threshold_db,
    )
    audio_blocks: queue.Queue[tuple[np.ndarray, float]] = queue.Queue(maxsize=64)
    dropped_blocks = 0

    def callback(indata, _frames, time_info, status) -> None:
        nonlocal dropped_blocks
        if status:
            dropped_blocks += 1
        # Use one clock so audio can match camera frames later.
        host_now = time.perf_counter()
        first_sample_time = host_now + (
            time_info.inputBufferAdcTime - time_info.currentTime
        )
        try:
            audio_blocks.put_nowait((indata[:, 0].copy(), first_sample_time))
        except queue.Full:
            dropped_blocks += 1

    device = normalize_device(args.device)
    print("Keep still and quiet during calibration, then tap the table.")
    print("Press Ctrl+C to stop.\n")

    try:
        with sd.InputStream(
            device=device,
            samplerate=args.sample_rate,
            blocksize=args.block_size,
            channels=1,
            dtype="float32",
            latency="low",
            callback=callback,
        ):
            last_status_time = 0.0
            while True:
                try:
                    block, block_time = audio_blocks.get(timeout=1.0)
                except queue.Empty:
                    print("No audio blocks received for 1 second.", file=sys.stderr)
                    continue
                event = detector.process_block(block, block_time)
                now = time.perf_counter()

                if not detector.calibrated and now - last_status_time >= 0.1:
                    percent = int(detector.calibration_progress * 100)
                    print(f"\rCalibrating ambient noise: {percent:3d}%", end="", flush=True)
                    last_status_time = now
                elif detector.calibrated and last_status_time != -1.0:
                    print(
                        f"\rReady, noise floor {detector.noise_floor_db:.1f} dBFS"
                        + " " * 12
                    )
                    last_status_time = -1.0

                if event is not None:
                    print(
                        f"TAP  time={event.timestamp:.6f}  "
                        f"strength={event.strength_db:.1f} dB"
                    )
    except KeyboardInterrupt:
        print(f"\nStopped. Dropped or status blocks: {dropped_blocks}")
        return 0
    except (sd.PortAudioError, OSError) as exc:
        print(f"Audio input failed: {exc}", file=sys.stderr)
        print(
            "Run 'python tap_detection.py --list-devices', then pass "
            "'--device NUMBER'.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

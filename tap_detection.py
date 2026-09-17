from __future__ import annotations

import argparse
import sys

import sounddevice as sd

from helper_audio import AudioTapDetector, MicrophoneInput


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
    device = normalize_device(args.device)
    microphone = MicrophoneInput(
        device=device,
        sample_rate=args.sample_rate,
        block_size=args.block_size,
    )
    print("Keep still and quiet during calibration, then tap the table.")
    print("Press Ctrl+C to stop.\n")

    try:
        with microphone:
            last_status_time = 0.0
            while True:
                block = microphone.get(timeout=1.0)
                if block is None:
                    print("No audio blocks received for 1 second.", file=sys.stderr)
                    continue
                event = detector.process_block(block.samples, block.first_sample_time)
                now = block.first_sample_time

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
        print(f"\nStopped. Dropped or status blocks: {microphone.dropped_blocks}")
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

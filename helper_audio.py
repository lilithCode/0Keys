from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import queue
import time

import numpy as np
import sounddevice as sd
from scipy.signal import butter, sosfilt


@dataclass(frozen=True)
class TapEvent:
    timestamp: float
    strength_db: float
    peak_db: float
    noise_floor_db: float


@dataclass(frozen=True)
class AudioBlock:
    samples: np.ndarray
    first_sample_time: float


class MicrophoneInput:
    def __init__(
        self,
        device: int | str | None = None,
        sample_rate: int = 48_000,
        block_size: int = 256,
    ) -> None:
        self.device = device
        self.sample_rate = sample_rate
        self.block_size = block_size
        self.dropped_blocks = 0
        self._blocks: queue.Queue[AudioBlock] = queue.Queue(maxsize=64)
        self._stream = None

    def _callback(self, indata, _frames, time_info, status) -> None:
        if status:
            self.dropped_blocks += 1
        host_now = time.perf_counter()
        first_sample_time = host_now + (
            time_info.inputBufferAdcTime - time_info.currentTime
        )
        block = AudioBlock(indata[:, 0].copy(), first_sample_time)
        try:
            self._blocks.put_nowait(block)
        except queue.Full:
            self.dropped_blocks += 1

    def __enter__(self) -> "MicrophoneInput":
        self._stream = sd.InputStream(
            device=self.device,
            samplerate=self.sample_rate,
            blocksize=self.block_size,
            channels=1,
            dtype="float32",
            latency="low",
            callback=self._callback,
        )
        self._stream.start()
        return self

    def __exit__(self, *_args) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def get(self, timeout: float | None = None) -> AudioBlock | None:
        try:
            return self._blocks.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain(self, maximum_blocks: int = 128) -> list[AudioBlock]:
        blocks = []
        for _ in range(maximum_blocks):
            try:
                blocks.append(self._blocks.get_nowait())
            except queue.Empty:
                break
        return blocks


class AudioTapDetector:
    def __init__(
        self,
        sample_rate: int = 48_000,
        low_hz: float = 600.0,
        high_hz: float = 8_000.0,
        filter_order: int = 4,
        calibration_seconds: float = 2.5,
        threshold_db: float = 14.0,
        minimum_rise_db: float = 5.0,
        minimum_crest_db: float = 4.0,
        refractory_seconds: float = 0.11,
        noise_history_seconds: float = 3.0,
    ) -> None:
        if not 0 < low_hz < high_hz < sample_rate / 2:
            raise ValueError("Filter cutoffs must satisfy 0 < low < high < Nyquist")

        self.sample_rate = sample_rate
        self.calibration_samples = int(calibration_seconds * sample_rate)
        self.threshold_db = threshold_db
        self.minimum_rise_db = minimum_rise_db
        self.minimum_crest_db = minimum_crest_db
        self.refractory_seconds = refractory_seconds

        self._sos = butter(
            filter_order,
            [low_hz, high_hz],
            btype="bandpass",
            fs=sample_rate,
            output="sos",
        )
        self._filter_state = np.zeros((self._sos.shape[0], 2), dtype=np.float64)
        self._noise_history_seconds = noise_history_seconds
        self._noise_db: deque[float] = deque()
        self._samples_seen = 0
        self._previous_peak_db = -120.0
        self._last_tap_time = -math.inf
        self._noise_floor_db = -90.0
        self._history_configured = False

    @staticmethod
    def _to_db(value: float) -> float:
        return 20.0 * math.log10(max(value, 1e-12))

    @property
    def calibrated(self) -> bool:
        return self._samples_seen >= self.calibration_samples

    @property
    def calibration_progress(self) -> float:
        if self.calibration_samples == 0:
            return 1.0
        return min(1.0, self._samples_seen / self.calibration_samples)

    @property
    def noise_floor_db(self) -> float:
        return self._noise_floor_db

    def process_block(
        self, samples: np.ndarray, first_sample_time: float
    ) -> TapEvent | None:
        block = np.asarray(samples, dtype=np.float64).reshape(-1)
        if block.size == 0:
            return None
        if not np.all(np.isfinite(block)):
            block = np.nan_to_num(block)

        if not self._history_configured:
            blocks_per_history = max(
                8,
                int(self._noise_history_seconds * self.sample_rate / block.size),
            )
            self._noise_db = deque(maxlen=blocks_per_history)
            self._history_configured = True

        filtered, self._filter_state = sosfilt(
            self._sos, block, zi=self._filter_state
        )
        absolute = np.abs(filtered)
        peak_index = int(np.argmax(absolute))
        peak_db = self._to_db(float(absolute[peak_index]))
        rms_db = self._to_db(float(np.sqrt(np.mean(filtered * filtered))))
        crest_db = peak_db - rms_db
        rise_db = peak_db - self._previous_peak_db
        self._previous_peak_db = peak_db

        was_calibrated = self.calibrated
        self._samples_seen += block.size

        if not was_calibrated:
            self._noise_db.append(rms_db)
            self._update_noise_floor()
            return None

        peak_time = first_sample_time + peak_index / self.sample_rate
        strength_db = peak_db - self._noise_floor_db
        outside_refractory = peak_time - self._last_tap_time >= self.refractory_seconds
        is_tap = (
            outside_refractory
            and strength_db >= self.threshold_db
            and rise_db >= self.minimum_rise_db
            and crest_db >= self.minimum_crest_db
        )

        if is_tap:
            self._last_tap_time = peak_time
            return TapEvent(
                timestamp=peak_time,
                strength_db=strength_db,
                peak_db=peak_db,
                noise_floor_db=self._noise_floor_db,
            )

        if rms_db <= self._noise_floor_db + self.threshold_db * 0.45:
            self._noise_db.append(rms_db)
            self._update_noise_floor()
        return None

    def _update_noise_floor(self) -> None:
        if self._noise_db:
            self._noise_floor_db = float(np.percentile(self._noise_db, 75))

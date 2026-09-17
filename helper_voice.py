from __future__ import annotations

import json
from pathlib import Path
import queue
import threading
import time

import numpy as np


RECORD_PHRASES = ("record", "ready", "start recording")


def is_record_command(result: str) -> bool:
    try:
        data = json.loads(result)
        words = data.get("result", [])
        phrase = data.get("text", "").strip().lower()
        return (phrase in RECORD_PHRASES
                and [word.get("word") for word in words] == phrase.split()
                and all(float(word.get("conf", 0)) >= 0.7 for word in words))
    except (ValueError, TypeError, AttributeError):
        return False


class VoiceRecordControl:
    def __init__(self, model_path: str, sample_rate: int = 48_000):
        if not Path(model_path).is_dir():
            raise RuntimeError("Voice model missing. Follow voice setup in README.md")
        try:
            from vosk import KaldiRecognizer, Model, SetLogLevel

            SetLogLevel(-1)
            self._model = Model(str(model_path))
            self._recognizer = KaldiRecognizer(self._model, sample_rate,
                                               json.dumps([*RECORD_PHRASES, "[unk]"]))
            self._recognizer.SetWords(True)
        except Exception as exc:
            raise RuntimeError(f"Could not load offline voice control: {exc}") from exc
        self._lock = threading.Lock()
        self._queue = queue.Queue(maxsize=128)
        self._stop = threading.Event()
        self._enabled = False
        self._generation = 0
        self._pending = False
        self._pending_at = 0.0
        self.error = ""
        self.feedback = "Waiting for setup"
        self._thread = threading.Thread(target=self._run, name="setup-voice", daemon=True)
        self._thread.start()

    def _clear_locked(self):
        self._generation += 1
        self._pending = False
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def set_listening(self, enabled: bool):
        with self._lock:
            enabled = enabled and not self.error
            if enabled != self._enabled:
                self._enabled = enabled
                self._clear_locked()
                self.feedback = "Listening. Say ready, then pause" if enabled else "Voice paused"

    def reset(self):
        with self._lock:
            self._clear_locked()
            self.feedback = "Audio gap. Say ready again"

    def submit(self, blocks):
        if not blocks:
            return
        with self._lock:
            if not self._enabled:
                return
            samples = np.concatenate([block.samples for block in blocks])
            pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes()
            try:
                self._queue.put_nowait((self._generation, time.monotonic(), pcm))
            except queue.Full:
                # Start fresh instead of acting on delayed speech.
                self._clear_locked()
                self.feedback = "Voice fell behind. Say ready again"

    def pop_record(self) -> bool:
        with self._lock:
            if not self._enabled or not self._pending:
                return False
            if time.monotonic() - self._pending_at > 1.0:
                self._clear_locked()
                return False
            self._enabled = False
            self._clear_locked()
            return True

    def _run(self):
        generation = -1
        try:
            while not self._stop.is_set():
                try:
                    current, submitted, pcm = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                with self._lock:
                    if current != self._generation or not self._enabled:
                        continue
                    if time.monotonic() - submitted > 1.0:
                        self._clear_locked()
                        continue
                if current != generation:
                    self._recognizer.Reset()
                    generation = current
                if self._recognizer.AcceptWaveform(pcm):
                    result = self._recognizer.Result()
                    command = is_record_command(result)
                    with self._lock:
                        if current == self._generation and self._enabled:
                            if command and time.monotonic() - submitted <= 1.0:
                                self._pending = True
                                self._pending_at = time.monotonic()
                                self.feedback = "Command heard"
                            else:
                                self.feedback = "No clear command. Say ready, then pause"
        except Exception as exc:
            with self._lock:
                self.error = f"Voice stopped: {exc}"
                self._enabled = False
                self._clear_locked()

    def close(self):
        self.set_listening(False)
        self._stop.set()
        self._thread.join(timeout=2.0)

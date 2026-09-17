import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from helper_audio import AudioBlock
from helper_voice import VoiceRecordControl, is_record_command


def result(text="record", confidence=0.95):
    return json.dumps({"text": text, "result": [{"word": word, "conf": confidence} for word in text.split()]})


class VoiceTests(unittest.TestCase):
    def test_only_confident_complete_record_command_is_accepted(self):
        self.assertTrue(is_record_command(result()))
        for value in (result(confidence=0.4), result("[unk]"), result("do not record"),
                      '{"partial": "record"}', '{"text": "record"}', "bad json", "null"):
            self.assertFalse(is_record_command(value))

    def test_ready_and_start_recording_are_setup_commands(self):
        for phrase in ("ready", "record", "start recording"):
            self.assertTrue(is_record_command(result(phrase, confidence=0.75)))
            self.assertFalse(is_record_command(result(phrase, confidence=0.4)))

    def test_commands_embedded_in_other_speech_are_rejected(self):
        for phrase in ("not ready", "do not record", "record that again"):
            self.assertFalse(is_record_command(result(phrase)))

    def make_voice(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        recognizer = MagicMock()
        recognizer.AcceptWaveform.return_value = True
        recognizer.Result.return_value = result()
        vosk = SimpleNamespace(Model=MagicMock(), KaldiRecognizer=MagicMock(return_value=recognizer),
                               SetLogLevel=MagicMock())
        with patch.dict("sys.modules", vosk=vosk):
            voice = VoiceRecordControl(directory.name)
        self.addCleanup(voice.close)
        return voice, recognizer

    def wait_for_command(self, voice):
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if voice.pop_record():
                return True
            threading.Event().wait(0.005)
        return False

    def test_pcm_conversion_and_one_command_per_recording(self):
        voice, recognizer = self.make_voice()
        voice.set_listening(True)
        voice.submit([AudioBlock(np.array([-2, 0, 2], dtype=float), 0)])
        self.assertTrue(self.wait_for_command(voice))
        self.assertFalse(voice.pop_record())
        pcm = recognizer.AcceptWaveform.call_args.args[0]
        np.testing.assert_array_equal(np.frombuffer(pcm, dtype="<i2"), [-32767, 0, 32767])
        recognizer.Reset.assert_called_once()

    def test_disabled_listener_does_not_queue_audio(self):
        voice, recognizer = self.make_voice()
        voice.submit([AudioBlock(np.zeros(256), 0)])
        self.assertTrue(voice._queue.empty())
        recognizer.AcceptWaveform.assert_not_called()

    def test_old_in_flight_result_cannot_start_next_pose(self):
        voice, recognizer = self.make_voice()
        entered = threading.Event()
        released = threading.Event()
        finished = threading.Event()
        self.addCleanup(released.set)

        def accept(_pcm):
            entered.set()
            released.wait(2)
            return True

        def response():
            finished.set()
            return result()

        recognizer.AcceptWaveform.side_effect = accept
        recognizer.Result.side_effect = response
        voice.set_listening(True)
        voice.submit([AudioBlock(np.zeros(256), 0)])
        self.assertTrue(entered.wait(1))
        voice.set_listening(False)
        voice.set_listening(True)
        released.set()
        self.assertTrue(finished.wait(1))
        drained = threading.Event()
        recognizer.AcceptWaveform.side_effect = lambda _pcm: drained.set() or False
        voice.submit([AudioBlock(np.zeros(256), 0)])
        self.assertTrue(drained.wait(1))
        self.assertFalse(voice.pop_record())
        self.assertFalse(voice._pending)

    def test_expired_command_is_discarded(self):
        voice, _ = self.make_voice()
        voice.set_listening(True)
        with voice._lock:
            voice._pending = True
            voice._pending_at = time.monotonic() - 2
        self.assertFalse(voice.pop_record())

    def test_recognizer_failure_disables_voice(self):
        voice, recognizer = self.make_voice()
        recognizer.AcceptWaveform.side_effect = RuntimeError("decoder failed")
        voice.set_listening(True)
        voice.submit([AudioBlock(np.zeros(256), 0)])
        voice._thread.join(timeout=1)
        self.assertIn("decoder failed", voice.error)
        self.assertFalse(voice.pop_record())

    def test_missing_model_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "README"):
                VoiceRecordControl(str(Path(directory) / "missing"))

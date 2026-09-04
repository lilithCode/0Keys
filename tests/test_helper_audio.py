import unittest

import numpy as np

from helper_audio import AudioTapDetector


class AudioTapDetectorTests(unittest.TestCase):
    def test_detects_one_impulse_without_duplicates(self):
        sample_rate = 48_000
        block_size = 256
        rng = np.random.default_rng(7)
        audio = rng.normal(0.0, 0.0002, sample_rate * 4)

        tap_sample = sample_rate * 3
        decay = 0.13 * np.exp(-np.arange(900) / 100.0)
        decay *= np.where(np.arange(decay.size) % 2 == 0, 1.0, -1.0)
        audio[tap_sample : tap_sample + decay.size] += decay

        detector = AudioTapDetector(sample_rate=sample_rate)
        events = []
        for start in range(0, audio.size, block_size):
            block = audio[start : start + block_size]
            event = detector.process_block(block, start / sample_rate)
            if event:
                events.append(event)

        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(events[0].timestamp, 3.0, delta=0.02)

    def test_rejects_steady_background_noise(self):
        sample_rate = 48_000
        block_size = 256
        rng = np.random.default_rng(11)
        audio = rng.normal(0.0, 0.0003, sample_rate * 5)
        detector = AudioTapDetector(sample_rate=sample_rate)

        events = []
        for start in range(0, audio.size, block_size):
            event = detector.process_block(
                audio[start : start + block_size], start / sample_rate
            )
            if event:
                events.append(event)

        self.assertEqual(events, [])


if __name__ == "__main__":
    unittest.main()

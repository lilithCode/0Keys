from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from helper_vision import HandSnapshot
from scripts import replay_keyboard


class ReplayTests(unittest.TestCase):
    def test_video_keeps_aspect_and_saves_view_for_future_replays(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            layout = root / "layout.json"
            output = root / "frames.jsonl"
            layout.write_text(json.dumps({"layout": "spaced_ansi_v1", "camera": 0,
                "rotation": 0, "mirrored": True, "flip_rows": False,
                "points": [[0, 0], [1, 0], [1, 1], [0, 1]]}))
            capture = MagicMock()
            capture.get.return_value = 30.0
            capture.grab.side_effect = [True, True, True, False]
            capture.retrieve.return_value = (True, np.zeros((24, 12, 3), dtype=np.uint8))
            tracker = MagicMock()
            tracker.process.side_effect = lambda frame, stamp: HandSnapshot(stamp, ())
            arguments = ["replay", "--extract", "--video", "reference.mp4", "--rotation", "90", "--mirror",
                         "--snapshots", str(output), "--calibration", str(layout),
                         "--profile", str(root / "missing-profile.json")]
            with patch("sys.argv", arguments), patch.object(cv2, "VideoCapture", return_value=capture), \
                    patch.object(replay_keyboard, "MediaPipeHandTracker") as factory, redirect_stdout(io.StringIO()):
                factory.return_value.__enter__.return_value = tracker
                replay_keyboard.main()
            self.assertEqual(tracker.process.call_args_list[0].args[0].shape, (12, 24, 3))
            lines = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(lines[0]["aspect"], 2.0)
            self.assertEqual(lines[0]["context"]["camera"], "video:reference.mp4")
            self.assertTrue(lines[0]["context"]["mirrored"])
            self.assertEqual(len(replay_keyboard.snapshots_from_json(output)), 3)
            with patch("sys.argv", ["replay", "--snapshots", str(output), "--calibration", str(layout),
                                    "--profile", str(root / "missing-profile.json")]), redirect_stdout(io.StringIO()) as result:
                replay_keyboard.main()
            self.assertEqual(json.loads(result.getvalue())["aspect"], 2.0)
            capture.release.assert_called_once()

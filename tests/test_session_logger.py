import json
import tempfile
import unittest

from session_logger import SessionLogger


class SessionLoggerTests(unittest.TestCase):
    def test_writes_json_line(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = SessionLogger(directory)
            logger.write("key_predicted", key="a", probability=0.8)
            record = json.loads(logger.path.read_text(encoding="utf-8"))

        self.assertEqual(record["event"], "key_predicted")
        self.assertEqual(record["key"], "a")
        self.assertAlmostEqual(record["probability"], 0.8)


if __name__ == "__main__":
    unittest.main()

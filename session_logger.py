from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any


class SessionLogger:
    def __init__(self, directory: str | Path = "logs") -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.path = directory / f"session_{timestamp}.jsonl"

    def write(self, event: str, **details: Any) -> None:
        record = {
            "recorded_at": datetime.now().isoformat(timespec="milliseconds"),
            "event": event,
            **details,
        }
        with self.path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, separators=(",", ":")) + "\n")

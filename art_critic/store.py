"""Append-only log of feedback rounds. This file is the preference dataset."""

from __future__ import annotations

import json
import time
from pathlib import Path


class FeedbackStore:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> list[dict]:
        if not self.path.exists():
            return []
        with self.path.open() as f:
            return [json.loads(line) for line in f if line.strip()]

    def append(self, event: dict) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(event) + "\n")

    def reset(self) -> Path | None:
        """Archive the current log under a timestamped name. Nothing is deleted."""
        if not self.path.exists():
            return None
        archived = self.path.with_name(f"{self.path.stem}-{time.strftime('%Y%m%d-%H%M%S')}{self.path.suffix}")
        self.path.rename(archived)
        return archived

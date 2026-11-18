"""Run the demo: `uv run python -m art_critic [--images DIR]`."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import uvicorn

from .harness import Harness
from .index import EmbeddingIndex
from .server import create_app
from .source import FolderSource
from .store import FeedbackStore


def build_harness(images: Path, data: Path, **kwargs) -> Harness:
    return Harness(
        FolderSource(images, data / "stills"),
        EmbeddingIndex(data / "chroma"),
        FeedbackStore(data / "feedback.jsonl"),
        **kwargs,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="RLHF demo harness for visual preferences, conditioned on prompts")
    parser.add_argument("--images", type=Path, default=Path("data/wikiart/images"), help="folder of images and videos to judge")
    parser.add_argument("--data", type=Path, default=None, help="where stills, the index and feedback live (default: data/state/<folder>)")
    parser.add_argument("--conditions", type=Path, default=None, help='JSON file of extra conditions: {"axis": ["value", ...]}')
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    # State is kept per media folder, inside this project, never beside the user's files.
    folder = args.images.resolve()
    data = args.data or Path("data/state") / f"{folder.parent.name}-{folder.name}-{hashlib.sha1(str(folder).encode()).hexdigest()[:6]}"
    extra = json.loads(args.conditions.read_text()) if args.conditions else None

    print(f"indexing {args.images} (first run embeds everything); state in {data} ...")
    harness = build_harness(args.images, data, extra_conditions=extra)
    vocab = harness.vocabulary()
    print(f"{len(harness.ids)} items, {sum(map(len, vocab.values()))} conditions on {len(vocab)} axes, {len(harness.events)} past rounds")
    uvicorn.run(create_app(harness), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()

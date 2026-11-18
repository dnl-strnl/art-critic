"""Where the things being judged come from.

FolderSource is the only implementation: every image and video under a folder,
with optional labels. A generator would implement the same protocol.

A video is treated as a handful of still frames for embedding and as the
original file for display, so nothing downstream needs to know the difference.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Protocol

from PIL import Image, ImageOps

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".m4v", ".mov", ".webm"}
VIDEO_FRAMES = 6
LABELS_FILE = "labels.json"


class Source(Protocol):
    def ids(self) -> list[str]: ...

    def kind(self, item_id: str) -> str: ...  # "image" or "video"

    def path(self, item_id: str) -> Path: ...  # what the browser displays

    def poster(self, item_id: str) -> Path: ...  # a still, for small previews

    def frames(self, item_id: str) -> list[Path]: ...  # stills to embed

    def labels(self, item_id: str) -> dict[str, str]: ...  # axis -> value


class FolderSource:
    """Every readable image and video under `folder`.

    If the folder holds a labels.json mapping relative paths to
    {axis: value}, those become labeled conditions (artist, style, ...).
    """

    def __init__(self, folder: Path, cache_dir: Path, size: int = 512):
        if not folder.is_dir():
            raise FileNotFoundError(f"media folder not found: {folder}")
        cache_dir.mkdir(parents=True, exist_ok=True)
        labels_path = folder / LABELS_FILE
        all_labels = json.loads(labels_path.read_text()) if labels_path.exists() else {}
        self._items: dict[str, dict] = {}
        for origin in sorted(folder.rglob("*")):
            ext = origin.suffix.lower()
            if ext not in IMAGE_EXTS | VIDEO_EXTS or not origin.is_file():
                continue
            rel = origin.relative_to(folder).as_posix()
            item_id = hashlib.sha1(rel.encode()).hexdigest()[:16]
            try:
                if ext in VIDEO_EXTS:
                    frames = _video_stills(origin, cache_dir, item_id, size)
                    item = {"kind": "video", "path": origin, "poster": frames[len(frames) // 2], "frames": frames}
                else:
                    still = _image_still(origin, cache_dir / f"{item_id}.jpg", size)
                    item = {"kind": "image", "path": still, "poster": still, "frames": [still]}
            except Exception as err:  # one bad file should not take the pool down
                print(f"skipping unreadable file {origin}: {err}")
                continue
            item["labels"] = {str(k): str(v) for k, v in all_labels.get(rel, {}).items()}
            self._items[item_id] = item

    def ids(self) -> list[str]:
        return list(self._items)

    def kind(self, item_id: str) -> str:
        return self._items[item_id]["kind"]

    def path(self, item_id: str) -> Path:
        return self._items[item_id]["path"]

    def poster(self, item_id: str) -> Path:
        return self._items[item_id]["poster"]

    def frames(self, item_id: str) -> list[Path]:
        return self._items[item_id]["frames"]

    def labels(self, item_id: str) -> dict[str, str]:
        return self._items[item_id]["labels"]


def _save_still(im: Image.Image, dest: Path, size: int) -> None:
    im = ImageOps.exif_transpose(im).convert("RGB")
    im.thumbnail((size, size))
    im.save(dest, "JPEG", quality=88)


def _image_still(origin: Path, dest: Path, size: int) -> Path:
    if not dest.exists():
        with Image.open(origin) as im:
            _save_still(im, dest, size)
    return dest


def _video_stills(origin: Path, cache_dir: Path, item_id: str, size: int) -> list[Path]:
    """Evenly spaced frames across the video, cached as JPEGs."""
    cached = sorted(cache_dir.glob(f"{item_id}-f*.jpg"))
    if cached:
        return cached
    import av

    stills: list[Path] = []
    with av.open(str(origin)) as container:
        stream = container.streams.video[0]
        if container.duration:
            seconds = container.duration / av.time_base
            targets = [int(seconds * (k + 0.5) / VIDEO_FRAMES / stream.time_base) for k in range(VIDEO_FRAMES)]
        else:
            targets = [0]
        for k, target in enumerate(targets):
            container.seek(target, stream=stream)  # lands on the keyframe before the target
            for frame in container.decode(stream):
                if frame.pts is None or frame.pts >= target:
                    dest = cache_dir / f"{item_id}-f{k}.jpg"
                    _save_still(frame.to_image(), dest, size)
                    stills.append(dest)
                    break
    if not stills:
        raise ValueError("no decodable video frames")
    return stills

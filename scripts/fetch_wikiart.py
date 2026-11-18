"""Download a style-balanced sample of WikiArt into data/wikiart/images/.

Labels (artist, style, genre) come with the dataset and are written to
labels.json beside the images. Source: the huggan/wikiart parquet shards on
Hugging Face. Each shard is about 520 MB and holds ~1,100 artworks. Neighbouring
shards share artists and styles, so the default takes shards spread across the
set. Shards are deleted once the sample is extracted; running again with other
shards adds to the pool.
"""

from __future__ import annotations

import argparse
import io
import json
import random
from collections import Counter
from pathlib import Path

import httpx
import pyarrow.parquet as pq
from PIL import Image

SHARD = "https://huggingface.co/datasets/huggan/wikiart/resolve/main/data/train-{:05d}-of-00072.parquet"
INFO = "https://huggingface.co/datasets/huggan/wikiart/resolve/main/dataset_infos.json"


def pretty(name: str, artist: bool = False) -> str | None:
    if name.lower().startswith("unknown"):
        return None
    name = name.replace("ã\xad", "í").replace("_", " ")
    return name.replace("-", " ").title() if artist else name


def download(url: str, dest: Path) -> None:
    with httpx.stream("GET", url, follow_redirects=True, timeout=60) as res:
        res.raise_for_status()
        size = int(res.headers.get("content-length", 0))
        if dest.exists() and dest.stat().st_size == size:
            return
        with dest.open("wb") as f:
            for chunk in res.iter_bytes(1 << 20):
                f.write(chunk)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--shards", type=int, nargs="+", default=[0, 24, 48, 66], help="which of the 72 shards (~520 MB each) to sample from")
    parser.add_argument("--per-style", type=int, default=60)
    parser.add_argument("--per-artist", type=int, default=30)
    parser.add_argument("--size", type=int, default=640, help="longest side of the saved image")
    parser.add_argument("--out", type=Path, default=Path("data/wikiart/images"))
    parser.add_argument("--keep-shards", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cache = args.out.parent / "shards"
    cache.mkdir(exist_ok=True)
    rng = random.Random(args.seed)

    files = []
    for n in args.shards:
        dest = cache / f"{n:05d}.parquet"
        print(f"downloading shard {n} ...", flush=True)
        download(SHARD.format(n), dest)
        files.append((n, pq.ParquetFile(dest)))

    # The shards store labels as integers; their names live in the dataset card.
    info = httpx.get(INFO, follow_redirects=True, timeout=60).raise_for_status().json()
    features = next(iter(info.values()))["features"]
    names = {axis: features[axis]["names"] for axis in ("artist", "style", "genre")}
    rows = []  # (shard, row in shard, labels)
    for n, f in files:
        table = f.read(columns=["artist", "style", "genre"]).to_pydict()
        for i in range(f.metadata.num_rows):
            label = {axis: pretty(names[axis][table[axis][i]], artist=axis == "artist") for axis in names}
            rows.append((n, i, {k: v for k, v in label.items() if v}))
    print(f"read labels for {len(rows)} artworks", flush=True)

    # Fill every style up to its cap, then top up well-represented artists.
    rng.shuffle(rows)
    chosen, styles, artists = {}, Counter(), Counter()
    for n, i, label in rows:
        if len(chosen) >= args.count:
            continue
        style, artist = label.get("style"), label.get("artist")
        if (style and styles[style] < args.per_style) or (artist and artists[artist] < args.per_artist):
            styles[style] += 1
            artists[artist] += 1
            chosen[(n, i)] = label

    labels = {}
    for n, f in files:
        start = 0
        for batch in f.iter_batches(batch_size=64, columns=["image"]):
            for j, image in enumerate(batch.column("image").to_pylist()):
                label = chosen.get((n, start + j))
                if label is None:
                    continue
                name = f"wikiart-{n:02d}-{start + j:05d}.jpg"
                try:
                    im = Image.open(io.BytesIO(image["bytes"])).convert("RGB")
                    im.thumbnail((args.size, args.size))
                    im.save(args.out / name, "JPEG", quality=88)
                    labels[name] = label
                except OSError as err:
                    print(f"failed {name}: {err}")
            start += len(batch)

    labels_path = args.out / "labels.json"
    merged = {**(json.loads(labels_path.read_text()) if labels_path.exists() else {}), **labels}
    labels_path.write_text(json.dumps(merged, indent=1, ensure_ascii=False))
    if not args.keep_shards:
        for f in cache.glob("*.parquet"):
            f.unlink()
        cache.rmdir()
    print(f"added {len(labels)}/{len(chosen)} images; {len(merged)} now in {args.out}")
    for axis in names:
        counts = Counter(label[axis] for label in merged.values() if axis in label)
        print(f"{axis}: {len(counts)} values, {sum(c >= 12 for c in counts.values())} with 12+ images")


if __name__ == "__main__":
    main()

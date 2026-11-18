"""ChromaDB as the embedding store and retrieval engine.

Two inner-product collections of raw CLIP vectors:

  items       one per image or video in the pool, with its labels as metadata
  conditions  one per condition (text embedding of its prompt)

Chroma answers three questions: which items carry a label, which items best
match a text condition, and which conditions sit closest to a learned taste.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable

import chromadb
import numpy as np
from chromadb.config import Settings

from .conditions import Condition
from .source import Source

BATCH = 500


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


class EmbeddingIndex:
    def __init__(self, path: Path):
        self.client = chromadb.PersistentClient(path=str(path), settings=Settings(anonymized_telemetry=False))
        self.items = self.client.get_or_create_collection("items", metadata={"hnsw:space": "ip"})
        self.conditions = self.client.get_or_create_collection("conditions", metadata={"hnsw:space": "ip"})

    # -- items ----------------------------------------------------------------

    def sync_items(self, source: Source, embed: Callable[[list[Path]], np.ndarray]) -> int:
        """Embed items that are new to the pool; drop ones that left it."""
        wanted = source.ids()
        have = set(self.items.get(include=[])["ids"])
        stale = sorted(have - set(wanted))
        for i in range(0, len(stale), BATCH):
            self.items.delete(ids=stale[i : i + BATCH])
        missing = [i for i in wanted if i not in have]
        for i in range(0, len(missing), BATCH):
            chunk = missing[i : i + BATCH]
            frames = [source.frames(item_id) for item_id in chunk]
            flat = embed([p for group in frames for p in group])
            # One vector per item: a video is the mean of its frames.
            ends = np.cumsum([len(group) for group in frames])
            vectors = np.stack([_unit(part.mean(axis=0)) for part in np.split(flat, ends[:-1])])
            metadatas = [{"kind": source.kind(item_id), **source.labels(item_id)} for item_id in chunk]
            self.items.add(ids=chunk, embeddings=vectors, metadatas=metadatas)
        return len(missing)

    def all_items(self) -> tuple[list[str], np.ndarray]:
        """Every id and its embedding, in a stable (sorted) order."""
        got = self.items.get(include=["embeddings"])
        order = np.argsort(got["ids"])
        return [got["ids"][i] for i in order], np.asarray(got["embeddings"], dtype=np.float32)[order]

    def with_label(self, axis: str, value: str) -> list[str]:
        return self.items.get(where={axis: value}, include=[])["ids"]

    def nearest(self, vector: np.ndarray, k: int) -> list[str]:
        """Ids of the `k` items with the largest inner product with `vector`."""
        return self.items.query(query_embeddings=[vector.tolist()], n_results=k, include=[])["ids"][0]

    # -- conditions -----------------------------------------------------------

    @staticmethod
    def _id(key: str) -> str:
        return hashlib.sha1(key.encode()).hexdigest()[:16]

    def sync_conditions(self, vocab: list[Condition], embed: Callable[[list[str]], np.ndarray]) -> None:
        """Embed vocabulary conditions that are new; drop retired ones. Custom ones are kept."""
        got = self.conditions.get(include=["metadatas"])
        wanted = {self._id(c.key): c for c in vocab}
        # A changed prompt counts as a new condition.
        current = {i for i, m in zip(got["ids"], got["metadatas"]) if i in wanted and m["prompt"] == wanted[i].prompt}
        retired = [i for i, m in zip(got["ids"], got["metadatas"]) if i not in current and not m["custom"]]
        if retired:
            self.conditions.delete(ids=retired)
        new = [c for i, c in wanted.items() if i not in current]
        if new:
            self.add_conditions(new, embed([c.prompt for c in new]))

    def add_conditions(self, conds: list[Condition], vectors: np.ndarray) -> None:
        self.conditions.upsert(
            ids=[self._id(c.key) for c in conds],
            embeddings=vectors,
            metadatas=[{"axis": c.axis, "value": c.value, "prompt": c.prompt, "labeled": c.labeled, "custom": c.custom} for c in conds],
        )

    def all_conditions(self) -> list[tuple[Condition, np.ndarray]]:
        got = self.conditions.get(include=["embeddings", "metadatas"])
        vectors = np.asarray(got["embeddings"], dtype=np.float32)
        return [(Condition(m["axis"], m["value"], m["prompt"], m["labeled"]), v) for m, v in zip(got["metadatas"], vectors)]

    def closest_conditions(self, direction: np.ndarray, k: int) -> list[dict]:
        """Vocabulary conditions most aligned with a taste direction."""
        res = self.conditions.query(query_embeddings=[direction.tolist()], n_results=k, where={"custom": False}, include=["metadatas"])
        return [{"axis": m["axis"], "value": m["value"]} for m in res["metadatas"][0]]

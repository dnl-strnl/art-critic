"""The RLHF loop: pick a condition, show items that fit it, take feedback, refit."""

from __future__ import annotations

import threading
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

import numpy as np

from . import conditions, embed, sampler
from .conditions import OPEN, Condition
from .index import EmbeddingIndex
from .reward import STARS, RewardModel, Signals, pair_features
from .source import Source
from .store import FeedbackStore

ROUND_SIZE = 4
PANEL_SIZE = 8
MIN_POOL = 2 * ROUND_SIZE
MIN_LABELED = 12  # a label value needs this many items to become a condition
TEXT_MEMBERS = 40  # a text condition's candidates: its best CLIP matches
STICK = 3  # rounds spent on a condition before auto-rotation moves on
COND_DIM = 16
# Prior strengths (L2) for the three feature blocks; see reward.py.
L2_MATCH, L2_TASTE, L2_SHIFT = 0.5, 0.02, 0.05


class Harness:
    def __init__(
        self,
        source: Source,
        index: EmbeddingIndex,
        store: FeedbackStore,
        embed_images: Callable[[list[Path]], np.ndarray] = embed.embed_images,
        embed_texts: Callable[[list[str]], np.ndarray] = embed.embed_texts,
        extra_conditions: dict[str, list[str]] | None = None,
        dim: int = 64,
        seed: int | None = None,
    ):
        self.source, self.index, self.store, self.embed_texts = source, index, store, embed_texts
        self.lock = threading.Lock()
        self.rng = np.random.default_rng(seed)

        index.sync_items(source, embed_images)
        self.ids, self.X = index.all_items()
        if len(self.ids) < MIN_POOL:
            raise ValueError(f"need at least {MIN_POOL} items, found {len(self.ids)}")
        self.row = {item_id: i for i, item_id in enumerate(self.ids)}
        self.proj = embed.fit_projection(self.X, dim)
        self.Z = self.proj(self.X)

        counts = Counter(pair for i in self.ids for pair in source.labels(i).items())
        label_values: dict[str, list[str]] = {}
        for (axis, value), n in sorted(counts.items()):
            if n >= MIN_LABELED and axis not in conditions.SPECIFIC_LABEL_AXES:
                label_values.setdefault(axis, []).append(value)
        index.sync_conditions(conditions.build(label_values, extra_conditions), embed_texts)
        self.conds: dict[str, Condition] = {}
        self.T: dict[str, np.ndarray] = {}
        for cond, vector in index.all_conditions():
            self.conds[cond.key], self.T[cond.key] = cond, vector
        self.conds[OPEN.key] = OPEN  # has no prompt, so no text vector
        # Feedback given under a condition that is no longer offered still teaches the model.
        logged = {e["condition"]["key"]: e["condition"] for e in store.load() if "condition" in e}
        past = [Condition(c["axis"], c["value"], c["prompt"]) for key, c in logged.items() if key not in self.conds]
        # Known to the model, but never dealt or listed.
        self.hidden = {OPEN.key, *(cond.key for cond in past)}
        if past:
            for cond, vector in zip(past, embed_texts([cond.prompt for cond in past])):
                self.conds[cond.key], self.T[cond.key] = cond, vector
        # Condition features: PCA over the conditions' text embeddings.
        self.cond_proj = embed.fit_projection(np.stack(list(self.T.values())), COND_DIM)
        self._members: dict[str, np.ndarray] = {}
        self._match: dict[str, np.ndarray] = {}

        d, k = self.Z.shape[1], len(self.cond_proj.components)
        self.model = RewardModel(1 + d + k * d, np.concatenate([[L2_MATCH], np.full(d, L2_TASTE), np.full(k * d, L2_SHIFT)]))
        self.pending: dict[str, dict] = {}
        self.auto: dict | None = None
        # Rounds logged before conditions existed are left in the file but not used.
        self.events = [e for e in store.load() if e.get("condition", {}).get("key") in self.conds]
        self._refit()

    # -- conditions -----------------------------------------------------------

    def vocabulary(self) -> dict[str, list[str]]:
        with self.lock:
            out: dict[str, list[str]] = defaultdict(list)
            for cond in self.conds.values():
                if cond.key not in self.hidden:
                    out[cond.axis].append(cond.value)
            return {axis: sorted(values) for axis, values in sorted(out.items())}

    def members(self, key: str) -> np.ndarray:
        """Rows of the items that fit a condition: by label, or by CLIP match."""
        if key not in self._members:
            cond = self.conds[key]
            if cond is OPEN:
                ids = self.ids
            elif cond.labeled:
                ids = self.index.with_label(cond.axis, cond.value)
            else:
                ids = self.index.nearest(self.T[key], min(TEXT_MEMBERS, len(self.ids)))
            self._members[key] = np.array(sorted(self.row[i] for i in ids if i in self.row))
        return self._members[key]

    def match(self, key: str) -> np.ndarray:
        """How well every item fits the condition's prompt, standardized over the pool."""
        if key not in self._match:
            if key == OPEN.key:  # no prompt, nothing to match
                self._match[key] = np.zeros(len(self.ids), dtype=np.float32)
            else:
                m = self.X @ self.T[key]
                self._match[key] = ((m - m.mean()) / m.std()).astype(np.float32)
        return self._match[key]

    def features(self, key: str, rows: np.ndarray, interaction: bool = True) -> np.ndarray:
        z_cond = None
        if interaction and key == OPEN.key:  # only the general-taste block is live
            z_cond = np.zeros(len(self.cond_proj.components), dtype=np.float32)
        elif interaction:
            z_cond = self.cond_proj(self.T[key][None])[0]
            z_cond = z_cond / max(np.linalg.norm(z_cond), 1e-8)
        return pair_features(self.match(key)[rows], self.Z[rows], z_cond)

    def score(self, key: str, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.model.predict(self.features(key, rows), offset=self.match(key)[rows])

    def _choose_condition(self, axis: str | None, value: str | None, text: str | None) -> tuple[Condition, bool]:
        if text and text.strip():
            cond = conditions.custom(text.strip())
            if cond.key not in self.conds:
                vector = self.embed_texts([cond.prompt])[0]
                self.index.add_conditions([cond], vector[None])
                self.conds[cond.key], self.T[cond.key] = cond, vector
            return cond, False
        if axis or value:
            if f"{axis}:{value}" in self.hidden:
                raise KeyError(f"{axis}:{value}")
            return self.conds[f"{axis}:{value}"], False

        # Auto-rotation: stay a few rounds, then move to the least-visited axis and value.
        if self.auto and self.auto["left"] > 0:
            self.auto["left"] -= 1
            return self.conds[self.auto["key"]], True
        by_axis = Counter(e["condition"]["axis"] for e in self.events)
        by_key = Counter(e["condition"]["key"] for e in self.events)
        offered = [c for c in self.conds.values() if not c.custom and c.key not in self.hidden]
        axis = min(sorted({c.axis for c in offered}), key=lambda a: (by_axis[a], self.rng.random()))
        cond = min((c for c in offered if c.axis == axis), key=lambda c: (by_key[c.key], self.rng.random()))
        self.auto = {"key": cond.key, "left": STICK - 1}
        return cond, True

    # -- the loop -------------------------------------------------------------

    def new_round(self, axis: str | None = None, value: str | None = None, text: str | None = None) -> dict:
        with self.lock:
            cond, auto = self._choose_condition(axis, value, text)
            key = cond.key
            unseen = self._unseen(key)
            mean, std = self.score(key, unseen)
            picks = np.array(sampler.choose(mean, std, np.arange(len(unseen)), self.rng, ROUND_SIZE))
            rows = unseen[picks]
            shown = [self.ids[r] for r in rows]
            # Both guesses are fixed now, before the answer exists.
            guesses = {
                "model_pick": shown[int(np.argmax(mean[picks]))],
                "match_pick": shown[int(np.argmax(self.match(key)[rows]))],
            }
            # The same four may be judged with the condition switched off; those guesses are fixed now too.
            # With no prompt, prompt matching can only guess at random.
            open_guesses = {
                "model_pick": shown[int(np.argmax(self.score(OPEN.key, rows)[0]))],
                "match_pick": shown[int(self.rng.integers(len(shown)))],
            }
            round_id = uuid.uuid4().hex[:12]
            self.pending = {round_id: {"shown": shown, "condition": cond.to_dict(), **guesses, "open": open_guesses}}
            return {
                "round_id": round_id,
                "condition": {**cond.to_dict(), "auto": auto, "labeled": cond.labeled},
                "items": [self._item(i) for i in shown],
                "state": self._state(key),
            }

    def submit(self, round_id: str, pick: str | None, thumbs: dict[str, int], ratings: dict[str, int], free: bool = False) -> dict:
        """Log a round's feedback and refit. `free` means it was given with the condition switched off."""
        with self.lock:
            pending = self.pending.get(round_id)
            if pending is None:
                raise KeyError(f"unknown or already-submitted round: {round_id}")
            shown = pending["shown"]
            if any(i not in shown for i in [*thumbs, *ratings, *([pick] if pick else [])]):
                raise ValueError("feedback refers to an item that was not in this round")
            if any(v not in (1, -1) for v in thumbs.values()):
                raise ValueError("thumbs must be 1 or -1")
            if any(v not in range(1, STARS + 1) for v in ratings.values()):
                raise ValueError(f"ratings must be 1..{STARS}")
            del self.pending[round_id]
            open_guesses = pending.pop("open")
            if free:
                pending = {**pending, "condition": OPEN.to_dict(), "dealt_under": pending["condition"], **open_guesses}
            event = {
                "round_id": round_id,
                "ts": time.time(),
                **pending,
                "pick": pick,
                "thumbs": thumbs,
                "ratings": ratings,
                "skipped": not (pick or thumbs or ratings),
            }
            self.store.append(event)
            self.events.append(event)
            self._refit()
            return {
                "model_hit": pick == event["model_pick"] if pick else None,
                "match_hit": pick == event["match_pick"] if pick else None,
            }

    def reset(self) -> None:
        with self.lock:
            self.store.reset()
            self.events, self.pending, self.auto = [], {}, None
            self._refit()

    # -- internals ------------------------------------------------------------

    def _unseen(self, key: str) -> np.ndarray:
        rows = self.members(key)
        unseen = np.array([r for r in rows if self.ids[r] not in self.seen[key]], dtype=int)
        return unseen if len(unseen) >= ROUND_SIZE else rows

    def design(self, interaction: bool = True) -> tuple[np.ndarray, np.ndarray, Signals]:
        """All feedback so far as a feature matrix, its match offsets, and signals over its rows.

        One row per judged (condition, item) pair. `interaction=False` gives the
        prompt-blind variant, for comparing against.
        """
        signals = Signals()
        pairs: dict[tuple[str, int], int] = {}  # (condition, item row) -> feature row

        def pair(key: str, item_id: str) -> int:
            return pairs.setdefault((key, self.row[item_id]), len(pairs))

        for e in self.events:
            key = e["condition"]["key"]
            shown = [i for i in e["shown"] if i in self.row]
            if e["pick"] in shown and len(shown) > 1:
                signals.choices.append(([pair(key, i) for i in shown], shown.index(e["pick"])))
            signals.thumbs += [(pair(key, i), v) for i, v in e["thumbs"].items() if i in self.row]
            signals.ratings += [(pair(key, i), v) for i, v in e["ratings"].items() if i in self.row]

        by_key: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for (key, row), i in pairs.items():
            by_key[key].append((i, row))
        width = self.model.dim if interaction else 1 + self.Z.shape[1]
        Phi = np.zeros((len(pairs), width), dtype=np.float32)
        offset = np.zeros(len(pairs), dtype=np.float32)
        for key, entries in by_key.items():
            idx, rows = (np.array(x) for x in zip(*entries))
            Phi[idx], offset[idx] = self.features(key, rows, interaction), self.match(key)[rows]
        return Phi, offset, signals

    def _refit(self) -> None:
        self.seen: dict[str, set[str]] = defaultdict(set)
        for e in self.events:
            # A round judged with the condition off was still dealt, and seen, under one.
            for key in {e["condition"]["key"], e.get("dealt_under", {}).get("key")} & self.conds.keys():
                self.seen[key].update(i for i in e["shown"] if i in self.row)
                if len(self.members(key)) - len(self.seen[key]) < ROUND_SIZE:
                    self.seen[key] = set()  # condition exhausted: start another pass
        Phi, offset, signals = self.design()
        self.model.fit(Phi, signals, offset)

    def _item(self, item_id: str) -> dict:
        return {"id": item_id, "kind": self.source.kind(item_id), "url": f"/media/{item_id}", "poster": f"/poster/{item_id}"}

    def _state(self, key: str) -> dict:
        unseen = self._unseen(key)
        mean, _ = self.score(key, unseen)
        order = np.argsort(-mean)
        lean = {"likes": [], "dislikes": []}
        if self.model.fitted:
            taste = self.proj.to_raw(self.model.w[1 : 1 + self.Z.shape[1]])
            lean = {"likes": self.index.closest_conditions(taste, 6), "dislikes": self.index.closest_conditions(-taste, 6)}

        stars: dict[tuple[str, str], list[int]] = defaultdict(list)
        for e in self.events:
            stars[(e["condition"]["axis"], e["condition"]["value"])] += list(e["ratings"].values())
        by_condition: dict[str, list[dict]] = defaultdict(list)
        for (axis, value), given in stars.items():
            if given:
                by_condition[axis].append({"value": value, "stars": float(np.mean(given)), "n": len(given)})
        for rows in by_condition.values():
            rows.sort(key=lambda r: -r["stars"])

        judged = [e for e in self.events if e["pick"]]
        return {
            "fitted": self.model.fitted,
            "pool": len(self.ids),
            "members": len(self.members(key)),
            "unseen": len(unseen),
            "counts": {
                "rounds": sum(not e["skipped"] for e in self.events),
                "skipped": sum(e["skipped"] for e in self.events),
                "picks": sum(bool(e["pick"]) for e in self.events),
                "thumbs": sum(len(e["thumbs"]) for e in self.events),
                "ratings": sum(len(e["ratings"]) for e in self.events),
                "conditions": len({e["condition"]["key"] for e in self.events} - {OPEN.key}),
            },
            # Before any feedback the score is prompt match alone, so these are zero-shot.
            "favorites": [self._item(self.ids[r]) for r in unseen[order[:PANEL_SIZE]]],
            "least": [self._item(self.ids[r]) for r in unseen[order[::-1][:PANEL_SIZE]]] if self.model.fitted else [],
            "lean": lean,
            "by_condition": dict(sorted(by_condition.items())),
            # 1 where a pre-registered guess matched the person's pick.
            "accuracy": {
                "model": [int(e["pick"] == e["model_pick"]) for e in judged],
                "match": [int(e["pick"] == e["match_pick"]) for e in judged],
            },
            "chance": 1 / ROUND_SIZE,
        }

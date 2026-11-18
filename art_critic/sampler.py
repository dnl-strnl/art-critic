"""Decides which images to show next: mostly exploit, some explore."""

from __future__ import annotations

import numpy as np


def choose(
    mean: np.ndarray,
    std: np.ndarray,
    candidates: np.ndarray,
    rng: np.random.Generator,
    n: int = 4,
    fitted: bool = True,
) -> list[int]:
    """Pick `n` rows from `candidates`.

    Half by predicted score (what the model thinks you'll like), one by highest
    uncertainty (what it knows least about), the rest at random so the model
    keeps seeing things it would never have chosen.
    """
    candidates = np.asarray(candidates)
    if not fitted or len(candidates) <= n:
        return rng.permutation(candidates)[:n].tolist()

    picked: list[int] = []
    remaining = candidates.copy()

    def take(scores: np.ndarray, count: int) -> None:
        nonlocal remaining
        order = np.argsort(-scores)[:count]
        picked.extend(remaining[order].tolist())
        remaining = np.delete(remaining, order)

    take(mean[remaining], n // 2)
    take(std[remaining], 1)
    take(rng.random(len(remaining)), n - len(picked))
    # Shuffle so screen position says nothing about why an image is there.
    return rng.permutation(picked).tolist()

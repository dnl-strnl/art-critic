"""A simulated person with a hidden taste, for tests."""

from __future__ import annotations

import numpy as np


class SyntheticUser:
    """Answers feedback prompts from a hidden per-image utility plus noise."""

    def __init__(self, utility: np.ndarray, rng: np.random.Generator, noise: float = 0.5):
        self.utility = (utility - utility.mean()) / utility.std()
        self.rng, self.noise = rng, noise
        self._star_edges = np.quantile(self.utility, [0.2, 0.4, 0.6, 0.8])

    def _felt(self, rows) -> np.ndarray:
        rows = np.atleast_1d(rows)
        return self.utility[rows] + self.noise * self.rng.standard_normal(len(rows))

    def pick(self, rows: list[int]) -> int:
        """Position of the favorite among the rows shown."""
        return int(np.argmax(self._felt(rows)))

    def thumb(self, row: int) -> int:
        return 1 if self._felt(row)[0] > 0 else -1

    def stars(self, row: int) -> int:
        return int(np.searchsorted(self._star_edges, self._felt(row)[0])) + 1

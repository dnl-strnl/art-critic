"""The reward model: preference for an item, conditioned on a prompt.

Each judged thing is a (condition, item) pair with features phi, and

    r(condition, item) = match + theta . phi

`match` is how well the item fits the condition's prompt (frozen CLIP, not
learned). theta is learned, and phi has three blocks (see pair_features):

  match        lets feedback turn prompt-matching up or down
  z(item)      general taste: what this person likes under any condition
  z(cond) x z(item)   how their taste shifts with the condition

Each feedback type is a different likelihood over the same r:

  best-of-N pick   softmax choice over the items shown
  thumbs up/down   Bernoulli, sigmoid(+-(r - b))
  1-5 stars        ordinal logistic, P(stars <= k) = sigmoid(c_k - r)

theta gets a Gaussian prior (the L2 term). The fit is the MAP estimate, and a
Laplace approximation around it gives a per-pair uncertainty: high for pairs
unlike anything judged so far, shrinking as feedback accumulates.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

STARS = 5


def pair_features(match: np.ndarray, z_item: np.ndarray, z_cond: np.ndarray | None = None) -> np.ndarray:
    """Features for n items under one condition.

    match: (n,) standardized prompt-match; z_item: (n, d); z_cond: (k,) or
    None to leave out the interaction block (a prompt-blind taste model).
    """
    parts = [match[:, None], z_item]
    if z_cond is not None:
        parts.append((z_cond[None, :, None] * z_item[:, None, :]).reshape(len(z_item), -1))
    return np.concatenate(parts, axis=1).astype(np.float32)


@dataclass
class Signals:
    """Feedback expressed as row indices into the feature matrix."""

    choices: list[tuple[list[int], int]] = field(default_factory=list)  # (rows shown, position picked)
    thumbs: list[tuple[int, int]] = field(default_factory=list)  # (row, +1 or -1)
    ratings: list[tuple[int, int]] = field(default_factory=list)  # (row, 1..5)

    def __len__(self) -> int:
        return len(self.choices) + len(self.thumbs) + len(self.ratings)


class RewardModel:
    def __init__(self, dim: int, l2: float | np.ndarray = 0.02):
        self.dim = dim
        self.l2 = np.broadcast_to(np.asarray(l2, dtype=np.float32), (dim,)).copy()
        self._clear()

    def _clear(self) -> None:
        self.w = np.zeros(self.dim, dtype=np.float32)
        self.cov = np.diag(1 / (2 * self.l2)).astype(np.float32)  # the prior
        self.bias = 0.0
        self.cutpoints = np.linspace(-1.5, 1.5, STARS - 1, dtype=np.float32)
        self.fitted = False

    def predict(self, Phi: np.ndarray, offset: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Score and its posterior standard deviation, per row of Phi."""
        mean = Phi @ self.w + (0 if offset is None else offset)
        return mean, np.sqrt(np.einsum("nd,nd->n", Phi @ self.cov, Phi))

    def fit(self, Phi: np.ndarray, signals: Signals, offset: np.ndarray | None = None) -> None:
        """Refit from scratch. Cheap enough to run after every round."""
        if len(signals) == 0:
            self._clear()
            return

        P = torch.from_numpy(np.ascontiguousarray(Phi, dtype=np.float32))
        off = torch.zeros(len(P)) if offset is None else torch.from_numpy(np.asarray(offset, dtype=np.float32))
        l2 = torch.from_numpy(self.l2)
        w = torch.zeros(self.dim, requires_grad=True)
        bias = torch.zeros((), requires_grad=True)
        # Cutpoints are c1, then positive gaps, so they stay ordered.
        cut_raw = torch.tensor([-1.5, 0.5413, 0.5413, 0.5413], requires_grad=True)

        if signals.choices:
            width = max(len(shown) for shown, _ in signals.choices)
            shown = torch.zeros(len(signals.choices), width, dtype=torch.long)
            mask = torch.zeros(len(signals.choices), width, dtype=torch.bool)
            for i, (rows, _) in enumerate(signals.choices):
                shown[i, : len(rows)] = torch.tensor(rows)
                mask[i, : len(rows)] = True
            picked = torch.tensor([pos for _, pos in signals.choices])
        t_rows = torch.tensor([row for row, _ in signals.thumbs], dtype=torch.long)
        t_sign = torch.tensor([float(s) for _, s in signals.thumbs])
        r_rows = torch.tensor([row for row, _ in signals.ratings], dtype=torch.long)
        r_star = torch.tensor([s - 1 for _, s in signals.ratings], dtype=torch.long)[:, None]  # 0..4

        def choice_nll(r):
            if not signals.choices:
                return 0.0
            logp = F.log_softmax(r[shown].masked_fill(~mask, float("-inf")), dim=1)
            return -logp[torch.arange(len(picked)), picked].sum()

        def separable_nll(r):
            """Thumbs and stars: each term depends on a single score."""
            out = F.softplus(-t_sign * (r[t_rows] - bias)).sum()
            if len(r_rows):
                cdf = torch.sigmoid(_ordered(cut_raw)[None] - r[r_rows][:, None])  # (T, 4)
                edge = torch.zeros(len(r_rows), 1)
                cdf = torch.cat([edge, cdf, edge + 1], dim=1)  # (T, 6)
                p = cdf.gather(1, r_star + 1) - cdf.gather(1, r_star)
                out = out - torch.log(p.clamp_min(1e-6)).sum()
            return out

        opt = torch.optim.LBFGS([w, bias, cut_raw], max_iter=200, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            r = off + P @ w
            loss = (l2 * w**2).sum() + choice_nll(r) + separable_nll(r)
            loss.backward()
            return loss

        opt.step(closure)

        # Laplace: covariance of theta is the inverse Hessian at the MAP. Every
        # loss reaches theta only through the scores, so H = P^T H_r P + prior.
        r = (off + P @ w).detach()
        H_r = torch.zeros(len(P), len(P), dtype=torch.float64)
        for rows, _ in signals.choices:
            idx = torch.tensor(rows)
            p = F.softmax(r[idx], dim=0).double()
            H_r[idx[:, None], idx[None, :]] += torch.diag(p) - torch.outer(p, p)
        if len(t_rows) or len(r_rows):
            leaf = r.clone().requires_grad_(True)
            (grad,) = torch.autograd.grad(separable_nll(leaf), leaf, create_graph=True)
            (diag,) = torch.autograd.grad(grad.sum(), leaf)
            H_r += torch.diag(diag.double())
        Pd = P.double()
        hessian = Pd.T @ H_r @ Pd + torch.diag(2 * l2.double())

        self.w = w.detach().numpy()
        self.cov = torch.linalg.inv(hessian).float().numpy()
        self.bias = bias.item()
        self.cutpoints = _ordered(cut_raw).detach().numpy()
        self.fitted = True


def _ordered(cut_raw: torch.Tensor) -> torch.Tensor:
    return torch.cat([cut_raw[:1], cut_raw[:1] + F.softplus(cut_raw[1:]).cumsum(dim=0)])

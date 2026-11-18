"""The reward model recovers a hidden taste from each feedback type, and one that shifts with the condition."""

import numpy as np
import pytest

from art_critic.reward import RewardModel, Signals, pair_features
from art_critic.synthetic import SyntheticUser

N, DIM, ROUNDS = 300, 64, 40


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    rank = lambda v: np.argsort(np.argsort(v))
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def unit_scale(Z: np.ndarray) -> np.ndarray:
    return (Z / np.sqrt((Z**2).sum(axis=1).mean())).astype(np.float32)


@pytest.fixture
def world():
    rng = np.random.default_rng(0)
    # Decaying variance per dimension, like PCA features.
    Z = unit_scale(rng.standard_normal((N, DIM)) * np.linspace(1.0, 0.2, DIM))
    utility = Z @ rng.standard_normal(DIM)
    return Z, utility, SyntheticUser(utility, rng), rng


def collect(user, rng, kinds, rounds=ROUNDS, n=N) -> Signals:
    signals = Signals()
    for _ in range(rounds):
        rows = rng.choice(n, 4, replace=False).tolist()
        if "pick" in kinds:
            signals.choices.append((rows, user.pick(rows)))
        if "thumbs" in kinds:
            signals.thumbs += [(r, user.thumb(r)) for r in rows]
        if "stars" in kinds:
            signals.ratings += [(r, user.stars(r)) for r in rows]
    return signals


@pytest.mark.parametrize("kinds", [{"pick"}, {"thumbs"}, {"stars"}, {"pick", "thumbs", "stars"}])
def test_learns_hidden_taste(world, kinds):
    Z, utility, user, rng = world
    model = RewardModel(DIM)
    model.fit(Z, collect(user, rng, kinds))
    mean, _ = model.predict(Z)
    assert spearman(mean, utility) > 0.4

    # Fresh rounds the model never trained on: beat the 25% chance rate clearly.
    held_out = [rng.choice(N, 4, replace=False) for _ in range(400)]
    hits = np.mean([np.argmax(mean[rows]) == np.argmax(utility[rows]) for rows in held_out])
    assert hits > 0.4


def test_combined_beats_picks_alone(world):
    Z, utility, user, rng = world
    scores = {}
    for name, kinds in [("pick", {"pick"}), ("all", {"pick", "thumbs", "stars"})]:
        model = RewardModel(DIM)
        model.fit(Z, collect(SyntheticUser(utility, np.random.default_rng(2)), np.random.default_rng(1), kinds))
        scores[name] = spearman(model.predict(Z)[0], utility)
    assert scores["all"] > scores["pick"]


def test_cutpoints_stay_ordered(world):
    Z, _, user, rng = world
    model = RewardModel(DIM)
    model.fit(Z, collect(user, rng, {"stars"}))
    assert (np.diff(model.cutpoints) > 0).all()


def test_no_feedback_means_the_offset_alone(world):
    Z, *_ = world
    model = RewardModel(DIM)
    model.fit(Z, Signals())
    offset = np.arange(N, dtype=np.float32)
    assert not model.fitted
    assert not model.predict(Z)[0].any() and (model.predict(Z, offset)[0] == offset).all()


def test_uncertainty_shrinks_with_data_and_is_lowest_on_judged_items(world):
    Z, _, user, rng = world
    none, few, many = RewardModel(DIM), RewardModel(DIM), RewardModel(DIM)
    signals = collect(user, rng, {"stars"}, rounds=3)
    few.fit(Z, signals)
    many.fit(Z, collect(user, rng, {"pick", "thumbs", "stars"}, rounds=60))
    prior, some, lots = (m.predict(Z)[1] for m in (none, few, many))
    assert (some <= prior + 1e-5).all() and prior.mean() > some.mean() > lots.mean()

    judged = sorted({row for row, _ in signals.ratings})
    drop = 1 - some / prior
    assert drop[judged].mean() > 3 * np.delete(drop, judged).mean()


def test_laplace_covariance_matches_autograd_hessian(world):
    """The closed-form Hessian is the real one, for all three losses at once."""
    import torch

    Z, _, user, rng = world
    Zs = Z[:, :6].copy()
    signals = collect(user, rng, {"pick", "thumbs", "stars"}, rounds=8)
    model = RewardModel(6)
    model.fit(Zs, signals)

    P = torch.from_numpy(Zs).double()
    cuts = torch.from_numpy(model.cutpoints).double()

    def loss(w):
        r = P @ w
        out = 0.02 * (w**2).sum()
        for rows, pos in signals.choices:
            out = out - torch.log_softmax(r[rows], dim=0)[pos]
        for row, sign in signals.thumbs:
            out = out + torch.nn.functional.softplus(-sign * (r[row] - model.bias))
        for row, star in signals.ratings:
            cdf = torch.cat([torch.zeros(1), torch.sigmoid(cuts - r[row]), torch.ones(1)])
            out = out - torch.log(cdf[star] - cdf[star - 1])
        return out

    hessian = torch.autograd.functional.hessian(loss, torch.from_numpy(model.w).double())
    np.testing.assert_allclose(model.cov, torch.linalg.inv(hessian).numpy(), rtol=2e-3, atol=1e-4)


def test_learns_taste_that_shifts_with_the_condition():
    """A prompt-blind model captures the shared taste; only the interaction block captures the shift."""
    rng = np.random.default_rng(3)
    n, d, n_cond, k = 150, 12, 10, 4
    Zx = unit_scale(rng.standard_normal((n, d)))
    Zc = rng.standard_normal((n_cond, k))
    Zc /= np.linalg.norm(Zc, axis=1, keepdims=True)
    match = rng.standard_normal((n_cond, n)).astype(np.float32)  # unrelated to taste here
    general, shift = rng.standard_normal(d), rng.standard_normal((k, d))
    # Hidden preference: a shared part plus a part that depends on the condition.
    utility = (Zx @ general)[None, :] + 1.5 * (Zc @ shift @ Zx.T)  # (n_cond, n)
    flat = lambda c, rows: c * n + np.asarray(rows)
    user = SyntheticUser(utility.ravel(), rng, noise=0.3)

    signals = Signals()
    for _ in range(160):
        c = int(rng.integers(n_cond))
        rows = flat(c, rng.choice(n, 4, replace=False)).tolist()
        signals.choices.append((rows, user.pick(rows)))
        signals.ratings += [(r, user.stars(r)) for r in rows]

    held_out = [(int(rng.integers(n_cond)), rng.choice(n, 4, replace=False)) for _ in range(800)]
    accuracy = {}
    for name, interaction in [("blind", False), ("conditioned", True)]:
        Phi = np.concatenate([pair_features(match[c], Zx, Zc[c] if interaction else None) for c in range(n_cond)])
        model = RewardModel(Phi.shape[1])
        model.fit(Phi, signals)
        mean, _ = model.predict(Phi)
        accuracy[name] = np.mean([np.argmax(mean[flat(c, rows)]) == np.argmax(utility[c, rows]) for c, rows in held_out])

    assert accuracy["blind"] > 0.3  # the shared taste alone beats chance
    assert accuracy["conditioned"] > accuracy["blind"] + 0.1
    assert accuracy["conditioned"] > 0.55

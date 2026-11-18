"""Frozen CLIP embeddings, and the PCA projection the reward model works in."""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MODEL, PRETRAINED = "ViT-B-32", "laion2b_s34b_b79k"


@functools.cache
def _clip():
    import open_clip
    import torch

    # Not MPS by default: torch 2.15 aborts the process inside MPS on macOS 14
    # (an uncatchable NSException), and CPU embeds a few hundred images in seconds.
    device = os.environ.get("ART_CRITIC_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL, pretrained=PRETRAINED)
    model.eval().to(device)
    return model, preprocess, open_clip.get_tokenizer(MODEL), device


def embed_images(paths: list[Path], batch: int = 32) -> np.ndarray:
    """L2-normalized image embeddings, one row per path."""
    import torch
    from PIL import Image

    model, preprocess, _, device = _clip()
    out = []
    with torch.no_grad():
        for i in range(0, len(paths), batch):
            ims = []
            for p in paths[i : i + batch]:
                with Image.open(p) as im:
                    ims.append(preprocess(im.convert("RGB")))
            feats = model.encode_image(torch.stack(ims).to(device))
            out.append(torch.nn.functional.normalize(feats, dim=-1).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def embed_texts(texts: list[str]) -> np.ndarray:
    """L2-normalized text embeddings, in the same space as the images."""
    import torch

    model, _, tokenizer, device = _clip()
    with torch.no_grad():
        feats = model.encode_text(tokenizer(texts).to(device))
        return torch.nn.functional.normalize(feats, dim=-1).cpu().numpy().astype(np.float32)


@dataclass
class Projection:
    """PCA from raw embedding space down to the reward model's feature space."""

    mean: np.ndarray  # (D,)
    components: np.ndarray  # (d, D), orthonormal rows
    scale: float

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean) @ self.components.T * self.scale).astype(np.float32)

    def to_raw(self, w: np.ndarray) -> np.ndarray:
        """The raw-space direction whose inner product ranks images like `w` does.

        w . project(x) = scale * (P^T w) . x - const, so P^T w is a valid
        max-inner-product query against the stored embeddings.
        """
        q = self.components.T @ w
        norm = np.linalg.norm(q)
        return (q / norm if norm > 0 else q).astype(np.float32)


def fit_projection(X: np.ndarray, dim: int = 64) -> Projection:
    mean = X.mean(axis=0)
    centered = X - mean
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[: min(dim, len(X) - 1)]
    Z = centered @ components.T
    # Unit mean squared norm, so the L2 strength means the same thing for any pool.
    scale = 1.0 / np.sqrt((Z**2).sum(axis=1).mean())
    return Projection(mean=mean, components=components, scale=float(scale))

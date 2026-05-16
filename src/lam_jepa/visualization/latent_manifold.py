from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import matplotlib.pyplot as plt
import torch


def _pca_2d(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(x, full_matrices=False)
    return u[:, :2] * s[:2]


def latent_to_2d(latents: torch.Tensor) -> np.ndarray:
    x = latents.detach().cpu().numpy()
    if x.ndim == 1:
        x = x[None, :]
    if x.shape[0] < 2 or x.shape[1] < 2:
        return np.zeros((x.shape[0], 2), dtype=float)
    try:
        from sklearn.decomposition import PCA  # type: ignore
        return PCA(n_components=2).fit_transform(x)
    except Exception:
        return _pca_2d(x)


def plot_latent_manifold(latents: torch.Tensor, labels: Sequence[float] | None = None, trajectory: torch.Tensor | None = None, title: str = "latent manifold", out_path: str | Path | None = None):
    xy = latent_to_2d(latents)
    fig, ax = plt.subplots(figsize=(8, 6))
    if labels is None:
        ax.scatter(xy[:, 0], xy[:, 1], s=18)
    else:
        labels_arr = np.asarray(list(labels), dtype=float)
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=labels_arr, s=18)
        fig.colorbar(sc, ax=ax)
    if trajectory is not None:
        txy = latent_to_2d(trajectory)
        ax.plot(txy[:, 0], txy[:, 1])
    ax.set_title(title)
    ax.set_xlabel("latent-1")
    ax.set_ylabel("latent-2")
    fig.tight_layout()
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=160)
    return fig, ax


def plot_intervention_heatmap(matrix: torch.Tensor, title: str = "intervention heatmap", out_path: str | Path | None = None):
    data = matrix.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(data, aspect="auto")
    fig.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("target concept")
    ax.set_ylabel("source concept")
    fig.tight_layout()
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=160)
    return fig, ax

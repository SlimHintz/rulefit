"""Winsorization and linear-term scaling utilities."""

from __future__ import annotations

import numpy as np


class Winsorizer:
    """Performs feature-wise winsorization."""

    def __init__(self, trim_quantile: float = 0.0) -> None:
        self.trim_quantile = trim_quantile
        self.winsor_lims: np.ndarray | None = None

    def train(self, x: np.ndarray) -> None:
        """Estimate winsorization lower/upper limits for each column."""
        n_features = x.shape[1]
        limits = np.empty((2, n_features), dtype=float)
        limits[0, :] = -np.inf
        limits[1, :] = np.inf
        if self.trim_quantile > 0:
            lower = np.percentile(x, self.trim_quantile * 100.0, axis=0)
            upper = np.percentile(x, 100.0 - self.trim_quantile * 100.0, axis=0)
            limits[0, :] = lower
            limits[1, :] = upper
        self.winsor_lims = limits

    def trim(self, x: np.ndarray) -> np.ndarray:
        if self.winsor_lims is None:
            raise ValueError("Winsorizer must be trained before calling trim().")
        return np.clip(x, self.winsor_lims[0, :], self.winsor_lims[1, :])


class FriedScale:
    """Scale linear terms according to Friedman et al. (2005)."""

    def __init__(self, winsorizer: Winsorizer | None = None) -> None:
        self.scale_multipliers: np.ndarray | None = None
        self.winsorizer = winsorizer

    def train(self, x: np.ndarray) -> None:
        """Estimate per-feature scale multipliers."""
        x_trimmed = self.winsorizer.trim(x) if self.winsorizer is not None else x
        multipliers = np.ones(x.shape[1], dtype=float)

        # Do not scale binary features (already indicator-like rules).
        uniq_counts = np.array([len(np.unique(x[:, i])) for i in range(x.shape[1])])
        idx = uniq_counts > 2
        multipliers[idx] = 0.4 / (1.0e-12 + np.std(x_trimmed[:, idx], axis=0))

        self.scale_multipliers = multipliers

    def scale(self, x: np.ndarray) -> np.ndarray:
        if self.scale_multipliers is None:
            raise ValueError("FriedScale must be trained before calling scale().")
        base = self.winsorizer.trim(x) if self.winsorizer is not None else x
        return base * self.scale_multipliers

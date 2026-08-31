"""Latent-space comparison: PCA and spherical PGA.

Direct component-score comparisons, with no graph inference at all
(mirrors what ``SPAICOM/semasia-datasets``'s own
``plot_correlation_heatmap.py`` does for PCA -- see
``src/plotting/latent.py::plot_pc_correlation_heatmap`` in that repo).

Principal Geodesic Analysis (PGA, Fletcher et al. 2004) generalizes PCA to
Riemannian manifolds via the tangent-space construction: Frechet mean,
Log map to the tangent space at the mean, ordinary PCA in that (Euclidean)
tangent space, Exp map back for the principal geodesics. For a *flat*
(Euclidean) latent space this is mathematically identical to ordinary
PCA -- Log/Exp are the identity -- so a meaningful PGA needs a curved
manifold. This module implements it on the unit hypersphere, which
matches the cosine-similarity geometry already used elsewhere in this
pipeline (e.g. ``LatentSpace.relative(mode='cosine')``).
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

# Re-exported: the geometry itself lives in :mod:`src.manifold`, which
# `src.alignment.preprocessing` also imports (importing it from here
# instead would close a cycle through `src.latent.space`).
from ..manifold import (
    spherical_exp_map,
    spherical_frechet_mean,
    spherical_log_map,
)

__all__ = [
    'pca_components',
    'pga_components',
    'spherical_exp_map',
    'spherical_frechet_mean',
    'spherical_log_map',
]


def pca_components(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Ordinary PCA of a point cloud.

    Parameters
    ----------
    X : np.ndarray, shape (n_points, n_features)
    k : int
        Number of components.

    Returns
    -------
    scores : np.ndarray, shape (n_points, k)
        Points projected onto the top-k principal components.
    singular_values : np.ndarray, shape (k,)
        Singular values of the centered data, descending (same
        quantity -- not ``explained_variance_`` -- as returned by
        :func:`pga_components`, so the two "spectra" are comparable).
    """
    k_eff = min(k, X.shape[0] - 1, X.shape[1])
    # svd_solver='full' (exact, deterministic): 'auto' silently switches to
    # the randomized solver at this k << min(n, d) regime, which is not
    # reproducible run-to-run without a fixed random_state.
    pca = PCA(n_components=k_eff, svd_solver='full')
    scores = pca.fit_transform(np.asarray(X, dtype=np.float64))
    return scores, pca.singular_values_


def pga_components(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Principal Geodesic Analysis of a point cloud, on the unit hypersphere.

    Projects rows of ``X`` onto the unit sphere, computes their Frechet
    mean, maps to the tangent space at the mean (Euclidean by
    construction), and runs ordinary PCA there.

    Parameters
    ----------
    X : np.ndarray, shape (n_points, n_features)
    k : int
        Number of principal geodesic components.

    Returns
    -------
    scores : np.ndarray, shape (n_points, k)
        Tangent-space PCA scores (coordinates along each principal
        geodesic).
    singular_values : np.ndarray, shape (k,)
        Singular values of the tangent-space PCA, descending.
    """
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    mu = spherical_frechet_mean(Xn)
    tangent = spherical_log_map(mu, Xn)
    k_eff = min(k, tangent.shape[0] - 1, tangent.shape[1])
    pca = PCA(n_components=k_eff, svd_solver='full')
    scores = pca.fit_transform(tangent)
    singular_values = pca.singular_values_
    return scores, singular_values

"""Riemannian primitives on the unit hypersphere.

Frechet mean and the Log/Exp maps that :func:`src.latent.pga_components`
and the ``'pga'`` method of :class:`src.alignment.LatentScaler` both need.

This lives outside both packages for the same reason :mod:`src.kernels`
does: :mod:`src.latent.space` imports from :mod:`src.alignment`, so a
preprocessing step reaching back into :mod:`src.latent` would close an
import cycle. The geometry belongs to neither package, so it sits above
both.

The sphere is the manifold of choice here because it matches the
cosine-similarity geometry already used elsewhere in the pipeline
(``LatentSpace.relative(mode='cosine')``, the ``'cosine'`` kernel). On a
*flat* space PGA degenerates to PCA -- Log and Exp are the identity --
so a curved manifold is what makes it a different method at all.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    'spherical_exp_map',
    'spherical_frechet_mean',
    'spherical_log_map',
]


def spherical_frechet_mean(
    X: np.ndarray, max_iter: int = 100, tol: float = 1e-8
) -> np.ndarray:
    """Karcher/Frechet mean of unit vectors on the hypersphere.

    Iterates: average the tangent vectors (Log map at the current
    estimate), take a small step along their mean in the tangent space,
    Exp map back to the sphere. Converges to the point minimizing the sum
    of squared geodesic distances.

    Parameters
    ----------
    X : np.ndarray, shape (n_points, n_features)
        Points on the unit sphere (rows are unit-norm; non-unit-norm
        rows are normalized first).
    max_iter : int, default=100
    tol : float, default=1e-8
        Stop when the update's norm drops below this.

    Returns
    -------
    np.ndarray, shape (n_features,)
        Unit-norm Frechet mean.
    """
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    mu = Xn.mean(axis=0)
    mu /= np.linalg.norm(mu)
    for _ in range(max_iter):
        tangent = spherical_log_map(mu, Xn)
        step = tangent.mean(axis=0)
        if np.linalg.norm(step) < tol:
            break
        mu = spherical_exp_map(mu, step[None, :])[0]
    return mu


def spherical_log_map(mu: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Riemannian log map at ``mu`` on the unit hypersphere.

    Parameters
    ----------
    mu : np.ndarray, shape (n_features,)
        Unit-norm base point.
    X : np.ndarray, shape (n_points, n_features)
        Unit-norm points to map into the tangent space at ``mu``.

    Returns
    -------
    np.ndarray, shape (n_points, n_features)
        Tangent vectors at ``mu`` (each orthogonal to ``mu``).
    """
    cos_theta = np.clip(X @ mu, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    diff = X - cos_theta[:, None] * mu[None, :]
    diff_norm = np.linalg.norm(diff, axis=1)
    scale = np.divide(
        theta, diff_norm, out=np.zeros_like(theta), where=diff_norm > 1e-12
    )
    return scale[:, None] * diff


def spherical_exp_map(mu: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Riemannian exp map at ``mu`` on the unit hypersphere.

    Parameters
    ----------
    mu : np.ndarray, shape (n_features,)
        Unit-norm base point.
    V : np.ndarray, shape (n_points, n_features)
        Tangent vectors at ``mu`` (orthogonal to ``mu``).

    Returns
    -------
    np.ndarray, shape (n_points, n_features)
        Unit-norm points on the sphere.
    """
    norms = np.linalg.norm(V, axis=1)
    safe_norms = np.where(norms > 1e-12, norms, 1.0)
    direction = V / safe_norms[:, None]
    out = (
        np.cos(norms)[:, None] * mu[None, :]
        + np.sin(norms)[:, None] * direction
    )
    return out / np.linalg.norm(out, axis=1, keepdims=True)

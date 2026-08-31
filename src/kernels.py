"""Kernel functions shared by the RKHS residual stage and pilot selection.

A :class:`Kernel` is a small, picklable descriptor of the kernel *and* its
fitted hyper-parameters. Bandwidths are fitted once on the calibration
cloud (median-pairwise-distance heuristic) and then reused verbatim for
out-of-sample points, exactly like the whitening statistics.

This lives outside :mod:`src.alignment` so that
:class:`src.anchors.Anchor` can select kernel-herding pilots in the same
RKHS geometry the aligner will later fit in, without the two packages
importing each other.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np

log = logging.getLogger(__name__)

KernelName = Literal['rbf', 'laplacian', 'polynomial', 'linear', 'cosine']

__all__ = ['Kernel', 'KernelName', 'median_squared_distance']


def _squared_distances(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pairwise squared Euclidean distances, shape ``(len(A), len(B))``."""
    sq = (
        np.sum(A**2, axis=1)[:, None]
        + np.sum(B**2, axis=1)[None, :]
        - 2.0 * (A @ B.T)
    )
    return np.maximum(sq, 0.0)


def median_squared_distance(
    X: np.ndarray, max_points: int = 2000, seed: int = 42
) -> float:
    """Median pairwise squared distance (the classic bandwidth heuristic).

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
    max_points : int, default=2000
        Sub-sample size used to keep the estimate ``O(max_points^2)``.
    seed : int, default=42
        Sub-sampling seed.

    Returns
    -------
    float
        Strictly positive median of the off-diagonal squared distances.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        X = X[rng.choice(X.shape[0], size=max_points, replace=False)]
    sq = _squared_distances(X, X)
    iu = np.triu_indices(sq.shape[0], k=1)
    med = float(np.median(sq[iu])) if iu[0].size else 1.0
    return med if med > 0 else 1.0


class Kernel:
    """Kernel with fittable bandwidth.

    Parameters
    ----------
    name : {'rbf', 'laplacian', 'polynomial', 'linear', 'cosine'}
        Kernel family. ``'rbf'`` is the safe default.
    gamma : float, optional
        Inverse bandwidth of ``'rbf'``/``'laplacian'``. ``None`` (the
        default) fits it from the calibration cloud with the median
        heuristic: ``gamma = 1 / median_squared_distance``.
    degree : int, default=3
        Degree of the polynomial kernel.
    coef0 : float, default=1.0
        Offset of the polynomial kernel.
    bandwidth_scale : float, default=1.0
        Multiplies the median-heuristic bandwidth (``> 1`` smooths,
        ``< 1`` sharpens). Ignored when ``gamma`` is given.
    max_points : int, default=2000
        Sub-sample size of the median heuristic.
    """

    def __init__(
        self,
        name: KernelName = 'rbf',
        gamma: float | None = None,
        degree: int = 3,
        coef0: float = 1.0,
        bandwidth_scale: float = 1.0,
        max_points: int = 2000,
    ) -> None:
        if name not in ('rbf', 'laplacian', 'polynomial', 'linear', 'cosine'):
            raise ValueError(f'Unknown kernel {name!r}.')
        if name == 'polynomial' and coef0 < 0:
            # (<x,y> + c)^d is a positive-definite kernel only for c >= 0.
            # With c < 0 the Gram matrix can be indefinite, which breaks
            # the RKHS the residual stage is defined over.
            raise ValueError(
                f'polynomial kernel needs coef0 >= 0 to be positive '
                f'definite, got {coef0}.'
            )
        self.name: KernelName = name
        self.gamma = None if gamma is None else float(gamma)
        self.degree = int(degree)
        self.coef0 = float(coef0)
        self.bandwidth_scale = float(bandwidth_scale)
        self.max_points = int(max_points)

    def fit(self, X: np.ndarray, seed: int = 42) -> Kernel:
        """Fit the bandwidth on ``X`` (no-op if ``gamma`` was given).

        Parameters
        ----------
        X : np.ndarray, shape (n, d)
            Calibration cloud in the space the kernel acts on.
        seed : int, default=42
            Sub-sampling seed of the median heuristic.

        Returns
        -------
        Kernel
            ``self``.
        """
        if self.gamma is None and self.name in ('rbf', 'laplacian'):
            med = median_squared_distance(
                X, max_points=self.max_points, seed=seed
            )
            if self.name == 'laplacian':
                med = np.sqrt(med)
            self.gamma = 1.0 / (self.bandwidth_scale * med)
            log.debug(
                'Kernel %r: median heuristic -> gamma=%.6g',
                self.name,
                self.gamma,
            )
        return self

    def __call__(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Gram matrix ``k(A_i, B_j)``, shape ``(len(A), len(B))``."""
        A = np.asarray(A, dtype=np.float64)
        B = np.asarray(B, dtype=np.float64)
        if A.shape[1] != B.shape[1]:
            raise ValueError(
                f'Kernel inputs must share their dimensionality, got '
                f'{A.shape[1]} and {B.shape[1]}.'
            )
        match self.name:
            case 'rbf':
                return np.exp(-self._gamma() * _squared_distances(A, B))
            case 'laplacian':
                dist = np.sqrt(_squared_distances(A, B))
                return np.exp(-self._gamma() * dist)
            case 'polynomial':
                return (A @ B.T + self.coef0) ** self.degree
            case 'linear':
                return A @ B.T
            case 'cosine':
                An = A / np.maximum(
                    np.linalg.norm(A, axis=1, keepdims=True), 1e-12
                )
                Bn = B / np.maximum(
                    np.linalg.norm(B, axis=1, keepdims=True), 1e-12
                )
                return An @ Bn.T
        raise ValueError(f'Unknown kernel {self.name!r}.')  # pragma: no cover

    def _gamma(self) -> float:
        if self.gamma is None:
            raise RuntimeError(
                f'Kernel({self.name!r}) has no bandwidth: call fit() first '
                'or pass gamma explicitly.'
            )
        return self.gamma

    def spec(self) -> dict[str, float | str | int | None]:
        """Constructor arguments, so an unfitted twin can be rebuilt.

        A kernel is fitted to one point cloud, and a method that lifts
        *two* spaces needs one bandwidth per space. This returns the
        recipe rather than the fitted object, so the caller can build a
        second kernel of the same family and fit it on the other side --
        the alternative, reusing one fitted bandwidth across two
        differently scaled spaces, silently measures one of them in the
        other's units.
        """
        return {
            'name': self.name,
            'gamma': self.gamma,
            'degree': self.degree,
            'coef0': self.coef0,
            'bandwidth_scale': self.bandwidth_scale,
            'max_points': self.max_points,
        }

    def summary(self) -> dict[str, float | str | int]:
        """Hyper-parameters, for experiment logging."""
        out: dict[str, float | str | int] = {'kernel': self.name}
        if self.name in ('rbf', 'laplacian'):
            out['kernel_gamma'] = float(self._gamma())
        if self.name == 'polynomial':
            out['kernel_degree'] = self.degree
            out['kernel_coef0'] = self.coef0
        return out

    def __repr__(self) -> str:
        return f'Kernel(name={self.name!r}, gamma={self.gamma})'

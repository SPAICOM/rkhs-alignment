"""Invertible per-space standardisation of a latent point cloud.

This is Step 0 of the RKA pipeline (``idea.md``): before any alignment is
attempted, each latent space is standardised *independently* using
calibration statistics, so that whatever misalignment remains is a genuine
coordinate-system mismatch rather than a scale/correlation mismatch.

:class:`LatentScaler` keeps the fitted statistics, so the very same
transform (and its exact inverse) can be applied to out-of-sample points
later -- the "whitening consistency" requirement of the pipeline: test
points must never re-estimate their own statistics.

The ``'whiten'`` method produces ``Z`` with ``(1/N) Z^T Z = I_d`` (the
``ddof=0`` convention of ``idea.md``, not numpy's ``ddof=1``) via the
symmetric (ZCA) inverse square root of the covariance -- the unique
whitening matrix that is itself symmetric, and so the one that rotates
the cloud least (Kessy, Lewandowski & Strimmer, *Optimal Whitening and
Decorrelation*, The American Statistician 72(4), 2018). ``'pca'`` is the
same factorisation stopped one step earlier: a rotation onto the
principal axes with the variances left alone. ``'pga'`` replaces the
linear chart with a Riemannian one on the unit sphere (Fletcher et al.,
*Principal Geodesic Analysis for the Study of Nonlinear Statistics of
Shape*, IEEE TMI 23(8), 2004), carrying the radius as an extra
coordinate so that it, too, inverts exactly. The ridge is
*relative* to the average feature variance, so a single ``eps`` behaves
identically across encoders whose latents differ in scale by orders of
magnitude, and the covariance is Ledoit-Wolf shrunk by default -- without
which a transform fitted on ``n ~ d`` samples does not survive contact
with out-of-sample points (see ``shrinkage``).
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
from sklearn.covariance import ledoit_wolf_shrinkage

from ..manifold import (
    spherical_exp_map,
    spherical_frechet_mean,
    spherical_log_map,
)

log = logging.getLogger(__name__)

# Below this norm a point has no usable direction on the sphere.
_MIN_RADIUS = 1e-12

ScalingMethod = Literal['none', 'center', 'standard', 'pca', 'pga', 'whiten']

__all__ = ['LatentScaler', 'ScalingMethod']


def _deterministic_signs(basis: np.ndarray) -> np.ndarray:
    """Fix each column's sign so a basis does not flip between runs.

    ``eigh``/``svd`` leave the sign of every axis to LAPACK, so the same
    data can produce ``+v`` on one machine and ``-v`` on another. The
    convention here is scikit-learn's: make the largest-magnitude entry
    of each column positive. ZCA is unaffected either way (``U D U^T``
    cannot see a column flip), but a PCA or PGA basis can.
    """
    pivot = np.argmax(np.abs(basis), axis=0)
    signs = np.sign(basis[pivot, np.arange(basis.shape[1])])
    signs[signs == 0] = 1.0
    return basis * signs


class LatentScaler:
    """Fitted, invertible standardisation of a latent point cloud.

    Every method is exactly invertible, which is what the transmit ->
    align -> receive pipeline needs: the sender's transform is undone on
    the receiver's side, so any loss here is loss in the delivered latent.
    All but ``'pga'`` are affine, ``Z = (X - mean) @ W`` with the inverse
    ``X = Z @ W_inv + mean``; ``'pga'`` is a Riemannian chart and inverts
    through the Exp map instead. Truncation (``n_components``) is the one
    deliberate exception: it makes the round trip a projection onto the
    retained subspace. :attr:`is_invertible` says which case you are in.

    Parameters
    ----------
    method : ScalingMethod, default='whiten'
        - ``'none'``     : identity (mean kept at zero).
        - ``'center'``   : subtract the feature means.
        - ``'standard'`` : per-feature z-scoring.
        - ``'pca'``      : mean-centred and rotated onto the principal
          axes, variances left as they are. Equivalently the (centred)
          Karhunen-Loeve transform; the two names describe one map.
        - ``'pga'``      : Principal Geodesic Analysis on the unit
          hypersphere (Fletcher et al. 2004).
        - ``'whiten'``   : mean-centred with identity covariance (ZCA).

        ``'pca'`` and ``'whiten'`` share a factorisation and differ only
        in the last step: PCA stops after the rotation, whitening goes on
        to divide each retained direction by ``sqrt(lambda)``. So PCA
        *decorrelates* (``cov(Z)`` is diagonal) where whitening
        *equalises* (``cov(Z) = I``). Keep PCA when the relative scale of
        the directions is signal -- an RBF bandwidth is one number for
        every direction, so whitening tells it that a direction carrying
        1% of the variance deserves the same bandwidth as the leading
        one, which PCA does not.

        Centring is not optional for ``'pca'``: the classical transform
        is defined on the covariance, and running it on the raw second
        moment instead returns the *mean direction* as the leading axis
        rather than the direction of largest variation.
    eps : float, default=1e-6
        Floor applied to the covariance eigenvalues before inversion.
        Only used by ``'whiten'``; guards against rank-deficient
        covariances (which happen whenever ``n_points < n_features``).
        ``'pca'`` and ``'pga'`` invert nothing, so they need no floor.
    eps_mode : {'relative', 'absolute'}, default='relative'
        How ``eps`` is read. ``'relative'`` scales it by the mean feature
        variance, so one value behaves the same across encoders whose
        latents differ in scale by orders of magnitude. ``'absolute'``
        uses it as a bare eigenvalue floor, which is what a reference
        implementation writing ``np.maximum(eigvals, 1e-6)`` does; pick it
        when reproducing such a pipeline exactly.
    shrinkage : {'auto'} | float | None, default='auto'
        Covariance shrinkage toward a scaled identity, for ``'whiten'``.
        ``'auto'`` uses the Ledoit-Wolf estimate of the optimal
        intensity, ``None`` or ``0.0`` disables it.

        This is not cosmetic. Whitening estimated from ``n ~ d`` samples
        is the pathological case: the sample covariance is *technically*
        full rank but its smallest eigenvalues are pure sampling noise,
        so inverting its square root produces a transform that is fine on
        the samples it was fitted to and catastrophic out of sample --
        measured on SEMASIA ViT pairs, held-out NMSE blows up by two
        orders of magnitude at exactly ``n = d``. A fixed ridge cannot
        fix this because the problem is mis-estimated *directions*, not a
        small determinant. Ledoit-Wolf shrinkage adapts to the regime
        (intensity ~0.6 at ``n = d/30``, ~0.02 at ``n = 4d``), so small
        pilot budgets stay stable and large ones are left alone.

        Ignored by ``'pca'`` and ``'pga'``, and not out of laziness:
        shrinking toward a *scaled identity* leaves the eigenvectors of
        the covariance exactly where they were, so it cannot change a
        principal-axis basis. It only matters when the eigenvalues are
        inverted, which is whitening.
    n_components : int | float | None, default=None
        Rank of the transform. ``None`` keeps every direction; an ``int``
        keeps the top-``k`` principal directions; a ``float`` in
        ``(0, 1]`` keeps as many as explain that fraction of the variance.
        Used by ``'pca'``, ``'pga'`` and ``'whiten'``.

        For ``'pga'`` this counts *geodesic* components, of which there
        are at most ``d - 1`` (the tangent space at a point of the sphere
        is one dimension smaller). The radial coordinate is always kept
        on top, so ``out_dim`` is ``k + 1`` and a full-rank PGA maps
        ``d`` dimensions to ``d``.

        Full-rank whitening equalises *every* direction, which is right
        when each carries signal -- the 12- and 16-dimensional latents the
        RKA paper works with -- and actively harmful on a 768-dimensional
        encoder latent whose spectrum decays fast, because it amplifies
        several hundred noise directions up to the scale of the signal.
        Truncating also puts the kernel residual back in a regime where it
        can generalise: an RBF fitted on 768 whitened dimensions from a
        thousand pilots interpolates and predicts nothing out of sample.

    Notes
    -----
    With ``n_components`` set, the transform is no longer square:
    :meth:`transform` returns ``k`` columns and
    :meth:`inverse_transform` maps them back to the original ``d``. The
    round trip is then a projection, exact only on the retained subspace.
    """

    def __init__(
        self,
        method: ScalingMethod = 'whiten',
        eps: float = 1e-6,
        shrinkage: str | float | None = 'auto',
        n_components: float | None = None,
        eps_mode: str = 'relative',
    ) -> None:
        if method not in (
            'none',
            'center',
            'standard',
            'pca',
            'pga',
            'whiten',
        ):
            raise ValueError(f'Unknown scaling method {method!r}.')
        if eps_mode not in ('relative', 'absolute'):
            raise ValueError(
                f"eps_mode must be 'relative' or 'absolute', got {eps_mode!r}."
            )
        if isinstance(shrinkage, str) and shrinkage != 'auto':
            raise ValueError(
                f"shrinkage must be 'auto', a float, or None; got "
                f'{shrinkage!r}.'
            )
        self.method: ScalingMethod = method
        self.eps = float(eps)
        self.eps_mode = eps_mode
        self.shrinkage = shrinkage
        self.n_components = n_components

        self.mean_: np.ndarray | None = None
        self.forward_: np.ndarray | None = None
        self.inverse_: np.ndarray | None = None
        self.shrinkage_: float = 0.0
        self.n_components_: int | None = None
        self.explained_variance_ratio_: float = 1.0
        self.out_dim_: int | None = None
        # 'pga' only: the chart it inverts through.
        self.sphere_mean_: np.ndarray | None = None
        self.tangent_basis_: np.ndarray | None = None
        self.log_radius_mean_: float = 0.0

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> LatentScaler:
        """Estimate the transform on the calibration point cloud ``X``.

        Parameters
        ----------
        X : np.ndarray, shape (n_points, n_features)

        Returns
        -------
        LatentScaler
            ``self``.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f'X must be 2-dimensional, got shape {X.shape}.')
        n, d = X.shape

        self.out_dim_ = None
        self.sphere_mean_ = None
        self.tangent_basis_ = None
        self.log_radius_mean_ = 0.0
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_

        match self.method:
            case 'none':
                self.mean_ = np.zeros(d)
                self.forward_ = np.eye(d)
                self.inverse_ = np.eye(d)
            case 'center':
                self.forward_ = np.eye(d)
                self.inverse_ = np.eye(d)
            case 'standard':
                std = Xc.std(axis=0)
                std[std <= 0] = 1.0
                self.forward_ = np.diag(1.0 / std)
                self.inverse_ = np.diag(std)
            case 'pca':
                # Rotate the *centred* cloud onto its principal axes and
                # keep the leading directions. Nothing is inverted, so
                # there is no ridge and no shrinkage -- the transform is a
                # (possibly truncated) rotation, and its own inverse.
                evals, evecs = self._principal_axes(Xc, n, d, shrink=False)
                total = float(evals.sum())
                k = self._rank(evals, total, d)
                if k >= n:
                    log.warning(
                        'PCA keeping %d directions from %d points: the '
                        'centred cloud spans at most %d, so the trailing '
                        'axes are arbitrary.',
                        k,
                        n,
                        max(n - 1, 0),
                    )
                self.n_components_ = k
                self.explained_variance_ratio_ = float(
                    evals[:k].sum() / max(total, 1e-30)
                )
                # Orthonormal columns, so the inverse is the transpose:
                # exact when k == d, a projection onto the retained
                # subspace otherwise.
                self.forward_ = evecs[:, :k]
                self.inverse_ = evecs[:, :k].T
            case 'pga':
                # Principal Geodesic Analysis on the unit hypersphere
                # (Fletcher et al., IEEE TMI 23(8), 2004): Frechet mean,
                # Log map to the tangent space there, ordinary PCA in that
                # (flat) tangent space, Exp map to invert.
                #
                # The sphere discards the radius, and this transform has
                # to be invertible, so the radius rides along as an extra
                # coordinate: the chart is the product R+ x S^(d-1), i.e.
                # polar coordinates with PGA on the angular part. Log
                # radius rather than radius, so the coordinate is additive
                # like the tangent ones and a scale mismatch between the
                # two spaces shows up as an offset.
                self.mean_ = np.zeros(d)  # centring here is intrinsic
                radius = np.linalg.norm(X, axis=1)
                usable = radius > _MIN_RADIUS
                if not usable.all():
                    log.warning(
                        '%d of %d points sit at (numerically) the origin, '
                        'where the sphere has no well-defined direction; '
                        'they are dropped from the chart estimate and '
                        'transform to the mean direction.',
                        int((~usable).sum()),
                        n,
                    )
                if usable.sum() < 2:
                    raise ValueError(
                        f"'pga' needs at least 2 points away from the "
                        f'origin, got {int(usable.sum())} of {n}.'
                    )
                # Degenerate rows would be 0/0; estimate the chart from
                # the rest and let them sit at the mean direction, which
                # is the only choice that keeps the radius round-tripping.
                self.sphere_mean_ = spherical_frechet_mean(
                    X[usable] / radius[usable, None]
                )
                self.log_radius_mean_ = float(np.log(radius[usable]).mean())

                tangent = spherical_log_map(
                    self.sphere_mean_, self._unit_directions(X, radius)
                )
                spread = np.linalg.norm(tangent, axis=1).max()
                if spread > 0.5 * np.pi:
                    log.warning(
                        'PGA tangent radius reaches %.2f rad (> pi/2): the '
                        'cloud wraps far enough around the sphere that the '
                        'tangent approximation is strained, and points at '
                        'pi from the mean have no unique Log.',
                        spread,
                    )

                # Tangent vectors are orthogonal to the mean, so their
                # covariance is singular along it by construction; the
                # tangent space has d - 1 usable dimensions.
                evals, evecs = self._principal_axes(
                    tangent, n, d, shrink=False
                )
                total = float(evals.sum())
                k = min(self._rank(evals, total, d), max(d - 1, 1))
                self.n_components_ = k
                self.explained_variance_ratio_ = float(
                    evals[:k].sum() / max(total, 1e-30)
                )
                self.tangent_basis_ = evecs[:, :k]
                # Affine slots stay empty: this branch is not a matrix.
                self.forward_ = None
                self.inverse_ = None
                self.out_dim_ = k + 1  # + the radial coordinate
            case 'whiten':
                if n < d:
                    log.warning(
                        'Whitening a %d-dimensional space from only %d '
                        'points: the covariance is rank-deficient, so the '
                        'shrinkage carries %d directions.',
                        d,
                        n,
                        d - n,
                    )
                evals, evecs = self._principal_axes(Xc, n, d, shrink=True)
                # trace(cov) is the eigenvalue sum, taken before the floor.
                ridge = (
                    self.eps
                    if self.eps_mode == 'absolute'
                    else self.eps * max(float(evals.sum()) / d, 1e-30)
                )
                evals = np.clip(evals, ridge, None)

                total = float(evals.sum())
                k = self._rank(evals, total, d)
                self.n_components_ = k
                self.explained_variance_ratio_ = float(
                    evals[:k].sum() / max(total, 1e-30)
                )
                evals, evecs = evals[:k], evecs[:, :k]

                # Square (k == d) reproduces the symmetric ZCA whitening;
                # truncated, it is PCA whitening onto the kept subspace and
                # inverse_ lifts back out of it. Either way this is the
                # 'pca' rotation above followed by a diagonal rescaling.
                self.forward_ = evecs * evals**-0.5
                self.inverse_ = (evecs * evals**0.5).T
                if k == d:
                    self.forward_ = self.forward_ @ evecs.T
                    self.inverse_ = evecs @ self.inverse_

        if self.out_dim_ is None:
            self.out_dim_ = int(self.forward_.shape[1])
        return self

    def _principal_axes(
        self, Xc: np.ndarray, n: int, d: int, shrink: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        """Covariance eigenpairs of centred data, largest eigenvalue first.

        Rank truncation takes a prefix, so the leading directions have to
        come first; ``eigh`` returns ascending order.

        Two routes. Shrinkage needs the covariance matrix itself, so that
        path forms it and calls ``eigh``. Without shrinkage the SVD of the
        centred data is used instead: forming ``Xc^T Xc`` squares the
        condition number, and past ``cond(Xc) ~ 1e8`` the trailing axes
        from ``eigh`` are visibly wrong while the SVD's are not.
        """
        if shrink:
            cov = self._covariance(Xc, n, d)
            evals, evecs = np.linalg.eigh(cov)
            order = np.argsort(evals)[::-1]
            evals, evecs = evals[order], evecs[:, order]
        else:
            self.shrinkage_ = 0.0
            _, sv, Vt = np.linalg.svd(Xc, full_matrices=True)
            # Pad: svd returns min(n, d) singular values, and a truncation
            # rule still has to see the empty directions as zero-variance.
            evals = np.zeros(d)
            evals[: sv.size] = sv**2 / n
            evecs = Vt.T
        return evals, _deterministic_signs(evecs)

    def _rank(self, evals: np.ndarray, total: float, d: int) -> int:
        """Number of principal directions the whitening keeps."""
        if self.n_components is None:
            return d
        if isinstance(self.n_components, (int, np.integer)):
            if self.n_components < 1:
                raise ValueError(
                    f'n_components must be >= 1, got {self.n_components}.'
                )
            return int(min(self.n_components, d))
        fraction = float(self.n_components)
        if not 0.0 < fraction <= 1.0:
            raise ValueError(
                f'A fractional n_components must lie in (0, 1], got '
                f'{fraction}.'
            )
        kept = np.cumsum(evals) / max(total, 1e-30)
        return int(np.searchsorted(kept, fraction) + 1)

    def _covariance(self, Xc: np.ndarray, n: int, d: int) -> np.ndarray:
        """Sample covariance, optionally shrunk toward a scaled identity."""
        sample = (Xc.T @ Xc) / n
        if not self.shrinkage:
            self.shrinkage_ = 0.0
            return sample

        if self.shrinkage == 'auto':
            self.shrinkage_ = float(
                ledoit_wolf_shrinkage(Xc, assume_centered=True)
            )
        else:
            self.shrinkage_ = float(np.clip(self.shrinkage, 0.0, 1.0))

        # Ledoit-Wolf target: the identity scaled to preserve the trace.
        mu = float(np.trace(sample)) / d
        a = self.shrinkage_
        return (1.0 - a) * sample + a * mu * np.eye(d)

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Standardise ``X`` with the fitted statistics."""
        self._check_fitted()
        X = np.asarray(X, dtype=np.float64)
        if X.shape[1] != self.mean_.shape[0]:
            raise ValueError(
                f'X has {X.shape[1]} features, scaler was fitted on '
                f'{self.mean_.shape[0]}.'
            )
        if self.method == 'pga':
            return self._pga_forward(X)
        return (X - self.mean_) @ self.forward_

    def _unit_directions(
        self, X: np.ndarray, radius: np.ndarray
    ) -> np.ndarray:
        """Directions of ``X``, with origin points placed at the mean."""
        unit = np.tile(self.sphere_mean_, (X.shape[0], 1))
        usable = radius > _MIN_RADIUS
        unit[usable] = X[usable] / radius[usable, None]
        return unit

    def _pga_forward(self, X: np.ndarray) -> np.ndarray:
        """Polar chart: log-radius, then coordinates along the geodesics."""
        radius = np.linalg.norm(X, axis=1)
        unit = self._unit_directions(X, radius)
        radius = np.maximum(radius, _MIN_RADIUS)
        tangent = spherical_log_map(self.sphere_mean_, unit)
        return np.column_stack(
            (
                np.log(radius) - self.log_radius_mean_,
                tangent @ self.tangent_basis_,
            )
        )

    def _pga_inverse(self, Z: np.ndarray) -> np.ndarray:
        """Undo :meth:`_pga_forward`: Exp map, then restore the radius."""
        radius = np.exp(Z[:, 0] + self.log_radius_mean_)
        tangent = Z[:, 1:] @ self.tangent_basis_.T
        unit = spherical_exp_map(self.sphere_mean_, tangent)
        return radius[:, None] * unit

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """Map standardised points back to the raw latent space.

        Exact for every method at full rank -- this is the receiver's half
        of the transmit -> align -> receive pipeline, so anything lost
        here is lost from the delivered latent. With ``n_components`` set
        the round trip is a projection onto the retained subspace, which
        is the point of asking for it.
        """
        self._check_fitted()
        Z = np.asarray(Z, dtype=np.float64)
        if Z.shape[1] != self.out_dim:
            raise ValueError(
                f'Z has {Z.shape[1]} components, scaler produces '
                f'{self.out_dim}.'
            )
        if self.method == 'pga':
            return self._pga_inverse(Z)
        return Z @ self.inverse_ + self.mean_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """:meth:`fit` then :meth:`transform` on the same data."""
        return self.fit(X).transform(X)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has been called."""
        return self.mean_ is not None

    @property
    def dim(self) -> int:
        """Dimensionality the scaler was fitted on."""
        self._check_fitted()
        return int(self.mean_.shape[0])

    @property
    def out_dim(self) -> int:
        """Dimensionality :meth:`transform` produces."""
        self._check_fitted()
        return int(self.out_dim_)

    @property
    def is_invertible(self) -> bool:
        """Whether :meth:`inverse_transform` undoes :meth:`transform` exactly.

        False only when ``n_components`` truncated the transform, which
        turns the round trip into a projection onto the retained
        subspace. Worth checking before sending a latent through the
        transmit -> align -> receive chain, where the inverse is what the
        receiver actually decodes.
        """
        self._check_fitted()
        return self.out_dim_ >= self.dim

    def _check_fitted(self) -> None:
        if self.mean_ is None:
            raise RuntimeError('LatentScaler.fit() must be called first.')

    def __repr__(self) -> str:
        state = f'dim={self.dim}' if self.is_fitted else 'unfitted'
        return f'LatentScaler(method={self.method!r}, {state})'

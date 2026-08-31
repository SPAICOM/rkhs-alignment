"""Orthogonal Procrustes between two paired latent point clouds.

Solves, in closed form,

    Q* = argmin_Q || X_src Q^T - X_tgt ||_F      s.t.  Q semi-orthogonal

with ``Q`` of shape ``(d_tgt, d_src)``. When the two spaces have the same
dimensionality this is the classical square orthogonal Procrustes problem;
when they differ it is the *rectangular* (Stiefel) problem, and ``Q`` can
only be semi-orthogonal on its smaller side:

===================  ==========================  ==========================
 dimensions           constraint satisfied        interpretation
===================  ==========================  ==========================
 d_src == d_tgt       Q^T Q = Q Q^T = I           rotation/reflection
 d_src <  d_tgt       Q^T Q = I_{d_src}           isometric embedding
 d_src >  d_tgt       Q Q^T = I_{d_tgt}           orthogonal projection
===================  ==========================  ==========================

:func:`check_paired_dims` makes that case analysis explicit (and refuses
silently-wrong inputs), because in this project the source and target are
two *different* encoders whose latent dimensionalities routinely disagree
(e.g. ``resnet50`` at 2048 vs ``vit_small`` at 384).

Note on conventions: ``idea.md`` stacks samples as *columns*
(``X in R^(dt x N)``); everything in this package stacks them as *rows*
(``X in R^(N x dt)``), matching :class:`src.latent.space.LatentSpace`. The
two are related by a transpose, so ``C = Y X^T`` there is
``X_tgt^T X_src`` here.
"""

from __future__ import annotations

import logging

import numpy as np

from .base import Aligner, check_paired_dims

log = logging.getLogger(__name__)

__all__ = ['ProcrustesAligner', 'ProcrustesFit', 'orthogonal_procrustes']


class ProcrustesFit:
    """Result of :func:`orthogonal_procrustes`.

    Attributes
    ----------
    Q : np.ndarray, shape (d_tgt, d_src)
        Semi-orthogonal map; apply it as ``X_src @ Q.T``.
    scale : float
        Optimal isotropic scale (``1.0`` unless ``scaling=True``).
    singular_values : np.ndarray, shape (min(d_src, d_tgt),)
        Singular values of the cross-covariance; their sum is the
        alignment "mass" explained by the orthogonal map.
    """

    __slots__ = ('Q', 'scale', 'singular_values')

    def __init__(
        self,
        Q: np.ndarray,
        scale: float,
        singular_values: np.ndarray,
    ) -> None:
        self.Q = Q
        self.scale = float(scale)
        self.singular_values = singular_values

    @property
    def regime(self) -> str:
        """Which semi-orthogonality the solution satisfies."""
        d_tgt, d_src = self.Q.shape
        if d_src == d_tgt:
            return 'square'
        return 'embedding' if d_src < d_tgt else 'projection'

    def apply(self, X: np.ndarray) -> np.ndarray:
        """Map source-space rows into the target space."""
        return self.scale * (np.asarray(X, dtype=np.float64) @ self.Q.T)

    def __repr__(self) -> str:
        d_tgt, d_src = self.Q.shape
        return (
            f'ProcrustesFit(d_src={d_src}, d_tgt={d_tgt}, '
            f'regime={self.regime!r}, scale={self.scale:.4f})'
        )


def orthogonal_procrustes(
    X_src: np.ndarray,
    X_tgt: np.ndarray,
    scaling: bool = False,
) -> ProcrustesFit:
    """Closed-form (rectangular) orthogonal Procrustes solution.

    Parameters
    ----------
    X_src : np.ndarray, shape (n, d_src)
        Source latents (rows are samples), assumed already centred.
    X_tgt : np.ndarray, shape (n, d_tgt)
        Paired target latents, assumed already centred.
    scaling : bool, default=False
        Also fit the optimal isotropic scale ``s`` of
        ``|| s X_src Q^T - X_tgt ||_F`` (the "similarity" rather than the
        "orthogonal" Procrustes problem). Redundant when both spaces are
        whitened.

    Returns
    -------
    ProcrustesFit
    """
    X_src = np.asarray(X_src, dtype=np.float64)
    X_tgt = np.asarray(X_tgt, dtype=np.float64)
    # The caller is usually an Aligner, which has already reported the
    # dimension regime; keep the primitive's own report at debug level.
    check_paired_dims(
        X_src,
        X_tgt,
        context='orthogonal_procrustes',
        level=logging.DEBUG,
    )

    # C = X_tgt^T X_src is `Y X^T` in the column-stacked convention of
    # idea.md; the thin SVD keeps Q rectangular (d_tgt, d_src).
    C = X_tgt.T @ X_src
    U, sigma, Vt = np.linalg.svd(C, full_matrices=False)
    Q = U @ Vt

    scale = 1.0
    if scaling:
        denom = float(np.sum(X_src**2))
        scale = float(np.sum(sigma) / denom) if denom > 0 else 1.0

    return ProcrustesFit(Q=Q, scale=scale, singular_values=sigma)


class ProcrustesAligner(Aligner):
    """Alignment by a single (rectangular) orthogonal map.

    This is the ``Q*`` stage of the RKA pipeline used on its own -- the
    "Ortho" baseline of the latent-translation literature -- and the
    ``lam -> inf`` limit of :class:`~src.alignment.rkhs.RKHSAligner`.

    Parameters
    ----------
    preprocess : ScalingMethod, default='whiten'
        Per-space standardisation (see :class:`~src.alignment.base.Aligner`).
    scaling : bool, default=False
        Fit the optimal isotropic scale as well. Whitening already
        removes global scale, so this only matters for
        ``preprocess='center'``.
    eps : float, default=1e-6
        Relative covariance ridge of the whitening step.
    seed : int, default=42
        Unused (the solution is deterministic); kept for a uniform
        constructor signature across methods.
    """

    def __init__(
        self,
        preprocess: str = 'whiten',
        scaling: bool = False,
        eps: float = 1e-6,
        n_components: float | None = None,
        shrinkage: str | float | None = 'auto',
        seed: int = 42,
    ) -> None:
        super().__init__(
            preprocess=preprocess,
            eps=eps,
            n_components=n_components,
            shrinkage=shrinkage,
            seed=seed,
        )
        self.scaling = bool(scaling)
        self.fit_: ProcrustesFit | None = None

    def hyperparameters(self) -> dict:
        params = super().hyperparameters()
        params['scaling'] = self.scaling
        return params

    @property
    def Q(self) -> np.ndarray:
        """The fitted semi-orthogonal map, shape ``(d_tgt, d_src)``."""
        self._check_fitted()
        return self.fit_.Q

    def _fit(
        self,
        Z_src: np.ndarray,
        Z_tgt: np.ndarray,
        labels: np.ndarray | None = None,
    ) -> None:
        self.fit_ = orthogonal_procrustes(Z_src, Z_tgt, scaling=self.scaling)

        n, d_src = Z_src.shape
        n_dirs = min(d_src, Z_tgt.shape[1])
        sigma = self.fit_.singular_values

        # How much of the target the rigid map actually accounts for.
        # This is the "how rigid is this pair" number to read: it is an
        # R^2, so it is bounded above by 1 whatever the standardisation
        # did, and a pair that scores low is one where a non-linear
        # correction has something left to explain.
        residual = Z_tgt - self.fit_.apply(Z_src)
        energy = float(np.sum(Z_tgt**2))
        rigid_r2 = 1.0 - float(np.sum(residual**2)) / max(energy, 1e-30)

        # The mean singular value of C/N. On *exactly* whitened spaces
        # these are cosines of the angles between matched directions and
        # the score lands in [0, 1] -- but the whitening is estimated,
        # and Ledoit-Wolf shrinkage on a wide latent (d = 4096 from 50k
        # rows) leaves the pilots over-dispersed in the weak directions,
        # which pushes this above 1. Read `procrustes_r2` instead
        # whenever this one exceeds 1; it is reporting that the
        # standardisation did not deliver unit covariance, not that the
        # pair is more than perfectly rigid.
        explained = float(np.sum(sigma) / (n * n_dirs))
        if explained > 1.0:
            log.debug(
                'procrustes_explained=%.3f exceeds 1: the whitening of a '
                '%d-dimensional source has not delivered unit covariance, '
                'so read procrustes_r2=%.3f instead.',
                explained,
                d_src,
                rigid_r2,
            )

        self.diagnostics_ = {
            'procrustes_regime': self.fit_.regime,
            'procrustes_scale': self.fit_.scale,
            'procrustes_explained': explained,
            'procrustes_r2': rigid_r2,
        }

    def _transform(self, Z_src: np.ndarray) -> np.ndarray:
        return self.fit_.apply(Z_src)

"""Unconstrained linear alignment -- the least-squares baseline family.

Where :class:`~src.alignment.procrustes.ProcrustesAligner` restricts the
map to be semi-orthogonal, this one lets it be an arbitrary ridge-
regularised linear map. It is a useful upper bound on what *any* purely
linear method can achieve on a given pair of encoders: if the RKHS
residual stage does not beat it, the leftover structure was linear all
along.

It also covers three of the four transformation classes Maiorca et al.
compare in "Latent Space Translation via Semantic Alignment" (Sec. 3.2),
each with its own config preset under ``config/hydra/alignment/``:

=================  =====================  ==============================
 preset             constructor            paper's name
=================  =====================  ==============================
 ``linear``         defaults                ``linear`` -- least squares
                                            with ``b = 0``
 ``affine``         ``fit_intercept``       ``affine`` -- ``Rx + b``
 ``l_ortho``        ``orthogonalize``       ``l-ortho`` -- the least-
                                            squares ``R``, projected
                                            onto the closest orthogonal
                                            matrix by SVD
=================  =====================  ==============================

Their fourth, ``ortho``, is Procrustes and lives in its own module. Note
that ``affine`` only differs from ``linear`` when the preprocessing does
*not* centre both spaces: under ``preprocess='standard'`` (the paper's
own standard scaling) the least-squares intercept is zero by
construction, so the two coincide. The paper reports them as different
methods because it fits the affine map by gradient descent, not in
closed form.
"""

from __future__ import annotations

import logging

import numpy as np

from .base import Aligner

log = logging.getLogger(__name__)

__all__ = ['LinearAligner']


class LinearAligner(Aligner):
    """Ridge-regularised linear map fitted by least squares.

    Parameters
    ----------
    preprocess : ScalingMethod, default='whiten'
        Per-space standardisation.
    alpha : float, default=1e-6
        Ridge strength, relative to the trace of the source Gram matrix
        (so it is invariant to the latent scale).
    fit_intercept : bool, default=False
        Fit an intercept, giving the ``affine`` class. Redundant for any
        centring ``preprocess``, which already puts it at zero.
    orthogonalize : bool, default=False
        Replace the fitted ``W`` by the closest semi-orthogonal matrix,
        ``U V^T`` from its SVD (the ``l-ortho`` class). This is *not* the
        Procrustes solution: Procrustes optimises over the orthogonal
        group directly, whereas this projects an unconstrained fit onto
        it afterwards. Comparing the two says how nearly orthogonal the
        unconstrained solution already was -- ``linear_orthogonality``
        reports it as the ratio of the fit to its own projection.
    eps : float, default=1e-6
        Relative covariance ridge of the whitening step.
    seed : int, default=42
        Unused; kept for a uniform constructor signature.
    """

    def __init__(
        self,
        preprocess: str = 'whiten',
        alpha: float = 1e-6,
        fit_intercept: bool = False,
        orthogonalize: bool = False,
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
        self.alpha = float(alpha)
        self.fit_intercept = bool(fit_intercept)
        self.orthogonalize = bool(orthogonalize)
        self.W_: np.ndarray | None = None
        self.b_: np.ndarray | None = None

    def hyperparameters(self) -> dict:
        params = super().hyperparameters()
        params.update(
            alpha=self.alpha,
            fit_intercept=self.fit_intercept,
            orthogonalize=self.orthogonalize,
        )
        return params

    def _fit(
        self,
        Z_src: np.ndarray,
        Z_tgt: np.ndarray,
        labels: np.ndarray | None = None,
    ) -> None:
        d_src = Z_src.shape[1]
        gram = Z_src.T @ Z_src
        ridge = self.alpha * max(float(np.trace(gram)) / d_src, 1e-30)
        gram[np.diag_indices_from(gram)] += ridge

        self.W_ = np.linalg.solve(gram, Z_src.T @ Z_tgt)  # (d_src, d_tgt)

        orthogonality = np.nan
        if self.orthogonalize:
            U, _, Vt = np.linalg.svd(self.W_, full_matrices=False)
            projected = U @ Vt
            # 1.0 means the least-squares fit was already an isometry, so
            # enforcing orthogonality costs nothing; 0 means it was not.
            orthogonality = float(
                np.sum(self.W_ * projected)
                / max(
                    np.linalg.norm(self.W_) * np.linalg.norm(projected),
                    1e-30,
                )
            )
            self.W_ = projected

        self.b_ = (
            Z_tgt.mean(axis=0) - Z_src.mean(axis=0) @ self.W_
            if self.fit_intercept
            else np.zeros(Z_tgt.shape[1])
        )
        self.diagnostics_ = {
            'linear_ridge': ridge,
            'linear_weight_norm': float(np.linalg.norm(self.W_)),
            'linear_orthogonality': orthogonality,
        }

    @property
    def map_parameters(self) -> int:
        """``W``, plus the intercept when one is fitted."""
        self._check_fitted()
        return self.W_.size + (self.b_.size if self.fit_intercept else 0)

    def _transform(self, Z_src: np.ndarray) -> np.ndarray:
        return Z_src @ self.W_ + self.b_

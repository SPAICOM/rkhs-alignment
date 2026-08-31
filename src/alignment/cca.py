"""Canonical-correlation alignment: whiten, rotate, colour.

CCA looks for the pair of bases -- one per space -- in which the two
point clouds are maximally correlated coordinate by coordinate:

    max_{a, b}  corr(Z_src a, Z_tgt b)

solved sequentially under ``a^T C_src a = b^T C_tgt b = 1``. Stacking the
first ``k`` solutions gives ``A in R^{d_src x k}`` and
``B in R^{d_tgt x k}``, and the alignment reads a source latent's
canonical scores and re-synthesises them with the *target's* canonical
basis:

    f(z) = (z^T A) B^+

which is exactly a whiten -> rotate -> colour pipeline: ``A`` whitens the
source with ``C_src^{-1/2}`` and rotates it onto the shared canonical
frame, ``B^+`` colours the result back into the target's own geometry.

Relation to the other methods here, which is the reason it earns a place
in the comparison:

- Against **Procrustes**: both end in a rotation, but Procrustes is
  constrained to be *rigid* in whatever coordinates it is handed, while
  CCA is free to re-weight directions by how well they actually
  correlate across the two encoders. The two coincide exactly when both
  spaces are already whitened from the same samples -- which is why this
  class always estimates its own whitening from the calibration pairs
  rather than leaning on :class:`~src.alignment.base.Aligner`'s.
- Against **Linear**: the unconstrained least-squares map is CCA at full
  rank *with* the canonical correlations left in as shrinkage. Dropping
  them (as here) keeps the target's scale instead of regressing toward
  its mean, which is what retrieval cares about.

Truncating to ``k < min(d_src, d_tgt)`` is the method's own regulariser:
the trailing canonical directions are the ones whose correlation is
estimated from noise, and they are exactly the ones a small pilot budget
gets wrong. It is also the method's *compression* knob: the map
factorises as ``z -> (z A) B^+``, so the transmitter can evaluate ``z A``
locally and put only those ``k`` numbers on the channel.

:class:`SVCCAAligner` is the SVCCA of Raghu et al. (2017): each space is
first projected onto its own top singular directions, and CCA is run in
those two truncated bases. The two truncation levels are per-space and
independent -- a 2048-dimensional ResNet transmitter and a 384-
dimensional ViT receiver have nothing in common in how fast their spectra
decay -- and, unlike the canonical rank, each is a *single-space*
statistic, so it can be estimated from a device's own unpaired data and
costs no pilots.

Why it is not just CCA with a smaller ``k``: CCA is invariant to any
invertible linear map of either space, so an *untruncated* SVD change of
basis would leave the solution untouched. What SVCCA changes is the
subspace the covariances are estimated in. Dropping the low-variance
directions before inverting a covariance removes precisely the
directions whose inverse-square-root is dominated by sampling noise, so
the ridge no longer has to carry them; on wide latents it also drops the
two eigendecompositions from ``d^3`` to ``k^3``.
"""

from __future__ import annotations

import logging

import numpy as np

from .base import Aligner

log = logging.getLogger(__name__)

__all__ = ['CCAAligner', 'SVCCAAligner']


def _inverse_sqrt(cov: np.ndarray, ridge: float) -> np.ndarray:
    """Symmetric ``C^{-1/2}`` of a ridged covariance.

    The ridge is *relative* to the mean eigenvalue, so one setting behaves
    the same across encoders whose latents differ in scale by orders of
    magnitude, and the eigenvalues are floored at it so a rank-deficient
    covariance -- which is what a pilot budget below the latent dimension
    always produces -- inverts to something finite.
    """
    d = cov.shape[0]
    shift = ridge * max(float(np.trace(cov)) / d, 1e-30)
    evals, evecs = np.linalg.eigh(cov + shift * np.eye(d))
    return (evecs * np.clip(evals, shift, None) ** -0.5) @ evecs.T


def _canonical_bases(
    Xc: np.ndarray, Yc: np.ndarray, k: int, reg: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Canonical bases of two *centred*, paired point clouds.

    Returns ``(A, B, rho)``: the first ``k`` canonical directions of each
    space and their canonical correlations. Whichever basis the two
    clouds are expressed in, this is the same three-line recipe --
    whiten both, take the SVD of the cross-covariance between the
    whitened clouds, un-whiten the singular vectors -- which is why
    :class:`SVCCAAligner` only has to hand it a projected pair.
    """
    n = Xc.shape[0]
    C_src_inv_sqrt = _inverse_sqrt(Xc.T @ Xc / n, reg)
    C_tgt_inv_sqrt = _inverse_sqrt(Yc.T @ Yc / n, reg)
    M = C_src_inv_sqrt @ (Xc.T @ Yc / n) @ C_tgt_inv_sqrt

    U, rho, Vt = np.linalg.svd(M, full_matrices=False)
    return C_src_inv_sqrt @ U[:, :k], C_tgt_inv_sqrt @ Vt[:k].T, rho[:k]


def _check_truncation(value: float | None, name: str) -> int | float | None:
    """Validate an SVD level: a rank, a variance fraction, or ``None``.

    The int/float split is the ``sklearn`` (and
    :class:`~src.alignment.preprocessing.LatentScaler`) convention, with
    its one sharp edge: ``1`` is a single component and ``1.0`` is all
    the variance.
    """
    if value is None:
        return None
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        if int(value) < 1:
            raise ValueError(
                f'{name} must be an int >= 1, a fraction in (0, 1], or '
                f'None; got {value}.'
            )
        return int(value)
    fraction = float(value)
    if not 0.0 < fraction <= 1.0:
        raise ValueError(
            f'A fractional {name} must lie in (0, 1], got {fraction}.'
        )
    return fraction


def _svd_basis(
    Zc: np.ndarray, level: float | None
) -> tuple[np.ndarray, float]:
    """Top principal directions of a centred cloud, and the variance kept.

    ``level`` is a rank, a fraction of the variance, or ``None`` for
    every direction the cloud actually spans. The numerically null
    directions are dropped in all three cases: they carry no data, and
    keeping them would hand CCA a covariance whose inverse square root is
    pure ridge.
    """
    _, sigma, Vt = np.linalg.svd(Zc, full_matrices=False)
    variance = sigma**2
    total = float(variance.sum())

    # np.linalg.matrix_rank's tolerance, on the singular values we have.
    rank = int(
        (sigma > sigma[0] * max(Zc.shape) * np.finfo(float).eps).sum()
        if sigma[0] > 0
        else 0
    )
    rank = max(rank, 1)

    if level is None:
        k = rank
    elif isinstance(level, int):
        k = min(level, rank)
    else:
        kept = np.cumsum(variance) / max(total, 1e-30)
        k = min(int(np.searchsorted(kept, level) + 1), rank)

    return Vt[:k].T, float(variance[:k].sum() / max(total, 1e-30))


class CCAAligner(Aligner):
    """Alignment through the canonical-correlation bases of the two spaces.

    Parameters
    ----------
    n_canonical : int | None, default=None
        Number ``k`` of canonical directions kept. ``None`` keeps
        ``min(d_src, d_tgt)`` -- the largest number of canonical pairs
        that exists, and exactly the rank of the rectangular Procrustes
        map on the same pair, so the two methods carry the same effective
        dimension and the comparison is not confounded by capacity.
    reg : float, default=1e-3
        Ridge on both within-space covariances, relative to their mean
        eigenvalue. CCA inverts *two* covariances rather than one, so it
        is markedly more sensitive to a small pilot budget than the other
        closed-form methods; this is the knob that keeps it usable there.
    preprocess : ScalingMethod, default='standard'
        Per-space standardisation. CCA estimates its own whitening from
        the calibration pairs regardless, so ``'whiten'`` here is
        redundant at best -- and at worst it whitens from a *different*
        sample (the local context) than the one CCA then un-whitens with,
        which mostly cancels out. Centring is all the method needs.
    eps : float, default=1e-6
        Relative covariance ridge of the whitening step.
    seed : int, default=42
        Unused (the solution is deterministic); kept for a uniform
        constructor signature.
    """

    def __init__(
        self,
        n_canonical: int | None = None,
        reg: float = 1e-3,
        preprocess: str = 'standard',
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
        if n_canonical is not None and int(n_canonical) < 1:
            raise ValueError(
                f'n_canonical must be >= 1 or None, got {n_canonical}.'
            )
        self.n_canonical = n_canonical
        self.reg = float(reg)

        self.mean_src_: np.ndarray | None = None
        self.mean_tgt_: np.ndarray | None = None
        self.A_: np.ndarray | None = None
        self.B_: np.ndarray | None = None
        self.correlations_: np.ndarray | None = None
        self.W_: np.ndarray | None = None

    def hyperparameters(self) -> dict:
        params = super().hyperparameters()
        params.update(n_canonical=self.n_canonical, reg=self.reg)
        return params

    def _fit(
        self,
        Z_src: np.ndarray,
        Z_tgt: np.ndarray,
        labels: np.ndarray | None = None,
    ) -> None:
        n, d_src = Z_src.shape
        d_tgt = Z_tgt.shape[1]
        k_max = min(d_src, d_tgt)
        k = (
            k_max
            if self.n_canonical is None
            else min(int(self.n_canonical), k_max)
        )

        # CCA is invariant to centring but not to the mean, and the base
        # class only guarantees it for a centring `preprocess`. Keep both
        # means so `_transform` can reproduce the same shift on new
        # points whatever the preprocessing did.
        self.mean_src_ = Z_src.mean(axis=0)
        self.mean_tgt_ = Z_tgt.mean(axis=0)
        Xc = Z_src - self.mean_src_
        Yc = Z_tgt - self.mean_tgt_

        self.A_, self.B_, self.correlations_ = _canonical_bases(
            Xc, Yc, k, self.reg
        )  # (d_src, k), (d_tgt, k)

        # Score with A, synthesise with the target basis. Collapsing the
        # two into one matrix keeps `transform` a single gemm and makes
        # the map directly comparable with the other linear methods.
        self.W_ = self.A_ @ np.linalg.pinv(self.B_)  # (d_src, d_tgt)
        self.diagnostics_ = self._canonical_diagnostics()
        if n < d_src + d_tgt:
            log.debug(
                'CCA fitted on %d pairs for (%d -> %d) dimensions: the '
                'canonical correlations are optimistically biased and the '
                'ridge (reg=%g) is carrying the fit.',
                n,
                d_src,
                d_tgt,
                self.reg,
            )

    @property
    def transmitted_symbols(self) -> int:
        """The canonical coordinates -- all the channel actually carries.

        ``f(z) = (z A) B^+`` factorises through ``k`` numbers: the
        transmitter evaluates ``z A`` with its own local matrix and puts
        that on the channel, the receiver applies ``B^+``. So a
        canonical-correlation map's rate is bounded by its canonical rank
        as well as by the standardised source dimension, and truncating
        to ``k`` is a compression of ``k / d_src`` whether or not the
        whitening was truncated to match.

        Which of the two binds depends on how the method was configured:
        under ``preprocess='whiten'`` with ``n_components=k`` the
        whitening does the truncating and the two agree, while under
        plain centring or z-scoring the canonical rank is the only thing
        holding the rate down -- and with ``n_canonical=None`` that is
        still ``min(d_src, d_tgt)``, so a source wider than the target
        never transmits its full dimension.

        In :class:`SVCCAAligner` the canonical rank always binds, since
        the SVD levels cap it; they buy accuracy rather than airtime.
        """
        self._check_fitted()
        return min(int(self.scaler_src_.out_dim), int(self.A_.shape[1]))

    def _canonical_diagnostics(self) -> dict:
        """Post-fit summary shared by both canonical-correlation classes."""
        return {
            'cca_n_canonical': len(self.correlations_),
            'cca_mean_correlation': float(np.mean(self.correlations_)),
            # Correlations pinned at ~1 across the board mean the fit is
            # interpolating: with n <= d + k every direction can be made
            # to correlate perfectly on the calibration set alone.
            'cca_min_correlation': float(np.min(self.correlations_)),
            'cca_weight_norm': float(np.linalg.norm(self.W_)),
        }

    def _transform(self, Z_src: np.ndarray) -> np.ndarray:
        # The canonical bases are estimated on centred data, so the map
        # has to centre and re-offset itself rather than leaning on the
        # `preprocess` step to have done it. Without this the class is
        # only correct under a centring `preprocess`, and silently
        # carries a `mean_src @ W - mean_tgt` bias under `none` -- which
        # a receiver's fixed probe reads as a shifted latent.
        return (Z_src - self.mean_src_) @ self.W_ + self.mean_tgt_

    # ------------------------------------------------------------------
    # What the method costs
    # ------------------------------------------------------------------

    @property
    def map_parameters(self) -> int:
        """The two factors, one held at each end of the link.

        The transmitter stores ``A`` and the receiver ``B^+``, so a
        rank-``k`` map costs ``k (d_src + d_tgt)`` numbers rather than
        the ``d_src d_tgt`` of the collapsed ``W`` -- which is only ever
        formed here to keep :meth:`_transform` a single gemm.
        """
        self._check_fitted()
        return int(self.A_.size + self.B_.size)


class SVCCAAligner(CCAAligner):
    """CCA in the top singular subspace of each space (Raghu et al., 2017).

    Each side is projected onto its own leading singular directions
    before a single canonical-correlation problem is solved in the two
    truncated bases, and the resulting target-side scores are lifted back
    out of the truncation. The map is still one matrix
    ``W = A B^+ : R^{d_src} -> R^{d_tgt}``, so it drops into the same
    comparison as every other linear method.

    The truncation is *per space*, and deliberately so: the transmitter
    and the receiver are different encoders with different widths and
    different spectral decay, and the level that keeps 99% of a
    2048-dimensional ResNet's variance has nothing to do with the one
    that does the same for a 384-dimensional ViT. It is also a
    single-space quantity -- unlike the canonical directions, which only
    exist for a *pair* -- so each device can estimate its own basis from
    all of its local data, and this class does exactly that whenever
    ``fit`` is given ``src_context`` / ``tgt_context``. Nothing about the
    truncation is paid for in pilots.

    Two knobs now compress, and they are not interchangeable:

    - ``svd_src`` / ``svd_tgt`` are *denoising*. They decide which
      directions the covariances are estimated in, which is what keeps
      the two inversions honest when the pilot budget is small.
    - ``n_canonical`` is the *rate*. The map factorises through ``k``
      canonical coordinates, so the transmitter sends ``z A`` -- ``k``
      numbers -- and the receiver applies ``B^+``. See
      :attr:`transmitted_symbols`.

    Parameters
    ----------
    svd_src, svd_tgt : int | float | None, default=None
        Truncation level of the source (transmitter) and target
        (receiver) space. An ``int`` keeps that many singular directions;
        a ``float`` in ``(0, 1]`` keeps as many as explain that fraction
        of the space's variance; ``None`` keeps every direction the data
        spans, which recovers plain CCA up to the ridge. Note ``1`` is
        one component and ``1.0`` is all the variance.
    n_canonical : int | None, default=None
        Canonical directions kept, as in :class:`CCAAligner`, but now
        capped by the two SVD levels: ``k <= min(k_src, k_tgt)``.
    reg : float, default=1e-3
        Ridge on both within-subspace covariances, relative to their mean
        eigenvalue. A well-chosen truncation is the better regulariser of
        the two -- it removes the noisy directions rather than damping
        them -- so this can usually sit lower here than in plain CCA.
    preprocess : ScalingMethod, default='center'
        Per-space standardisation. Centring is the SVCCA default because
        the truncation is *not* invariant to what comes before it:
        z-scoring re-weights every feature to unit variance and so
        reshuffles the spectrum the truncation then ranks, and whitening
        flattens it outright, at which point "the top k directions" means
        nothing. Use ``'standard'`` only when the raw features differ in
        scale for reasons that are not semantic.

    Attributes
    ----------
    P_src_, P_tgt_ : np.ndarray
        The two SVD bases, ``(d, k)`` with orthonormal columns.
    """

    def __init__(
        self,
        svd_src: float | None = None,
        svd_tgt: float | None = None,
        n_canonical: int | None = None,
        reg: float = 1e-3,
        preprocess: str = 'center',
        eps: float = 1e-6,
        n_components: float | None = None,
        shrinkage: str | float | None = 'auto',
        seed: int = 42,
    ) -> None:
        super().__init__(
            n_canonical=n_canonical,
            reg=reg,
            preprocess=preprocess,
            eps=eps,
            n_components=n_components,
            shrinkage=shrinkage,
            seed=seed,
        )
        self.svd_src = _check_truncation(svd_src, 'svd_src')
        self.svd_tgt = _check_truncation(svd_tgt, 'svd_tgt')

        self.P_src_: np.ndarray | None = None
        self.P_tgt_: np.ndarray | None = None

    def hyperparameters(self) -> dict:
        params = super().hyperparameters()
        params.update(svd_src=self.svd_src, svd_tgt=self.svd_tgt)
        return params

    def _fit(
        self,
        Z_src: np.ndarray,
        Z_tgt: np.ndarray,
        labels: np.ndarray | None = None,
    ) -> None:
        # Same contract as the parent: the map centres itself, so it is
        # correct under any `preprocess` rather than only a centring one.
        self.mean_src_ = Z_src.mean(axis=0)
        self.mean_tgt_ = Z_tgt.mean(axis=0)
        Xc = Z_src - self.mean_src_
        Yc = Z_tgt - self.mean_tgt_

        self.P_src_, var_src = self._basis(
            Z_src, self._context_src, self.svd_src
        )
        self.P_tgt_, var_tgt = self._basis(
            Z_tgt, self._context_tgt, self.svd_tgt
        )
        k_src, k_tgt = self.P_src_.shape[1], self.P_tgt_.shape[1]

        k_max = min(k_src, k_tgt)
        k = (
            k_max
            if self.n_canonical is None
            else min(int(self.n_canonical), k_max)
        )

        A_p, B_p, self.correlations_ = _canonical_bases(
            Xc @ self.P_src_, Yc @ self.P_tgt_, k, self.reg
        )
        # Lift both bases out of the truncation, so the map is stated in
        # the same standardised coordinates as every other method's.
        self.A_ = self.P_src_ @ A_p  # (d_src, k)
        self.B_ = self.P_tgt_ @ B_p  # (d_tgt, k)
        # P_tgt_ is an isometry, so pinv(P B) = pinv(B) P^T exactly; the
        # pseudo-inverse is taken in the small basis and lifted, not
        # recomputed on the (d_tgt, k) matrix.
        self.W_ = self.A_ @ np.linalg.pinv(B_p) @ self.P_tgt_.T

        from_context = (
            self._context_src is not None or self._context_tgt is not None
        )
        self.diagnostics_ = {
            **self._canonical_diagnostics(),
            'svcca_rank_src': int(k_src),
            'svcca_rank_tgt': int(k_tgt),
            'svcca_variance_src': var_src,
            'svcca_variance_tgt': var_tgt,
            'svcca_basis_from': 'context' if from_context else 'pilots',
        }

        n = Z_src.shape[0]
        if n < k_src + k_tgt:
            log.debug(
                'SVCCA fitted on %d pairs in (%d -> %d) truncated '
                'dimensions: the canonical correlations are '
                'optimistically biased; lower svd_src/svd_tgt before '
                'raising reg=%g, since truncating removes the noisy '
                'directions instead of damping them.',
                n,
                k_src,
                k_tgt,
                self.reg,
            )

    def _basis(
        self,
        Z: np.ndarray,
        context: np.ndarray | None,
        level: float | None,
    ) -> tuple[np.ndarray, float]:
        """SVD basis of one space, from its context when it has one.

        The pilots are the fallback, not the preference: a rank the
        pilot budget cannot support is silently clipped to ``min(n, d)``,
        and a basis estimated from a handful of paired samples is exactly
        the noisy object the truncation exists to remove.
        """
        source = Z if context is None else context
        return _svd_basis(source - source.mean(axis=0), level)

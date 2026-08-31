"""Common contract for latent-space alignment methods.

Every method in this package answers the same question: given paired
calibration latents from a *source* encoder (the transmitter) and a
*target* encoder (the receiver), build a map

    T : R^{d_src} -> R^{d_tgt}

that sends new source latents into the target's own raw latent space, so
that the receiver's private decoder -- trained only on its own latents --
can consume them unchanged.

:class:`Aligner` factors out the two things every method shares:

1. **Standardisation** (Step 0 of ``idea.md``). Each space gets its own
   fitted :class:`~src.alignment.preprocessing.LatentScaler`; subclasses
   only ever see standardised coordinates, and :meth:`Aligner.transform`
   undoes the target-side scaling so the output is a raw target latent.
2. **Validation and bookkeeping**: paired-shape checks (including the
   rectangular-Procrustes dimension regime) and a :meth:`Aligner.summary`
   dict of hyper-parameters plus fit diagnostics, ready to be logged.

Subclasses implement :meth:`Aligner._fit` and :meth:`Aligner._transform`
on standardised coordinates.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from .preprocessing import LatentScaler, ScalingMethod

log = logging.getLogger(__name__)

__all__ = ['Aligner', 'check_paired_dims']


def check_paired_dims(
    X_src: np.ndarray,
    X_tgt: np.ndarray,
    context: str = 'alignment',
    level: int = logging.INFO,
) -> tuple[int, int, int]:
    """Validate a paired calibration set and report its dimension regime.

    Parameters
    ----------
    X_src : np.ndarray, shape (n, d_src)
        Source (transmitter) latents.
    X_tgt : np.ndarray, shape (n, d_tgt)
        Target (receiver) latents; row ``i`` must be the *same* input
        sample as row ``i`` of ``X_src``.
    context : str, default='alignment'
        Name used in error messages.
    level : int, default=logging.INFO
        Level at which the dimension regime is reported. Nested callers
        drop it to ``logging.DEBUG`` so the message is emitted once.

    Returns
    -------
    (n, d_src, d_tgt) : tuple[int, int, int]

    Raises
    ------
    ValueError
        If either array is not 2-D, if the row counts disagree, or if
        the calibration set is empty.
    """
    if X_src.ndim != 2 or X_tgt.ndim != 2:
        raise ValueError(
            f'{context}: both point clouds must be 2-dimensional '
            f'(n_samples, n_features); got {X_src.shape} and {X_tgt.shape}.'
        )
    n, d_src = X_src.shape
    n_tgt, d_tgt = X_tgt.shape
    if n != n_tgt:
        raise ValueError(
            f'{context}: source and target must be paired sample-by-sample, '
            f'but got {n} source rows and {n_tgt} target rows.'
        )
    if n == 0:
        raise ValueError(f'{context}: empty calibration set.')

    if d_src > d_tgt:
        log.log(
            level,
            '%s: rectangular Procrustes in projection regime '
            '(d_src=%d > d_tgt=%d); Q Q^T = I_%d, so the map discards '
            '%d source directions.',
            context,
            d_src,
            d_tgt,
            d_tgt,
            d_src - d_tgt,
        )
    elif d_src < d_tgt:
        log.log(
            level,
            '%s: rectangular Procrustes in embedding regime '
            '(d_src=%d < d_tgt=%d); Q^T Q = I_%d, so the image spans only '
            '%d of the %d target directions.',
            context,
            d_src,
            d_tgt,
            d_src,
            d_src,
            d_tgt,
        )
    if n < max(d_src, d_tgt):
        log.warning(
            '%s: only %d paired samples for dimensions (%d -> %d); the '
            'cross-covariance is rank-deficient and the fit will be '
            'under-determined.',
            context,
            n,
            d_src,
            d_tgt,
        )
    return n, d_src, d_tgt


def _standardise(
    scaler: LatentScaler, X: np.ndarray | None
) -> np.ndarray | None:
    """Apply a fitted scaler to optional unpaired context latents."""
    if X is None:
        return None
    return scaler.transform(np.asarray(X, dtype=np.float64))


class Aligner(ABC):
    """Base class for a source -> target latent-space map.

    Parameters
    ----------
    preprocess : ScalingMethod, default='whiten'
        Per-space standardisation fitted on the calibration data and
        applied to every subsequent point. One of ``'none'``,
        ``'center'``, ``'standard'``, ``'pca'``, ``'pga'`` or
        ``'whiten'`` -- see
        :class:`~src.alignment.preprocessing.LatentScaler`.
    eps : float, default=1e-6
        Relative covariance ridge of the whitening step.
    shrinkage : {'auto'} | float | None, default='auto'
        Covariance shrinkage of the whitening step; ``None`` disables it
        and recovers the plain sample covariance.
    n_components : int | float | None, default=None
        Rank of the whitening (see
        :class:`~src.alignment.preprocessing.LatentScaler`). ``None``
        keeps every direction. Truncating matters on wide encoder
        latents, where full-rank whitening amplifies hundreds of noise
        directions and leaves the kernel residual unable to generalise.
    seed : int, default=42
        Seed for any randomised component of the method.
    """

    def __init__(
        self,
        preprocess: ScalingMethod = 'whiten',
        eps: float = 1e-6,
        n_components: float | None = None,
        shrinkage: str | float | None = 'auto',
        seed: int = 42,
    ) -> None:
        self.preprocess: ScalingMethod = preprocess
        self.eps = float(eps)
        self.n_components = n_components
        self.shrinkage = shrinkage
        self.seed = int(seed)

        self.scaler_src_: LatentScaler | None = None
        self.scaler_tgt_: LatentScaler | None = None
        # Standardised unpaired context, held only for the duration of a
        # `fit` (see `Aligner.fit`).
        self._context_src: np.ndarray | None = None
        self._context_tgt: np.ndarray | None = None
        self.n_calibration_: int | None = None
        self.dim_src_: int | None = None
        self.dim_tgt_: int | None = None
        self.diagnostics_: dict[str, Any] = {}
        self.preprocess_diagnostics_: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Short identifier used in logs and result tables."""
        return type(self).__name__.removesuffix('Aligner').lower()

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has been called successfully."""
        return self.scaler_tgt_ is not None

    # ------------------------------------------------------------------
    # What the method costs
    # ------------------------------------------------------------------

    @property
    def paired_samples_used(self) -> int:
        """Paired calibration samples the fitted map actually consumes.

        This is the quantity that costs airtime, and it is *not* the same
        for every method at the same budget. A rigid or least-squares map
        estimates a cross-covariance and so uses every pilot it is given;
        an anchor-based method with a closed-form inverse only ever
        touches its ``K`` anchors, and would return the identical map if
        the rest of the budget were never exchanged. Ranking two such
        methods at equal ``N`` compares them at unequal cost.
        """
        self._check_fitted()
        return int(self.n_calibration_)

    @property
    def transmitted_symbols(self) -> int:
        """Coefficients the transmitter puts on the channel per sample.

        The rate, in the semantic-communication sense. For a method that
        sends a *coordinate* representation -- Procrustes, the linear
        classes, RKA -- that is the dimension of the standardised source
        space, so a truncated whitening is a compression: keep ``k``
        principal directions and ``k`` numbers are sent instead of
        ``d_src``. For an anchor-based method it is the number of anchors
        instead (see
        :class:`~src.alignment.relative.RelativeRepresentationAligner`).

        Two methods are only rate-comparable when this agrees, which is
        what ``symbols`` in the comparison config enforces.
        """
        self._check_fitted()
        return int(self.scaler_src_.out_dim)

    @property
    def map_parameters(self) -> int:
        """Numbers that must be stored, or shipped, to apply the map.

        The receiver has to hold this to decode anything, so it is the
        method's standing memory cost -- a second axis on which the
        methods are not comparable, since a kernel method carries its
        calibration points into deployment while a linear one does not.

        Counted between the *standardised* spaces, so a truncated
        whitening shrinks it: the two scalers are excluded because every
        method carries the same pair of them.
        """
        self._check_fitted()
        return int(self.scaler_src_.out_dim * self.scaler_tgt_.out_dim)

    def hyperparameters(self) -> dict[str, Any]:
        """Method hyper-parameters (subclasses extend this)."""
        return {
            'preprocess': self.preprocess,
            'eps': self.eps,
            'n_components': self.n_components,
            'shrinkage': self.shrinkage,
        }

    def summary(self) -> dict[str, Any]:
        """Hyper-parameters plus post-fit diagnostics, for logging."""
        out: dict[str, Any] = {'method': self.name}
        out.update(self.hyperparameters())
        if self.is_fitted:
            out.update(
                n_calibration=self.n_calibration_,
                dim_src=self.dim_src_,
                dim_tgt=self.dim_tgt_,
                paired_samples_used=self.paired_samples_used,
                map_parameters=self.map_parameters,
                transmitted_symbols=self.transmitted_symbols,
            )
            out.update(self.preprocess_diagnostics_)
            out.update(self.diagnostics_)
        return out

    def __repr__(self) -> str:
        state = (
            f'{self.dim_src_}->{self.dim_tgt_}'
            if self.is_fitted
            else 'unfitted'
        )
        return f'{type(self).__name__}({state})'

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X_src: np.ndarray,
        X_tgt: np.ndarray,
        labels: np.ndarray | None = None,
        src_context: np.ndarray | None = None,
        tgt_context: np.ndarray | None = None,
    ) -> Aligner:
        """Fit the map on paired calibration latents.

        Parameters
        ----------
        X_src : np.ndarray, shape (n, d_src)
            Source latents; row ``i`` is the same input sample as row
            ``i`` of ``X_tgt``. These are the *pilots*: the samples both
            devices must exchange.
        X_tgt : np.ndarray, shape (n, d_tgt)
            Target latents.
        labels : np.ndarray, optional
            Per-sample class labels; used only by methods that need them
            (e.g. label-stratified anchor selection).
        src_context, tgt_context : np.ndarray, optional
            Unpaired local latents used to estimate that side's
            standardisation -- and any other single-space statistic a
            method can estimate without pairing, such as SVCCA's SVD
            basis. Defaults to the pilots themselves.

            Standardising a latent space needs no pairing and therefore
            no pilots: each device can whiten from all of its own data,
            and only the paired samples cost airtime. Tying the whitening
            to the pilot budget makes the low-budget regime look far
            worse than it is -- measured on the Digits pair, estimating
            the whitening from all local data instead of from the pilots
            is worth +14 accuracy points at ``N = 5`` and +7 at
            ``N = 8``, converging by ``N ~ 100``.

        Returns
        -------
        Aligner
            ``self``.
        """
        X_src = np.asarray(X_src, dtype=np.float64)
        X_tgt = np.asarray(X_tgt, dtype=np.float64)
        n, d_src, d_tgt = check_paired_dims(X_src, X_tgt, context=self.name)
        self.n_calibration_, self.dim_src_, self.dim_tgt_ = n, d_src, d_tgt

        make = lambda: LatentScaler(  # noqa: E731
            self.preprocess,
            eps=self.eps,
            n_components=self.n_components,
            shrinkage=self.shrinkage,
        )
        self.scaler_src_, self.scaler_tgt_ = make(), make()
        self.scaler_src_.fit(X_src if src_context is None else src_context)
        self.scaler_tgt_.fit(X_tgt if tgt_context is None else tgt_context)
        Z_src = self.scaler_src_.transform(X_src)
        Z_tgt = self.scaler_tgt_.transform(X_tgt)

        # Kept apart from `diagnostics_`, which every subclass assigns
        # wholesale in `_fit`.
        self.preprocess_diagnostics_ = {}
        if self.n_components is not None:
            self.preprocess_diagnostics_ = {
                'rank_src': self.scaler_src_.out_dim,
                'rank_tgt': self.scaler_tgt_.out_dim,
                'variance_kept_src': (
                    self.scaler_src_.explained_variance_ratio_
                ),
                'variance_kept_tgt': (
                    self.scaler_tgt_.explained_variance_ratio_
                ),
            }
        self.diagnostics_ = {}

        # Subclasses that estimate a *per-space* statistic -- a basis, a
        # spectrum -- may take it from the context rather than from the
        # pilots, because such a statistic needs no pairing and so costs
        # no airtime (:class:`~src.alignment.cca.SVCCAAligner` does this
        # for its SVD bases). Held only across the fit, so a fitted
        # aligner never carries a copy of the training split around.
        self._context_src = _standardise(self.scaler_src_, src_context)
        self._context_tgt = _standardise(self.scaler_tgt_, tgt_context)
        try:
            self._fit(Z_src, Z_tgt, labels=labels)
        finally:
            self._context_src = self._context_tgt = None
        return self

    def transform(self, X_src: np.ndarray) -> np.ndarray:
        """Map source latents into the target's raw latent space.

        Parameters
        ----------
        X_src : np.ndarray, shape (m, d_src)

        Returns
        -------
        np.ndarray, shape (m, d_tgt)
        """
        self._check_fitted()
        X_src = np.asarray(X_src, dtype=np.float64)
        if X_src.ndim != 2 or X_src.shape[1] != self.dim_src_:
            raise ValueError(
                f'{self.name}: expected source latents of shape '
                f'(m, {self.dim_src_}), got {X_src.shape}.'
            )
        Z_hat = self._transform(self.scaler_src_.transform(X_src))
        return self.scaler_tgt_.inverse_transform(Z_hat)

    def fit_transform(
        self,
        X_src: np.ndarray,
        X_tgt: np.ndarray,
        labels: np.ndarray | None = None,
        **context: np.ndarray | None,
    ) -> np.ndarray:
        """:meth:`fit` on the calibration pair, then map the source."""
        return self.fit(X_src, X_tgt, labels=labels, **context).transform(
            X_src
        )

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    @abstractmethod
    def _fit(
        self,
        Z_src: np.ndarray,
        Z_tgt: np.ndarray,
        labels: np.ndarray | None = None,
    ) -> None:
        """Fit on *standardised* paired latents."""

    @abstractmethod
    def _transform(self, Z_src: np.ndarray) -> np.ndarray:
        """Map *standardised* source latents to standardised target ones."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(
                f'{type(self).__name__}.fit() must be called first.'
            )

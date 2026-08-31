"""Kernel CCA on kernel data (Huang, Lee and Hsiao, 2009).

The paper's contribution is a *placement*, not a new estimator: instead
of deriving a kernelised eigenproblem, it turns each space into a
"kernel data" matrix and then runs classical linear CCA on it. Two steps,
in the authors' own words (Section 2.2):

    (a) transform the data points to a kernel representation,
        ``K^(i) = kappa_i(X^(i), X^(i))``, whose ``(j, j')`` entry is
        ``kappa_i(x_j, x_j')``;
    (b) KCCA is the classical LCCA on *regularised* kernel data.

Read row-wise, that is a feature map: sample ``j`` of space ``i`` is
represented by the vector of its kernel values against a basis set,
``psi_i(x) = (kappa_i(x, u_1), ..., kappa_i(x, u_m))``. Propositions 1
and 2 of the appendix are what license it -- for a translation-invariant
kernel with a shrinking window those coordinates become dense in
``L_2(P_i)``, so the LCCA of the kernel data converges to the canonical
analysis of the two RKHSs.

The reason it earns a place next to :class:`~src.alignment.cca.CCAAligner`
is that ``kappa_i`` is fitted per space and the canonical stage is
unchanged: :func:`~src.alignment.cca._canonical_bases` is called here
verbatim, on ``psi_1(Z_src)`` and ``psi_2(Z_tgt)`` instead of on
``Z_src`` and ``Z_tgt``. Everything CCA-shaped about the method -- the
rate being the canonical rank, the ridge on the two covariances -- carries
over unchanged, and the only new object is the feature map.

Regularisation (Section 3.1)
----------------------------
A Gram matrix has "effective rank much lower than its size", so the
covariances the canonical stage inverts are singular by construction.
The paper deliberately does *not* fix this with a ridge -- "it lessens the
numerical instability, [but] does not solve the problem of inferior
estimation" -- and instead reduces the number of columns of the kernel
data before CCA ever sees it. Three routes, all available here through
``basis``:

``'svd'``
    Optimal basis subset, Eqs. (5)-(6). Eigendecompose ``K = U L U'`` and
    keep the leading columns ``U~``, so the kernel data becomes
    ``K~ = K U~``. The paper sizes ``U~`` by eigenvalue mass (99% in its
    experiments), which is what a fractional ``n_basis_src`` means here.
``'subset'``
    Random basis subset, Eq. (7). Draw ``m << n`` points ``X~`` and use
    the thin matrix ``K~ = kappa(X, X~)``. "The most economic one among
    the three regularization methods", and the only one whose deployed
    map is sparse -- it stores ``m`` kernel centres rather than all ``n``.
``'hybrid'``
    The paper's own compromise: a random subset first, then the SVD cut
    inside it, via ``n_subset``.

Which subset ``'subset'`` draws is left to the repo's pilot designs
(``subset_strategy``) rather than fixed to uniform sampling. The paper
uses uniform, and stratified uniform in its Pendigits example -- both are
available, alongside the k-means and kernel-herding designs the rest of
this package already uses for the same job.

From canonical scores back to a latent
--------------------------------------
CCA between two Euclidean spaces synthesises a target latent with
``B^+``: the canonical basis is invertible on its own range, so scoring
with ``A`` and colouring with ``B^+`` lands back in the target space.
There is no such inverse here -- ``psi_2`` maps a ``d_tgt``-dimensional
latent into ``m_2`` kernel coordinates and cannot be run backwards -- so
the pre-image is *learned*, as a ridge readout ``R`` from the target's
own canonical scores onto its raw latents:

    R = argmin_R || (psi_2(Z_tgt) B) R - Z_tgt ||^2 + ridge

    f(z) = (psi_1(z) A) R + mean_tgt

which is the exact non-linear analogue of ``B^+``: fit on the *target*
side, applied to the *source* side's scores, and correct to the extent
that the canonical correlations are real. That is ``readout='target'``.

The default is the other one. ``readout='source'`` fits the same readout
on the *source* scores, ``psi_1(Z_src) A``, which is what Example 3 of
the paper does when it feeds the KCCA-found variates of its inputs to a
discriminant fitted on those same variates -- reduced-rank kernel
regression through a KCCA-chosen subspace. Both use the identical
canonical solution and differ only in how the decoder is estimated;
measured on a smooth synthetic pair over pilot budgets from 30 to 500 and
three strengths of non-linearity, ``'source'`` was never worse and won by
3-5% of held-out NMSE, because it is fitted on the scores it will
actually be applied to. Keep ``'target'`` when the question is whether
the *shared subspace* is right rather than how well a decoder rides on
top of it -- it is the variant the receiver could in principle estimate
without the transmitter.

Association measures (Section 4)
--------------------------------
The canonical correlations are reported in the summary as the paper's own
statistics: ``r_max = rho_1`` (Eq. 8), ``r_log = -sum log(1 - rho_v^2)``
(Eq. 9), and the Bartlett-style independence test of Eq. (13),

    (n - (m_1 + m_2 + 1) / 2) r_log  ~  chi^2_{m_1 m_2}

under independence -- Eq. (13) with the degrees of freedom corrected for
column-centred kernel data, see :func:`_bartlett_p`. Here they are fit
diagnostics rather than a hypothesis test: a p-value pinned at 0 with
``rho_min ~ 1`` is the kernel data interpolating the pilots, which is the
failure mode the reduced-column step exists to prevent.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
from scipy.stats import chi2

from ..kernels import Kernel, KernelName
from .base import Aligner
from .cca import _canonical_bases, _check_truncation
from .pilots import PILOT_STRATEGIES, select_pilots

log = logging.getLogger(__name__)

__all__ = ['KCCAAligner', 'rule_of_thumb_gamma']

BasisMethod = Literal['svd', 'subset', 'hybrid']

_BASIS_METHODS: tuple[str, ...] = ('svd', 'subset', 'hybrid')

# Beyond this many basis-pool points the O(n^2 m) kernel evaluation and
# the SVD of the kernel data start to dominate the whole fit.
_LARGE_POOL_WARNING = 4000


def rule_of_thumb_gamma(X: np.ndarray, bandwidth_scale: float = 1.0) -> float:
    """The paper's Gaussian window width, as an inverse bandwidth.

    Section 3.2 fixes the window width at ``sigma_l = sqrt(10 S_l)``
    coordinatewise, ``S_l`` being the one-dimensional sample variance of
    coordinate ``l`` -- "not be optimal or even far from being optimal,
    [but] it gives robust and satisfactory results".

    A coordinatewise width is an anisotropic kernel;
    :class:`~src.kernels.Kernel` is isotropic, so what is used here is the
    mean coordinate variance. The two coincide exactly under
    ``preprocess='standard'``, where every coordinate has been scaled to
    unit variance already -- which is why that is this method's default
    standardisation.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Cloud the kernel will act on.
    bandwidth_scale : float, default=1.0
        Multiplies ``sigma^2`` (``> 1`` smooths, ``< 1`` sharpens), with
        the same sense as :class:`~src.kernels.Kernel`'s.

    Returns
    -------
    float
        ``gamma = 1 / (2 sigma^2)`` for ``exp(-gamma ||x - y||^2)``.
    """
    variance = float(np.mean(np.var(np.asarray(X, dtype=np.float64), axis=0)))
    sigma_sq = bandwidth_scale * 10.0 * max(variance, 1e-30)
    return 1.0 / (2.0 * sigma_sq)


class _ReducedKernel:
    """One space's kernel data map, ``x -> psi(x)`` (Eqs. 5-7).

    Holds the kernel, the basis points it is evaluated against and the
    optional column projection of the SVD routes, plus the column means
    of the calibration kernel data. Centring is what restricts the
    canonical analysis to *centred* variates, which is how the appendix
    disposes of the trivial pair ``rho_0 = 1, f_0 = g_0 = 1``: without it
    the leading canonical direction is the constant function and carries
    no information about either space.

    Note this is column-centring of the ``n x m`` kernel *data matrix*,
    the thing classical LCCA would centre, and not the double-centring of
    an ``n x n`` Gram matrix that
    :class:`~src.alignment.rkhs.RKHSAligner` does. The two agree only
    when the basis set is the whole sample and the projection is trivial.
    """

    def __init__(
        self,
        kernel: Kernel,
        points: np.ndarray,
        projection: np.ndarray | None = None,
    ) -> None:
        self.kernel = kernel
        self.points = points
        self.projection = projection
        self.col_mean_: np.ndarray | None = None

    @property
    def n_features(self) -> int:
        """Columns of the reduced kernel data, ``m``."""
        if self.projection is not None:
            return int(self.projection.shape[1])
        return int(self.points.shape[0])

    @property
    def n_parameters(self) -> int:
        """Numbers the map must carry to be evaluated on a new point.

        The kernel centres *and* the projection: a kernel machine is not
        free of its training data, and stating the cost is the honest
        comparison against a linear map. This is where ``'subset'`` pays
        off -- it stores ``m`` points where ``'svd'`` stores all ``n``.
        """
        projection = 0 if self.projection is None else self.projection.size
        return int(self.points.size + projection)

    def fit(self, Z: np.ndarray) -> np.ndarray:
        """Evaluate on the calibration cloud and learn the column means."""
        F = self._features(Z)
        self.col_mean_ = F.mean(axis=0)
        return F - self.col_mean_

    def __call__(self, Z: np.ndarray) -> np.ndarray:
        """Centred kernel data of new points, shape ``(len(Z), m)``.

        The means are the *calibration* ones, so a block of new points
        transforms identically whether it arrives all at once or a row at
        a time.
        """
        if self.col_mean_ is None:
            raise RuntimeError('_ReducedKernel.fit() must be called first.')
        return self._features(Z) - self.col_mean_

    def _features(self, Z: np.ndarray) -> np.ndarray:
        F = self.kernel(Z, self.points)
        return F if self.projection is None else F @ self.projection


def _subset_size(level: float | None, n_pool: int) -> int:
    """Basis-point count from an int, a fraction of the pool, or ``None``."""
    if level is None:
        return n_pool
    if isinstance(level, int):
        return min(level, n_pool)
    return max(1, min(round(level * n_pool), n_pool))


def _column_rank(
    sigma: np.ndarray, level: float | None, shape: tuple[int, int]
) -> int:
    """How many leading singular directions of the kernel data to keep.

    ``level`` is a count, a fraction of the *eigenvalue* mass, or
    ``None`` for every direction the kernel data actually spans. The
    fraction is on the singular values themselves rather than on their
    squares: the paper sizes its optimal basis so that the retained
    columns "make up 99% of eigenvalues of ``Lambda``", and for the
    symmetric ``K`` of the ``'svd'`` route the singular values *are* the
    eigenvalues. That is a different criterion from
    :class:`~src.alignment.cca.SVCCAAligner`'s float, which is a fraction
    of the variance and so sums squares.
    """
    rank = int(
        (sigma > sigma[0] * max(shape) * np.finfo(float).eps).sum()
        if sigma[0] > 0
        else 0
    )
    rank = max(rank, 1)
    if level is None:
        return rank
    if isinstance(level, int):
        return min(level, rank)
    kept = np.cumsum(sigma) / max(float(sigma.sum()), 1e-30)
    return min(int(np.searchsorted(kept, level) + 1), rank)


class KCCAAligner(Aligner):
    """Kernel canonical-correlation alignment on reduced kernel data.

    Fits the canonical stage of :class:`~src.alignment.cca.CCAAligner` to
    the two spaces' kernel data rather than to their coordinates, then
    decodes the shared canonical scores into the receiver's raw latents
    with a ridge readout. The map is genuinely non-linear -- unlike every
    other closed-form method here except
    :class:`~src.alignment.rkhs.RKHSAligner`, which reaches the same place
    from the other side, by *adding* a kernel correction to a rigid map
    instead of running the whole alignment in feature space.

    Parameters
    ----------
    kernel : {'rbf', 'laplacian', 'polynomial', 'linear', 'cosine'}
        Family of ``kappa_1``. The paper uses the Gaussian kernel
        throughout, which is ``'rbf'`` here and the default.
    kernel_tgt : str, optional
        Family of ``kappa_2``. ``None`` reuses ``kernel``. The two are
        separate kernels in the paper and the bandwidths are fitted per
        space regardless; this is for the case where the receiver's space
        should not be kernelised at all (Example 3 keeps its class
        indicator linear), for which ``'linear'`` is the setting.
    gamma : float, optional
        Inverse bandwidth of both kernels. ``None`` fits one per space by
        ``bandwidth``.
    bandwidth : {'rule_of_thumb', 'median'}, default='rule_of_thumb'
        How an unset ``gamma`` is chosen. ``'rule_of_thumb'`` is the
        paper's ``sigma^2 = 10 S`` (:func:`rule_of_thumb_gamma`);
        ``'median'`` is the median-pairwise-distance heuristic the rest of
        this package uses, which makes the kernel comparable with
        :class:`~src.alignment.rkhs.RKHSAligner`'s.
    bandwidth_scale : float, default=1.0
        Multiplier on whichever bandwidth heuristic is used.
    degree, coef0 : int, float
        Polynomial-kernel parameters.
    basis : {'svd', 'subset', 'hybrid'}, default='svd'
        Reduced-column route of Section 3.1; see the module docstring.
        The paper's own guidance: ``'svd'`` for small problems, where it
        is the better approximation, and ``'subset'`` as ``n`` grows,
        "for its simplicity and being economic".
    n_basis_src, n_basis_tgt : int | float | None, default=0.99
        Size of each space's reduced kernel data. Its meaning follows
        ``basis``: under ``'subset'`` a count of basis *points* (or a
        fraction of the pool); under ``'svd'`` / ``'hybrid'`` a count of
        singular directions, or a fraction of the eigenvalue mass -- the
        paper's 99% being the default. ``None`` reduces nothing, which
        hands the canonical stage a rank-deficient covariance and leaves
        ``reg`` to carry it.
    n_subset : int | None, default=None
        Random-subset size of ``basis='hybrid'``, drawn before the SVD
        cut. ``None`` uses the whole pool, at which point ``'hybrid'``
        degenerates to ``'svd'``.
    subset_strategy : str, default='random'
        How ``'subset'`` and ``'hybrid'`` draw their basis points; any
        design in :data:`~src.alignment.pilots.PILOT_STRATEGIES`. The
        paper uses ``'random'``, and ``'stratified'`` in its Pendigits
        example. Unlike a *pilot* design this one costs no airtime -- the
        basis set is a single-space object, chosen from local data.
    n_canonical : int | None, default=None
        Canonical directions kept, capped by ``min(m_1, m_2)``. This is
        the rate: the transmitter puts ``k`` numbers on the channel (see
        :attr:`transmitted_symbols`).
    reg : float, default=1e-3
        Ridge on both kernel-data covariances, relative to their mean
        eigenvalue. The paper argues the reduced-column step, not a
        ridge, is the right regulariser -- so treat this as a floor that
        keeps the two inversions finite and buy conditioning with
        ``n_basis_*`` instead.
    readout : {'source', 'target'}, default='source'
        Which side's canonical scores the pre-image readout is fitted on.
        ``'source'`` is the paper's Example 3 usage and the stronger of
        the two out of sample; ``'target'`` is the strict analogue of
        CCA's ``B^+``. See the module docstring.
    readout_reg : float, default=1e-6
        Relative ridge of that readout.
    max_points : int | None, default=2000
        Cap on the calibration points the basis is drawn from and the
        kernel data is estimated on. Guards the ``O(n^2 m)`` kernel
        evaluation when the calibration set is a whole training split.
        Unlike :class:`~src.alignment.cca.SVCCAAligner`, this method
        ignores ``src_context`` / ``tgt_context`` beyond the
        standardisation the base class fits from them: see the comment in
        :meth:`_fit`.
    preprocess : ScalingMethod, default='standard'
        Per-space standardisation. Unlike CCA, this method is *not*
        invariant to a change of basis -- the kernel sees the
        coordinates -- so the default is the z-scoring that makes the
        paper's coordinatewise window width exact.
    eps : float, default=1e-6
        Relative covariance ridge of the whitening step.
    seed : int, default=42
        Seed of the basis-subset draw and the bandwidth heuristic.

    Attributes
    ----------
    A_, B_ : np.ndarray
        Canonical bases of the two kernel data matrices, ``(m_i, k)``.
    R_ : np.ndarray
        Ridge readout from canonical scores to raw target latents,
        ``(k, d_tgt)``.
    correlations_ : np.ndarray
        The ``k`` retained canonical correlations.
    correlations_full_ : np.ndarray
        All ``min(m_1, m_2)`` of them, which is what the association
        measures of Section 4 are computed from.
    """

    def __init__(
        self,
        kernel: KernelName = 'rbf',
        kernel_tgt: KernelName | None = None,
        gamma: float | None = None,
        bandwidth: str = 'rule_of_thumb',
        bandwidth_scale: float = 1.0,
        degree: int = 3,
        coef0: float = 1.0,
        basis: BasisMethod = 'svd',
        n_basis_src: float | None = 0.99,
        n_basis_tgt: float | None = 0.99,
        n_subset: int | None = None,
        subset_strategy: str = 'random',
        n_canonical: int | None = None,
        reg: float = 1e-3,
        readout: str = 'source',
        readout_reg: float = 1e-6,
        max_points: int | None = 2000,
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
        if basis not in _BASIS_METHODS:
            raise ValueError(
                f'Unknown basis method {basis!r}. '
                f'Supported: {", ".join(_BASIS_METHODS)}.'
            )
        if bandwidth not in ('rule_of_thumb', 'median'):
            raise ValueError(
                f'Unknown bandwidth heuristic {bandwidth!r}; expected '
                "'rule_of_thumb' or 'median'."
            )
        if readout not in ('target', 'source'):
            raise ValueError(
                f"Unknown readout {readout!r}; expected 'source' or 'target'."
            )
        if subset_strategy not in PILOT_STRATEGIES:
            raise ValueError(
                f'Unknown subset strategy {subset_strategy!r}. '
                f'Supported: {", ".join(PILOT_STRATEGIES)}.'
            )
        if n_canonical is not None and int(n_canonical) < 1:
            raise ValueError(
                f'n_canonical must be >= 1 or None, got {n_canonical}.'
            )

        self.kernel = kernel
        self.kernel_tgt = kernel_tgt
        self.gamma = None if gamma is None else float(gamma)
        self.bandwidth = bandwidth
        self.bandwidth_scale = float(bandwidth_scale)
        self.degree = int(degree)
        self.coef0 = float(coef0)

        self.basis: BasisMethod = basis
        self.n_basis_src = _check_truncation(n_basis_src, 'n_basis_src')
        self.n_basis_tgt = _check_truncation(n_basis_tgt, 'n_basis_tgt')
        self.n_subset = None if n_subset is None else int(n_subset)
        self.subset_strategy = subset_strategy

        self.n_canonical = n_canonical
        self.reg = float(reg)
        self.readout = readout
        self.readout_reg = float(readout_reg)
        self.max_points = None if max_points is None else int(max_points)

        self.mean_tgt_: np.ndarray | None = None
        self.phi_src_: _ReducedKernel | None = None
        self.phi_tgt_: _ReducedKernel | None = None
        self.A_: np.ndarray | None = None
        self.B_: np.ndarray | None = None
        self.R_: np.ndarray | None = None
        self.correlations_: np.ndarray | None = None
        self.correlations_full_: np.ndarray | None = None

    def hyperparameters(self) -> dict:
        params = super().hyperparameters()
        params.update(
            kernel=self.kernel,
            kernel_tgt=self.kernel_tgt or self.kernel,
            gamma=self.gamma,
            bandwidth=self.bandwidth,
            bandwidth_scale=self.bandwidth_scale,
            basis=self.basis,
            n_basis_src=self.n_basis_src,
            n_basis_tgt=self.n_basis_tgt,
            n_subset=self.n_subset,
            subset_strategy=self.subset_strategy,
            n_canonical=self.n_canonical,
            reg=self.reg,
            readout=self.readout,
            readout_reg=self.readout_reg,
            max_points=self.max_points,
        )
        if 'polynomial' in (self.kernel, self.kernel_tgt):
            params.update(degree=self.degree, coef0=self.coef0)
        return params

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def _fit(
        self,
        Z_src: np.ndarray,
        Z_tgt: np.ndarray,
        labels: np.ndarray | None = None,
    ) -> None:
        n = Z_src.shape[0]
        rng = np.random.default_rng(self.seed)

        # Step (a): kernel data for each space, reduced in columns. Both
        # the kernel and the basis are estimated on the pilots and not on
        # any unpaired context, unlike
        # :class:`~src.alignment.cca.SVCCAAligner`'s SVD basis: they are
        # single-space objects, but the canonical stage that consumes
        # them is not, and it can only support as many kernel columns as
        # there are paired samples. A basis drawn from a larger local
        # pool outgrows the budget rather than improving on it -- on a
        # smooth synthetic pair at 30 to 100 pilots it cost 0.127 -> 0.197
        # held-out NMSE.
        columns = max(1, n - 1)
        self.phi_src_, info_src = self._reduced_kernel(
            Z_src, self.kernel, self.n_basis_src, labels, rng, columns
        )
        self.phi_tgt_, info_tgt = self._reduced_kernel(
            Z_tgt,
            self.kernel_tgt or self.kernel,
            self.n_basis_tgt,
            labels,
            rng,
            columns,
        )
        Psi_src = self.phi_src_.fit(Z_src)  # (n, m1), centred
        Psi_tgt = self.phi_tgt_.fit(Z_tgt)  # (n, m2), centred
        m_src, m_tgt = Psi_src.shape[1], Psi_tgt.shape[1]

        # Step (b): classical LCCA on the kernel data. The full canonical
        # spectrum is kept because the association measures of Section 4
        # are defined over all of it, not just the retained pairs.
        k_max = min(m_src, m_tgt)
        k = (
            k_max
            if self.n_canonical is None
            else min(int(self.n_canonical), k_max)
        )
        A, B, self.correlations_full_ = _canonical_bases(
            Psi_src, Psi_tgt, k_max, self.reg
        )
        self.A_, self.B_ = A[:, :k], B[:, :k]
        self.correlations_ = self.correlations_full_[:k]

        # The pre-image: canonical scores -> the receiver's raw latents.
        U, V = Psi_src @ self.A_, Psi_tgt @ self.B_
        self.mean_tgt_ = Z_tgt.mean(axis=0)
        self.R_ = _ridge_readout(
            V if self.readout == 'target' else U,
            Z_tgt - self.mean_tgt_,
            self.readout_reg,
        )
        self._record_diagnostics(n, U, V, Z_tgt, info_src, info_tgt)

        if n < m_src + m_tgt:
            log.debug(
                'KCCA fitted on %d pairs with (%d, %d) kernel columns: the '
                'canonical correlations are optimistically biased. Reduce '
                'n_basis_src/n_basis_tgt before raising reg=%g -- the '
                'reduced-column step removes the noisy directions where '
                'the ridge only damps them.',
                n,
                m_src,
                m_tgt,
                self.reg,
            )

    def _reduced_kernel(
        self,
        Z: np.ndarray,
        name: KernelName,
        n_basis: float | None,
        labels: np.ndarray | None,
        rng: np.random.Generator,
        max_columns: int,
    ) -> tuple[_ReducedKernel, dict]:
        """Build one space's reduced kernel data map (Eqs. 5-7).

        ``max_columns`` is the rank the calibration set can support --
        the centred kernel data of ``n`` pilots spans at most ``n - 1``
        directions, so columns past that are exactly null directions that
        ``reg`` would invert into noise. It binds only where the
        requested basis outgrows the budget, which is the regime the
        paper's own sizing rule does not visit (its kernel data and its
        LCCA see the same ``n``).
        """
        pool, keep = self._pool(Z, rng)
        kernel = Kernel(
            name=name,
            gamma=self.gamma,
            degree=self.degree,
            coef0=self.coef0,
            bandwidth_scale=self.bandwidth_scale,
        )
        if (
            kernel.gamma is None
            and self.bandwidth == 'rule_of_thumb'
            and name in ('rbf', 'laplacian')
        ):
            kernel.gamma = rule_of_thumb_gamma(pool, self.bandwidth_scale)
        kernel.fit(pool, seed=self.seed)  # no-op once gamma is set

        # Basis points: the whole calibration set, or a subset of it.
        size = None
        if self.basis == 'subset':
            # No projection follows, so the points *are* the columns and
            # the budget's cap applies to them directly.
            size = min(_subset_size(n_basis, pool.shape[0]), max_columns)
        elif self.basis == 'hybrid' and self.n_subset is not None:
            # Here they are only the pre-subset the SVD then cuts down.
            size = min(self.n_subset, pool.shape[0])

        points = pool
        if size is not None:
            points = pool[
                select_pilots(
                    pool,
                    size,
                    strategy=self.subset_strategy,
                    # Follow whatever `max_points` sub-sampled, so the
                    # labels still index the rows they are stratifying.
                    labels=None
                    if labels is None
                    else np.asarray(labels)[
                        slice(None) if keep is None else keep
                    ],
                    kernel=kernel,
                    seed=self.seed,
                )
            ]

        # Column projection: the leading singular directions of the
        # kernel data, for the two SVD routes.
        projection, mass = None, 1.0
        if self.basis in ('svd', 'hybrid'):
            M = kernel(pool, points)
            _, sigma, Vt = np.linalg.svd(M, full_matrices=False)
            columns = min(_column_rank(sigma, n_basis, M.shape), max_columns)
            projection = Vt[:columns].T
            mass = float(sigma[:columns].sum() / max(sigma.sum(), 1e-30))

        reduced = _ReducedKernel(kernel, points, projection)
        return reduced, {
            'basis_points': int(points.shape[0]),
            'columns': reduced.n_features,
            'eigenvalue_mass': mass,
            **kernel.summary(),
        }

    def _pool(
        self, Z: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """The cloud the kernel and the basis are estimated on."""
        source = Z
        keep = None
        if self.max_points is not None and source.shape[0] > self.max_points:
            keep = np.sort(
                rng.choice(
                    source.shape[0], size=self.max_points, replace=False
                )
            )
            source = source[keep]
        elif source.shape[0] > _LARGE_POOL_WARNING:
            log.warning(
                'Building kernel data against %d points: the reduction is '
                'O(n^2 m) in time and O(n m) in memory. Consider '
                'max_points.',
                source.shape[0],
            )
        return source, keep

    def _record_diagnostics(
        self,
        n: int,
        U: np.ndarray,
        V: np.ndarray,
        Z_tgt: np.ndarray,
        info_src: dict,
        info_tgt: dict,
    ) -> None:
        rho = self.correlations_
        Yc = Z_tgt - self.mean_tgt_
        energy = max(float(np.sum(Yc**2)), 1e-30)
        m_src, m_tgt = info_src['columns'], info_tgt['columns']

        self.diagnostics_ = {
            'kcca_n_canonical': len(rho),
            'kcca_mean_correlation': float(np.mean(rho)),
            # Pinned at ~1 across the board means the kernel data is
            # interpolating: with more columns than pilots every
            # direction can be made to correlate perfectly in-sample.
            'kcca_min_correlation': float(np.min(rho)),
            'kcca_columns_src': int(m_src),
            'kcca_columns_tgt': int(m_tgt),
            'kcca_basis_points_src': info_src['basis_points'],
            'kcca_basis_points_tgt': info_tgt['basis_points'],
            'kcca_eigenvalue_mass_src': info_src['eigenvalue_mass'],
            'kcca_eigenvalue_mass_tgt': info_tgt['eigenvalue_mass'],
            'kcca_gamma_src': info_src.get('kernel_gamma'),
            'kcca_gamma_tgt': info_tgt.get('kernel_gamma'),
            # Section 4's association measures, over the full spectrum.
            'kcca_r_max': float(self.correlations_full_[0]),
            'kcca_r_log': _log_association(self.correlations_full_),
            'kcca_independence_p': _bartlett_p(
                n, m_src, m_tgt, self.correlations_full_
            ),
            # How much of the target latents the *target's own* canonical
            # scores reconstruct: the ceiling the source side is being
            # decoded against, and a property of the receiver alone.
            'kcca_readout_r2': 1.0
            - float(np.sum((V @ self.R_ - Yc) ** 2) / energy),
            # The same through the deployed path. The gap between the two
            # is what the canonical correlations promised and the source
            # scores did not deliver.
            'kcca_transfer_r2': 1.0
            - float(np.sum((U @ self.R_ - Yc) ** 2) / energy),
        }

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def _transform(self, Z_src: np.ndarray) -> np.ndarray:
        return self.phi_src_(Z_src) @ self.A_ @ self.R_ + self.mean_tgt_

    def transmit(self, X_src: np.ndarray) -> np.ndarray:
        """The ``k`` canonical scores, i.e. what crosses the channel.

        The transmitter holds ``kappa_1``, its basis points and ``A``, so
        it can evaluate this locally; the receiver needs only ``R``. This
        and :meth:`receive` are :meth:`transform` split at the channel,
        and composing them reproduces it exactly.
        """
        self._check_fitted()
        Z = self.scaler_src_.transform(np.asarray(X_src, dtype=np.float64))
        return self.phi_src_(Z) @ self.A_

    def receive(self, U: np.ndarray) -> np.ndarray:
        """Decode canonical scores into the receiver's raw latents."""
        self._check_fitted()
        Z_hat = np.asarray(U, dtype=np.float64) @ self.R_ + self.mean_tgt_
        return self.scaler_tgt_.inverse_transform(Z_hat)

    # ------------------------------------------------------------------
    # What the method costs
    # ------------------------------------------------------------------

    @property
    def transmitted_symbols(self) -> int:
        """The canonical coordinates -- see :meth:`transmit`.

        Unlike the linear methods there is no competing bound from the
        standardised source dimension: the kernel data has ``m_1``
        columns whatever ``d_src`` is, so the canonical rank is the only
        thing setting the rate.
        """
        self._check_fitted()
        return int(self.A_.shape[1])

    @property
    def map_parameters(self) -> int:
        """The transmitter's kernel data map and ``A``, plus ``R``.

        The target-side objects (``kappa_2``, its basis points, ``B``) are
        fitting scaffolding: they build ``R`` and are never evaluated
        again, so they are not part of the deployed map and are not
        counted. What is counted is the price of the non-linearity -- the
        source basis points travel into deployment, exactly as
        :class:`~src.alignment.rkhs.RKHSAligner`'s retained pilots do.
        """
        self._check_fitted()
        return int(self.phi_src_.n_parameters + self.A_.size + self.R_.size)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _ridge_readout(S: np.ndarray, Y: np.ndarray, reg: float) -> np.ndarray:
    """Ridge least squares ``argmin_R ||S R - Y||^2``, both centred.

    The ridge is relative to the mean eigenvalue of ``S^T S``, so one
    setting behaves the same whatever scale the canonical scores came out
    at, and a rank-deficient score matrix -- which is what a pilot budget
    below the canonical rank produces -- still solves.
    """
    k = S.shape[1]
    G = S.T @ S
    G[np.diag_indices_from(G)] += reg * max(float(np.trace(G)) / k, 1e-30)
    return np.linalg.solve(G, S.T @ Y)


def _log_association(rho: np.ndarray) -> float:
    """``r_log = -sum_v log(1 - rho_v^2)``, Eq. (9).

    Correlations are clipped just below 1 so a canonical pair that
    interpolates the calibration set reports a large finite number rather
    than an infinity that would poison the summary.
    """
    r = np.clip(np.asarray(rho, dtype=np.float64), 0.0, 1.0 - 1e-12)
    return float(-np.sum(np.log1p(-(r**2))))


def _bartlett_p(n: int, m_src: int, m_tgt: int, rho: np.ndarray) -> float:
    """p-value of the independence test of Eq. (13).

    ``(n - (m_1 + m_2 + 1)/2) r_log`` is referred to a chi-squared
    distribution under independence of the two spaces.

    One deliberate deviation, in the degrees of freedom. Eq. (13) states
    ``(m_1 - 1)(m_2 - 1)``, which is classical Bartlett on ``m_1 - 1``
    and ``m_2 - 1`` variables: one direction per side is assumed spent on
    the constant function, as it would be if the basis functions spanned
    it exactly and centring therefore annihilated a dimension. Column-
    centring the kernel data does not do that -- it projects each column
    onto ``1^perp``, leaving all ``m_i`` of them linearly independent --
    so the statistic here has ``m_1 m_2`` free directions, and using the
    paper's count over-rejects badly at small ``m``.

    Measured on Case III-1 of the paper's Table 2 (bivariate standard
    normal, ``rho = 0``, ``n = 500``, 300 runs), where the rule-of-thumb
    window leaves ``m_1 = m_2 = 2``:

    ===========================  =====================
    degrees of freedom            type-I error at 0.05
    ===========================  =====================
    ``(m_1 - 1)(m_2 - 1) = 1``    0.40
    ``m_1 m_2 = 4``               0.053
    ===========================  =====================

    against the 0.04 the paper reports. The two counts converge as the
    bases grow, so this only bites in the regime the paper's own
    experiments do not visit.

    Returns NaN where the asymptotics have nothing to stand on --
    ``n <= m_1 + m_2``, fewer pilots than the two bases together -- which
    is the honest answer, since the correlations there are a function of
    an interpolating fit rather than of the data. It also assumes the
    smooth, light-tailed kernel data of the paper's Gaussian kernel: a
    polynomial kernel's is heavy-tailed enough that the chi-squared
    approximation reports significance on independent inputs.
    """
    df = m_src * m_tgt
    factor = n - 0.5 * (m_src + m_tgt + 1)
    if df < 1 or factor <= 0 or n <= m_src + m_tgt:
        return float('nan')
    return float(chi2.sf(factor * _log_association(rho), df))

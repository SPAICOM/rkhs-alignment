"""Residual Kernel Alignment (RKA): a rigid map plus an orthogonal wiggle.

Implements the two-stage map of ``idea.md``,

    f(z) = Q z + g(z),

where ``Q`` is the best semi-orthogonal (Procrustes) map between the two
whitened latent spaces and ``g`` is a kernel-ridge correction fitted to
whatever ``Q`` could not explain. The two stages decouple -- and therefore
both admit closed forms -- because ``g`` is constrained to be empirically
uncorrelated with the source directions ``Q`` already used:

    G X^T = 0     (Step 4 of the pipeline)

Without that constraint a plain kernel ridge regression of the residual
would quietly re-absorb linear structure that belongs to ``Q``, and the
split between "rigid" and "residual" would stop meaning anything.

Pipeline (rows are samples here; ``idea.md`` stacks them as columns, so
every expression below is the transpose of the one written there):

===== ===================================================================
Step   Operation
===== ===================================================================
0      Whiten both spaces (handled by :class:`~src.alignment.base.Aligner`)
1      ``Q = orthogonal_procrustes(Z_src, Z_tgt)``
2      ``E = Z_tgt - Z_src Q^T``
3      Centred Gram ``Kc``; smoother ``H = Kc (Kc + N lam I)^-1``
4      ``B = Z_src^T H Z_src``;  ``H_perp = H - H Z_src B^-1 Z_src^T H``
5      ``G = H_perp E``  (fitted);  ``A = (Kc + N lam I)^-1 M``  (dual)
6      ``f(z*) = Q z* + kc(z*)^T A``
===== ===================================================================

Everything is solved through :func:`numpy.linalg.solve` rather than
explicit inverses, and the whole ``lam`` grid shares a *single*
eigendecomposition of ``Kc``.

That eigendecomposition is purely a computational device -- it is not a
modelling choice and it is deliberately absent from ``idea.md``, which
states the same estimator in matrix form. With ``Kc = V diag(s) V^T`` and
``V`` orthogonal, every object the pipeline needs is a reweighting of the
*same* basis::

    (Kc + N lam I)^-1 = V diag(1 / (s + N lam)) V^T
    H_lam             = V diag(s / (s + N lam)) V^T

``V`` does not depend on ``lam``, so ``V^T X`` and ``V^T E`` are computed
once (this is what :attr:`VtX` and :attr:`VtE` hold) and each grid point
costs ``O(N d^2)`` instead of an ``O(N^3)`` solve. The reweighting
``h(s) = s / (s + N lam)`` is a low-pass filter on the eigenbasis of the
kernel operator -- the same picture as graph-Fourier filtering, with the
Gram matrix playing the role of the graph -- and the results are
identical to forming ``H_lam`` explicitly, to machine precision (and
better conditioned at small ``lam``, since ``H_lam`` is never built).
"""

from __future__ import annotations

import logging

import numpy as np

from ..kernels import Kernel, KernelName
from .base import Aligner
from .procrustes import ProcrustesFit, orthogonal_procrustes

log = logging.getLogger(__name__)

__all__ = ['RKHSAligner']

# Beyond this many calibration points the O(N^3) eigendecomposition and
# the O(N^2) Gram matrix start to dominate the whole experiment.
_LARGE_N_WARNING = 8000


class _ResidualSolver:
    """Eigendecomposed residual problem, reusable across ``lam`` values.

    Holds the one-off ``O(N^3)`` work (centred Gram matrix and its
    eigendecomposition) so that :meth:`solve` is ``O(N d^2)`` per ``lam``.
    """

    def __init__(
        self,
        X: np.ndarray,
        E: np.ndarray,
        kernel: Kernel,
        center: bool,
        ridge_B: float,
        orthogonal: bool,
    ) -> None:
        self.n, self.d_src = X.shape
        self.X = X
        self.E = E
        self.kernel = kernel
        self.center = center
        self.ridge_B = float(ridge_B)
        self.orthogonal = bool(orthogonal)

        K = kernel(X, X)
        if center:
            # Kernel-PCA double centring: Kc = J K J with J = I - 11^T/N,
            # i.e. the Gram matrix of the *centred* feature map
            # phi~(x) = phi(x) - mean_a phi(x_a). Expanding that inner
            # product is where all three terms come from:
            #   <phi(x_i) - pbar, phi(x_j) - pbar>
            #     = K_ij - mean_a K_aj - mean_b K_ib + mean_ab K_ab
            # The grand mean is the <pbar, pbar> term, not a fudge: drop
            # it and the result is no longer a Gram matrix of anything,
            # so it stops being PSD (its smallest eigenvalue goes deeply
            # negative) and the eigenvalue reweighting below is solving a
            # problem with no RKHS behind it.
            #
            # Note this is centring in *feature* space, which is not the
            # same as centring X before evaluating the kernel: for an RBF
            # or Laplacian kernel the latter is an exact no-op (K depends
            # only on ||x_i - x_j||), and only for the linear kernel do
            # the two coincide.
            #
            # Why centre at all: the model has no intercept in feature
            # space, so without this the RKHS term soaks up a constant
            # offset instead of genuine non-linear structure. It also
            # makes 1 an eigenvector of Kc at eigenvalue 0, hence
            # 1^T H = 0, hence the fitted correction G has exactly zero
            # empirical mean -- the residual stage cannot smuggle in a
            # translation that belongs to Step 0.
            self.col_mean_ = K.mean(axis=0)
            self.grand_ = float(K.mean())
            Kc = K - self.col_mean_[None, :] - self.col_mean_[:, None]
            Kc += self.grand_
        else:
            self.col_mean_ = np.zeros(self.n)
            self.grand_ = 0.0
            Kc = K
        Kc = 0.5 * (Kc + Kc.T)  # enforce exact symmetry before eigh

        s, V = np.linalg.eigh(Kc)
        # A Gram matrix of a positive-definite kernel is PSD for *any*
        # point set, so the only negative eigenvalues that should appear
        # here are rounding noise -- and they do: centring alone leaves
        # the constant direction at ~1e-14, either sign. Anything larger
        # means the kernel itself is not positive definite (a polynomial
        # kernel with negative coef0, say), which invalidates the
        # representer-theorem argument the whole stage rests on.
        self.min_eigenvalue_ = float(s.min())
        self.max_eigenvalue_ = float(s.max())
        tolerance = 1e-8 * max(self.max_eigenvalue_, 1.0)
        if self.min_eigenvalue_ < -tolerance:
            log.warning(
                'Centred Gram matrix has eigenvalue %.3e (largest %.3e): '
                'the kernel is not positive semi-definite, so it does not '
                'define an RKHS and the residual stage is not solving the '
                'stated problem.',
                self.min_eigenvalue_,
                self.max_eigenvalue_,
            )
        self.s = np.clip(s, 0.0, None)
        self.V = V
        self.VtX = V.T @ X
        self.VtE = V.T @ E

        # Capacity of the residual stage. Rows of G live in
        # range(K) n ker(X), so the usable dimension is the numerical rank
        # of the centred Gram minus the rank of X. When that is <= 0 the
        # constraint admits only G = 0 and RKA returns the Procrustes
        # solution at every lambda -- silently, because the ridge on B
        # keeps the linear algebra well posed. Two distinct causes land
        # here: too few pilots (rank X = N - 1 >= rank K), and a kernel
        # whose RKHS is the linear functions the constraint removes
        # (linear/cosine, where rank K <= d_src).
        cutoff = max(self.s.max(), 1.0) * self.n * np.finfo(float).eps
        self.gram_rank_ = int((self.s > cutoff).sum())
        self.x_rank_ = int(np.linalg.matrix_rank(X))
        self.capacity_ = self.gram_rank_ - self.x_rank_
        self.gram_cond_ = (
            float(self.s.max() / max(self.s[self.s > cutoff].min(), 1e-300))
            if self.gram_rank_
            else float('inf')
        )

        if self.capacity_ <= 0:
            log.warning(
                'Residual stage has no degrees of freedom: rank(K)=%d, '
                'rank(X)=%d over N=%d pilots in %d dimensions. The '
                'constraint G X^T = 0 then forces G = 0, so this fit '
                'returns the Procrustes solution at every lambda. Use more '
                'pilots (N > d_src), reduce d_src, or pick a kernel whose '
                'RKHS is larger than the linear functions.',
                self.gram_rank_,
                self.x_rank_,
                self.n,
                self.d_src,
            )

    def solve(self, lam: float) -> dict[str, np.ndarray | float]:
        """Fit the residual stage at regularisation ``lam``.

        Parameters
        ----------
        lam : float
            Ridge strength; the effective shift is ``N * lam``, matching
            ``H = K (K + N lam I)^-1``.

        Returns
        -------
        dict
            ``A`` (dual coefficients, ``(N, d_tgt)``), ``G`` (fitted
            residual on the calibration points, ``(N, d_tgt)``) and
            ``dof`` (trace of the smoother, i.e. effective degrees of
            freedom).
        """
        if lam <= 0:
            raise ValueError(f'lam must be strictly positive, got {lam}.')
        shift = self.s + self.n * lam
        h = self.s / shift  # eigenvalues of H_lam

        if self.orthogonal:
            # B = X^T H X and P = X^T H E, both assembled in the
            # eigenbasis so H is never formed explicitly.
            hVtX = h[:, None] * self.VtX
            B = self.VtX.T @ hVtX
            if self.ridge_B > 0:
                B[np.diag_indices_from(B)] += self.ridge_B * max(
                    float(np.trace(B)) / self.d_src, 1e-30
                )
            P = hVtX.T @ self.VtE
            VtM = self.VtE - self.VtX @ np.linalg.solve(B, P)
        else:
            VtM = self.VtE

        A = self.V @ (VtM / shift[:, None])
        G = self.V @ (h[:, None] * VtM)
        return {'A': A, 'G': G, 'dof': float(np.sum(h))}

    def center_cross(self, k: np.ndarray) -> np.ndarray:
        """Centre an out-of-sample cross-kernel block ``(m, N)``.

        Same identity as the training block, with ``pbar`` still the mean
        over the *training* points::

            <phi(x*) - pbar, phi(x_j) - pbar>
              = k(x*, x_j) - mean_b k(x*, x_b)   # this row's own mean
                           - col_mean_[j]        # training column mean
                           + grand_

        The row mean is taken over the training columns, so test points
        are never centred against each other -- a block of test points
        must transform identically whether it arrives all at once or one
        row at a time.
        """
        if not self.center:
            return k
        return (
            k - k.mean(axis=1, keepdims=True) - self.col_mean_[None, :]
        ) + self.grand_


class RKHSAligner(Aligner):
    """Residual Kernel Alignment: Procrustes plus a constrained RKHS term.

    Parameters
    ----------
    kernel : {'rbf', 'laplacian', 'polynomial', 'linear', 'cosine'}
        Kernel family of the residual stage. ``'rbf'`` is the default.
    gamma : float, optional
        Inverse bandwidth; ``None`` fits it with the median-pairwise-
        distance heuristic on the whitened source cloud.
    bandwidth_scale : float, default=1.0
        Multiplier on the median-heuristic bandwidth.
    degree, coef0 : int, float
        Polynomial-kernel parameters.
    lam : float, default=1e-3
        Ridge strength of the residual stage (used when ``lam_grid`` is
        ``None``).
    lam_grid : list[float], optional
        Log-spaced grid to sweep. When given, ``lam`` is selected on a
        held-out slice of the calibration set and the model is refitted
        on all of it; the whole sweep is kept in :attr:`lambda_path_`.
    val_fraction : float, default=0.2
        Held-out fraction used for that selection.
    selection : {'nmse', 'cosine'}, default='nmse'
        Criterion optimised over ``lam_grid``.
    center_kernel : bool, default=True
        Double-centre the Gram matrix (strongly recommended: the model
        has no explicit intercept in feature space).
    orthogonal_residual : bool, default=True
        Enforce ``G X^T = 0``. Setting it to ``False`` degrades the
        method to plain kernel ridge on the Procrustes residual, which is
        the natural ablation for the identifiability constraint.
    ridge_B : float, default=1e-8
        Relative ridge added to ``B = X^T H X`` before solving; needed
        when ``N`` is small relative to ``d_src``.
    max_points : int, optional
        Cap on the calibration points fed to the kernel stage (``Q`` is
        always fitted on all of them). Guards the ``O(N^3)`` solve.
    preprocess : ScalingMethod, default='whiten'
        Per-space standardisation; ``'whiten'`` is Step 0 of the pipeline.
    eps : float, default=1e-6
        Relative covariance ridge of the whitening step.
    seed : int, default=42
        Seed of the bandwidth heuristic, the validation split and any
        sub-sampling.
    """

    def __init__(
        self,
        kernel: KernelName = 'rbf',
        gamma: float | None = None,
        bandwidth_scale: float = 1.0,
        degree: int = 3,
        coef0: float = 1.0,
        lam: float = 1e-3,
        lam_grid: list[float] | None = None,
        val_fraction: float = 0.2,
        selection: str = 'nmse',
        center_kernel: bool = True,
        orthogonal_residual: bool = True,
        ridge_B: float = 1e-8,
        max_points: int | None = None,
        preprocess: str = 'whiten',
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
        if selection not in ('nmse', 'cosine'):
            raise ValueError(f'Unknown selection criterion {selection!r}.')

        self.kernel_spec = Kernel(
            name=kernel,
            gamma=gamma,
            degree=degree,
            coef0=coef0,
            bandwidth_scale=bandwidth_scale,
        )
        self.lam = float(lam)
        self.lam_grid = (
            None if lam_grid is None else [float(v) for v in lam_grid]
        )
        self.val_fraction = float(val_fraction)
        self.selection = selection
        self.center_kernel = bool(center_kernel)
        self.orthogonal_residual = bool(orthogonal_residual)
        self.ridge_B = float(ridge_B)
        self.max_points = None if max_points is None else int(max_points)

        self.procrustes_: ProcrustesFit | None = None
        self.lam_: float | None = None
        self.lambda_path_: list[dict[str, float]] = []
        self.A_: np.ndarray | None = None
        self._solver: _ResidualSolver | None = None
        self._X_kernel: np.ndarray | None = None

    def hyperparameters(self) -> dict:
        params = super().hyperparameters()
        params.update(self.kernel_spec.summary())
        params.update(
            lam=self.lam,
            lam_grid=self.lam_grid,
            selection=self.selection,
            center_kernel=self.center_kernel,
            orthogonal_residual=self.orthogonal_residual,
            ridge_B=self.ridge_B,
            max_points=self.max_points,
        )
        return params

    @property
    def Q(self) -> np.ndarray:
        """The fitted semi-orthogonal map, shape ``(d_tgt, d_src)``."""
        self._check_fitted()
        return self.procrustes_.Q

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def _fit(
        self,
        Z_src: np.ndarray,
        Z_tgt: np.ndarray,
        labels: np.ndarray | None = None,
    ) -> None:
        # --- Steps 1-2: rigid map and its residual -------------------
        self.procrustes_ = orthogonal_procrustes(Z_src, Z_tgt)
        E_full = Z_tgt - self.procrustes_.apply(Z_src)

        # --- Optional sub-sampling of the O(N^3) kernel stage --------
        rng = np.random.default_rng(self.seed)
        idx = np.arange(Z_src.shape[0])
        if self.max_points is not None and idx.size > self.max_points:
            idx = np.sort(
                rng.choice(idx.size, size=self.max_points, replace=False)
            )
            log.info(
                'RKHS stage sub-sampled to %d of %d calibration points.',
                idx.size,
                Z_src.shape[0],
            )
        elif idx.size > _LARGE_N_WARNING:
            log.warning(
                'Fitting the RKHS stage on %d points: the exact solve is '
                'O(N^3) in time and O(N^2) in memory. Consider max_points.',
                idx.size,
            )
        X, E = Z_src[idx], E_full[idx]

        # --- Step 3: kernel machinery --------------------------------
        self.kernel_spec.fit(X, seed=self.seed)

        # --- Regularisation selection --------------------------------
        self.lam_ = (
            self.lam
            if not self.lam_grid
            else self._select_lambda(X, Z_tgt[idx])
        )

        # --- Steps 4-5: constrained fit on the full calibration set ---
        self._solver = _ResidualSolver(
            X,
            E,
            kernel=self.kernel_spec,
            center=self.center_kernel,
            ridge_B=self.ridge_B,
            orthogonal=self.orthogonal_residual,
        )
        solution = self._solver.solve(self.lam_)
        self.A_ = solution['A']
        self._X_kernel = X

        self._record_diagnostics(X, E, Z_tgt, solution)

    def _select_lambda(self, X: np.ndarray, Y: np.ndarray) -> float:
        """Sweep ``lam_grid`` on a held-out slice of the calibration set.

        The rigid stage is refitted on the training slice alone. Reusing
        the residual of the *full* ``Q`` would let the validation rows
        influence the target they are then scored against -- weakly, since
        each row is one of ``N`` in a cross-covariance, but the whole
        point of the split is that it does not happen at all.
        """
        n = X.shape[0]
        n_val = max(1, round(self.val_fraction * n))
        if n_val >= n:
            raise ValueError(
                f'val_fraction={self.val_fraction} leaves no training rows '
                f'out of {n} calibration samples.'
            )
        perm = np.random.default_rng(self.seed).permutation(n)
        val_idx, fit_idx = perm[:n_val], perm[n_val:]

        # The residual to model is the one this sub-fit's own Q leaves
        # behind, so the selection never sees the validation targets.
        q = orthogonal_procrustes(X[fit_idx], Y[fit_idx])
        E_fit = Y[fit_idx] - q.apply(X[fit_idx])
        E_val = Y[val_idx] - q.apply(X[val_idx])

        solver = _ResidualSolver(
            X[fit_idx],
            E_fit,
            kernel=self.kernel_spec,
            center=self.center_kernel,
            ridge_B=self.ridge_B,
            orthogonal=self.orthogonal_residual,
        )
        k_val = solver.center_cross(self.kernel_spec(X[val_idx], X[fit_idx]))

        self.lambda_path_ = []
        for lam in self.lam_grid:
            G_val = k_val @ solver.solve(lam)['A']
            self.lambda_path_.append(
                {
                    'lam': float(lam),
                    'val_nmse': _nmse(G_val, E_val),
                    'val_cosine': _mean_cosine(G_val, E_val),
                }
            )

        key = 'val_nmse' if self.selection == 'nmse' else 'val_cosine'
        best = (
            min(self.lambda_path_, key=lambda r: r[key])
            if self.selection == 'nmse'
            else max(self.lambda_path_, key=lambda r: r[key])
        )
        log.info(
            'RKHS: selected lam=%.4g (%s=%.4f) out of %d grid points.',
            best['lam'],
            key,
            best[key],
            len(self.lambda_path_),
        )
        return float(best['lam'])

    def _record_diagnostics(
        self,
        X: np.ndarray,
        E: np.ndarray,
        Z_tgt: np.ndarray,
        solution: dict,
    ) -> None:
        G = solution['G']
        norm_G = float(np.linalg.norm(G))
        norm_X = float(np.linalg.norm(X))
        self.diagnostics_ = {
            'rkhs_lam': self.lam_,
            'rkhs_dof': solution['dof'],
            'rkhs_n_kernel_points': int(X.shape[0]),
            # Should be ~0 (the constant direction that centring removes).
            # A meaningfully negative value means the kernel is not PSD.
            'rkhs_gram_min_eig': self._solver.min_eigenvalue_,
            # Conditioning over the non-null spectrum; the raw max/min
            # ratio is meaningless because centring always leaves a zero.
            'rkhs_gram_cond': self._solver.gram_cond_,
            'rkhs_gram_rank': self._solver.gram_rank_,
            # rank(K) - rank(X): the residual's degrees of freedom. Zero
            # or less means RKA is exactly Procrustes.
            'rkhs_capacity': self._solver.capacity_,
            'procrustes_regime': self.procrustes_.regime,
            # Share of the target that the rigid stage left on the table.
            'rkhs_residual_share': float(
                np.linalg.norm(E) / max(np.linalg.norm(Z_tgt), 1e-12)
            ),
            # Fraction of that residual the RKHS term recovers in-sample.
            'rkhs_residual_r2': 1.0 - _nmse(G, E),
            # Sanity check of Step 4: this must be ~0 (machine precision).
            'rkhs_orthogonality': float(
                np.linalg.norm(G.T @ X) / max(norm_G * norm_X, 1e-12)
            ),
        }
        if self.lambda_path_:
            self.diagnostics_['rkhs_lam_grid_size'] = len(self.lambda_path_)

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    @property
    def map_parameters(self) -> int:
        """``Q``, the dual coefficients, *and* the retained pilots.

        This is the price of the non-linearity and it is worth stating
        plainly: a kernel machine evaluates its correction against stored
        calibration points, so those points are part of the deployed map.
        The receiver therefore holds ``d_tgt x d_src`` for the rigid
        stage plus ``N_kernel x (d_src + d_tgt)`` for the residual, where
        a purely linear method holds only the first term.
        """
        self._check_fitted()
        return self.procrustes_.Q.size + self._X_kernel.size + self.A_.size

    def _transform(self, Z_src: np.ndarray) -> np.ndarray:
        linear = self.procrustes_.apply(Z_src)
        k = self._solver.center_cross(self.kernel_spec(Z_src, self._X_kernel))
        return linear + k @ self.A_

    def transform_linear(self, X_src: np.ndarray) -> np.ndarray:
        """Map source latents with the rigid stage only (``g == 0``).

        This is the ``lam -> inf`` limit of the method and the pure
        Procrustes baseline, useful both as a reference and as the unit
        test of the implementation.
        """
        self._check_fitted()
        X_src = np.asarray(X_src, dtype=np.float64)
        Z = self.scaler_src_.transform(X_src)
        return self.scaler_tgt_.inverse_transform(self.procrustes_.apply(Z))


# ---------------------------------------------------------------------
# Scoring helpers used during the lambda sweep
# ---------------------------------------------------------------------


def _nmse(pred: np.ndarray, true: np.ndarray) -> float:
    """Squared error normalised by the energy of ``true``."""
    denom = float(np.sum(true**2))
    return float(np.sum((pred - true) ** 2) / max(denom, 1e-30))


def _mean_cosine(pred: np.ndarray, true: np.ndarray) -> float:
    """Mean row-wise cosine similarity."""
    pn = np.maximum(np.linalg.norm(pred, axis=1), 1e-12)
    tn = np.maximum(np.linalg.norm(true, axis=1), 1e-12)
    return float(np.mean(np.sum(pred * true, axis=1) / (pn * tn)))

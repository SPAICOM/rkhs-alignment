"""Neural alignment baselines: Direct MLP and Residual MLP.

The two comparators of the RKA paper (Sec. V, Fig. 3), kept
architecturally identical so the only thing separating them is *what*
they are asked to learn:

- :class:`DirectMLPAligner` learns the complete TX-to-RX mapping from
  scratch.
- :class:`ResidualMLPAligner` keeps the Procrustes solution as a fixed
  backbone and learns only the correction on top of it -- the same
  decomposition ``f(x) = Qx + g(x)`` as
  :class:`~src.alignment.rkhs.RKHSAligner`, but with ``g`` a trained
  network instead of a closed-form constrained kernel ridge.

The pair therefore isolates two separate questions: whether a Procrustes
backbone helps at all (Direct vs Residual), and whether the RKHS residual
is more sample-efficient than a learned one (Residual MLP vs RKA). The
paper's answer to both is yes, most visibly in the low-pilot regime.

Both are trained with L-BFGS rather than SGD: with one 16-unit hidden
layer and a pilot budget in the tens, a full-batch quasi-Newton solver is
both faster and better conditioned than a stochastic one.

One caveat on the paper's defaults. It fixes the hidden layer at 16 units
for 12- and 16-dimensional latents, where that is ample. Carried
unchanged onto SEMASIA ViT latents (192-1024 dimensions) the same 16
units become a hard rank-16 bottleneck on the whole map, which is most of
why :class:`DirectMLPAligner` collapses there while
:class:`ResidualMLPAligner` -- whose network only has to carry a residual
on top of a full-rank ``Q`` -- does not. Raise ``hidden_units`` to
compare architectures on their merits rather than to reproduce the
paper's numbers.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor

from .base import Aligner
from .procrustes import ProcrustesFit, orthogonal_procrustes

if TYPE_CHECKING:
    from collections.abc import Sequence

log = logging.getLogger(__name__)

__all__ = ['DirectMLPAligner', 'MLPAligner', 'ResidualMLPAligner']


def _net_parameters(net) -> int:
    """Weights plus biases of a fitted scikit-learn MLP."""
    return sum(w.size for w in net.coefs_) + sum(
        b.size for b in net.intercepts_
    )


class MLPAligner(Aligner):
    """Shared plumbing for the two MLP baselines.

    Parameters
    ----------
    hidden_units : int | Sequence[int], default=16
        Hidden layer widths. An ``int`` is a single layer (the paper's
        setting); a sequence stacks layers. Raise it for wide latents --
        16 units between a 768- and a 384-dimensional space is a rank-16
        bottleneck on the entire map.
    activation : str, default='tanh'
        Hidden-layer non-linearity.
    alpha : float, default=1e-1
        L2 penalty, the paper's regularisation for both baselines.
    solver : {'lbfgs', 'adam'}, default='lbfgs'
        Optimiser. L-BFGS suits the small-network, few-pilot regime;
        ``'adam'`` scales better once the network is large.
    max_iter : int, default=1000
        Iteration cap.
    preprocess : ScalingMethod, default='whiten'
        Per-space standardisation, matching the other aligners so the
        comparison is like-for-like.
    eps : float, default=1e-6
        Relative covariance ridge of the whitening step.
    seed : int, default=42
        Weight-initialisation seed.
    """

    def __init__(
        self,
        hidden_units: int | Sequence[int] = 16,
        activation: str = 'tanh',
        alpha: float = 1e-1,
        solver: str = 'lbfgs',
        max_iter: int = 1000,
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
        self.hidden_units = (
            (int(hidden_units),)
            if isinstance(hidden_units, (int, np.integer))
            else tuple(int(h) for h in hidden_units)
        )
        self.activation = activation
        self.alpha = float(alpha)
        self.solver = solver
        self.max_iter = int(max_iter)
        self.net_: MLPRegressor | None = None

    def hyperparameters(self) -> dict:
        params = super().hyperparameters()
        params.update(
            hidden_units=list(self.hidden_units),
            activation=self.activation,
            alpha=self.alpha,
            solver=self.solver,
            max_iter=self.max_iter,
        )
        return params

    def _train(self, Z_src: np.ndarray, target: np.ndarray) -> MLPRegressor:
        """Fit the regressor, with non-convergence downgraded."""
        net = MLPRegressor(
            hidden_layer_sizes=self.hidden_units,
            activation=self.activation,
            solver=self.solver,
            alpha=self.alpha,
            max_iter=self.max_iter,
            random_state=self.seed,
        )
        # With very few pilots the solver routinely hits the iteration
        # cap; that is the regime under study, not a misconfiguration, so
        # it should not spray warnings over a sweep.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', ConvergenceWarning)
            net.fit(Z_src, target)
        return net


class DirectMLPAligner(MLPAligner):
    """Learn the whole TX-to-RX map with a small MLP.

    No geometric prior at all: the network has to discover the rotation
    and the non-linearity together, which is what makes it the most
    pilot-hungry method in the comparison.
    """

    @property
    def name(self) -> str:
        return 'direct_mlp'

    def _fit(
        self,
        Z_src: np.ndarray,
        Z_tgt: np.ndarray,
        labels: np.ndarray | None = None,
    ) -> None:
        self.net_ = self._train(Z_src, Z_tgt)
        self.diagnostics_ = {'mlp_train_loss': float(self.net_.loss_)}

    @property
    def map_parameters(self) -> int:
        """Every weight and bias of the network."""
        self._check_fitted()
        return _net_parameters(self.net_)

    def _transform(self, Z_src: np.ndarray) -> np.ndarray:
        return self.net_.predict(Z_src)


class ResidualMLPAligner(MLPAligner):
    """Procrustes backbone plus a *learned* residual.

    The direct counterpart to :class:`~src.alignment.rkhs.RKHSAligner`:
    same two-stage decomposition, same fixed ``Q``, but the residual is a
    trained MLP rather than a constrained kernel ridge -- and, unlike the
    RKHS one, it carries no orthogonality constraint, so nothing stops it
    re-absorbing linear structure that already belongs to ``Q``.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.procrustes_: ProcrustesFit | None = None

    @property
    def name(self) -> str:
        return 'residual_mlp'

    def _fit(
        self,
        Z_src: np.ndarray,
        Z_tgt: np.ndarray,
        labels: np.ndarray | None = None,
    ) -> None:
        self.procrustes_ = orthogonal_procrustes(Z_src, Z_tgt)
        residual = Z_tgt - self.procrustes_.apply(Z_src)
        self.net_ = self._train(Z_src, residual)

        fitted = self.net_.predict(Z_src)
        self.diagnostics_ = {
            'mlp_train_loss': float(self.net_.loss_),
            'procrustes_regime': self.procrustes_.regime,
            # Unconstrained, so this is generally far from zero -- the
            # contrast with rkhs_orthogonality is the point.
            'mlp_orthogonality': float(
                np.linalg.norm(fitted.T @ Z_src)
                / max(
                    np.linalg.norm(fitted) * np.linalg.norm(Z_src),
                    1e-12,
                )
            ),
        }

    @property
    def map_parameters(self) -> int:
        """The fixed rigid backbone plus the learned correction."""
        self._check_fitted()
        return self.procrustes_.Q.size + _net_parameters(self.net_)

    def _transform(self, Z_src: np.ndarray) -> np.ndarray:
        return self.procrustes_.apply(Z_src) + self.net_.predict(Z_src)

    def transform_linear(self, X_src: np.ndarray) -> np.ndarray:
        """Map with the rigid stage only, ignoring the learned residual."""
        self._check_fitted()
        Z = self.scaler_src_.transform(np.asarray(X_src, dtype=np.float64))
        return self.scaler_tgt_.inverse_transform(self.procrustes_.apply(Z))

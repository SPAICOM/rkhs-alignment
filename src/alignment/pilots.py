"""Semantic-pilot selection for fitting an alignment map.

A *semantic pilot* is a paired calibration sample ``(x_i, y_i)``: the TX
and RX representations of the same input, which the two devices exchange
to estimate the alignment. Pilots cost airtime, so the question every
method faces is how few of them can be used before the map degrades --
and, given a budget, *which* ones to spend it on.

Anchors and pilots are the same object seen from two angles: a subset of
the latent distribution chosen to represent it well. So rather than
inventing a parallel mechanism, this module reuses
:class:`src.anchors.Anchor` and simply demands the answer come back as
*sample indices* -- pilots must be actual observations, since the RX side
has to be able to look up its own representation of the very same input.
That rules out strategies returning synthetic points (k-means centroids
without ``medoids``), which :func:`select_pilots` rejects explicitly.

The interesting strategy here is ``'herding'``, the kernel-aware design
of the RKA paper (Sec. IV): it picks the pilot set whose kernel mean
embedding best matches the full candidate pool's, in the same RKHS the
residual stage will later be fitted in.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from ..anchors import Anchor, AnchorStrategy

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..kernels import Kernel

log = logging.getLogger(__name__)

__all__ = [
    'PILOT_STRATEGIES',
    'scheduled_bandwidth',
    'select_pilot_path',
    'select_pilots',
]

# Strategies that yield real sample indices, hence usable as pilots.
PILOT_STRATEGIES: tuple[str, ...] = (
    'random',
    'kmeans',
    'fps',
    'stratified',
    'herding',
    'round_robin',
)

# Strategies whose answer for budget k is the k-point prefix of their
# answer for any larger budget, so a sweep can select once and slice.
# 'fps' and 'herding' are greedy; 'round_robin' builds one ordering of the
# whole pool from the seed alone and truncates it.
_GREEDY_STRATEGIES: frozenset[str] = frozenset(
    {'fps', 'herding', 'round_robin'}
)


def scheduled_bandwidth(
    n_pilots: int,
    n_pool: int,
    base: float = 3.9,
    power: float = 0.7,
) -> float:
    """Budget-dependent bandwidth multiplier for kernel herding.

    ``base * (n_pilots / n_pool) ** power``, so the herding kernel widens
    as the budget grows relative to the pool. The motivation is that a
    sharp kernel makes the redundancy term local, so once the modes are
    covered the score degenerates to a density ranking; a wider kernel
    keeps a diversity pressure alive.

    Measured on the Digits pool, however, this buys nothing: averaged
    over resampled pools the fixed default (0.2541) is best or within
    noise at every budget, and no bandwidth trend with ``n_pilots``
    survives averaging. The helper is kept because the mechanism is real
    and may bite on other data, but do not switch it on without checking
    it against a fixed bandwidth first -- and note that a budget-dependent
    kernel makes the selection *non-nested*, so a sweep can no longer
    slice one ordering (see :func:`select_pilot_path`).

    Parameters
    ----------
    n_pilots : int
        Budget the bandwidth is being chosen for.
    n_pool : int
        Size of the candidate pool.
    base, power : float
        Schedule coefficients.

    Returns
    -------
    float
        Multiplier to pass as ``Kernel(bandwidth_scale=...)``.
    """
    if n_pool <= 0:
        raise ValueError(f'n_pool must be positive, got {n_pool}.')
    return float(base * (n_pilots / n_pool) ** power)


def select_pilots(
    X: np.ndarray,
    n_pilots: int,
    strategy: AnchorStrategy = 'random',
    labels: np.ndarray | None = None,
    kernel: Kernel | None = None,
    seed: int = 42,
) -> np.ndarray:
    """Choose which calibration samples to spend the pilot budget on.

    Parameters
    ----------
    X : np.ndarray, shape (n_pool, d_src)
        Candidate pool, in the *transmitter's* latent space. Selection is
        driven by the TX side alone, since that is what the transmitter
        can evaluate before anything is exchanged.
    n_pilots : int
        Budget ``N``. Returned unchanged (as ``arange``) when it meets or
        exceeds the pool size.
    strategy : {'random', 'kmeans', 'fps', 'stratified', 'herding'}
        Selection rule; see :class:`src.anchors.Anchor`. ``'kmeans'`` is
        forced to its medoid variant so the result stays index-based.
    labels : np.ndarray, optional
        Per-sample class labels, required by ``'stratified'``.
    kernel : Kernel, optional
        RKHS for ``'herding'``; defaults to an RBF kernel with the
        median-distance bandwidth fitted on ``X``.
    seed : int, default=42
        Seed of the randomised strategies.

    Returns
    -------
    np.ndarray, shape (<= n_pilots,)
        Sorted row indices into ``X``. Can be shorter than requested:
        ``'stratified'`` drops per-class medoid collisions.
    """
    X = np.asarray(X)
    n_pool = X.shape[0]
    if n_pilots <= 0:
        raise ValueError(f'n_pilots must be positive, got {n_pilots}.')
    if strategy not in PILOT_STRATEGIES:
        raise ValueError(
            f'Unknown pilot strategy {strategy!r}. '
            f'Supported: {", ".join(PILOT_STRATEGIES)}.'
        )
    if n_pilots >= n_pool:
        log.debug(
            'Pilot budget %d covers the whole pool of %d; using all of it.',
            n_pilots,
            n_pool,
        )
        return np.arange(n_pool)

    fit_kwargs: dict = {'n_anchors': n_pilots}
    if strategy == 'kmeans':
        # Centroids are not observations, so they cannot be pilots.
        fit_kwargs['medoids'] = True
    if strategy in ('stratified', 'round_robin'):
        if labels is None:
            raise ValueError(
                f'Pilot strategy {strategy!r} requires per-sample labels.'
            )
        fit_kwargs['labels'] = labels
    if strategy == 'herding':
        fit_kwargs['kernel'] = kernel

    anchor = Anchor(X, strategy=strategy, seed=seed).fit(**fit_kwargs)
    indices = anchor.indices
    if indices is None:  # pragma: no cover - guarded by PILOT_STRATEGIES
        raise ValueError(
            f'Pilot strategy {strategy!r} did not return sample indices; '
            'pilots must be real observations so the receiver can look up '
            'its own representation of the same input.'
        )
    return np.sort(np.asarray(indices, dtype=int))


def select_pilot_path(
    X: np.ndarray,
    counts: Sequence[int],
    strategy: AnchorStrategy = 'random',
    labels: np.ndarray | None = None,
    kernel: Kernel | None = None,
    seed: int = 42,
) -> dict[int, np.ndarray]:
    """Pilot sets for a whole sweep of budgets at once.

    For the greedy strategies this is not just a convenience loop: kernel
    herding and farthest-point sampling never revisit a choice, so the
    budget-``k`` answer is the ``k``-point prefix of the budget-``K``
    answer. Selecting once at the largest budget and slicing is therefore
    *identical* to selecting per budget, and turns a sweep over ``B``
    budgets from ``B`` selections into one -- which matters when the
    candidate pool is a full 50k-row training split and each herding pass
    costs a pool-sized Gram row-sum.

    Parameters
    ----------
    X : np.ndarray, shape (n_pool, d_src)
        Candidate pool, in the transmitter's latent space.
    counts : Sequence[int]
        Pilot budgets to produce.
    strategy, labels, kernel, seed
        As in :func:`select_pilots`.

    Returns
    -------
    dict[int, np.ndarray]
        Budget -> sorted row indices into ``X``.
    """
    budgets = sorted({int(c) for c in counts})
    if not budgets:
        return {}
    largest = budgets[-1]

    if strategy not in _GREEDY_STRATEGIES or len(budgets) == 1:
        return {
            n: select_pilots(
                X,
                n_pilots=n,
                strategy=strategy,
                labels=labels,
                kernel=kernel,
                seed=seed,
            )
            for n in budgets
        }

    if largest >= X.shape[0]:
        everything = np.arange(X.shape[0])
        return {
            n: everything[:n] if n < everything.size else everything
            for n in budgets
        }

    fit_kwargs: dict = {'n_anchors': largest}
    if strategy == 'herding':
        fit_kwargs['kernel'] = kernel
    if strategy == 'round_robin':
        fit_kwargs['labels'] = labels
    anchor = Anchor(X, strategy=strategy, seed=seed).fit(**fit_kwargs)

    order = anchor.order
    if order is None:  # pragma: no cover - greedy strategies always set it
        raise RuntimeError(
            f'Strategy {strategy!r} was treated as greedy but exposed no '
            'selection order.'
        )
    log.debug(
        'Selected a %d-point %s path over a pool of %d; slicing %d budgets.',
        largest,
        strategy,
        X.shape[0],
        len(budgets),
    )
    return {n: np.sort(order[:n]) for n in budgets}

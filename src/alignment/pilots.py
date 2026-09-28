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

Three further designs -- ``'anti_herding'``, ``'ball'`` and
``'classes:k'`` -- are *adversarial*: nobody would deploy them, and that
is the point. A map is only identifiable from pilots that cover the
distribution it will be used on, so a design that withholds that cover is
what separates methods by what they fall back on. RKA falls back on the
rigid map ``Q z`` and pure kernel ridge falls back on the receiver's mean,
so the gap between them is a measurement rather than an anecdote.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from ..anchors import Anchor, AnchorStrategy
from ..kernels import Kernel

if TYPE_CHECKING:
    from collections.abc import Sequence

log = logging.getLogger(__name__)

__all__ = [
    'ADVERSARIAL_DESIGNS',
    'PILOT_STRATEGIES',
    'parse_design',
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
    'anti_herding',
    'ball',
)

# Designs whose whole purpose is to withhold cover of the pool. They are
# not ways to spend a budget well; they are the stress test that says
# what a method does when the calibration set cannot identify the map.
# `classes` takes a parameter, as `classes:2`.
ADVERSARIAL_DESIGNS: tuple[str, ...] = ('anti_herding', 'ball', 'classes')

# How much of the pool the 'ball' design draws its blob from: the pilots
# are the points nearest one random pool point, so the budget itself sets
# the radius until it would exceed this share.
_BALL_SHARE = 0.15

# Chunk of the pool scored at once when evaluating the pool's mean
# embedding, as in `Anchor._fit_herding`.
_HERDING_CHUNK = 512

# Strategies whose answer for budget k is the k-point prefix of their
# answer for any larger budget, so a sweep can select once and slice.
# 'fps' and 'herding' are greedy; 'round_robin' builds one ordering of the
# whole pool from the seed alone and truncates it.
#
# The adversarial designs are nested too: anti-herding is greedy like
# herding, and 'ball' and 'classes:k' rank the pool once from the seed
# and truncate.
_GREEDY_STRATEGIES: frozenset[str] = frozenset(
    {'fps', 'herding', 'round_robin', 'anti_herding', 'ball', 'classes'}
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


def parse_design(strategy: str) -> tuple[str, int | None]:
    """Split ``'classes:2'`` into its name and parameter.

    Only ``'classes'`` takes one, and it is required there: how many of
    the receiver's classes the pilots are allowed to come from is the
    whole severity of that design, so defaulting it would hide the knob
    the study is sweeping.
    """
    name, _, argument = str(strategy).partition(':')
    if name != 'classes':
        if argument:
            raise ValueError(
                f'Pilot strategy {strategy!r} takes no parameter.'
            )
        return name, None
    if not argument.isdigit() or int(argument) < 1:
        raise ValueError(
            f'Design {strategy!r} needs a positive class count, as '
            "'classes:2'."
        )
    return name, int(argument)


def _anti_herding_order(
    X: np.ndarray, n_pilots: int, kernel: Kernel | None, seed: int
) -> np.ndarray:
    """Kernel herding's exact mirror: ``argmin`` of the same score.

    Herding takes ``argmax_i mu_i - (1/t) sum_{p in P_t} k(x_i, x_p)``,
    which keeps the set both central and spread out. Minimising the same
    score instead takes the points the pool's mean embedding reaches
    least, and then prefers points *close* to what is already selected --
    so the set collapses into a corner of the support and its MMD to the
    pool grows. Same kernel, same bookkeeping, opposite sign: that is
    what makes it the principled adversary for a method fitted in this
    RKHS rather than merely a bad draw.
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    kernel = (Kernel('rbf') if kernel is None else kernel).fit(X, seed=seed)

    mu = np.empty(n)
    for start in range(0, n, _HERDING_CHUNK):
        stop = min(start + _HERDING_CHUNK, n)
        mu[start:stop] = kernel(X[start:stop], X).mean(axis=1)

    order = np.empty(n_pilots, dtype=int)
    affinity = np.zeros(n)
    available = np.ones(n, dtype=bool)
    for t in range(n_pilots):
        score = mu if t == 0 else mu - affinity / t
        pick = int(np.argmin(np.where(available, score, np.inf)))
        order[t] = pick
        available[pick] = False
        affinity += kernel(X[pick][None, :], X).ravel()
    return order


def _ball_order(X: np.ndarray, n_pilots: int, seed: int) -> np.ndarray:
    """The points nearest one random pool point: a tight blob.

    Support bias with no label structure, which is what separates it from
    ``'classes:k'``: the pilots are unrepresentative in geometry while
    the classes stay mixed, so a method cannot recover by having seen
    every label.
    """
    X = np.asarray(X, dtype=np.float64)
    rng = np.random.default_rng(seed)
    centre = X[rng.integers(X.shape[0])]
    distance = np.linalg.norm(X - centre, axis=1)
    reach = max(n_pilots, int(_BALL_SHARE * X.shape[0]))
    return np.argsort(distance)[:reach][:n_pilots]


def _classes_order(
    labels: np.ndarray | None, n_pilots: int, n_classes: int, seed: int
) -> np.ndarray:
    """Pilots drawn from only ``n_classes`` of the receiver's classes.

    Support bias aligned with the label structure the decoder reads out,
    so the transported latents of the unseen classes land wherever the
    map extrapolates them.
    """
    if labels is None:
        raise ValueError("Design 'classes:k' requires per-sample labels.")
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    present = np.unique(labels)
    if n_classes > present.size:
        raise ValueError(
            f'classes:{n_classes} asks for more classes than the pool has '
            f'({present.size}).'
        )
    keep = rng.choice(present, size=n_classes, replace=False)
    candidates = np.flatnonzero(np.isin(labels, keep))
    if candidates.size < n_pilots:
        log.warning(
            'classes:%d leaves %d candidates, short of the %d pilots asked '
            'for; the design returns what it has.',
            n_classes,
            candidates.size,
            n_pilots,
        )
        return rng.permutation(candidates)
    return rng.permutation(candidates)[:n_pilots]


def _design_order(
    X: np.ndarray,
    n_pilots: int,
    strategy: str,
    labels: np.ndarray | None,
    kernel: Kernel | None,
    seed: int,
) -> np.ndarray | None:
    """Selection order of an adversarial design, or ``None`` for the rest."""
    name, parameter = parse_design(strategy)
    if name not in ADVERSARIAL_DESIGNS:
        return None
    if name == 'anti_herding':
        return _anti_herding_order(X, n_pilots, kernel, seed)
    if name == 'ball':
        return _ball_order(X, n_pilots, seed)
    return _classes_order(labels, n_pilots, int(parameter), seed)


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
    name, _ = parse_design(strategy)
    if name not in (*PILOT_STRATEGIES, *ADVERSARIAL_DESIGNS):
        raise ValueError(
            f'Unknown pilot strategy {strategy!r}. '
            f'Supported: {", ".join(PILOT_STRATEGIES)}, classes:<k>.'
        )
    if n_pilots >= n_pool:
        log.debug(
            'Pilot budget %d covers the whole pool of %d; using all of it.',
            n_pilots,
            n_pool,
        )
        return np.arange(n_pool)

    order = _design_order(X, n_pilots, strategy, labels, kernel, seed)
    if order is not None:
        return np.sort(order)

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
    name, _ = parse_design(strategy)

    if name not in _GREEDY_STRATEGIES or len(budgets) == 1:
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

    order = _design_order(X, largest, strategy, labels, kernel, seed)
    if order is not None:
        return {n: np.sort(order[:n]) for n in budgets}

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

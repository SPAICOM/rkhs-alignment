"""Configuration-driven plumbing shared by the experiment scripts.

Every entry point under ``scripts/`` needs the same few things before any
alignment happens: work out which encoder pairs to run, load exactly those
agents with their rows paired, give every receiver the private decoder
that measures post-alignment accuracy, and build the aligner named by the
config. That logic lives here so the scripts cannot drift apart on it.

Everything takes the Hydra ``DictConfig`` directly -- these are experiment
helpers, not library code, and the config *is* the experiment.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from .decoder import Decoder, MLPDecoder, TimmDecoder
from .latent import LatentSpace, load_agents
from .reporting import chart_slug, rate_slug

if TYPE_CHECKING:
    from .alignment import Aligner

log = logging.getLogger(__name__)

# Keys an alignment preset may carry that describe how the method should
# be *run* rather than how it should be *built*, and which therefore must
# be stripped before the constructor sees them.
ALIGNER_META_KEYS: tuple[str, ...] = ('pilot_strategy', 'rate_key')

__all__ = [
    'ALIGNER_META_KEYS',
    'apply_symbol_budget',
    'available_agents',
    'build_aligner',
    'build_decoder',
    'channel_rate',
    'check_budgets_fit_pool',
    'configure_method',
    'declared_widths',
    'decoder_checkpoint',
    'describe_pairs',
    'draw_pool',
    'effective_rank',
    'load_named_agents',
    'load_pair_data',
    'method_pilot_strategy',
    'native_accuracy',
    'rank_request',
    'resolve_budgets',
    'resolve_chart_budgets',
    'resolve_charts',
    'resolve_pairs',
    'resolve_star',
    'resolve_symbols',
    'usable_rank',
    'widest_agent',
]


# ---------------------------------------------------------------------
# Alignment presets
# ---------------------------------------------------------------------


def build_aligner(method_cfg: DictConfig, **overrides) -> Aligner:
    """Instantiate an alignment preset, minus its run-time metadata.

    Every preset under ``config/hydra/alignment/`` declares the pilot
    design the method expects (:func:`method_pilot_strategy`) alongside
    its hyper-parameters, because the two belong together: a method's
    published results assume a particular calibration design, and pairing
    them in one file is what stops a run silently evaluating a method
    under someone else's. Those keys are not constructor arguments,
    though, so they are dropped here rather than at each call site.
    """
    fields = {
        key: value
        for key, value in method_cfg.items()
        if key not in ALIGNER_META_KEYS
    }
    return hydra.utils.instantiate(fields, **overrides)


def apply_symbol_budget(cfg: DictConfig) -> dict[str, int]:
    """Point every method at the same channel rate.

    Methods buy their rate with different knobs -- an anchor-based
    equalizer transmits one coefficient per anchor, a coordinate method
    transmits its truncated whitened latent -- so matching them takes
    more than setting a single field. Each preset names its own knob in
    ``rate_key``; this writes ``cfg.symbols`` into whichever one that is,
    in place, before anything is instantiated.

    Without it a comparison at "the same k" is not a comparison at the
    same rate, and the cheaper method is being asked to do the same job
    on fewer symbols.

    Returns
    -------
    dict[str, int]
        The knob each method had set, for logging.
    """
    budget = cfg.get('symbols')
    if budget is None:
        return {}

    applied: dict[str, int] = {}
    for name, method_cfg in cfg.methods.items():
        key = method_cfg.get('rate_key')
        if key is None:
            log.warning(
                'Method %s declares no `rate_key`, so `symbols=%s` cannot '
                'be applied to it; its rate is whatever its preset says.',
                name,
                budget,
            )
            continue
        method_cfg[key] = int(budget)
        applied[name] = int(budget)
    return applied


def method_pilot_strategy(
    method_cfg: DictConfig, default: str | None = None
) -> str | None:
    """The pilot design a preset asks for, or ``default`` if it names none."""
    return str(method_cfg.get('pilot_strategy') or default or '') or None


def resolve_pairs(
    cfg: DictConfig, agent_names: list[str]
) -> list[tuple[str, str]]:
    """Turn the ``pairs`` / ``receiver`` config into explicit pairs.

    Parameters
    ----------
    cfg : DictConfig
        The run configuration.
    agent_names : list[str]
        Agents the data source offers, in config order.

    Returns
    -------
    list[tuple[str, str]]
        ``(source, target)`` model names.
    """
    if cfg.get('pairs'):
        pairs = [(str(p['source']), str(p['target'])) for p in cfg.pairs]
    else:
        # Star topology: everybody transmits to a single receiver.
        receiver = cfg.get('receiver') or agent_names[0]
        senders = [name for name in agent_names if name != receiver]
        max_pairs = cfg.get('max_pairs')
        if max_pairs:
            senders = senders[: int(max_pairs)]
        pairs = [(sender, receiver) for sender in senders]

    if not pairs:
        raise ValueError(
            'No alignment pairs to run: set `pairs`, or `receiver` plus a '
            'data source with at least two agents.'
        )
    for source, target in pairs:
        if source == target:
            raise ValueError(
                f'Pair ({source} -> {target}) aligns an encoder with '
                'itself; drop it from `pairs`.'
            )
    return pairs


def load_named_agents(
    cfg: DictConfig, names: Sequence[str]
) -> dict[str, dict[str, LatentSpace]]:
    """Load exactly the named agents, with their rows paired."""
    data_cfg = OmegaConf.to_container(cfg.data, resolve=True)
    source = data_cfg.pop('source')
    data_cfg.pop('seed', None)
    data_cfg.pop('models', None)
    return load_agents(
        source, models=sorted(set(names)), seed=cfg.seed, **data_cfg
    )


def declared_widths(cfg: DictConfig) -> dict[str, int] | None:
    """Agent widths the data config states outright, if it states them.

    Only the synthetic source does: it *generates* its agents, so their
    dimensions are an input (``agent_dims``) rather than something to be
    discovered. Reading them here means a 42-agent synthetic run does not
    have to materialise all 42 latent spaces just to find the widest one.

    ``None`` means the source does not declare them and the widths have
    to be measured, which costs a load.
    """
    dims = cfg.data.get('agent_dims')
    if cfg.data.source != 'synthetic' or not dims:
        return None
    names = available_agents(cfg)
    # `make_synthetic_agents` cycles `agent_dims` to cover `n_agents`.
    return {name: int(dims[i % len(dims)]) for i, name in enumerate(names)}


def widest_agent(names: Sequence[str], widths: dict[str, int]) -> str:
    """The widest agent, ties broken by the order the config lists them.

    This is the receiver. The rule is not cosmetic: the receiver's own
    latent space is where every transported latent has to land, so a
    narrow receiver caps what the channel can carry no matter how much
    the transmitter has to say -- and it caps it for every method at
    once, which makes the comparison a measurement of the receiver
    rather than of the alignment.
    """
    return max(names, key=lambda name: (widths[name], -names.index(name)))


def resolve_star(
    cfg: DictConfig,
) -> tuple[list[tuple[str, str]], dict[str, dict[str, LatentSpace]]]:
    """The pairs to run and the agents they need, receiver first.

    Explicit ``pairs`` or an explicit ``receiver`` are obeyed as given --
    naming one is how a study says it wants a direction the rule would
    not pick. Otherwise the receiver is the *widest* agent available, and
    everything else transmits into it.

    Resolving the star costs more than reading ``data.models[0]``,
    because the widest agent cannot be known until the widths are, and
    only the synthetic source declares them. The load is paid once per
    run, against a sweep that is orders of magnitude more expensive, and
    it buys a config that keeps pointing at the right receiver when the
    model list changes underneath it.
    """
    names = available_agents(cfg)
    if cfg.get('pairs') or cfg.get('receiver'):
        pairs = resolve_pairs(cfg, names)
        return pairs, load_pair_data(cfg, pairs)

    widths = declared_widths(cfg)
    if widths is None:
        candidates = load_named_agents(cfg, names)
        widths = {
            name: int(space['train'].latent.shape[1])
            for name, space in candidates.items()
        }
    else:
        candidates = None

    receiver = widest_agent(names, widths)
    log.info(
        'Receiver resolved by width: %s (d=%d), over %s.',
        receiver,
        widths[receiver],
        ', '.join(f'{n}={widths[n]}' for n in names if n != receiver),
    )
    pairs = resolve_pairs(OmegaConf.merge(cfg, {'receiver': receiver}), names)
    if candidates is None:
        return pairs, load_pair_data(cfg, pairs)
    # Already loaded to measure the widths; `max_pairs` may have dropped
    # some of them, and the caller should not see agents it is not using.
    needed = {name for pair in pairs for name in pair}
    return pairs, {n: s for n, s in candidates.items() if n in needed}


def load_pair_data(
    cfg: DictConfig, pairs: list[tuple[str, str]]
) -> dict[str, dict[str, LatentSpace]]:
    """Load exactly the agents the resolved pairs reference."""
    return load_named_agents(cfg, [name for pair in pairs for name in pair])


def available_agents(cfg: DictConfig) -> list[str]:
    """Agent names the data source offers, without loading any latents."""
    models = cfg.data.get('models')
    if models:
        return [str(m) for m in models]
    if cfg.data.source == 'synthetic':
        n_agents = int(cfg.data.get('n_agents', 8))
        width = max(2, len(str(n_agents - 1)))
        return [f'agent_{i:0{width}d}' for i in range(n_agents)]
    raise ValueError(
        f'Cannot enumerate agents for source {cfg.data.source!r}: list them '
        'under `data.models` or give explicit `pairs`.'
    )


# ---------------------------------------------------------------------
# Sweep axes
# ---------------------------------------------------------------------
#
# The two sweeps under ``scripts/`` -- lambda against the pilot budget,
# and the field against the pilot budget -- are run over the same axes
# and have to resolve them identically, or the second cannot look up what
# the first measured. The shared definitions live here for the same
# reason the rest of this module does.


def draw_pool(
    space: LatentSpace, pool_size: int | None, seed: int
) -> np.ndarray:
    """Row indices of the candidate pool pilots are selected from.

    ``None`` means the whole training split. That is rarely what anyone
    wants: kernel herding matches the pool's mean embedding, and
    computing that embedding is *quadratic* in the pool -- 17s over a
    50k split against 0.7s over a 10k one, paid before a single aligner
    is fitted. It is also deterministic over a fixed pool, so on the full
    split it returns the same pilots for every seed and the randomised
    designs get an error bar the deterministic ones do not.

    Both studies draw from the same size, declared once in
    ``axes/default.yaml``: the lambda measured in one is applied in the
    other, and a lambda measured on pilots herded from 50k does not
    belong to a run whose pilots were herded from 10k.
    """
    n = space.n_points
    if pool_size is None or pool_size >= n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=int(pool_size), replace=False))


def check_budgets_fit_pool(
    counts: Sequence[int], pool_size: int | None, n_available: int
) -> None:
    """Refuse a pilot budget the candidate pool cannot actually fill.

    Selection returns the whole pool once the budget reaches its size, so
    an over-large budget does not fail -- it silently collapses. Two
    budgets past the limit then select *identical* pilots, score
    identically, and draw as a flat tail on the figure that looks like
    saturation and is nothing but the pool running out.

    Raising is the right response rather than warning: the run would
    otherwise spend hours producing a curve whose right-hand end is an
    artefact, and there is no reading of the result that recovers it.
    """
    limit = n_available if pool_size is None else min(pool_size, n_available)
    largest = max(int(c) for c in counts)
    if largest < limit:
        return
    raise ValueError(
        f'The largest pilot budget ({largest}) meets or exceeds the '
        f'{limit} candidates available to select it from '
        f'(pool_size={pool_size}, training split={n_available}). '
        'Selection would return the whole pool for every budget past '
        f'{limit}, so those points would be identical rather than a '
        'curve. Raise `pilots.pool_size`, or lower the top of '
        '`pilots.counts` / `pilots.multipliers`.'
    )


def usable_rank(
    pairs: list[tuple[str, str]],
    agents: dict[str, dict[str, LatentSpace]],
) -> tuple[int, int]:
    """``(d_r, max_rank)``: the receiver's width, and what a chart can keep.

    ``d_r`` is what the symbol grid is written against -- it is the
    receiver's latent dimension, the width of the space anything has to
    be delivered into. The truncation itself is bounded by the *narrower*
    of the two spaces in every pair, since a rank the transmitter cannot
    produce is not a rate anyone can send at.
    """
    targets = {t for _, t in pairs}
    d_r = min(agents[t]['train'].latent.shape[1] for t in targets)
    max_rank = min(
        min(
            agents[s]['train'].latent.shape[1],
            agents[t]['train'].latent.shape[1],
        )
        for s, t in pairs
    )
    return d_r, max_rank


def describe_pairs(
    pairs: list[tuple[str, str]],
    agents: dict[str, dict[str, LatentSpace]],
) -> str:
    """Who transmits to whom, and at what width.

    Worth printing at the top of every run. ``receiver: null`` resolves
    to the *first* entry of ``data.models``, which is a silent choice
    that decides ``d_r`` -- and therefore the rate, and therefore the
    pilot budgets, and therefore every filename. A run pointed at the
    wrong end of the pair is not distinguishable from a correct one by
    its numbers alone.
    """
    lines = []
    for source, target in pairs:
        d_s = agents[source]['train'].latent.shape[1]
        d_t = agents[target]['train'].latent.shape[1]
        lines.append(f'  TX {source} (d={d_s})  ->  RX {target} (d={d_t})')
    return '\n'.join(lines)


def resolve_symbols(cfg: DictConfig, d_r: int, max_rank: int) -> int | None:
    """The rank to compress to, or ``None`` when nothing is compressed.

    The requested rate is an explicit ``symbols``, else
    ``ceil(d_r / symbol_divisor)``. Expressing it relative to ``d_r``
    rather than as a bare number is what lets one config follow the
    encoder pair: the interesting quantity is the compression factor, and
    ``d_r`` changes underneath it.

    What the request *means* depends on whether it compresses anything.
    ``n_components`` is applied to both charts -- source and receiver
    alike -- which is what makes a truncated run rate-matched, and is the
    behaviour the compression studies are built on. But it also means a
    rank the transmitter cannot exceed anyway is not a free choice: asked
    for 384 symbols on a 384-wide transmitter and an 888-wide receiver,
    it leaves the transmitter untouched and quietly throws away 504 of
    the receiver's own directions. That is a compression of the wrong
    side of the channel.

    So a request at or above ``max_rank`` -- the narrowest space in the
    pair, and therefore the most coefficients the transmitter could
    possibly put on the channel -- is read as *no compression*, and
    returns ``None``. Every space then keeps its largest available
    dimension: the transmitter sends its full width, the receiver decodes
    into all of its own. Only a request genuinely below that is a rate,
    and it is applied globally as before.
    """
    if cfg.get('symbols') is not None:
        requested = int(cfg.symbols)
    else:
        divisor = float(cfg.symbol_divisor)
        if divisor <= 0:
            raise ValueError(
                f'symbol_divisor must be positive, got {divisor}.'
            )
        requested = max(1, math.ceil(d_r / divisor))

    if requested >= max_rank:
        log.info(
            'Requested rate %d is not below the %d dimensions the '
            'narrowest space in the run offers, so nothing is '
            'compressed: every space keeps its full width.',
            requested,
            max_rank,
        )
        return None
    return requested


def channel_rate(symbols: int | None, max_rank: int) -> int:
    """Coefficients the transmitter actually puts on the channel.

    The compression rank when there is one, otherwise the transmitter's
    full width. This is the number the pilot budget is scaled against --
    an uncompressed run still spends a definite number of coefficients,
    and ``pilots per symbol`` has to mean the same thing either way.
    """
    return max_rank if symbols is None else int(symbols)


def resolve_budgets(cfg: DictConfig, symbols: int) -> list[int]:
    """The pilot axis: absolute ``counts``, else ``ceil(symbols * m)``.

    Deriving from the rate is the default so the axis follows ``d_r``
    rather than being pinned to one encoder pair; a study that wants the
    same absolute budgets at two rates says so with ``counts``.

    Parameters
    ----------
    cfg : DictConfig
        The run configuration.
    symbols : int
        The rate as :func:`resolve_symbols` returned it -- an integer,
        never ``cfg.symbols``. ``symbols: null`` in the config means
        "derive the rate from ``d_r``", and it has already been derived
        by the time the budget axis is built; passing the raw config
        value here would multiply by ``None``.
    """
    if cfg.pilots.get('counts'):
        counts = [int(c) for c in cfg.pilots.counts]
    else:
        counts = [
            max(1, math.ceil(symbols * float(m)))
            for m in cfg.pilots.multipliers
        ]
    if not counts:
        raise ValueError(
            'The pilot axis is empty: set `pilots.counts` or '
            '`pilots.multipliers`.'
        )
    return sorted(set(counts))


def rank_request(request: Any, max_rank: int) -> int:
    """One entry of the ``ranks`` axis, as an absolute number of symbols.

    An ``int`` is that rank outright. A ``float`` in ``(0, 1]`` is the
    *fraction of the channel* to keep, resolved against ``max_rank`` --
    the most coefficients the transmitter could put on the wire -- so
    ``0.8`` is "compress by 20%" and follows the encoder pair instead of
    being pinned to one width.

    The fraction is resolved to an integer here, before it reaches any
    chart, and that is the whole point. ``LatentScaler.n_components``
    also takes a float, but there it means *explained variance*: passing
    0.8 through would keep however many directions carry 80% of each
    space's variance, which is a different number on each side of the
    channel and therefore not a rate at all. Both charts have to keep the
    same ``k`` or the run is not rate-matched.
    """
    if isinstance(request, bool):
        raise TypeError(f'A rank must be a number, got {request!r}.')
    if isinstance(request, (int, np.integer)):
        if request < 1:
            raise ValueError(
                f'An absolute rank must be >= 1, got {request}. Write a '
                'float in (0, 1] for a fraction of the channel.'
            )
        return int(request)
    fraction = float(request)
    if not 0.0 < fraction <= 1.0:
        raise ValueError(
            f'A fractional rank must lie in (0, 1], got {fraction}. Write '
            'an integer for an absolute number of symbols.'
        )
    return max(1, math.ceil(fraction * max_rank))


def resolve_chart_budgets(
    cfg: DictConfig,
    charts: Sequence[DictConfig],
    method_cfg: DictConfig,
    symbols: int | None,
    max_rank: int,
) -> list[tuple[DictConfig, int | None, int, list[int]]]:
    """``[(chart, rank, rate, counts)]`` -- the budget axis, per chart.

    ``pilots.multipliers`` are pilots *per symbol*, so the axis has to be
    scaled against what the chart in hand actually puts on the channel:
    a chart keeping ``k`` directions is measured at ``ceil(k * m)``, not
    at budgets derived from whatever single rate the run resolved. With a
    ``ranks`` axis the two differ for every rank but one, and scaling them
    all against the same rate would compare a 48-symbol chart and a
    384-symbol chart at the same absolute ``N`` -- eight pilots per symbol
    against one.

    Absolute ``pilots.counts`` still win and still apply to every chart:
    an axis stated outright is stated for the whole run.

    The rank is read through ``method_cfg``'s own ``rate_key``. Every
    method in a run is rate-matched through its own knob, so any of them
    reports the same rank; the caller passes whichever it already has.
    """
    plan = []
    for chart in charts:
        merged = configure_method(method_cfg, chart, symbols)
        rank = effective_rank(merged, method_cfg)
        rate = channel_rate(rank, max_rank)
        plan.append((chart, rank, rate, resolve_budgets(cfg, rate)))
    return plan


def resolve_charts(cfg: DictConfig, max_rank: int) -> list[DictConfig]:
    """The chart axis, crossed with the compression ranks if there are any.

    ``ranks`` is the second axis of the preprocessing grid: a study that
    wants one lambda per (chart, rank) states the ranks once and every
    chart is measured at each of them. It is expressed here rather than by
    writing out one ``charts`` entry per rank because the two are
    independent questions -- which factorisation, and how much of it -- and
    crossing them by hand is what makes a five-rank sweep a fifteen-line
    config that cannot be diffed against a three-rank one.

    Each expanded chart carries the rank in ``n_components`` and a
    ``name``, so :func:`~src.reporting.chart_slug` can tell them apart:
    the slug does not print ``n_components`` (``rate_slug`` already
    reports the effective rank), and without a name every rank of one
    chart would land in the same figure directory.

    The ``resolve_symbols`` rule applies per rank: a request at or above
    ``max_rank`` cannot compress the transmitter, so it is read as *no
    compression* -- ``n_components`` unset, tagged ``kfull`` -- rather
    than silently discarding the receiver's surplus directions. Several
    such requests collapse to one chart rather than fitting the same cell
    repeatedly.

    ``ranks: null`` (the default) returns ``cfg.charts`` untouched, so a
    config that never mentions the axis behaves exactly as before.
    """
    charts = list(cfg.charts)
    ranks = cfg.get('ranks')
    if not ranks:
        return charts

    out: list[DictConfig] = []
    seen: set[tuple[str, int | None]] = set()
    for chart in charts:
        if chart.get('n_components') is not None:
            raise ValueError(
                f'Chart {chart_slug(chart)!r} pins n_components='
                f'{chart["n_components"]} while `ranks` is also set, so the '
                'rank it is measured at is ambiguous. Drop one of them: '
                '`ranks` for a grid, the chart key for a one-off.'
            )
        base = chart_slug(chart)
        for request in ranks:
            symbols = rank_request(request, max_rank)
            rank = None if symbols >= max_rank else symbols
            if (base, rank) in seen:
                log.info(
                    'Rank %s resolves to %d symbols, which is not below '
                    'max_rank=%d, so it is the same uncompressed chart as '
                    'an earlier request; skipping it.',
                    request,
                    symbols,
                    max_rank,
                )
                continue
            seen.add((base, rank))
            # Built from a plain container rather than merged: a
            # composed config is in struct mode, and neither
            # `n_components` nor `name` need be present on the entry the
            # user wrote.
            fields = dict(OmegaConf.to_container(chart, resolve=True))
            fields['n_components'] = rank
            fields['name'] = f'{base}-{rate_slug(rank)}'
            out.append(OmegaConf.create(fields))
    return out


def chart_fields(chart: DictConfig) -> dict[str, Any]:
    """The preprocessing keys one chart entry writes into every method."""
    return {key: value for key, value in chart.items() if key != 'name'}


def configure_method(
    method_cfg: DictConfig, chart: DictConfig, symbols: int | None
) -> DictConfig:
    """One method, at a given rate and on a given preprocessing chart.

    The rate goes in first, through whichever knob the preset names in
    ``rate_key`` (writing ``n_components`` blindly would silently miss any
    method whose rate lives somewhere else), and the chart is merged on
    top -- so a chart that sets ``n_components`` deliberately overrides
    the rate for itself, which is how one run covers several ranks.

    Returns a *detached* copy: the caller is building one aligner per
    cell of a grid and must not mutate the shared preset.
    """
    fields: dict[str, Any] = {}
    key = method_cfg.get('rate_key')
    if key is None:
        log.warning(
            'Method preset declares no `rate_key`, so symbols=%s cannot be '
            'applied; its rate is whatever the preset says.',
            symbols,
        )
    else:
        # `None` is written out rather than left alone: "no compression"
        # has to mean every space keeps its largest dimension, not
        # whatever rank the preset happened to ship with.
        fields[key] = None if symbols is None else int(symbols)
    fields.update(chart_fields(chart))
    return OmegaConf.merge(method_cfg, fields)


def effective_rank(merged: DictConfig, method_cfg: DictConfig) -> int | None:
    """The rank a configured method actually keeps, for the filename.

    ``None`` means nothing was truncated -- no ``rate_key`` to write into,
    or a chart that set the rank back to null. Naming such a run ``k384``
    would claim a truncation that did not happen.
    """
    key = method_cfg.get('rate_key')
    if key is None:
        return None
    value = merged.get(key)
    return None if value is None else int(value)


def native_accuracy(
    pairs: list[tuple[str, str]],
    agents: dict[str, dict[str, LatentSpace]],
    decoders: dict[str, Decoder | None],
) -> dict[str, float]:
    """The receivers' accuracy on their own latents -- the ceiling.

    Every alignment in these studies is working toward this number and
    none can exceed it, so it is drawn on the figures as a rule. Averaged
    over the pairs, which is exact for the single-receiver star topology
    and the right summary if it ever grows a second one.
    """
    scores = [
        decoders[target].score(
            agents[target]['test'].latent, agents[target]['test'].labels
        )
        for _, target in pairs
        if decoders.get(target) is not None
    ]
    return {'accuracy': float(np.mean(scores))} if scores else {}


# ---------------------------------------------------------------------
# Receiver decoders
# ---------------------------------------------------------------------


def decoder_checkpoint(cfg: DictConfig, model: str) -> Path:
    """Path of a cached decoder checkpoint for one receiver.

    Keyed on the decoder's *architecture* as well as the receiver and the
    seed. ``MLPDecoder.load`` validates only ``input_dim``, so a cache
    keyed on the seed alone would hand back a 512-unit head to a run that
    asked for 1024 -- silently, and the whole study would then be scored
    by an instrument it did not configure. Changing any hyper-parameter
    now names a different file instead.
    """
    slug = model.replace('.', '_').replace('/', '_')
    dataset = cfg.data.get('dataset', cfg.data.source)
    spec = json.dumps(
        OmegaConf.to_container(cfg.decoder.mlp, resolve=True), sort_keys=True
    )
    digest = hashlib.sha1(spec.encode()).hexdigest()[:8]
    return (
        Path(cfg.decoder.checkpoint_dir)
        / dataset
        / slug
        / f'seed{cfg.seed}-{digest}.pt'
    )


def build_decoder(cfg: DictConfig, space: LatentSpace) -> Decoder:
    """Fit (or restore) the receiver's private decoder on its own latents.

    Parameters
    ----------
    cfg : DictConfig
        The run configuration.
    space : LatentSpace
        The receiver's *train* split; the decoder never sees any
        transported latent.

    Returns
    -------
    Decoder
    """
    if space.labels is None:
        raise ValueError(
            f'Agent {space.model_name!r} has no labels, so no decoder can be '
            'fitted. Set decoder.enabled=false to skip the downstream metric.'
        )

    if cfg.decoder.kind == 'linear':
        decoder = TimmDecoder(
            model_name=space.model_name,
            input_dim=space.dim,
            n_classes=int(cfg.decoder.n_classes),
            l2=float(cfg.decoder.l2),
        )
        return decoder.fit(space.latent, space.labels)

    if cfg.decoder.kind != 'mlp':
        raise ValueError(f'Unknown decoder kind {cfg.decoder.kind!r}.')

    path = decoder_checkpoint(cfg, space.model_name)
    if cfg.decoder.cache and path.exists():
        decoder = MLPDecoder.load(path)
        # Both, not just the width. The receiver's decoder is the
        # instrument every number in a study is read off, and a head
        # belonging to another encoder is not detectable downstream --
        # it just reports a quietly wrong ceiling. Width alone cannot
        # tell these apart: `vit_base_patch16` and `vit_base_patch32`
        # are both d=768, `vit_small_patch16` and `vit_small_patch32`
        # both d=384.
        if (
            decoder.input_dim == space.dim
            and decoder.model_name == space.model_name
        ):
            log.info(
                'Restored decoder for %s from %s.', space.model_name, path
            )
            return decoder
        log.warning(
            'Checkpoint %s holds a head for %r at dim=%s, but this run '
            'asked for %r at dim=%d; refitting.',
            path,
            decoder.model_name,
            decoder.input_dim,
            space.model_name,
            space.dim,
        )

    log.info('Fitting an MLP decoder for %s.', space.model_name)
    decoder = MLPDecoder(
        model_name=space.model_name,
        input_dim=space.dim,
        seed=cfg.seed,
        **OmegaConf.to_container(cfg.decoder.mlp, resolve=True),
    )
    decoder.fit(space.latent, space.labels)
    if cfg.decoder.cache:
        decoder.save(path)
        log.info('Saved decoder checkpoint to %s.', path)
    return decoder

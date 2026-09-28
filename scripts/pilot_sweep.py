"""Figure (ii): the field against the pilot budget.

One figure, one panel per metric, plotting RKA, Procrustes and the two
neural baselines against the number of semantic pilots ``N``. Colour is
the method, line style is the pilot design, and the band is +/- 1 sd over
the seeds.

Every method is fitted on every design in ``pilots.strategies``. ``drop``
then chooses what the figure shows: a bare ``method`` leaves a method out
altogether, ``method:design`` leaves one curve out and keeps the method's
others -- the way to put two designs on the kernel methods and one on the
baselines without the eight curves the full grid would draw.

Nothing here is left to an inner selection:

- the rate is pinned (``symbols``), written into whichever knob each
  preset declares, so every method spends the same airtime;
- RKA's ``lambda`` is not selected per fit but *read back* per budget from
  the CSVs ``lambda_sweep.py`` wrote, at the same chart and the same rate.
  Choosing it on a held-out slice of the pilots is itself
  sample-inefficient, and at the small budgets that would show up as a
  property of RKA rather than of the selection rule.

The lookup matches on the CSVs' *columns*, never on their filenames, so
renaming an output cannot silently change which lambda is used. It also
means ``lambda_sweep.py`` has to have run first on the same axes -- which
is what ``config/hydra/axes/default.yaml`` exists to guarantee.

Because the schedule is imported, the fit has to match the run it was
measured in. Both sweeps therefore honour ``use_local_context``, which is
on by default: each device estimates its chart from its whole local
split, and the lambda measured by ``lambda_sweep.py`` belongs to the same
chart this run fits on.

Every seed writes its own records CSV as it finishes and ``resume=true``
skips the ones already on disk, so a run that dies costs one seed rather
than the grid, and the figure is drawn from whatever seeds are there.

Examples
--------
    just pilot-sweep
    uv run scripts/pilot_sweep.py 'seeds=[0, 1, 2]'
    uv run scripts/pilot_sweep.py 'charts=[{preprocess: pca}]'
    uv run scripts/pilot_sweep.py data=semasia_mnist
    uv run scripts/pilot_sweep.py plot_only=true 'drop=[direct_mlp]'
    uv run scripts/pilot_sweep.py \\
        'drop=[direct_mlp:stratified, residual_mlp:stratified]'
"""

from __future__ import annotations

import csv
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

import wandb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.alignment import Aligner, alignment_metrics, select_pilot_path
from src.experiment import (
    build_aligner,
    build_decoder,
    channel_rate,
    check_budgets_fit_pool,
    configure_method,
    describe_pairs,
    draw_pool,
    native_accuracy,
    resolve_chart_budgets,
    resolve_charts,
    resolve_star,
    resolve_symbols,
    usable_rank,
)
from src.plotting import plot_pilot_efficiency, use_project_style
from src.reporting import (
    chart_slug,
    figure_dir,
    pairs_slug,
    pilot_stem,
    rate_slug,
    result_dir,
)

if TYPE_CHECKING:
    from src.decoder import Decoder
    from src.latent import LatentSpace

log = logging.getLogger(__name__)

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / 'config' / 'hydra')
ROOT = Path(__file__).resolve().parents[1]

_TEXT: frozenset[str] = frozenset(
    {'method', 'strategy', 'chart', 'dataset', 'pairs', 'source', 'target'}
)


# ---------------------------------------------------------------------
# The lambda schedule
# ---------------------------------------------------------------------


def lambda_schedule(
    directory: Path,
    pairs_tag: str,
    chart_tag: str,
    symbols: int | None,
    metric: str,
    method: str | None = None,
    bandwidth: float | None = None,
    pattern: str = 'lambda_*.csv',
) -> dict[int, float]:
    """Budget -> the ``lambda`` that maximised ``metric`` in the sweep.

    Reads every cell CSV a lambda sweep left for this dataset and keeps
    the rows describing *this* run: same encoder pairs, same
    preprocessing chart, same rate. Nothing is refitted -- the sweep
    already paid for these numbers.

    Two more filters apply only to CSVs that carry the column. ``method``
    matches the ``method`` column the ablation sweep writes, so each
    variant reads its own lambda. ``bandwidth`` matches
    ``bandwidth_scale``: a bandwidth sweep's CSV holds curves at every
    scale, and the best lambda there may belong to a bandwidth the fit
    applying it will never use.
    """
    key = f'{metric}_mean'
    best: dict[int, tuple[float, float]] = {}
    scanned = 0

    for path in sorted(directory.glob(pattern)):
        with path.open() as handle:
            rows = list(csv.DictReader(handle))
        if not rows or key not in rows[0]:
            log.warning('%s carries no %s column; skipping.', path.name, key)
            continue
        scanned += 1
        for row in rows:
            rank = int(row['symbols']) if row.get('symbols') else None
            scale = row.get('bandwidth_scale')
            if (
                row.get('pairs') != pairs_tag
                or row.get('chart') != chart_tag
                or rank != symbols
                or (method is not None and row.get('method', method) != method)
                or (
                    bandwidth is not None
                    and scale not in (None, '')
                    and float(scale) != bandwidth
                )
            ):
                continue
            n_pilots = int(row['n_pilots'])
            score = float(row[key])
            if n_pilots not in best or score > best[n_pilots][0]:
                best[n_pilots] = (score, float(row['lam']))

    log.info(
        'Scanned %d lambda-sweep CSVs under %s; %d budgets match '
        'pairs=%s chart=%s %s%s.',
        scanned,
        directory,
        len(best),
        pairs_tag,
        chart_tag,
        rate_slug(symbols),
        '' if method is None else f' method={method}',
    )
    return {n: lam for n, (_, lam) in best.items()}


def lambda_schedules(
    cfg: DictConfig,
    directory: Path,
    pairs_tag: str,
    chart_tag: str,
    symbols: int | None,
) -> dict[str, dict[int, float]]:
    """One schedule per method in ``lam_schedule.methods``.

    Each method reads the rows at its own ``bandwidth_scale``, and its own
    ``method`` rows where the CSVs name one. Stops the run when a method
    has nothing measured, since fitting it unscheduled would quietly fall
    back on its held-out selection.
    """
    schedules: dict[str, dict[int, float]] = {}
    for name in cfg.lam_schedule.methods or ():
        name = str(name)
        scale = cfg.methods[name].get('bandwidth_scale')
        schedule = lambda_schedule(
            directory,
            pairs_tag,
            chart_tag,
            symbols,
            str(cfg.lam_schedule.metric),
            method=name,
            bandwidth=None if scale is None else float(scale),
            pattern=str(cfg.lam_schedule.get('pattern') or 'lambda_*.csv'),
        )
        if not schedule:
            raise SystemExit(
                f'No lambda measured for {name} under {directory}/ for '
                f'chart={chart_tag} at {rate_slug(symbols)}. Run its lambda '
                'sweep first, or set lam_schedule.enabled=false.'
            )
        log.info(
            'Lambda schedule for %s (%s), %d budgets: %s',
            name,
            cfg.lam_schedule.metric,
            len(schedule),
            ', '.join(f'{n}:{lam:.4g}' for n, lam in sorted(schedule.items())),
        )
        schedules[name] = schedule
    return schedules


def lookup_lambda(
    schedule: dict[int, float], n_pilots: int, nearest: bool
) -> float:
    """The scheduled ``lambda`` for one budget.

    An exact match is the intended case -- both studies read their budget
    axis out of the same config. For anything else the nearest measured
    budget *in log space* is the honest fallback (the axis is geometric),
    and it is reported as such rather than silently applied.
    """
    if n_pilots in schedule:
        return schedule[n_pilots]
    if not nearest:
        raise KeyError(
            f'No lambda measured at N={n_pilots}. Measured budgets: '
            f'{sorted(schedule)}. Run the lambda sweep at this budget, or '
            'set lam_schedule.nearest=true.'
        )
    closest = min(
        schedule, key=lambda n: abs(math.log(n) - math.log(n_pilots))
    )
    log.warning(
        'No lambda measured at N=%d; using the one from N=%d (%.4g).',
        n_pilots,
        closest,
        schedule[closest],
    )
    return schedule[closest]


def pinned(method_cfg: DictConfig, lam: float | None) -> DictConfig:
    """Pin one method's ``lambda``, disabling its own selection.

    ``lam_grid=None`` matters as much as ``lam``: left set, the aligner
    would hold out a slice of the pilots and re-select, which is the
    behaviour the schedule exists to replace.
    """
    if lam is None:
        return method_cfg
    return OmegaConf.merge(method_cfg, {'lam': float(lam), 'lam_grid': None})


# ---------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------


def evaluate(
    cfg: DictConfig,
    method_cfg: DictConfig,
    pilots: np.ndarray,
    agents: dict[str, dict[str, LatentSpace]],
    pair: tuple[str, str],
    decoder: Decoder | None,
    seed: int,
) -> dict[str, float]:
    """Fit one method on one pilot set and score the whole test split."""
    source, target = pair
    src_train, src_test = agents[source]['train'], agents[source]['test']
    tgt_train, tgt_test = agents[target]['train'], agents[target]['test']

    context: dict[str, np.ndarray] = {}
    if cfg.use_local_context:
        context = {
            'src_context': src_train.latent,
            'tgt_context': tgt_train.latent,
        }

    labels = src_train.labels
    aligner: Aligner = build_aligner(method_cfg, seed=seed)
    aligner.fit(
        src_train.latent[pilots],
        tgt_train.latent[pilots],
        labels=None if labels is None else labels[pilots],
        **context,
    )
    return alignment_metrics(
        aligner.transform(src_test.latent),
        tgt_test.latent,
        decoder=decoder,
        labels=tgt_test.labels,
        ks=tuple(cfg.eval.topk),
    )


def run_seed(
    cfg: DictConfig,
    pairs: list[tuple[str, str]],
    agents: dict[str, dict[str, LatentSpace]],
    decoders: dict[str, Decoder | None],
    chart: DictConfig,
    symbols: int,
    counts: list[int],
    schedules: dict[str, dict[int, float]],
    seed: int,
    context: dict[str, Any],
) -> list[dict[str, Any]]:
    """Every ``(design, budget, method, pair)`` cell, for one realisation.

    Pilot selection sits outside the method loop, so at a given budget
    every method is fitted on exactly the same samples -- which is what
    makes them comparable at all. ``schedules`` maps a method to its
    lambda per budget; a method without one keeps its preset.
    """
    metrics = list(cfg.eval.metrics)
    configured = {
        name: configure_method(method_cfg, chart, symbols)
        for name, method_cfg in cfg.methods.items()
    }
    records: list[dict[str, Any]] = []

    for strategy in cfg.pilots.strategies:
        strategy = str(strategy)
        for source, target in pairs:
            src_train = agents[source]['train']
            pool = draw_pool(src_train, cfg.pilots.pool_size, seed)
            labels = src_train.labels
            log.info(
                'seed=%d %s: selecting up to %d pilots from a pool of %d.',
                seed,
                strategy,
                max(counts),
                pool.size,
            )
            # Every budget from one selection: herding is greedy, so the
            # budget-k answer is the k-prefix of the largest.
            paths = select_pilot_path(
                src_train.latent[pool],
                counts=counts,
                strategy=strategy,
                labels=None if labels is None else labels[pool],
                seed=seed,
            )

            for n_pilots in counts:
                pilots = pool[paths[n_pilots]]
                for name, method_cfg in configured.items():
                    applied = (
                        lookup_lambda(
                            schedules[name],
                            n_pilots,
                            bool(cfg.lam_schedule.nearest),
                        )
                        if schedules.get(name)
                        else None
                    )
                    scores = evaluate(
                        cfg,
                        pinned(method_cfg, applied),
                        pilots,
                        agents,
                        (source, target),
                        decoders.get(target),
                        seed,
                    )
                    records.append(
                        context
                        | {
                            'method': name,
                            'strategy': strategy,
                            'n_pilots': n_pilots,
                            'n_pilots_used': int(pilots.size),
                            'seed': seed,
                            'pool_size': int(pool.size),
                            'source': source,
                            'target': target,
                            # None, not '', for the methods that take no
                            # schedule: the CSV writes both as an empty
                            # field, but wandb rejects a mixed column.
                            'lam': applied,
                            **{m: scores[m] for m in metrics},
                        }
                    )
                log.info(
                    'seed=%d %s N=%d: %s',
                    seed,
                    strategy,
                    n_pilots,
                    '  '.join(
                        f'{r["method"]}={r[metrics[0]]:.4f}'
                        for r in records[-len(configured) :]
                    ),
                )
    return records


# ---------------------------------------------------------------------
# Persistence and reduction
# ---------------------------------------------------------------------


def write_records(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write one seed's records, via a temporary so a crash cannot cut it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.csv.partial')
    with tmp.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def read_records(paths: list[Path]) -> list[dict[str, Any]]:
    """Every cell from every seed file, with the numerics parsed."""
    records: list[dict[str, Any]] = []
    for path in paths:
        with path.open() as handle:
            for row in csv.DictReader(handle):
                parsed: dict[str, Any] = {}
                for key, value in row.items():
                    if key in _TEXT:
                        parsed[key] = value
                        continue
                    try:
                        parsed[key] = float(value)
                    except (TypeError, ValueError):
                        parsed[key] = value or None
                for key in ('n_pilots', 'seed', 'symbols', 'n_pilots_used'):
                    if isinstance(parsed.get(key), float):
                        parsed[key] = int(parsed[key])
                records.append(parsed)
    return records


def parse_drop(
    entries: Any, methods: list[str], strategies: list[str]
) -> tuple[set[str], set[tuple[str, str]]]:
    """Split ``drop`` into whole methods and single ``method:design`` cells.

    A bare name takes a method off the figure entirely; ``method:design``
    takes one curve off and leaves the method's other designs standing --
    which is what a figure that compares two pilot designs for the kernel
    methods against one design for the neural baselines is asking for.

    Both are read after the fitting, so the records keep every cell and
    the figure is a `plot_only=true` redraw away from including it again.

    Names are checked against the run: a typo would otherwise drop
    nothing, and the figure it wrote would be the one it was meant to
    replace.
    """
    known_methods, known_strategies = set(methods), set(strategies)
    dropped: set[str] = set()
    cells: set[tuple[str, str]] = set()

    for entry in entries or ():
        text = str(entry).strip()
        method, sep, strategy = (part.strip() for part in text.partition(':'))
        if method not in known_methods:
            raise SystemExit(
                f'drop={text!r} names no method in this run. Methods: '
                f'{sorted(known_methods)}.'
            )
        if not sep:
            dropped.add(method)
        elif strategy not in known_strategies:
            raise SystemExit(
                f'drop={text!r} names no pilot design in this run. Designs: '
                f'{sorted(known_strategies)}.'
            )
        else:
            cells.add((method, strategy))
    return dropped, cells


def drop_slug(dropped: set[str], cells: set[tuple[str, str]]) -> str:
    """The ``no-...`` marker naming a figure with curves left out."""
    return '-'.join(sorted(dropped) + sorted(f'{m}.{s}' for m, s in cells))


def aggregate(
    records: list[dict[str, Any]], metrics: list[str], order: list[str]
) -> list[dict[str, Any]]:
    """Average over pairs within a seed, then mean +/- sd over seeds.

    That order is deliberate: the band is then the spread of the
    *experiment* -- which pool was drawn, which pilots the design picked
    out of it -- rather than the spread of the pair mix, which is a
    property of the encoder zoo and would not shrink with more seeds.
    """
    per_seed: dict[tuple, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in records:
        key = (row['method'], row['strategy'], row['n_pilots'], row['seed'])
        for metric in metrics:
            if row.get(metric) is not None:
                per_seed[key][metric].append(float(row[metric]))

    by_cell: dict[tuple, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for (method, strategy, n_pilots, _), values in per_seed.items():
        for metric, samples in values.items():
            by_cell[(method, strategy, n_pilots)][metric].append(
                float(np.mean(samples))
            )

    summary: list[dict[str, Any]] = []
    for (method, strategy, n_pilots), values in by_cell.items():
        row: dict[str, Any] = {
            'method': method,
            'strategy': strategy,
            'n_pilots': n_pilots,
            'n_seeds': len(next(iter(values.values()))),
        }
        for metric in metrics:
            if metric not in values:
                continue
            row[f'{metric}_mean'] = float(np.mean(values[metric]))
            row[f'{metric}_std'] = float(np.std(values[metric]))
        summary.append(row)

    # Colour is assigned by first-seen order against a four-slot palette,
    # so the config's method order is what keeps each curve on the hue it
    # holds in every other figure.
    rank = {name: i for i, name in enumerate(order)}
    return sorted(
        summary,
        key=lambda r: (
            rank.get(r['method'], len(order)),
            r['method'],
            r['strategy'],
            r['n_pilots'],
        ),
    )


def native_ceiling(records: list[dict[str, Any]]) -> dict[str, float] | None:
    """The accuracy ceiling to draw, recovered from the records.

    The sweep carries each receiver's own accuracy on every row it
    writes, so the ceiling costs nothing here -- the alternative is
    refitting a decoder to reproduce a number the run already had.
    """
    values = [
        float(r['native_accuracy'])
        for r in records
        if isinstance(r.get('native_accuracy'), float)
    ]
    return {'accuracy': float(np.mean(values))} if values else None


def summary_table(summary: list[dict[str, Any]], metrics: list[str]) -> str:
    """Fixed-width view of the curves (also the accessibility fallback)."""
    columns = ['method', 'strategy', 'n_pilots', 'n_seeds'] + [
        f'{m}_{stat}'
        for m in metrics
        for stat in ('mean', 'std')
        if any(f'{m}_{stat}' in r for r in summary)
    ]
    body = [
        [
            f'{r[c]:.4f}'
            if isinstance(r.get(c), float)
            else str(r.get(c, '-'))
            for c in columns
        ]
        for r in summary
    ]
    widths = [
        max(len(c), *(len(line[i]) for line in body))
        for i, c in enumerate(columns)
    ]
    head = '  '.join(c.ljust(w) for c, w in zip(columns, widths))
    rule = '  '.join('-' * w for w in widths)
    return '\n'.join(
        [head, rule]
        + [
            '  '.join(v.ljust(w) for v, w in zip(line, widths))
            for line in body
        ]
    )


@hydra.main(
    version_base=None, config_path=CONFIG_DIR, config_name='pilot_sweep'
)
def main(cfg: DictConfig) -> None:
    """Run the comparison over every chart and seed; draw one figure each."""
    logging.getLogger('src').setLevel(logging.INFO)
    log.info('Configuration:\n%s', OmegaConf.to_yaml(cfg))

    metrics = list(cfg.eval.metrics)
    order = list(cfg.methods)
    dropped, dropped_cells = parse_drop(
        cfg.drop, order, [str(s) for s in cfg.pilots.strategies]
    )
    seeds = [int(s) for s in cfg.seeds]

    dataset = str(cfg.data.get('dataset', cfg.data.source))
    pairs, agents = resolve_star(cfg)
    log.info(
        'Alignment direction (receiver %s):\n%s',
        'pinned' if cfg.get('receiver') else 'resolved by width',
        describe_pairs(pairs, agents),
    )

    decoders: dict[str, Decoder | None] = {}
    if cfg.decoder.enabled:
        for target in {t for _, t in pairs}:
            decoders[target] = build_decoder(cfg, agents[target]['train'])
    reference = native_accuracy(pairs, agents, decoders)

    # Budgets below the source width are rank-deficient by design; that
    # regime is the subject of the study, not a misconfiguration.
    for name in ('src.alignment.base', 'src.alignment.preprocessing'):
        logging.getLogger(name).setLevel(logging.ERROR)

    d_r, max_rank = usable_rank(pairs, agents)
    symbols = resolve_symbols(cfg, d_r, max_rank)
    rate = channel_rate(symbols, max_rank)
    # The chart axis crossed with `ranks`, when the config gives any;
    # both studies expand it the same way, and each chart carries the
    # budget axis scaled to its own rank -- or the lambda measured at one
    # (chart, rank, budget) could not be looked up at the other.
    charts = resolve_charts(cfg, max_rank)
    plan = resolve_chart_budgets(
        cfg, charts, next(iter(cfg.methods.values())), symbols, max_rank
    )
    counts = sorted({n for *_, budgets in plan for n in budgets})
    strategies = [str(s) for s in cfg.pilots.strategies]
    check_budgets_fit_pool(
        counts,
        cfg.pilots.pool_size,
        min(agents[src]['train'].n_points for src, _ in pairs),
    )
    log.info(
        'd_r=%d, max_rank=%d -> %s; %d charts x %d seeds x %d designs x '
        '%d methods, %d (chart, budget) cells.',
        d_r,
        max_rank,
        f'compressing to {symbols} symbols'
        if symbols is not None
        else f'no compression, {rate} symbols at full width',
        len(charts),
        len(seeds),
        len(strategies),
        len(cfg.methods),
        sum(len(b) for *_, b in plan),
    )
    for chart, rank, chart_rate, budgets in plan:
        log.info(
            '  chart=%-16s %-6s rate=%-5d budgets %s',
            chart_slug(chart),
            rate_slug(rank),
            chart_rate,
            ', '.join(str(n) for n in budgets),
        )

    results = result_dir(cfg.output.results, str(cfg.output.study), dataset)
    lam_results = result_dir(
        cfg.output.results, str(cfg.lam_schedule.study), dataset
    )
    use_project_style(ROOT / 'config' / 'plotting' / 'plt.mplstyle')

    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.name
        or f'pilots-{dataset}-{pairs_slug(pairs)}-{rate_slug(symbols)}',
        group=cfg.wandb.group,
        job_type='pilot-sweep',
        tags=list(cfg.wandb.tags),
        mode=cfg.wandb.mode,
        config=OmegaConf.to_container(cfg, resolve=True)
        | {
            'resolved_symbols': symbols,
            'resolved_channel_rate': rate,
            'resolved_counts': counts,
            'resolved_seeds': seeds,
            'd_r': d_r,
        },
    )

    written: list[Path] = []
    try:
        for chart, rank, _chart_rate, budgets in plan:
            tag = chart_slug(chart)
            stem = pilot_stem(dataset, pairs, chart, rank, budgets, strategies)

            schedules: dict[str, dict[int, float]] = {}
            if cfg.lam_schedule.enabled and not cfg.plot_only:
                schedules = lambda_schedules(
                    cfg, lam_results, pairs_slug(pairs), tag, rank
                )

            context = {
                'dataset': dataset,
                'pairs': pairs_slug(pairs),
                'chart': tag,
                'symbols': rank,
                'native_accuracy': reference.get('accuracy'),
            }

            for seed in seeds:
                path = results / f'{stem}_seed{seed}.csv'
                if path.exists() and (cfg.resume or cfg.plot_only):
                    log.info(
                        'Reusing the seed already on disk at %s '
                        '(resume=true); pass resume=false to refit it.',
                        path,
                    )
                    continue
                if cfg.plot_only:
                    log.warning(
                        'plot_only=true and %s is missing; seed %d will be '
                        'absent from the bands.',
                        path.name,
                        seed,
                    )
                    continue
                log.info(
                    '=== chart=%s  %s  seed=%d ===',
                    tag,
                    rate_slug(rank),
                    seed,
                )
                write_records(
                    path,
                    run_seed(
                        cfg,
                        pairs,
                        agents,
                        decoders,
                        chart,
                        symbols,
                        budgets,
                        schedules,
                        seed,
                        context,
                    ),
                )

            paths = sorted(results.glob(f'{stem}_seed*.csv'))
            if not paths:
                log.warning('No records for chart=%s; nothing to draw.', tag)
                continue

            records = [
                r
                for r in read_records(paths)
                if r['method'] not in dropped
                and (r['method'], r['strategy']) not in dropped_cells
            ]
            if not records:
                log.warning(
                    'Every record for chart=%s was dropped; nothing to draw.',
                    tag,
                )
                continue
            summary = aggregate(records, metrics, order)

            # A figure with a curve left out is a different figure, so
            # it gets a different name: `drop` is a presentation choice
            # made after the fitting, and it must not overwrite the
            # canonical one drawn from the same records.
            marker = drop_slug(dropped, dropped_cells)
            shown = f'{stem}_no-{marker}' if marker else stem

            reduced = results / f'{shown}.csv'
            with reduced.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
                writer.writeheader()
                writer.writerows(summary)

            figures = figure_dir(
                cfg.output.figures, str(cfg.output.study), dataset, tag
            )
            title = cfg.output.get('title') or (
                f'Semantic pilots — {dataset}, '
                f'{chart.get("preprocess", "whiten")} chart, '
                + ('full width' if rank is None else f'{rank} symbols')
            )
            written += plot_pilot_efficiency(
                summary,
                metrics=metrics,
                out_path=figures / shown,
                reference=native_ceiling(records),
                title=title,
                formats=tuple(cfg.output.formats),
            )

            found = ', '.join(
                str(s) for s in sorted({r['seed'] for r in records})
            )
            print(
                f'\n{tag} — {len(paths)} seeds ({found}), '
                f'{len(records)} cells:\n'
            )
            print(summary_table(summary, metrics))

            # A seed that died mid-grid leaves a partial file, so the
            # cells are not necessarily balanced -- say so rather than
            # letting an error bar over two seeds sit next to one over
            # five.
            coverage = {r['n_seeds'] for r in summary}
            if len(coverage) > 1:
                print(
                    f'\nWarning: uneven seed coverage across cells '
                    f'({sorted(coverage)} seeds). Re-run the missing seeds '
                    'before reading the bands.'
                )

            table = wandb.Table(columns=list(summary[0]))
            for row in summary:
                table.add_data(*row.values())
            run.log({f'curves/{tag}': table})

        for path in written:
            if path.suffix == '.png':
                run.log({f'figure/{path.stem}': wandb.Image(str(path))})
        run.summary.update(
            {f'native/{k}': v for k, v in reference.items()}
            | {
                'symbols': symbols,
                'channel_rate': rate,
                'd_r': d_r,
                'n_seeds': len(seeds),
            }
        )
    finally:
        run.finish()

    if not written:
        raise SystemExit('No figures produced; nothing was written.')

    if reference:
        print(
            '\nNative RX ceiling: '
            + '  '.join(f'{k}={v:.4f}' for k, v in reference.items())
        )
    print('\nWrote: ' + ', '.join(str(p) for p in written))


if __name__ == '__main__':
    main()

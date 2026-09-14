"""Figure (i): RKA against Procrustes, over the RKHS regularisation.

One figure is one metric against ``lambda``: the RKA curve, and the flat
Procrustes line it is read against. The line is flat because Procrustes
has no ``lambda`` -- it *is* the ``lambda -> infinity`` limit of the curve
above it, so the vertical gap between them is exactly what the kernel
residual buys.

That figure is then repeated over the axes the study varies, all of which
live in the config rather than in a shell loop:

- ``charts``: the preprocessing configuration, merged into RKA and
  Procrustes alike so the two always share a chart (``axes/default.yaml``
  says why that is not optional);
- ``ranks``: how many directions each chart keeps, crossed with
  ``charts`` so every factorisation is measured at every compression;
- ``pilots``: the calibration budget ``N``, either absolute or derived
  from the rate as ``ceil(symbols * multiplier)``.

So one invocation writes ``charts x ranks x budgets x metrics`` figures,
and one CSV per ``(chart, rank, budget)`` cell. Every output is named
after the whole configuration that produced it -- dataset, encoder pair,
chart, rate, budget, lambda grid -- because a figure loses its path the
moment it is copied into a manuscript.

Those CSVs are the input to ``pilot_sweep.py``, which reads RKA's best
lambda per budget back out of them instead of re-selecting it.

Cells are written as they finish and ``resume=true`` skips the ones
already on disk, so a run that dies part-way costs the current cell and
nothing else -- re-invoking the script picks up where it stopped.

Examples
--------
    just lambda-sweep
    uv run scripts/lambda_sweep.py symbol_divisor=7
    uv run scripts/lambda_sweep.py 'pilots.counts=[500, 1000]'
    uv run scripts/lambda_sweep.py 'charts=[{preprocess: pca}]'
    uv run scripts/lambda_sweep.py 'ranks=[384, 192, 96]'
    uv run scripts/lambda_sweep.py lam.min=1e-6 lam.max=1e-1 lam.per_decade=8
    uv run scripts/lambda_sweep.py plot_only=true
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
from src.plotting import plot_regularization_metric, use_project_style
from src.reporting import (
    chart_slug,
    figure_dir,
    lambda_stem,
    pairs_slug,
    rate_slug,
    result_dir,
)

if TYPE_CHECKING:
    from src.decoder import Decoder
    from src.latent import LatentSpace

log = logging.getLogger(__name__)

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / 'config' / 'hydra')
ROOT = Path(__file__).resolve().parents[1]

# Columns of a cell CSV that identify the cell rather than describe the
# curve. This is a contract, not a dump: `pilot_sweep.py` finds the
# lambda measured at a given (chart, rate, budget) by globbing the
# results directory and filtering on these fields. It never parses a
# filename, so renaming an output cannot silently change what it reads.
CONTEXT: tuple[str, ...] = (
    'dataset',
    'pairs',
    'chart',
    'preprocess',
    'symbols',
    'n_pilots',
    'pilots_per_symbol',
    'strategy',
)

# Which of those survive a round trip through the CSV as text.
_TEXT: frozenset[str] = frozenset(
    {'dataset', 'pairs', 'chart', 'preprocess', 'strategy'}
)


def lambda_grid(cfg: DictConfig) -> list[float]:
    """The lambdas to sweep: the explicit grid, or a generated log range."""
    grid = cfg.lam.get('grid')
    if grid:
        return sorted(float(v) for v in grid)

    low = math.log10(float(cfg.lam.min))
    high = math.log10(float(cfg.lam.max))
    per_decade = float(cfg.lam.per_decade)
    n = round((high - low) * per_decade) + 1
    if n < 2:
        raise ValueError(
            f'lam spans {high - low:g} decades at {per_decade:g} points '
            'per decade, which is fewer than two lambdas.'
        )
    return [
        float(10.0 ** (low + i * (high - low) / (n - 1))) for i in range(n)
    ]


# ---------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------


def pilot_paths(
    cfg: DictConfig,
    pairs: list[tuple[str, str]],
    agents: dict[str, dict[str, LatentSpace]],
    counts: list[int],
) -> dict[tuple[str, int], dict[int, np.ndarray]]:
    """Pilot sets per ``(source, repeat)``, for every budget at once.

    Selection sees only the transmitter's raw latents, so it depends on
    neither the chart nor ``lambda``. Every cell of the grid therefore
    shares one selection, and every point on a curve sees the same
    calibration data -- which is what makes the sweep a measurement of
    regularisation rather than of pilot luck.

    ``select_pilot_path`` also collapses the budget axis into a single
    greedy pass: kernel herding never revisits a choice, so the budget-k
    answer is the k-prefix of the largest. On a 50k-row training split
    that is the difference between one pool-sized Gram sweep and eleven.
    """
    paths: dict[tuple[str, int], dict[int, np.ndarray]] = {}
    for repeat in range(int(cfg.pilots.n_repeats)):
        seed = int(cfg.seed) + 1000 * repeat
        for source, _ in pairs:
            if (source, repeat) in paths:
                continue
            train = agents[source]['train']
            pool = draw_pool(train, cfg.pilots.pool_size, seed)
            labels = train.labels
            log.info(
                'Selecting up to %d %s pilots for %s from a pool of %d '
                '(repeat %d).',
                max(counts),
                cfg.pilots.strategy,
                source,
                pool.size,
                repeat,
            )
            within = select_pilot_path(
                train.latent[pool],
                counts=counts,
                strategy=str(cfg.pilots.strategy),
                labels=None if labels is None else labels[pool],
                seed=seed,
            )
            paths[(source, repeat)] = {
                n: pool[idx] for n, idx in within.items()
            }
    return paths


class PilotSets:
    """Pilot sets per ``(source, repeat)``, selected on first use.

    Selection is the most expensive thing a resumed run can do for
    nothing. Kernel herding is greedy over the whole training split and
    its cost scales with the *largest* budget on the axis, so a sweep
    whose cells are all already on disk would pay for a 24576-pilot
    selection up front and then never look at the result -- which is
    precisely the run someone repeats most often, to redraw a figure or
    to add one cell to a finished grid.

    Deferring it to the first cell that actually has to be fitted makes a
    fully-resumed run cost a data load and a redraw, and changes nothing
    about a run that does fit: the selection still happens exactly once,
    and every cell still sees the same pilots.
    """

    def __init__(
        self,
        cfg: DictConfig,
        pairs: list[tuple[str, str]],
        agents: dict[str, dict[str, LatentSpace]],
        counts: list[int],
    ) -> None:
        self._args = (cfg, pairs, agents, counts)
        self._paths: dict[tuple[str, int], dict[int, np.ndarray]] | None = None

    def __getitem__(self, key: tuple[str, int]) -> dict[int, np.ndarray]:
        if self._paths is None:
            self._paths = pilot_paths(*self._args)
        return self._paths[key]


def fit_on(
    aligner: Aligner,
    pilots: np.ndarray,
    agents: dict[str, dict[str, LatentSpace]],
    pair: tuple[str, str],
) -> Aligner:
    """Fit one aligner on one pilot set."""
    source, target = pair
    src_train, tgt_train = agents[source]['train'], agents[target]['train']
    labels = src_train.labels
    return aligner.fit(
        src_train.latent[pilots],
        tgt_train.latent[pilots],
        labels=None if labels is None else labels[pilots],
    )


def evaluate(
    cfg: DictConfig,
    aligner: Aligner,
    agents: dict[str, dict[str, LatentSpace]],
    pair: tuple[str, str],
    decoder: Decoder | None,
) -> dict[str, float]:
    """Score a fitted aligner on the whole test split."""
    source, target = pair
    tgt_test = agents[target]['test']
    return alignment_metrics(
        aligner.transform(agents[source]['test'].latent),
        tgt_test.latent,
        decoder=decoder,
        labels=tgt_test.labels,
        ks=tuple(cfg.eval.topk),
    )


def score(
    cfg: DictConfig,
    aligner: Aligner,
    pilots: np.ndarray,
    agents: dict[str, dict[str, LatentSpace]],
    pair: tuple[str, str],
    decoder: Decoder | None,
) -> dict[str, float]:
    """Fit one aligner on one pilot set and score the whole test split."""
    return evaluate(
        cfg, fit_on(aligner, pilots, agents, pair), agents, pair, decoder
    )


def sweep_cell(
    cfg: DictConfig,
    pairs: list[tuple[str, str]],
    agents: dict[str, dict[str, LatentSpace]],
    decoders: dict[str, Decoder | None],
    paths: PilotSets,
    chart: DictConfig,
    symbols: int,
    n_pilots: int,
    grid: list[float],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, float]]]:
    """One (chart, budget) cell: the swept curve and the flat baselines."""
    metrics = list(cfg.eval.metrics)
    swept = str(cfg.sweep_method)
    repeats = range(int(cfg.pilots.n_repeats))

    curve: dict[float, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    flat: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for name, method_cfg in cfg.methods.items():
        configured = configure_method(method_cfg, chart, symbols)

        if name != swept:
            for repeat in repeats:
                for source, target in pairs:
                    scores = score(
                        cfg,
                        build_aligner(configured),
                        paths[(source, repeat)][n_pilots],
                        agents,
                        (source, target),
                        decoders.get(target),
                    )
                    for metric in metrics:
                        flat[name][metric].append(scores[metric])
            log.info(
                'N=%d baseline %s: %s',
                n_pilots,
                name,
                ', '.join(
                    f'{m}={np.mean(flat[name][m]):.4f}' for m in metrics
                ),
            )
            continue

        # One fit per pilot set, then one re-solve per lambda. Everything
        # a fit computes before the ridge -- the rigid stage, the
        # bandwidth, the eigendecomposition of the centred Gram matrix --
        # is independent of lambda, and that eigendecomposition is the
        # O(N^3) part: refitting it for every point of a 25-point grid
        # made an N=8192 cell cost ~20 minutes, almost all of it
        # recomputing one matrix. `set_lam` is exactly the fit a fresh
        # aligner would produce (tests/test_alignment.py pins that). A
        # method without it is refitted per lambda, as before.
        for repeat in repeats:
            for source, target in pairs:
                pilots = paths[(source, repeat)][n_pilots]
                aligner: Aligner | None = None
                for lam in grid:
                    # `lam_grid=None` matters as much as `lam`: left set,
                    # the aligner holds out a slice of the pilots and
                    # re-selects, and the point of a sweep is to see the
                    # whole curve rather than be handed one point off it.
                    if aligner is not None and hasattr(aligner, 'set_lam'):
                        aligner.set_lam(float(lam))
                    else:
                        pinned = OmegaConf.merge(
                            configured, {'lam': float(lam), 'lam_grid': None}
                        )
                        aligner = fit_on(
                            build_aligner(pinned),
                            pilots,
                            agents,
                            (source, target),
                        )
                    scores = evaluate(
                        cfg,
                        aligner,
                        agents,
                        (source, target),
                        decoders.get(target),
                    )
                    for metric in metrics:
                        curve[float(lam)][metric].append(scores[metric])

        for lam in grid:
            log.info(
                'N=%d lam=%.4g: %s',
                n_pilots,
                lam,
                ', '.join(
                    f'{m}={np.mean(curve[float(lam)][m]):.4f}' for m in metrics
                ),
            )

    baselines = {
        name: {m: float(np.mean(v)) for m, v in values.items()}
        for name, values in flat.items()
    }
    return summarise(curve, metrics), baselines


def summarise(
    curve: dict[float, dict[str, list[float]]], metrics: list[str]
) -> list[dict[str, Any]]:
    """Mean and spread over the (pair, repeat) cells at each ``lambda``."""
    rows = []
    for lam, values in sorted(curve.items()):
        row: dict[str, Any] = {'lam': lam, 'n_cells': len(values[metrics[0]])}
        for metric in metrics:
            row[f'{metric}_mean'] = float(np.mean(values[metric]))
            row[f'{metric}_std'] = float(np.std(values[metric]))
        rows.append(row)
    return rows


# ---------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------


def annotate(
    rows: list[dict[str, Any]],
    context: dict[str, Any],
    baselines: dict[str, dict[str, float]],
) -> list[dict[str, Any]]:
    """Put the cell's identity, and its flat baselines, on every row.

    The baselines are carried rather than recomputed because comparing
    cells later means comparing each curve against *its own* reference,
    and the alternative is refitting to recover a number the run already
    had.
    """
    missing = [key for key in CONTEXT if key not in context]
    if missing:
        raise ValueError(
            f'The cell is missing the context columns {missing}, which '
            '`pilot_sweep.py` filters on. See CONTEXT.'
        )
    extra = {
        f'{name}_{metric}': value
        for name, values in baselines.items()
        for metric, value in values.items()
    }
    return [context | row | extra for row in rows]


def write_cell(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write one cell's CSV, via a temporary so a crash cannot truncate it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.csv.partial')
    with tmp.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def read_cell(path: Path) -> list[dict[str, Any]]:
    """Read one cell's CSV back, with the numeric columns parsed."""
    with path.open() as handle:
        rows = list(csv.DictReader(handle))

    parsed = []
    for row in rows:
        out: dict[str, Any] = {}
        for key, value in row.items():
            if key in _TEXT:
                out[key] = value
                continue
            try:
                out[key] = float(value)
            except (TypeError, ValueError):
                out[key] = value or None
        for key in ('symbols', 'n_pilots', 'n_cells'):
            if isinstance(out.get(key), float):
                out[key] = int(out[key])
        parsed.append(out)
    return parsed


def split_baselines(
    rows: list[dict[str, Any]], names: list[str], metrics: list[str]
) -> dict[str, dict[str, float]]:
    """Recover the flat baselines from a cell read back off disk."""
    return {
        name: {
            metric: float(rows[0][f'{name}_{metric}'])
            for metric in metrics
            if rows[0].get(f'{name}_{metric}') is not None
        }
        for name in names
        if any(rows[0].get(f'{name}_{m}') is not None for m in metrics)
    }


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------


def best_lambda(
    rows: list[dict[str, Any]], metrics: list[str]
) -> dict[str, float]:
    """The ``lambda`` maximising each metric."""
    return {
        metric: float(max(rows, key=lambda r: r[f'{metric}_mean'])['lam'])
        for metric in metrics
    }


def cell_title(
    cfg: DictConfig, chart: DictConfig, symbols: int | None, n_pilots: int
) -> str:
    """Figure title for one cell.

    Built from words rather than from the filename slugs: the project
    style renders titles through LaTeX where it is installed, and a slug
    carrying underscores does not survive that.
    """
    if cfg.output.get('title'):
        return str(cfg.output.title)
    dataset = cfg.data.get('dataset', cfg.data.source)
    rate = 'full width' if symbols is None else f'{symbols} symbols'
    return (
        f'RKA vs Procrustes — {dataset}, '
        f'{chart.get("preprocess", "whiten")} chart, '
        f'{rate}, {n_pilots} pilots'
    )


def draw(
    cfg: DictConfig,
    rows: list[dict[str, Any]],
    baselines: dict[str, dict[str, float]],
    reference: dict[str, float],
    directory: Path,
    stem: str,
    title: str,
) -> list[Path]:
    """One figure per metric, for one cell."""
    return [
        path
        for metric in cfg.eval.metrics
        for path in plot_regularization_metric(
            rows,
            metric=str(metric),
            out_path=directory / f'{stem}_{metric}',
            baselines=baselines,
            reference=reference,
            title=title,
            method=str(cfg.sweep_method),
            formats=tuple(cfg.output.formats),
        )
    ]


def overview_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    """Fixed-width view of the grid (also the accessibility fallback)."""

    def cell(row: dict[str, Any], column: str) -> str:
        value = row.get(column)
        if value is None:
            return '-'
        if not isinstance(value, float):
            return str(value)
        return f'{value:+.4f}' if column.startswith('gain_') else f'{value:g}'

    body = [[cell(row, c) for c in columns] for row in rows]
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
    version_base=None, config_path=CONFIG_DIR, config_name='lambda_sweep'
)
def main(cfg: DictConfig) -> None:
    """Sweep lambda over every (chart, budget) cell; write the figures."""
    logging.getLogger('src').setLevel(logging.INFO)
    log.info('Configuration:\n%s', OmegaConf.to_yaml(cfg))

    metrics = list(cfg.eval.metrics)
    swept = str(cfg.sweep_method)
    if swept not in cfg.methods:
        raise SystemExit(
            f'sweep_method={swept!r} is not one of the configured methods '
            f'({", ".join(cfg.methods)}).'
        )
    flat_names = [name for name in cfg.methods if name != swept]

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
    # The budget axis is scaled against what actually goes on the
    # channel, which is the transmitter's full width when nothing is
    # compressed -- so `pilots per symbol` means the same thing either way.
    rate = channel_rate(symbols, max_rank)
    # The chart axis crossed with `ranks`, when the config gives any;
    # both studies expand it the same way, or the lambda measured at
    # one (chart, rank) could not be looked up at the other. Each chart
    # then carries its own budget axis, scaled to the rank it keeps.
    charts = resolve_charts(cfg, max_rank)
    plan = resolve_chart_budgets(
        cfg, charts, cfg.methods[swept], symbols, max_rank
    )
    # One pilot selection serves the whole grid, so it has to cover every
    # budget any chart asks for. Herding is greedy, so the union costs one
    # pass to its maximum rather than one pass per chart.
    counts = sorted({n for *_, budgets in plan for n in budgets})
    grid = lambda_grid(cfg)
    check_budgets_fit_pool(
        counts,
        cfg.pilots.pool_size,
        min(agents[src]['train'].n_points for src, _ in pairs),
    )
    log.info(
        'd_r=%d, max_rank=%d -> %s; %d charts, %d cells, %d lambdas '
        '(%.3g..%.3g).',
        d_r,
        max_rank,
        f'compressing to {symbols} symbols'
        if symbols is not None
        else f'no compression, {rate} symbols at full width',
        len(charts),
        sum(len(b) for *_, b in plan),
        len(grid),
        grid[0],
        grid[-1],
    )
    # Spelled out per chart because the budgets are no longer one axis:
    # `pilots per symbol` is scaled to the rank each chart keeps, so a
    # reader has to be able to see which N belong to which rank.
    for chart, rank, chart_rate, budgets in plan:
        log.info(
            '  chart=%-16s %-6s rate=%-5d budgets %s',
            chart_slug(chart),
            rate_slug(rank),
            chart_rate,
            ', '.join(str(n) for n in budgets),
        )

    results = result_dir(cfg.output.results, str(cfg.output.study), dataset)
    use_project_style(ROOT / 'config' / 'plotting' / 'plt.mplstyle')

    # Not selected yet: `PilotSets` defers the herding pass until a
    # cell actually needs fitting, so a fully-resumed run does not pay
    # for pilots it will never use.
    paths = PilotSets(cfg, pairs, agents, counts)

    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.name
        or f'lambda-{dataset}-{pairs_slug(pairs)}-{rate_slug(symbols)}',
        group=cfg.wandb.group,
        job_type='lambda-sweep',
        tags=list(cfg.wandb.tags),
        mode=cfg.wandb.mode,
        config=OmegaConf.to_container(cfg, resolve=True)
        | {
            'resolved_symbols': symbols,
            'resolved_channel_rate': rate,
            'resolved_counts': counts,
            'resolved_budgets_per_chart': {
                chart_slug(c): b for c, _, _, b in plan
            },
            'resolved_lam_grid': grid,
            'd_r': d_r,
        },
    )

    figures_written: list[Path] = []
    overview: list[dict[str, Any]] = []
    try:
        for chart, rank, chart_rate, budgets in plan:
            tag = chart_slug(chart)
            figures = figure_dir(
                cfg.output.figures, str(cfg.output.study), dataset, tag
            )

            for n_pilots in budgets:
                stem = lambda_stem(dataset, pairs, chart, rank, n_pilots, grid)
                path = results / f'{stem}.csv'

                if path.exists() and (cfg.resume or cfg.plot_only):
                    rows = read_cell(path)
                    baselines = split_baselines(rows, flat_names, metrics)
                    # Named in full, and attributed: this is the local
                    # results tree, never wandb -- these scripts only ever
                    # write to wandb, they never read a run back.
                    log.info(
                        'Reusing the cell already on disk at %s '
                        '(resume=true); pass resume=false to refit it.',
                        path,
                    )
                elif cfg.plot_only:
                    log.warning(
                        'plot_only=true and %s is missing; skipping the '
                        'chart=%s N=%d cell.',
                        path.name,
                        tag,
                        n_pilots,
                    )
                    continue
                else:
                    log.info(
                        '=== chart=%s  %s  N=%d ===',
                        tag,
                        rate_slug(rank),
                        n_pilots,
                    )
                    rows, baselines = sweep_cell(
                        cfg,
                        pairs,
                        agents,
                        decoders,
                        paths,
                        chart,
                        symbols,
                        n_pilots,
                        grid,
                    )
                    rows = annotate(
                        rows,
                        {
                            'dataset': dataset,
                            'pairs': pairs_slug(pairs),
                            'chart': tag,
                            'preprocess': str(
                                chart.get('preprocess', 'whiten')
                            ),
                            'symbols': rank,
                            'n_pilots': n_pilots,
                            'pilots_per_symbol': round(
                                n_pilots / chart_rate, 4
                            ),
                            'strategy': str(cfg.pilots.strategy),
                        },
                        baselines,
                    )
                    write_cell(path, rows)

                figures_written += draw(
                    cfg,
                    rows,
                    baselines,
                    reference,
                    figures,
                    stem,
                    cell_title(cfg, chart, rank, n_pilots),
                )

                best = best_lambda(rows, metrics)
                entry: dict[str, Any] = {
                    'chart': tag,
                    'symbols': rank,
                    'n_pilots': n_pilots,
                }
                for metric in metrics:
                    peak = max(r[f'{metric}_mean'] for r in rows)
                    line = baselines.get('procrustes', {}).get(metric)
                    entry[f'lam_{metric}'] = best[metric]
                    entry[f'rka_{metric}'] = peak
                    entry[f'proc_{metric}'] = line
                    entry[f'gain_{metric}'] = (
                        peak - line if line is not None else None
                    )
                overview.append(entry)

                table = wandb.Table(columns=list(rows[0]))
                for row in rows:
                    table.add_data(*row.values())
                run.log({f'curve/{tag}/n{n_pilots}': table})
                run.summary.update(
                    {
                        f'best_lam/{tag}/n{n_pilots}/{m}': v
                        for m, v in best.items()
                    }
                )

        for path in figures_written:
            if path.suffix == '.png':
                run.log({f'figure/{path.stem}': wandb.Image(str(path))})
        run.summary.update(
            {f'native/{k}': v for k, v in reference.items()}
            | {
                'symbols': symbols,
                'channel_rate': rate,
                'd_r': d_r,
                'n_cells': len(overview),
            }
        )
    finally:
        run.finish()

    if not overview:
        raise SystemExit('No cells produced; nothing was written.')

    print(
        f'\n{dataset} — {pairs_slug(pairs)}, {rate_slug(symbols)} '
        f'({rate} symbols on the channel), '
        f'{len(grid)} lambdas over {grid[0]:.3g}..{grid[-1]:.3g}\n'
    )
    print(
        overview_table(
            overview,
            ['chart', 'symbols', 'n_pilots']
            + [
                f'{part}_{metric}'
                for metric in metrics
                for part in ('lam', 'rka', 'proc', 'gain')
            ],
        )
    )
    if reference:
        print(
            '\nNative RX ceiling: '
            + '  '.join(f'{k}={v:.4f}' for k, v in reference.items())
        )
    print(
        f'\nWrote {len(figures_written)} figures under '
        f'{figure_dir(cfg.output.figures, str(cfg.output.study), dataset)}/ '
        f'and {len(overview)} CSVs under {results}/'
    )
    print('Next: uv run scripts/pilot_sweep.py')


if __name__ == '__main__':
    main()

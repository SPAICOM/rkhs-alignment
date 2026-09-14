"""Figures (ii) and (iii), averaged over encoder pairs instead of seeds.

``pilot_sweep.py`` and ``dimension_sweep.py`` each draw one encoder pair,
with the band over pilot realisations. Whether a gap belongs to the method
or to the pair is a different question, and this script answers it: it
fits nothing, pools what those two studies already wrote for every pair in
the config, and redraws both figures with the band over *pairs*.

The order of averaging is the point. Seeds are averaged within a pair
first, then mean and standard deviation are taken across pairs, so the
band is the spread between encoder pairs and does not shrink because one
pair happened to be run with more seeds.

Everything is matched on the CSVs' columns -- ``pairs``, ``chart``,
``symbols``, ``n_pilots``, ``strategy`` -- never on their filenames, as in
the other studies. A pair with nothing on disk stops the run and the
message is the list of commands that produce it; a pair that is only
partly there stops it too, because a mean over a different set of pairs
at each point is not a curve.

Next to the figures it prints, per pair, the margin RKA holds over the
best of every other method at the points the figures are read at, since
an average can hide a pair where the ordering flips.

Examples
--------
    just pair-average
    uv run scripts/pair_average.py 'pilot.drop=[direct_mlp:round_robin]'
"""

from __future__ import annotations

import csv
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.plotting import (
    plot_dimension_sweep,
    plot_pilot_efficiency,
    use_project_style,
)
from src.reporting import (
    budget_slug,
    chart_slug,
    figure_dir,
    pairs_slug,
    ranks_slug,
    rate_slug,
    result_dir,
    strategies_slug,
)

log = logging.getLogger(__name__)

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / 'config' / 'hydra')
ROOT = Path(__file__).resolve().parents[1]

_TEXT: frozenset[str] = frozenset(
    {
        'dataset',
        'pairs',
        'chart',
        'base_chart',
        'preprocess',
        'strategy',
        'method',
        'source',
        'target',
        'decoder',
    }
)
_INTS: tuple[str, ...] = (
    'n_pilots',
    'seed',
    'symbols',
    'requested',
    'n_pilots_used',
)


# ---------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------


def read_csv(path: Path) -> list[dict[str, Any]]:
    """One CSV, with the numeric columns parsed."""
    rows = []
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
            for key in _INTS:
                if isinstance(parsed.get(key), float):
                    parsed[key] = int(parsed[key])
            rows.append(parsed)
    return rows


def pilot_records(
    directory: Path,
    tags: list[str],
    chart: str,
    symbols: int,
    counts: list[int],
    strategies: list[str],
) -> list[dict[str, Any]]:
    """Every per-seed pilot-sweep record describing one of the pairs."""
    wanted, budgets, designs = set(tags), set(counts), set(strategies)
    return [
        row
        for path in sorted(directory.glob('pilots_*_seed*.csv'))
        for row in read_csv(path)
        if row.get('pairs') in wanted
        and row.get('chart') == chart
        and row.get('symbols') == symbols
        and row.get('n_pilots') in budgets
        and row.get('strategy') in designs
    ]


def dimension_rows(
    directory: Path,
    tags: list[str],
    base_chart: str,
    n_pilots: int,
    strategy: str,
    ranks: list[int],
) -> list[dict[str, Any]]:
    """One dimension-sweep point per ``(pair, method, rank)``.

    A pair can have several summaries on disk -- one per rank axis it was
    drawn over -- and they describe the same fits. The newest file wins
    each point, so a rerun after a fix supersedes what it fixed.
    """
    wanted, axis = set(tags), set(ranks)
    points: dict[tuple[str, str, int], dict[str, Any]] = {}
    paths = sorted(
        directory.glob('dims_*.csv'), key=lambda p: p.stat().st_mtime
    )
    for path in paths:
        for row in read_csv(path):
            if (
                row.get('pairs') in wanted
                and row.get('base_chart') == base_chart
                and row.get('n_pilots') == n_pilots
                and row.get('strategy') == strategy
                and row.get('requested') in axis
            ):
                points[(row['pairs'], row['method'], row['requested'])] = row
    return list(points.values())


# ---------------------------------------------------------------------
# Averaging
# ---------------------------------------------------------------------


def average_over_pairs(
    rows: list[dict[str, Any]],
    keys: tuple[str, ...],
    metrics: list[str],
) -> list[dict[str, Any]]:
    """Mean within each pair over its repeats, then mean +/- sd over pairs.

    ``keys`` name one point of the curve (``method``, ``strategy``,
    ``n_pilots`` for the pilot figure). Whatever else varies within a
    ``(pair, *keys)`` group -- seeds -- is averaged away first.
    """
    within: dict[tuple, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        group = (row['pairs'], *(row[k] for k in keys))
        for metric in metrics:
            if row.get(metric) is not None:
                within[group][metric].append(float(row[metric]))

    across: dict[tuple, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for (_, *point), values in within.items():
        for metric, samples in values.items():
            across[tuple(point)][metric].append(float(np.mean(samples)))

    out = []
    for point, values in across.items():
        row: dict[str, Any] = dict(zip(keys, point))
        row['n_pairs'] = len(next(iter(values.values())))
        for metric, samples in values.items():
            row[f'{metric}_mean'] = float(np.mean(samples))
            row[f'{metric}_std'] = float(np.std(samples))
        out.append(row)
    return out


def missing_points(
    rows: list[dict[str, Any]],
    tags: list[str],
    keys: tuple[str, ...],
    expected: list[tuple],
) -> dict[str, list[tuple]]:
    """``pair -> [points it has no row for]``, over the expected grid."""
    have = defaultdict(set)
    for row in rows:
        have[row['pairs']].add(tuple(row[k] for k in keys))
    return {
        tag: [p for p in expected if p not in have[tag]]
        for tag in tags
        if any(p not in have[tag] for p in expected)
    }


def margins(
    rows: list[dict[str, Any]],
    at: dict[str, Any],
    metric: str,
    ours: str = 'rkhs',
) -> list[tuple[str, float, str, float]]:
    """Per pair: ``(pair, ours - best other, best other, ours)`` at a point."""
    by_pair: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        if (
            all(row.get(k) == v for k, v in at.items())
            and row.get(metric) is not None
        ):
            by_pair[row['pairs']][row['method']].append(float(row[metric]))
    out = []
    for pair, methods in sorted(by_pair.items()):
        means = {m: float(np.mean(v)) for m, v in methods.items()}
        if ours not in means or len(means) < 2:
            continue
        rival = max((m for m in means if m != ours), key=means.get)
        out.append((pair, means[ours] - means[rival], rival, means[ours]))
    return out


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------


def commands_for(cfg: DictConfig, source: str, target: str) -> list[str]:
    """The per-pair runs this study reads, as `just` commands."""
    chart = '{' + ','.join(f'{k}:{v}' for k, v in cfg.chart.items()) + '}'
    common = (
        f"'data.models=[{source},{target}]' receiver={target} "
        f"'charts=[{chart}]'"
    )
    counts = ','.join(str(c) for c in cfg.pilot.counts)
    symbols = int(cfg.pilot.symbols)
    others = [int(r) for r in cfg.dimension.ranks if int(r) != symbols]
    n = int(cfg.dimension.n_pilots)
    decoder = f'decoder.kind={cfg.decoder}'
    strategies = ','.join(str(s) for s in cfg.pilot.strategies)
    ranks = ','.join(str(r) for r in cfg.dimension.ranks)
    seeds = ','.join(str(r) for r in cfg.pilot.seeds)
    return [
        (
            f"just lambda-sweep {common} 'ranks=[{symbols}]' "
            f"'pilots.counts=[{counts}]' {decoder}"
        ),
        (
            f'just lambda-sweep {common} '
            f"'ranks=[{','.join(map(str, others))}]' "
            f"'pilots.counts=[{n}]' {decoder}"
        ),
        (
            f"just pilot-sweep {common} 'ranks=[{symbols}]' "
            f"'pilots.counts=[{counts}]' "
            f"'pilots.strategies=[{strategies}]' 'seeds=[{seeds}]' {decoder}"
        ),
        (
            f"just dimension-sweep {common} 'ranks=[{ranks}]' "
            f'pilots.n_pilots={n} {decoder}'
        ),
    ]


def parse_drop(entries: Any) -> tuple[set[str], set[tuple[str, str]]]:
    """``method`` or ``method:design`` entries, as in ``pilot_sweep.py``."""
    methods, cells = set(), set()
    for entry in entries or ():
        method, sep, design = str(entry).partition(':')
        if sep:
            cells.add((method.strip(), design.strip()))
        else:
            methods.add(method.strip())
    return methods, cells


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a CSV via a temporary, so a crash cannot truncate it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'{path.name}.partial')
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with tmp.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def print_margins(title: str, rows: list[tuple[str, float, str, float]]):
    """One line per pair: RKA's value and its margin over the best rival."""
    print(f'\n{title}')
    for pair, gap, rival, value in rows:
        print(
            f'  {pair:<46s} RKA {value:.4f}   {gap * 100:+6.2f} pp over '
            f'{rival}'
        )


@hydra.main(
    version_base=None, config_path=CONFIG_DIR, config_name='pair_average'
)
def main(cfg: DictConfig) -> None:
    """Pool the per-pair studies and draw both figures averaged over pairs."""
    logging.getLogger('src').setLevel(logging.INFO)
    metrics = list(cfg.eval.metrics)
    dataset = str(cfg.data.get('dataset', cfg.data.source))
    pairs = [(str(p.source), str(p.target)) for p in cfg.pairs]
    tags = [pairs_slug([pair]) for pair in pairs]
    if len(set(tags)) != len(tags):
        raise SystemExit('`pairs` lists the same encoder pair twice.')

    base = chart_slug(cfg.chart)
    symbols = int(cfg.pilot.symbols)
    counts = [int(c) for c in cfg.pilot.counts]
    strategies = [str(s) for s in cfg.pilot.strategies]
    ranks = [int(r) for r in cfg.dimension.ranks]
    n_pilots = int(cfg.dimension.n_pilots)
    root = cfg.output.results

    pilots = pilot_records(
        result_dir(root, str(cfg.pilot.study), dataset),
        tags,
        f'{base}-{rate_slug(symbols)}',
        symbols,
        counts,
        strategies,
    )
    dims = dimension_rows(
        result_dir(root, str(cfg.dimension.study), dataset),
        tags,
        base,
        n_pilots,
        str(cfg.dimension.strategy),
        ranks,
    )

    # --- coverage: every pair at every point, or nothing is drawn -----
    pilot_keys = ('method', 'strategy', 'n_pilots')
    dim_keys = ('method', 'requested')
    pilot_methods = sorted({r['method'] for r in pilots})
    dim_methods = sorted({r['method'] for r in dims})
    gaps = {
        'pilot sweep': missing_points(
            pilots,
            tags,
            pilot_keys,
            [
                (m, s, n)
                for m in pilot_methods
                for s in strategies
                for n in counts
            ],
        )
        if pilots
        else {tag: ['everything'] for tag in tags},
        'dimension sweep': missing_points(
            dims,
            tags,
            dim_keys,
            [(m, k) for m in dim_methods for k in ranks],
        )
        if dims
        else {tag: ['everything'] for tag in tags},
    }
    if any(gaps.values()):
        lines = []
        for study, missing in gaps.items():
            for tag, points in missing.items():
                lines.append(
                    f'  {study}: {tag} lacks {len(points)} point(s), e.g. '
                    f'{points[0]}'
                )
        todo = [
            (s, t)
            for (s, t), tag in zip(pairs, tags)
            if any(tag in missing for missing in gaps.values())
        ]
        commands = '\n\n'.join(
            '\n'.join(commands_for(cfg, s, t)) for s, t in todo
        )
        raise SystemExit(
            'Not every pair is on disk at every point:\n'
            + '\n'.join(lines)
            + '\n\nRun, per pair (each resumes what is already there):\n\n'
            + commands
        )
    decoders = {r.get('decoder') for r in dims} - {None}
    if decoders and decoders != {str(cfg.decoder)}:
        raise SystemExit(
            f'The dimension-sweep rows were scored with decoder '
            f'{sorted(decoders)}, but this study is configured for '
            f'decoder={cfg.decoder}.'
        )

    # --- figure (ii) ---------------------------------------------------
    dropped, dropped_cells = parse_drop(cfg.pilot.drop)
    order = list(dict.fromkeys(r['method'] for r in pilots))
    pilot_summary = sorted(
        average_over_pairs(pilots, pilot_keys, metrics),
        key=lambda r: (order.index(r['method']), r['strategy'], r['n_pilots']),
    )
    natives = defaultdict(list)
    for row in pilots:
        if isinstance(row.get('native_accuracy'), float):
            natives[row['pairs']].append(row['native_accuracy'])
    reference = (
        {'accuracy': float(np.mean([np.mean(v) for v in natives.values()]))}
        if natives
        else None
    )

    stem = '_'.join(
        (
            'pairavg-pilots',
            dataset,
            pairs_slug(pairs),
            f'{base}-{rate_slug(symbols)}',
            budget_slug(counts),
            strategies_slug(strategies),
        )
    )
    results = result_dir(root, str(cfg.output.study), dataset)
    write_rows(results / f'{stem}.csv', pilot_summary)
    use_project_style(ROOT / 'config' / 'plotting' / 'plt.mplstyle')
    shown = [
        r
        for r in pilot_summary
        if r['method'] not in dropped
        and (r['method'], r['strategy']) not in dropped_cells
    ]
    written = plot_pilot_efficiency(
        shown,
        metrics=metrics,
        out_path=figure_dir(
            cfg.output.figures, str(cfg.output.study), dataset, base
        )
        / stem,
        reference=reference,
        title=(
            f'Semantic pilots — {dataset}, {base} chart, {symbols} symbols, '
            f'mean ± sd over {len(pairs)} encoder pairs'
        ),
        formats=tuple(cfg.output.formats),
    )

    # --- figure (iii) --------------------------------------------------
    order = list(dict.fromkeys(r['method'] for r in dims))
    dim_summary = average_over_pairs(dims, dim_keys, metrics)
    delivered = defaultdict(list)
    for row in dims:
        delivered[(row['method'], row['requested'])].append(row['symbols'])
    for row in dim_summary:
        # Drawn at what every pair delivered, never at what one did.
        row['symbols'] = min(delivered[(row['method'], row['requested'])])
    dim_summary.sort(key=lambda r: (order.index(r['method']), r['requested']))

    stem = '_'.join(
        (
            'pairavg-dims',
            dataset,
            pairs_slug(pairs),
            base,
            f'n{n_pilots}',
            strategies_slug([str(cfg.dimension.strategy)]),
            ranks_slug(ranks),
        )
    )
    write_rows(results / f'{stem}.csv', dim_summary)
    written += plot_dimension_sweep(
        dim_summary,
        metrics=metrics,
        out_path=figure_dir(
            cfg.output.figures, str(cfg.output.study), dataset, base
        )
        / stem,
        reference=reference,
        hue_of=OmegaConf.to_container(cfg.hue_of, resolve=True),
        title=(
            f'Compression — {dataset}, {base} chart, N={n_pilots} '
            f'{cfg.dimension.strategy} pilots, mean ± sd over {len(pairs)} '
            'encoder pairs'
        ),
        formats=tuple(cfg.output.formats),
    )

    # --- where the average could hide a flip --------------------------
    for metric in metrics:
        for strategy in strategies:
            print_margins(
                f'{metric}, pilot sweep, {strategy} pilots, N={max(counts)}, '
                f'k={symbols}:',
                margins(
                    pilots,
                    {'strategy': strategy, 'n_pilots': max(counts)},
                    metric,
                ),
            )
        for k in ranks:
            print_margins(
                f'{metric}, dimension sweep, N={n_pilots}, k={k}:',
                margins(dims, {'requested': k}, metric),
            )

    print('\nWrote: ' + ', '.join(str(p) for p in written))


if __name__ == '__main__':
    main()

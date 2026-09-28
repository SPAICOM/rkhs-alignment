"""Ablation of RKA over the RKHS regularisation.

RKA is ``f(z) = Q z + g(z)``: a rigid Procrustes map ``Q``, then a
kernel-ridge correction ``g`` fitted to its residual under the constraint
``G X^T = 0``. This study removes those parts one at a time and sweeps
every variant over ``lambda`` the way ``lambda_sweep.py`` sweeps RKA:

- ``rkhs``      -- RKA itself;
- ``rkhs_free`` -- Procrustes plus kernel ridge on its residual, without
  the orthogonality constraint;
- ``krr``       -- pure kernel alignment: no Procrustes, so ``E = Y``,
  and no constraint.

As ``lambda`` grows the first two fall back onto the Procrustes line,
while pure kernel alignment has no rigid stage to fall back on and
collapses to the receiver's mean.

The sweep itself is ``lambda_sweep.py``'s, imported rather than copied:
one pilot selection serves every variant, and each variant is fitted once
per bandwidth and re-solved per ``lambda``. Each variant writes its own
CSV, in the same format as a figure (i) cell plus a ``method`` column.
Then, per metric, the run draws

- one comparison figure with every variant over ``lambda``, each at its
  best bandwidth (or at ``compare_bandwidth``), against Procrustes;
- one figure per variant, of the same kind figure (i) draws for RKA.

Outputs go under ``kernel_ablation/``, where ``pilot_sweep.py`` and
``dimension_sweep.py`` never glob for lambda cells.

Examples
--------
    just kernel-ablation-sweep
    just kernel-ablation-sweep compare_bandwidth=1.0
    just kernel-ablation-sweep plot_only=true
    uv run scripts/kernel_ablation_sweep.py 'variants=[rkhs,krr]'
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

import wandb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lambda_sweep import (
    PilotSets,
    annotate,
    bandwidth_axis,
    lambda_grid,
    overview_table,
    read_cell,
    split_baselines,
    sweep_cell,
    write_cell,
)

from src.experiment import (
    build_decoder,
    channel_rate,
    check_budgets_fit_pool,
    describe_pairs,
    native_accuracy,
    resolve_chart_budgets,
    resolve_charts,
    resolve_star,
    resolve_symbols,
    usable_rank,
)
from src.plotting import (
    plot_dimension_sweep,
    plot_regularization_ablation,
    plot_regularization_bandwidths,
    plot_regularization_metric,
    use_project_style,
)
from src.reporting import (
    chart_slug,
    dimension_stem,
    figure_dir,
    lambda_stem,
    pairs_slug,
    rate_slug,
    result_dir,
    strategies_slug,
)

if TYPE_CHECKING:
    from src.decoder import Decoder

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = str(ROOT / 'config' / 'hydra')


def variant_config(cfg: DictConfig, variant: str) -> DictConfig:
    """``cfg`` with ``variant`` swept and every non-variant method flat.

    The other variants are dropped rather than left in: ``sweep_cell``
    fits anything it does not sweep as a flat baseline, which for a
    kernel variant means a held-out lambda selection nobody asked for.
    """
    flat = [name for name in cfg.methods if name not in cfg.variants]
    view = OmegaConf.merge(cfg, {'sweep_method': variant})
    with open_dict(view):
        view.methods = {name: cfg.methods[name] for name in [variant, *flat]}
    return view


def curve_at(
    rows: list[dict[str, Any]], metric: str, bandwidth: float | None
) -> tuple[list[dict[str, Any]], float | None]:
    """One lambda curve out of a cell, and the bandwidth it was read at.

    A cell without a bandwidth axis is already one curve. On a surface,
    ``bandwidth=None`` takes the bandwidth holding the best point for
    ``metric``.
    """
    if 'bandwidth_scale' not in rows[0]:
        return rows, None
    if bandwidth is None:
        best = max(rows, key=lambda r: r[f'{metric}_mean'])
        bandwidth = float(best['bandwidth_scale'])
    curve = [r for r in rows if float(r['bandwidth_scale']) == bandwidth]
    if not curve:
        raise SystemExit(
            f'compare_bandwidth={bandwidth:g} is not on the bandwidth axis '
            'of the cell on disk.'
        )
    return curve, bandwidth


def draw_ranks(
    cfg: DictConfig,
    peaks: list[dict[str, Any]],
    reference: dict[str, float],
    dataset: str,
    pairs: list[tuple[str, str]],
    chart: DictConfig,
    figures: Path,
) -> list[Path]:
    """Each variant's best point against the compression rank.

    Drawn only when the run spans more than one rank at a budget, and
    read off the cells already written: the peak over lambda per
    (variant, rank), which is how figure (iii) reads RKA too. One figure
    per budget, since a rank axis at two budgets is two figures.
    """
    metrics = list(cfg.eval.metrics)
    written: list[Path] = []
    budgets = sorted({row['n_pilots'] for row in peaks})
    for n_pilots in budgets:
        rows = [row for row in peaks if row['n_pilots'] == n_pilots]
        if len({row['symbols'] for row in rows}) < 2:
            continue
        stem = 'kernel_ablation_dims' + dimension_stem(
            dataset,
            pairs,
            chart,
            n_pilots,
            str(cfg.pilots.strategy),
            sorted({int(row['symbols']) for row in rows}),
            str(cfg.decoder.kind),
        ).removeprefix('dims')
        written += plot_dimension_sweep(
            rows,
            metrics=metrics,
            out_path=figures / stem,
            reference=reference if cfg.output.native_rx else None,
            title=str(cfg.output.title) if cfg.output.get('title') else None,
            formats=tuple(cfg.output.formats),
            panel=tuple(float(v) for v in cfg.output.panel),
            text_scale=float(cfg.output.text_scale),
        )
    return written


def draw_variant(
    cfg: DictConfig,
    variant: str,
    rows: list[dict[str, Any]],
    baselines: dict[str, dict[str, float]],
    reference: dict[str, float],
    path: Path,
) -> list[Path]:
    """The figure (i) view of one variant: its surface, or its curve."""
    shared: dict[str, Any] = {
        'baselines': baselines,
        'reference': reference if cfg.output.native_rx else None,
        'title': str(cfg.output.title) if cfg.output.get('title') else None,
        'formats': tuple(cfg.output.formats),
        'panel': tuple(float(v) for v in cfg.output.panel),
        'text_scale': float(cfg.output.text_scale),
        'annotate': bool(cfg.output.annotate),
    }
    written: list[Path] = []
    for metric in cfg.eval.metrics:
        out = path.with_name(f'{path.name}_{metric}')
        if 'bandwidth_scale' in rows[0]:
            written += plot_regularization_bandwidths(
                rows, metric=str(metric), out_path=out, **shared
            )
        else:
            written += plot_regularization_metric(
                rows,
                metric=str(metric),
                out_path=out,
                method=variant,
                **shared,
            )
    return written


@hydra.main(
    version_base=None,
    config_path=CONFIG_DIR,
    config_name='kernel_ablation_sweep',
)
def main(cfg: DictConfig) -> None:
    """Sweep every variant over lambda per cell; draw them side by side."""
    logging.getLogger('src').setLevel(logging.INFO)
    log.info('Configuration:\n%s', OmegaConf.to_yaml(cfg))

    metrics = list(cfg.eval.metrics)
    variants = [str(v) for v in cfg.variants]
    missing = [v for v in variants if v not in cfg.methods]
    if missing:
        raise SystemExit(
            f'variants {missing} are not configured methods '
            f'({", ".join(cfg.methods)}).'
        )
    flat_names = [name for name in cfg.methods if name not in variants]

    bandwidths = bandwidth_axis(cfg)
    if bandwidths is not None:
        lacking = [
            v for v in variants if 'bandwidth_scale' not in cfg.methods[v]
        ]
        if lacking:
            raise SystemExit(
                f'bandwidths= was given, but {lacking} take no '
                '`bandwidth_scale`.'
            )
    compare = cfg.get('compare_bandwidth')
    if compare is not None:
        compare = float(compare)
        if bandwidths is None or compare not in bandwidths:
            raise SystemExit(
                f'compare_bandwidth={compare:g} is not on the bandwidth '
                f'axis {bandwidths}.'
            )

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

    for name in ('src.alignment.base', 'src.alignment.preprocessing'):
        logging.getLogger(name).setLevel(logging.ERROR)

    d_r, max_rank = usable_rank(pairs, agents)
    symbols = resolve_symbols(cfg, d_r, max_rank)
    rate = channel_rate(symbols, max_rank)
    charts = resolve_charts(cfg, max_rank)
    plan = resolve_chart_budgets(
        cfg, charts, cfg.methods[variants[0]], symbols, max_rank
    )
    counts = sorted({n for *_, budgets in plan for n in budgets})
    grid = lambda_grid(cfg)
    check_budgets_fit_pool(
        counts,
        cfg.pilots.pool_size,
        min(agents[src]['train'].n_points for src, _ in pairs),
    )

    study = str(cfg.output.study)
    results = result_dir(cfg.output.results, study, dataset)
    use_project_style(ROOT / 'config' / 'plotting' / 'plt.mplstyle')
    paths = PilotSets(cfg, pairs, agents, counts)
    compared = strategies_slug(variants) + (
        ''
        if bandwidths is None
        else f'_at-bw{"best" if compare is None else f"{compare:g}"}'
    ).replace('.', 'p')

    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.name
        or '-'.join(
            ('kernel_ablation', dataset, pairs_slug(pairs), rate_slug(symbols))
        ),
        group=cfg.wandb.group,
        job_type='kernel-ablation-sweep',
        tags=list(cfg.wandb.tags),
        mode=cfg.wandb.mode,
        config=OmegaConf.to_container(cfg, resolve=True)
        | {
            'resolved_symbols': symbols,
            'resolved_counts': counts,
            'resolved_lam_grid': grid,
            'resolved_bandwidths': bandwidths,
            'd_r': d_r,
        },
    )

    figures_written: list[Path] = []
    overview: list[dict[str, Any]] = []
    peaks: list[dict[str, Any]] = []
    try:
        for chart, rank, chart_rate, budgets in plan:
            tag = chart_slug(chart)
            figures = figure_dir(cfg.output.figures, study, dataset, tag)

            for n_pilots in budgets:
                stem = 'kernel_ablation' + lambda_stem(
                    dataset,
                    pairs,
                    chart,
                    rank,
                    n_pilots,
                    grid,
                    bandwidths=bandwidths,
                ).removeprefix('lambda')

                cells: dict[str, tuple[list, dict]] = {}
                for variant in variants:
                    path = results / f'{stem}_{variant}.csv'
                    if path.exists() and (cfg.resume or cfg.plot_only):
                        rows = read_cell(path)
                        baselines = split_baselines(rows, flat_names, metrics)
                        log.info(
                            'Reusing the %s cell already on disk at %s '
                            '(resume=true); pass resume=false to refit it.',
                            variant,
                            path,
                        )
                    elif cfg.plot_only:
                        log.warning(
                            'plot_only=true and %s is missing; skipping it.',
                            path.name,
                        )
                        continue
                    else:
                        log.info(
                            '=== %s  chart=%s  %s  N=%d ===',
                            variant,
                            tag,
                            rate_slug(rank),
                            n_pilots,
                        )
                        rows, baselines = sweep_cell(
                            variant_config(cfg, variant),
                            pairs,
                            agents,
                            decoders,
                            paths,
                            chart,
                            symbols,
                            n_pilots,
                            grid,
                            bandwidths,
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
                                'method': variant,
                            },
                            baselines,
                        )
                        write_cell(path, rows)

                    cells[variant] = (rows, baselines)
                    figures_written += draw_variant(
                        cfg,
                        variant,
                        rows,
                        baselines,
                        reference,
                        figures / f'{stem}_{variant}',
                    )
                    table = wandb.Table(columns=list(rows[0]))
                    for row in rows:
                        table.add_data(*row.values())
                    run.log({f'curve/{variant}/{tag}/n{n_pilots}': table})

                if not cells:
                    continue
                flat_line = next(iter(cells.values()))[1]
                for name, values in flat_line.items():
                    peaks.append(
                        {'method': name, 'symbols': rank, 'n_pilots': n_pilots}
                        | {f'{m}_mean': v for m, v in values.items()}
                        | {f'{m}_std': 0.0 for m in values}
                    )
                for variant, (rows_v, _) in cells.items():
                    best = {
                        m: max(float(r[f'{m}_mean']) for r in rows_v)
                        for m in metrics
                    }
                    peaks.append(
                        {
                            'method': variant,
                            'symbols': rank,
                            'n_pilots': n_pilots,
                        }
                        | {f'{m}_mean': v for m, v in best.items()}
                        | {f'{m}_std': 0.0 for m in best}
                    )

                # Procrustes is refitted with every variant on the same
                # pilots, so any variant's copy of the line will do.
                flat = next(iter(cells.values()))[1]
                for metric in metrics:
                    curves: dict[str, list[dict[str, Any]]] = {}
                    for variant, (rows, _) in cells.items():
                        curve, scale = curve_at(rows, metric, compare)
                        curves[variant] = curve
                        best = max(curve, key=lambda r: r[f'{metric}_mean'])
                        line = flat.get('procrustes', {}).get(metric)
                        overview.append(
                            {
                                'metric': metric,
                                'n_pilots': n_pilots,
                                'variant': variant,
                                'bw': scale,
                                'lam': float(best['lam']),
                                'peak': best[f'{metric}_mean'],
                                'right_end': max(
                                    curve, key=lambda r: r['lam']
                                )[f'{metric}_mean'],
                                'proc': line,
                                'gain_vs_proc': (
                                    None
                                    if line is None
                                    else best[f'{metric}_mean'] - line
                                ),
                            }
                        )
                    ylim = (cfg.output.get('ylim') or {}).get(metric)
                    name = f'{stem}_{compared}_{metric}'
                    if ylim is not None:
                        # Named, so a zoomed figure never replaces the
                        # full-range one.
                        name += '_y' + 'to'.join(
                            f'{float(v):g}'.replace('.', 'p') for v in ylim
                        )
                    figures_written += plot_regularization_ablation(
                        curves,
                        metric=metric,
                        out_path=figures / name,
                        baselines=flat,
                        reference=reference if cfg.output.native_rx else None,
                        title=(
                            str(cfg.output.title)
                            if cfg.output.get('title')
                            else None
                        ),
                        ylim=None if ylim is None else tuple(ylim),
                        formats=tuple(cfg.output.formats),
                        panel=tuple(float(v) for v in cfg.output.panel),
                        text_scale=float(cfg.output.text_scale),
                        annotate=bool(cfg.output.annotate),
                    )

        if peaks:
            # The *base* chart, not an expanded one: the figure spans
            # the rank axis, so naming it `whiten-k32` would claim a
            # rank it does not hold.
            figures_written += draw_ranks(
                cfg,
                peaks,
                reference,
                dataset,
                pairs,
                cfg.charts[0],
                figure_dir(cfg.output.figures, study, dataset),
            )

        for path in figures_written:
            if path.suffix == '.png':
                run.log({f'figure/{path.stem}': wandb.Image(str(path))})
        run.summary.update({f'native/{k}': v for k, v in reference.items()})
    finally:
        run.finish()

    if not overview:
        raise SystemExit('No cells produced; nothing was written.')

    print(
        f'\n{dataset} — {pairs_slug(pairs)}, {rate_slug(symbols)} '
        f'({rate} symbols on the channel), '
        f'{len(grid)} lambdas over {grid[0]:.3g}..{grid[-1]:.3g}'
        + (
            ''
            if bandwidths is None
            else ', each variant at '
            + (
                'its best bandwidth'
                if compare is None
                else f'bandwidth {compare:g}'
            )
        )
        + '\n'
    )
    print(
        overview_table(
            overview,
            [
                'metric',
                'n_pilots',
                'variant',
                'bw',
                'lam',
                'peak',
                'right_end',
                'proc',
                'gain_vs_proc',
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
        f'{figure_dir(cfg.output.figures, study, dataset)}/ and '
        f'{len(variants)} CSVs per cell under {results}/'
    )


if __name__ == '__main__':
    main()

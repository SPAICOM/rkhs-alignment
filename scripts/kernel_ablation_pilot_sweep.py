"""Ablation of RKA over the pilot budget.

Figure (ii), with RKA's two ablations against it instead of the neural
baselines:

- ``rkhs``      -- RKA itself, *read back* from ``pilot_sweep.py``'s
  records and never refitted;
- ``rkhs_free`` -- Procrustes plus kernel ridge on its residual, without
  the orthogonality constraint;
- ``krr``       -- pure kernel alignment: no Procrustes, no constraint.

The fitting is ``pilot_sweep.py``'s own ``run_seed``, restricted to the
methods in ``fit``: the same pool per seed, the same herding pass, and
each ablation's lambda read back per budget from its own lambda sweep, as
RKA's was. The reused records must cover every (design, budget, seed) of
the run, and Procrustes is refitted to check that the pilots really are
the ones RKA saw -- its scores have to match the reused ones exactly.

Records land under ``kernel_ablation/`` as one CSV per seed, so ``resume=true``
skips seeds already on disk and ``plot_only=true`` redraws from them.

Examples
--------
    just kernel-ablation-pilot-sweep
    just kernel-ablation-pilot-sweep plot_only=true
    uv run scripts/kernel_ablation_pilot_sweep.py 'seeds=[0,1]'
"""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

import wandb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pilot_sweep import (
    aggregate,
    lambda_schedules,
    native_ceiling,
    read_records,
    run_seed,
    summary_table,
    write_records,
)

from src.experiment import (
    build_decoder,
    check_budgets_fit_pool,
    describe_pairs,
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
    strategies_slug,
)

if TYPE_CHECKING:
    from src.decoder import Decoder

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = str(ROOT / 'config' / 'hydra')

# The columns that identify one fitted cell of a pilot sweep.
_CELL: tuple[str, ...] = ('method', 'strategy', 'n_pilots', 'seed')


def reused_records(
    directory: Path,
    context: dict[str, Any],
    methods: list[str],
    strategies: list[str],
    counts: list[int],
    seeds: list[int],
) -> list[dict[str, Any]]:
    """The cells of ``methods`` another pilot sweep already fitted.

    Found by their columns, not their filenames, and required to be
    complete and unambiguous: a missing cell would silently thin a band,
    and two copies of one cell mean two runs wrote the same thing under
    different names and nothing here can tell which is right.
    """
    wanted = {
        (m, s, n, k)
        for m in methods
        for s in strategies
        for n in counts
        for k in seeds
    }
    found: dict[tuple, dict[str, Any]] = {}
    for path in sorted(directory.glob('pilots_*_seed*.csv')):
        for row in read_records([path]):
            if any(row.get(key) != value for key, value in context.items()):
                continue
            cell = tuple(row[key] for key in _CELL)
            if cell not in wanted:
                continue
            if cell in found:
                raise SystemExit(
                    f'Cell {cell} appears in more than one record under '
                    f'{directory}/; remove the stale one.'
                )
            found[cell] = row
    missing = sorted(wanted - set(found))
    if missing:
        raise SystemExit(
            f'{len(missing)} cells of {methods} are missing under '
            f'{directory}/, e.g. {missing[:3]}. Run `just pilot-sweep` on '
            'the same axes first.'
        )
    return list(found.values())


def check_same_pilots(
    fitted: list[dict[str, Any]],
    reused: list[dict[str, Any]],
    method: str,
    metrics: list[str],
) -> None:
    """Stop unless ``method`` scores identically in both record sets.

    A deterministic method fitted on the same pilots gives the same
    numbers, so any gap means the pilots, the chart or the decoder
    differ from the run being reused.
    """
    reference = {
        tuple(r[key] for key in _CELL): r
        for r in reused
        if r['method'] == method
    }
    worst = 0.0
    for row in fitted:
        if row['method'] != method:
            continue
        other = reference[tuple(row[key] for key in _CELL)]
        for metric in metrics:
            worst = max(worst, abs(float(row[metric]) - float(other[metric])))
    if worst > 1e-9:
        raise SystemExit(
            f'{method} refitted here differs from its reused records by up '
            f'to {worst:.3g}: the pilots, the chart or the decoder do not '
            'match the run being reused.'
        )
    log.info('%s matches its reused records exactly.', method)


@hydra.main(
    version_base=None,
    config_path=CONFIG_DIR,
    config_name='kernel_ablation_pilot_sweep',
)
def main(cfg: DictConfig) -> None:
    """Fit the ablations per seed, merge in RKA's records, draw figure."""
    logging.getLogger('src').setLevel(logging.INFO)
    log.info('Configuration:\n%s', OmegaConf.to_yaml(cfg))

    metrics = list(cfg.eval.metrics)
    fit = [str(m) for m in cfg.fit]
    reuse = [str(m) for m in cfg.reuse.methods]
    show = [str(m) for m in cfg.show]
    unknown = [m for m in (*fit, *reuse) if m not in cfg.methods]
    if unknown:
        raise SystemExit(f'{unknown} are not configured methods.')
    unshown = [m for m in show if m not in (*fit, *reuse)]
    if unshown:
        raise SystemExit(f'show={unshown} is neither fitted nor reused.')
    seeds = [int(s) for s in cfg.seeds]
    strategies = [str(s) for s in cfg.pilots.strategies]

    dataset = str(cfg.data.get('dataset', cfg.data.source))
    pairs, agents = resolve_star(cfg)
    log.info(
        'Alignment direction (receiver %s):\n%s',
        'pinned' if cfg.get('receiver') else 'resolved by width',
        describe_pairs(pairs, agents),
    )

    decoders: dict[str, Decoder | None] = {}
    if cfg.decoder.enabled and not cfg.plot_only:
        for target in {t for _, t in pairs}:
            decoders[target] = build_decoder(cfg, agents[target]['train'])
    reference = native_accuracy(pairs, agents, decoders) if decoders else {}

    for name in ('src.alignment.base', 'src.alignment.preprocessing'):
        logging.getLogger(name).setLevel(logging.ERROR)

    d_r, max_rank = usable_rank(pairs, agents)
    symbols = resolve_symbols(cfg, d_r, max_rank)
    charts = resolve_charts(cfg, max_rank)
    plan = resolve_chart_budgets(
        cfg, charts, cfg.methods[fit[0]], symbols, max_rank
    )
    counts = sorted({n for *_, budgets in plan for n in budgets})
    check_budgets_fit_pool(
        counts,
        cfg.pilots.pool_size,
        min(agents[src]['train'].n_points for src, _ in pairs),
    )

    root = cfg.output.results
    results = result_dir(root, str(cfg.output.study), dataset)
    lam_results = result_dir(root, str(cfg.lam_schedule.study), dataset)
    reuse_results = result_dir(root, str(cfg.reuse.study), dataset)
    use_project_style(ROOT / 'config' / 'plotting' / 'plt.mplstyle')

    # `run_seed` fits every entry of `methods`, so the view holds the
    # fitted methods alone.
    view = OmegaConf.merge(cfg, {})
    with open_dict(view):
        view.methods = {name: cfg.methods[name] for name in fit}

    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.name
        or f'kernel-ablation-pilots-{dataset}-{pairs_slug(pairs)}-'
        f'{rate_slug(symbols)}',
        group=cfg.wandb.group,
        job_type='kernel-ablation-pilot-sweep',
        tags=list(cfg.wandb.tags),
        mode=cfg.wandb.mode,
        config=OmegaConf.to_container(cfg, resolve=True)
        | {'resolved_counts': counts, 'resolved_seeds': seeds, 'd_r': d_r},
    )

    written: list[Path] = []
    try:
        for chart, rank, _chart_rate, budgets in plan:
            tag = chart_slug(chart)
            stem = pilot_stem(dataset, pairs, chart, rank, budgets, strategies)
            fitted_stem = f'{stem}_{strategies_slug(fit)}'
            context = {
                'dataset': dataset,
                'pairs': pairs_slug(pairs),
                'chart': tag,
                'symbols': rank,
                'native_accuracy': reference.get('accuracy'),
            }

            schedules: dict[str, dict[int, float]] = {}
            if cfg.lam_schedule.enabled and not cfg.plot_only:
                schedules = lambda_schedules(
                    view, lam_results, pairs_slug(pairs), tag, rank
                )

            for seed in seeds:
                path = results / f'{fitted_stem}_seed{seed}.csv'
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
                    '=== chart=%s  %s  seed=%d ===', tag, rate_slug(rank), seed
                )
                write_records(
                    path,
                    run_seed(
                        view,
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

            paths = sorted(results.glob(f'{fitted_stem}_seed*.csv'))
            if not paths:
                log.warning('No records for chart=%s; nothing to draw.', tag)
                continue
            fitted = read_records(paths)
            done = sorted({int(r['seed']) for r in fitted})
            reused = reused_records(
                reuse_results,
                {
                    'dataset': dataset,
                    'pairs': pairs_slug(pairs),
                    'chart': tag,
                    'symbols': rank,
                },
                reuse,
                strategies,
                budgets,
                done,
            )
            for method in set(fit) & set(reuse):
                check_same_pilots(fitted, reused, method, metrics)

            records = [r for r in reused if r['method'] in show] + [
                r
                for r in fitted
                if r['method'] in show and r['method'] not in reuse
            ]
            summary = aggregate(records, metrics, show)
            shown = f'{stem}_{strategies_slug(show)}'

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

            print(
                f'\n{tag} — seeds {", ".join(map(str, done))}, '
                f'{len(records)} cells:\n'
            )
            print(summary_table(summary, metrics))
            coverage = {r['n_seeds'] for r in summary}
            if len(coverage) > 1 or len(done) != len(seeds):
                print(
                    f'\nWarning: {len(done)} of {len(seeds)} seeds on disk, '
                    f'coverage {sorted(coverage)} across cells.'
                )

            table = wandb.Table(columns=list(summary[0]))
            for row in summary:
                table.add_data(*row.values())
            run.log({f'curves/{tag}': table})

        for path in written:
            if path.suffix == '.png':
                run.log({f'figure/{path.stem}': wandb.Image(str(path))})
    finally:
        run.finish()

    if not written:
        raise SystemExit('No figures produced; nothing was written.')
    print('\nWrote: ' + ', '.join(str(p) for p in written))


if __name__ == '__main__':
    main()

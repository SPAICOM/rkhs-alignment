"""RKA against the alignment methods of the literature, at a fixed budget.

The companion to ``pilot_sweep.py``, with the axes swapped. There the
pilot budget varies and the method field is small; here the budget is
*pinned* and the whole field of published alignment methods is run on it,
so what the figure ranks is the methods themselves rather than their
sample efficiency.

Each method is calibrated with the pilot design its own preset declares
(``pilot_strategy``) -- kernel herding for RKA and the neural baselines,
k-means for the anchor-based equalizers, farthest-point sampling for the
linear classes -- so every method is measured in the configuration its
authors intended. Methods sharing a design share the pilot set exactly.
Setting ``pilots.strategy`` overrides all of them and puts the field on
one shared set, which is the stricter comparison: it isolates the map
from the calibration design.

The budget has to be generous enough for every method in the field to be
operating, and RKA sets that floor: its residual lives in
``range(K) n ker(X)``, so it has ``rank(K) - rank(X)`` degrees of freedom
-- roughly ``N - d_src`` -- and needs the budget to clear about twice the
wider latent dimension before the correction has as many directions to
spend as the rigid stage. Below that the comparison is measuring
Procrustes under two different names.

The field, and where each comes from:

===============  =====================================================
 method           reference
===============  =====================================================
 ``rkhs``         Residual Kernel Alignment (ours)
 ``procrustes``   the ``ortho`` class; Maiorca et al. 2024
 ``linear``       the ``linear`` class; Maiorca et al. 2024
 ``l_ortho``      the ``l-ortho`` class; Maiorca et al. 2024
 ``affine``       the ``affine`` class; Maiorca et al. 2024
 ``cca``          canonical-correlation alignment
 ``rr``           inverse relative projection; Maiorca et al. 2025
 ``ppfe``         Parseval Frame Equalizer; Fiorellino et al. 2026
===============  =====================================================

Each is a preset under ``config/hydra/alignment/``, so the field is edited
from the command line rather than in code.

Examples
--------
    uv run scripts/method_comparison.py
    uv run scripts/method_comparison.py pilots.strategy=herding
    uv run scripts/method_comparison.py data=semasia_mnist pilots.n_pilots=1536
    uv run scripts/method_comparison.py '~methods.affine' '~methods.cca'
    uv run scripts/method_comparison.py -m pilots.n_pilots=2048,3072,4096
"""

from __future__ import annotations

import csv
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

import wandb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.alignment import Aligner, alignment_metrics, select_pilots
from src.experiment import (
    apply_symbol_budget,
    available_agents,
    build_aligner,
    build_decoder,
    load_pair_data,
    method_pilot_strategy,
    resolve_pairs,
)
from src.plotting import plot_method_comparison, use_project_style

if TYPE_CHECKING:
    from src.decoder import Decoder
    from src.latent import LatentSpace

log = logging.getLogger(__name__)

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / 'config' / 'hydra')
ROOT = Path(__file__).resolve().parents[1]


def resolve_strategies(cfg: DictConfig) -> dict[str, str]:
    """The pilot design each method is calibrated with.

    ``pilots.strategy`` is the override: set it and every method is fitted
    on one shared pilot set, which isolates the map from the calibration
    design and makes the ranking a like-for-like comparison of the maps
    alone. Left null -- the default -- each method uses the design its own
    preset declares, so every method is measured in the configuration its
    authors intended rather than under somebody else's.

    The second reading is the fairer one for a published comparison and
    the noisier one to interpret: two methods can then differ both in
    their map and in the pilots that map was fitted on.
    """
    override = cfg.pilots.get('strategy')
    if override:
        return {name: str(override) for name in cfg.methods}

    fallback = str(cfg.pilots.fallback_strategy)
    strategies = {
        name: method_pilot_strategy(method_cfg, fallback)
        for name, method_cfg in cfg.methods.items()
    }
    for name, strategy in strategies.items():
        if not cfg.methods[name].get('pilot_strategy'):
            log.warning(
                'Method %s declares no `pilot_strategy`; falling back to '
                '%r. Add one to its preset under config/hydra/alignment/.',
                name,
                strategy,
            )
    log.info(
        'Per-method pilot designs: %s',
        ', '.join(f'{n}={s}' for n, s in sorted(strategies.items())),
    )
    return strategies


def pilot_sets(
    cfg: DictConfig,
    pairs: list[tuple[str, str]],
    agents: dict[str, dict[str, LatentSpace]],
    strategies: dict[str, str],
) -> dict[tuple[str, str, int], np.ndarray]:
    """One pilot set per ``(source, strategy, repeat)``.

    Keyed by strategy rather than by method, so the methods that share a
    design -- every linear class uses farthest-point sampling, both
    anchor-based methods use k-means -- are fitted on the *same* samples
    and stay directly comparable with each other. Selection still happens
    once, up front, outside the method loop.
    """
    sets: dict[tuple[str, str, int], np.ndarray] = {}
    for repeat in range(int(cfg.pilots.n_repeats)):
        seed = int(cfg.seed) + 1000 * repeat
        for source, _ in pairs:
            train = agents[source]['train']
            budget = int(cfg.pilots.n_pilots or train.n_points)
            for strategy in set(strategies.values()):
                if (source, strategy, repeat) in sets:
                    continue
                sets[(source, strategy, repeat)] = select_pilots(
                    train.latent,
                    n_pilots=budget,
                    strategy=strategy,
                    labels=train.labels,
                    seed=seed,
                )
    return sets


def evaluate(
    cfg: DictConfig,
    method_cfg: DictConfig,
    pilots: np.ndarray,
    agents: dict[str, dict[str, LatentSpace]],
    pair: tuple[str, str],
    decoder: Decoder | None,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Fit one method on one pilot set and score it on the test split."""
    source, target = pair
    src_train, src_test = agents[source]['train'], agents[source]['test']
    tgt_train, tgt_test = agents[target]['train'], agents[target]['test']

    aligner: Aligner = build_aligner(method_cfg)
    labels = src_train.labels
    # Only the pilots are exchanged; each device standardises its own
    # space from everything it holds locally, which costs no airtime.
    aligner.fit(
        src_train.latent[pilots],
        tgt_train.latent[pilots],
        labels=None if labels is None else labels[pilots],
        src_context=src_train.latent,
        tgt_context=tgt_train.latent,
    )
    scores = alignment_metrics(
        aligner.transform(src_test.latent),
        tgt_test.latent,
        decoder=decoder,
        labels=tgt_test.labels,
        ks=tuple(cfg.eval.topk),
    )
    return scores, aligner.summary()


def run_field(
    cfg: DictConfig,
    pairs: list[tuple[str, str]],
    agents: dict[str, dict[str, LatentSpace]],
    decoders: dict[str, Decoder | None],
    sets: dict[tuple[str, str, int], np.ndarray],
    strategies: dict[str, str],
    metrics: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Score every method on every ``(pair, repeat)`` cell.

    Returns the per-cell records and the fit diagnostics each aligner
    reports, which is where the method-specific health checks live --
    ``rr_projector_cond``, ``cca_min_correlation``,
    ``linear_orthogonality``, ``rkhs_lam``.

    Each record also carries what the fit *cost*: ``paired_used`` is the
    paired samples the map actually consumed, which is well below the
    budget for anchor-based methods, and ``map_params`` is the size of
    the deployed map. Neither is a metric, but a ranking that ignores
    them is comparing methods at different prices.
    """
    records: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []

    for name, method_cfg in cfg.methods.items():
        strategy = strategies[name]
        for repeat in range(int(cfg.pilots.n_repeats)):
            for source, target in pairs:
                pilots = sets[(source, strategy, repeat)]
                scores, summary = evaluate(
                    cfg,
                    method_cfg,
                    pilots,
                    agents,
                    (source, target),
                    decoders.get(target),
                )
                records.append(
                    {
                        'method': name,
                        'strategy': strategy,
                        'repeat': repeat,
                        'source': source,
                        'target': target,
                        'n_pilots': int(pilots.size),
                        # What the method actually spent, as opposed to
                        # what it was offered: see `Aligner`.
                        'paired_used': int(summary['paired_samples_used']),
                        'map_params': int(summary['map_parameters']),
                        'symbols': int(summary['transmitted_symbols']),
                        **{m: scores[m] for m in metrics},
                    }
                )
                diagnostics.append(
                    {
                        'method': name,
                        'strategy': strategy,
                        'repeat': repeat,
                        'source': source,
                        'target': target,
                        **{
                            k: v
                            for k, v in summary.items()
                            if isinstance(v, (int, float, str, bool))
                        },
                    }
                )
        cells = [r for r in records if r['method'] == name]
        log.info(
            'method %s (pilots: %s) done over %d cells: %s',
            name,
            strategy,
            len(cells),
            ', '.join(
                f'{m}={np.mean([c[m] for c in cells]):.4f}' for m in metrics
            ),
        )
    return records, diagnostics


def aggregate(
    records: list[dict[str, Any]],
    strategies: dict[str, str],
    metrics: list[str],
) -> list[dict[str, Any]]:
    """Average over TX-RX pairs within a repeat, then over repeats.

    Averaging pairs first means the reported spread is the variability of
    the *experiment* (which pilots were drawn), not of the pair mix.
    """
    per_repeat: dict[tuple, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in records:
        for metric in metrics:
            per_repeat[(row['method'], row['repeat'])][metric].append(
                row[metric]
            )

    # Cost is a property of the method, not of the metric, so it is
    # carried through rather than averaged into the spread.
    cost: dict[str, dict[str, int]] = {}
    for row in records:
        seen = cost.setdefault(
            row['method'],
            {'symbols': 0, 'paired_used': 0, 'map_params': 0},
        )
        seen['symbols'] = max(seen['symbols'], row['symbols'])
        seen['paired_used'] = max(seen['paired_used'], row['paired_used'])
        seen['map_params'] = max(seen['map_params'], row['map_params'])

    by_method: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for (method, _), values in per_repeat.items():
        for metric in metrics:
            by_method[method][metric].append(float(np.mean(values[metric])))

    summary = []
    for method, values in by_method.items():
        row: dict[str, Any] = {
            'method': method,
            'strategy': strategies[method],
            'symbols': cost[method]['symbols'],
            'paired_used': cost[method]['paired_used'],
            'map_params': cost[method]['map_params'],
            'n_repeats': len(next(iter(values.values()))),
        }
        for metric in metrics:
            row[f'{metric}_mean'] = float(np.mean(values[metric]))
            row[f'{metric}_std'] = float(np.std(values[metric]))
        summary.append(row)
    return sorted(summary, key=lambda r: -r[f'{metrics[0]}_mean'])


def paired_deltas(
    records: list[dict[str, Any]],
    metric: str,
    highlight: str,
) -> dict[str, tuple[float, float, int, int]]:
    """Per-cell differences against ``highlight``: mean, sd, wins, cells.

    Every method is fitted on the same pilots and scored on the same test
    rows, so the difference between two of them is a *paired* quantity
    and carries far less variance than either measurement does on its
    own. Comparing each against the marginal noise floor throws that
    away: a 0.003 gap that holds in six draws out of six is a real
    ordering, even though 0.003 sits inside the 0.005 binomial error on
    any single accuracy.
    """
    cells: dict[tuple, dict[str, float]] = defaultdict(dict)
    for row in records:
        key = (row['repeat'], row['source'], row['target'])
        cells[key][row['method']] = row[metric]

    gaps: dict[str, list[float]] = defaultdict(list)
    for scores in cells.values():
        if highlight not in scores:
            continue
        for method, value in scores.items():
            gaps[method].append(value - scores[highlight])

    return {
        method: (
            float(np.mean(g)),
            float(np.std(g)),
            int(sum(x > 0 for x in g)),
            len(g),
        )
        for method, g in gaps.items()
    }


def accuracy_noise_floor(
    pairs: list[tuple[str, str]],
    agents: dict[str, dict[str, LatentSpace]],
) -> float:
    """One standard error on a test-split accuracy, at p = 0.5.

    Accuracy is a mean over test rows, so it carries sampling error even
    when the fit is perfectly deterministic -- and with a deterministic
    pilot design and one repeat, the reported standard deviation is
    exactly zero, which invites reading a 0.0002 gap as a ranking. The
    binomial standard error is the honest floor: two methods closer
    together than this are tied, however many decimal places the table
    prints.

    Evaluated at p = 0.5 so it is a *bound* rather than a per-method
    figure -- the true error shrinks as accuracy approaches 1.
    """
    n = min(agents[target]['test'].n_points for _, target in pairs)
    return 0.5 / np.sqrt(n)


def native_reference(
    pairs: list[tuple[str, str]],
    agents: dict[str, dict[str, LatentSpace]],
    decoders: dict[str, Decoder | None],
) -> dict[str, float]:
    """The receivers' own accuracy on their own latents -- the ceiling."""
    scores = [
        decoders[target].score(
            agents[target]['test'].latent, agents[target]['test'].labels
        )
        for _, target in pairs
        if decoders.get(target) is not None
    ]
    return {'accuracy': float(np.mean(scores))} if scores else {}


def format_table(
    summary: list[dict[str, Any]],
    metrics: list[str],
    highlight: str,
    noise_floor: float = 0.0,
    deltas: dict[str, tuple[float, float, int, int]] | None = None,
) -> str:
    """Fixed-width ranking, with each method's gap to ``highlight``.

    The gap column is the *paired* difference against ``highlight`` when
    ``deltas`` is given, with the number of cells it held in -- 6/6 is an
    ordering, 3/6 is a coin flip. Without it the gap is a difference of
    means and anything inside ``noise_floor`` is marked ``~``.
    """
    baseline = next(
        (r for r in summary if r['method'] == highlight),
        None,
    )
    columns = [
        'method',
        'strategy',
        'symbols',
        'paired_used',
        'map_params',
    ] + [f'{m}_{stat}' for m in metrics for stat in ('mean', 'std')]
    if baseline is not None:
        columns.append(f'd_{metrics[0]}')

    cells = []
    for row in summary:
        line = [
            f'{row[c]:.4f}' if isinstance(row[c], float) else str(row[c])
            for c in columns
            if c in row
        ]
        if baseline is not None:
            entry = (deltas or {}).get(row['method'])
            if entry is not None and entry[3] > 1:
                mean, sd, wins, total = entry
                # Two standard errors on the paired mean: a gap that
                # clears it is an ordering, not a draw.
                tie = '' if abs(mean) > 2 * sd / np.sqrt(total) else '~'
                line.append(f'{mean:+.4f}{tie} {wins}/{total}')
            else:
                gap = (
                    row[f'{metrics[0]}_mean'] - baseline[f'{metrics[0]}_mean']
                )
                tie = '~' if abs(gap) < noise_floor else ''
                line.append(f'{gap:+.4f}{tie}')
        cells.append(line)

    widths = [
        max(len(c), *(len(row[i]) for row in cells))
        for i, c in enumerate(columns)
    ]
    head = '  '.join(c.ljust(w) for c, w in zip(columns, widths))
    rule = '  '.join('-' * w for w in widths)
    body = [
        '  '.join(v.ljust(w) for v, w in zip(row, widths)) for row in cells
    ]
    return '\n'.join([head, rule, *body])


def write_csv(rows: list[dict[str, Any]], path: Path) -> Path:
    """Dump a list of flat records, unioning their keys."""
    fields: list[str] = []
    for row in rows:
        fields.extend(k for k in row if k not in fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, restval='')
        writer.writeheader()
        writer.writerows(rows)
    return path


@hydra.main(
    version_base=None, config_path=CONFIG_DIR, config_name='method_comparison'
)
def main(cfg: DictConfig) -> None:
    """Run the whole method field at one pilot budget and write the figure."""
    logging.getLogger('src').setLevel(logging.INFO)
    log.info('Configuration:\n%s', OmegaConf.to_yaml(cfg))

    metrics = list(cfg.eval.metrics)
    highlight = str(cfg.highlight)

    # Rate-match before anything is built, so every method is offered the
    # same number of channel symbols and the ranking is a like-for-like
    # comparison at a fixed rate rather than at a fixed hyper-parameter.
    applied = apply_symbol_budget(cfg)
    if applied:
        log.info(
            'Channel rate pinned to %s symbols: %s',
            cfg.symbols,
            ', '.join(
                f'{n}.{cfg.methods[n].rate_key}' for n in sorted(applied)
            ),
        )

    pairs = resolve_pairs(cfg, available_agents(cfg))
    agents = load_pair_data(cfg, pairs)

    decoders: dict[str, Decoder | None] = {}
    if cfg.decoder.enabled:
        for target in {t for _, t in pairs}:
            decoders[target] = build_decoder(cfg, agents[target]['train'])

    # A pinned budget below the latent dimension is rank-deficient by
    # design -- that regime is the subject of the study, not a
    # misconfiguration, so the per-fit warnings about it are silenced.
    for name in ('src.alignment.base', 'src.alignment.preprocessing'):
        logging.getLogger(name).setLevel(logging.ERROR)

    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.name
        or f'methods-{cfg.data.get("dataset", cfg.data.source)}',
        group=cfg.wandb.group,
        job_type='method-comparison',
        tags=list(cfg.wandb.tags),
        mode=cfg.wandb.mode,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    try:
        strategies = resolve_strategies(cfg)
        sets = pilot_sets(cfg, pairs, agents, strategies)
        budget = int(next(iter(sets.values())).size)
        records, diagnostics = run_field(
            cfg, pairs, agents, decoders, sets, strategies, metrics
        )
        summary = aggregate(records, strategies, metrics)
        reference = native_reference(pairs, agents, decoders)
        floor = accuracy_noise_floor(pairs, agents)
        deltas = paired_deltas(records, metrics[0], highlight)

        use_project_style(ROOT / 'config' / 'plotting' / 'plt.mplstyle')
        dataset = cfg.data.get('dataset', cfg.data.source)
        default_title = (
            f'Alignment methods at $N = {budget}$ semantic pilots — {dataset}'
        )
        out = Path(cfg.output_dir) / f'method_comparison_{dataset}_n{budget}'
        # Persist the aggregate so the figure can be restyled, and the
        # per-cell records so the ranking can be re-tested, without
        # paying for the sweep again.
        written_csv = [
            write_csv(summary, out.with_suffix('.csv')),
            write_csv(records, out.with_name(out.name + '_records.csv')),
            write_csv(
                diagnostics, out.with_name(out.name + '_diagnostics.csv')
            ),
        ]
        written = plot_method_comparison(
            summary,
            metrics=metrics,
            out_path=out,
            reference=reference,
            highlight=highlight,
            title=cfg.plot_title or default_title,
        )

        table = wandb.Table(columns=list(summary[0]))
        for row in summary:
            table.add_data(*row.values())
        run.log({'method_comparison': table})
        for path in written:
            if path.suffix == '.png':
                run.log({'figure': wandb.Image(str(path))})
        run.summary.update(
            {f'native/{k}': v for k, v in reference.items()}
            | {
                f'{row["method"]}/{m}': row[f'{m}_mean']
                for row in summary
                for m in metrics
            }
            | {
                'accuracy_noise_floor': floor,
                'n_pilots': budget,
                'n_pairs': len(pairs),
                'n_records': len(records),
            }
        )
    finally:
        run.finish()

    print(f'\nFixed pilot budget: N = {budget}')
    print('\n' + format_table(summary, metrics, highlight, floor, deltas))
    cells = max((d[3] for d in deltas.values()), default=0)
    if cells > 1:
        print(
            f'\nGap column is the paired difference against {highlight} over '
            f'{cells} (repeat, pair) cells, with how many it held in. '
            '~ marks a gap inside two standard errors of that mean.'
        )
    else:
        print(
            f'\n~ marks a gap inside the {floor:.4f} accuracy noise floor '
            '(one binomial standard error on the test split). With a single '
            'cell the paired test is unavailable, so this bound is '
            'conservative: raise pilots.n_repeats to resolve smaller gaps.'
        )
    if reference:
        print(
            '\nNative RX ceiling: '
            + '  '.join(f'{k}={v:.4f}' for k, v in reference.items())
        )
    print('\nWrote: ' + ', '.join(str(p) for p in [*written, *written_csv]))


if __name__ == '__main__':
    main()

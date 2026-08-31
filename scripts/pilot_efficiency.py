"""Sample efficiency of alignment methods versus the semantic-pilot budget.

Reproduces Figs. 2 and 3 of the RKA paper on SEMASIA latents: how well does
each alignment method transport a transmitter's latents into a receiver's
space when it is only allowed ``N`` paired calibration samples, and how
much does it matter *which* ``N`` are spent?

For every ``(pilot strategy, N, repeat, TX->RX pair)`` the script draws a
pilot set from the TX training pool, fits every configured method on those
pilots alone, and scores the map on the held-out test split. Results are
averaged over the pairs within each repeat, then reported as mean +/- one
standard deviation over repeats.

Two factors vary independently, so one run covers either paper figure:

  Fig. 2  methods={procrustes, rkhs}, pilots.strategies=[random, herding]
  Fig. 3  methods={procrustes, rkhs, direct_mlp, residual_mlp},
          pilots.strategies=[herding]

Examples
--------
    uv run scripts/pilot_efficiency.py
    uv run scripts/pilot_efficiency.py data=semasia_mnist
    uv run scripts/pilot_efficiency.py 'pilots.counts=[25,50,100,200]'
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

from src.alignment import Aligner, alignment_metrics, select_pilot_path
from src.experiment import (
    available_agents,
    build_aligner,
    build_decoder,
    load_pair_data,
    resolve_pairs,
)
from src.plotting import plot_pilot_efficiency, use_project_style

if TYPE_CHECKING:
    from src.decoder import Decoder
    from src.latent import LatentSpace

log = logging.getLogger(__name__)

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / 'config' / 'hydra')
ROOT = Path(__file__).resolve().parents[1]

# Strategies whose answer depends on the seed; the rest are
# deterministic and can be reused across repeats of a fixed pool.
_RANDOMISED = frozenset({'random', 'kmeans', 'stratified'})


def draw_pool(
    space: LatentSpace, pool_size: int | None, seed: int
) -> np.ndarray:
    """Row indices of the candidate pool this repeat selects pilots from.

    ``pool_size=None`` (the default) means the whole training split: the
    transmitter picks its pilots from everything it has. Setting an
    integer re-draws a smaller pool per repeat instead, which is the only
    way the *deterministic* strategies (kernel herding, farthest-point)
    acquire an error bar -- over a fixed pool they return the same pilots
    every time.
    """
    n = space.n_points
    if pool_size is None or pool_size >= n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=pool_size, replace=False))


def evaluate_pilots(
    cfg: DictConfig,
    method_name: str,
    method_cfg: DictConfig,
    pilots: np.ndarray,
    agents: dict[str, dict[str, LatentSpace]],
    pair: tuple[str, str],
    decoder: Decoder | None,
) -> dict[str, float]:
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
    return alignment_metrics(
        aligner.transform(src_test.latent),
        tgt_test.latent,
        decoder=decoder,
        labels=tgt_test.labels,
        ks=tuple(cfg.eval.topk),
    )


def run_sweep(
    cfg: DictConfig,
    pairs: list[tuple[str, str]],
    agents: dict[str, dict[str, LatentSpace]],
    decoders: dict[str, Decoder | None],
) -> list[dict[str, Any]]:
    """Every (strategy, N, repeat, pair, method) cell of the study.

    The pilot selection sits outside the method loop so each pilot set is
    chosen once and every method is scored on exactly the same samples --
    which is what makes the methods comparable at a given budget.
    """
    metrics = list(cfg.eval.metrics)
    records: list[dict[str, Any]] = []

    # Small pilot budgets make the whitening and the cross-covariance
    # rank-deficient by construction -- that regime is the subject of the
    # study, not a misconfiguration, so the per-fit warnings about it are
    # silenced for the sweep rather than repeated a thousand times.
    log.info(
        'Latent dims run to %d; budgets below that are rank-deficient by '
        'design. Silencing the per-fit rank warnings for the sweep.',
        max(s['train'].dim for s in agents.values()),
    )
    noisy = [
        logging.getLogger('src.alignment.base'),
        logging.getLogger('src.alignment.preprocessing'),
    ]
    previous = [logger.level for logger in noisy]
    for logger in noisy:
        logger.setLevel(logging.ERROR)

    try:
        records = _sweep_cells(cfg, pairs, agents, decoders, metrics)
    finally:
        for logger, level in zip(noisy, previous):
            logger.setLevel(level)
    return records


def _sweep_cells(
    cfg: DictConfig,
    pairs: list[tuple[str, str]],
    agents: dict[str, dict[str, LatentSpace]],
    decoders: dict[str, Decoder | None],
    metrics: list[str],
) -> list[dict[str, Any]]:
    """Inner loop of :func:`run_sweep`."""
    counts = [int(c) for c in cfg.pilots.counts]
    records: list[dict[str, Any]] = []
    paths: dict[tuple, dict[int, np.ndarray]] = {}

    for strategy in cfg.pilots.strategies:
        strategy = str(strategy)
        # A deterministic strategy over a fixed pool returns the same
        # pilots every repeat, so repeating it would re-fit identical
        # models to produce a band that is zero by construction.
        deterministic = (
            strategy not in _RANDOMISED and cfg.pilots.pool_size is None
        )
        n_repeats = 1 if deterministic else int(cfg.pilots.n_repeats)
        if deterministic:
            log.info(
                'Strategy %s is deterministic over the full pool; running '
                'a single repeat instead of %s.',
                strategy,
                cfg.pilots.n_repeats,
            )

        for repeat in range(n_repeats):
            seed = int(cfg.seed) + 1000 * repeat
            for source, target in pairs:
                src_train = agents[source]['train']
                pool = draw_pool(src_train, cfg.pilots.pool_size, seed)
                labels = src_train.labels

                # All budgets for this (source, strategy, draw) at once,
                # so a greedy strategy runs one selection instead of one
                # per budget. Deterministic strategies over a fixed pool
                # also repeat identically, so the key drops the seed.
                key = (
                    source,
                    strategy,
                    int(pool.size),
                    seed if strategy in _RANDOMISED else -1,
                )
                if key not in paths:
                    paths[key] = select_pilot_path(
                        src_train.latent[pool],
                        counts=counts,
                        strategy=strategy,
                        labels=None if labels is None else labels[pool],
                        seed=seed,
                    )

                for n_pilots, chosen in paths[key].items():
                    pilots = pool[chosen]
                    for method_name, method_cfg in cfg.methods.items():
                        scores = evaluate_pilots(
                            cfg,
                            method_name,
                            method_cfg,
                            pilots,
                            agents,
                            (source, target),
                            decoders.get(target),
                        )
                        records.append(
                            {
                                'method': method_name,
                                'strategy': strategy,
                                'n_pilots': n_pilots,
                                'n_pilots_used': int(pilots.size),
                                'repeat': repeat,
                                'source': source,
                                'target': target,
                                **{m: scores[m] for m in metrics},
                            }
                        )
            log.info(
                'strategy=%s repeat=%d/%d done (%d records).',
                strategy,
                repeat + 1,
                n_repeats,
                len(records),
            )
    return records


def aggregate(
    records: list[dict[str, Any]], metrics: list[str]
) -> list[dict[str, Any]]:
    """Average over TX-RX pairs within a repeat, then over repeats.

    Averaging pairs first means the reported spread is the variability of
    the *experiment* (which pilots were drawn), not of the pair mix.
    """
    per_repeat: dict[tuple, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in records:
        key = (row['method'], row['strategy'], row['n_pilots'], row['repeat'])
        for metric in metrics:
            per_repeat[key][metric].append(row[metric])

    by_cell: dict[tuple, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for (method, strategy, n_pilots, _), values in per_repeat.items():
        for metric in metrics:
            by_cell[(method, strategy, n_pilots)][metric].append(
                float(np.mean(values[metric]))
            )

    summary = []
    for (method, strategy, n_pilots), values in by_cell.items():
        row: dict[str, Any] = {
            'method': method,
            'strategy': strategy,
            'n_pilots': n_pilots,
            'n_repeats': len(next(iter(values.values()))),
        }
        for metric in metrics:
            row[f'{metric}_mean'] = float(np.mean(values[metric]))
            row[f'{metric}_std'] = float(np.std(values[metric]))
        summary.append(row)
    return sorted(
        summary, key=lambda r: (r['method'], r['strategy'], r['n_pilots'])
    )


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


def format_table(summary: list[dict[str, Any]], metrics: list[str]) -> str:
    """Fixed-width view of the aggregated curves (also the a11y fallback)."""
    columns = ['method', 'strategy', 'n_pilots'] + [
        f'{m}_{stat}' for m in metrics for stat in ('mean', 'std')
    ]
    cells = [
        [
            f'{r[c]:.4f}' if isinstance(r[c], float) else str(r[c])
            for c in columns
        ]
        for r in summary
    ]
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


@hydra.main(
    version_base=None, config_path=CONFIG_DIR, config_name='pilot_efficiency'
)
def main(cfg: DictConfig) -> None:
    """Run the pilot-budget sweep and write the figure."""
    logging.getLogger('src').setLevel(logging.INFO)
    log.info('Configuration:\n%s', OmegaConf.to_yaml(cfg))

    metrics = list(cfg.eval.metrics)
    pairs = resolve_pairs(cfg, available_agents(cfg))
    agents = load_pair_data(cfg, pairs)

    decoders: dict[str, Decoder | None] = {}
    if cfg.decoder.enabled:
        for target in {t for _, t in pairs}:
            decoders[target] = build_decoder(cfg, agents[target]['train'])

    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.name
        or f'pilots-{cfg.data.get("dataset", "synthetic")}',
        group=cfg.wandb.group,
        job_type='pilot-efficiency',
        tags=list(cfg.wandb.tags),
        mode=cfg.wandb.mode,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    try:
        records = run_sweep(cfg, pairs, agents, decoders)
        summary = aggregate(records, metrics)
        reference = native_reference(pairs, agents, decoders)

        use_project_style(ROOT / 'config' / 'plotting' / 'plt.mplstyle')
        dataset = cfg.data.get('dataset', cfg.data.source)
        source = cfg.data.source
        default_title = (
            f'Semantic pilots — SEMASIA {dataset}'
            if source == 'semasia'
            else f'Semantic pilots — {dataset}'
        )
        out = Path(cfg.output_dir) / f'pilot_efficiency_{dataset}'
        # Persist the aggregate so the figure can be restyled without
        # paying for the sweep again.
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.with_suffix('.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)

        written = plot_pilot_efficiency(
            summary,
            metrics=metrics,
            out_path=out,
            reference=reference,
            title=cfg.plot_title or default_title,
        )

        table = wandb.Table(columns=list(summary[0]))
        for row in summary:
            table.add_data(*row.values())
        run.log({'pilot_efficiency': table})
        for path in written:
            if path.suffix == '.png':
                run.log({'figure': wandb.Image(str(path))})
        run.summary.update(
            {f'native/{k}': v for k, v in reference.items()}
            | {'n_pairs': len(pairs), 'n_records': len(records)}
        )
    finally:
        run.finish()

    print('\n' + format_table(summary, metrics))
    if reference:
        print(
            '\nNative RX ceiling: '
            + '  '.join(f'{k}={v:.4f}' for k, v in reference.items())
        )
    print(
        '\nWrote: '
        + ', '.join(str(p) for p in [*written, out.with_suffix('.csv')])
    )


if __name__ == '__main__':
    main()

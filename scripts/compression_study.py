"""Post-alignment performance against the compression factor.

The rate–distortion view of semantic channel equalization, on one fixed
encoder pair. Every method here compresses the semantic message; they
differ in what they compress it *with*, and this study holds the
compression fixed and lets the mechanism vary.

    procrustes  transmits the truncated whitened latent -- k coefficients
                -- and the receiver applies one rigid rotation.
    cca         the same k coefficients, but the retained basis is the
                canonical one rather than the rotation: truncated
                whitening followed by CCA is SVCCA.
    rkhs        the same k coefficients, and the receiver applies the
                rotation plus a kernel correction fitted on the pilots.
    ppfe        transmits frame coefficients c = F x, one per anchor, so
                its anchor count *is* k and is not a free parameter.

Two things vary, and they are not the same thing:

- ``symbols`` (x-axis) is the **rate**: coefficients per sample, paid on
  every transmission forever. The compression factor reported alongside
  is ``symbols / d_src``.
- ``pilots.counts`` is the **calibration budget**: paired samples
  exchanged once. The three coordinate methods are run at each budget;
  PPFE is not swept over it, because its anchor count is pinned by the
  rate.

The interaction between the two is the point of the figure. RKA's
residual lives in ``range(K) n ker(X)`` and so has ``= N - k`` degrees of
freedom: raising the rate *spends* the capacity that the pilot budget
provides, and past ``k = N`` there is none left and RKA collapses onto
Procrustes exactly. PPFE has the mirror-image ceiling -- it cannot
transmit more coefficients than it has anchors to build them from. Both
show up in the results as curves that stop being distinguishable, which
is why the realised rate and the realised capacity are reported per row
rather than assumed from the config.

Examples
--------
    uv run scripts/compression_study.py
    uv run scripts/compression_study.py 'symbols=[8,16,32,64]'
    uv run scripts/compression_study.py 'pilots.counts=[100,1000]'
    uv run scripts/compression_study.py data=semasia_mnist \\
        'pairs=[{source: vit_base_patch16_224.augreg_in21k,
                 target: vit_small_patch16_224.augreg_in1k}]'
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
    available_agents,
    build_aligner,
    build_decoder,
    load_pair_data,
    method_pilot_strategy,
    resolve_pairs,
)
from src.plotting import plot_compression_study, use_project_style

if TYPE_CHECKING:
    from src.decoder import Decoder
    from src.latent import LatentSpace

log = logging.getLogger(__name__)

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / 'config' / 'hydra')
ROOT = Path(__file__).resolve().parents[1]


def draw_pilots(
    cfg: DictConfig,
    space: LatentSpace,
    budget: int,
    strategy: str,
    repeat: int,
) -> np.ndarray:
    """Pilot indices for one (budget, strategy, repeat) cell.

    The candidate pool is re-drawn per repeat so the deterministic
    designs -- herding, farthest-point, k-means over a fixed pool --
    acquire genuine variability instead of returning the same set every
    time and reporting a spread of exactly zero.
    """
    rng = np.random.default_rng(int(cfg.seed) + 1000 * repeat)
    pool_size = cfg.pilots.get('pool_size')
    pool = (
        np.arange(space.n_points)
        if not pool_size or pool_size >= space.n_points
        else np.sort(
            rng.choice(space.n_points, size=int(pool_size), replace=False)
        )
    )
    chosen = select_pilots(
        space.latent[pool],
        n_pilots=min(int(budget), pool.size),
        strategy=strategy,
        labels=None if space.labels is None else space.labels[pool],
        seed=int(cfg.seed) + 1000 * repeat,
    )
    return pool[chosen]


def evaluate(
    cfg: DictConfig,
    method_cfg: DictConfig,
    symbols: int,
    pilots: np.ndarray,
    agents: dict[str, dict[str, LatentSpace]],
    pair: tuple[str, str],
    decoder: Decoder | None,
) -> dict[str, Any]:
    """Fit one method at one rate on one pilot set, and score it."""
    source, target = pair
    src_train, src_test = agents[source]['train'], agents[source]['test']
    tgt_train, tgt_test = agents[target]['train'], agents[target]['test']

    # The rate is bought with a different knob per method; the preset
    # names its own in `rate_key`.
    fields = OmegaConf.merge(
        method_cfg, {str(method_cfg.rate_key): int(symbols)}
    )
    aligner: Aligner = build_aligner(fields)
    labels = src_train.labels
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
    summary = aligner.summary()
    # Realised, not requested: an anchor method cannot transmit more
    # coefficients than it has anchors, and RKA's capacity can be zero.
    scores['symbols'] = float(summary['transmitted_symbols'])
    scores['paired_used'] = float(summary['paired_samples_used'])
    scores['map_params'] = float(summary['map_parameters'])
    scores['capacity'] = float(summary.get('rkhs_capacity', np.nan))
    return scores


def run_sweep(
    cfg: DictConfig,
    pairs: list[tuple[str, str]],
    agents: dict[str, dict[str, LatentSpace]],
    decoders: dict[str, Decoder | None],
    metrics: list[str],
) -> list[dict[str, Any]]:
    """Every (method, budget, rate, repeat, pair) cell of the study."""
    swept = set(cfg.pilots.get('sweep_methods') or cfg.methods.keys())
    records: list[dict[str, Any]] = []

    for name, method_cfg in cfg.methods.items():
        strategy = method_pilot_strategy(
            method_cfg, str(cfg.pilots.fallback_strategy)
        )
        # A method whose rate is pinned by its own construction is run at
        # one budget only; sweeping it would repeat the same map.
        budgets = (
            [int(b) for b in cfg.pilots.counts]
            if name in swept
            else [int(cfg.pilots.counts[-1])]
        )
        for budget in budgets:
            for symbols in cfg.symbols:
                for repeat in range(int(cfg.pilots.n_repeats)):
                    for source, target in pairs:
                        pilots = draw_pilots(
                            cfg,
                            agents[source]['train'],
                            budget,
                            strategy,
                            repeat,
                        )
                        scores = evaluate(
                            cfg,
                            method_cfg,
                            int(symbols),
                            pilots,
                            agents,
                            (source, target),
                            decoders.get(target),
                        )
                        records.append(
                            {
                                'method': name,
                                'strategy': strategy,
                                'n_pilots': int(pilots.size),
                                'requested_symbols': int(symbols),
                                'repeat': repeat,
                                'source': source,
                                'target': target,
                                **{
                                    k: v
                                    for k, v in scores.items()
                                    if k in metrics
                                    or k
                                    in (
                                        'symbols',
                                        'paired_used',
                                        'map_params',
                                        'capacity',
                                    )
                                },
                            }
                        )
            log.info(
                'method=%-11s N=%-5d done (%d rates x %d repeats)',
                name,
                budget,
                len(cfg.symbols),
                int(cfg.pilots.n_repeats),
            )
    return records


def aggregate(
    records: list[dict[str, Any]], metrics: list[str], d_src: int
) -> list[dict[str, Any]]:
    """Mean and spread over (pair, repeat) for each (method, N, rate)."""
    cells: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        cells[
            (row['method'], row['n_pilots'], row['requested_symbols'])
        ].append(row)

    summary = []
    for (method, budget, requested), rows in cells.items():
        entry: dict[str, Any] = {
            'method': method,
            'n_pilots': budget,
            'requested_symbols': requested,
            'symbols': int(np.mean([r['symbols'] for r in rows])),
            'compression': float(
                np.mean([r['symbols'] for r in rows]) / d_src
            ),
            'paired_used': int(np.mean([r['paired_used'] for r in rows])),
            'map_params': int(np.mean([r['map_params'] for r in rows])),
            'capacity': float(np.mean([r['capacity'] for r in rows])),
            'n_cells': len(rows),
        }
        for metric in metrics:
            values = [r[metric] for r in rows]
            entry[metric] = float(np.mean(values))
            entry[f'{metric}_std'] = float(np.std(values))
        summary.append(entry)
    return sorted(
        summary, key=lambda r: (r['method'], r['n_pilots'], r['symbols'])
    )


def native_reference(
    pairs: list[tuple[str, str]],
    agents: dict[str, dict[str, LatentSpace]],
    decoders: dict[str, Decoder | None],
) -> dict[str, float]:
    """The receivers' accuracy on their own latents."""
    scores = [
        decoders[target].score(
            agents[target]['test'].latent, agents[target]['test'].labels
        )
        for _, target in pairs
        if decoders.get(target) is not None
    ]
    return {'accuracy': float(np.mean(scores))} if scores else {}


def format_table(summary: list[dict[str, Any]], metrics: list[str]) -> str:
    """Fixed-width view of the sweep (also the accessibility fallback)."""
    columns = [
        'method',
        'n_pilots',
        'symbols',
        'compression',
        'paired_used',
        'capacity',
    ] + metrics
    cells = [
        [
            f'{r[c]:.4f}'
            if isinstance(r[c], float) and c != 'capacity'
            else (
                '-'
                if c == 'capacity' and r[c] != r[c]
                else f'{r[c]:.0f}'
                if c == 'capacity'
                else str(r[c])
            )
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
    version_base=None,
    config_path=CONFIG_DIR,
    config_name='compression_study',
)
def main(cfg: DictConfig) -> None:
    """Sweep the compression factor and write the figure."""
    logging.getLogger('src').setLevel(logging.INFO)
    log.info('Configuration:\n%s', OmegaConf.to_yaml(cfg))

    metrics = list(cfg.eval.metrics)
    pairs = resolve_pairs(cfg, available_agents(cfg))
    agents = load_pair_data(cfg, pairs)
    d_src = agents[pairs[0][0]]['train'].dim

    decoders: dict[str, Decoder | None] = {}
    if cfg.decoder.enabled:
        for target in {t for _, t in pairs}:
            decoders[target] = build_decoder(cfg, agents[target]['train'])

    # Low budgets and aggressive truncation are the subject here, not a
    # misconfiguration, so the per-fit rank warnings are silenced.
    for name in (
        'src.alignment.base',
        'src.alignment.preprocessing',
        'src.alignment.relative',
        'src.alignment.rkhs',
    ):
        logging.getLogger(name).setLevel(logging.ERROR)

    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.name
        or f'compression-{cfg.data.get("dataset", cfg.data.source)}',
        group=cfg.wandb.group,
        job_type='compression-study',
        tags=list(cfg.wandb.tags),
        mode=cfg.wandb.mode,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    try:
        records = run_sweep(cfg, pairs, agents, decoders, metrics)
        summary = aggregate(records, metrics, d_src)
        reference = native_reference(pairs, agents, decoders)

        use_project_style(ROOT / 'config' / 'plotting' / 'plt.mplstyle')
        dataset = cfg.data.get('dataset', cfg.data.source)
        out = Path(cfg.output_dir) / f'compression_{dataset}'
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.with_suffix('.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)

        written = plot_compression_study(
            summary,
            metrics=metrics,
            out_path=out,
            reference=reference,
            hue_of=OmegaConf.to_container(cfg.get('hue_of') or {}),
            title=cfg.plot_title
            or f'Rate vs performance — {pairs[0][0]} → {pairs[0][1]}',
        )

        table = wandb.Table(columns=list(summary[0]))
        for row in summary:
            table.add_data(*row.values())
        run.log({'compression_study': table})
        for path in written:
            if path.suffix == '.png':
                run.log({'figure': wandb.Image(str(path))})
        run.summary.update(
            {f'native/{k}': v for k, v in reference.items()}
            | {'d_src': d_src, 'n_records': len(records)}
        )
    finally:
        run.finish()

    print(f'\nTransmitter dimension d_src = {d_src}')
    print('\n' + format_table(summary, metrics))
    if reference:
        print(
            '\nNative RX ceiling: '
            + '  '.join(f'{k}={v:.4f}' for k, v in reference.items())
        )
    print(
        "\ncapacity is RKA's residual degrees of freedom (~N - symbols); "
        'at <= 0 it is Procrustes exactly.'
    )
    print(
        '\nWrote: '
        + ', '.join(str(p) for p in [*written, out.with_suffix('.csv')])
    )


if __name__ == '__main__':
    main()

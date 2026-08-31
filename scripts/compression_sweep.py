"""Alignment quality against the compression dimension, across encoder pairs.

The rate–distortion study, widened. One sweep over the number of
transmitted coefficients, run on several deliberately dissimilar encoder
pairs, with every alignment method in the comparison rate-matched to the
same budget at every point.

The rate is what the methods are held to, and they buy it differently:

===================  ==============  =========================================
 method               knob            what crosses the channel
===================  ==============  =========================================
 ``procrustes``       n_components    truncated *whitened* coordinates (KLT)
 ``pga_procrustes``   n_components    tangent coordinates on the sphere,
                                      plus one radial coefficient
 ``rkhs``             n_components    truncated whitened coordinates (KLT)
 ``pca_rkhs``         n_components    truncated coordinates, *unwhitened*
 ``cca``              n_canonical     canonical scores
 ``svcca``            n_canonical     canonical scores in the SVD subspace
 ``kcca``             n_canonical     kernel canonical scores
 ``ppfe``             n_anchors       frame coefficients, one per anchor
===================  ==============  =========================================

Two of those pairs are ablations rather than baselines. ``pca_rkhs``
strips the whitening out of RKA's Karhunen-Loeve step and keeps only the
truncation, which isolates which half of the transform the method lives
on; ``pga_procrustes`` replaces the whitened chart with the tangent space
of the unit sphere, testing whether the latents are better described by
direction than by coordinates. Both are reported alongside the method
they ablate, and share its colour in the figures.

Output is one figure per metric rather than a table: the interesting
object is a curve, and four metrics disagree about the ranking often
enough that collapsing them loses the finding. The realised rate is
plotted, not the requested one -- a canonical method cannot exceed
``min(d_src, d_tgt)`` scores, and PGA sends one coefficient more than it
is asked for.

Examples
--------
    uv run scripts/compression_sweep.py
    uv run scripts/compression_sweep.py 'symbols=[4,16,64,256]'
    uv run scripts/compression_sweep.py pilots.n_pilots=1000 pilots.n_repeats=1
    uv run scripts/compression_sweep.py '~methods.kcca'
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
from src.experiment import build_aligner, build_decoder, method_pilot_strategy
from src.latent import load_agents
from src.plotting import plot_compression_facets, use_project_style

if TYPE_CHECKING:
    from src.decoder import Decoder
    from src.latent import LatentSpace

log = logging.getLogger(__name__)

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / 'config' / 'hydra')
ROOT = Path(__file__).resolve().parents[1]


def resolve_pairs(cfg: DictConfig) -> list[dict[str, str]]:
    """The configured pairs, each with a short label for the panels."""
    pairs = [
        {
            'source': str(p['source']),
            'target': str(p['target']),
            'label': str(p.get('label') or f'{p["source"]} → {p["target"]}'),
        }
        for p in cfg.pairs
    ]
    if not pairs:
        raise ValueError('`pairs` is empty: nothing to sweep.')
    return pairs


def load_data(
    cfg: DictConfig, pairs: list[dict[str, str]]
) -> dict[str, dict[str, LatentSpace]]:
    """Load exactly the agents the pairs reference, once."""
    data_cfg = OmegaConf.to_container(cfg.data, resolve=True)
    source = data_cfg.pop('source')
    data_cfg.pop('seed', None)
    data_cfg.pop('models', None)
    needed = sorted({p[end] for p in pairs for end in ('source', 'target')})
    log.info('Loading %d agents: %s', len(needed), ', '.join(needed))
    return load_agents(source, models=needed, seed=cfg.seed, **data_cfg)


def draw_pilots(
    cfg: DictConfig, space: LatentSpace, strategy: str, repeat: int
) -> np.ndarray:
    """Pilot indices for one (strategy, repeat) cell of one transmitter.

    The pool is re-drawn per repeat so the deterministic designs --
    herding, k-means over a fixed pool -- get genuine variability rather
    than returning the same set every time and reporting a spread of
    exactly zero.
    """
    seed = int(cfg.seed) + 1000 * repeat
    rng = np.random.default_rng(seed)
    pool_size = cfg.pilots.get('pool_size')
    pool = (
        np.arange(space.n_points)
        if not pool_size or pool_size >= space.n_points
        else np.sort(
            rng.choice(space.n_points, size=int(pool_size), replace=False)
        )
    )
    budget = min(int(cfg.pilots.n_pilots or space.n_points), pool.size)
    chosen = select_pilots(
        space.latent[pool],
        n_pilots=budget,
        strategy=strategy,
        labels=None if space.labels is None else space.labels[pool],
        seed=seed,
    )
    return pool[chosen]


def evaluate(
    cfg: DictConfig,
    method_cfg: DictConfig,
    symbols: int,
    pilots: np.ndarray,
    src: dict[str, LatentSpace],
    tgt: dict[str, LatentSpace],
    decoder: Decoder | None,
) -> dict[str, float]:
    """Fit one method at one rate on one pilot set, and score it."""
    fields = OmegaConf.merge(
        method_cfg, {str(method_cfg.rate_key): int(symbols)}
    )
    aligner: Aligner = build_aligner(fields)
    labels = src['train'].labels
    # Only the pilots are exchanged; each device standardises its own
    # space from everything it holds locally, which costs no airtime.
    aligner.fit(
        src['train'].latent[pilots],
        tgt['train'].latent[pilots],
        labels=None if labels is None else labels[pilots],
        src_context=src['train'].latent,
        tgt_context=tgt['train'].latent,
    )
    scores = alignment_metrics(
        aligner.transform(src['test'].latent),
        tgt['test'].latent,
        decoder=decoder,
        labels=tgt['test'].labels,
        ks=tuple(cfg.eval.topk),
    )
    summary = aligner.summary()
    scores['symbols'] = float(summary['transmitted_symbols'])
    scores['paired_used'] = float(summary['paired_samples_used'])
    scores['map_params'] = float(summary['map_parameters'])
    scores['capacity'] = float(summary.get('rkhs_capacity', np.nan))
    return scores


def run_sweep(
    cfg: DictConfig,
    pairs: list[dict[str, str]],
    agents: dict[str, dict[str, LatentSpace]],
    decoders: dict[str, Decoder | None],
    metrics: list[str],
) -> list[dict[str, Any]]:
    """Every (pair, method, rate, repeat) cell of the study."""
    extra = ('symbols', 'paired_used', 'map_params', 'capacity')
    records: list[dict[str, Any]] = []

    for pair in pairs:
        src, tgt = agents[pair['source']], agents[pair['target']]
        for name, method_cfg in cfg.methods.items():
            strategy = method_pilot_strategy(
                method_cfg, str(cfg.pilots.fallback_strategy)
            )
            for repeat in range(int(cfg.pilots.n_repeats)):
                pilots = draw_pilots(cfg, src['train'], strategy, repeat)
                for symbols in cfg.symbols:
                    scores = evaluate(
                        cfg,
                        method_cfg,
                        int(symbols),
                        pilots,
                        src,
                        tgt,
                        decoders.get(pair['target']),
                    )
                    records.append(
                        {
                            'pair': pair['label'],
                            'source': pair['source'],
                            'target': pair['target'],
                            'method': name,
                            'strategy': strategy,
                            'requested_symbols': int(symbols),
                            'repeat': repeat,
                            'n_pilots': int(pilots.size),
                            **{
                                k: v
                                for k, v in scores.items()
                                if k in metrics or k in extra
                            },
                        }
                    )
            log.info(
                '%s | %-15s done (%d rates x %d repeats)',
                pair['label'],
                name,
                len(cfg.symbols),
                int(cfg.pilots.n_repeats),
            )
    return records


def aggregate(
    records: list[dict[str, Any]], metrics: list[str]
) -> list[dict[str, Any]]:
    """Mean and spread over repeats, per (pair, method, requested rate)."""
    cells: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        cells[(row['pair'], row['method'], row['requested_symbols'])].append(
            row
        )

    summary = []
    for (pair, method, requested), rows in cells.items():
        entry: dict[str, Any] = {
            'pair': pair,
            'method': method,
            'requested_symbols': requested,
            # The realised rate, which is what the figures plot: a
            # canonical method caps at min(d_src, d_tgt) and PGA sends an
            # extra radial coefficient.
            'symbols': round(np.mean([r['symbols'] for r in rows])),
            'paired_used': int(np.mean([r['paired_used'] for r in rows])),
            'map_params': int(np.mean([r['map_params'] for r in rows])),
            'capacity': float(np.mean([r['capacity'] for r in rows])),
            'n_repeats': len(rows),
        }
        for metric in metrics:
            values = [r[metric] for r in rows]
            entry[metric] = float(np.mean(values))
            entry[f'{metric}_std'] = float(np.std(values))
        summary.append(entry)
    return sorted(
        summary, key=lambda r: (r['pair'], r['method'], r['symbols'])
    )


def native_reference(
    pairs: list[dict[str, str]],
    agents: dict[str, dict[str, LatentSpace]],
    decoders: dict[str, Decoder | None],
) -> dict[str, float]:
    """Each receiver's accuracy on its own latents, keyed by pair label."""
    out: dict[str, float] = {}
    for pair in pairs:
        decoder = decoders.get(pair['target'])
        if decoder is None:
            continue
        test = agents[pair['target']]['test']
        out[pair['label']] = decoder.score(test.latent, test.labels)
    return out


@hydra.main(
    version_base=None,
    config_path=CONFIG_DIR,
    config_name='compression_sweep',
)
def main(cfg: DictConfig) -> None:
    """Run the sweep and write one figure per metric."""
    logging.getLogger('src').setLevel(logging.INFO)
    log.info('Configuration:\n%s', OmegaConf.to_yaml(cfg))

    metrics = list(cfg.eval.metrics)
    pairs = resolve_pairs(cfg)
    agents = load_data(cfg, pairs)

    decoders: dict[str, Decoder | None] = {}
    if cfg.decoder.enabled:
        for target in {p['target'] for p in pairs}:
            decoders[target] = build_decoder(cfg, agents[target]['train'])

    # Aggressive truncation and small budgets are the subject of the
    # study, not a misconfiguration, so the per-fit rank warnings are
    # silenced rather than repeated a few hundred times.
    for name in (
        'src.alignment.base',
        'src.alignment.preprocessing',
        'src.alignment.relative',
        'src.alignment.rkhs',
        'src.alignment.cca',
        'src.alignment.kcca',
    ):
        logging.getLogger(name).setLevel(logging.ERROR)

    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.name
        or f'compression-{cfg.data.get("dataset", cfg.data.source)}',
        group=cfg.wandb.group,
        job_type='compression-sweep',
        tags=list(cfg.wandb.tags),
        mode=cfg.wandb.mode,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    try:
        records = run_sweep(cfg, pairs, agents, decoders, metrics)
        summary = aggregate(records, metrics)
        reference = native_reference(pairs, agents, decoders)

        use_project_style(ROOT / 'config' / 'plotting' / 'plt.mplstyle')
        dataset = cfg.data.get('dataset', cfg.data.source)
        stem = Path(cfg.output_dir) / f'compression_{dataset}'
        stem.parent.mkdir(parents=True, exist_ok=True)
        with stem.with_suffix('.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)

        hue_of = OmegaConf.to_container(cfg.get('hue_of') or {})
        labels = [p['label'] for p in pairs]
        written = []
        for metric in metrics:
            written += plot_compression_facets(
                summary,
                metric=metric,
                out_path=stem.with_name(f'{stem.name}_{metric}'),
                facets=labels,
                # The native rule is the receiver reading its own
                # latents, which is only a ceiling for the task metric.
                reference=reference if metric == 'accuracy' else None,
                hue_of=hue_of,
                title=cfg.plot_title
                or f'{_pretty(metric)} vs compression — {dataset}',
            )

        table = wandb.Table(columns=list(summary[0]))
        for row in summary:
            table.add_data(*row.values())
        run.log({'compression_sweep': table})
        for path in written:
            if path.suffix == '.png':
                run.log({path.stem: wandb.Image(str(path))})
        run.summary.update(
            {f'native/{k}': v for k, v in reference.items()}
            | {'n_pairs': len(pairs), 'n_records': len(records)}
        )
    finally:
        run.finish()

    print(f'\n{len(records)} fits over {len(pairs)} pairs.')
    for pair in pairs:
        src_dim = agents[pair['source']]['train'].dim
        tgt_dim = agents[pair['target']]['train'].dim
        native = reference.get(pair['label'])
        native_txt = '' if native is None else f' | native RX {native:.4f}'
        print(f'  {pair["label"]}: {src_dim} -> {tgt_dim}{native_txt}')
    print(
        '\nWrote: '
        + ', '.join(str(p) for p in [*written, stem.with_suffix('.csv')])
    )


def _pretty(metric: str) -> str:
    """Metric name for a figure title."""
    return {
        'mrr': 'Mean reciprocal rank',
        'top1': 'Top-1 retrieval',
        'top5': 'Top-5 retrieval',
        'accuracy': 'Post-alignment accuracy',
    }.get(metric, metric)


if __name__ == '__main__':
    main()

"""How much does the RKHS residual actually buy, and at which ``lambda``?

Reproduces Fig. 1 of the RKA paper: post-alignment performance as a
function of the RKHS regularisation ``lambda``, read against a flat
Procrustes reference. Two things should be visible, and both are checks on
the implementation as much as results:

- an interior optimum, where the residual stage lifts the curve above the
  Procrustes line;
- convergence back *onto* that line as ``lambda`` grows, since ``A* -> 0``
  and ``f(x) -> Q*x`` (paper Eq. 17).

The best ``lambda`` per metric is printed at the end, ready to be pinned
into the pilot study rather than left to the aligner's own internal
selection -- which optimises held-out error on the *whitened residual*,
not the downstream metric anyone reports.

Examples
--------
    uv run scripts/regularization_study.py
    uv run scripts/regularization_study.py data=semasia_mnist
    uv run scripts/regularization_study.py pilots.n_pilots=2000
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
    resolve_pairs,
)
from src.plotting import plot_regularization_study, use_project_style

if TYPE_CHECKING:
    from src.decoder import Decoder
    from src.latent import LatentSpace

log = logging.getLogger(__name__)

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / 'config' / 'hydra')
ROOT = Path(__file__).resolve().parents[1]


def pilot_sets(
    cfg: DictConfig,
    pairs: list[tuple[str, str]],
    agents: dict[str, dict[str, LatentSpace]],
) -> dict[tuple[str, int], np.ndarray]:
    """One pilot set per ``(source, repeat)``, shared by every ``lambda``.

    Every point on the curve has to see the same calibration data, or the
    sweep would be measuring pilot luck rather than regularisation.
    """
    sets: dict[tuple[str, int], np.ndarray] = {}
    for repeat in range(int(cfg.pilots.n_repeats)):
        seed = int(cfg.seed) + 1000 * repeat
        for source, _ in pairs:
            if (source, repeat) in sets:
                continue
            train = agents[source]['train']
            budget = cfg.pilots.n_pilots or train.n_points
            labels = train.labels
            sets[(source, repeat)] = select_pilots(
                train.latent,
                n_pilots=int(budget),
                strategy=str(cfg.pilots.strategy),
                labels=labels,
                seed=seed,
            )
    return sets


def score(
    cfg: DictConfig,
    aligner: Aligner,
    pilots: np.ndarray,
    agents: dict[str, dict[str, LatentSpace]],
    pair: tuple[str, str],
    decoder: Decoder | None,
) -> dict[str, float]:
    """Fit one aligner on one pilot set and score the whole test split."""
    source, target = pair
    src_train, src_test = agents[source]['train'], agents[source]['test']
    tgt_train, tgt_test = agents[target]['train'], agents[target]['test']

    labels = src_train.labels
    aligner.fit(
        src_train.latent[pilots],
        tgt_train.latent[pilots],
        labels=None if labels is None else labels[pilots],
    )
    return alignment_metrics(
        aligner.transform(src_test.latent),
        tgt_test.latent,
        decoder=decoder,
        labels=tgt_test.labels,
        ks=tuple(cfg.eval.topk),
    )


def sweep_lambda(
    cfg: DictConfig,
    pairs: list[tuple[str, str]],
    agents: dict[str, dict[str, LatentSpace]],
    decoders: dict[str, Decoder | None],
    sets: dict[tuple[str, int], np.ndarray],
    metrics: list[str],
) -> list[dict[str, Any]]:
    """Score the swept method at every ``lambda``, over pairs and repeats."""
    rows: list[dict[str, Any]] = []
    for lam in cfg.lam_grid:
        method_cfg = OmegaConf.merge(
            cfg.sweep_method, {'lam': float(lam), 'lam_grid': None}
        )
        for repeat in range(int(cfg.pilots.n_repeats)):
            for source, target in pairs:
                aligner = build_aligner(method_cfg)
                scores = score(
                    cfg,
                    aligner,
                    sets[(source, repeat)],
                    agents,
                    (source, target),
                    decoders.get(target),
                )
                rows.append(
                    {
                        'lam': float(lam),
                        'repeat': repeat,
                        'source': source,
                        'target': target,
                        **{m: scores[m] for m in metrics},
                    }
                )
        log.info(
            'lam=%.4g done: %s',
            lam,
            ', '.join(
                f'{m}={np.mean([r[m] for r in rows if r["lam"] == lam]):.4f}'
                for m in metrics
            ),
        )
    return rows


def sweep_baselines(
    cfg: DictConfig,
    pairs: list[tuple[str, str]],
    agents: dict[str, dict[str, LatentSpace]],
    decoders: dict[str, Decoder | None],
    sets: dict[tuple[str, int], np.ndarray],
    metrics: list[str],
) -> dict[str, dict[str, float]]:
    """Score every lambda-independent baseline on the same pilot sets."""
    baselines: dict[str, dict[str, float]] = {}
    for name, method_cfg in (cfg.baselines or {}).items():
        collected: dict[str, list[float]] = defaultdict(list)
        for repeat in range(int(cfg.pilots.n_repeats)):
            for source, target in pairs:
                aligner = build_aligner(method_cfg)
                scores = score(
                    cfg,
                    aligner,
                    sets[(source, repeat)],
                    agents,
                    (source, target),
                    decoders.get(target),
                )
                for metric in metrics:
                    collected[metric].append(scores[metric])
        baselines[name] = {m: float(np.mean(v)) for m, v in collected.items()}
        log.info(
            'baseline %s: %s',
            name,
            ', '.join(f'{m}={v:.4f}' for m, v in baselines[name].items()),
        )
    return baselines


def aggregate(
    rows: list[dict[str, Any]], metrics: list[str]
) -> list[dict[str, Any]]:
    """Mean and spread over the (pair, repeat) cells at each ``lambda``."""
    grouped: dict[float, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        for metric in metrics:
            grouped[row['lam']][metric].append(row[metric])

    summary = []
    for lam, values in sorted(grouped.items()):
        entry: dict[str, Any] = {
            'lam': lam,
            'n_cells': len(values[metrics[0]]),
        }
        for metric in metrics:
            entry[f'{metric}_mean'] = float(np.mean(values[metric]))
            entry[f'{metric}_std'] = float(np.std(values[metric]))
        summary.append(entry)
    return summary


def best_lambda(
    summary: list[dict[str, Any]], metrics: list[str]
) -> dict[str, float]:
    """The ``lambda`` maximising each metric."""
    return {
        metric: max(summary, key=lambda r: r[f'{metric}_mean'])['lam']
        for metric in metrics
    }


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
    """Fixed-width view of the curve (also the accessibility fallback)."""
    columns = ['lam'] + [
        f'{m}_{stat}' for m in metrics for stat in ('mean', 'std')
    ]
    cells = [
        [
            f'{r[c]:.6g}' if isinstance(r[c], float) else str(r[c])
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
    config_name='regularization_study',
)
def main(cfg: DictConfig) -> None:
    """Sweep the RKHS regularisation and write the figure."""
    logging.getLogger('src').setLevel(logging.INFO)
    log.info('Configuration:\n%s', OmegaConf.to_yaml(cfg))

    metrics = list(cfg.eval.metrics)
    pairs = resolve_pairs(cfg, available_agents(cfg))
    agents = load_pair_data(cfg, pairs)

    decoders: dict[str, Decoder | None] = {}
    if cfg.decoder.enabled:
        for target in {t for _, t in pairs}:
            decoders[target] = build_decoder(cfg, agents[target]['train'])

    # Low pilot budgets are rank-deficient by design; see pilot_efficiency.
    for name in ('src.alignment.base', 'src.alignment.preprocessing'):
        logging.getLogger(name).setLevel(logging.ERROR)

    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.name
        or f'lambda-{cfg.data.get("dataset", cfg.data.source)}',
        group=cfg.wandb.group,
        job_type='regularization-study',
        tags=list(cfg.wandb.tags),
        mode=cfg.wandb.mode,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    try:
        sets = pilot_sets(cfg, pairs, agents)
        baselines = sweep_baselines(
            cfg, pairs, agents, decoders, sets, metrics
        )
        summary = aggregate(
            sweep_lambda(cfg, pairs, agents, decoders, sets, metrics), metrics
        )
        reference = native_reference(pairs, agents, decoders)
        best = best_lambda(summary, metrics)

        dataset = cfg.data.get('dataset', cfg.data.source)
        out = Path(cfg.output_dir) / f'regularization_{dataset}'
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.with_suffix('.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)

        use_project_style(ROOT / 'config' / 'plotting' / 'plt.mplstyle')
        written = plot_regularization_study(
            summary,
            metrics=metrics,
            out_path=out,
            baselines=baselines,
            reference=reference,
            title=cfg.plot_title or f'RKHS regularisation — SEMASIA {dataset}',
        )

        table = wandb.Table(columns=list(summary[0]))
        for row in summary:
            table.add_data(*row.values())
        run.log({'regularization': table})
        for path in written:
            if path.suffix == '.png':
                run.log({'figure': wandb.Image(str(path))})
        run.summary.update(
            {f'best_lam/{k}': v for k, v in best.items()}
            | {f'native/{k}': v for k, v in reference.items()}
            | {
                f'baseline/{name}/{m}': v
                for name, values in baselines.items()
                for m, v in values.items()
            }
        )
    finally:
        run.finish()

    print('\n' + format_table(summary, metrics))
    print('\nBaselines (flat in lambda):')
    for name, values in baselines.items():
        print(
            f'  {name:14s} '
            + '  '.join(f'{m}={v:.4f}' for m, v in values.items())
        )
    if reference:
        print(
            '\nNative RX ceiling: '
            + '  '.join(f'{k}={v:.4f}' for k, v in reference.items())
        )

    print('\nBest lambda per metric:')
    for metric, lam in best.items():
        gain = {
            name: max(r[f'{metric}_mean'] for r in summary if r['lam'] == lam)
            - values.get(metric, float('nan'))
            for name, values in baselines.items()
        }
        margin = '  '.join(f'vs {n}: {g:+.4f}' for n, g in gain.items())
        print(f'  {metric:10s} lam={lam:<10.4g} {margin}')

    primary = metrics[0]
    print(
        f'\nPin it into the pilot study with:\n'
        f'  uv run scripts/pilot_efficiency.py '
        f'alignment@methods.rkhs=rkhs '
        f'methods.rkhs.lam={best[primary]:g} '
        f'methods.rkhs.lam_grid=null'
    )
    print(
        '\nWrote: '
        + ', '.join(str(p) for p in [*written, out.with_suffix('.csv')])
    )


if __name__ == '__main__':
    main()

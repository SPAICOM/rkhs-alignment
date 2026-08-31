"""Which kernel explains the Procrustes residual, and at what bandwidth?

RKA's whole claim rests on there being structure left over after the rigid
map that a kernel can capture. That makes the kernel a modelling choice,
not a default, and this script measures it directly rather than through
the downstream metric it is usually judged by.

The quantity to watch is ``residual_r2``: how much of the *held-out*
Procrustes residual the fitted correction explains,

    E_test  = Y_test - Q z_test          what the rigid map left behind
    G_test  = f(z_test) - Q z_test       what the kernel predicted
    R^2     = 1 - ||E_test - G_test||^2 / ||E_test||^2

This is the kernel's own job description. A kernel that scores zero has
found nothing generalisable, whatever it did to the calibration set, and
a *negative* score means the correction actively moves the prediction
away from the target out of sample.

Two sweeps, both reported against the rigid baseline:

- ``kernel``  : family x bandwidth for the residual stage. Note that
  ``linear`` and ``cosine`` are expected to score ~0 by construction, not
  by accident: their RKHS *is* the space of linear functions, which is
  exactly what the constraint ``G X^T = 0`` removes, so the fit has no
  degrees of freedom left. The script reports their capacity so that
  shows up as a fact rather than a puzzle.
- ``pilots``  : bandwidth of the kernel used for *herding*, holding the
  aligner fixed. The RKA paper's argument is that pilots should be chosen
  in the same RKHS the residual is later fitted in; the experiment
  scripts have always let that kernel default instead, so this sweep is
  what says whether matching them is worth anything.

Examples
--------
    uv run scripts/kernel_study.py
    uv run scripts/kernel_study.py sweep=pilots
    uv run scripts/kernel_study.py data=semasia_mnist pilots.n_pilots=1536
    uv run scripts/kernel_study.py 'kernels=[rbf,laplacian]'
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

from src.alignment import (
    Kernel,
    ProcrustesAligner,
    RKHSAligner,
    alignment_metrics,
    select_pilots,
)
from src.experiment import (
    available_agents,
    build_decoder,
    load_pair_data,
    resolve_pairs,
)
from src.plotting import plot_kernel_study, use_project_style

if TYPE_CHECKING:
    from src.decoder import Decoder
    from src.latent import LatentSpace

log = logging.getLogger(__name__)

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / 'config' / 'hydra')
ROOT = Path(__file__).resolve().parents[1]


def residual_r2(
    aligner: RKHSAligner, X_src: np.ndarray, Y_tgt: np.ndarray
) -> float:
    """Held-out fraction of the Procrustes residual the kernel explains.

    Both predictions are taken in the receiver's raw space, so this is
    the residual the receiver would actually have seen.
    """
    rigid = aligner.transform_linear(X_src)
    predicted = aligner.transform(X_src) - rigid
    left_behind = Y_tgt - rigid
    energy = float(np.sum(left_behind**2))
    if energy <= 0:
        return 0.0
    return 1.0 - float(np.sum((left_behind - predicted) ** 2)) / energy


def fit_and_score(
    cfg: DictConfig,
    aligner: RKHSAligner,
    pilots: np.ndarray,
    agents: dict[str, dict[str, LatentSpace]],
    pair: tuple[str, str],
    decoder: Decoder | None,
) -> dict[str, Any]:
    """One fit, scored on the test split plus the residual diagnostic."""
    source, target = pair
    src_train, src_test = agents[source]['train'], agents[source]['test']
    tgt_train, tgt_test = agents[target]['train'], agents[target]['test']

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
    scores['residual_r2'] = residual_r2(
        aligner, src_test.latent, tgt_test.latent
    )
    scores['capacity'] = float(summary.get('rkhs_capacity', 0))
    scores['lam'] = float(summary.get('rkhs_lam', 0.0))
    scores['gram_cond'] = float(summary.get('rkhs_gram_cond', 0.0))
    return scores


def rigid_reference(
    cfg: DictConfig,
    pilots: dict[str, np.ndarray],
    pairs: list[tuple[str, str]],
    agents: dict[str, dict[str, LatentSpace]],
    decoders: dict[str, Decoder | None],
    metrics: list[str],
) -> dict[str, float]:
    """Procrustes alone, on the same pilots -- the line to beat."""
    collected: dict[str, list[float]] = defaultdict(list)
    for source, target in pairs:
        aligner = ProcrustesAligner(
            preprocess=cfg.aligner.preprocess,
            n_components=cfg.aligner.n_components,
            eps=cfg.aligner.eps,
            seed=cfg.seed,
        )
        src_train, src_test = agents[source]['train'], agents[source]['test']
        tgt_train, tgt_test = agents[target]['train'], agents[target]['test']
        aligner.fit(
            src_train.latent[pilots[source]],
            tgt_train.latent[pilots[source]],
            src_context=src_train.latent,
            tgt_context=tgt_train.latent,
        )
        scores = alignment_metrics(
            aligner.transform(src_test.latent),
            tgt_test.latent,
            decoder=decoders.get(target),
            labels=tgt_test.labels,
            ks=tuple(cfg.eval.topk),
        )
        for metric in metrics:
            if metric in scores:
                collected[metric].append(scores[metric])
    out = {m: float(np.mean(v)) for m, v in collected.items()}
    # The rigid map is the zero-correction point by definition.
    out['residual_r2'] = 0.0
    return out


def make_aligner(cfg: DictConfig, **overrides) -> RKHSAligner:
    """An RKHS aligner from the study's fixed settings plus overrides."""
    fields = OmegaConf.to_container(cfg.aligner, resolve=True)
    fields.update(overrides)
    return RKHSAligner(seed=cfg.seed, **fields)


def sweep_kernels(
    cfg: DictConfig,
    pilots: dict[str, np.ndarray],
    pairs: list[tuple[str, str]],
    agents: dict[str, dict[str, LatentSpace]],
    decoders: dict[str, Decoder | None],
    metrics: list[str],
) -> list[dict[str, Any]]:
    """Kernel family x bandwidth for the residual stage."""
    rows: list[dict[str, Any]] = []
    for name in cfg.kernels:
        for scale in cfg.bandwidth_scales:
            cells: dict[str, list[float]] = defaultdict(list)
            for source, target in pairs:
                scores = fit_and_score(
                    cfg,
                    make_aligner(
                        cfg, kernel=str(name), bandwidth_scale=float(scale)
                    ),
                    pilots[source],
                    agents,
                    (source, target),
                    decoders.get(target),
                )
                for key, value in scores.items():
                    cells[key].append(value)
            row = {
                'kernel': str(name),
                'bandwidth_scale': float(scale),
                **{k: float(np.mean(v)) for k, v in cells.items()},
            }
            rows.append(row)
            log.info(
                'kernel=%-11s scale=%-6g residual_r2=%+.4f capacity=%d %s',
                name,
                scale,
                row['residual_r2'],
                int(row['capacity']),
                ' '.join(f'{m}={row[m]:.4f}' for m in metrics if m in row),
            )
    return rows


def sweep_pilot_bandwidth(
    cfg: DictConfig,
    pairs: list[tuple[str, str]],
    agents: dict[str, dict[str, LatentSpace]],
    decoders: dict[str, Decoder | None],
    metrics: list[str],
) -> list[dict[str, Any]]:
    """Bandwidth of the *herding* kernel, with the aligner held fixed.

    Kernel herding matches the selected subset's mean embedding to the
    pool's, and how local that match is depends entirely on the kernel's
    bandwidth. A sharp kernel makes the redundancy term myopic and the
    score degenerates into a density ranking; a wide one keeps a global
    diversity pressure. Nothing in the pipeline ties this bandwidth to
    the one the residual stage later uses, which is precisely what the
    sweep is here to test.
    """
    rows: list[dict[str, Any]] = []
    for scale in cfg.pilot_bandwidth_scales:
        cells: dict[str, list[float]] = defaultdict(list)
        for source, target in pairs:
            train = agents[source]['train']
            kernel = Kernel(
                str(cfg.pilot_kernel), bandwidth_scale=float(scale)
            )
            chosen = select_pilots(
                train.latent,
                n_pilots=int(cfg.pilots.n_pilots or train.n_points),
                strategy='herding',
                labels=train.labels,
                kernel=kernel,
                seed=cfg.seed,
            )
            scores = fit_and_score(
                cfg,
                make_aligner(cfg),
                chosen,
                agents,
                (source, target),
                decoders.get(target),
            )
            for key, value in scores.items():
                cells[key].append(value)
        row = {
            'kernel': f'herding-{cfg.pilot_kernel}',
            'bandwidth_scale': float(scale),
            **{k: float(np.mean(v)) for k, v in cells.items()},
        }
        rows.append(row)
        log.info(
            'pilot bandwidth=%-6g residual_r2=%+.4f %s',
            scale,
            row['residual_r2'],
            ' '.join(f'{m}={row[m]:.4f}' for m in metrics if m in row),
        )
    return rows


def format_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    """Fixed-width view of the sweep (also the accessibility fallback)."""
    cells = [
        [
            f'{r[c]:.4f}' if isinstance(r[c], float) else str(r[c])
            for c in columns
        ]
        for r in rows
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
    version_base=None, config_path=CONFIG_DIR, config_name='kernel_study'
)
def main(cfg: DictConfig) -> None:
    """Run the configured sweep and write the figure."""
    logging.getLogger('src').setLevel(logging.INFO)
    log.info('Configuration:\n%s', OmegaConf.to_yaml(cfg))

    metrics = list(cfg.eval.metrics)
    panels = ['residual_r2', *metrics]
    pairs = resolve_pairs(cfg, available_agents(cfg))
    agents = load_pair_data(cfg, pairs)

    decoders: dict[str, Decoder | None] = {}
    if cfg.decoder.enabled:
        for target in {t for _, t in pairs}:
            decoders[target] = build_decoder(cfg, agents[target]['train'])

    for name in ('src.alignment.base', 'src.alignment.preprocessing'):
        logging.getLogger(name).setLevel(logging.ERROR)

    # One pilot set per source, shared by the kernel sweep so that only
    # the kernel varies. The pilot sweep re-selects by construction.
    pilots = {
        source: select_pilots(
            agents[source]['train'].latent,
            n_pilots=int(
                cfg.pilots.n_pilots or agents[source]['train'].n_points
            ),
            strategy=str(cfg.pilots.strategy),
            labels=agents[source]['train'].labels,
            seed=cfg.seed,
        )
        for source, _ in pairs
    }

    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.name
        or f'kernels-{cfg.data.get("dataset", cfg.data.source)}',
        group=cfg.wandb.group,
        job_type='kernel-study',
        tags=list(cfg.wandb.tags),
        mode=cfg.wandb.mode,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    try:
        reference = rigid_reference(
            cfg, pilots, pairs, agents, decoders, metrics
        )
        log.info(
            'Procrustes reference: %s',
            ' '.join(f'{k}={v:.4f}' for k, v in reference.items()),
        )
        rows = (
            sweep_pilot_bandwidth(cfg, pairs, agents, decoders, metrics)
            if cfg.sweep == 'pilots'
            else sweep_kernels(cfg, pilots, pairs, agents, decoders, metrics)
        )

        use_project_style(ROOT / 'config' / 'plotting' / 'plt.mplstyle')
        dataset = cfg.data.get('dataset', cfg.data.source)
        out = Path(cfg.output_dir) / f'kernel_study_{cfg.sweep}_{dataset}'
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.with_suffix('.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        # Families the constraint annihilates carry no line worth drawing:
        # their curve is a flat zero and it crowds out the rest.
        drawable = [r for r in rows if r['capacity'] > 0]
        written = plot_kernel_study(
            drawable or rows,
            metrics=panels,
            out_path=out,
            reference=reference,
            title=cfg.plot_title
            or (
                f'Herding bandwidth — {dataset}'
                if cfg.sweep == 'pilots'
                else f'Kernels on the Procrustes residual — {dataset}'
            ),
        )

        table = wandb.Table(columns=list(rows[0]))
        for row in rows:
            table.add_data(*row.values())
        run.log({'kernel_study': table})
        for path in written:
            if path.suffix == '.png':
                run.log({'figure': wandb.Image(str(path))})
        best = max(rows, key=lambda r: r['residual_r2'])
        run.summary.update(
            {f'rigid/{k}': v for k, v in reference.items()}
            | {
                'best/kernel': best['kernel'],
                'best/bandwidth_scale': best['bandwidth_scale'],
                'best/residual_r2': best['residual_r2'],
            }
        )
    finally:
        run.finish()

    columns = ['kernel', 'bandwidth_scale', 'residual_r2', 'capacity'] + [
        m for m in metrics if m in rows[0]
    ]
    print('\n' + format_table(rows, columns))
    print(
        '\nProcrustes reference: '
        + '  '.join(f'{k}={v:.4f}' for k, v in reference.items())
    )
    print(
        f'\nBest residual_r2: {best["residual_r2"]:+.4f} at '
        f'kernel={best["kernel"]}, bandwidth_scale={best["bandwidth_scale"]}'
    )
    print(
        '\nWrote: '
        + ', '.join(str(p) for p in [*written, out.with_suffix('.csv')])
    )


if __name__ == '__main__':
    main()

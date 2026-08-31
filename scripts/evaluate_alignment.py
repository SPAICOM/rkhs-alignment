"""Fit and score latent-space alignment maps between pairs of encoders.

For every ``(source, target)`` pair the script

1. loads both agents' train/test latents with rows paired sample-by-sample,
2. fits the configured aligner on the train split (the calibration set),
3. transports the source's *held-out* latents into the target's raw
   latent space, and
4. scores the result three ways -- reconstruction, retrieval, and the
   receiver's own decoder accuracy on the transported latents.

The alignment method itself is chosen entirely from configuration: the
``alignment`` config group carries a ``_target_`` and that method's own
hyper-parameters, and Hydra instantiates the matching class from
:mod:`src.alignment`.

Examples
--------
    uv run scripts/evaluate_alignment.py alignment=rkhs alignment.lam=1e-2
    uv run scripts/evaluate_alignment.py alignment=relative \
        alignment.n_anchors=256
    uv run scripts/evaluate_alignment.py data=synthetic receiver=agent_00
    uv run scripts/evaluate_alignment.py -m alignment=rkhs \
        alignment.lam=1e-4,1e-3,1e-2
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

import wandb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.alignment import Aligner, alignment_metrics
from src.experiment import (
    available_agents,
    build_aligner,
    build_decoder,
    load_pair_data,
    resolve_pairs,
)

if TYPE_CHECKING:
    from src.decoder import Decoder
    from src.latent import LatentSpace

log = logging.getLogger(__name__)

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / 'config' / 'hydra')

# Columns of the printed / logged results table, in order.
_REPORT_COLUMNS = (
    'source',
    'target',
    'dim_src',
    'dim_tgt',
    'nmse',
    'r2',
    'cosine',
    'mrr',
    'top1',
    'accuracy',
    'accuracy_oracle',
    'accuracy_ratio',
)


# ---------------------------------------------------------------------
# One pair
# ---------------------------------------------------------------------


def run_pair(
    cfg: DictConfig,
    source: str,
    target: str,
    agents: dict[str, dict[str, LatentSpace]],
    decoders: dict[str, Decoder | None],
) -> dict[str, Any]:
    """Fit and score one ``source -> target`` alignment.

    Returns
    -------
    dict[str, Any]
        Metrics, the aligner's own summary, and (when available) the
        rigid-stage reference and the regularisation sweep.
    """
    src_train, src_test = agents[source]['train'], agents[source]['test']
    tgt_train, tgt_test = agents[target]['train'], agents[target]['test']

    aligner: Aligner = build_aligner(cfg.alignment)
    log.info(
        'Fitting %s: %s (d=%d) -> %s (d=%d) on %d calibration samples.',
        aligner.name,
        source,
        src_train.dim,
        target,
        tgt_train.dim,
        src_train.n_points,
    )
    aligner.fit(src_train.latent, tgt_train.latent, labels=src_train.labels)

    Y_pred = aligner.transform(src_test.latent)
    decoder = decoders.get(target)
    metrics = alignment_metrics(
        Y_pred,
        tgt_test.latent,
        decoder=decoder,
        labels=tgt_test.labels,
        ks=tuple(cfg.eval.topk),
    )

    record: dict[str, Any] = {
        'source': source,
        'target': target,
        **metrics,
        **aligner.summary(),
    }

    # The rigid stage on its own, as an in-run reference point.
    if cfg.eval.log_linear_reference and hasattr(aligner, 'transform_linear'):
        reference = alignment_metrics(
            aligner.transform_linear(src_test.latent),
            tgt_test.latent,
            decoder=decoder,
            labels=tgt_test.labels,
            ks=tuple(cfg.eval.topk),
        )
        record.update({f'linear_ref/{k}': v for k, v in reference.items()})

    record['lambda_path'] = list(getattr(aligner, 'lambda_path_', []) or [])
    return record


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------


def format_table(records: list[dict[str, Any]]) -> str:
    """Render the per-pair results as a fixed-width text table."""
    columns = [c for c in _REPORT_COLUMNS if any(c in r for r in records)]
    widths = {
        c: max(len(c), *(len(_cell(r.get(c))) for r in records))
        for c in columns
    }
    header = '  '.join(c.ljust(widths[c]) for c in columns)
    rule = '  '.join('-' * widths[c] for c in columns)
    rows = [
        '  '.join(_cell(r.get(c)).ljust(widths[c]) for c in columns)
        for r in records
    ]
    return '\n'.join([header, rule, *rows])


def _cell(value: Any) -> str:
    if value is None:
        return '-'
    if isinstance(value, float):
        return f'{value:.4f}'
    return str(value)


def log_to_wandb(
    run: Any, records: list[dict[str, Any]], columns: list[str]
) -> dict[str, float]:
    """Push per-pair rows, the sweep curves and the aggregate summary."""
    table = wandb.Table(columns=columns)
    for step, record in enumerate(records):
        table.add_data(*[record.get(c) for c in columns])

        scalars = {
            f'{k}': v
            for k, v in record.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }
        pair = f'{record["source"]}->{record["target"]}'
        run.log(
            {f'pair/{k}': v for k, v in scalars.items()} | {'pair_index': step}
        )

        if record['lambda_path']:
            sweep = wandb.Table(columns=['lam', 'val_nmse', 'val_cosine'])
            for point in record['lambda_path']:
                sweep.add_data(
                    point['lam'], point['val_nmse'], point['val_cosine']
                )
            run.log(
                {
                    f'lambda_sweep/{pair}': wandb.plot.line(
                        sweep,
                        'lam',
                        'val_nmse',
                        title=f'Regularisation sweep: {pair}',
                    )
                }
            )

    run.log({'results': table})

    aggregate = {}
    for key in _REPORT_COLUMNS:
        values = [
            r[key]
            for r in records
            if isinstance(r.get(key), (int, float))
            and not isinstance(r.get(key), bool)
        ]
        if values:
            aggregate[f'mean/{key}'] = float(np.mean(values))
    run.summary.update(aggregate)
    return aggregate


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------


@hydra.main(
    version_base=None, config_path=CONFIG_DIR, config_name='evaluate_alignment'
)
def main(cfg: DictConfig) -> None:
    """Run the configured alignment experiment."""
    logging.getLogger('src').setLevel(logging.INFO)
    log.info('Configuration:\n%s', OmegaConf.to_yaml(cfg))

    pairs = resolve_pairs(cfg, available_agents(cfg))
    agents = load_pair_data(cfg, pairs)

    missing = {n for pair in pairs for n in pair} - set(agents)
    if missing:
        raise ValueError(
            f'Pairs reference agents that were not loaded: {sorted(missing)}.'
        )

    # One decoder per receiver, shared by every pair that targets it.
    decoders: dict[str, Decoder | None] = {}
    if cfg.decoder.enabled:
        for target in {t for _, t in pairs}:
            decoders[target] = build_decoder(cfg, agents[target]['train'])

    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.name
        or f'{cfg.alignment._target_.rsplit(".", 1)[-1]}-{cfg.data.source}',
        group=cfg.wandb.group,
        job_type=cfg.wandb.job_type,
        tags=list(cfg.wandb.tags),
        mode=cfg.wandb.mode,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    try:
        records = [
            run_pair(cfg, source, target, agents, decoders)
            for source, target in pairs
        ]
        columns = sorted(
            {k for r in records for k, v in r.items() if k != 'lambda_path'}
        )
        aggregate = log_to_wandb(run, records, columns)
    finally:
        run.finish()

    print('\n' + format_table(records))
    print(
        '\nMean over pairs: '
        + '  '.join(
            f'{k.removeprefix("mean/")}={v:.4f}'
            for k, v in aggregate.items()
            if k.removeprefix('mean/') in ('nmse', 'cosine', 'mrr', 'accuracy')
        )
    )


if __name__ == '__main__':
    main()

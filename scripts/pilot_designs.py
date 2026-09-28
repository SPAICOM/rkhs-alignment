"""RKA against pure kernel alignment, per pilot design.

Figure (ii)'s axis -- the pilot budget at a fixed truncation -- drawn once
per pilot *design*, so the question is not how the field moves with ``N``
but how each method's ``N``-curve changes when the pilots stop covering
the pool. That is the argument for the rigid stage, because the methods
differ in what they fall back on when the calibration set cannot identify
the map::

    RKA   lam -> inf  ->  Q z, the Procrustes map
    krr   lam -> inf  ->  0,   the receiver's mean (chance accuracy)

so a design that pushes the test set off the pilots' support should cost
pure kernel alignment far more than it costs RKA.

The designs are ``src/alignment/pilots.py`` strategies, so herding and its
mirror share one kernel and one implementation:

- ``herding``      the reference, and the kernel's own best case;
- ``random``       i.i.d. from the pool, the undesigned baseline;
- ``anti_herding`` herding's exact mirror, maximising the same MMD;
- ``classes:k``    pilots from only ``k`` classes: support bias aligned
  with the label structure the decoder reads out;
- ``ball``         the pilots nearest one random pool point: support bias
  with no label structure.

Charts come from each device's whole local split (``use_local_context``),
so a biased design biases the cross-map and never the whitening, and
``lambda`` is selected per cell on a held-out slice of that cell's own
pilots -- what a deployment actually gets.

``import_records`` merges a CSV written by an earlier run (or by the
exploratory script this study grew out of) in the same schema, so the
figures can be drawn without refitting anything.

Examples
--------
    just pilot-designs
    just pilot-designs plot_only=true
    just pilot-designs 'designs=[herding,anti_herding]' 'seeds=[0]'
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
    Aligner,
    alignment_metrics,
    select_pilot_path,
)
from src.experiment import (
    build_aligner,
    build_decoder,
    configure_method,
    draw_pool,
    load_named_agents,
)
from src.plotting import plot_pilot_efficiency, use_project_style
from src.reporting import figure_dir, pairs_slug, result_dir

if TYPE_CHECKING:
    from src.decoder import Decoder
    from src.latent import LatentSpace

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = str(ROOT / 'config' / 'hydra')

# One record per fitted cell. `coverage` is the share of the receiver's
# classes the design let through, which is what separates `classes:k`
# from the geometric designs at the same budget.
FIELDS: tuple[str, ...] = (
    'source',
    'target',
    'k',
    'design',
    'n_pilots',
    'seed',
    'coverage',
    'method',
    'lam',
    'accuracy',
    'top1',
    'mrr',
)

_TEXT = frozenset({'source', 'target', 'design', 'method'})


def read_records(path: Path) -> list[dict[str, Any]]:
    """Records from one CSV, with the numeric columns parsed."""
    rows: list[dict[str, Any]] = []
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
            for key in ('k', 'n_pilots', 'seed'):
                if isinstance(parsed.get(key), float):
                    parsed[key] = int(parsed[key])
            rows.append(parsed)
    return rows


def write_records(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write one (pair, seed) CSV, via a temporary so a crash cannot cut it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.csv.partial')
    with tmp.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FIELDS))
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def run_cell(
    cfg: DictConfig,
    agents: dict[str, dict[str, LatentSpace]],
    pair: tuple[str, str],
    decoder: Decoder | None,
    seed: int,
) -> list[dict[str, Any]]:
    """Every (design, budget, method) cell of one pair and one seed."""
    source, target = pair
    src_train, src_test = agents[source]['train'], agents[source]['test']
    tgt_train, tgt_test = agents[target]['train'], agents[target]['test']
    counts = [int(c) for c in cfg.pilots.counts]
    labels = src_train.labels
    n_classes = int(np.unique(tgt_test.labels).size)

    context: dict[str, np.ndarray] = {}
    if cfg.use_local_context:
        context = {
            'src_context': src_train.latent,
            'tgt_context': tgt_train.latent,
        }

    pool = draw_pool(src_train, cfg.pilots.pool_size, seed)
    records: list[dict[str, Any]] = []
    for design in cfg.designs:
        design = str(design)
        paths = select_pilot_path(
            src_train.latent[pool],
            counts=counts,
            strategy=design,
            labels=None if labels is None else labels[pool],
            seed=seed,
        )
        for n_pilots in counts:
            pilots = pool[paths[n_pilots]]
            if pilots.size < n_pilots:
                log.warning(
                    '%s left %d of %d pilots at N=%d; skipping the cell.',
                    design,
                    pilots.size,
                    n_pilots,
                    n_pilots,
                )
                continue
            coverage = (
                float('nan')
                if labels is None
                else float(np.unique(labels[pilots]).size / n_classes)
            )
            for name, method_cfg in cfg.methods.items():
                configured = configure_method(
                    method_cfg, cfg.chart, int(cfg.symbols)
                )
                aligner: Aligner = build_aligner(configured, seed=seed)
                aligner.fit(
                    src_train.latent[pilots],
                    tgt_train.latent[pilots],
                    labels=None if labels is None else labels[pilots],
                    **context,
                )
                scores = alignment_metrics(
                    aligner.transform(src_test.latent),
                    tgt_test.latent,
                    decoder=decoder,
                    labels=tgt_test.labels,
                    ks=(1, 5),
                )
                records.append(
                    {
                        'source': source,
                        'target': target,
                        'k': int(cfg.symbols),
                        'design': design,
                        'n_pilots': n_pilots,
                        'seed': seed,
                        'coverage': coverage,
                        'method': name,
                        'lam': getattr(aligner, 'lam_', None),
                        'accuracy': scores['accuracy'],
                        'top1': scores['top1'],
                        'mrr': scores['mrr'],
                    }
                )
            log.info(
                'seed=%d %s N=%d: %s',
                seed,
                design,
                n_pilots,
                '  '.join(
                    f'{r["method"]}={r["accuracy"]:.4f}'
                    for r in records[-len(cfg.methods) :]
                ),
            )
    return records


def summarise(
    records: list[dict[str, Any]], metrics: list[str], order: list[str]
) -> dict[str, list[dict[str, Any]]]:
    """Mean +/- sd over (pair, seed) per design, in the plotter's schema.

    The spread is over realisations *and* pairs: a design is a claim
    about the calibration set, not about one encoder pair, so a band that
    hid the pair-to-pair variation would overstate it.
    """
    cells: dict[tuple, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in records:
        key = (row['design'], row['method'], int(row['n_pilots']))
        for metric in metrics:
            value = row.get(metric)
            if value is not None:
                cells[key][metric].append(float(value))

    rank = {name: i for i, name in enumerate(order)}
    by_design: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (design, method, n_pilots), values in cells.items():
        row: dict[str, Any] = {
            'method': method,
            # The plotter's second channel is the pilot design; here every
            # figure holds one, so it is constant within a figure.
            'strategy': design,
            'n_pilots': n_pilots,
            'n_seeds': len(next(iter(values.values()))),
        }
        for metric, samples in values.items():
            row[f'{metric}_mean'] = float(np.mean(samples))
            row[f'{metric}_std'] = float(np.std(samples))
        by_design[design].append(row)

    return {
        design: sorted(
            rows,
            key=lambda r: (rank.get(r['method'], len(order)), r['n_pilots']),
        )
        for design, rows in by_design.items()
    }


@hydra.main(
    version_base=None, config_path=CONFIG_DIR, config_name='pilot_designs'
)
def main(cfg: DictConfig) -> None:
    """Fit every (design, budget, method) cell; draw one figure per design."""
    logging.getLogger('src').setLevel(logging.INFO)
    log.info('Configuration:\n%s', OmegaConf.to_yaml(cfg))

    metrics = list(cfg.eval.metrics)
    order = list(cfg.methods)
    seeds = [int(s) for s in cfg.seeds]
    pairs = [(str(p.source), str(p.target)) for p in cfg.pairs]
    dataset = str(cfg.data.get('dataset', cfg.data.source))

    for name in ('src.alignment.base', 'src.alignment.preprocessing'):
        logging.getLogger(name).setLevel(logging.ERROR)

    results = result_dir(cfg.output.results, str(cfg.output.study), dataset)
    use_project_style(ROOT / 'config' / 'plotting' / 'plt.mplstyle')

    records: list[dict[str, Any]] = []
    if cfg.get('import_records'):
        imported = read_records(ROOT / str(cfg.import_records))
        wanted = {str(d) for d in cfg.designs}
        records += [r for r in imported if r['design'] in wanted]
        log.info(
            'Imported %d records from %s (%d after filtering to this run).',
            len(imported),
            cfg.import_records,
            len(records),
        )

    if not cfg.plot_only:
        agents = load_named_agents(cfg, sorted({m for p in pairs for m in p}))
        decoders: dict[str, Decoder | None] = {}
        if cfg.decoder.enabled:
            for target in {t for _, t in pairs}:
                decoders[target] = build_decoder(cfg, agents[target]['train'])

        for pair in pairs:
            for seed in seeds:
                stem = f'designs_{dataset}_{pairs_slug([pair])}_k{cfg.symbols}'
                path = results / f'{stem}_seed{seed}.csv'
                if path.exists() and cfg.resume:
                    log.info('Reusing %s (resume=true).', path.name)
                    records += read_records(path)
                    continue
                log.info('=== %s -> %s  seed=%d ===', *pair, seed)
                fitted = run_cell(
                    cfg, agents, pair, decoders.get(pair[1]), seed
                )
                write_records(path, fitted)
                records += fitted

    if not records:
        raise SystemExit(
            'No records: fit some, or point `import_records` at a CSV.'
        )

    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.name or f'designs-{dataset}-k{cfg.symbols}',
        group=cfg.wandb.group,
        job_type='pilot-designs',
        tags=list(cfg.wandb.tags),
        mode=cfg.wandb.mode,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    written: list[Path] = []
    try:
        figures = figure_dir(
            cfg.output.figures, str(cfg.output.study), dataset
        )
        by_design = summarise(records, metrics, order)
        for design, rows in by_design.items():
            slug = design.replace(':', '')
            written += plot_pilot_efficiency(
                rows,
                metrics=metrics,
                out_path=figures / f'designs_{dataset}_k{cfg.symbols}_{slug}',
                title=f'pilot design: {design}',
                formats=tuple(cfg.output.formats),
                panel=tuple(float(v) for v in cfg.output.panel),
                text_scale=float(cfg.output.text_scale),
            )
            pairs_seen = len({(r['source'], r['target']) for r in records})
            print(f'\n{design} — {pairs_seen} pairs, {len(seeds)} seeds\n')
            head = f'{"method":<12}{"N":>7}' + ''.join(
                f'{m:>12}' for m in metrics
            )
            print(head)
            print('-' * len(head))
            for row in rows:
                print(
                    f'{row["method"]:<12}{row["n_pilots"]:>7}'
                    + ''.join(f'{row[f"{m}_mean"]:12.4f}' for m in metrics)
                )
        for path in written:
            if path.suffix == '.png':
                run.log({f'figure/{path.stem}': wandb.Image(str(path))})
    finally:
        run.finish()

    print(f'\nWrote {len(written)} figures under {figures}/')


if __name__ == '__main__':
    main()

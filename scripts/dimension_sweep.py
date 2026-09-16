"""Figure (iii): the field against the compression dimension.

One figure, one panel per metric, plotting RKA, Procrustes, CCA, SVCCA and
Proto-PFE against the number of transmitted symbols ``k`` at a single,
fixed pilot budget ``N``. It is the pilot sweep turned on its side: there
the rate is pinned and ``N`` moves, here ``N`` is pinned and the rate
moves -- which is the axis a claim about robustness to compression has to
be read on.

Two kinds of curve share the figure, and the distinction is the point of
the design:

- **read back** -- RKA and Procrustes are not refitted. ``lambda_sweep.py``
  already measured both at every ``(chart, rank, budget)`` cell, and RKA's
  point is the one at the lambda that maximised ``read_back.metric`` in
  that cell. Every metric is reported at that one lambda, the rule
  ``pilot_sweep.py`` applies too.
- **fitted** -- the entries of ``methods``, fitted here on the same
  pilots: same seed, same pool, same design. CCA and SVCCA take no chart:
  CCA is invariant to any invertible map of either space, so whitening it
  changes nothing, and SVCCA's SVD ranks exactly the spectrum whitening
  would flatten; their rate goes through ``n_canonical``. Proto-PFE takes
  no chart either -- its rate is the anchor count. A method listed in
  ``charted`` would instead be fitted per ``(chart, rank)`` on the
  truncated-whitened coordinates RKA and Procrustes had.

Reading one set of numbers back and fitting the other is only sound if
both saw the same pilots, decoder and preprocessing, and the lambda
sweep's CSVs record none of those. So the run checks it: Procrustes is
refitted per ``(chart, rank)`` and compared with the value in the CSV
before anything is drawn.

With ``seeds`` set the figure is an average over pilot realisations
instead. The lambda sweep measured one pilot set, so nothing can be read
back: every seed draws its own pilots, exactly as ``pilot_sweep.py`` does
for that seed, and every method is fitted on them -- RKA at the lambda the
sweep selected for that ``(chart, rank)``. Each row keeps its seed, so a
study pooling several pairs can average within a pair first.

Every rank has to sit strictly below ``N``. At ``N <= k`` the whitened
chart is rank-deficient and RKA's residual has no degrees of freedom, so
the figure would measure the calibration rather than the compression.

The x position is the rate a method actually delivered, not the one it
was asked for: a canonical method cannot send more scores than its SVD
truncation kept, and plotting the request would credit it with airtime
it did not spend.

Examples
--------
    just dimension-sweep
    just dimension-sweep decoder.kind=linear
    uv run scripts/dimension_sweep.py pilots.n_pilots=4096
    uv run scripts/dimension_sweep.py 'ranks=[16, 32, 64, 128]'
    uv run scripts/dimension_sweep.py plot_only=true
    uv run scripts/dimension_sweep.py 'seeds=[0,1,2,3,4]'
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
    build_aligner,
    build_decoder,
    channel_rate,
    check_budgets_fit_pool,
    configure_method,
    describe_pairs,
    draw_pool,
    native_accuracy,
    resolve_charts,
    resolve_star,
    usable_rank,
)
from src.plotting import plot_dimension_sweep, use_project_style
from src.reporting import (
    chart_slug,
    dimension_fits_stem,
    dimension_stem,
    figure_dir,
    pairs_slug,
    rate_slug,
    result_dir,
)

if TYPE_CHECKING:
    from src.decoder import Decoder
    from src.latent import LatentSpace

log = logging.getLogger(__name__)

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / 'config' / 'hydra')
ROOT = Path(__file__).resolve().parents[1]

_TEXT: frozenset[str] = frozenset(
    {
        'dataset',
        'pairs',
        'chart',
        'base_chart',
        'preprocess',
        'strategy',
        'method',
        'decoder',
        'source',
    }
)


# ---------------------------------------------------------------------
# Reading the lambda sweep back
# ---------------------------------------------------------------------


def _cell_rank(row: dict[str, str]) -> int | None:
    """The ``symbols`` column: an int, or ``None`` for an untruncated cell."""
    value = row.get('symbols')
    return int(float(value)) if value else None


def read_back(
    directory: Path,
    pairs_tag: str,
    chart_tag: str,
    rank: int | None,
    n_pilots: int,
    strategy: str,
    methods: list[str],
    swept: str,
    metrics: list[str],
    select: str,
) -> dict[str, Any] | None:
    """The lambda sweep's numbers at one ``(chart, rank, budget)`` cell.

    Matches on the CSVs' columns, never on their names, exactly as
    ``pilot_sweep.py`` does. Several files can describe the same cell --
    a coarse lambda grid and a later fine one, say -- and the row with the
    best ``select`` over all of them wins: that is the lambda the method
    would be run at.

    Returns
    -------
    dict or None
        ``{'lam': float, 'values': {method: {metric: float}}}``, or
        ``None`` when no CSV describes the cell.
    """
    key = f'{select}_mean'
    best: dict[str, str] | None = None
    for path in sorted(directory.glob('lambda_*.csv')):
        with path.open() as handle:
            for row in csv.DictReader(handle):
                if (
                    row.get('pairs') != pairs_tag
                    or row.get('chart') != chart_tag
                    or _cell_rank(row) != rank
                    or int(float(row.get('n_pilots') or -1)) != n_pilots
                    or row.get('strategy') != strategy
                    or not row.get(key)
                ):
                    continue
                if best is None or float(row[key]) > float(best[key]):
                    best = row
    if best is None:
        return None

    values: dict[str, dict[str, float]] = {}
    for name in methods:
        columns = {
            metric: f'{metric}_mean' if name == swept else f'{name}_{metric}'
            for metric in metrics
        }
        missing = [c for c in columns.values() if not best.get(c)]
        if missing:
            raise ValueError(
                f'The lambda sweep cell chart={chart_tag} '
                f'{rate_slug(rank)} N={n_pilots} has no {missing} column, '
                f'so {name!r} cannot be read back. Is it one of that '
                "study's methods, and every metric one of its `eval.metrics`?"
            )
        values[name] = {m: float(best[c]) for m, c in columns.items()}
    return {'lam': float(best['lam']), 'values': values}


def check_ranks_below_budget(rates: list[int], n_pilots: int) -> None:
    """Refuse a rank the pilot budget cannot support.

    With ``N <= k`` pilots the whitened ``k``-chart is rank-deficient and
    RKA's residual stage has ``rank(K) - rank(X) <= 0`` degrees of
    freedom, so it returns Procrustes exactly: the curve would bend for a
    reason that has nothing to do with compression.
    """
    offending = sorted(r for r in rates if r >= n_pilots)
    if offending:
        raise ValueError(
            f'Ranks {offending} are not below the pilot budget '
            f'N={n_pilots}. Every rank on the axis needs N > k: raise '
            '`pilots.n_pilots` or drop those ranks.'
        )


# ---------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------


class PilotSet:
    """One realisation's pilots for every source, selected on first use.

    Drawn the way both other studies draw them -- ``draw_pool`` then
    ``select_pilot_path`` with the same seed for both -- so ``seed=cfg.seed``
    reproduces the lambda sweep's first repeat, and any other seed
    reproduces that seed of ``pilot_sweep.py`` at ``N = n_pilots``. Deferred
    because a resumed run that has nothing to fit and verification switched
    off should not pay for a kernel-herding pass over the training split.
    """

    def __init__(
        self,
        cfg: DictConfig,
        pairs: list[tuple[str, str]],
        agents: dict[str, dict[str, LatentSpace]],
        seed: int,
    ) -> None:
        self._cfg, self._pairs, self._agents = cfg, pairs, agents
        self.seed = int(seed)
        self._indices: dict[str, np.ndarray] | None = None

    def __getitem__(self, source: str) -> np.ndarray:
        if self._indices is None:
            cfg = self._cfg
            n = int(cfg.pilots.n_pilots)
            seed = self.seed
            self._indices = {}
            for name in dict.fromkeys(s for s, _ in self._pairs):
                train = self._agents[name]['train']
                pool = draw_pool(train, cfg.pilots.pool_size, seed)
                log.info(
                    'Selecting %d %s pilots for %s from a pool of %d '
                    '(seed %d).',
                    n,
                    cfg.pilots.strategy,
                    name,
                    pool.size,
                    seed,
                )
                labels = train.labels
                within = select_pilot_path(
                    train.latent[pool],
                    counts=[n],
                    strategy=str(cfg.pilots.strategy),
                    labels=None if labels is None else labels[pool],
                    seed=seed,
                )
                self._indices[name] = pool[within[n]]
        return self._indices[source]


def fit_and_score(
    cfg: DictConfig,
    method_cfg: DictConfig,
    pilots: PilotSet,
    agents: dict[str, dict[str, LatentSpace]],
    pairs: list[tuple[str, str]],
    decoders: dict[str, Decoder | None],
    seed: int | None = None,
) -> tuple[dict[str, float], int]:
    """Fit one configured method on every pair; mean metrics, realised rate.

    The rate is the smallest any pair delivered, so a point is never
    plotted at more symbols than one of its pairs actually sent. ``seed``
    overrides the preset's own (the config seed) for a per-realisation
    fit, as ``pilot_sweep.py`` does; ``None`` leaves the preset untouched,
    which is what makes a fit on the lambda sweep's pilots reproduce it.
    """
    metrics = list(cfg.eval.metrics)
    scores: dict[str, list[float]] = defaultdict(list)
    delivered: list[int] = []
    for source, target in pairs:
        src_train, src_test = agents[source]['train'], agents[source]['test']
        tgt_train, tgt_test = agents[target]['train'], agents[target]['test']
        idx = pilots[source]
        context: dict[str, np.ndarray] = {}
        if cfg.use_local_context:
            context = {
                'src_context': src_train.latent,
                'tgt_context': tgt_train.latent,
            }
        labels = src_train.labels
        aligner: Aligner = build_aligner(
            method_cfg, **({} if seed is None else {'seed': seed})
        )
        aligner.fit(
            src_train.latent[idx],
            tgt_train.latent[idx],
            labels=None if labels is None else labels[idx],
            **context,
        )
        result = alignment_metrics(
            aligner.transform(src_test.latent),
            tgt_test.latent,
            decoder=decoders.get(target),
            labels=tgt_test.labels,
            ks=tuple(cfg.eval.topk),
        )
        for metric in metrics:
            if metric in result:
                scores[metric].append(float(result[metric]))
        delivered.append(int(aligner.transmitted_symbols))
    return {m: float(np.mean(v)) for m, v in scores.items()}, min(delivered)


# ---------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a CSV via a temporary, so a crash cannot truncate it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f'{path.name}.partial')
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with tmp.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def read_rows(path: Path) -> list[dict[str, Any]]:
    """Read records back, with the numerics parsed."""
    if not path.exists():
        return []
    rows = []
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
            for key in ('rank', 'requested', 'symbols', 'n_pilots', 'seed'):
                if isinstance(parsed.get(key), float):
                    parsed[key] = int(parsed[key])
            rows.append(parsed)
    return rows


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------


def expand_charts(
    cfg: DictConfig, max_rank: int
) -> list[tuple[DictConfig, list[tuple[DictConfig, int | None, int]]]]:
    """``[(base chart, [(chart, rank, rate)])]``, one figure per base chart.

    Expanded with :func:`~src.experiment.resolve_charts`, the function the
    lambda sweep expands its ``ranks`` with, so every chart tag here is one
    that sweep wrote.
    """
    plan = []
    for base in cfg.charts:
        single = OmegaConf.merge(
            cfg, {'charts': [OmegaConf.to_container(base, resolve=True)]}
        )
        ranks = [
            (
                chart,
                chart.get('n_components'),
                channel_rate(chart.get('n_components'), max_rank),
            )
            for chart in resolve_charts(single, max_rank)
        ]
        if len({rate for *_, rate in ranks}) < 2:
            raise SystemExit(
                f'Chart {chart_slug(base)!r} resolves to a single rank, and '
                'a dimension sweep needs an axis: set `ranks`, e.g. '
                "'ranks=[16,32,64,128]'."
            )
        plan.append((base, ranks))
    return plan


def summary_table(
    summary: list[dict[str, Any]], metrics: list[str], order: list[str]
) -> str:
    """Methods by rank, per metric, plus what each keeps under compression.

    ``kept`` is the value at the smallest rank over the value at the
    largest: the share of its own uncompressed performance a method still
    delivers at the tightest rate on the axis. It is the number a claim of
    robustness to compression is about, so it is printed rather than left
    to be read off the slope.
    """
    rates = sorted({r['requested'] for r in summary})
    lines = []
    for metric in metrics:
        header = (
            [f'{metric:<10s}'] + [f'k={k:>5d}' for k in rates] + ['  kept']
        )
        lines.append('  '.join(header))
        lines.append('-' * len('  '.join(header)))
        for method in order:
            by_rate = {
                r['requested']: r
                for r in summary
                if r['method'] == method and r.get(metric) is not None
            }
            if not by_rate:
                continue
            cells = []
            for k in rates:
                row = by_rate.get(k)
                if row is None:
                    cells.append(f'{"-":>7s}')
                    continue
                mark = '*' if row['symbols'] != row['requested'] else ' '
                cells.append(f'{row[metric]:.4f}{mark}')
            ends = [by_rate[k][metric] for k in rates if k in by_rate]
            kept = ends[0] / ends[-1] if ends[-1] else float('nan')
            lines.append('  '.join([f'{method:<10s}', *cells, f'{kept:6.1%}']))
        lines.append('')
    if any(r['symbols'] != r['requested'] for r in summary):
        lines.append(
            '* delivered fewer symbols than requested (drawn at the '
            'delivered rate).'
        )
    return '\n'.join(lines).rstrip()


@hydra.main(
    version_base=None, config_path=CONFIG_DIR, config_name='dimension_sweep'
)
def main(cfg: DictConfig) -> None:
    """Read back, fit the rest, verify, and draw one figure per chart."""
    logging.getLogger('src').setLevel(logging.INFO)
    log.info('Configuration:\n%s', OmegaConf.to_yaml(cfg))

    metrics = list(cfg.eval.metrics)
    read_names = [str(m) for m in cfg.read_back.methods]
    fit_names = [str(m) for m in cfg.methods]
    order = read_names + fit_names
    n_pilots = int(cfg.pilots.n_pilots)
    strategy = str(cfg.pilots.strategy)
    decoder_kind = str(cfg.decoder.kind) if cfg.decoder.enabled else 'none'
    seeds = [int(x) for x in cfg.seeds] if cfg.get('seeds') else None
    if seeds and len(set(seeds)) != len(seeds):
        raise SystemExit(f'seeds={seeds} repeats a seed.')
    needs_refit = set(read_names) if seeds else set()
    if cfg.verify.enabled:
        needs_refit.add('procrustes')
    if needs_refit - set(cfg.refit):
        raise SystemExit(
            f'{sorted(needs_refit - set(cfg.refit))} must be refitted '
            '(for `seeds` or `verify`) but have no preset under `refit`.'
        )
    if cfg.verify.enabled and 'procrustes' not in read_names:
        raise SystemExit(
            'verify.enabled=true checks the read-back Procrustes numbers, '
            'but `procrustes` is not in read_back.methods.'
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

    _, max_rank = usable_rank(pairs, agents)
    plan = expand_charts(cfg, max_rank)
    rates = sorted({rate for _, ranks in plan for *_, rate in ranks})
    try:
        check_ranks_below_budget(rates, n_pilots)
        check_budgets_fit_pool(
            [n_pilots],
            cfg.pilots.pool_size,
            min(agents[s]['train'].n_points for s, _ in pairs),
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    pairs_tag = pairs_slug(pairs)
    lam_results = result_dir(
        cfg.output.results, str(cfg.read_back.study), dataset
    )
    results = result_dir(cfg.output.results, str(cfg.output.study), dataset)

    # Everything the figure needs from the lambda sweep, gathered before a
    # single fit: a missing cell is a reason not to start, and it is far
    # cheaper to say so now than after the herding pass.
    backs: dict[tuple[str, int | None], dict[str, Any]] = {}
    missing: list[str] = []
    for _, ranks in plan:
        for chart, rank, _ in ranks:
            cell = read_back(
                lam_results,
                pairs_tag,
                chart_slug(chart),
                rank,
                n_pilots,
                strategy,
                read_names,
                str(cfg.read_back.swept),
                metrics,
                str(cfg.read_back.metric),
            )
            if cell is None:
                missing.append(f'{chart_slug(chart)} N={n_pilots}')
            else:
                backs[(chart_slug(chart), rank)] = cell
    if missing:
        raise SystemExit(
            f'No lambda-sweep cell under {lam_results}/ for '
            f'pairs={pairs_tag}, strategy={strategy}: '
            f'{", ".join(missing)}. Run `just '
            'lambda-sweep` with these ranks and N among its '
            '`pilots.counts` first.'
        )

    log.info(
        'max_rank=%d; %d charts x %d ranks (%s) at N=%d %s pilots; %s.',
        max_rank,
        len(plan),
        len(rates),
        ', '.join(str(r) for r in rates),
        n_pilots,
        strategy,
        f'{len(seeds)} seeds ({", ".join(map(str, seeds))}), refitting '
        f'{", ".join(read_names + fit_names)} on each'
        if seeds
        else f'read back {", ".join(read_names)}, fitting '
        f'{", ".join(fit_names)} on the lambda sweep pilots',
    )

    # The lambda sweep's own pilots: what `verify` refits on, and the one
    # realisation when `seeds` is null.
    lambda_pilots = PilotSet(cfg, pairs, agents, int(cfg.seed))
    use_project_style(ROOT / 'config' / 'plotting' / 'plt.mplstyle')

    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=cfg.wandb.name or f'dims-{dataset}-{pairs_tag}-n{n_pilots}',
        group=cfg.wandb.group,
        job_type='dimension-sweep',
        tags=list(cfg.wandb.tags),
        mode=cfg.wandb.mode,
        config=OmegaConf.to_container(cfg, resolve=True)
        | {'resolved_rates': rates, 'max_rank': max_rank},
    )

    written: list[Path] = []
    try:
        # --- the consistency check -----------------------------------
        if cfg.verify.enabled and not cfg.plot_only:
            metric = 'accuracy' if cfg.decoder.enabled else 'mrr'
            for _, ranks in plan:
                for chart, rank, _ in ranks:
                    tag = chart_slug(chart)
                    refit, _ = fit_and_score(
                        cfg,
                        configure_method(cfg.refit.procrustes, chart, rank),
                        lambda_pilots,
                        agents,
                        pairs,
                        decoders,
                    )
                    stored = backs[(tag, rank)]['values']['procrustes'][metric]
                    gap = abs(refit[metric] - stored)
                    log.info(
                        'verify %s %s: procrustes %s refit=%.4f, lambda '
                        'sweep=%.4f.',
                        tag,
                        rate_slug(rank),
                        metric,
                        refit[metric],
                        stored,
                    )
                    if gap > float(cfg.verify.tolerance):
                        raise SystemExit(
                            f'Procrustes refitted at chart={tag} '
                            f'{rate_slug(rank)} N={n_pilots} scores '
                            f'{metric}={refit[metric]:.4f}, but the lambda '
                            f'sweep recorded {stored:.4f}. The two runs do '
                            'not share pilots, decoder or preprocessing -- '
                            'check `decoder.kind`, `seed`, '
                            '`pilots.pool_size` and `pilots.strategy` '
                            'against the lambda sweep, and whether a '
                            'preset changed since it ran.'
                        )

        # --- the fitted methods -------------------------------------
        # One realisation (`seeds: null`) fits only `methods`, on the lambda
        # sweep's own pilots, and reads RKA and Procrustes back. With seeds,
        # every realisation fits everything -- the read-back methods too, on
        # their chart, RKA at the lambda the sweep selected for that cell.
        # A method on a chart is fitted per (chart, rank); the rest take
        # none, so one fit per rank serves every chart. Records are keyed on
        # (method, chart, rank, seed); a row written before either axis
        # existed reads as chart `none`, seed `None` -- the one realisation.
        charted = {str(m) for m in cfg.charted}
        unknown = charted - set(fit_names)
        if unknown:
            raise SystemExit(
                f'charted={sorted(unknown)} names no method in `methods` '
                f'({", ".join(fit_names)}).'
            )
        on_chart = charted | (set(read_names) if seeds else set())
        fit_order = (read_names if seeds else []) + fit_names
        realisations: list[int | None] = list(seeds) if seeds else [None]

        cells: list[tuple[str, str, DictConfig, int | None, int]] = []
        for name in fit_order:
            if name in on_chart:
                cells += [
                    (name, chart_slug(chart), chart, rank, rate)
                    for _, ranks in plan
                    for chart, rank, rate in ranks
                ]
            else:
                cells += [
                    (name, 'none', OmegaConf.create({}), rate, rate)
                    for rate in rates
                ]

        fits_path = results / (
            dimension_fits_stem(
                dataset, pairs, n_pilots, strategy, decoder_kind
            )
            + '.csv'
        )
        fits = read_rows(fits_path) if (cfg.resume or cfg.plot_only) else []
        have = {
            (
                r['method'],
                r.get('chart') or 'none',
                r['requested'],
                r.get('seed'),
            )
            for r in fits
        }
        pilot_sets: dict[int, PilotSet] = {}
        for seed in realisations:
            for name, tag, chart, rank, rate in cells:
                if (name, tag, rate, seed) in have:
                    continue
                if cfg.plot_only:
                    log.warning(
                        'plot_only=true and %s at chart=%s k=%d seed=%s is '
                        'not on disk; it will be missing from the figure.',
                        name,
                        tag,
                        rate,
                        seed,
                    )
                    continue
                if seed is None:
                    pilots = lambda_pilots
                else:
                    pilots = pilot_sets.setdefault(
                        seed, PilotSet(cfg, pairs, agents, seed)
                    )
                lam = None
                if name in read_names:
                    # Refitted exactly as the lambda sweep fitted it: the
                    # rank goes in as the sweep resolved it, and RKA takes
                    # the lambda that sweep selected for this cell.
                    method_cfg = configure_method(cfg.refit[name], chart, rank)
                    if name == str(cfg.read_back.swept):
                        lam = backs[(tag, rank)]['lam']
                        method_cfg = OmegaConf.merge(
                            method_cfg, {'lam': lam, 'lam_grid': None}
                        )
                else:
                    # The rate goes in as a number even when the chart is
                    # untruncated: a method whose rate knob is not a width
                    # (the anchor count, say) has no "full" setting.
                    method_cfg = configure_method(
                        cfg.methods[name], chart, rate
                    )
                values, delivered = fit_and_score(
                    cfg,
                    method_cfg,
                    pilots,
                    agents,
                    pairs,
                    decoders,
                    seed=seed,
                )
                if delivered != rate:
                    log.warning(
                        '%s asked for %d symbols delivered %d; it is drawn '
                        'at %d.',
                        name,
                        rate,
                        delivered,
                        delivered,
                    )
                log.info(
                    'seed=%s k=%d %s (chart=%s): %s',
                    seed,
                    rate,
                    name,
                    tag,
                    '  '.join(f'{m}={v:.4f}' for m, v in values.items()),
                )
                fits.append(
                    {
                        'dataset': dataset,
                        'pairs': pairs_tag,
                        'strategy': strategy,
                        'n_pilots': n_pilots,
                        'seed': seed,
                        'method': name,
                        'chart': tag,
                        'requested': rate,
                        'symbols': delivered,
                        'lam': lam,
                        **values,
                    }
                )
                write_rows(fits_path, fits)

        # --- one figure per base chart -------------------------------
        wanted_seeds = set(realisations)
        for base, ranks in plan:
            # Every row names the run it came from, so a study that pools
            # several of these CSVs (`pair_average.py`) can filter on
            # columns rather than parse a filename.
            identity = {
                'dataset': dataset,
                'pairs': pairs_tag,
                'base_chart': chart_slug(base),
                'n_pilots': n_pilots,
                'strategy': strategy,
                'decoder': decoder_kind,
            }
            chart_rates = {rate for *_, rate in ranks}
            chart_tags = {chart_slug(chart) for chart, *_ in ranks}

            def as_configured(row: dict[str, Any]) -> bool:
                """Fitted the way the method is configured *now*: a method
                moved on or off a chart leaves its old records cached."""
                tag = row.get('chart') or 'none'
                if row['method'] in on_chart:
                    return tag in chart_tags
                return tag == 'none'

            summary: list[dict[str, Any]] = []
            if not seeds:
                for chart, rank, rate in ranks:
                    cell = backs[(chart_slug(chart), rank)]
                    summary.extend(
                        identity
                        | {
                            'method': name,
                            'chart': chart_slug(chart),
                            'requested': rate,
                            'symbols': rate,
                            'lam': cell['lam']
                            if name == cfg.read_back.swept
                            else None,
                            'source': str(cfg.read_back.study),
                            **cell['values'][name],
                        }
                        for name in read_names
                    )
            summary += [
                identity
                | {
                    'seed': r.get('seed'),
                    'method': r['method'],
                    'chart': r.get('chart') or 'none',
                    'requested': r['requested'],
                    'symbols': r['symbols'],
                    'lam': r.get('lam'),
                    'source': 'fitted',
                    **{m: r.get(m) for m in metrics},
                }
                for r in fits
                if r['method'] in fit_order
                and r['requested'] in chart_rates
                and r.get('seed') in wanted_seeds
                and as_configured(r)
            ]
            rank_of = {name: i for i, name in enumerate(order)}
            summary.sort(
                key=lambda r: (
                    rank_of.get(r['method'], 99),
                    r['requested'],
                    -1 if r.get('seed') is None else r['seed'],
                )
            )

            # The figure and the table read one point per (method, rank):
            # with seeds, the mean over them and its spread. The CSV keeps
            # every seed, so a pooled study can average within the pair.
            shown = summary
            if seeds:
                grouped: dict[tuple[str, int], list[dict[str, Any]]] = (
                    defaultdict(list)
                )
                for row in summary:
                    grouped[(row['method'], row['requested'])].append(row)
                shown = []
                for (method, k), group in grouped.items():
                    point: dict[str, Any] = {
                        'method': method,
                        'requested': k,
                        'symbols': min(g['symbols'] for g in group),
                        'n_seeds': len(group),
                    }
                    for metric in metrics:
                        values = [
                            g[metric]
                            for g in group
                            if g.get(metric) is not None
                        ]
                        if not values:
                            continue
                        point[metric] = point[f'{metric}_mean'] = float(
                            np.mean(values)
                        )
                        point[f'{metric}_std'] = float(np.std(values))
                    shown.append(point)
                shown.sort(
                    key=lambda r: (
                        rank_of.get(r['method'], 99),
                        r['requested'],
                    )
                )
                short = sorted(
                    {
                        (p['method'], p['requested'])
                        for p in shown
                        if p['n_seeds'] < len(seeds)
                    }
                )
                if short:
                    log.warning(
                        'Uneven seed coverage: %d (method, rank) points have '
                        'fewer than %d seeds, e.g. %s.',
                        len(short),
                        len(seeds),
                        short[0],
                    )

            tag = chart_slug(base)
            stem = dimension_stem(
                dataset,
                pairs,
                base,
                n_pilots,
                strategy,
                sorted(chart_rates),
                decoder_kind,
                seeds=seeds,
            )
            write_rows(results / f'{stem}.csv', summary)
            title = cfg.output.get('title') or (
                f'Compression — {dataset}, {base.get("preprocess", "whiten")} '
                f'chart, N={n_pilots} {strategy} pilots'
                + (f', {len(seeds)} seeds' if seeds else '')
            )
            written += plot_dimension_sweep(
                shown,
                metrics=metrics,
                out_path=figure_dir(
                    cfg.output.figures, str(cfg.output.study), dataset, tag
                )
                / stem,
                reference=reference or None,
                hue_of=OmegaConf.to_container(cfg.hue_of, resolve=True),
                title=title,
                formats=tuple(cfg.output.formats),
            )

            print(
                f'\n{tag} — {pairs_tag}, N={n_pilots} {strategy} pilots'
                + (f', mean over seeds {seeds}' if seeds else '')
                + '\n'
            )
            print(summary_table(shown, metrics, order))

            table = wandb.Table(columns=list(summary[0]))
            for row in summary:
                table.add_data(*(row.get(c) for c in table.columns))
            run.log({f'curves/{tag}': table})

        for path in written:
            if path.suffix == '.png':
                run.log({f'figure/{path.stem}': wandb.Image(str(path))})
        run.summary.update(
            {f'native/{k}': v for k, v in reference.items()}
            | {'n_pilots': n_pilots, 'max_rank': max_rank}
        )
    finally:
        run.finish()

    if not written:
        raise SystemExit('No figures produced; nothing was written.')
    if reference:
        print(
            '\nNative RX ceiling: '
            + '  '.join(f'{k}={v:.4f}' for k, v in reference.items())
        )
    print('\nWrote: ' + ', '.join(str(p) for p in written))


if __name__ == '__main__':
    main()

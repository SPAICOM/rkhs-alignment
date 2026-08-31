"""Pilot-budget sweep on Digits, built from the project's own classes.

Reproduces the reference RKA experiment through :mod:`src.alignment` and
:mod:`src.anchors` rather than a private transcription, so the library is
what is actually being exercised. The reference's conventions are matched
exactly, and each one is a configuration choice rather than a rewrite:

- whitening estimated **once** from the whole calibration set, with an
  absolute eigenvalue floor and no shrinkage
  (``LatentScaler(eps_mode='absolute', shrinkage=None)``); the pilots are
  a subset of the already-whitened data, so the aligners themselves run
  with ``preprocess='none'``;
- the receiver's classifier fitted once in that whitened RX space and
  held fixed, so the native ceiling is a single number;
- median-heuristic RBF bandwidth on the pilots, double-centred Gram, and
  the reported RKA accuracy taken as the best over the lambda grid -- an
  oracle, since the maximum is over test accuracy.

Two selection rules, both from :func:`src.alignment.select_pilots`:

- ``round_robin`` -- nested stratified random ordering (classes visited
  round-robin, one unused member each), averaged over ``n_seeds``;
- ``herding`` -- global kernel herding over the whole calibration pool,
  deterministic, hence one curve with no error bar.

Both are prefix-consistent, so :func:`src.alignment.select_pilot_path`
selects once per seed and slices every budget out of it.

Example
-------
    uv run scripts/digits_reference_study.py
    uv run scripts/digits_reference_study.py 'pilots.counts=[20,60,140]'
    uv run scripts/digits_reference_study.py pilots.herding_bandwidth=1.0
"""

from __future__ import annotations

import csv
import logging
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from sklearn.datasets import load_digits
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.alignment import (
    LatentScaler,
    ProcrustesAligner,
    RKHSAligner,
    select_pilot_path,
)
from src.kernels import Kernel
from src.plotting import plot_pilot_efficiency, use_project_style

warnings.filterwarnings('ignore', category=ConvergenceWarning)

log = logging.getLogger(__name__)

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / 'config' / 'hydra')
ROOT = Path(__file__).resolve().parents[1]


def latent_representation(model: MLPClassifier, X: np.ndarray) -> np.ndarray:
    """Hidden-layer representation of one agent."""
    Z = X @ model.coefs_[0] + model.intercepts_[0]
    if model.activation == 'tanh':
        return np.tanh(Z)
    if model.activation == 'relu':
        return np.maximum(Z, 0.0)
    return Z


class Setup:
    """Whitened latents, the fixed RX head, and the native ceiling."""

    def __init__(self, cfg: DictConfig) -> None:
        digits = load_digits()
        X_raw = digits.data.astype(float) / 16.0

        X_model, X_rest, y_model, y_rest = train_test_split(
            X_raw,
            digits.target,
            test_size=cfg.data.holdout_fraction,
            stratify=digits.target,
            random_state=cfg.data.split_seed,
        )
        X_cal, X_test, self.y_cal, self.y_test = train_test_split(
            X_rest,
            y_rest,
            test_size=cfg.data.test_fraction,
            stratify=y_rest,
            random_state=cfg.data.split_seed + 1,
        )

        tx = MLPClassifier(
            hidden_layer_sizes=(cfg.data.tx_dim,),
            activation='tanh',
            solver='lbfgs',
            alpha=1e-3,
            max_iter=600,
            random_state=cfg.data.tx_seed,
        ).fit(X_model, y_model)
        rx = MLPClassifier(
            hidden_layer_sizes=(cfg.data.rx_dim,),
            activation='relu',
            solver='lbfgs',
            alpha=3e-3,
            max_iter=600,
            random_state=cfg.data.rx_seed,
        ).fit(X_model, y_model)

        zt_cal, zt_test = (
            latent_representation(tx, X) for X in (X_cal, X_test)
        )
        zr_model, zr_cal, zr_test = (
            latent_representation(rx, X) for X in (X_model, X_cal, X_test)
        )

        # Fitted once on the whole calibration set. The pilots are a
        # subset of the result, so the aligners take preprocess='none'.
        whiten = lambda: LatentScaler(  # noqa: E731
            'whiten',
            eps=cfg.whitening.eps,
            eps_mode=cfg.whitening.eps_mode,
            shrinkage=cfg.whitening.shrinkage,
        )
        scaler_t, scaler_r = whiten().fit(zt_cal), whiten().fit(zr_cal)
        self.zt_cal = scaler_t.transform(zt_cal)
        self.zt_test = scaler_t.transform(zt_test)
        self.zr_cal = scaler_r.transform(zr_cal)
        self.zr_test = scaler_r.transform(zr_test)

        self.rx_head = LogisticRegression(max_iter=1500, random_state=30).fit(
            scaler_r.transform(zr_model), y_model
        )
        self.native = accuracy_score(
            self.y_test, self.rx_head.predict(self.zr_test)
        )
        self.d_t = self.zt_cal.shape[1]
        self.d_r = self.zr_cal.shape[1]

    def accuracy(self, Z_hat: np.ndarray) -> float:
        return accuracy_score(self.y_test, self.rx_head.predict(Z_hat))


def evaluate_alignment(
    cfg: DictConfig, setup: Setup, pilots: np.ndarray
) -> tuple[float, float, float]:
    """Procrustes and best-over-lambda RKA for one pilot subset."""
    Zx, Zy = setup.zt_cal[pilots], setup.zr_cal[pilots]

    rigid = ProcrustesAligner(preprocess='none').fit(Zx, Zy)
    proc_acc = setup.accuracy(rigid.transform(setup.zt_test))

    best_acc, best_lambda = -np.inf, None
    for lam in np.logspace(
        cfg.lambda_grid.log_min, cfg.lambda_grid.log_max, cfg.lambda_grid.n
    ):
        aligner = RKHSAligner(
            preprocess='none',
            lam=float(lam),
            lam_grid=None,
            ridge_B=cfg.rkhs.ridge_B,
            max_points=None,
        ).fit(Zx, Zy)
        acc = setup.accuracy(aligner.transform(setup.zt_test))
        if acc > best_acc:
            best_acc, best_lambda = acc, float(lam)
    return proc_acc, best_acc, best_lambda


def run_sweep(
    cfg: DictConfig, setup: Setup
) -> tuple[list[dict[str, Any]], dict[int, float]]:
    """Every (strategy, budget, seed) cell."""
    counts = [int(n) for n in cfg.pilots.counts]
    records: list[dict[str, Any]] = []
    best_lambdas: dict[int, float] = {}

    # Kernel herding is deterministic on a fixed pool, so on the whole
    # calibration set it yields one number per budget and no error bar.
    # That is misleading: a single realisation of this curve shows swings
    # of ~0.01 that vanish under averaging. Re-drawing the pool gives it a
    # band on the same footing as the randomised strategy. Set
    # `pool_size: null` to recover the reference's single-pool protocol.
    n_pool = len(setup.zt_cal)
    pool_size = cfg.pilots.pool_size
    resampled = bool(pool_size) and pool_size < n_pool
    n_draws = int(cfg.pilots.n_seeds) if resampled else 1

    for draw in range(n_draws):
        rng = np.random.default_rng(int(cfg.seed) + 1000 * draw)
        pool = (
            np.sort(rng.choice(n_pool, pool_size, replace=False))
            if resampled
            else np.arange(n_pool)
        )
        kernel = Kernel(
            'rbf', bandwidth_scale=cfg.pilots.herding_bandwidth
        ).fit(setup.zt_cal[pool], seed=cfg.seed)
        paths = select_pilot_path(
            setup.zt_cal[pool], counts, strategy='herding', kernel=kernel
        )
        for budget, chosen in paths.items():
            proc, rka, lam = evaluate_alignment(cfg, setup, pool[chosen])
            best_lambdas[budget] = lam
            records.extend(
                {
                    'method': method,
                    'strategy': 'herding',
                    'n_pilots': budget,
                    'repeat': draw,
                    'accuracy': value,
                }
                for method, value in (('procrustes', proc), ('rkhs', rka))
            )
    log.info('kernel herding done (%d pool draws).', n_draws)

    for seed in range(1, int(cfg.pilots.n_seeds) + 1):
        rng = np.random.default_rng(int(cfg.seed) + 1000 * (seed - 1))
        pool = (
            np.sort(rng.choice(n_pool, pool_size, replace=False))
            if resampled
            else np.arange(n_pool)
        )
        paths = select_pilot_path(
            setup.zt_cal[pool],
            counts,
            strategy='round_robin',
            labels=setup.y_cal[pool],
            seed=seed,
        )
        for budget, chosen in paths.items():
            proc, rka, _ = evaluate_alignment(cfg, setup, pool[chosen])
            records.extend(
                {
                    'method': method,
                    'strategy': 'round_robin',
                    'n_pilots': budget,
                    'repeat': seed,
                    'accuracy': value,
                }
                for method, value in (('procrustes', proc), ('rkhs', rka))
            )
        if seed % 10 == 0:
            log.info('round-robin seed %d/%d done.', seed, cfg.pilots.n_seeds)
    return records, best_lambdas


def aggregate(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mean and standard deviation over seeds."""
    grouped: dict[tuple, list[float]] = defaultdict(list)
    for row in records:
        grouped[(row['method'], row['strategy'], row['n_pilots'])].append(
            row['accuracy']
        )
    return [
        {
            'method': method,
            'strategy': strategy,
            'n_pilots': budget,
            'n_repeats': len(values),
            'accuracy_mean': float(np.mean(values)),
            'accuracy_std': float(np.std(values)),
        }
        for (method, strategy, budget), values in sorted(grouped.items())
    ]


def format_table(
    summary: list[dict[str, Any]], best_lambdas: dict[int, float]
) -> str:
    """The reference script's own table layout."""
    lookup = {(r['method'], r['strategy'], r['n_pilots']): r for r in summary}
    budgets = sorted({r['n_pilots'] for r in summary})
    header = (
        '  N | Proc round-robin | RKA round-robin  | Proc herding | '
        'RKA herding | best lambda'
    )
    lines = [header]
    for n in budgets:
        pr = lookup[('procrustes', 'round_robin', n)]
        rr = lookup[('rkhs', 'round_robin', n)]
        ph = lookup[('procrustes', 'herding', n)]
        rh = lookup[('rkhs', 'herding', n)]
        lines.append(
            f'{n:3d} | {pr["accuracy_mean"]:.4f}±{pr["accuracy_std"]:.4f}  | '
            f'{rr["accuracy_mean"]:.4f}±{rr["accuracy_std"]:.4f}  | '
            f'{ph["accuracy_mean"]:12.4f} | {rh["accuracy_mean"]:11.4f} | '
            f'{best_lambdas[n]:.3e}'
        )
    return '\n'.join(lines)


@hydra.main(
    version_base=None, config_path=CONFIG_DIR, config_name='digits_reference'
)
def main(cfg: DictConfig) -> None:
    """Sweep the pilot budget through the project's aligners."""
    logging.getLogger('src').setLevel(logging.WARNING)
    log.info('Configuration:\n%s', OmegaConf.to_yaml(cfg))

    setup = Setup(cfg)
    log.info(
        'TX d_t=%d, RX d_r=%d | calibration %d, test %d',
        setup.d_t,
        setup.d_r,
        len(setup.zt_cal),
        len(setup.y_test),
    )
    print(f'Native RX accuracy: {setup.native:.4f}')

    records, best_lambdas = run_sweep(cfg, setup)
    summary = aggregate(records)

    out = Path(cfg.output_dir) / 'digits_pilot_study'
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.with_suffix('.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    use_project_style(ROOT / 'config' / 'plotting' / 'plt.mplstyle')
    written = plot_pilot_efficiency(
        summary,
        metrics=['accuracy'],
        out_path=out,
        reference={'accuracy': setup.native},
        title=cfg.plot_title
        or f'Digits — post-alignment accuracy ($d_t$ = {setup.d_t})',
    )

    print('\n' + format_table(summary, best_lambdas))
    print(
        '\nWrote: '
        + ', '.join(str(p) for p in [*written, out.with_suffix('.csv')])
    )


if __name__ == '__main__':
    main()

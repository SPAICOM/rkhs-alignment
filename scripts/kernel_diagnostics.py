"""Why does the unconstrained kernel beat RKA on accuracy, and where not?

Four read-only studies behind the `kernel_ablation` figures. None of them
writes a result CSV: each answers one question about the mechanism and
prints a table, so the claims in the paper can be reproduced from one
command rather than from a notebook.

``study=mechanism``
    The class geometry of the transported test latents, against the
    receiver's own latents on the same chart. Least squares predicts the
    conditional mean, so it shrinks: smaller scale, smaller within-class
    spread, a higher Fisher ratio than the receiver itself -- which is
    what a linear probe rewards and what retrieval punishes. RKA cannot
    shrink, because ``Q`` is an isometry and ``G X^T = 0`` freezes the
    linear part.

``study=whitening``
    Why that shrinkage survives whitening. Both charts have unit
    covariance by construction, but the cross-covariance still has
    singular values below one -- the canonical correlations. Least
    squares keeps them, Procrustes replaces every one with 1.

``study=stationarity``
    Whether the two-stage solve is the joint optimum. Alternating the two
    closed-form blocks leaves RKA where it is (the constraint decouples
    them) and keeps descending without the constraint, which is the
    identifiability argument as a measurement.

``study=screen``
    Where RKA could win: prices every (transmitter, receiver) pair by its
    rms canonical correlation, then fits RKA and pure kernel alignment on
    the least and most predictable pairs.

Examples
--------
    just kernel-diagnostics
    just kernel-diagnostics study=whitening
    just kernel-diagnostics study=screen screen.n_extremes=3
    uv run scripts/kernel_diagnostics.py study=stationarity n_pilots=4096
"""

from __future__ import annotations

import itertools
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.alignment import alignment_metrics, select_pilot_path
from src.alignment.procrustes import orthogonal_procrustes
from src.alignment.rkhs import _ResidualSolver
from src.experiment import (
    build_aligner,
    build_decoder,
    configure_method,
    draw_pool,
    load_named_agents,
)

if TYPE_CHECKING:
    from src.latent import LatentSpace

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = str(ROOT / 'config' / 'hydra')


# ---------------------------------------------------------------------
# Shared pieces
# ---------------------------------------------------------------------


def herded(
    space: LatentSpace, n_pilots: int, pool_size: int, seed: int
) -> np.ndarray:
    """Indices of ``n_pilots`` kernel-herded pilots, as the studies use."""
    pool = draw_pool(space, pool_size, seed)
    within = select_pilot_path(
        space.latent[pool], counts=[n_pilots], strategy='herding', seed=seed
    )
    return pool[within[n_pilots]]


def fitted(
    cfg: DictConfig,
    name: str,
    pilots: np.ndarray,
    src: dict[str, LatentSpace],
    tgt: dict[str, LatentSpace],
    lam: float | None = None,
):
    """One method of ``cfg.methods``, at a pinned ``lam`` and rank."""
    chart = OmegaConf.create({'preprocess': 'whiten'})
    method = configure_method(cfg.methods[name], chart, int(cfg.symbols))
    if lam is not None:
        method = OmegaConf.merge(
            method,
            {
                'lam': float(lam),
                'lam_grid': None,
                'bandwidth_scale': float(cfg.bandwidth_scale),
            },
        )
    return build_aligner(method, seed=int(cfg.seed)).fit(
        src['train'].latent[pilots], tgt['train'].latent[pilots]
    )


def table(rows: list[tuple], headers: tuple[str, ...], width: int = 10) -> str:
    """Fixed-width table; the first column is left-aligned text."""
    head = f'{headers[0]:<14s}' + ''.join(
        f'{h:>{width}s}' for h in headers[1:]
    )
    body = [
        f'{row[0]!s:<14s}'
        + ''.join(
            f'{v:{width}.4f}' if isinstance(v, float) else f'{v:>{width}}'
            for v in row[1:]
        )
        for row in rows
    ]
    return '\n'.join([head, '-' * len(head), *body])


def spread(Z: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    """Total, within-class and between-class spread of the rows of ``Z``."""
    Z = Z - Z.mean(axis=0)
    classes = np.unique(labels)
    means = np.stack([Z[labels == c].mean(axis=0) for c in classes])
    within = float(
        sum(
            np.sum((Z[labels == c] - means[i]) ** 2)
            for i, c in enumerate(classes)
        )
    )
    counts = np.array([np.sum(labels == c) for c in classes])
    return (
        float(np.sum(Z**2)),
        within,
        float(np.sum(counts[:, None] * means**2)),
    )


# ---------------------------------------------------------------------
# The studies
# ---------------------------------------------------------------------


def study_mechanism(cfg: DictConfig, agents: dict, pair: tuple[str, str]):
    """Class geometry of each method's transported test latents."""
    tx, rx = pair
    src, tgt = agents[tx], agents[rx]
    pilots = herded(
        src['train'],
        int(cfg.n_pilots),
        int(cfg.pilots.pool_size),
        int(cfg.seed),
    )
    decoder = build_decoder(cfg, tgt['train'])
    labels = tgt['test'].labels
    lam = float(cfg.mechanism.lam)

    rows = []
    for name in cfg.mechanism.methods:
        name = str(name)
        aligner = fitted(
            cfg,
            name,
            pilots,
            src,
            tgt,
            lam if name in cfg.kernel_methods else None,
        )
        Y_hat = aligner.transform(src['test'].latent)
        scores = alignment_metrics(
            Y_hat,
            tgt['test'].latent,
            decoder=decoder,
            labels=labels,
            ks=(1, 5),
        )
        # The receiver's own latents on this chart: the ceiling for any
        # map that spends the same number of symbols.
        scaler = aligner.scaler_tgt_
        Y_k = scaler.inverse_transform(scaler.transform(tgt['test'].latent))
        total, within, between = spread(scaler.transform(Y_hat), labels)
        t_total, t_within, t_between = spread(scaler.transform(Y_k), labels)
        rows.append(
            (
                name,
                scores['accuracy'],
                scores['mrr'],
                float(np.sqrt(total / t_total)),
                within / t_within,
                between / t_between,
                (between / within) / (t_between / t_within),
                float(
                    np.sum(
                        (scaler.transform(Y_hat) - scaler.transform(Y_k)) ** 2
                    )
                    / t_total
                ),
            )
        )

    print(
        f'\nN={int(cfg.n_pilots)}, k={int(cfg.symbols)}, lam={lam:g}. Ratios '
        'are against the receiver on the same chart (1.0 = unchanged).\n'
    )
    print(
        table(
            rows,
            (
                'method',
                'acc',
                'mrr',
                'scale',
                'within',
                'between',
                'fisher',
                'nmse',
            ),
        )
    )


def study_whitening(cfg: DictConfig, agents: dict, pair: tuple[str, str]):
    """Canonical correlations, and the output scale they imply."""
    tx, rx = pair
    src, tgt = agents[tx], agents[rx]
    pilots = herded(
        src['train'],
        int(cfg.n_pilots),
        int(cfg.pilots.pool_size),
        int(cfg.seed),
    )
    aligner = fitted(cfg, 'procrustes', pilots, src, tgt)

    rows = []
    for split, X, Y in (
        ('pilots', src['train'].latent[pilots], tgt['train'].latent[pilots]),
        ('test', src['test'].latent, tgt['test'].latent),
    ):
        Zx = aligner.scaler_src_.transform(X)
        Zy = aligner.scaler_tgt_.transform(Y)
        rho = np.linalg.svd(Zy.T @ Zx / Zx.shape[0], compute_uv=False)
        rows.append(
            (
                split,
                float(np.abs(1 - Zx.var(axis=0)).mean()),
                float(np.abs(1 - Zy.var(axis=0)).mean()),
                float(rho.max()),
                float(rho.mean()),
                float(np.sqrt((rho**2).mean())),
                float(rho.min()),
            )
        )

    print(
        f'\nk={int(cfg.symbols)}: both charts are whitened, yet a '
        'least-squares map keeps only rms(rho) of the target scale, while '
        'Procrustes keeps 1.\n'
    )
    print(
        table(
            rows,
            (
                'split',
                'src |1-var|',
                'tgt |1-var|',
                'max',
                'mean',
                'rms',
                'min',
            ),
            width=12,
        )
    )


def study_stationarity(cfg: DictConfig, agents: dict, pair: tuple[str, str]):
    """Alternate the two closed-form blocks, with and without the
    constraint."""
    tx, rx = pair
    src, tgt = agents[tx], agents[rx]
    pilots = herded(
        src['train'],
        int(cfg.n_pilots),
        int(cfg.pilots.pool_size),
        int(cfg.seed),
    )
    lam = float(cfg.stationarity.lam)
    base = fitted(cfg, 'krr', pilots, src, tgt, lam)
    kernel, scaler_x, scaler_y = (
        base.kernel_spec,
        base.scaler_src_,
        base.scaler_tgt_,
    )
    X = scaler_x.transform(src['train'].latent[pilots])
    Y = scaler_y.transform(tgt['train'].latent[pilots])
    n = X.shape[0]

    for constrained in (False, True):
        print(
            f'\n{"RKA (G^T X = 0)" if constrained else "RKA free"}: '
            'alternating the rigid and residual blocks\n'
        )
        rows = []
        Q = orthogonal_procrustes(X, Y).Q
        previous = None
        for it in range(int(cfg.stationarity.iterations)):
            solver = _ResidualSolver(
                X,
                Y - X @ Q.T,
                kernel=kernel,
                center=True,
                ridge_B=1e-8,
                orthogonal=constrained,
                lam_scaling=str(cfg.methods.rkhs.lam_scaling),
            )
            out = solver.solve(lam)
            G, A = out['G'], out['A']
            penalty = (
                lam
                * solver.lam_scale_
                * float(
                    np.sum(
                        A * (solver.V @ (solver.s[:, None] * (solver.V.T @ A)))
                    )
                )
            )
            rows.append(
                (
                    it,
                    float(np.sum((Y - X @ Q.T - G) ** 2) / n + penalty),
                    float(
                        np.linalg.norm(X.T @ G)
                        / max(np.linalg.norm(X) * np.linalg.norm(G), 1e-12)
                    ),
                    0.0
                    if previous is None
                    else float(np.linalg.norm(Q - previous)),
                )
            )
            previous = Q
            Q = orthogonal_procrustes(X, Y - G).Q
        print(table(rows, ('iter', 'objective', '|X^T G|', '|dQ|'), width=14))


def study_screen(cfg: DictConfig, agents: dict, _pair: tuple[str, str]):
    """Price every pair by rms(rho), then fit RKA and krr on the extremes."""
    receivers = [str(m) for m in cfg.screen.receivers]
    transmitters = [str(m) for m in cfg.screen.transmitters]
    k, n = int(cfg.symbols), int(cfg.n_pilots)

    pilots = {
        tx: herded(
            agents[tx]['train'], n, int(cfg.pilots.pool_size), int(cfg.seed)
        )
        for tx in transmitters
    }

    rho: dict[tuple[str, str], float] = {}
    for rx, tx in itertools.product(receivers, transmitters):
        aligner = fitted(cfg, 'procrustes', pilots[tx], agents[tx], agents[rx])
        Zx = aligner.scaler_src_.transform(agents[tx]['test'].latent)
        Zy = aligner.scaler_tgt_.transform(agents[rx]['test'].latent)
        s = np.linalg.svd(Zy.T @ Zx / Zx.shape[0], compute_uv=False)
        rho[(tx, rx)] = float(np.sqrt((s**2).mean()))

    order = sorted(rho, key=rho.get)
    print(
        f'\nrms canonical correlation at k={k} (low = much to shrink away)\n'
    )
    for tx, rx in order:
        print(f'  {rho[(tx, rx)]:.3f}  {tx:<45s} -> {rx}')

    extremes = int(cfg.screen.n_extremes)
    chosen = order[:extremes] + order[-extremes:]
    decoders = {
        rx: build_decoder(cfg, agents[rx]['train']) for rx in receivers
    }
    grid = [float(v) for v in cfg.screen.lam_grid]

    print(f'\n\nfitting the extremes at k={k}, N={n}\n')
    rows = []
    for tx, rx in chosen:
        best: dict[str, dict[str, float]] = {}
        for name in ('rkhs', 'krr'):
            aligner = fitted(
                cfg, name, pilots[tx], agents[tx], agents[rx], grid[0]
            )
            scored = []
            for lam in grid:
                aligner.set_lam(lam)
                scored.append(
                    alignment_metrics(
                        aligner.transform(agents[tx]['test'].latent),
                        agents[rx]['test'].latent,
                        decoder=decoders[rx],
                        labels=agents[rx]['test'].labels,
                        ks=(1, 5),
                    )
                )
            best[name] = {
                m: max(s[m] for s in scored) for m in ('accuracy', 'top1')
            }
        rows.append(
            (
                f'{tx.split(".")[0][:12]}->{rx.split(".")[0][:12]}',
                rho[(tx, rx)],
                best['rkhs']['accuracy'],
                best['krr']['accuracy'],
                best['rkhs']['accuracy'] - best['krr']['accuracy'],
                best['rkhs']['top1'],
                best['krr']['top1'],
                best['rkhs']['top1'] - best['krr']['top1'],
            )
        )
    print(
        table(
            rows,
            (
                'pair',
                'rho',
                'RKA acc',
                'krr acc',
                'd_acc',
                'RKA top1',
                'krr top1',
                'd_top1',
            ),
        )
    )


STUDIES = {
    'mechanism': study_mechanism,
    'whitening': study_whitening,
    'stationarity': study_stationarity,
    'screen': study_screen,
}


@hydra.main(
    version_base=None, config_path=CONFIG_DIR, config_name='kernel_diagnostics'
)
def main(cfg: DictConfig) -> None:
    """Run one diagnostic and print its table."""
    logging.getLogger('src').setLevel(logging.WARNING)
    study = str(cfg.study)
    if study not in STUDIES:
        raise SystemExit(
            f'study={study!r} is not one of {", ".join(sorted(STUDIES))}.'
        )

    pair = (str(cfg.transmitter), str(cfg.receiver))
    names = set(pair)
    if study == 'screen':
        names |= {str(m) for m in cfg.screen.receivers}
        names |= {str(m) for m in cfg.screen.transmitters}
    log.info('Loading %d encoders...', len(names))
    agents = load_named_agents(cfg, sorted(names))

    STUDIES[study](cfg, agents, pair)


if __name__ == '__main__':
    main()

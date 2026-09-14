"""Redraw the compression figures from stored results, without refitting.

The sweep writes its summary to ``compression_<dataset>.csv``; this reads
that back and re-renders the per-metric figures, optionally dropping
methods. Refitting to change which curves are drawn would be an hour and
a half of compute to answer a question the CSV already contains.

    uv run scripts/replot_compression.py
    uv run scripts/replot_compression.py --drop kcca pga_procrustes pca_rkhs
    uv run scripts/replot_compression.py --suffix _full --drop kcca
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.plotting import plot_compression_facets, use_project_style

ROOT = Path(__file__).resolve().parents[1]

SWEEP_CONFIG = ROOT / 'config' / 'hydra' / 'compression_sweep.yaml'


def load_hue_of(path: Path) -> dict[str, str]:
    """The sweep's `hue_of` map: colour is the family, style the member.

    Read from the sweep config rather than restated here, so a method
    added there cannot silently arrive without a hue family and push the
    figure past the four-slot validated palette. Parsed as plain YAML --
    the key is a literal, so Hydra composition is not needed.
    """
    with path.open() as handle:
        loaded = yaml.safe_load(handle) or {}
    return dict(loaded.get('hue_of') or {})


PRETTY = {
    'mrr': 'Mean reciprocal rank',
    'top1': 'Top-1 retrieval',
    'top5': 'Top-5 retrieval',
    'accuracy': 'Post-alignment accuracy',
}
NUMERIC = [
    'requested_symbols',
    'symbols',
    'paired_used',
    'map_params',
    'capacity',
    'n_repeats',
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--csv', default='figures/compression_cifar10.csv')
    ap.add_argument('--drop', nargs='*', default=[], help='methods to omit')
    ap.add_argument(
        '--metrics',
        nargs='*',
        default=['mrr', 'top1', 'top5', 'accuracy'],
    )
    ap.add_argument(
        '--suffix', default='', help='appended to each output stem'
    )
    ap.add_argument(
        '--native',
        nargs='*',
        default=[],
        metavar='PAIR=ACC',
        help=(
            'receiver-on-its-own-latents accuracy per pair, drawn as a '
            'ceiling on the accuracy panel; the sweep prints these as '
            '"native RX"'
        ),
    )
    args = ap.parse_args()
    native = {}
    for item in args.native:
        label, _, value = item.partition('=')
        native[label] = float(value)

    path = Path(args.csv)
    rows = list(csv.DictReader(path.open()))
    if not rows:
        raise SystemExit(f'{path} is empty.')

    drop = set(args.drop)
    kept = [r for r in rows if r['method'] not in drop]
    if not kept:
        raise SystemExit('every row was dropped; nothing to plot.')

    # csv gives strings; the plotting code expects the numbers back.
    metric_cols = {c for m in args.metrics for c in (m, f'{m}_std')}
    for r in kept:
        for key in list(r):
            if key in NUMERIC or key in metric_cols:
                r[key] = float(r[key])

    methods = sorted({r['method'] for r in kept})
    facets = list(dict.fromkeys(r['pair'] for r in kept))
    print(
        f'{len(kept)} of {len(rows)} rows, {len(methods)} methods: '
        + ', '.join(methods)
    )
    if drop:
        print(f'dropped: {", ".join(sorted(drop))}')

    use_project_style(ROOT / 'config' / 'plotting' / 'plt.mplstyle')
    dataset = path.stem.replace('compression_', '')
    written = []
    for metric in args.metrics:
        written += plot_compression_facets(
            kept,
            metric=metric,
            out_path=path.with_name(f'{path.stem}_{metric}{args.suffix}'),
            facets=facets,
            reference=native if metric == 'accuracy' and native else None,
            hue_of={
                k: v
                for k, v in load_hue_of(SWEEP_CONFIG).items()
                if k not in drop and v not in drop
            },
            title=f'{PRETTY.get(metric, metric)} vs compression — {dataset}',
        )
    print('Wrote: ' + ', '.join(str(p) for p in written))


if __name__ == '__main__':
    main()

"""Where a study's outputs land, and what they are called.

Three entry points write everything that ends up in the paper --
``scripts/lambda_sweep.py``, ``scripts/pilot_sweep.py`` and
``scripts/dimension_sweep.py`` -- and all are swept over the same axes:
dataset, encoder pair, preprocessing chart, channel rate, pilot budget.
The layout below follows from two decisions.

Figures and data live in separate trees (``figures/`` and ``results/``)
that share the same subdirectory structure, because they are consumed
differently: figures are browsed by eye, CSVs are globbed and filtered by
a script. Mixing them is what made the flat ``figures/`` directory
unreadable at four hundred files.

The *filename* then carries the whole configuration, even the part the
directory already implies. A figure is copied out of this tree the moment
it is used -- into a manuscript, a slide, an email -- and the path does
not survive that move; the name is the only provenance left. The
directories only group.

Nothing here parses a name back. ``pilot_sweep.py`` has to find the
lambda that ``lambda_sweep.py`` measured at a given (chart, rate, budget)
and it reads that off the *columns* of the CSVs, which every writer here
is required to carry. Filenames are for people.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

__all__ = [
    'budget_slug',
    'chart_slug',
    'dimension_fits_stem',
    'dimension_stem',
    'figure_dir',
    'lam_slug',
    'lambda_stem',
    'model_slug',
    'pairs_slug',
    'pilot_stem',
    'ranks_slug',
    'rate_slug',
    'result_dir',
    'strategies_slug',
]

# Chart keys that earn a place in the filename. `n_components` is
# deliberately absent: it *is* the channel rate, and `rate_slug` already
# reports the effective one, so repeating it here would print `k384`
# twice in every name.
_CHART_ABBREV: dict[str, str] = {'eps': 'eps', 'shrinkage': 'shr'}

# Anything outside this set is replaced, so a stem is safe on every
# filesystem and survives a shell glob without quoting. The dot is not in
# it: a stem carrying one is truncated by anything reaching for
# `Path.with_suffix`, and a name is the wrong place to find that out.
_UNSAFE = re.compile(r'[^A-Za-z0-9_+-]+')

# timm's ViT names end `..._patch<k>_<resolution>`. The resolution is
# constant across a SEMASIA config, so it distinguishes nothing.
_RESOLUTION = re.compile(r'(_p\d+)_\d{3}$')


def _real(value: float) -> str:
    """Format one number for a filename: compact, and free of dots.

    ``1e-08`` stays as it is; ``0.0001`` becomes ``0p0001``. The ``p``
    is the usual electronics convention for a decimal point, and it
    keeps the whole stem in one dot-free token.
    """
    return f'{value:g}'.replace('.', 'p')


def _num(value: Any) -> str:
    """Format one scalar for a filename."""
    if value is None:
        return 'none'
    if isinstance(value, bool):
        return 'on' if value else 'off'
    if isinstance(value, float):
        return _real(value)
    return _UNSAFE.sub('-', str(value))


def model_slug(name: str) -> str:
    """Compact tag for one encoder, e.g. ``vit_small_p16``.

    Drops the pre-training tag after the dot and the input resolution.
    What a filename has to answer is *which encoders were in the run*,
    and neither of those two fields separates the encoders in any config
    here -- while both cost a dozen characters in a name that already
    carries five other axes.

    The resolution is only stripped when it follows a patch size, so a
    model whose trailing digits are part of its identity (``regnety_016``)
    keeps them.
    """
    head = str(name).split('.', 1)[0].replace('patch', 'p')
    return _UNSAFE.sub('-', _RESOLUTION.sub(r'\1', head))


def pairs_slug(pairs: Sequence[tuple[str, str]]) -> str:
    """Tag for the encoder pairs a run averages over.

    The star topology these studies use has one receiver and several
    transmitters, and the receiver is the interesting half -- it fixes
    ``d_r``, and therefore the rate the whole sweep is written against.
    So it is always named; the transmitters are named only when there is
    one of them to name.
    """
    receivers = {t for _, t in pairs}
    sources = [s for s, _ in pairs]
    if len(receivers) != 1:
        return f'{len(pairs)}pairs'
    receiver = model_slug(next(iter(receivers)))
    if len(sources) == 1:
        return f'{model_slug(sources[0])}-to-{receiver}'
    return f'{len(sources)}tx-to-{receiver}'


def chart_slug(chart: Mapping[str, Any]) -> str:
    """Tag for one preprocessing configuration.

    ``name`` wins when the config sets one, so a chart that differs in
    something this does not print can still be told apart in the tree.
    """
    if chart.get('name'):
        return _UNSAFE.sub('-', str(chart['name']))
    parts = [_num(chart.get('preprocess', 'whiten'))]
    parts += [
        f'{abbrev}{_num(chart[key])}'
        for key, abbrev in _CHART_ABBREV.items()
        if key in chart
    ]
    return '-'.join(parts)


def rate_slug(symbols: int | None) -> str:
    """Tag for the channel rate: ``k384``, or ``kfull`` when untruncated.

    ``None`` means the run never applied a rank -- either nothing was
    asked for, or the method declares no ``rate_key`` to write it into.
    Naming such a run ``k384`` would claim a truncation that did not
    happen.
    """
    return 'kfull' if symbols is None else f'k{int(symbols)}'


def lam_slug(grid: Sequence[float]) -> str:
    """Tag for a lambda grid: its span and how finely it is read."""
    values = sorted(float(v) for v in grid)
    return f'lam{_real(values[0])}to{_real(values[-1])}x{len(values)}'


def budget_slug(counts: Sequence[int]) -> str:
    """Tag for a pilot-budget axis: its span and how many points."""
    values = sorted(int(c) for c in counts)
    return f'N{values[0]}to{values[-1]}x{len(values)}'


def ranks_slug(rates: Sequence[int]) -> str:
    """Tag for a compression axis: its span and how many points."""
    values = sorted(int(r) for r in rates)
    return f'k{values[0]}to{values[-1]}x{len(values)}'


def strategies_slug(strategies: Sequence[str]) -> str:
    """Tag for the pilot designs a comparison puts on one figure."""
    return '-'.join(_UNSAFE.sub('-', str(s)) for s in strategies)


def lambda_stem(
    dataset: str,
    pairs: Sequence[tuple[str, str]],
    chart: Mapping[str, Any],
    symbols: int | None,
    n_pilots: int,
    lam_grid: Sequence[float],
) -> str:
    """Name of one cell of the lambda sweep: one chart, one budget."""
    return '_'.join(
        (
            'lambda',
            _UNSAFE.sub('-', str(dataset)),
            pairs_slug(pairs),
            chart_slug(chart),
            rate_slug(symbols),
            f'n{int(n_pilots)}',
            lam_slug(lam_grid),
        )
    )


def pilot_stem(
    dataset: str,
    pairs: Sequence[tuple[str, str]],
    chart: Mapping[str, Any],
    symbols: int | None,
    counts: Sequence[int],
    strategies: Sequence[str],
) -> str:
    """Name of one pilot-budget comparison: one chart, the whole axis."""
    return '_'.join(
        (
            'pilots',
            _UNSAFE.sub('-', str(dataset)),
            pairs_slug(pairs),
            chart_slug(chart),
            rate_slug(symbols),
            budget_slug(counts),
            strategies_slug(strategies),
        )
    )


def dimension_stem(
    dataset: str,
    pairs: Sequence[tuple[str, str]],
    chart: Mapping[str, Any],
    n_pilots: int,
    strategy: str,
    rates: Sequence[int],
    decoder: str,
) -> str:
    """Name of one compression comparison: one chart, one budget.

    ``chart`` is the base chart the ranks expand, not one expanded rank:
    the figure spans the whole rank axis. The decoder is named because
    the accuracy panel is read off it, and a linear and an MLP head put
    the same curves at different heights.
    """
    return '_'.join(
        (
            'dims',
            _UNSAFE.sub('-', str(dataset)),
            pairs_slug(pairs),
            chart_slug(chart),
            f'n{int(n_pilots)}',
            strategies_slug([strategy]),
            ranks_slug(rates),
            f'dec-{_UNSAFE.sub("-", str(decoder))}',
        )
    )


def dimension_fits_stem(
    dataset: str,
    pairs: Sequence[tuple[str, str]],
    n_pilots: int,
    strategy: str,
    decoder: str,
) -> str:
    """Name of the fitted methods' records behind a compression figure.

    No chart and no method list: every row carries both, so one record
    serves the whole figure family, and adding a method to a study fits
    that method alone rather than orphaning the records already on disk.
    """
    return '_'.join(
        (
            'dimsfits',
            _UNSAFE.sub('-', str(dataset)),
            pairs_slug(pairs),
            f'n{int(n_pilots)}',
            strategies_slug([strategy]),
            f'dec-{_UNSAFE.sub("-", str(decoder))}',
        )
    )


def figure_dir(
    root: str | Path, study: str, dataset: str, chart: str | None = None
) -> Path:
    """``<root>/<study>/<dataset>[/<chart>]``, created on demand."""
    path = Path(root) / study / _UNSAFE.sub('-', str(dataset))
    if chart is not None:
        path = path / _UNSAFE.sub('-', chart)
    path.mkdir(parents=True, exist_ok=True)
    return path


def result_dir(root: str | Path, study: str, dataset: str) -> Path:
    """``<root>/<study>/<dataset>``, created on demand.

    Flatter than the figure tree on purpose: this is the directory
    ``pilot_sweep.py`` globs for every lambda cell of a dataset, and a
    per-chart subdirectory would only make it re-walk them.
    """
    path = Path(root) / study / _UNSAFE.sub('-', str(dataset))
    path.mkdir(parents=True, exist_ok=True)
    return path

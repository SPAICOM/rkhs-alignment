"""Figures for the alignment studies.

One figure, one panel per metric, plotting each metric against the pilot
budget ``N``. Two factors are on screen at once, so they get separate
channels: **colour is the alignment method**, assigned in fixed order and
never cycled, while **line style and marker are the pilot-selection
strategy**. Identity is therefore never carried by colour alone, and a
run comparing four methods under one strategy reads the same way as one
comparing two methods under two strategies.

The shaded band is +/- 1 standard deviation over the independent pilot
realisations; the dotted rule on the accuracy panel is the receiver's own
native accuracy, the ceiling any alignment is working toward.
"""

from __future__ import annotations

import logging
from pathlib import Path
from shutil import which
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

log = logging.getLogger(__name__)

__all__ = [
    'METHOD_COLORS',
    'plot_compression_facets',
    'plot_compression_study',
    'plot_kernel_study',
    'plot_method_comparison',
    'plot_pilot_efficiency',
    'plot_regularization_study',
    'use_project_style',
]

# Validated categorical palette (light surface): worst all-pairs CVD
# dE 9.2, normal-vision dE 16.3.
_PALETTE: tuple[str, ...] = ('#2a78d6', '#eb6834', '#1baf7a', '#4a3aa7')

# Colour belongs to the method, not to its position in a given run: a
# figure that drops one method must not repaint the others.
METHOD_COLORS: dict[str, str] = {
    'procrustes': _PALETTE[0],
    'rkhs': _PALETTE[1],
    'direct_mlp': _PALETTE[2],
    'residual_mlp': _PALETTE[3],
}

# Second channel: the pilot-selection strategy.
_STRATEGY_STYLES: tuple[tuple[str, str], ...] = (
    ('-', 'o'),
    ('--', 's'),
    (':', '^'),
    ('-.', 'D'),
)

_INK = '#0b0b0b'
_MUTED = '#52514e'
_SURFACE = '#fcfcfb'
_GRID = '#e6e5e1'
# Neutral fill for the method-comparison bars. Identity there is carried
# by the axis label of each row, so the only thing colour has to say is
# "this one is ours"; a second hue would imply a grouping that is not in
# the data.
_NEUTRAL_FILL = '#b9b7b1'

_PRETTY = {
    'accuracy': 'Post-alignment accuracy',
    'mrr': 'Mean reciprocal rank',
    'top1': 'Top-1 retrieval',
    'top5': 'Top-5 retrieval',
    'nmse': 'Normalised MSE',
    'residual_r2': 'Residual explained (held-out $R^2$)',
    'cosine': 'Cosine similarity',
    'r2': r'$R^2$',
}


def use_project_style(style_path: str | Path | None = None) -> None:
    """Apply the repo's matplotlib style, without requiring LaTeX.

    ``config/plotting/plt.mplstyle`` sets ``text.usetex: True``, which
    raises at render time on a machine with no TeX installation. The
    style is applied either way and that one key is turned back off when
    TeX is missing.
    """
    if style_path is not None and Path(style_path).exists():
        plt.style.use(str(style_path))
    if mpl.rcParams.get('text.usetex') and which('latex') is None:
        log.info('No LaTeX installation found; rendering with mathtext.')
        mpl.rcParams['text.usetex'] = False


def plot_pilot_efficiency(
    summary: list[dict[str, Any]],
    metrics: list[str],
    out_path: str | Path,
    reference: dict[str, float] | None = None,
    title: str | None = None,
    shade_below: tuple[float, str] | None = None,
    formats: tuple[str, ...] = ('png', 'pdf'),
) -> list[Path]:
    """Plot each metric against the pilot budget.

    Parameters
    ----------
    summary : list[dict]
        One row per ``(method, strategy, n_pilots)``, carrying
        ``f'{metric}_mean'`` and ``f'{metric}_std'`` for every metric.
    metrics : list[str]
        Metrics to panel, in order.
    out_path : str | Path
        Destination stem; one file per entry of ``formats``.
    reference : dict[str, float], optional
        Horizontal reference per metric (e.g. the receiver's native
        accuracy).
    title : str, optional
        Figure title.
    shade_below : tuple[float, str], optional
        ``(x, label)``: shade the region left of ``x`` and annotate it.
        Used to mark a regime boundary, e.g. the pilot budget below which
        the orthogonality constraint admits only the zero residual.
    formats : tuple[str, ...], default=('png', 'pdf')
        Extensions to write.

    Returns
    -------
    list[Path]
        The files written.
    """
    methods = _ordered(summary, 'method')
    strategies = _ordered(summary, 'strategy')
    colors = _assign_colors(methods)
    styles = dict(zip(strategies, _STRATEGY_STYLES))
    budgets = sorted({r['n_pilots'] for r in summary})

    fig, axes = plt.subplots(
        1, len(metrics), figsize=(9.0 * len(metrics), 7.0), squeeze=False
    )
    fig.patch.set_facecolor(_SURFACE)

    for ax, metric in zip(axes[0], metrics):
        ax.set_facecolor(_SURFACE)
        if shade_below is not None:
            boundary, note = shade_below
            ax.axvspan(
                min(budgets) * 0.8,
                boundary,
                color='#e6e5e1',
                alpha=0.55,
                linewidth=0,
                zorder=0,
            )
            ax.annotate(
                note,
                xy=(np.sqrt(min(budgets) * 0.8 * boundary), 0.02),
                xycoords=('data', 'axes fraction'),
                ha='center',
                color=_MUTED,
                fontsize=12,
            )
        for index, method in enumerate(methods):
            for strategy in strategies:
                rows = sorted(
                    (
                        r
                        for r in summary
                        if r['method'] == method and r['strategy'] == strategy
                    ),
                    key=lambda r: r['n_pilots'],
                )
                if not rows:
                    continue
                x = np.array([r['n_pilots'] for r in rows], dtype=float)
                mu = np.array([r[f'{metric}_mean'] for r in rows])
                sd = np.array([r.get(f'{metric}_std', 0.0) for r in rows])
                line, marker = styles[strategy]
                label = (
                    f'{_label(method)} - {_label(strategy)}'
                    if len(strategies) > 1
                    else _label(method)
                )
                # Series drawn earlier get a slightly wider line, so a
                # pair that coincides exactly (which is the expected
                # result below the feasibility threshold) still reads as
                # two curves rather than one.
                width = 2.0 + 1.8 * (len(methods) - 1 - index)
                ax.plot(
                    x,
                    mu,
                    line,
                    marker=marker,
                    color=colors[method],
                    label=label,
                    linewidth=width,
                    markersize=8 + 3 * (len(methods) - 1 - index),
                    markeredgecolor=_SURFACE,
                    markeredgewidth=1.2,
                )
                ax.fill_between(
                    x,
                    mu - sd,
                    mu + sd,
                    color=colors[method],
                    alpha=0.12,
                    linewidth=0,
                )

        if reference and metric in reference:
            ax.axhline(
                reference[metric],
                linestyle=(0, (1, 3)),
                color=_MUTED,
                linewidth=1.6,
            )
            ax.annotate(
                'native RX',
                xy=(0.0, reference[metric]),
                xycoords=('axes fraction', 'data'),
                xytext=(4, 5),
                textcoords='offset points',
                ha='left',
                color=_MUTED,
                fontsize=13,
            )

        # Budgets are geometric, so a linear axis crushes the small ones
        # together and their labels collide.
        ax.set_xscale('log')
        ax.set_xticks(budgets)
        ax.set_xticklabels([str(b) for b in budgets])
        ax.minorticks_off()
        ax.set_xlabel('Number of semantic pilots $N$', color=_INK)
        ax.set_ylabel(_PRETTY.get(metric, metric), color=_INK)
        ax.grid(True, color='#e6e5e1', linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color('#d8d7d2')
        ax.tick_params(colors=_MUTED, labelsize=14)
        ax.xaxis.label.set_fontsize(16)
        ax.yaxis.label.set_fontsize(16)

    # One shared legend below the panels: with methods x strategies the
    # entry count grows fast, and an in-panel box lands on the curves.
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        fontsize=13,
        labelcolor=_INK,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.0),
        ncol=min(len(labels), 4),
    )
    rows = int(np.ceil(len(labels) / min(len(labels), 4)))
    if title:
        fig.suptitle(title, color=_INK, fontsize=18)
    fig.tight_layout(rect=(0, 0.035 * rows + 0.02, 1, 1))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in formats:
        path = out_path.with_suffix(f'.{ext}')
        fig.savefig(path, dpi=200, facecolor=_SURFACE, bbox_inches='tight')
        written.append(path)
    plt.close(fig)
    return written


def _assign_colors(methods: list[str]) -> dict[str, str]:
    """Map each method to its own fixed hue.

    Known methods keep their slot whatever else is in the figure; any
    unknown name takes the next free slot, in order, never a generated
    hue.
    """
    colors = {m: METHOD_COLORS[m] for m in methods if m in METHOD_COLORS}
    spare = [c for c in _PALETTE if c not in colors.values()]
    unknown = [m for m in methods if m not in colors]
    if len(unknown) > len(spare):
        raise ValueError(
            f'{len(methods)} methods need more hues than the '
            f'{len(_PALETTE)}-slot validated palette provides; facet the '
            'run rather than adding hues.'
        )
    colors.update(dict(zip(unknown, spare)))
    return colors


def _ordered(rows: list[dict[str, Any]], key: str) -> list[str]:
    """Distinct values of ``key``, in first-seen order."""
    seen: list[str] = []
    for row in rows:
        if row[key] not in seen:
            seen.append(row[key])
    return seen


def _label(name: str) -> str:
    """Human-readable series label."""
    return {
        'rkhs': 'RKA (ours)',
        'procrustes': 'Procrustes',
        'direct_mlp': 'Direct MLP',
        'residual_mlp': 'Residual MLP',
        'linear': 'Linear',
        'affine': 'Affine',
        'l_ortho': 'Linear + ortho',
        'cca': 'CCA',
        'svcca': 'SVCCA',
        'kcca': 'KCCA',
        'pca_rkhs': 'PCA-RKA',
        'pga_procrustes': 'PGA-Procrustes',
        'relative': 'Relative rep.',
        'rr': 'RR (inverse proj.)',
        'ppfe': 'Proto-PFE',
        'herding': 'kernel herding',
        'random': 'random',
        'stratified': 'stratified',
        'round_robin': 'round-robin',
        'fps': 'farthest-point',
        'kmeans': 'k-means medoids',
        'rbf': 'RBF',
        'laplacian': 'Laplacian',
        'polynomial': 'polynomial',
        'cosine': 'cosine',
    }.get(name, name)


def plot_regularization_study(
    curves: list[dict[str, Any]],
    metrics: list[str],
    out_path: str | Path,
    baselines: dict[str, dict[str, float]] | None = None,
    reference: dict[str, float] | None = None,
    title: str | None = None,
    formats: tuple[str, ...] = ('png', 'pdf'),
) -> list[Path]:
    """Metric versus the RKHS regularisation, against flat baselines.

    The swept method gets a curve over ``lam``; every baseline is a
    horizontal line, since none of them depends on ``lam``. Reading the
    curve against those lines is the whole point: where it rises above
    them the residual stage is earning its place, and as ``lam`` grows it
    must fall back onto the Procrustes line, which is the pipeline's own
    consistency check.

    Parameters
    ----------
    curves : list[dict]
        One row per ``lam``, with ``lam`` and ``f'{metric}_mean'`` /
        ``f'{metric}_std'`` keys.
    metrics : list[str]
        Metrics to panel, in order.
    out_path : str | Path
        Destination stem.
    baselines : dict[str, dict[str, float]], optional
        ``{method: {metric: value}}``, drawn as horizontal lines.
    reference : dict[str, float], optional
        Receiver's native performance, drawn as a dotted rule.
    title : str, optional
        Figure title.
    formats : tuple[str, ...], default=('png', 'pdf')
        Extensions to write.

    Returns
    -------
    list[Path]
        The files written.
    """
    baselines = baselines or {}
    rows = sorted(curves, key=lambda r: r['lam'])
    lam = np.array([r['lam'] for r in rows], dtype=float)
    colors = _assign_colors(['rkhs', *baselines])

    fig, axes = plt.subplots(
        1, len(metrics), figsize=(9.0 * len(metrics), 7.0), squeeze=False
    )
    fig.patch.set_facecolor(_SURFACE)

    for ax, metric in zip(axes[0], metrics):
        ax.set_facecolor(_SURFACE)
        mu = np.array([r[f'{metric}_mean'] for r in rows])
        sd = np.array([r.get(f'{metric}_std', 0.0) for r in rows])
        ax.plot(
            lam,
            mu,
            '-',
            marker='o',
            color=colors['rkhs'],
            label=_label('rkhs'),
            linewidth=2.0,
            markersize=8,
            markeredgecolor=_SURFACE,
            markeredgewidth=1.2,
        )
        ax.fill_between(
            lam,
            mu - sd,
            mu + sd,
            color=colors['rkhs'],
            alpha=0.12,
            linewidth=0,
        )

        for name, values in baselines.items():
            if metric in values:
                ax.axhline(
                    values[metric],
                    linestyle='--',
                    color=colors[name],
                    linewidth=2.0,
                    label=_label(name),
                )

        if reference and metric in reference:
            ax.axhline(
                reference[metric],
                linestyle=(0, (1, 3)),
                color=_MUTED,
                linewidth=1.6,
            )
            ax.annotate(
                'native RX',
                xy=(0.0, reference[metric]),
                xycoords=('axes fraction', 'data'),
                xytext=(4, 5),
                textcoords='offset points',
                ha='left',
                color=_MUTED,
                fontsize=13,
            )

        best = rows[int(np.argmax(mu))]
        ax.axvline(best['lam'], color=_MUTED, linewidth=1.0, alpha=0.5)
        ax.annotate(
            rf'best $\lambda$ = {best["lam"]:.3g}',
            xy=(best['lam'], mu.max()),
            xytext=(6, -14),
            textcoords='offset points',
            color=_MUTED,
            fontsize=13,
        )

        ax.set_xscale('log')
        ax.set_xlabel(r'RKHS regularisation $\lambda$', color=_INK)
        ax.set_ylabel(_PRETTY.get(metric, metric), color=_INK)
        ax.grid(True, color='#e6e5e1', linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color('#d8d7d2')
        ax.tick_params(colors=_MUTED, labelsize=14)
        ax.xaxis.label.set_fontsize(16)
        ax.yaxis.label.set_fontsize(16)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        fontsize=13,
        labelcolor=_INK,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.0),
        ncol=min(len(labels), 4),
    )
    if title:
        fig.suptitle(title, color=_INK, fontsize=18)
    fig.tight_layout(rect=(0, 0.09, 1, 1))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in formats:
        path = out_path.with_suffix(f'.{ext}')
        fig.savefig(path, dpi=200, facecolor=_SURFACE, bbox_inches='tight')
        written.append(path)
    plt.close(fig)
    return written


def plot_method_comparison(
    summary: list[dict[str, Any]],
    metrics: list[str],
    out_path: str | Path,
    reference: dict[str, float] | None = None,
    highlight: str = 'rkhs',
    title: str | None = None,
    formats: tuple[str, ...] = ('png', 'pdf'),
) -> list[Path]:
    """Rank every alignment method at one fixed pilot budget.

    A horizontal bar per method, one panel per metric, sorted by the
    first metric so the ranking is the shape of the figure rather than
    something the reader has to reconstruct. Every row is direct-labelled
    on the axis and its value is written at the bar end, so the figure
    doubles as the results table.

    Colour carries one bit only -- whether the row is ``highlight``, the
    method under test -- because identity is already on the axis. Giving
    ten methods ten hues would exceed the validated palette and imply a
    grouping the data does not have.

    Parameters
    ----------
    summary : list[dict]
        One row per method, carrying ``method`` and ``f'{metric}_mean'``
        / ``f'{metric}_std'`` for every metric.
    metrics : list[str]
        Metrics to panel, in order. The first one sets the row order.
    out_path : str | Path
        Destination stem; one file per entry of ``formats``.
    reference : dict[str, float], optional
        Per-metric reference line (the receiver's native performance).
    highlight : str, default='rkhs'
        Method key painted in the accent hue.
    title : str, optional
        Figure title.
    formats : tuple[str, ...], default=('png', 'pdf')
        Extensions to write.

    Returns
    -------
    list[Path]
        The files written.
    """
    lead = metrics[0]
    # Ascending, because a horizontal bar axis counts upward from the
    # bottom: this puts the winner at the top of the panel.
    rows = sorted(summary, key=lambda r: r[f'{lead}_mean'])
    names = [r['method'] for r in rows]
    y = np.arange(len(rows), dtype=float)
    accent = METHOD_COLORS.get(highlight, _PALETTE[1])
    colors = [accent if n == highlight else _NEUTRAL_FILL for n in names]

    fig, axes = plt.subplots(
        1,
        len(metrics),
        figsize=(7.5 * len(metrics), 0.62 * len(rows) + 2.6),
        squeeze=False,
        sharey=True,
    )
    fig.patch.set_facecolor(_SURFACE)

    for ax, metric in zip(axes[0], metrics):
        ax.set_facecolor(_SURFACE)
        mu = np.array([r[f'{metric}_mean'] for r in rows])
        sd = np.array([r.get(f'{metric}_std', 0.0) for r in rows])

        ax.barh(
            y,
            mu,
            height=0.62,
            color=colors,
            linewidth=0,
            zorder=2,
        )
        # Drawn separately from `barh(xerr=...)` so the caps keep the ink
        # colour rather than inheriting the bar's.
        ax.errorbar(
            mu,
            y,
            xerr=sd,
            fmt='none',
            ecolor=_MUTED,
            elinewidth=1.4,
            capsize=4,
            capthick=1.4,
            zorder=3,
        )

        span = float(mu.max() - min(mu.min(), 0.0)) or 1.0
        for index, (value, spread) in enumerate(zip(mu, sd)):
            ax.annotate(
                f'{value:.3f}',
                xy=(value + spread + 0.015 * span, index),
                va='center',
                ha='left',
                color=_INK,
                fontsize=12,
                zorder=4,
            )

        if reference and metric in reference:
            ax.axvline(
                reference[metric],
                linestyle=(0, (1, 3)),
                color=_MUTED,
                linewidth=1.6,
                zorder=1,
            )
            ax.annotate(
                'native RX',
                xy=(reference[metric], 1.0),
                xycoords=('data', 'axes fraction'),
                xytext=(0, 4),
                textcoords='offset points',
                ha='center',
                color=_MUTED,
                fontsize=12,
            )

        upper = max(
            float((mu + sd).max()),
            float(reference.get(metric, 0.0)) if reference else 0.0,
        )
        ax.set_xlim(0.0, upper * 1.16)
        ax.set_ylim(-0.7, len(rows) - 0.3)
        ax.set_xlabel(_PRETTY.get(metric, metric), color=_INK)
        ax.grid(True, axis='x', color=_GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ('top', 'right', 'left'):
            ax.spines[side].set_visible(False)
        ax.spines['bottom'].set_color('#d8d7d2')
        ax.tick_params(colors=_MUTED, labelsize=13)
        ax.xaxis.label.set_fontsize(16)

    axes[0][0].set_yticks(y)
    axes[0][0].set_yticklabels([_label(n) for n in names], fontsize=14)
    for label, name in zip(axes[0][0].get_yticklabels(), names):
        label.set_color(_INK if name == highlight else _MUTED)

    if title:
        fig.suptitle(title, color=_INK, fontsize=18)
    fig.tight_layout(rect=(0, 0, 1, 0.97 if title else 1.0))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in formats:
        path = out_path.with_suffix(f'.{ext}')
        fig.savefig(path, dpi=200, facecolor=_SURFACE, bbox_inches='tight')
        written.append(path)
    plt.close(fig)
    return written


def plot_kernel_study(
    rows: list[dict[str, Any]],
    metrics: list[str],
    out_path: str | Path,
    reference: dict[str, float] | None = None,
    x_key: str = 'bandwidth_scale',
    series_key: str = 'kernel',
    title: str | None = None,
    formats: tuple[str, ...] = ('png', 'pdf'),
) -> list[Path]:
    """Metric against kernel bandwidth, one line per kernel family.

    The first panel is always ``residual_r2`` -- how much of the
    Procrustes residual the kernel explains *out of sample*, which is the
    quantity the kernel choice is actually responsible for. Downstream
    metrics follow, so it is visible whether explaining more of the
    residual translates into anything the receiver notices.

    Parameters
    ----------
    rows : list[dict]
        One row per ``(series, x)``, carrying ``series_key``, ``x_key``
        and each metric.
    metrics : list[str]
        Metrics to panel, in order.
    out_path : str | Path
        Destination stem.
    reference : dict[str, float], optional
        Per-metric horizontal rule (the rigid baseline).
    x_key, series_key : str
        Column names for the x-axis and the colour channel.
    title : str, optional
        Figure title.
    formats : tuple[str, ...], default=('png', 'pdf')
        Extensions to write.

    Returns
    -------
    list[Path]
        The files written.
    """
    series = _ordered(rows, series_key)
    colors = _assign_colors(series)
    styles = dict(zip(series, _STRATEGY_STYLES))

    fig, axes = plt.subplots(
        1, len(metrics), figsize=(8.0 * len(metrics), 6.4), squeeze=False
    )
    fig.patch.set_facecolor(_SURFACE)

    for ax, metric in zip(axes[0], metrics):
        ax.set_facecolor(_SURFACE)
        for name in series:
            points = sorted(
                (r for r in rows if r[series_key] == name),
                key=lambda r: r[x_key],
            )
            if not points:
                continue
            line, marker = styles[name]
            ax.plot(
                [p[x_key] for p in points],
                [p[metric] for p in points],
                line,
                marker=marker,
                color=colors[name],
                label=_label(name),
                linewidth=2.0,
                markersize=8,
                markeredgecolor=_SURFACE,
                markeredgewidth=1.2,
            )

        if reference and metric in reference:
            ax.axhline(
                reference[metric],
                linestyle='--',
                color=_MUTED,
                linewidth=1.6,
            )
            ax.annotate(
                'Procrustes',
                xy=(1.0, reference[metric]),
                xycoords=('axes fraction', 'data'),
                xytext=(-4, 5),
                textcoords='offset points',
                ha='right',
                color=_MUTED,
                fontsize=13,
            )

        # Zero is the meaningful floor for an R^2: below it the kernel is
        # predicting the residual worse than predicting nothing at all.
        if metric == 'residual_r2':
            ax.axhline(0.0, color=_MUTED, linewidth=1.0, alpha=0.4)

        ax.set_xscale('log')
        ax.set_xlabel('Bandwidth scale', color=_INK)
        ax.set_ylabel(_PRETTY.get(metric, metric), color=_INK)
        ax.grid(True, color=_GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color('#d8d7d2')
        ax.tick_params(colors=_MUTED, labelsize=14)
        ax.xaxis.label.set_fontsize(16)
        ax.yaxis.label.set_fontsize(16)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        fontsize=13,
        labelcolor=_INK,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.0),
        ncol=min(len(labels), 4),
    )
    if title:
        fig.suptitle(title, color=_INK, fontsize=18)
    fig.tight_layout(rect=(0, 0.08, 1, 1))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in formats:
        path = out_path.with_suffix(f'.{ext}')
        fig.savefig(path, dpi=200, facecolor=_SURFACE, bbox_inches='tight')
        written.append(path)
    plt.close(fig)
    return written


def plot_compression_study(
    rows: list[dict[str, Any]],
    metrics: list[str],
    out_path: str | Path,
    reference: dict[str, float] | None = None,
    hue_of: dict[str, str] | None = None,
    title: str | None = None,
    formats: tuple[str, ...] = ('png', 'pdf'),
) -> list[Path]:
    """Metric against the channel rate, one line per (method, pilot budget).

    Two factors again, so two channels: **colour is the method** and
    **line style is the pilot budget**. That split is the point of the
    figure -- the rate is on the x-axis and is matched across methods by
    construction, so what the reader has to separate is which map is being
    used from how much calibration it was given.

    Rows whose realised rate falls short of the requested one are still
    plotted at the rate they achieved, which is how a method's own rate
    ceiling shows up -- a canonical method cannot exceed
    ``min(d_src, d_tgt)`` pairs however many symbols it is offered.

    ``hue_of`` lets two methods share a colour, separated by line style.
    The palette carries four validated hues, so a fifth series is folded
    rather than invented -- and only where the fold states something
    true: ``{'svcca': 'cca'}`` marks them as one family differing by a
    truncation, not as two unrelated methods that ran out of colours.

    Parameters
    ----------
    rows : list[dict]
        One row per ``(method, n_pilots, symbols)``, carrying those keys
        and each metric.
    metrics : list[str]
        Metrics to panel, in order.
    out_path : str | Path
        Destination stem.
    reference : dict[str, float], optional
        Per-metric horizontal rule (the receiver's own performance).
    title : str, optional
        Figure title.
    formats : tuple[str, ...], default=('png', 'pdf')
        Extensions to write.

    Returns
    -------
    list[Path]
        The files written.
    """
    hue_of = hue_of or {}
    methods = _ordered(rows, 'method')
    budgets = sorted({r['n_pilots'] for r in rows})
    colors = _assign_colors(
        list(dict.fromkeys(hue_of.get(m, m) for m in methods))
    )
    # Style separates the series that share a hue, and the pilot budgets
    # within each: one slot per (method, budget) pair that is drawn.
    series = [
        (m, b)
        for m in methods
        for b in budgets
        if any(r['method'] == m and r['n_pilots'] == b for r in rows)
    ]
    styles = {
        key: _STRATEGY_STYLES[
            sum(
                1
                for earlier in series[: series.index(key)]
                if hue_of.get(earlier[0], earlier[0])
                == hue_of.get(key[0], key[0])
            )
            % len(_STRATEGY_STYLES)
        ]
        for key in series
    }

    fig, axes = plt.subplots(
        1, len(metrics), figsize=(8.0 * len(metrics), 6.6), squeeze=False
    )
    fig.patch.set_facecolor(_SURFACE)

    for ax, metric in zip(axes[0], metrics):
        ax.set_facecolor(_SURFACE)
        for method in methods:
            for budget in budgets:
                points = sorted(
                    (
                        r
                        for r in rows
                        if r['method'] == method and r['n_pilots'] == budget
                    ),
                    key=lambda r: r['symbols'],
                )
                if not points:
                    continue
                line, marker = styles[(method, budget)]
                ax.plot(
                    [p['symbols'] for p in points],
                    [p[metric] for p in points],
                    line,
                    marker=marker,
                    color=colors[hue_of.get(method, method)],
                    label=(
                        f'{_label(method)} · N={budget}'
                        if len(budgets) > 1
                        else _label(method)
                    ),
                    linewidth=2.0,
                    markersize=7,
                    markeredgecolor=_SURFACE,
                    markeredgewidth=1.0,
                )

        if reference and metric in reference:
            ax.axhline(
                reference[metric],
                linestyle=(0, (1, 3)),
                color=_MUTED,
                linewidth=1.6,
            )
            ax.annotate(
                'native RX',
                xy=(0.0, reference[metric]),
                xycoords=('axes fraction', 'data'),
                xytext=(4, 5),
                textcoords='offset points',
                ha='left',
                color=_MUTED,
                fontsize=13,
            )

        ax.set_xscale('log', base=2)
        # A method that cannot reach the requested rate lands on its own
        # ceiling (347, 384), which sits close enough to a grid value to
        # collide with its label. Keep a tick only where it is separated
        # from the last kept one, preferring the requested grid.
        ticks: list[int] = []
        for value in sorted(
            {r['symbols'] for r in rows},
            key=lambda v: (v, -rows[0].get('requested_symbols', 0)),
        ):
            if not ticks or value >= ticks[-1] * 1.25:
                ticks.append(value)
        ax.set_xticks(ticks)
        ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        ax.minorticks_off()
        ax.set_xlabel('Transmitted coefficients per sample', color=_INK)
        ax.set_ylabel(_PRETTY.get(metric, metric), color=_INK)
        ax.grid(True, color=_GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color('#d8d7d2')
        ax.tick_params(colors=_MUTED, labelsize=13)
        ax.xaxis.label.set_fontsize(16)
        ax.yaxis.label.set_fontsize(16)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        fontsize=12,
        labelcolor=_INK,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.0),
        ncol=min(len(labels), 4),
    )
    legend_rows = int(np.ceil(len(labels) / min(len(labels), 4)))
    if title:
        fig.suptitle(title, color=_INK, fontsize=18)
    fig.tight_layout(rect=(0, 0.045 * legend_rows + 0.02, 1, 1))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in formats:
        path = out_path.with_suffix(f'.{ext}')
        fig.savefig(path, dpi=200, facecolor=_SURFACE, bbox_inches='tight')
        written.append(path)
    plt.close(fig)
    return written


def plot_compression_facets(
    rows: list[dict[str, Any]],
    metric: str,
    out_path: str | Path,
    facets: list[str] | None = None,
    reference: dict[str, float] | None = None,
    hue_of: dict[str, str] | None = None,
    title: str | None = None,
    formats: tuple[str, ...] = ('png', 'pdf'),
) -> list[Path]:
    """One metric against the rate, one panel per encoder pair.

    Small multiples rather than a single crowded axis: encoder pairs
    differ in latent width and in how much the two spaces share, so their
    curves live at different heights and overlaying them would compare
    the pairs rather than the methods. Sharing the y-axis keeps the
    panels readable against each other.

    Colour is the method *family* and line style separates its members --
    Procrustes with the two charts it can be fitted in, RKA with the two
    truncations, the three canonical variants. The palette carries four
    validated hues, so eight series are folded along a real distinction
    instead of being given invented colours.

    Parameters
    ----------
    rows : list[dict]
        One row per ``(pair, method, symbols)``, carrying those keys and
        ``metric``.
    metric : str
        The single metric this figure shows.
    out_path : str | Path
        Destination stem.
    facets : list[str], optional
        Pair labels, in panel order. Defaults to first-seen order.
    reference : dict[str, float], optional
        Per-pair horizontal rule (the receiver's own performance).
    hue_of : dict[str, str], optional
        ``method -> hue owner``; see the note above.
    title : str, optional
        Figure title.
    formats : tuple[str, ...], default=('png', 'pdf')
        Extensions to write.

    Returns
    -------
    list[Path]
        The files written.
    """
    hue_of = hue_of or {}
    facets = facets or _ordered(rows, 'pair')
    methods = _ordered(rows, 'method')
    colors = _assign_colors(
        list(dict.fromkeys(hue_of.get(m, m) for m in methods))
    )
    # One style slot per member of a hue group, assigned in first-seen
    # order so a family reads as a family.
    styles: dict[str, tuple[str, str]] = {}
    for method in methods:
        hue = hue_of.get(method, method)
        taken = sum(1 for m in styles if hue_of.get(m, m) == hue)
        styles[method] = _STRATEGY_STYLES[taken % len(_STRATEGY_STYLES)]

    fig, axes = plt.subplots(
        1,
        len(facets),
        figsize=(5.6 * len(facets), 5.4),
        squeeze=False,
        sharey=True,
    )
    fig.patch.set_facecolor(_SURFACE)

    for ax, facet in zip(axes[0], facets):
        ax.set_facecolor(_SURFACE)
        panel = [r for r in rows if r['pair'] == facet]
        for method in methods:
            points = sorted(
                (r for r in panel if r['method'] == method),
                key=lambda r: r['symbols'],
            )
            if not points:
                continue
            line, marker = styles[method]
            ax.plot(
                [p['symbols'] for p in points],
                [p[metric] for p in points],
                line,
                marker=marker,
                color=colors[hue_of.get(method, method)],
                label=_label(method),
                linewidth=1.9,
                markersize=6,
                markeredgecolor=_SURFACE,
                markeredgewidth=1.0,
            )

        if reference and facet in reference:
            ax.axhline(
                reference[facet],
                linestyle=(0, (1, 3)),
                color=_MUTED,
                linewidth=1.5,
            )

        ax.set_xscale('log', base=2)
        ticks: list[int] = []
        for value in sorted({r['symbols'] for r in panel}):
            if not ticks or value >= ticks[-1] * 1.25:
                ticks.append(value)
        ax.set_xticks(ticks)
        ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        ax.minorticks_off()
        ax.set_title(facet, color=_INK, fontsize=13, pad=8)
        ax.set_xlabel('Compression dimension', color=_INK)
        ax.grid(True, color=_GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color('#d8d7d2')
        ax.tick_params(colors=_MUTED, labelsize=12)
        ax.xaxis.label.set_fontsize(14)

    axes[0][0].set_ylabel(_PRETTY.get(metric, metric), color=_INK)
    axes[0][0].yaxis.label.set_fontsize(15)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        fontsize=12,
        labelcolor=_INK,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.0),
        ncol=min(len(labels), 4),
    )
    legend_rows = int(np.ceil(len(labels) / min(len(labels), 4)))
    if title:
        fig.suptitle(title, color=_INK, fontsize=17)
    fig.tight_layout(rect=(0, 0.055 * legend_rows + 0.02, 1, 1))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in formats:
        path = out_path.with_suffix(f'.{ext}')
        fig.savefig(path, dpi=200, facecolor=_SURFACE, bbox_inches='tight')
        written.append(path)
    plt.close(fig)
    return written

"""Output names and the study configs are a contract between the scripts.

``lambda_sweep.py`` writes a CSV per (chart, rate, budget) cell;
``pilot_sweep.py`` reads RKA's measured lambda back out of them, and
``dimension_sweep.py`` reads RKA's and Procrustes' scores. That
handshake rests on two things this module pins down: the two configs
resolve their shared axes identically, and a stem is a stem -- one
dot-free token that survives being handed to a plotting helper.
"""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from src.experiment import check_budgets_fit_pool
from src.plotting import _with_extension
from src.reporting import (
    bandwidth_slug,
    budget_slug,
    chart_slug,
    dimension_fits_stem,
    dimension_stem,
    figure_dir,
    lam_slug,
    lambda_stem,
    model_slug,
    pairs_slug,
    pilot_stem,
    ranks_slug,
    rate_slug,
    result_dir,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = str(ROOT / 'config' / 'hydra')

CHART = {'preprocess': 'pca'}
PAIRS = [
    ('regnety_016.pycls_in1k', 'vit_small_patch16_224.augreg_in1k'),
]
GRID = [1e-8, 1e-4, 1e3]
SCALES = [0.25, 1.0, 4.0]
COUNTS = [423, 768, 3840]


def study(name: str, *overrides: str):
    """Compose one study config the way its script does."""
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return compose(config_name=name, overrides=list(overrides))


# ---------------------------------------------------------------------
# Slugs
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ('name', 'expected'),
    [
        # The pre-training tag and the input resolution are dropped; the
        # patch size is not, because it is what separates two ViTs.
        ('vit_small_patch16_224.augreg_in1k', 'vit_small_p16'),
        ('vit_base_patch32_224.augreg_in21k', 'vit_base_p32'),
        # Trailing digits that are part of the identity survive: there is
        # no patch size in front of them to make them a resolution.
        ('regnety_016.pycls_in1k', 'regnety_016'),
    ],
)
def test_model_slug_keeps_what_separates_the_encoders(name, expected):
    assert model_slug(name) == expected


def test_pairs_slug_always_names_the_receiver():
    one = pairs_slug([('a', 'rx')])
    many = pairs_slug([('a', 'rx'), ('b', 'rx'), ('c', 'rx')])
    assert one == 'a-to-rx'
    assert many == '3tx-to-rx'


def test_rate_slug_does_not_claim_a_truncation_that_did_not_happen():
    assert rate_slug(384) == 'k384'
    assert rate_slug(None) == 'kfull'


def test_chart_slug_separates_charts_that_differ():
    assert chart_slug({'preprocess': 'pca'}) != chart_slug(
        {'preprocess': 'whiten'}
    )
    assert chart_slug({'preprocess': 'whiten', 'eps': 1e-3}) != chart_slug(
        {'preprocess': 'whiten', 'eps': 1e-6}
    )
    assert chart_slug({'name': 'ablation', 'preprocess': 'pca'}) == 'ablation'


def test_slugs_report_the_span_and_the_resolution_of_an_axis():
    # Two grids over the same window at different resolutions have to be
    # told apart, or one overwrites the other's CSV.
    assert lam_slug(GRID) != lam_slug([1e-8, 1e-6, 1e-4, 1e3])
    assert budget_slug(COUNTS) == 'N423to3840x3'
    assert ranks_slug([128, 16, 888]) == 'k16to888x3'
    assert bandwidth_slug(SCALES) == 'bw0p25to4x3'


# ---------------------------------------------------------------------
# Stems
# ---------------------------------------------------------------------

STEMS = [
    lambda_stem('cifar10', PAIRS, CHART, 384, 653, GRID),
    pilot_stem('cifar10', PAIRS, CHART, 384, COUNTS, ['stratified']),
    dimension_stem(
        'cifar10', PAIRS, CHART, 8192, 'herding', [16, 64, 888], 'linear'
    ),
    dimension_stem(
        'cifar10',
        PAIRS,
        CHART,
        8192,
        'herding',
        [16, 64],
        'linear',
        seeds=[0, 1, 2],
    ),
    dimension_fits_stem('cifar10', PAIRS, 8192, 'herding', 'linear'),
    lambda_stem('cifar10', PAIRS, CHART, 384, 653, GRID, bandwidths=SCALES),
]


@pytest.mark.parametrize('stem', STEMS)
def test_a_stem_carries_no_dot(stem):
    """``Path.with_suffix`` truncates at the last dot, so a stem has none.

    This is not hypothetical: a lambda grid printed with ``%g`` puts a
    dot in the middle of every name, and the figure written from it lands
    at ``..._lam0.pdf`` with the rest of the configuration gone.
    """
    assert '.' not in stem


@pytest.mark.parametrize('stem', STEMS)
def test_appending_an_extension_keeps_the_whole_stem(stem):
    assert _with_extension(Path('/tmp') / stem, 'pdf').name == f'{stem}.pdf'


def test_a_stem_names_every_axis_of_the_cell():
    stem = lambda_stem('cifar10', PAIRS, CHART, 384, 653, GRID)
    for part in ('cifar10', 'regnety_016', 'vit_small_p16', 'pca', 'k384'):
        assert part in stem
    assert '_n653_' in stem


def test_cells_that_differ_in_any_axis_get_different_names():
    base = lambda_stem('cifar10', PAIRS, CHART, 384, 653, GRID)
    variants = [
        lambda_stem('mnist', PAIRS, CHART, 384, 653, GRID),
        lambda_stem(
            'cifar10', PAIRS, {'preprocess': 'whiten'}, 384, 653, GRID
        ),
        lambda_stem('cifar10', PAIRS, CHART, 192, 653, GRID),
        lambda_stem('cifar10', PAIRS, CHART, 384, 768, GRID),
        lambda_stem('cifar10', PAIRS, CHART, 384, 653, [1e-8, 1e3]),
        lambda_stem('cifar10', [('x', 'y')], CHART, 384, 653, GRID),
        # A bandwidth run must not land on the single-bandwidth CSV it
        # came from: `pilot_sweep.py` reads the best lambda per budget
        # out of these and has no bandwidth column to filter on.
        lambda_stem(
            'cifar10', PAIRS, CHART, 384, 653, GRID, bandwidths=SCALES
        ),
        lambda_stem(
            'cifar10', PAIRS, CHART, 384, 653, GRID, bandwidths=[0.5, 1.0]
        ),
    ]
    assert len(set(variants)) == len(variants)
    assert base not in variants


def test_figures_and_results_live_in_parallel_trees(tmp_path):
    figures = figure_dir(
        tmp_path / 'figures', 'lambda_sweep', 'cifar10', 'pca'
    )
    results = result_dir(tmp_path / 'results', 'lambda_sweep', 'cifar10')
    assert figures.is_dir() and results.is_dir()
    assert figures.relative_to(tmp_path / 'figures').parts == (
        'lambda_sweep',
        'cifar10',
        'pca',
    )
    assert results.relative_to(tmp_path / 'results').parts == (
        'lambda_sweep',
        'cifar10',
    )


# ---------------------------------------------------------------------
# The shared axes
# ---------------------------------------------------------------------


def test_both_studies_resolve_the_same_budget_axis():
    """The lambda lookup matches budgets exactly, so the axes must agree."""
    lam = study('lambda_sweep')
    pilots = study('pilot_sweep')
    assert lam.pilots.counts == pilots.pilots.counts
    assert lam.pilots.multipliers == pilots.pilots.multipliers
    assert lam.symbols == pilots.symbols
    assert lam.symbol_divisor == pilots.symbol_divisor
    assert lam.charts == pilots.charts


def test_overriding_the_shared_axis_moves_both_studies():
    override = 'pilots.multipliers=[1.5,3.0]'
    lam = study('lambda_sweep', override)
    pilots = study('pilot_sweep', override)
    assert list(lam.pilots.multipliers) == [1.5, 3.0]
    assert list(lam.pilots.multipliers) == list(pilots.pilots.multipliers)


def test_the_swept_method_is_one_of_the_configured_ones():
    lam = study('lambda_sweep')
    assert lam.sweep_method in lam.methods


def test_the_scheduled_methods_are_configured_and_take_a_lambda():
    pilots = study('pilot_sweep')
    for name in pilots.lam_schedule.methods:
        assert name in pilots.methods, name
        # A schedule can only be applied to a method that has a lambda to
        # pin; writing one into a preset without it would be silent.
        assert 'lam' in pilots.methods[name], name


def test_the_pilot_study_reads_the_lambda_study_it_names():
    """The schedule points at a study directory, not at a filename."""
    pilots = study('pilot_sweep')
    lam = study('lambda_sweep')
    assert pilots.lam_schedule.study == lam.output.study
    assert pilots.output.results == lam.output.results


def test_the_scheduled_metric_is_one_the_lambda_study_measures():
    pilots = study('pilot_sweep')
    lam = study('lambda_sweep')
    assert pilots.lam_schedule.metric in lam.eval.metrics


def test_both_studies_draw_from_the_same_candidate_pool():
    """A lambda measured on pilots herded from one pool does not belong
    to a run whose pilots were herded from another."""
    lam = study('lambda_sweep')
    pilots = study('pilot_sweep')
    assert lam.pilots.pool_size == pilots.pilots.pool_size


def test_a_budget_the_pool_cannot_fill_is_refused():
    """Past the pool size every budget selects the same pilots, so the
    curve would flatten for a reason that is not in the data."""
    check_budgets_fit_pool([100, 500], pool_size=10000, n_available=50000)
    with pytest.raises(ValueError, match='meets or exceeds'):
        check_budgets_fit_pool([100, 24576], 10000, 50000)
    # `null` pool means the whole split, which is still a limit.
    with pytest.raises(ValueError, match='meets or exceeds'):
        check_budgets_fit_pool([60000], None, 50000)


# ---------------------------------------------------------------------
# The dimension sweep
# ---------------------------------------------------------------------


def dimension_script():
    """``scripts/dimension_sweep.py`` as a module (``scripts/`` is not a
    package, and its ``main`` only runs under ``__main__``)."""
    path = ROOT / 'scripts' / 'dimension_sweep.py'
    spec = importlib.util.spec_from_file_location('dimension_sweep', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_dimension_study_fits_on_the_lambda_studys_pilots():
    """Its fitted curves are drawn against read-back ones, so every input
    to the pilot selection has to be the lambda sweep's."""
    dims = study('dimension_sweep')
    lam = study('lambda_sweep')
    assert dims.seed == lam.seed
    assert dims.pilots.pool_size == lam.pilots.pool_size
    assert dims.pilots.strategy == lam.pilots.strategy


def test_the_dimension_study_reads_what_the_lambda_study_wrote():
    dims = study('dimension_sweep')
    lam = study('lambda_sweep')
    assert dims.read_back.study == lam.output.study
    assert dims.output.results == lam.output.results
    assert dims.read_back.swept == lam.sweep_method
    assert set(dims.read_back.methods) <= set(lam.methods)
    assert dims.read_back.metric in lam.eval.metrics
    assert set(dims.eval.metrics) <= set(lam.eval.metrics)
    # Refits -- the check, and every seed -- use exactly the presets whose
    # numbers the lambda sweep wrote.
    for name in dims.read_back.methods:
        assert dims.refit[name] == lam.methods[name], name


def test_the_dimension_study_fits_methods_it_does_not_read_back():
    dims = study('dimension_sweep')
    assert not set(dims.methods) & set(dims.read_back.methods)
    # `charted` only says how a fitted method is fitted; naming anything
    # else would be silently ignored.
    assert set(dims.charted) <= set(dims.methods)


def test_a_rank_the_pilot_budget_cannot_support_is_refused():
    module = dimension_script()
    module.check_ranks_below_budget([16, 128, 888], 1024)
    with pytest.raises(ValueError, match=r'\[888\]'):
        module.check_ranks_below_budget([16, 888], 888)


def write_lambda_cell(path, chart, symbols, n_pilots, points):
    """A lambda-sweep cell CSV with the columns that study writes."""
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                'dataset',
                'pairs',
                'chart',
                'preprocess',
                'symbols',
                'n_pilots',
                'pilots_per_symbol',
                'strategy',
                'lam',
                'n_cells',
                'accuracy_mean',
                'accuracy_std',
                'mrr_mean',
                'mrr_std',
                'procrustes_accuracy',
                'procrustes_mrr',
            ],
        )
        writer.writeheader()
        for lam, acc, mrr in points:
            writer.writerow(
                {
                    'dataset': 'cifar10',
                    'pairs': 'a-to-b',
                    'chart': chart,
                    'preprocess': 'whiten',
                    'symbols': '' if symbols is None else symbols,
                    'n_pilots': n_pilots,
                    'pilots_per_symbol': 1.0,
                    'strategy': 'herding',
                    'lam': lam,
                    'n_cells': 1,
                    'accuracy_mean': acc,
                    'accuracy_std': 0.0,
                    'mrr_mean': mrr,
                    'mrr_std': 0.0,
                    'procrustes_accuracy': 0.80,
                    'procrustes_mrr': 0.10,
                }
            )


def test_read_back_takes_every_metric_at_the_best_lambda(tmp_path):
    """MRR is reported where accuracy peaked, not at its own optimum --
    one fit per point, the rule the pilot sweep applies."""
    module = dimension_script()
    write_lambda_cell(
        tmp_path / 'lambda_x_k32.csv',
        'whiten-k32',
        32,
        8192,
        [(1e-4, 0.85, 0.30), (1e-3, 0.88, 0.20), (1e-2, 0.82, 0.10)],
    )
    # Same chart at another budget and another rank: neither may leak in.
    write_lambda_cell(
        tmp_path / 'lambda_x_k32_n4096.csv',
        'whiten-k32',
        32,
        4096,
        [(1e-3, 0.99, 0.99)],
    )
    write_lambda_cell(
        tmp_path / 'lambda_x_k64.csv',
        'whiten-k64',
        64,
        8192,
        [(1e-3, 0.99, 0.99)],
    )
    cell = module.read_back(
        tmp_path,
        'a-to-b',
        'whiten-k32',
        32,
        8192,
        'herding',
        ['procrustes', 'rkhs'],
        'rkhs',
        ['accuracy', 'mrr'],
        'accuracy',
    )
    assert cell['lam'] == 1e-3
    assert cell['values']['rkhs'] == {'accuracy': 0.88, 'mrr': 0.20}
    assert cell['values']['procrustes'] == {'accuracy': 0.80, 'mrr': 0.10}


def test_read_back_matches_an_untruncated_cell_and_misses_cleanly(tmp_path):
    module = dimension_script()
    write_lambda_cell(
        tmp_path / 'lambda_x_kfull.csv',
        'whiten-kfull',
        None,
        8192,
        [(1e-3, 0.9, 0.5)],
    )
    args = (['procrustes', 'rkhs'], 'rkhs', ['accuracy'], 'accuracy')
    found = module.read_back(
        tmp_path, 'a-to-b', 'whiten-kfull', None, 8192, 'herding', *args
    )
    assert found['values']['rkhs']['accuracy'] == 0.9
    assert (
        module.read_back(
            tmp_path, 'a-to-b', 'whiten-kfull', None, 8192, 'random', *args
        )
        is None
    )


# ---------------------------------------------------------------------
# The pair average
# ---------------------------------------------------------------------


def pair_average_script():
    """``scripts/pair_average.py`` as a module."""
    path = ROOT / 'scripts' / 'pair_average.py'
    spec = importlib.util.spec_from_file_location('pair_average', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_pair_average_reads_what_the_per_pair_studies_write():
    avg = study('pair_average')
    pilots = study('pilot_sweep')
    dims = study('dimension_sweep')
    assert avg.pilot.study == pilots.output.study
    assert avg.dimension.study == dims.output.study
    assert avg.output.results == pilots.output.results == dims.output.results
    assert avg.dimension.strategy == dims.pilots.strategy
    assert set(avg.pilot.strategies) <= {
        'herding',
        'round_robin',
        'random',
        'stratified',
        'kmeans',
        'fps',
    }
    assert set(avg.eval.metrics) <= set(dims.eval.metrics)
    # The pilot figure is read at one rate, and the dimension figure has to
    # include it or the two cannot be read against each other.
    assert avg.pilot.symbols in avg.dimension.ranks


def test_seeds_are_averaged_within_a_pair_before_the_spread_over_pairs():
    """A pair run with more seeds must not weigh more, and the band has to
    be the spread between pairs rather than between seeds."""
    module = pair_average_script()
    rows = [
        # Pair a: three seeds at 0.80, 0.82, 0.84 -> mean 0.82.
        *[
            {'pairs': 'a', 'method': 'rkhs', 'n_pilots': 8, 'accuracy': v}
            for v in (0.80, 0.82, 0.84)
        ],
        # Pair b: one seed at 0.90.
        {'pairs': 'b', 'method': 'rkhs', 'n_pilots': 8, 'accuracy': 0.90},
    ]
    (point,) = module.average_over_pairs(
        rows, ('method', 'n_pilots'), ['accuracy']
    )
    assert point['n_pairs'] == 2
    assert point['accuracy_mean'] == pytest.approx(0.86)
    assert point['accuracy_std'] == pytest.approx(0.04)
    # Sample sd over the two pair means is 0.08 / sqrt(2), so the standard
    # error is 0.04, scaled by the t quantile at one degree of freedom.
    assert point['accuracy_ci95'] == pytest.approx(12.7062047 * 0.04)


def test_a_pair_missing_a_point_is_reported_not_averaged_over():
    module = pair_average_script()
    rows = [
        {'pairs': 'a', 'method': 'rkhs', 'requested': 16},
        {'pairs': 'a', 'method': 'rkhs', 'requested': 32},
        {'pairs': 'b', 'method': 'rkhs', 'requested': 16},
    ]
    missing = module.missing_points(
        rows, ['a', 'b'], ('method', 'requested'), [('rkhs', 16), ('rkhs', 32)]
    )
    assert missing == {'b': [('rkhs', 32)]}


def test_a_seeded_dimension_run_never_takes_the_single_runs_name():
    single = dimension_stem(
        'cifar10', PAIRS, CHART, 8192, 'herding', [16, 64], 'linear'
    )
    seeded = dimension_stem(
        'cifar10', PAIRS, CHART, 8192, 'herding', [16, 64], 'linear', [0, 1]
    )
    assert single != seeded
    assert single.endswith('dec-linear')


def write_rows(path, rows):
    """A CSV with the union of the rows' columns, blanks for the rest."""
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_the_pair_average_reads_only_the_seeds_it_is_given(tmp_path):
    """A pair that has run more seeds than the list must not weigh them in."""
    module = pair_average_script()
    base = {
        'pairs': 'a-to-b',
        'chart': 'whiten-k32',
        'symbols': 32,
        'n_pilots': 128,
        'strategy': 'herding',
        'method': 'rkhs',
    }
    write_rows(
        tmp_path / 'pilots_a_seed0.csv',
        [base | {'seed': s, 'accuracy': 0.8 + s / 100} for s in (0, 1, 2)],
    )
    rows = module.pilot_records(
        tmp_path, ['a-to-b'], 'whiten-k32', 32, [128], ['herding'], [0, 1]
    )
    assert sorted(r['seed'] for r in rows) == [0, 1]


def test_dimension_rows_keep_every_seed_or_only_the_single_run(tmp_path):
    module = pair_average_script()
    base = {
        'pairs': 'a-to-b',
        'base_chart': 'whiten',
        'n_pilots': 8192,
        'strategy': 'herding',
        'method': 'rkhs',
        'requested': 32,
    }
    # The single realisation, written before seeds existed: no column.
    write_rows(tmp_path / 'dims_single.csv', [base | {'accuracy': 0.9}])
    write_rows(
        tmp_path / 'dims_seeded.csv',
        [base | {'seed': s, 'accuracy': 0.8 + s / 100} for s in (0, 1)],
    )
    args = (tmp_path, ['a-to-b'], 'whiten', 8192, 'herding', [32])

    single = module.dimension_rows(*args, None)
    assert [r['accuracy'] for r in single] == [0.9]

    seeded = module.dimension_rows(*args, [0, 1])
    assert sorted(r['seed'] for r in seeded) == [0, 1]
    # Both seeds survive to be averaged within the pair.
    (point,) = module.average_over_pairs(
        seeded, ('method', 'requested'), ['accuracy']
    )
    assert point['accuracy_mean'] == pytest.approx(0.805)


def test_a_pair_missing_a_seed_is_reported():
    module = pair_average_script()
    rows = [
        {'pairs': 'a', 'method': 'rkhs', 'requested': 16, 'seed': 0},
        {'pairs': 'a', 'method': 'rkhs', 'requested': 16, 'seed': 1},
        {'pairs': 'b', 'method': 'rkhs', 'requested': 16, 'seed': 0},
    ]
    missing = module.missing_points(
        rows,
        ['a', 'b'],
        ('method', 'requested', 'seed'),
        [('rkhs', 16, 0), ('rkhs', 16, 1)],
    )
    assert missing == {'b': [('rkhs', 16, 1)]}

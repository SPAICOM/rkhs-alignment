"""Which encoder receives, and therefore what the whole run is indexed by.

``receiver`` is the most consequential thing a study config does not say
out loud. It fixes ``d_r``, which fixes the rate, which fixes the pilot
budgets, which fixes every filename -- and a run pointed at the wrong end
of the pair is not distinguishable from a correct one by its numbers
alone. The rule is that the receiver is the *widest* agent available,
because a narrow receiver caps what the channel can carry for every
method at once.

Everything here runs on the synthetic source, so the tests stay offline.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from src.experiment import (
    channel_rate,
    configure_method,
    declared_widths,
    effective_rank,
    rank_request,
    resolve_chart_budgets,
    resolve_charts,
    resolve_star,
    resolve_symbols,
    usable_rank,
    widest_agent,
)
from src.reporting import chart_slug, rate_slug

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / 'config' / 'hydra')


def study(*overrides: str):
    """Compose the lambda sweep on the synthetic source."""
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return compose(
            config_name='lambda_sweep',
            overrides=['data=synthetic', 'max_pairs=2', *overrides],
        )


# ---------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------


def test_the_widest_agent_wins_whatever_its_position():
    names = ['a', 'b', 'c']
    assert widest_agent(names, {'a': 384, 'b': 888, 'c': 512}) == 'b'
    # Listed first, but narrowest: position must not rescue it.
    assert widest_agent(names, {'a': 12, 'b': 888, 'c': 512}) == 'b'


def test_a_tie_is_broken_by_the_order_the_config_lists_them():
    """Ties have to resolve the same way on every machine and every run."""
    widths = {'a': 888, 'b': 888, 'c': 512}
    # `a` and `b` tie at the top; whichever the config lists first wins,
    # and the narrower `c` wins nothing either way.
    assert widest_agent(['a', 'b', 'c'], widths) == 'a'
    assert widest_agent(['c', 'b', 'a'], widths) == 'b'


def test_the_receiver_is_the_widest_agent_in_the_run():
    cfg = study()
    pairs, agents = resolve_star(cfg)
    receivers = {t for _, t in pairs}
    assert len(receivers) == 1
    receiver = receivers.pop()

    widths = {
        name: space['train'].latent.shape[1] for name, space in agents.items()
    }
    assert widths[receiver] == max(widths.values())
    # And every sender is strictly narrower than it, on this config.
    assert all(widths[s] < widths[receiver] for s, _ in pairs)


def test_d_r_is_read_off_the_receiver():
    cfg = study()
    pairs, agents = resolve_star(cfg)
    receiver = pairs[0][1]
    d_r, _ = usable_rank(pairs, agents)
    assert d_r == agents[receiver]['train'].latent.shape[1]


def test_the_rate_cannot_exceed_what_the_transmitter_can_emit():
    """``max_rank`` is the narrowest space, not the receiver's.

    With a receiver wider than its transmitters -- which is now the
    normal case -- ``d_r`` is no longer an achievable rate: the rate is
    the rank of the *source* whitening, so the transmitter caps it.
    """
    cfg = study()
    pairs, agents = resolve_star(cfg)
    d_r, max_rank = usable_rank(pairs, agents)
    narrowest = min(
        min(
            agents[s]['train'].latent.shape[1],
            agents[t]['train'].latent.shape[1],
        )
        for s, t in pairs
    )
    assert max_rank == narrowest
    assert max_rank <= d_r


# ---------------------------------------------------------------------
# Overriding it
# ---------------------------------------------------------------------


def test_an_explicit_receiver_beats_the_rule():
    """Naming one is how a study asks for a direction the rule would not."""
    cfg = study('receiver=agent_00')
    pairs, agents = resolve_star(cfg)
    assert {t for _, t in pairs} == {'agent_00'}
    assert agents['agent_00']['train'].latent.shape[1] < max(
        space['train'].latent.shape[1] for space in agents.values()
    )


def test_explicit_pairs_beat_the_rule():
    cfg = study(
        'pairs=[{source: agent_05, target: agent_03}]',
    )
    pairs, _ = resolve_star(cfg)
    assert pairs == [('agent_05', 'agent_03')]


def test_only_the_agents_the_pairs_reference_are_returned():
    """`max_pairs` drops senders; the caller must not see the leftovers."""
    cfg = study('max_pairs=2')
    pairs, agents = resolve_star(cfg)
    assert len(pairs) == 2
    assert set(agents) == {name for pair in pairs for name in pair}


# ---------------------------------------------------------------------
# Finding the widths without paying for them
# ---------------------------------------------------------------------


def test_a_source_that_generates_its_agents_declares_their_widths():
    """42 synthetic agents must not be materialised to rank two of them."""
    cfg = study()
    widths = declared_widths(cfg)
    assert widths is not None
    assert len(widths) == int(cfg.data.n_agents)
    assert widths['agent_00'] == int(cfg.data.agent_dims[0])


def test_a_source_that_does_not_declare_widths_says_so():
    """SEMASIA names carry no width, so they have to be measured."""
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name='lambda_sweep', overrides=['data=semasia'])
    assert declared_widths(cfg) is None


@pytest.mark.parametrize('name', ['lambda_sweep', 'pilot_sweep'])
def test_both_studies_leave_the_receiver_to_the_rule(name):
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name=name)
    assert cfg.receiver is None
    assert cfg.pairs is None


# ---------------------------------------------------------------------
# Compression, or the absence of it
# ---------------------------------------------------------------------


def test_a_request_below_max_rank_is_a_compression():
    cfg = study('symbols=16')
    pairs, agents = resolve_star(cfg)
    _, max_rank = usable_rank(pairs, agents)
    assert max_rank > 16
    assert resolve_symbols(cfg, *usable_rank(pairs, agents)) == 16


def test_a_request_the_transmitter_cannot_exceed_compresses_nothing():
    """``n_components`` reaches both charts, so an unreachable rank would
    leave the transmitter alone and truncate only the receiver -- which is
    compressing the wrong side of the channel."""
    cfg = study()
    pairs, agents = resolve_star(cfg)
    d_r, max_rank = usable_rank(pairs, agents)
    assert d_r > max_rank, 'this fixture needs a receiver wider than its TX'
    # symbol_divisor=1 asks for d_r, which the TX cannot reach.
    assert resolve_symbols(cfg, d_r, max_rank) is None


def test_no_compression_leaves_every_chart_at_its_own_full_width():
    cfg = study()
    pairs, agents = resolve_star(cfg)
    d_r, max_rank = usable_rank(pairs, agents)
    symbols = resolve_symbols(cfg, d_r, max_rank)
    method = next(iter(cfg.methods.values()))
    merged = configure_method(method, {'preprocess': 'whiten'}, symbols)
    # Written out as None rather than left alone, so a preset shipping a
    # rank of its own cannot reintroduce a truncation here.
    assert merged[method.rate_key] is None
    assert effective_rank(merged, method) is None
    assert rate_slug(effective_rank(merged, method)) == 'kfull'


def test_the_pilot_axis_is_scaled_by_what_reaches_the_channel():
    """An uncompressed run still spends a definite number of symbols."""
    cfg = study()
    pairs, agents = resolve_star(cfg)
    d_r, max_rank = usable_rank(pairs, agents)
    symbols = resolve_symbols(cfg, d_r, max_rank)
    assert symbols is None
    assert channel_rate(symbols, max_rank) == max_rank
    assert channel_rate(16, max_rank) == 16


# ---------------------------------------------------------------------
# The rank axis
# ---------------------------------------------------------------------


def test_no_ranks_leaves_the_chart_axis_exactly_as_it_was():
    """A config that never mentions `ranks` must behave as before."""
    cfg = study('charts=[{preprocess: pca}, {preprocess: whiten}]')
    assert cfg.ranks is None
    charts = resolve_charts(cfg, max_rank=384)
    assert [dict(c) for c in charts] == [
        {'preprocess': 'pca'},
        {'preprocess': 'whiten'},
    ]


def test_ranks_cross_every_chart():
    cfg = study(
        'charts=[{preprocess: pca}, {preprocess: whiten}]', 'ranks=[192, 96]'
    )
    charts = resolve_charts(cfg, max_rank=384)
    assert [chart_slug(c) for c in charts] == [
        'pca-k192',
        'pca-k96',
        'whiten-k192',
        'whiten-k96',
    ]
    assert [c.n_components for c in charts] == [192, 96, 192, 96]


def test_a_rank_the_transmitter_cannot_exceed_is_no_compression():
    """Same rule `resolve_symbols` applies, per rank."""
    cfg = study('charts=[{preprocess: pca}]', 'ranks=[512, 384, 96]')
    charts = resolve_charts(cfg, max_rank=384)
    # 512 and 384 both fail to compress, and collapse to one chart.
    assert [chart_slug(c) for c in charts] == ['pca-kfull', 'pca-k96']
    assert charts[0].n_components is None


def test_every_rank_of_one_chart_gets_its_own_slug():
    """`chart_slug` does not print `n_components`; the name must carry it."""
    cfg = study('charts=[{preprocess: pca}]', 'ranks=[192, 96, 48]')
    slugs = [chart_slug(c) for c in resolve_charts(cfg, max_rank=384)]
    assert len(set(slugs)) == len(slugs)


def test_pinning_n_components_alongside_ranks_is_an_error():
    cfg = study(
        'charts=[{preprocess: pca, n_components: 128}]', 'ranks=[192, 96]'
    )
    with pytest.raises(ValueError, match='ambiguous'):
        resolve_charts(cfg, max_rank=384)


@pytest.mark.parametrize('name', ['lambda_sweep', 'pilot_sweep'])
def test_both_studies_expand_the_rank_axis_identically(name):
    """The lambda measured at one (chart, rank) is looked up at the other."""
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(
            config_name=name,
            overrides=[
                'data=synthetic',
                'max_pairs=2',
                'charts=[{preprocess: pca}]',
                'ranks=[192, 96]',
            ],
        )
    assert [chart_slug(c) for c in resolve_charts(cfg, max_rank=384)] == [
        'pca-k192',
        'pca-k96',
    ]


# ---------------------------------------------------------------------
# The budget axis follows the rank
# ---------------------------------------------------------------------


def plan(*overrides: str, max_rank: int = 384):
    cfg = study(*overrides)
    charts = resolve_charts(cfg, max_rank)
    return cfg, resolve_chart_budgets(
        cfg, charts, cfg.methods[cfg.sweep_method], None, max_rank
    )


def test_each_rank_is_measured_at_its_own_budgets():
    """`multipliers` are pilots per symbol, so the axis follows the rank."""
    _, p = plan(
        'charts=[{preprocess: pca}]',
        'ranks=[384, 96]',
        'pilots.multipliers=[8.0, 16.0]',
    )
    ranks = [rank for _, rank, _, _ in p]
    budgets = [b for *_, b in p]
    assert ranks == [None, 96]  # 384 >= max_rank -> kfull
    assert budgets == [[3072, 6144], [768, 1536]]


def test_every_cell_is_read_at_the_same_pilots_per_symbol():
    _, p = plan(
        'charts=[{preprocess: pca}]',
        'ranks=[192, 96, 48]',
        'pilots.multipliers=[8.0, 32.0]',
    )
    for _, _, rate, budgets in p:
        assert [n / rate for n in budgets] == [8.0, 32.0]


def test_an_uncompressed_chart_is_scaled_to_the_full_width():
    _, p = plan(
        'charts=[{preprocess: pca}]',
        'ranks=[999]',
        'pilots.multipliers=[8.0]',
        max_rank=384,
    )
    (_, rank, rate, budgets) = p[0]
    assert rank is None and rate == 384 and budgets == [3072]


def test_absolute_counts_win_and_apply_to_every_chart():
    _, p = plan(
        'charts=[{preprocess: pca}]',
        'ranks=[192, 48]',
        'pilots.counts=[100, 200]',
    )
    assert [b for *_, b in p] == [[100, 200], [100, 200]]


def test_without_ranks_every_chart_keeps_the_run_wide_axis():
    """No `ranks` means one rank, so nothing about the axis changes."""
    _, p = plan(
        'charts=[{preprocess: pca}, {preprocess: whiten}]',
        'pilots.multipliers=[8.0]',
    )
    assert len({tuple(b) for *_, b in p}) == 1


def test_a_fractional_rank_is_a_fraction_of_the_channel():
    """0.8 keeps 80% of `max_rank` -- "compress by 20%"."""
    assert rank_request(0.8, 192) == 154
    assert rank_request(0.6, 192) == 116
    assert rank_request(0.2, 192) == 39
    assert rank_request(1.0, 192) == 192


def test_an_integer_rank_is_absolute():
    assert rank_request(96, 192) == 96
    assert rank_request(400, 192) == 400  # caller reads it as no compression


def test_a_fractional_rank_follows_the_pair():
    """The same config, two pairs, two ranks -- the point of a fraction."""
    assert rank_request(0.5, 192) == 96
    assert rank_request(0.5, 768) == 384


@pytest.mark.parametrize('bad', [0.0, -0.5, 1.5, 0, -3])
def test_a_rank_outside_the_two_conventions_is_rejected(bad):
    with pytest.raises(ValueError):
        rank_request(bad, 192)


def test_percentages_reach_the_chart_as_integers():
    """`n_components` must never see the fraction: there it means variance."""
    cfg = study('charts=[{preprocess: pca}]', 'ranks=[0.8, 0.6, 0.2]')
    charts = resolve_charts(cfg, max_rank=192)
    assert [chart_slug(c) for c in charts] == [
        'pca-k154',
        'pca-k116',
        'pca-k39',
    ]
    assert [c.n_components for c in charts] == [154, 116, 39]
    assert all(isinstance(c.n_components, int) for c in charts)

"""The alignment presets under ``config/hydra/alignment/`` are the API.

Every study selects its methods by name from that directory, so a preset
that fails to compose, fails to instantiate, or forgets to declare the
pilot design its method expects is a broken entry point -- and one that
only shows up as a crash (or, worse, as a silently mis-calibrated method)
half an hour into a run.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from src.alignment import PILOT_STRATEGIES, Aligner
from src.experiment import ALIGNER_META_KEYS, build_aligner

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / 'config' / 'hydra')
PRESETS = sorted(
    p.stem for p in (Path(CONFIG_DIR) / 'alignment').glob('*.yaml')
)


def preset(name: str):
    """Compose one alignment preset in isolation."""
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(
            config_name='evaluate_alignment',
            overrides=[f'alignment={name}', 'data=synthetic'],
        )
    return cfg.alignment


def test_the_preset_directory_is_not_empty():
    assert PRESETS, f'No alignment presets found under {CONFIG_DIR}.'


@pytest.mark.parametrize('name', PRESETS)
def test_preset_builds_an_aligner(name):
    """Composing and building must work without any run-level context.

    ``build_aligner`` is the only route the scripts use, precisely so
    that the metadata keys never reach a constructor; calling it here is
    what proves the stripping is in place for every preset.
    """
    aligner = build_aligner(preset(name))
    assert isinstance(aligner, Aligner)


@pytest.mark.parametrize('name', PRESETS)
def test_preset_declares_a_usable_pilot_design(name):
    """A method's calibration design travels with the method.

    Every preset names the pilot strategy its published results assume,
    so a study that does not pin one still evaluates each method the way
    its authors intended instead of under whatever the run happened to
    default to.
    """
    strategy = preset(name).get('pilot_strategy')
    assert strategy is not None
    assert strategy in PILOT_STRATEGIES


@pytest.mark.parametrize('name', PRESETS)
def test_metadata_keys_are_not_constructor_arguments(name):
    """The stripping is load-bearing, so check it actually bites."""
    fields = preset(name)
    assert any(key in fields for key in ALIGNER_META_KEYS)
    aligner = build_aligner(fields)
    for key in ALIGNER_META_KEYS:
        assert not hasattr(aligner, key)

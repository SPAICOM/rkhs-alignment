"""The receiver's probe is the measuring instrument, not a model.

Every study ranks alignment methods by what this probe makes of the
transported latents, so a probe that is well-behaved on its *own* latents
but erratic on anything else does not fail loudly -- it silently reorders
the results table. These tests pin the property that matters for that
role: the probe must respond to the latent, not to its scale.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.decoder import TimmDecoder

SEED = 0


def labelled_latents(
    n: int = 600, d: int = 40, scale: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Class-clustered latents with a long tail of near-empty directions.

    The tail is the point: a probe fitted without real regularisation
    puts enormous weight on directions its training data barely occupies,
    and those weights only misbehave when something *else* excites them.
    """
    rng = np.random.default_rng(SEED)
    y = rng.integers(0, 4, size=n)
    centres = rng.normal(size=(4, d)) * np.linspace(1.0, 1e-3, d)
    X = centres[y] + 0.25 * rng.normal(size=(n, d)) * np.linspace(1.0, 1e-3, d)
    return scale * X, y


def test_ridge_is_relative_so_the_probe_is_scale_free():
    """The same ``l2`` must mean the same thing at any latent scale.

    Encoder latents differ in magnitude by orders of magnitude across
    models, so an absolute ridge is a different amount of regularisation
    for every receiver in the study.
    """
    X, y = labelled_latents()
    small = TimmDecoder(input_dim=X.shape[1], n_classes=4, l2=1e-3).fit(X, y)
    large = TimmDecoder(input_dim=X.shape[1], n_classes=4, l2=1e-3).fit(
        1000.0 * X, y
    )

    # A ridge that tracks the data's own scale, not a fixed shift.
    assert large.ridge_ > 1e5 * small.ridge_
    assert large.score(1000.0 * X, y) == pytest.approx(
        small.score(X, y), abs=0.01
    )
    # The fit is the same map, just read in rescaled coordinates.
    assert np.allclose(large.W_ * 1000.0, small.W_, rtol=1e-6)


def test_regularisation_bounds_the_weights_on_starved_directions():
    """Without a real ridge the probe interpolates and its weights blow up."""
    X, y = labelled_latents()
    strong = TimmDecoder(input_dim=X.shape[1], n_classes=4, l2=1e-2).fit(X, y)
    weak = TimmDecoder(input_dim=X.shape[1], n_classes=4, l2=1e-12).fit(X, y)

    assert np.linalg.norm(weak.W_) > 10 * np.linalg.norm(strong.W_)
    # Both still read their own latents well; the damage only shows up on
    # inputs that excite the starved directions, which is exactly what a
    # transported latent does.
    assert strong.score(X, y) > 0.9
    assert weak.score(X, y) > 0.9


def test_probe_survives_energy_in_the_directions_its_data_starves():
    """A perturbation the probe should shrug off must not flip its answer.

    This is the alignment setting in miniature: a transported latent is
    the receiver's own latent plus a small error that does not respect
    the receiver's covariance. A probe fit for the job stays accurate; an
    unregularised one does not.
    """
    rng = np.random.default_rng(SEED)
    X, y = labelled_latents()
    tail = np.zeros(X.shape[1])
    tail[X.shape[1] // 2 :] = 1.0
    nudged = X + 0.05 * rng.normal(size=X.shape) * tail

    strong = TimmDecoder(input_dim=X.shape[1], n_classes=4, l2=1e-2).fit(X, y)
    weak = TimmDecoder(input_dim=X.shape[1], n_classes=4, l2=1e-12).fit(X, y)

    assert strong.score(nudged, y) > strong.score(X, y) - 0.02
    assert weak.score(nudged, y) < strong.score(nudged, y) - 0.1

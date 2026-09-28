"""Sanity checks for the alignment toolkit.

The RKHS tests are the ones ``idea.md`` calls out by name as the way to
tell a correct implementation from a subtly wrong one:

- the orthogonality constraint ``G X^T = 0`` must hold to machine
  precision (otherwise Step 4 is broken and the two stages are no longer
  identifiable);
- as ``lam`` grows the correction must vanish and the map must collapse
  onto the pure Procrustes baseline;
- the held-out error over a ``lam`` grid must be U-shaped, with the
  optimum strictly inside the grid.
"""

from __future__ import annotations

from itertools import pairwise
from unittest import mock

import numpy as np
import pytest
from scipy.linalg import null_space

from src.alignment import (
    PILOT_STRATEGIES,
    CCAAligner,
    CKAAligner,
    DirectMLPAligner,
    KCCAAligner,
    LatentScaler,
    LinearAligner,
    PPFEAligner,
    ProcrustesAligner,
    RelativeRepresentationAligner,
    ResidualMLPAligner,
    RKHSAligner,
    SVCCAAligner,
    check_paired_dims,
    cka_score,
    orthogonal_procrustes,
    parseval_frame,
    prune_anchors,
    reconstruction_metrics,
    retrieval_metrics,
    select_pilot_path,
    select_pilots,
)
from src.alignment.cca import _robust_svd
from src.anchors import Anchor
from src.kernels import Kernel
from src.latent import make_synthetic_agents
from src.manifold import spherical_exp_map, spherical_log_map

SEED = 0


def paired_data(
    n: int = 500,
    d_src: int = 10,
    d_tgt: int = 10,
    nonlinear: float = 1.0,
    noise: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Paired latents related by a rotation plus a smooth non-linearity.

    The projection feeding the sine is normalised by ``sqrt(d_src)`` so
    its argument stays order-1: an unnormalised one oscillates far faster
    than any realistic sample budget can resolve, which would make the
    residual stage look useless for reasons that have nothing to do with
    the implementation. The observation noise is what gives the
    regularisation sweep something to over-fit, and hence its U shape.
    """
    rng = np.random.default_rng(SEED)
    Z = rng.normal(size=(n, d_src))
    Q = np.linalg.qr(rng.normal(size=(d_tgt, d_tgt)))[0][:, :d_src]
    mixing = rng.normal(size=(d_src, d_tgt)) / np.sqrt(d_src)
    Y = Z @ Q.T + nonlinear * np.sin(Z @ mixing)
    return Z, Y + noise * rng.normal(size=(n, d_tgt))


# ---------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------


def test_whitening_gives_identity_covariance_and_inverts():
    rng = np.random.default_rng(SEED)
    X = rng.normal(size=(400, 12)) @ rng.normal(size=(12, 12)) + 3.0

    # Shrinkage off: only then is the covariance *exactly* the identity.
    scaler = LatentScaler('whiten', eps=1e-10, shrinkage=None)
    Z = scaler.fit_transform(X)

    cov = (Z - Z.mean(axis=0)).T @ (Z - Z.mean(axis=0)) / Z.shape[0]
    assert np.allclose(cov, np.eye(12), atol=1e-6)
    assert np.allclose(scaler.inverse_transform(Z), X, atol=1e-6)


def test_shrinkage_tracks_the_sample_regime():
    """Ledoit-Wolf intensity should fall away as n grows past d."""
    rng = np.random.default_rng(SEED)
    mixing = rng.normal(size=(40, 40))

    intensities = []
    for n in (50, 200, 2000):
        X = rng.normal(size=(n, 40)) @ mixing
        scaler = LatentScaler('whiten').fit(X)
        intensities.append(scaler.shrinkage_)

    assert intensities == sorted(intensities, reverse=True)
    assert intensities[0] > 5 * intensities[-1]
    assert LatentScaler('whiten', shrinkage=None).fit(X).shrinkage_ == 0.0


def test_shrinkage_keeps_whitening_usable_when_n_matches_d():
    """Without it, a transform fitted at n == d does not generalise.

    The sample covariance is technically full rank there but its small
    eigenvalues are noise, so inverting its square root is fine in-sample
    and wild out of sample. This is the failure that made the pilot sweep
    non-monotonic.
    """
    rng = np.random.default_rng(SEED)
    d = 60
    mixing = rng.normal(size=(d, d))
    fit = rng.normal(size=(d, d)) @ mixing
    held_out = rng.normal(size=(500, d)) @ mixing

    def out_of_sample_scale(shrinkage):
        scaler = LatentScaler('whiten', shrinkage=shrinkage).fit(fit)
        return float(np.abs(scaler.transform(held_out)).max())

    assert out_of_sample_scale(None) > 20 * out_of_sample_scale('auto')


@pytest.mark.parametrize(
    'method', ['none', 'center', 'standard', 'pca', 'whiten']
)
def test_scaler_roundtrip(method):
    rng = np.random.default_rng(SEED)
    X = rng.normal(size=(200, 7)) * 5.0 + 1.0
    scaler = LatentScaler(method).fit(X)
    assert np.allclose(
        scaler.inverse_transform(scaler.transform(X)), X, atol=1e-8
    )


def test_pca_decorrelates_without_rescaling():
    """PCA diagonalises the covariance but leaves the variances alone."""
    rng = np.random.default_rng(SEED)
    d = 6
    mixing = rng.normal(size=(d, d))
    X = rng.normal(size=(1000, d)) @ mixing

    Z = LatentScaler('pca', shrinkage=None).fit_transform(X)
    C = np.cov(Z, rowvar=False, bias=True)
    off = C - np.diag(np.diag(C))

    assert np.abs(off).max() < 1e-10  # decorrelated
    # ...but NOT unit variance, which is the whole difference from 'whiten'
    assert np.abs(np.diag(C) - 1.0).max() > 0.1
    # the retained spectrum is the covariance spectrum, descending
    evals = np.linalg.eigvalsh(np.cov(X, rowvar=False, bias=True))[::-1]
    assert np.allclose(np.diag(C), evals)
    assert np.all(np.diff(np.diag(C)) <= 1e-12)


def test_pca_is_a_rotation():
    """Full-rank PCA preserves distances; whitening does not."""
    rng = np.random.default_rng(SEED)
    X = rng.normal(size=(300, 5)) @ rng.normal(size=(5, 5))

    Z = LatentScaler('pca', shrinkage=None).fit_transform(X)
    Xc = X - X.mean(axis=0)
    assert np.allclose(np.linalg.norm(Z, axis=1), np.linalg.norm(Xc, axis=1))

    W = LatentScaler('whiten', shrinkage=None).fit_transform(X)
    assert not np.allclose(
        np.linalg.norm(W, axis=1), np.linalg.norm(Xc, axis=1)
    )


def test_pca_ignores_shrinkage():
    """Shrinking toward a scaled identity cannot move the eigenvectors."""
    rng = np.random.default_rng(SEED)
    X = rng.normal(size=(40, 12)) @ rng.normal(size=(12, 12))

    plain = LatentScaler('pca', shrinkage=None).fit(X)
    shrunk = LatentScaler('pca', shrinkage='auto').fit(X)

    assert shrunk.shrinkage_ == 0.0
    assert np.allclose(plain.forward_, shrunk.forward_)


def test_pca_truncation_projects():
    """Truncated PCA keeps the leading subspace and projects onto it."""
    rng = np.random.default_rng(SEED)
    d, k = 8, 3
    X = rng.normal(size=(400, d)) @ np.diag(np.linspace(10.0, 0.1, d))

    scaler = LatentScaler('pca', shrinkage=None, n_components=k).fit(X)
    Z = scaler.transform(X)
    assert Z.shape == (400, k)
    assert scaler.out_dim == k

    # Idempotent: round-tripping lands in the subspace and stays there.
    back = scaler.inverse_transform(Z)
    assert np.allclose(scaler.transform(back), Z)
    # and the projection is the best rank-k one, so it beats any other
    # k columns of the raw data on reconstruction error.
    err = np.linalg.norm(X - back)
    assert err < np.linalg.norm(X - X[:, :k] @ np.eye(k, d))


def test_pca_then_rescale_equals_whiten():
    """'whiten' is 'pca' followed by a division by sqrt(lambda).

    Compared at ``k < d``, where whitening stays in the principal basis.
    At full rank it rotates back (ZCA), which is the same transform
    composed with an orthogonal matrix -- see the next test.
    """
    rng = np.random.default_rng(SEED)
    X = rng.normal(size=(500, 6)) @ rng.normal(size=(6, 6))

    Z = LatentScaler('pca', shrinkage=None, n_components=4).fit_transform(X)
    manual = Z / Z.std(axis=0)  # ddof=0, i.e. divide by sqrt(lambda)

    pca_white = LatentScaler(
        'whiten', shrinkage=None, eps=1e-12, n_components=4
    ).fit_transform(X)
    assert np.allclose(manual, pca_white)


def test_full_rank_whiten_is_zca():
    """At ``k == d`` whitening is symmetric, and is PCA whitening rotated.

    ZCA is the unique symmetric whitening matrix (Kessy, Lewandowski &
    Strimmer 2018), so this pins which of the whitening family is used.
    """
    rng = np.random.default_rng(SEED)
    d = 6
    X = rng.normal(size=(500, d)) @ rng.normal(size=(d, d))

    zca = LatentScaler('whiten', shrinkage=None, eps=1e-12).fit(X)
    assert zca.out_dim == d
    assert np.allclose(zca.forward_, zca.forward_.T)

    pca = LatentScaler('pca', shrinkage=None).fit(X)
    Z = pca.transform(X)
    pca_forward = pca.forward_ / Z.std(axis=0)
    rotation = np.linalg.solve(pca_forward, zca.forward_)
    assert np.allclose(rotation @ rotation.T, np.eye(d))


ALL_METHODS = ['none', 'center', 'standard', 'pca', 'pga', 'whiten']


@pytest.mark.parametrize('method', ALL_METHODS)
def test_every_method_inverts_exactly(method):
    """transmit -> align -> receive: the receiver's inverse must be exact.

    Anything this loses is lost from the delivered latent, so every
    full-rank method is required to round-trip to machine precision --
    including ``'pga'``, which is not affine.
    """
    rng = np.random.default_rng(SEED)
    X = rng.normal(size=(200, 7)) * 5.0 + 1.0
    scaler = LatentScaler(method).fit(X)

    assert scaler.is_invertible
    assert scaler.out_dim == X.shape[1]
    assert np.allclose(scaler.inverse_transform(scaler.transform(X)), X)

    # ...and on points the scaler was never fitted to.
    held_out = rng.normal(size=(50, 7)) * 5.0 + 1.0
    assert np.allclose(
        scaler.inverse_transform(scaler.transform(held_out)), held_out
    )


@pytest.mark.parametrize('method', ['pca', 'pga', 'whiten'])
def test_truncation_reports_itself_as_lossy(method):
    """Truncation is the one lossy case, and it must say so."""
    rng = np.random.default_rng(SEED)
    X = rng.normal(size=(300, 8)) @ np.diag(np.linspace(9.0, 0.1, 8))
    scaler = LatentScaler(method, n_components=3).fit(X)

    assert not scaler.is_invertible
    assert scaler.out_dim < X.shape[1]
    # A projection: lossy once, idempotent thereafter.
    Z = scaler.transform(X)
    back = scaler.inverse_transform(Z)
    assert not np.allclose(back, X)
    assert np.allclose(scaler.transform(back), Z, atol=1e-8)


@pytest.mark.parametrize('method', ALL_METHODS)
def test_aligner_pipeline_round_trip(method):
    """The whole Tx -> align -> Rx chain lands in the raw target space."""
    rng = np.random.default_rng(SEED)
    n, d_src = 150, 6
    Z = rng.normal(size=(n, d_src))
    rotation = np.linalg.qr(rng.normal(size=(d_src, d_src)))[0]
    X_src = Z @ rotation + 3.0
    X_tgt = Z * 2.0 - 1.0

    aligner = ProcrustesAligner(preprocess=method).fit(X_src, X_tgt)
    out = aligner.transform(X_src)
    assert out.shape == X_tgt.shape
    assert np.all(np.isfinite(out))

    # An identity pairing must survive the chain untouched: preprocessing
    # on the sender, the identity map, and the inverse on the receiver.
    same = ProcrustesAligner(preprocess=method).fit(X_tgt, X_tgt)
    assert np.allclose(same.transform(X_tgt), X_tgt, atol=1e-6)


def test_pga_is_a_spherical_chart():
    """PGA reproduces the sphere's geometry, not a linear approximation."""
    rng = np.random.default_rng(SEED)
    d = 5
    # A cloud genuinely on a sphere of radius 3, in a cap.
    base = np.zeros(d)
    base[0] = 1.0
    V = rng.normal(size=(400, d)) * 0.3
    V -= np.outer(V @ base, base)  # tangent at `base`
    unit = spherical_exp_map(base, V)
    X = 3.0 * unit

    scaler = LatentScaler('pga').fit(X)
    # The Frechet mean sits on the sphere, near the cap's centre.
    assert np.isclose(np.linalg.norm(scaler.sphere_mean_), 1.0)
    assert scaler.sphere_mean_ @ base > 0.9
    # Constant radius -> the radial coordinate carries no information.
    Zc = scaler.transform(X)
    assert np.abs(Zc[:, 0]).max() < 1e-9
    assert np.allclose(scaler.inverse_transform(Zc), X)


def test_pga_recovers_radius_and_direction_separately():
    """The radial coordinate is log-scale and the rest is angular."""
    rng = np.random.default_rng(SEED)
    X = rng.normal(size=(200, 6))
    scaler = LatentScaler('pga').fit(X)

    Z = scaler.transform(X)
    # Scaling a point moves only the radial coordinate, by log(factor).
    Z2 = scaler.transform(X * 4.0)
    assert np.allclose(Z2[:, 0] - Z[:, 0], np.log(4.0))
    assert np.allclose(Z2[:, 1:], Z[:, 1:])


def test_pga_differs_from_pca_on_curved_data():
    """On a curved cloud PGA is a different chart, with a different job.

    PCA minimises *ambient* squared error and is optimal at that by
    Eckart-Young, so it wins any comparison phrased in those terms. PGA
    minimises squared *geodesic* distance in the tangent space, and keeps
    the radius as its own exact coordinate. Both are asserted here; the
    ambient comparison deliberately is not.
    """
    rng = np.random.default_rng(SEED)
    d = 5
    base = np.zeros(d)
    base[0] = 1.0
    V = rng.normal(size=(1500, d)) * 0.35
    V -= np.outer(V @ base, base)
    X = spherical_exp_map(base, V) * np.exp(rng.normal(size=(1500, 1)) * 0.4)

    pga = LatentScaler('pga').fit(X)
    pca = LatentScaler('pca').fit(X)
    # Both invert exactly at full rank...
    assert np.allclose(pga.inverse_transform(pga.transform(X)), X)
    assert np.allclose(pca.inverse_transform(pca.transform(X)), X)
    # ...but they are different charts.
    assert not np.allclose(pga.transform(X), pca.transform(X))

    def geodesic_and_radial_error(scaler):
        back = scaler.inverse_transform(scaler.transform(X))
        r_in = np.linalg.norm(X, axis=1)
        r_out = np.linalg.norm(back, axis=1)
        cos = np.sum((X / r_in[:, None]) * (back / r_out[:, None]), axis=1)
        angle = np.arccos(np.clip(cos, -1.0, 1.0))
        return (
            float(np.sqrt((angle**2).mean())),
            float(np.abs(np.log(r_out / r_in)).max()),
        )

    # At an equal angular budget -- k geodesic directions against k
    # principal ones -- PGA is closer in the metric it optimises.
    for k in (2, 3):
        geo_pga, rad_pga = geodesic_and_radial_error(
            LatentScaler('pga', n_components=k).fit(X)
        )
        geo_pca, rad_pca = geodesic_and_radial_error(
            LatentScaler('pca', n_components=k).fit(X)
        )
        assert geo_pga < geo_pca
        # And truncation never touches the radius: it has its own
        # coordinate, which the truncation does not reach.
        assert rad_pga < 1e-12
        assert rad_pca > 0.1


def test_pga_flat_data_matches_pca_directions():
    """On a narrow cap the sphere is flat, so PGA and PCA must agree."""
    rng = np.random.default_rng(SEED)
    d = 5
    base = np.zeros(d)
    base[0] = 1.0
    V = rng.normal(size=(2000, d)) * 1e-4  # tiny cap: curvature negligible
    V -= np.outer(V @ base, base)
    X = spherical_exp_map(base, V)

    pga = LatentScaler('pga').fit(X)
    tangent = spherical_log_map(pga.sphere_mean_, X)
    flat = LatentScaler('pca', shrinkage=None).fit(tangent)
    # Same leading directions, up to the sign convention both apply.
    overlap = np.abs(
        np.sum(pga.tangent_basis_[:, :3] * flat.forward_[:, :3], axis=0)
    )
    assert np.allclose(overlap, 1.0, atol=1e-6)


def test_pga_survives_origin_points(caplog):
    """A point at the origin has no direction; warn rather than divide."""
    rng = np.random.default_rng(SEED)
    X = rng.normal(size=(60, 4))
    X[7] = 0.0

    with caplog.at_level('WARNING', logger='src.alignment.preprocessing'):
        scaler = LatentScaler('pga').fit(X)
    assert 'origin' in caplog.text

    Z = scaler.transform(X)
    assert np.all(np.isfinite(Z))
    # every other point still round-trips exactly
    back = scaler.inverse_transform(Z)
    keep = np.arange(len(X)) != 7
    assert np.allclose(back[keep], X[keep])


def test_deterministic_signs_survive_a_reflection():
    """The basis must not flip when LAPACK feels like it."""
    rng = np.random.default_rng(SEED)
    X = rng.normal(size=(300, 6)) @ np.diag(np.linspace(4.0, 0.5, 6))
    a = LatentScaler('pca', shrinkage=None).fit(X).forward_
    b = LatentScaler('pca', shrinkage=None).fit(X.copy()).forward_
    assert np.array_equal(a, b)
    # largest-magnitude entry of each column is positive
    pivot = np.argmax(np.abs(a), axis=0)
    assert np.all(a[pivot, np.arange(a.shape[1])] > 0)


# ---------------------------------------------------------------------
# Procrustes, including the rectangular regimes
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ('d_src', 'd_tgt', 'regime'),
    [(10, 10, 'square'), (6, 14, 'embedding'), (14, 6, 'projection')],
)
def test_rectangular_procrustes_regimes(d_src, d_tgt, regime):
    rng = np.random.default_rng(SEED)
    X = rng.normal(size=(300, d_src))
    Q_true = np.linalg.qr(rng.normal(size=(max(d_src, d_tgt),) * 2))[0][
        :d_tgt, :d_src
    ]
    fit = orthogonal_procrustes(X, X @ Q_true.T)

    assert fit.Q.shape == (d_tgt, d_src)
    assert fit.regime == regime
    # Semi-orthogonality holds on the smaller side only.
    small = min(d_src, d_tgt)
    gram = fit.Q.T @ fit.Q if d_src <= d_tgt else fit.Q @ fit.Q.T
    assert np.allclose(gram, np.eye(small), atol=1e-10)


def test_procrustes_recovers_a_known_rotation():
    rng = np.random.default_rng(SEED)
    X = rng.normal(size=(400, 8))
    Q_true = np.linalg.qr(rng.normal(size=(8, 8)))[0]
    assert np.allclose(orthogonal_procrustes(X, X @ Q_true.T).Q, Q_true)


def test_check_paired_dims_rejects_unpaired_input():
    with pytest.raises(ValueError, match='paired sample-by-sample'):
        check_paired_dims(np.zeros((10, 3)), np.zeros((9, 3)))
    with pytest.raises(ValueError, match='2-dimensional'):
        check_paired_dims(np.zeros(10), np.zeros((10, 3)))


# ---------------------------------------------------------------------
# RKHS residual alignment -- the checks idea.md asks for
# ---------------------------------------------------------------------


def test_residual_correction_is_orthogonal_to_the_source():
    X, Y = paired_data()
    aligner = RKHSAligner(lam=1e-3).fit(X, Y)
    assert aligner.summary()['rkhs_orthogonality'] < 1e-6


def test_dropping_the_constraint_breaks_orthogonality():
    X, Y = paired_data()
    aligner = RKHSAligner(lam=1e-3, orthogonal_residual=False).fit(X, Y)
    assert aligner.summary()['rkhs_orthogonality'] > 1e-3


# RKA and its two ablations: no constraint, then no rigid stage either.
_RKA_VARIANTS = [
    {},
    {'orthogonal_residual': False},
    {'orthogonal_residual': False, 'rigid_stage': False},
]


@pytest.mark.parametrize('variant', _RKA_VARIANTS)
def test_resolving_at_another_lambda_matches_a_fresh_fit(variant):
    """`set_lam` reuses the eigendecomposition; the model must not notice."""
    X, Y = paired_data()
    swept = RKHSAligner(lam=1e-6, lam_grid=None, **variant).fit(X, Y)
    for lam in (1e-4, 1e-2, 1.0):
        fresh = RKHSAligner(lam=lam, lam_grid=None, **variant).fit(X, Y)
        swept.set_lam(lam)
        assert np.allclose(swept.transform(X), fresh.transform(X), atol=1e-10)
        for key in ('rkhs_dof', 'rkhs_residual_r2', 'rkhs_orthogonality'):
            assert np.isclose(
                swept.summary()[key], fresh.summary()[key], atol=1e-10
            ), key


def test_large_lambda_collapses_onto_procrustes():
    X, Y = paired_data()
    rkhs = RKHSAligner(lam=1e9).fit(X, Y)
    procrustes = ProcrustesAligner().fit(X, Y)

    assert np.abs(rkhs.A_).max() < 1e-8
    assert np.allclose(rkhs.transform(X), procrustes.transform(X), atol=1e-6)


def test_dropping_the_rigid_stage_is_plain_kernel_ridge():
    """``Q = 0`` and ``E = Y``: centred kernel ridge from source to target."""
    X, Y = paired_data()
    n, lam = X.shape[0], 1e-3
    aligner = RKHSAligner(
        lam=lam,
        lam_grid=None,
        lam_scaling='n',
        orthogonal_residual=False,
        rigid_stage=False,
    ).fit(X, Y)

    Z_src = aligner.scaler_src_.transform(X)
    J = np.eye(n) - 1.0 / n
    Kc = J @ aligner.kernel_spec(Z_src, Z_src) @ J
    A = np.linalg.solve(
        Kc + n * lam * np.eye(n), aligner.scaler_tgt_.transform(Y)
    )
    expected = aligner.scaler_tgt_.inverse_transform(Kc @ A)

    assert np.allclose(aligner.transform(X), expected, atol=1e-8)
    assert not aligner.Q.any()
    assert aligner.summary()['procrustes_regime'] == 'none'


def test_without_the_rigid_stage_large_lambda_collapses_onto_the_mean():
    X, Y = paired_data()
    aligner = RKHSAligner(
        lam=1e9, lam_grid=None, orthogonal_residual=False, rigid_stage=False
    ).fit(X, Y)
    mean = np.broadcast_to(Y.mean(axis=0), Y.shape)
    assert np.allclose(aligner.transform(X), mean, atol=1e-6)
    assert np.allclose(aligner.transform_linear(X), mean, atol=1e-10)


def test_plain_lambda_is_textbook_kernel_ridge():
    """`lam_scaling='plain'` shifts the Gram eigenvalues by `lam` itself."""
    X, Y = paired_data()
    n, lam = X.shape[0], 0.5
    aligner = RKHSAligner(
        lam=lam,
        lam_grid=None,
        lam_scaling='plain',
        orthogonal_residual=False,
        rigid_stage=False,
    ).fit(X, Y)

    assert aligner.summary()['rkhs_lam_effective'] == pytest.approx(lam)

    Z_src = aligner.scaler_src_.transform(X)
    J = np.eye(n) - 1.0 / n
    Kc = J @ aligner.kernel_spec(Z_src, Z_src) @ J
    A = np.linalg.solve(Kc + lam * np.eye(n), aligner.scaler_tgt_.transform(Y))
    expected = aligner.scaler_tgt_.inverse_transform(Kc @ A)
    assert np.allclose(aligner.transform(X), expected, atol=1e-8)


def test_the_conventions_are_one_path():
    """'plain' at `lam` is 'n' at `lam / N`: the same fit, relabelled."""
    X, Y = paired_data()
    n = X.shape[0]
    plain = RKHSAligner(lam=0.5, lam_grid=None, lam_scaling='plain').fit(X, Y)
    scaled = RKHSAligner(lam=0.5 / n, lam_grid=None, lam_scaling='n').fit(X, Y)
    assert np.allclose(plain.transform(X), scaled.transform(X), atol=1e-10)


def test_the_constraint_needs_the_rigid_stage():
    with pytest.raises(ValueError, match='orthogonal_residual=False'):
        RKHSAligner(rigid_stage=False)


def test_trace_scaling_is_a_reparametrisation():
    """`trace` relabels the lambda axis; it does not change the model.

    Every fit reachable under the literal `N*lam` convention must be
    reachable under `trace` at `lam / s_mean`, bit for bit. If this ever
    fails, the scaling has stopped being a change of units and started
    being a different estimator.
    """
    X, Y = paired_data()
    plain = RKHSAligner(lam=1e-3, lam_grid=None, lam_scaling='n').fit(X, Y)
    s_mean = plain._solver.s_mean_
    scaled = RKHSAligner(
        lam=1e-3 / s_mean, lam_grid=None, lam_scaling='trace'
    ).fit(X, Y)

    assert s_mean == pytest.approx(0.63, abs=0.1)  # rbf at median bandwidth
    assert np.array_equal(plain.A_, scaled.A_)
    assert np.array_equal(plain.transform(X), scaled.transform(X))
    # The shift is the quantity the two parametrisations agree on.
    assert plain.summary()['rkhs_lam_effective'] == pytest.approx(
        scaled.summary()['rkhs_lam_effective']
    )


def test_trace_scaling_makes_lambda_kernel_independent():
    """One grid, two kernels whose Gram matrices differ by ~10^5.

    Under `n` a shared grid regularises them by wildly different amounts,
    which is what made the kernel comparison in `alignment/rkhs.yaml`
    compare placement on the grid rather than the kernels.
    """
    X, Y = paired_data()
    shifts = {}
    for scaling in ('n', 'trace'):
        shifts[scaling] = [
            RKHSAligner(
                kernel=kernel, lam=1e-2, lam_grid=None, lam_scaling=scaling
            )
            .fit(X, Y)
            .summary()['rkhs_lam_effective']
            for kernel in ('rbf', 'polynomial')
        ]

    plain_ratio = shifts['n'][1] / shifts['n'][0]
    scaled_ratio = shifts['trace'][1] / shifts['trace'][0]
    assert plain_ratio == pytest.approx(1.0)  # same lam -> same shift
    assert scaled_ratio > 1e3  # ... on Gram matrices orders of magnitude apart


def test_gram_flatness_falls_with_the_source_dimension():
    """The concentration diagnostic: s_max/s_mean -> 1 in high dimension.

    At that point every Gram eigenvalue is equal, so `lam` selects an
    overall shrinkage rather than a smooth subspace and the residual can
    only interpolate the pilots.
    """
    rng = np.random.default_rng(0)
    flatness = []
    for d in (8, 128):
        X = rng.standard_normal((300, d))
        Y = X @ np.linalg.qr(rng.standard_normal((d, d)))[0].T
        aligner = RKHSAligner(lam=1e-3, lam_grid=None).fit(X, Y)
        flatness.append(aligner.summary()['rkhs_gram_flatness'])

    assert flatness[0] > 5 * flatness[1]
    assert flatness[1] < 5  # d=128, N=300: measured ~4.1


def test_unknown_lambda_scaling_is_rejected():
    with pytest.raises(ValueError, match='lam_scaling'):
        RKHSAligner(lam_scaling='per_sample')


def test_lambda_sweep_is_u_shaped():
    X, Y = paired_data(n=900, nonlinear=1.5)
    grid = list(np.logspace(-8, 3, 23))
    aligner = RKHSAligner(lam_grid=grid).fit(X, Y)

    losses = [point['val_nmse'] for point in aligner.lambda_path_]
    best = int(np.argmin(losses))
    assert 0 < best < len(grid) - 1, 'optimum should be interior to the grid'
    assert losses[0] > losses[best] and losses[-1] > losses[best]
    assert aligner.lam_ == pytest.approx(grid[best])


def test_transform_reproduces_the_training_fit():
    """Out-of-sample centring must agree with the training Gram matrix."""
    X, Y = paired_data()
    aligner = RKHSAligner(lam=1e-2).fit(X, Y)

    fitted = aligner._solver.solve(aligner.lam_)['G']
    expected = aligner.scaler_tgt_.inverse_transform(
        aligner.procrustes_.apply(aligner.scaler_src_.transform(X)) + fitted
    )
    assert np.allclose(aligner.transform(X), expected, atol=1e-8)


def test_rkhs_beats_procrustes_on_a_non_linear_pair():
    X, Y = paired_data(n=900, nonlinear=1.5)
    fit, test = slice(0, 700), slice(700, 900)

    rkhs = RKHSAligner(lam_grid=list(np.logspace(-6, 2, 17)))
    rkhs.fit(X[fit], Y[fit])
    procrustes = ProcrustesAligner().fit(X[fit], Y[fit])

    assert (
        reconstruction_metrics(rkhs.transform(X[test]), Y[test])['nmse']
        < reconstruction_metrics(procrustes.transform(X[test]), Y[test])[
            'nmse'
        ]
    )


def test_max_points_caps_the_kernel_stage():
    X, Y = paired_data(n=400)
    aligner = RKHSAligner(lam=1e-3, max_points=150).fit(X, Y)
    assert aligner.summary()['rkhs_n_kernel_points'] == 150
    assert aligner.transform(X[:20]).shape == (20, Y.shape[1])


# ---------------------------------------------------------------------
# Relative representations
# ---------------------------------------------------------------------


@pytest.mark.parametrize(('K', 'd'), [(8, 20), (20, 20), (50, 20)])
def test_parseval_frame_is_perfectly_conditioned_in_both_regimes(K, d):
    """``P^+ = P^T`` whichever side the frame is redundant on.

    ``K <= d`` gives orthonormal rows (a compression, decoding to the
    orthogonal projection onto the anchor span) and ``K >= d``
    orthonormal columns -- the Parseval frame the equalizer of Fiorellino
    et al. is built on, where the reconstruction formula is exact. Both
    have condition number 1, which is the property the decode needs.
    """
    rng = np.random.default_rng(SEED)
    P = parseval_frame(rng.normal(size=(K, d)))

    identity = np.eye(min(K, d))
    gram = P @ P.T if d >= K else P.T @ P
    assert np.allclose(gram, identity, atol=1e-10)
    assert np.allclose(np.linalg.pinv(P), P.T, atol=1e-10)
    assert np.linalg.cond(P) == pytest.approx(1.0)


def test_more_anchors_than_dimensions_reconstructs_exactly():
    """A redundant frame spans the space, so nothing is projected away."""
    X, _ = paired_data(d_src=10, d_tgt=10)
    aligner = RelativeRepresentationAligner(
        n_anchors=64, strategy='random', parseval=True, similarity='inner'
    ).fit(X, X)

    # Source and target are the same space here, so the two frames match
    # and the round trip is the identity -- exactly the Parseval
    # reconstruction formula. The tolerance is float32's, not float64's:
    # `Anchor` stores its point cloud single-precision.
    assert np.allclose(aligner.transform(X), X, atol=1e-5)


def test_anchor_budget_is_capped_at_the_calibration_set(caplog):
    X, Y = paired_data(n=40, d_src=10, d_tgt=10)
    with caplog.at_level('WARNING'):
        aligner = RelativeRepresentationAligner(
            n_anchors=200, strategy='random'
        ).fit(X, Y)
    assert aligner.summary()['rr_n_anchors_effective'] == 40
    assert 'exceeds the 40 available' in caplog.text


def test_parseval_projector_has_orthonormal_rows():
    X, Y = paired_data(d_src=12, d_tgt=12)
    aligner = RelativeRepresentationAligner(
        n_anchors=8, strategy='random', parseval=True
    ).fit(X, Y)
    for P in (aligner.P_src_, aligner.P_tgt_):
        assert np.allclose(P @ P.T, np.eye(8), atol=1e-8)
    assert aligner.summary()['rr_projector_cond'] == pytest.approx(1.0)


def test_parseval_makes_the_pseudo_inverse_a_transpose():
    """``P P^T = I`` implies ``P^+ = P^T``, so the decode is exact.

    Uses ``inner`` so no norm restoration is in play and the decode is
    purely the projection.
    """
    X, Y = paired_data(d_src=12, d_tgt=12)
    aligner = RelativeRepresentationAligner(
        n_anchors=8,
        strategy='random',
        parseval=True,
        decode='pinv',
        similarity='inner',
    ).fit(X, Y)

    # `readout_` holds (P^+)^T; under a Parseval frame that is exactly P.
    assert np.allclose(aligner.readout_, aligner.P_tgt_, atol=1e-10)

    # Decoding is then the orthogonal projection onto the anchor span.
    P = aligner.P_tgt_
    Z_src = aligner.scaler_src_.transform(X)
    decoded = aligner.scaler_tgt_.transform(aligner.transform(X))
    assert np.allclose(decoded, (Z_src @ aligner.P_src_.T) @ P, atol=1e-8)
    assert np.allclose(P.T @ P @ P.T @ P, P.T @ P, atol=1e-10)


def test_cosine_pinv_restores_the_norm_the_cosine_discarded():
    """The inverse projection returns a direction; the scale is refitted.

    The target carries a large mean, as real post-activation latents do:
    that is what an under-scaled reconstruction gets swamped by, so a
    zero-mean fixture would not exercise this at all.
    """
    X, Y = paired_data(n=600, d_src=12, d_tgt=12)
    Y = Y + 5.0
    common = {
        'n_anchors': 8,
        'strategy': 'random',
        'decode': 'pinv',
        'parseval': True,
    }

    aligner = RelativeRepresentationAligner(similarity='cosine', **common)
    aligner.fit(X, Y)
    scale = aligner.summary()['rr_norm_scale']

    # Cosine needs a real correction; inner is already on scale.
    assert scale > 2.0
    inner = RelativeRepresentationAligner(similarity='inner', **common)
    assert inner.fit(X, Y).summary()['rr_norm_scale'] == 1.0

    # Without it the prediction collapses onto the target mean, so every
    # row points the same way and retrieval degenerates.
    unscaled = aligner.scaler_tgt_.inverse_transform(
        aligner._transform(aligner.scaler_src_.transform(X)) / scale
    )
    assert (
        retrieval_metrics(aligner.transform(X), Y)['mrr']
        > 5 * (retrieval_metrics(unscaled, Y)['mrr'])
    )


def test_index_transfer_requires_an_index_based_strategy():
    X, Y = paired_data()
    aligner = RelativeRepresentationAligner(
        n_anchors=8, strategy='kmeans', medoids=False
    )
    with pytest.raises(ValueError, match='index-based'):
        aligner.fit(X, Y)


def test_cluster_transfer_works_with_plain_centroids():
    X, Y = paired_data()
    aligner = RelativeRepresentationAligner(
        n_anchors=8,
        strategy='kmeans',
        medoids=False,
        anchor_transfer='clusters',
    ).fit(X, Y)
    assert aligner.transform(X[:30]).shape == (30, Y.shape[1])


# ---------------------------------------------------------------------
# Literature baselines
# ---------------------------------------------------------------------


def test_cca_recovers_a_known_rotation():
    """A rotation is a perfect canonical correlation in every direction."""
    rng = np.random.default_rng(SEED)
    X = rng.normal(size=(400, 8))
    Q = np.linalg.qr(rng.normal(size=(8, 8)))[0]
    Y = X @ Q.T

    aligner = CCAAligner(reg=1e-8).fit(X, Y)
    assert np.allclose(aligner.transform(X), Y, atol=1e-6)
    assert aligner.summary()['cca_min_correlation'] == pytest.approx(
        1.0, abs=1e-6
    )


def test_cca_truncation_limits_the_rank_of_the_map():
    X, Y = paired_data(n=500, d_src=12, d_tgt=16)
    full = CCAAligner().fit(X, Y)
    truncated = CCAAligner(n_canonical=4).fit(X, Y)

    assert full.summary()['cca_n_canonical'] == 12  # min(d_src, d_tgt)
    assert truncated.summary()['cca_n_canonical'] == 4
    assert np.linalg.matrix_rank(truncated.W_, tol=1e-8) == 4


def svd_decaying_data(
    n: int = 600, d: int = 24, rank: int = 6, noise: float = 1e-3
) -> np.ndarray:
    """A cloud whose variance lives in `rank` directions, plus dust."""
    rng = np.random.default_rng(SEED)
    basis = np.linalg.qr(rng.normal(size=(d, d)))[0]
    scale = np.concatenate(
        [np.geomspace(10.0, 1.0, rank), np.full(d - rank, noise)]
    )
    return (rng.normal(size=(n, d)) * scale) @ basis.T


def test_cca_rate_is_the_canonical_rank_when_that_binds():
    """`transmitted_symbols` = min(whitening rank, canonical rank)."""
    X, Y = paired_data(n=500, d_src=12, d_tgt=16)

    # No whitening: the canonical rank is the only thing holding it down.
    assert CCAAligner(n_canonical=3).fit(X, Y).transmitted_symbols == 3
    # Still capped by min(d_src, d_tgt) with no truncation at all.
    assert CCAAligner().fit(X, Y).transmitted_symbols == 12
    # A truncated whitening binds instead when it is the smaller of the two.
    coarse = CCAAligner(preprocess='whiten', n_components=5, n_canonical=9)
    assert coarse.fit(X, Y).transmitted_symbols == 5


def test_cca_is_unbiased_under_every_preprocess():
    """The map centres and re-offsets itself, so `none` is not special."""
    X, Y = paired_data(n=400, d_src=10, d_tgt=10)
    X, Y = X + 7.0, Y - 3.0  # two offsets nothing else would remove

    fit = lambda method, reg=1e-3: (  # noqa: E731
        CCAAligner(preprocess=method, reg=reg).fit(X, Y).transform(X)
    )

    # Whether the base class removes the mean or the map does it itself,
    # the prediction is the same to machine precision -- and it lands on
    # the target's own offset rather than 7 @ W away from it.
    assert np.allclose(fit('none'), fit('center'), atol=1e-10)
    assert np.allclose(fit('none').mean(axis=0), Y.mean(axis=0), atol=1e-8)

    # Z-scoring is a linear change of basis, which CCA is invariant to
    # everywhere except in the ridge: the residual gap is the ridge's,
    # and it vanishes with it.
    assert not np.allclose(fit('none'), fit('standard'), atol=1e-6)
    assert np.allclose(fit('none', 1e-8), fit('standard', 1e-8), atol=1e-6)


def test_svcca_untruncated_matches_plain_cca():
    """With no truncation the two solve the same problem."""
    X, Y = paired_data(n=500, d_src=10, d_tgt=10)
    cca = CCAAligner(preprocess='center', reg=1e-8).fit(X, Y)
    svcca = SVCCAAligner(reg=1e-8).fit(X, Y)

    assert np.allclose(cca.transform(X), svcca.transform(X), atol=1e-6)


def test_svcca_recovers_a_known_rotation():
    rng = np.random.default_rng(SEED)
    X = rng.normal(size=(400, 8))
    Q = np.linalg.qr(rng.normal(size=(8, 8)))[0]

    aligner = SVCCAAligner(reg=1e-8).fit(X, X @ Q.T)
    assert np.allclose(aligner.transform(X), X @ Q.T, atol=1e-6)


def test_svcca_truncation_levels_are_independent_per_space():
    X, Y = paired_data(n=500, d_src=12, d_tgt=16)
    aligner = SVCCAAligner(svd_src=5, svd_tgt=9).fit(X, Y)
    summary = aligner.summary()

    assert summary['svcca_rank_src'] == 5
    assert summary['svcca_rank_tgt'] == 9
    assert aligner.P_src_.shape == (12, 5)
    assert aligner.P_tgt_.shape == (16, 9)
    # Orthonormal bases, and the map is stated in the full spaces.
    assert np.allclose(aligner.P_src_.T @ aligner.P_src_, np.eye(5))
    assert aligner.W_.shape == (12, 16)
    # k <= min(k_src, k_tgt): the SVD levels cap the canonical rank.
    assert summary['cca_n_canonical'] == 5
    assert np.linalg.matrix_rank(aligner.W_, tol=1e-8) == 5


def test_svcca_fractional_level_reads_as_explained_variance():
    """A float keeps the directions that carry the variance, not more."""
    X = svd_decaying_data(rank=6)
    Y = (
        X
        @ np.linalg.qr(np.random.default_rng(SEED).normal(size=(24, 24)))[0].T
    )

    aligner = SVCCAAligner(svd_src=0.999, svd_tgt=0.999).fit(X, Y)
    assert aligner.summary()['svcca_rank_src'] == 6
    assert aligner.summary()['svcca_variance_src'] >= 0.999


def test_svcca_canonical_rank_is_the_transmitted_rate():
    """n_canonical is the compression knob: k numbers on the channel."""
    X, Y = paired_data(n=500, d_src=12, d_tgt=16)
    aligner = SVCCAAligner(n_canonical=3).fit(X, Y)

    summary = aligner.summary()
    assert summary['transmitted_symbols'] == 3
    assert summary['map_parameters'] == 3 * (12 + 16)
    assert np.linalg.matrix_rank(aligner.W_, tol=1e-8) == 3


def test_svcca_takes_its_bases_from_the_unpaired_context():
    """The SVD basis needs no pairing, so it is not capped by the budget."""
    X, Y = paired_data(n=600, d_src=20, d_tgt=20)
    pilots = np.arange(8)

    lean = SVCCAAligner(svd_src=15, svd_tgt=15).fit(X[pilots], Y[pilots])
    rich = SVCCAAligner(svd_src=15, svd_tgt=15).fit(
        X[pilots], Y[pilots], src_context=X, tgt_context=Y
    )

    # Without context the basis comes from 8 pilots and cannot exceed
    # the rank they span once centred (n - 1); with it, all 15
    # directions are available.
    assert lean.summary()['svcca_rank_src'] == 7
    assert lean.summary()['svcca_basis_from'] == 'pilots'
    assert rich.summary()['svcca_rank_src'] == 15
    assert rich.summary()['svcca_basis_from'] == 'context'
    # And the context is not retained on the fitted aligner.
    assert rich._context_src is None


def test_svcca_rejects_malformed_truncation_levels():
    with pytest.raises(ValueError, match='svd_src'):
        SVCCAAligner(svd_src=0)
    with pytest.raises(ValueError, match='svd_tgt'):
        SVCCAAligner(svd_tgt=1.5)


def test_kcca_beats_linear_cca_when_the_pair_is_non_linear():
    """The whole point of kernelising -- and the price of the capacity.

    Held out, against the same canonical stage run on the coordinates.
    The kernel version wins where the rotation is genuinely a poor model
    and loses where it is nearly the right one, which is what a
    higher-capacity method fitted from the same pilots should do.
    """
    train, test = slice(0, 400), slice(400, None)

    def gap(nonlinear):
        X, Y = paired_data(
            n=800, d_src=6, d_tgt=6, nonlinear=nonlinear, noise=0.05
        )

        def nmse(aligner):
            error = aligner.transform(X[test]) - Y[test]
            return float(np.sum(error**2) / np.sum(Y[test] ** 2))

        linear = CCAAligner(preprocess='none').fit(X[train], Y[train])
        kernel = KCCAAligner().fit(X[train], Y[train])
        return nmse(kernel) / nmse(linear)

    assert gap(3.0) < 0.6
    assert gap(0.25) > 1.0


def test_kcca_reduces_to_the_paper_s_kernel_and_basis_sizes():
    """Sec. 3.2's window width, and Sec. 3.1's 99% eigenvalue basis."""
    X, Y = paired_data(n=400, d_src=5, d_tgt=5)
    aligner = KCCAAligner(n_basis_src=0.99, n_basis_tgt=0.99).fit(X, Y)
    summary = aligner.summary()

    # sigma^2 = 10 S with S = 1 after z-scoring, and gamma = 1 / 2 sigma^2.
    assert summary['kcca_gamma_src'] == pytest.approx(0.05)
    # The eigenvalue mass actually kept clears the level it was asked for,
    # and does so with far fewer columns than the 400 kernel functions.
    assert summary['kcca_eigenvalue_mass_src'] >= 0.99
    assert summary['kcca_columns_src'] < 400


def test_kcca_basis_routes_agree_on_the_columns_they_keep():
    """All three routes of Sec. 3.1 give a working map of the stated size."""
    X, Y = paired_data(n=400, d_src=6, d_tgt=6)
    common = {'n_basis_src': 20, 'n_basis_tgt': 20}

    svd = KCCAAligner(basis='svd', **common).fit(X, Y)
    subset = KCCAAligner(basis='subset', **common).fit(X, Y)
    hybrid = KCCAAligner(basis='hybrid', n_subset=100, **common).fit(X, Y)

    for aligner in (svd, subset, hybrid):
        assert aligner.summary()['kcca_columns_src'] == 20
        assert aligner.transform(X[:10]).shape == (10, 6)

    # Only the subset route is sparse in kernel centres: the SVD routes
    # evaluate against the whole pool (or the pre-subset), so they carry
    # far more of the calibration data into deployment.
    assert subset.summary()['kcca_basis_points_src'] == 20
    assert svd.summary()['kcca_basis_points_src'] == 400
    assert hybrid.summary()['kcca_basis_points_src'] == 100
    assert subset.map_parameters < hybrid.map_parameters < svd.map_parameters


def test_kcca_rate_is_the_canonical_rank():
    """The map factorises through k scores, and the split reproduces it."""
    X, Y = paired_data(n=400, d_src=6, d_tgt=8)
    aligner = KCCAAligner(n_canonical=4).fit(X, Y)

    assert aligner.summary()['transmitted_symbols'] == 4
    assert aligner.transmit(X[:10]).shape == (10, 4)
    # transmit -> channel -> receive is exactly `transform`, which is what
    # makes the rate claim true rather than notional.
    assert np.allclose(
        aligner.receive(aligner.transmit(X)), aligner.transform(X)
    )
    # The rate is not bounded by d_src the way a coordinate method's is:
    # the kernel data has m_1 columns whatever the latent width.
    assert KCCAAligner().fit(X, Y).transmitted_symbols > 6


def test_kcca_columns_are_capped_by_the_pilot_budget():
    """More kernel columns than pilots is rank the budget cannot support.

    The centred kernel data of ``n`` pilots spans at most ``n - 1``
    directions, so a basis asked for more than that would hand the
    canonical stage null directions for the ridge to invert into noise.
    """
    X, Y = paired_data(n=400, d_src=6, d_tgt=6)
    pilots = np.arange(30)

    for kwargs in (
        {'basis': 'svd'},
        {'basis': 'subset'},
        {'basis': 'hybrid', 'n_subset': 200},
    ):
        aligner = KCCAAligner(n_basis_src=100, n_basis_tgt=100, **kwargs).fit(
            X[pilots], Y[pilots]
        )
        assert aligner.summary()['kcca_columns_src'] == 29


def test_kcca_ignores_unpaired_context_but_still_standardises_from_it():
    """The basis stays on the pilots; only the scalers see the context.

    Unlike SVCCA's SVD basis, the kernel basis is consumed by a stage
    that needs pairing, so drawing it from a larger local pool outgrows
    the pilot budget rather than improving on it. The base class still
    fits its standardisation from the context, which is free.
    """
    X, Y = paired_data(n=600, d_src=6, d_tgt=6)
    pilots = np.arange(0, 300, 6)  # 50 of them

    lean = KCCAAligner().fit(X[pilots], Y[pilots])
    rich = KCCAAligner().fit(
        X[pilots], Y[pilots], src_context=X[:300], tgt_context=Y[:300]
    )

    # Same basis size either way -- it is drawn from the 50 pilots.
    assert (
        lean.summary()['kcca_basis_points_src']
        == rich.summary()['kcca_basis_points_src']
        == 50
    )
    # But the standardisation differs, so the two maps are not identical.
    assert not np.allclose(
        rich.scaler_src_.transform(X[:5]), lean.scaler_src_.transform(X[:5])
    )
    assert rich._context_tgt is None


def test_kcca_centres_its_kernel_data_and_lands_on_the_target_offset():
    """Centring is what disposes of the trivial canonical pair.

    Without it the leading canonical direction is the constant function
    at ``rho_0 = 1``, which carries nothing about either space, and the
    readout has no intercept to absorb the offset with.
    """
    X, Y = paired_data(n=300, d_src=5, d_tgt=5)
    aligner = KCCAAligner().fit(X, Y)

    # The kernel data is centred on the calibration set, so the scores
    # the readout is fitted on are too.
    assert np.allclose(aligner.transmit(X).mean(axis=0), 0.0, atol=1e-8)
    # And the map lands on the target's own offset, not a shifted one.
    assert np.allclose(
        aligner.transform(X).mean(axis=0), Y.mean(axis=0), atol=1e-6
    )


def test_kcca_independence_test_matches_the_paper_s_table_2():
    """Sec. 4.2's test: power on dependence, calibrated under the null.

    Cases I and III-1 of the paper's Table 2, at a tenth of its 100
    replicate runs. The published KCCA column reads 1.00 and 0.04.
    """

    def p_value(X, Y, run):
        fit = KCCAAligner(seed=run).fit(X, Y)
        return fit.summary()['kcca_independence_p']

    power, type_i = 0, 0
    for run in range(10):
        rng = np.random.default_rng(run)
        x = rng.normal(size=(500, 1))
        power += p_value(x, x**2, run) < 0.05
        z = rng.normal(size=(500, 2))
        type_i += p_value(z[:, :1], z[:, 1:], run) < 0.05

    assert power == 10  # Y = X^2 is invisible to a linear correlation
    assert type_i <= 2  # nominal 0.5 rejections out of 10


def test_kcca_rejects_malformed_configuration():
    with pytest.raises(ValueError, match='basis'):
        KCCAAligner(basis='eigen')
    with pytest.raises(ValueError, match='bandwidth'):
        KCCAAligner(bandwidth='silverman')
    with pytest.raises(ValueError, match='readout'):
        KCCAAligner(readout='both')
    with pytest.raises(ValueError, match='subset strategy'):
        KCCAAligner(subset_strategy='greedy')
    with pytest.raises(ValueError, match='n_basis_src'):
        KCCAAligner(n_basis_src=1.5)


# ---------------------------------------------------------------------
# CKA-based matching
# ---------------------------------------------------------------------


def test_cka_falls_monotonically_as_the_pairing_is_shuffled():
    """Table 1 of Maniparambil et al., which is what licenses Eq. (3).

    CKA is maximal on the ground-truth ordering and walks down towards
    zero as a growing fraction of the rows is permuted away. That is the
    whole basis of the method: if the score peaks at the true pairing,
    the permutation maximising it *is* the correspondence, and matching
    becomes an optimisation rather than a similarity lookup. The paper
    measures 0.72 -> 0.01 over the same sweep on 5k COCO pairs.
    """
    n = 300
    X, Y = paired_data(n=n, d_src=8, d_tgt=8)
    K = Kernel('rbf').fit(X)(X, X)
    L = Kernel('rbf').fit(Y)(Y, Y)

    rng = np.random.default_rng(SEED)
    scores = []
    for fraction in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        order = np.arange(n)
        shuffled = rng.choice(n, size=int(fraction * n), replace=False)
        order[shuffled] = rng.permutation(shuffled)
        scores.append(cka_score(K[np.ix_(order, order)], L))

    assert cka_score(K, K) == pytest.approx(1.0)
    assert all(before > after for before, after in pairwise(scores))
    assert scores[0] > 0.8 and scores[-1] < 0.1


def test_local_cka_equals_the_definition_it_is_derived_from():
    """Eq. (6) is "global CKA of the base set plus one query pair".

    The implementation never builds those ``(M+1) x (M+1)`` Grams -- it
    expands ``tr(KCLC)`` so the whole score matrix falls out of a single
    ``(p, M) x (M, q)`` product -- so this pins the closed form against
    the definition it replaces, one augmented Gram pair at a time.
    """
    X, Y = paired_data(n=140, d_src=6, d_tgt=6)
    aligner = CKAAligner(n_anchors=16, strategy='random').fit(X[:80], Y[:80])
    fast = aligner.local_cka(X[80:86], Y[80:90])

    base_src, base_tgt = aligner.base_src_, aligner.base_tgt_
    Z_src = aligner.scaler_src_.transform(X[80:86])
    Z_tgt = aligner.scaler_tgt_.transform(Y[80:90])
    slow = np.empty(fast.shape)
    for i, z in enumerate(Z_src):
        augmented_src = np.vstack([base_src.anchors, z])
        K = base_src.kernel(augmented_src, augmented_src)
        for j, h in enumerate(Z_tgt):
            augmented_tgt = np.vstack([base_tgt.anchors, h])
            L = base_tgt.kernel(augmented_tgt, augmented_tgt)
            slow[i, j] = cka_score(K, L)

    assert np.allclose(fast, slow, atol=1e-10)


@pytest.mark.parametrize('method', ['lsa', 'qap'])
def test_cka_recovers_a_shuffled_pairing(method):
    """The paper's caption-matching task, on a synthetic encoder pair.

    Both solvers of Sec. 4 are seeded by the same base set and differ
    only in what they optimise: ``'lsa'`` solves the localised surrogate
    exactly, ``'qap'`` runs FAQ on the global objective of Eq. (5).
    """
    X, Y = paired_data(n=600)
    aligner = CKAAligner(n_anchors=64).fit(X[:300], Y[:300])

    permutation = np.random.default_rng(SEED).permutation(80)
    matched = aligner.match(X[300:380], Y[300:380][permutation], method=method)
    assert np.mean(matched == np.argsort(permutation)) > 0.9


def test_stretching_is_what_makes_two_kernels_comparable():
    """Sec. 4.3's stretching matrix, worth +8.5 QAP points in Table 6.

    ``S = diag(1/std(x_l))`` is exactly ``preprocess='standard'``, and
    the reason it is load-bearing is that each side fits its own kernel
    bandwidth from a *pooled* distance scale: let one space's coordinates
    span two orders of magnitude and a single isotropic bandwidth sees
    only the loudest few, so the base Gram stops resolving the geometry
    the other side is being matched against.
    """
    X, Y = paired_data(n=400)
    Y = Y * 10.0 ** np.random.default_rng(3).uniform(-1, 2, size=Y.shape[1])

    stretched, plain = (
        CKAAligner(n_anchors=64, preprocess=preprocess)
        .fit(X[:200], Y[:200])
        .summary()
        for preprocess in ('standard', 'none')
    )
    assert stretched['cka_matching_accuracy'] == 1.0
    assert plain['cka_matching_accuracy'] < 0.4
    assert stretched['cka_score'] > 1.5 * plain['cka_score']


def test_cka_evaluates_its_kernels_only_against_the_base_set():
    """The base set is the whole kernel: no ``n x n`` Gram exists here.

    That is what separates this from :class:`RKHSAligner`, which carries
    its retained pilots into deployment, and it is why the rate is the
    base-set size rather than anything about the latent width.
    """
    X, Y = paired_data(n=400)
    aligner = CKAAligner(n_anchors=24).fit(X, Y)

    assert aligner.base_src_.gram.shape == (24, 24)
    assert aligner.base_tgt_.gram.shape == (24, 24)
    assert aligner.transmit(X[:5]).shape == (5, 24)
    assert aligner.transmitted_symbols == 24


def test_cka_base_set_is_the_same_samples_in_both_spaces():
    """Anchors are aligned *pairs*, so they have to be real samples."""
    X, Y = paired_data(n=200)
    aligner = CKAAligner(n_anchors=12, strategy='fps').fit(X, Y)

    indices = aligner.anchor_.indices
    assert np.allclose(
        aligner.base_src_.anchors, aligner.scaler_src_.transform(X)[indices]
    )
    assert np.allclose(
        aligner.base_tgt_.anchors, aligner.scaler_tgt_.transform(Y)[indices]
    )


def test_cka_base_set_must_be_index_based():
    """A centroid exists only in the space that computed it."""
    X, Y = paired_data(n=200)
    with pytest.raises(ValueError, match='index-based'):
        CKAAligner(n_anchors=8, strategy='kmeans', medoids=False).fit(X, Y)


def test_cka_base_set_is_capped_at_the_calibration_set(caplog):
    X, Y = paired_data(n=40)
    with caplog.at_level('WARNING'):
        aligner = CKAAligner(n_anchors=200, strategy='random').fit(X, Y)
    assert aligner.summary()['cka_n_anchors'] == 40
    assert 'exceeds the 40 available' in caplog.text


def test_cka_zero_shot_decodes_cost_only_the_base_set():
    """Two of the three decodes need no pilots beyond the anchors.

    The receiver's readout regression and its matching support set are
    both fitted on its own local latents, which have no partner, so the
    airtime bill is the base set alone. Only ``readout='source'`` is a
    paired regression and spends the whole budget.
    """
    X, Y = paired_data(n=300)
    paired = CKAAligner(n_anchors=32, readout='source').fit(X, Y)
    assert paired.paired_samples_used == 300

    for kwargs in ({'readout': 'target'}, {'decode': 'match'}):
        zero_shot = CKAAligner(n_anchors=32, **kwargs).fit(X, Y)
        assert zero_shot.paired_samples_used == 32
        assert zero_shot.transmitted_symbols == 32


def test_cka_feature_mismatch_predicts_the_zero_shot_readout():
    """The diagnostic that says whether the free decode can be afforded.

    ``readout='target'`` applies a decoder fitted on the receiver's own
    anchor features to the transmitter's, so it is exactly as good as the
    two feature spaces coinciding. Near an isometry it matches the paired
    fit for a fifth of the airtime; once the two encoders genuinely
    disagree it collapses, and ``cka_feature_mismatch`` moves first.
    """
    errors, mismatches = {}, {}
    for nonlinear in (0.0, 1.0):
        X, Y = paired_data(n=600, nonlinear=nonlinear)
        for readout in ('source', 'target'):
            aligner = CKAAligner(n_anchors=64, readout=readout).fit(
                X[:300], Y[:300]
            )
            errors[nonlinear, readout] = reconstruction_metrics(
                aligner.transform(X[300:]), Y[300:]
            )['nmse']
            mismatches[nonlinear] = aligner.summary()['cka_feature_mismatch']

    assert mismatches[0.0] < 0.25 < mismatches[1.0]
    # Near an isometry the free decode is as good as the paid one ...
    assert errors[0.0, 'target'] < 1.5 * errors[0.0, 'source']
    # ... and it is the mismatch, not the budget, that breaks it.
    assert errors[1.0, 'target'] > 5 * errors[1.0, 'source']


@pytest.mark.parametrize('decode', ['ridge', 'match'])
def test_cka_channel_split_reproduces_the_map(decode):
    """``transmit`` then ``receive`` is ``transform``, split at the channel.

    What crosses is the raw ``k(z, A)`` row: the centring and the
    ``HSIC(G, G)^{-1/2}`` scale are functions of the base Gram, which the
    receiver already holds, so spending airtime on them would be waste.
    """
    X, Y = paired_data(n=300)
    aligner = CKAAligner(n_anchors=32, decode=decode).fit(X, Y)
    assert np.allclose(
        aligner.receive(aligner.transmit(X[:25])), aligner.transform(X[:25])
    )


def test_hard_matching_returns_real_target_latents():
    """``match_temperature=0`` is retrieval, so its output is a sample.

    Raising the temperature mixes the top candidates instead, which is
    what a map wants -- a piecewise-constant map cannot land between two
    support points -- but the hard rule is the paper's, and it never
    invents a latent the receiver has not actually seen.
    """
    X, Y = paired_data(n=200)
    aligner = CKAAligner(
        n_anchors=16, decode='match', match_temperature=0.0
    ).fit(X, Y)

    support = aligner.scaler_tgt_.inverse_transform(aligner.support_)
    for row in aligner.transform(X[:30]):
        assert np.isclose(np.abs(support - row).sum(axis=1), 0.0).any()

    soft = CKAAligner(
        n_anchors=16, decode='match', match_temperature=0.25
    ).fit(X, Y)
    assert (
        reconstruction_metrics(soft.transform(X), Y)['nmse']
        < (reconstruction_metrics(aligner.transform(X), Y)['nmse'])
    )


def test_cka_rejects_malformed_configuration():
    with pytest.raises(ValueError, match='decode'):
        CKAAligner(decode='nearest')
    with pytest.raises(ValueError, match='readout'):
        CKAAligner(readout='both')
    with pytest.raises(ValueError, match='n_anchors'):
        CKAAligner(n_anchors=1)

    X, Y = paired_data(n=80)
    aligner = CKAAligner(n_anchors=8).fit(X, Y)
    with pytest.raises(ValueError, match='match method'):
        aligner.match(X[:10], Y[:10], method='greedy')
    with pytest.raises(ValueError, match='equal size'):
        aligner.match(X[:10], Y[:12])


def test_l_ortho_orthogonalises_the_least_squares_map():
    """The projection is a genuine isometry, but not the Procrustes one."""
    X, Y = paired_data(n=400, d_src=10, d_tgt=10)
    plain = LinearAligner(preprocess='standard').fit(X, Y)
    l_ortho = LinearAligner(preprocess='standard', orthogonalize=True).fit(
        X, Y
    )

    W = l_ortho.W_
    assert np.allclose(W.T @ W, np.eye(10), atol=1e-8)
    assert not np.allclose(W, plain.W_, atol=1e-3)

    # Procrustes optimises over the orthogonal group; l-ortho projects
    # onto it after the fact, so the two must not coincide.
    procrustes = ProcrustesAligner(preprocess='standard').fit(X, Y)
    assert not np.allclose(W, procrustes.Q.T, atol=1e-3)

    # And the diagnostic says how far the unconstrained fit already was.
    score = l_ortho.summary()['linear_orthogonality']
    assert 0.0 < score <= 1.0


def test_prototype_anchors_average_the_shared_support_set():
    """Both agents must mean the *same* samples, not the same cluster.

    With ``n_prototype_samples`` subsampling the cluster, averaging every
    member on the target side would build a different anchor than the
    source did -- and the anchor correspondence is the only thing holding
    the relative spaces together.
    """
    rng = np.random.default_rng(SEED)
    Z = rng.normal(size=(200, 6))
    anchor = Anchor(Z, strategy='kmeans', seed=SEED)
    anchor.fit(n_anchors=5, n_samples=3, medoids=False)

    for i in range(5):
        support = anchor.support_indices[i]
        assert support.size <= 3
        assert set(support).issubset(set(anchor.cluster_indices[i].tolist()))
        assert np.allclose(
            anchor.anchors[i], Z[support].mean(axis=0), atol=1e-5
        )


def test_ppfe_transfers_prototypes_through_the_support_set():
    X, Y = paired_data(n=300, d_src=10, d_tgt=14)
    aligner = PPFEAligner(n_anchors=20, n_prototype_samples=4).fit(X, Y)

    Z_tgt = aligner.scaler_tgt_.transform(Y)
    support = aligner.anchor_.support_indices
    expected = np.stack([Z_tgt[support[i]].mean(axis=0) for i in range(20)])
    # `P_tgt_` is the Parsevalised frame, so compare the row spaces the
    # prototypes span rather than the raw vectors.
    assert np.allclose(aligner.P_tgt_, parseval_frame(expected), atol=1e-8)


def test_ppfe_needs_a_redundant_frame_to_reconstruct():
    """Below ``N = d`` the equalizer compresses; above it, it is exact."""
    X, _ = paired_data(n=400, d_src=12, d_tgt=12)
    compressing = PPFEAligner(n_anchors=4).fit(X, X)
    redundant = PPFEAligner(n_anchors=64).fit(X, X)

    error = lambda a: reconstruction_metrics(a.transform(X), X)['nmse']  # noqa: E731
    assert error(compressing) > 0.1
    assert error(redundant) < 1e-8


def test_anchor_pruning_drops_correlated_anchors():
    rng = np.random.default_rng(SEED)
    base = rng.normal(size=(6, 12))
    # Twelve anchors that are really six directions, each duplicated with
    # a small jitter -- exactly the correlation the pruning targets.
    anchors = np.repeat(base, 2, axis=0) + 1e-3 * rng.normal(size=(12, 12))

    kept = prune_anchors(anchors, threshold=0.2, seed=SEED)
    assert kept.size == 6
    unit = anchors[kept] / np.linalg.norm(anchors[kept], axis=1, keepdims=True)
    off_diagonal = np.abs(unit @ unit.T) - np.eye(6)
    assert off_diagonal.max() < 0.8

    # A zero threshold is the no-op: every anchor survives.
    assert prune_anchors(anchors, threshold=0.0, seed=SEED).size == 12


def test_pruning_improves_the_conditioning_of_the_decode():
    X, Y = paired_data(n=400, d_src=12, d_tgt=12)
    common = {
        'n_anchors': 60,
        'strategy': 'random',
        'parseval': False,
        'preprocess': 'center',
    }
    plain = RelativeRepresentationAligner(**common).fit(X, Y)
    pruned = RelativeRepresentationAligner(
        prune_threshold=0.3, n_subspaces=4, **common
    ).fit(X, Y)

    assert (
        pruned.summary()['rr_projector_cond']
        < plain.summary()['rr_projector_cond']
    )
    assert pruned.summary()['rr_n_anchors_pruned'] < 60


def test_subspace_ensemble_is_the_mean_of_its_members():
    """The readout collapses the omega reconstructions into one matrix."""
    X, Y = paired_data(n=300, d_src=10, d_tgt=10)
    aligner = RelativeRepresentationAligner(
        n_anchors=40,
        strategy='random',
        parseval=False,
        similarity='inner',
        preprocess='center',
        prune_threshold=0.25,
        n_subspaces=5,
    ).fit(X, Y)

    Z = aligner.scaler_src_.transform(X)
    R = aligner._project(Z, aligner.P_src_)
    members = [
        R[:, s] @ np.linalg.pinv(aligner.P_tgt_[s]).T
        for s in aligner.subspaces_
    ]
    assert len(aligner.subspaces_) == 5
    assert np.allclose(
        aligner._transform(Z), np.mean(members, axis=0), atol=1e-10
    )


# ---------------------------------------------------------------------
# Shared contract
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    'make_aligner',
    [
        lambda: ProcrustesAligner(),
        lambda: LinearAligner(),
        lambda: RelativeRepresentationAligner(n_anchors=8, strategy='random'),
        lambda: RKHSAligner(lam=1e-3),
        lambda: CCAAligner(),
        lambda: PPFEAligner(n_anchors=40),
        lambda: LinearAligner(orthogonalize=True),
        lambda: RelativeRepresentationAligner(
            n_anchors=20,
            strategy='random',
            parseval=False,
            prune_threshold=0.2,
            n_subspaces=3,
        ),
    ],
)
def test_every_aligner_maps_into_the_target_space(make_aligner):
    X, Y = paired_data(n=300, d_src=9, d_tgt=15)
    aligner = make_aligner().fit(X, Y)

    assert aligner.transform(X[:25]).shape == (25, 15)
    assert aligner.summary()['dim_src'] == 9
    assert aligner.summary()['dim_tgt'] == 15

    with pytest.raises(ValueError, match='expected source latents'):
        aligner.transform(np.zeros((5, 4)))


def test_unfitted_aligner_refuses_to_transform():
    with pytest.raises(RuntimeError, match='fit\\(\\) must be called'):
        ProcrustesAligner().transform(np.zeros((3, 4)))


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------


def test_synthetic_agents_share_their_rows():
    agents = make_synthetic_agents(
        n_agents=3, n_train=120, n_test=60, ground_dim=5, agent_dims=[5, 8, 11]
    )
    assert [s['train'].dim for s in agents.values()] == [5, 8, 11]
    labels = [s['train'].labels for s in agents.values()]
    assert all(np.array_equal(labels[0], other) for other in labels[1:])


def test_agent_dims_below_ground_dim_are_rejected():
    with pytest.raises(ValueError, match='>= ground_dim'):
        make_synthetic_agents(n_agents=2, ground_dim=16, agent_dims=[16, 8])


# ---------------------------------------------------------------------
# Semantic-pilot selection
# ---------------------------------------------------------------------


def clustered_pool(seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Three blobs of very unequal mass, so sampling bias is visible."""
    rng = np.random.default_rng(seed)
    parts, labels = [], []
    for cls, (n, mu) in enumerate([(700, 0.0), (200, 8.0), (100, -8.0)]):
        parts.append(rng.normal(size=(n, 6)) + mu)
        labels.append(np.full(n, cls))
    return np.vstack(parts), np.concatenate(labels)


def _mmd2(pool: np.ndarray, chosen: np.ndarray, kernel: Kernel) -> float:
    """Squared MMD between the pool and the selected subset."""
    sub = pool[chosen]
    return float(
        kernel(pool, pool).mean()
        - 2 * kernel(sub, pool).mean()
        + kernel(sub, sub).mean()
    )


@pytest.mark.parametrize('strategy', PILOT_STRATEGIES)
def test_every_pilot_strategy_returns_usable_indices(strategy):
    X, y = clustered_pool()
    idx = select_pilots(X, 40, strategy=strategy, labels=y, seed=SEED)

    assert idx.ndim == 1 and idx.size <= 40
    assert idx.size == np.unique(idx).size, 'pilots must not repeat'
    assert idx.min() >= 0 and idx.max() < X.shape[0]
    assert np.array_equal(idx, np.sort(idx))


def test_kernel_herding_beats_random_at_matching_the_pool():
    """The MMD in Eq. 18 is what herding minimises, so measure that."""
    X, _ = clustered_pool()
    kernel = Kernel('rbf').fit(X, seed=SEED)

    herded = _mmd2(X, select_pilots(X, 40, strategy='herding'), kernel)
    draws = [
        _mmd2(X, select_pilots(X, 40, strategy='random', seed=s), kernel)
        for s in range(12)
    ]

    assert herded < min(draws), 'herding should beat every random draw'
    assert herded < 0.25 * float(np.mean(draws))


_ADVERSARIAL = ['anti_herding', 'ball', 'classes:2']


@pytest.mark.parametrize('design', _ADVERSARIAL)
def test_adversarial_designs_return_usable_indices(design):
    X, y = clustered_pool()
    idx = select_pilots(X, 40, strategy=design, labels=y, seed=SEED)

    assert idx.ndim == 1 and idx.size <= 40
    assert idx.size == np.unique(idx).size, 'pilots must not repeat'
    assert idx.min() >= 0 and idx.max() < X.shape[0]
    assert np.array_equal(idx, np.sort(idx))


def test_anti_herding_is_herdings_mirror():
    """Same score, opposite sign: it maximises the MMD herding minimises."""
    X, _ = clustered_pool()
    kernel = Kernel('rbf').fit(X, seed=SEED)

    herded = _mmd2(X, select_pilots(X, 40, strategy='herding'), kernel)
    against = _mmd2(X, select_pilots(X, 40, strategy='anti_herding'), kernel)
    draws = [
        _mmd2(X, select_pilots(X, 40, strategy='random', seed=s), kernel)
        for s in range(8)
    ]

    assert herded < min(draws), 'herding should beat every random draw'
    assert against > max(draws), 'anti-herding should lose to every draw'


@pytest.mark.parametrize('design', _ADVERSARIAL)
def test_adversarial_designs_are_nested(design):
    """A sweep slices one ordering, so the small budget is a prefix."""
    X, y = clustered_pool()
    path = select_pilot_path(
        X, counts=[10, 40], strategy=design, labels=y, seed=SEED
    )
    assert set(path[10].tolist()) <= set(path[40].tolist())
    assert path[10].size == 10 and path[40].size == 40


def test_classes_design_withholds_the_other_classes():
    X, y = clustered_pool()
    idx = select_pilots(X, 30, strategy='classes:2', labels=y, seed=SEED)
    assert np.unique(y[idx]).size == 2, 'only two classes may appear'
    assert np.unique(y).size > 2, 'the pool must have classes to withhold'


def test_classes_design_needs_labels_and_a_count():
    X, _ = clustered_pool()
    with pytest.raises(ValueError, match='requires per-sample labels'):
        select_pilots(X, 10, strategy='classes:2', seed=SEED)
    with pytest.raises(ValueError, match='positive class count'):
        select_pilots(X, 10, strategy='classes:0', seed=SEED)
    with pytest.raises(ValueError, match='takes no parameter'):
        select_pilots(X, 10, strategy='herding:2', seed=SEED)


def test_herding_fits_the_bandwidth_of_a_supplied_kernel():
    """A caller passes a kernel *spec*; the pool supplies its bandwidth.

    The herding score is a mean-embedding match, so the bandwidth has to
    belong to the cloud being selected from -- and requiring callers to
    pre-fit it made an unfitted spec raise on first evaluation instead.
    """
    rng = np.random.default_rng(SEED)
    X = rng.normal(size=(200, 6))

    sharp = select_pilots(
        X, 20, strategy='herding', kernel=Kernel('rbf', bandwidth_scale=0.1)
    )
    wide = select_pilots(
        X, 20, strategy='herding', kernel=Kernel('rbf', bandwidth_scale=10.0)
    )
    assert sharp.shape == (20,)
    # The bandwidth is load-bearing: it has to change the selection.
    assert not np.array_equal(sharp, wide)

    # An explicitly parameterised kernel is left exactly as given.
    fixed = Kernel('rbf', gamma=0.25)
    select_pilots(X, 10, strategy='herding', kernel=fixed)
    assert fixed.gamma == 0.25


def test_kernel_herding_is_deterministic():
    X, _ = clustered_pool()
    a = select_pilots(X, 30, strategy='herding', seed=0)
    b = select_pilots(X, 30, strategy='herding', seed=12345)
    assert np.array_equal(a, b)


def test_kernel_herding_preserves_pool_proportions():
    X, y = clustered_pool()
    idx = select_pilots(X, 50, strategy='herding')
    share = np.bincount(y[idx], minlength=3) / idx.size
    expected = np.bincount(y, minlength=3) / y.size
    assert np.max(np.abs(share - expected)) < 0.05


def test_select_pilots_rejects_bad_input():
    X, _ = clustered_pool()
    with pytest.raises(ValueError, match='Unknown pilot strategy'):
        select_pilots(X, 10, strategy='nope')
    with pytest.raises(ValueError, match='requires per-sample labels'):
        select_pilots(X, 10, strategy='stratified')
    with pytest.raises(ValueError, match='must be positive'):
        select_pilots(X, 0)


def test_pilot_budget_at_or_above_pool_size_keeps_everything():
    X, _ = clustered_pool()
    assert np.array_equal(
        select_pilots(X, X.shape[0] + 5, strategy='herding'),
        np.arange(X.shape[0]),
    )


# ---------------------------------------------------------------------
# Neural baselines
# ---------------------------------------------------------------------


@pytest.mark.parametrize('aligner_cls', [DirectMLPAligner, ResidualMLPAligner])
def test_mlp_baselines_map_into_the_target_space(aligner_cls):
    X, Y = paired_data(n=300, d_src=9, d_tgt=15)
    aligner = aligner_cls(max_iter=200).fit(X, Y)
    assert aligner.transform(X[:25]).shape == (25, 15)


def test_residual_mlp_keeps_procrustes_as_its_backbone():
    X, Y = paired_data(n=400)
    residual = ResidualMLPAligner(max_iter=200).fit(X, Y)
    procrustes = ProcrustesAligner().fit(X, Y)
    assert np.allclose(
        residual.transform_linear(X), procrustes.transform(X), atol=1e-8
    )


def test_learned_residual_is_not_orthogonally_constrained():
    """The contrast with RKA: nothing stops the MLP re-absorbing Q."""
    X, Y = paired_data(n=400, nonlinear=1.5)
    mlp = ResidualMLPAligner(max_iter=300).fit(X, Y)
    rkhs = RKHSAligner(lam=1e-3).fit(X, Y)
    assert mlp.summary()['mlp_orthogonality'] > 1e-3
    assert rkhs.summary()['rkhs_orthogonality'] < 1e-6


# ---------------------------------------------------------------------
# Rank-truncated whitening
# ---------------------------------------------------------------------


def anisotropic(n: int = 400, d: int = 40, seed: int = SEED) -> np.ndarray:
    """A cloud with a fast-decaying spectrum, like a real encoder latent."""
    rng = np.random.default_rng(seed)
    spectrum = np.diag(np.geomspace(5.0, 0.005, d))
    return rng.normal(size=(n, d)) @ spectrum @ rng.normal(size=(d, d))


def test_truncated_whitening_keeps_the_leading_subspace():
    X = anisotropic()
    scaler = LatentScaler('whiten', n_components=8).fit(X)

    assert scaler.out_dim == 8
    assert scaler.transform(X).shape == (X.shape[0], 8)
    assert scaler.inverse_transform(scaler.transform(X)).shape == X.shape
    # Eight directions of a decaying spectrum carry most of the variance
    # but not all of it -- the round trip is a projection, not an identity.
    assert 0.3 < scaler.explained_variance_ratio_ < 1.0


def test_variance_fraction_selects_the_rank():
    X = anisotropic()
    loose = LatentScaler('whiten', n_components=0.5).fit(X)
    tight = LatentScaler('whiten', n_components=0.99).fit(X)

    assert loose.out_dim < tight.out_dim <= X.shape[1]
    assert loose.explained_variance_ratio_ >= 0.5
    assert tight.explained_variance_ratio_ >= 0.99


def test_full_rank_whitening_is_unchanged_and_exactly_invertible():
    X = anisotropic()
    full = LatentScaler('whiten', n_components=None, shrinkage=None).fit(X)
    explicit = LatentScaler(
        'whiten', n_components=X.shape[1], shrinkage=None
    ).fit(X)

    assert full.out_dim == X.shape[1]
    assert np.allclose(full.inverse_transform(full.transform(X)), X, atol=1e-6)
    # Asking for every component must reproduce the untruncated transform
    # up to the sign of each eigenvector, so compare via the round trip.
    assert np.allclose(
        explicit.inverse_transform(explicit.transform(X)), X, atol=1e-6
    )


def test_bad_n_components_is_rejected():
    X = anisotropic(n=100, d=10)
    with pytest.raises(ValueError, match='n_components must be >= 1'):
        LatentScaler('whiten', n_components=0).fit(X)
    with pytest.raises(ValueError, match=r'must lie in \(0, 1\]'):
        LatentScaler('whiten', n_components=1.5).fit(X)


@pytest.mark.parametrize(
    'make_aligner',
    [
        lambda k: ProcrustesAligner(n_components=k),
        lambda k: LinearAligner(n_components=k),
        lambda k: RKHSAligner(n_components=k, lam=1e-3, lam_grid=None),
        lambda k: ResidualMLPAligner(n_components=k, max_iter=100),
    ],
)
def test_truncation_flows_through_every_aligner(make_aligner):
    X = anisotropic(d=40)
    Y = anisotropic(d=25, seed=SEED + 1)
    aligner = make_aligner(8).fit(X, Y)

    assert aligner.transform(X[:20]).shape == (20, 25)
    assert aligner.summary()['rank_src'] == 8
    assert aligner.summary()['rank_tgt'] == 8


def test_truncation_restores_a_decaying_gram_spectrum():
    """Why truncation matters: it is what gives ``lam`` something to do.

    In many whitened dimensions every pair of points is nearly
    equidistant, so the centred RBF Gram matrix tends to the identity and
    its spectrum flattens. A flat spectrum means no smoothing regime
    exists -- the ridge either interpolates or vanishes.
    """
    X = anisotropic(n=300, d=256)

    def gram_concentration(Z):
        K = Kernel('rbf').fit(Z, seed=SEED)(Z, Z)
        col = K.mean(axis=0)
        Kc = K - col[None, :] - col[:, None] + K.mean()
        s = np.clip(np.linalg.eigvalsh(0.5 * (Kc + Kc.T)), 0, None)[::-1]
        share = np.cumsum(s) / s.sum()
        return int(np.searchsorted(share, 0.9) + 1) / len(Z)

    flat = gram_concentration(LatentScaler('whiten').fit_transform(X))
    peaked = gram_concentration(
        LatentScaler('whiten', n_components=12).fit_transform(X)
    )
    assert peaked < 0.5 * flat


# ---------------------------------------------------------------------
# The rank condition hidden in the orthogonality constraint
# ---------------------------------------------------------------------


def constrained_pair(
    n: int, d_src: int = 60, d_tgt: int = 40, seed: int = SEED
) -> tuple[np.ndarray, np.ndarray]:
    """Paired latents with a genuine non-linear part to recover."""
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, d_src))
    linear = rng.normal(size=(d_src, d_tgt))
    wiggle = rng.normal(size=(d_src, d_tgt)) / np.sqrt(d_src)
    return Z, Z @ linear + 0.8 * np.sin(Z @ wiggle)


def residual_strength(aligner: RKHSAligner) -> float:
    """``||G|| / ||E||``: how much of the residual the fit actually keeps."""
    solution = aligner._solver.solve(aligner.lam_)
    return float(
        np.linalg.norm(solution['G']) / np.linalg.norm(aligner._solver.E)
    )


@pytest.mark.parametrize('n_pilots', [30, 59, 60, 61])
def test_constraint_annihilates_the_residual_below_full_rank(n_pilots):
    """``G X^T = 0`` has only the zero solution when ``N <= d_src``.

    Each row of ``G`` must lie in ``ker(X)``, and -- because the pilots
    are centred and ``G = A K`` with ``K 1 = 0`` -- also in ``1^perp``.
    That leaves ``N - 1 - min(d_src, N - 1)`` free directions, which is
    zero for every ``N <= d_src + 1``. RKA is then *exactly* Procrustes,
    at every lambda and for every kernel.
    """
    X, Y = constrained_pair(n_pilots, d_src=60)
    aligner = RKHSAligner(lam=1e-4, lam_grid=None).fit(X, Y)

    assert residual_strength(aligner) < 1e-5

    # Compare the maps, not individual entries: a handful of near-zero
    # coordinates make an element-wise relative tolerance meaningless.
    rigid = ProcrustesAligner().fit(X, Y).transform(X)
    deviation = np.linalg.norm(aligner.transform(X) - rigid)
    assert deviation / np.linalg.norm(rigid) < 1e-6


def test_residual_capacity_grows_with_the_surplus_over_d_src():
    """Above ``N = d_src`` the residual gets ``N - d_src`` directions."""
    strengths = [
        residual_strength(
            RKHSAligner(lam=1e-4, lam_grid=None).fit(
                *constrained_pair(n, d_src=60)
            )
        )
        for n in (61, 80, 150, 400, 1200)
    ]
    assert strengths[0] < 1e-5, 'N = d_src + 1 still has no free direction'
    assert strengths == sorted(strengths), 'capacity must grow with N'
    assert strengths[-1] > 0.1


def test_only_the_source_dimension_enters_the_constraint():
    """``d_tgt`` is absent from ``G X^T = 0``, so it cannot gate anything."""
    wide_target = residual_strength(
        RKHSAligner(lam=1e-4, lam_grid=None).fit(
            *constrained_pair(200, d_src=40, d_tgt=200)
        )
    )
    wide_source = residual_strength(
        RKHSAligner(lam=1e-4, lam_grid=None).fit(
            *constrained_pair(200, d_src=200, d_tgt=40)
        )
    )
    assert wide_target > 0.05
    assert wide_source < 1e-5


def test_feasible_dimension_matches_the_rank_formula():
    """``dim V = N - 1 - min(d_src, N - 1)`` free directions per output.

    Centring is what costs the extra one: it puts the all-ones vector in
    ``ker(X)``, while ``K 1 = 0`` simultaneously bars it from ``range(K)``,
    so that direction is available to neither.
    """
    rng = np.random.default_rng(SEED)
    n = 40
    ones = np.ones(n) / np.sqrt(n)

    for d_src in (5, 20, 37, 38, 39, 60):
        X = rng.normal(size=(d_src, n))
        X = X - X.mean(axis=1, keepdims=True)  # Step 0 centring
        kernel_basis = null_space(X)
        off_constant = kernel_basis - np.outer(ones, ones @ kernel_basis)
        free = int(
            (np.linalg.svd(off_constant, compute_uv=False) > 1e-10).sum()
        )
        assert free == n - 1 - min(d_src, n - 1)


# ---------------------------------------------------------------------
# Positive semi-definiteness of the Gram matrix
# ---------------------------------------------------------------------


def test_rbf_gram_is_psd_whatever_the_pilots_are():
    """A positive-definite kernel gives a PSD Gram for *any* point set.

    So pilot selection cannot make the Gram indefinite. It can make it
    *singular* -- duplicate pilots repeat a row -- which is a
    conditioning problem, not a definiteness one.
    """
    rng = np.random.default_rng(SEED)
    clouds = {
        'gaussian': rng.normal(size=(120, 30)),
        'one tight cluster': rng.normal(scale=1e-3, size=(120, 30)),
        'two far clusters': np.vstack(
            [rng.normal(size=(60, 30)), rng.normal(size=(60, 30)) + 50.0]
        ),
        'collinear': np.outer(np.linspace(-1, 1, 120), rng.normal(size=30)),
    }
    for name, Z in clouds.items():
        gram = Kernel('rbf').fit(Z, seed=SEED)(Z, Z)
        smallest = np.linalg.eigvalsh(0.5 * (gram + gram.T)).min()
        assert smallest > -1e-8 * max(np.abs(gram).max(), 1.0), name


def test_duplicate_pilots_make_the_gram_singular_not_indefinite():
    rng = np.random.default_rng(SEED)
    Z = rng.normal(size=(60, 20))
    doubled = np.vstack([Z, Z])

    gram = Kernel('rbf').fit(doubled, seed=SEED)(doubled, doubled)
    eigenvalues = np.linalg.eigvalsh(0.5 * (gram + gram.T))
    assert eigenvalues.min() > -1e-8 * eigenvalues.max()  # still PSD
    assert eigenvalues.min() < 1e-8 * eigenvalues.max()  # but singular


def test_polynomial_kernel_refuses_a_negative_offset():
    """``(<x,y> + c)^d`` is only positive definite for ``c >= 0``."""
    with pytest.raises(ValueError, match='coef0 >= 0'):
        Kernel('polynomial', coef0=-1.0)


def test_gram_conditioning_is_reported():
    X, Y = paired_data(n=300, d_src=10)
    summary = RKHSAligner(lam=1e-3, lam_grid=None).fit(X, Y).summary()

    # Centring always leaves the constant direction at ~0, so the minimum
    # eigenvalue is machine noise of either sign -- but never a real
    # negative eigenvalue.
    assert abs(summary['rkhs_gram_min_eig']) < 1e-6
    assert summary['rkhs_gram_cond'] > 0


def test_capacity_diagnostic_reports_the_usable_dimension():
    """``rank(K) - rank(X)`` is the residual's degrees of freedom."""
    X, Y = constrained_pair(300, d_src=20, d_tgt=15)
    summary = RKHSAligner(lam=1e-3, lam_grid=None).fit(X, Y).summary()

    assert summary['rkhs_gram_rank'] == 299  # N - 1 after centring
    assert summary['rkhs_capacity'] == 299 - 20
    # Conditioning must ignore the null direction centring always creates,
    # or every run would report ~1e15 regardless of the kernel.
    assert summary['rkhs_gram_cond'] < 1e6


@pytest.mark.parametrize('kernel', ['linear', 'cosine'])
def test_a_linear_kernel_has_no_capacity_by_construction(kernel):
    """Its RKHS *is* the linear functions the constraint removes.

    ``rank(K) <= d_src`` for a linear or cosine kernel, so the residual
    has ``rank(K) - rank(X) <= 0`` free directions however many pilots
    are supplied -- these two kernels can never do anything in RKA.
    """
    X, Y = constrained_pair(300, d_src=20, d_tgt=15)
    aligner = RKHSAligner(kernel=kernel, lam=1e-3, lam_grid=None).fit(X, Y)

    assert aligner.summary()['rkhs_capacity'] <= 0
    assert residual_strength(aligner) < 1e-5
    rigid = ProcrustesAligner().fit(X, Y).transform(X)
    assert (
        np.linalg.norm(aligner.transform(X) - rigid) / np.linalg.norm(rigid)
        < 1e-6
    )


def test_capacity_warning_fires_when_the_residual_is_annihilated(caplog):
    X, Y = constrained_pair(25, d_src=40, d_tgt=15)
    with caplog.at_level('WARNING', logger='src.alignment.rkhs'):
        RKHSAligner(lam=1e-3, lam_grid=None).fit(X, Y)
    assert 'no degrees of freedom' in caplog.text


def test_context_data_estimates_the_standardisation():
    """Whitening needs no pairing, so it need not be limited to pilots.

    Each device can standardise its own space from everything it holds
    locally; only the paired samples cost airtime. Restricting the
    whitening to a handful of pilots is what makes the low-budget regime
    look far worse than it is.
    """
    X, Y = paired_data(n=600, d_src=12, d_tgt=12)
    pilots = slice(0, 15)

    from_pilots = ProcrustesAligner().fit(X[pilots], Y[pilots])
    from_local = ProcrustesAligner().fit(
        X[pilots], Y[pilots], src_context=X, tgt_context=Y
    )

    # The alignment still uses only the pilots ...
    assert from_local.n_calibration_ == 15
    # ... but the standardisation is estimated from everything.
    assert not np.allclose(
        from_pilots.scaler_src_.forward_, from_local.scaler_src_.forward_
    )
    # and it is a better estimate: closer to the transform the full data
    # would give.
    reference = LatentScaler('whiten').fit(X).forward_
    assert np.linalg.norm(
        from_local.scaler_src_.forward_ - reference
    ) < np.linalg.norm(from_pilots.scaler_src_.forward_ - reference)


def test_context_defaults_to_the_pilots():
    X, Y = paired_data(n=200, d_src=8, d_tgt=8)
    with_default = ProcrustesAligner().fit(X, Y)
    explicit = ProcrustesAligner().fit(X, Y, src_context=X, tgt_context=Y)
    assert np.allclose(
        with_default.transform(X), explicit.transform(X), atol=1e-12
    )


# ---------------------------------------------------------------------
# Nested stratified (round-robin) selection
# ---------------------------------------------------------------------


def test_round_robin_prefixes_are_class_balanced():
    """Every prefix should be as balanced as its length allows."""
    X, y = clustered_pool()
    for n in (9, 30, 60, 150):
        idx = select_pilots(X, n, strategy='round_robin', labels=y, seed=3)
        counts = np.bincount(y[idx], minlength=3)
        assert idx.size == n
        assert counts.max() - counts.min() <= 1


def test_round_robin_is_nested_across_budgets():
    """Larger budgets must contain the smaller ones, at a fixed seed."""
    X, y = clustered_pool()
    budgets = [10, 25, 50, 120]
    chosen = {
        n: set(
            select_pilots(
                X, n, strategy='round_robin', labels=y, seed=5
            ).tolist()
        )
        for n in budgets
    }
    for smaller, larger in pairwise(budgets):
        assert chosen[smaller] < chosen[larger]


def test_round_robin_varies_with_the_seed_and_needs_labels():
    X, y = clustered_pool()
    a = select_pilots(X, 40, strategy='round_robin', labels=y, seed=1)
    b = select_pilots(X, 40, strategy='round_robin', labels=y, seed=2)
    assert not np.array_equal(a, b)
    with pytest.raises(ValueError, match='requires per-sample labels'):
        select_pilots(X, 40, strategy='round_robin')


def test_round_robin_beats_uniform_at_class_balance():
    """Its whole point: uniform sampling under-represents rare classes."""
    X, y = clustered_pool()  # 700 / 200 / 100
    balanced = np.bincount(
        y[select_pilots(X, 30, strategy='round_robin', labels=y, seed=0)],
        minlength=3,
    )
    uniform = np.bincount(
        y[select_pilots(X, 30, strategy='random', seed=0)], minlength=3
    )
    assert balanced.min() >= 10  # ten per class
    assert uniform.min() < balanced.min()


def test_scheduled_bandwidth_grows_with_the_budget():
    from src.alignment import scheduled_bandwidth

    values = [scheduled_bandwidth(n, 500) for n in (20, 100, 240)]
    assert values == sorted(values)
    assert scheduled_bandwidth(500, 500) == pytest.approx(3.9)
    with pytest.raises(ValueError, match='n_pool must be positive'):
        scheduled_bandwidth(10, 0)


@pytest.mark.parametrize('method', ['pca', 'pga'])
def test_unshrunk_factorisation_never_forms_the_left_factor(method):
    """The SVD path must not materialise an ``n x n`` matrix.

    Whitening statistics are estimated from all local data rather than
    from the pilots (``Aligner.fit``), so ``n`` here is the whole latent
    bank -- tens of thousands of rows. Asking for the full left factor
    costs ``n^2`` doubles (20 GB on a 50k bank) and it is discarded
    unread on the next line; only the right singular vectors are ever
    used. This asserts the shape contract directly, because the cost is
    invisible in the returned values: the two routes agree exactly, and
    the only symptom of a regression is a run that takes twenty times
    longer.
    """
    rng = np.random.default_rng(SEED)
    n, d = 600, 40
    X = rng.normal(size=(n, d)) @ np.diag(np.linspace(10.0, 0.1, d))

    seen = []
    real_svd = np.linalg.svd

    def spy(a, *args, **kwargs):
        out = real_svd(a, *args, **kwargs)
        seen.append(np.shape(a))
        return out

    with mock.patch.object(np.linalg, 'svd', spy):
        scaler = LatentScaler(method, n_components=5).fit(X)

    assert seen, 'the unshrunk path should factorise something'
    # Every factorisation is of a matrix no larger than d x d: the QR
    # reduction happens first, so nothing of size n reaches the SVD.
    assert all(max(shape) <= d for shape in seen), seen
    assert scaler.out_dim in (5, 6)  # 'pga' carries the radius too


def test_canonical_svd_falls_back_when_the_fast_driver_fails():
    """A singular cross-covariance must not abort the fit.

    numpy's ``gesdd`` does not always converge on a numerically singular
    matrix, which a kernel feature map reliably produces -- its effective
    rank is far below its column count. Measured on a 1980-column map
    the spectrum spanned 3e-18 to 1, and the fast driver raised where
    ``gesvd`` returned. The fallback has to yield a genuine
    factorisation, not merely avoid the exception.
    """
    rng = np.random.default_rng(SEED)
    d, k = 40, 6
    M = rng.normal(size=(d, k)) @ rng.normal(size=(k, d))  # rank k << d

    with mock.patch.object(
        np.linalg, 'svd', side_effect=np.linalg.LinAlgError('did not converge')
    ):
        U, rho, Vt = _robust_svd(M)

    assert np.allclose(U * rho @ Vt, M, atol=1e-8)
    assert np.all(np.diff(rho) <= 1e-12)  # descending
    assert np.allclose(U.T @ U, np.eye(U.shape[1]), atol=1e-8)
    assert np.allclose(Vt @ Vt.T, np.eye(Vt.shape[0]), atol=1e-8)


def test_canonical_svd_reports_non_finite_input_as_itself():
    """A NaN is an upstream bug, not a conditioning problem.

    Retrying it with a slower driver would only fail again, and more
    slowly; the error should name the real cause.
    """
    M = np.eye(5)
    M[2, 3] = np.nan
    with pytest.raises(np.linalg.LinAlgError, match='non-finite'):
        _robust_svd(M)

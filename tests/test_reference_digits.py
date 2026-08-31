"""Cross-check against a standalone reference implementation.

The strongest available correctness check on :class:`RKHSAligner`: build
the RKA paper's own experimental setting from scratch -- two independently
trained MLPs on sklearn Digits, 12- and 16-dimensional hidden layers -- and
require the packaged aligner to reproduce what a direct transcription of
Algorithm 1 produces on the same latents.

It is also the control for the SEMASIA results. RKA shows a clear gain
here and none at all on full-rank 768-dimensional ViT latents; this test
pins down that the difference is the regime, not the implementation.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from src.alignment import ProcrustesAligner, RKHSAligner
from src.alignment.preprocessing import LatentScaler

LAMBDAS = np.logspace(-6, 3, 30)


def hidden_activation(model: MLPClassifier, X: np.ndarray) -> np.ndarray:
    """First-layer activations, i.e. the agent's latent space."""
    A = X @ model.coefs_[0] + model.intercepts_[0]
    return np.tanh(A) if model.activation == 'tanh' else np.maximum(A, 0.0)


@pytest.fixture(scope='module')
def digits_agents() -> dict[str, np.ndarray]:
    """Two independently trained agents on Digits (d_t=12, d_r=16)."""
    digits = load_digits()
    X_pix = digits.data.astype(float) / 16.0
    X_model, X_tmp, y_model, y_tmp = train_test_split(
        X_pix,
        digits.target,
        test_size=0.40,
        stratify=digits.target,
        random_state=1,
    )
    X_cal, X_test, _, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=2
    )
    tx = MLPClassifier(
        hidden_layer_sizes=(12,),
        activation='tanh',
        solver='lbfgs',
        alpha=1e-3,
        max_iter=600,
        random_state=10,
    ).fit(X_model, y_model)
    rx = MLPClassifier(
        hidden_layer_sizes=(16,),
        activation='relu',
        solver='lbfgs',
        alpha=3e-3,
        max_iter=600,
        random_state=20,
    ).fit(X_model, y_model)

    return {
        'Zt_cal': hidden_activation(tx, X_cal),
        'Zr_cal': hidden_activation(rx, X_cal),
        'Zt_test': hidden_activation(tx, X_test),
        'Zr_test': hidden_activation(rx, X_test),
        'Zr_model': hidden_activation(rx, X_model),
        'y_model': y_model,
        'y_test': y_test,
    }


def reference_rka(Zt_cal, Zr_cal, Zt_test, lam: float) -> np.ndarray:
    """Algorithm 1 transcribed directly, column-stacked as in the paper."""
    mu_t, Wt = _whitener(Zt_cal)
    mu_r, Wr = _whitener(Zr_cal)
    X = ((Zt_cal - mu_t) @ Wt).T
    Y = ((Zr_cal - mu_r) @ Wr).T
    Xte = ((Zt_test - mu_t) @ Wt).T
    n = X.shape[1]

    U, _, Vt = np.linalg.svd(Y @ X.T, full_matrices=False)
    Q = U @ Vt
    E = Y - Q @ X

    K0 = np.exp(-_sqdist(X.T, X.T) / _bandwidth(X.T))
    ones = np.ones((n, n)) / n
    K = K0 - ones @ K0 - K0 @ ones + ones @ K0 @ ones
    Kte0 = np.exp(-_sqdist(Xte.T, X.T) / _bandwidth(X.T))
    Kte = (
        Kte0 - Kte0.mean(1, keepdims=True) - K0.mean(0, keepdims=True)
    ) + K0.mean()

    evals, evecs = np.linalg.eigh(K)
    evals = np.maximum(evals, 0.0)
    H = (evecs * (evals / (evals + n * lam))) @ evecs.T
    B = X @ H @ X.T + 1e-8 * np.eye(X.shape[0])
    Lam = E @ H @ X.T @ np.linalg.inv(B)
    A = ((E - Lam @ X) @ evecs * (1.0 / (evals + n * lam))) @ evecs.T
    return (Q @ Xte + A @ Kte.T).T


def _whitener(Z, eps=1e-6):
    mu = Z.mean(axis=0)
    Zc = Z - mu
    vals, vecs = np.linalg.eigh(Zc.T @ Zc / len(Z))
    vals = np.maximum(vals, eps)
    return mu, vecs @ np.diag(vals**-0.5) @ vecs.T


def _sqdist(A, B):
    return np.maximum(
        (A**2).sum(1)[:, None] + (B**2).sum(1)[None, :] - 2.0 * A @ B.T, 0.0
    )


def _bandwidth(Z):
    D = _sqdist(Z, Z)
    return np.median(D[D > 0])


def test_matches_the_reference_transcription(digits_agents):
    """Same latents, same lambda: the packaged aligner must agree.

    Compared in the receiver's *raw* space, not its whitened one. Six of
    the sixteen RX ReLU units are dead, so six covariance eigenvalues are
    exactly zero and whitening amplifies them by whatever floor each
    implementation happens to clip at -- 1e-6 absolute in the reference,
    a trace-relative ridge here. Those directions carry no information
    and the inverse transform scales them straight back down, so raw
    space is where the two are actually comparable.

    Shrinkage is off because it is a deliberate departure from the
    reference, not an implementation detail (see
    :class:`~src.alignment.preprocessing.LatentScaler`).
    """
    d = digits_agents
    lam = 5.69e-4

    ours = RKHSAligner(lam=lam, lam_grid=None, shrinkage=None)
    ours.fit(d['Zt_cal'], d['Zr_cal'])
    mine = ours.transform(d['Zt_test'])

    mu_r, W_r = _whitener(d['Zr_cal'])
    theirs = (
        reference_rka(d['Zt_cal'], d['Zr_cal'], d['Zt_test'], lam=lam)
        @ np.linalg.pinv(W_r)
        + mu_r
    )
    relative = np.abs(mine - theirs).max() / np.abs(theirs).max()
    assert relative < 1e-3


def test_reproduces_the_reference_gain_over_procrustes(digits_agents):
    """The paper's headline: an interior lambda beats plain Procrustes."""
    d = digits_agents
    proc = ProcrustesAligner().fit(d['Zt_cal'], d['Zr_cal'])
    scaler = proc.scaler_tgt_
    head = LogisticRegression(max_iter=1500, random_state=30).fit(
        scaler.transform(d['Zr_model']), d['y_model']
    )

    def accuracy(Y_hat, sc):
        return accuracy_score(d['y_test'], head.predict(sc.transform(Y_hat)))

    procrustes_acc = accuracy(proc.transform(d['Zt_test']), scaler)
    scores = []
    for lam in LAMBDAS:
        rkhs = RKHSAligner(lam=float(lam), lam_grid=None).fit(
            d['Zt_cal'], d['Zr_cal']
        )
        scores.append(accuracy(rkhs.transform(d['Zt_test']), rkhs.scaler_tgt_))
    scores = np.asarray(scores)

    # A real gain, at an interior lambda, collapsing back to the rigid
    # solution at the top of the grid (Eq. 17).
    assert scores.max() - procrustes_acc > 0.01
    assert 0 < int(np.argmax(scores)) < len(LAMBDAS) - 1
    assert scores[-1] == pytest.approx(procrustes_acc, abs=1e-9)


def test_flat_gram_spectrum_is_what_kills_the_gain(digits_agents):
    """The Digits latents are low-dimensional enough for lambda to bite.

    Padding them out to 300 dimensions with noise -- the SEMASIA regime --
    concentrates the pairwise distances and flattens the centred Gram
    spectrum, which is precisely when no lambda helps.
    """
    rng = np.random.default_rng(0)
    Zt = digits_agents['Zt_cal']
    padded = np.hstack([Zt, rng.normal(scale=0.1, size=(len(Zt), 288))])

    def spread(Z):
        W = LatentScaler('whiten').fit_transform(Z)
        D = _sqdist(W, W)
        off = D[~np.eye(len(W), dtype=bool)]
        return off.std() / off.mean()

    assert spread(Zt) > 4 * spread(padded)

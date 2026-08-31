"""Metrics for scoring a fitted latent-space alignment.

Three complementary families, all computed on *held-out* points:

- **Reconstruction** -- how close the mapped source latent is to the true
  target latent (``nmse``, ``r2``, ``cosine``). Scale-free, so numbers are
  comparable across encoder pairs with very different latent magnitudes.
- **Retrieval** -- whether the mapped latent lands nearest to its own
  counterpart rather than to somebody else's (``mrr``, ``top1``, ``top5``).
  This is the metric that matters when the receiver does nearest-neighbour
  lookups instead of decoding.
- **Downstream** -- the receiver's own decoder accuracy on the mapped
  latents (``accuracy``), i.e. the semantic-communication end metric.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ..decoder import Decoder

__all__ = [
    'alignment_metrics',
    'decoder_metrics',
    'mean_reciprocal_rank',
    'reconstruction_metrics',
    'retrieval_metrics',
]


def _row_normalize(X: np.ndarray) -> np.ndarray:
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def reconstruction_metrics(
    Y_pred: np.ndarray, Y_true: np.ndarray
) -> dict[str, float]:
    """Point-wise agreement between mapped and true target latents.

    Parameters
    ----------
    Y_pred : np.ndarray, shape (m, d_tgt)
        Source latents mapped into the target space.
    Y_true : np.ndarray, shape (m, d_tgt)
        The receiver's own latents for the same inputs.

    Returns
    -------
    dict[str, float]
        ``mse``, ``nmse`` (energy-normalised), ``r2`` (variance-explained,
        centred) and ``cosine`` (mean row-wise cosine similarity).
    """
    Y_pred = np.asarray(Y_pred, dtype=np.float64)
    Y_true = np.asarray(Y_true, dtype=np.float64)
    if Y_pred.shape != Y_true.shape:
        raise ValueError(
            f'Shape mismatch: predictions {Y_pred.shape} vs targets '
            f'{Y_true.shape}.'
        )

    sq_err = float(np.sum((Y_pred - Y_true) ** 2))
    energy = float(np.sum(Y_true**2))
    variance = float(np.sum((Y_true - Y_true.mean(axis=0)) ** 2))
    cosines = np.sum(_row_normalize(Y_pred) * _row_normalize(Y_true), axis=1)

    return {
        'mse': sq_err / Y_true.size,
        'nmse': sq_err / max(energy, 1e-30),
        'r2': 1.0 - sq_err / max(variance, 1e-30),
        'cosine': float(np.mean(cosines)),
    }


def mean_reciprocal_rank(queries: np.ndarray, candidates: np.ndarray) -> float:
    """Mean Reciprocal Rank of paired retrieval (row ``i`` <-> row ``i``).

    For each query, ranks all candidates by cosine similarity and finds
    the rank of the true match. Ties are broken in the query's favour: a
    candidate strictly more similar than the true match increases the
    rank, equally similar candidates do not.

    Parameters
    ----------
    queries : np.ndarray, shape (m, d)
    candidates : np.ndarray, shape (m, d)
        ``candidates[i]`` is the true match for ``queries[i]``.

    Returns
    -------
    float
        Mean of ``1 / rank_i``, in ``(0, 1]``.
    """
    ranks = _paired_ranks(queries, candidates)
    return float(np.mean(1.0 / ranks))


def _paired_ranks(
    Y_pred: np.ndarray, Y_true: np.ndarray, chunk: int = 1024
) -> np.ndarray:
    """Rank of each row's true counterpart under cosine similarity.

    Computed in row chunks: on a full SEMASIA test split the similarity
    matrix is 10000 x 10000, which is 800 MB held all at once and is
    evaluated once per fitted map during a sweep.
    """
    Q = _row_normalize(np.asarray(Y_pred, dtype=np.float64))
    C = _row_normalize(np.asarray(Y_true, dtype=np.float64))
    if Q.shape != C.shape:
        raise ValueError(
            f'Shape mismatch: predictions {Q.shape} vs targets {C.shape}.'
        )

    ranks = np.empty(Q.shape[0], dtype=np.int64)
    for start in range(0, Q.shape[0], chunk):
        stop = min(start + chunk, Q.shape[0])
        sim = Q[start:stop] @ C.T
        true_sim = sim[np.arange(stop - start), np.arange(start, stop)]
        ranks[start:stop] = 1 + np.sum(sim > true_sim[:, None], axis=1)
    return ranks


def retrieval_metrics(
    Y_pred: np.ndarray, Y_true: np.ndarray, ks: tuple[int, ...] = (1, 5)
) -> dict[str, float]:
    """Nearest-neighbour retrieval of the true counterpart.

    Parameters
    ----------
    Y_pred, Y_true : np.ndarray, shape (m, d_tgt)
        Mapped and true target latents, paired row by row.
    ks : tuple[int, ...], default=(1, 5)
        Cut-offs for top-k accuracy.

    Returns
    -------
    dict[str, float]
        ``mrr`` plus one ``top{k}`` entry per requested ``k``.
    """
    ranks = _paired_ranks(Y_pred, Y_true)
    out = {'mrr': float(np.mean(1.0 / ranks))}
    for k in ks:
        out[f'top{k}'] = float(np.mean(ranks <= k))
    return out


def decoder_metrics(
    decoder: Decoder,
    Y_pred: np.ndarray,
    labels: np.ndarray,
    Y_true: np.ndarray | None = None,
) -> dict[str, float]:
    """Receiver-decoder accuracy on the mapped latents.

    Parameters
    ----------
    decoder : Decoder
        The receiver's private decoder, fitted on *its own* raw latents.
    Y_pred : np.ndarray, shape (m, d_tgt)
        Source latents mapped into the target space.
    labels : np.ndarray, shape (m,)
        Ground-truth labels of those samples.
    Y_true : np.ndarray, optional
        The receiver's own latents, to also report the ceiling accuracy
        and the fraction of it that survives the transport.

    Returns
    -------
    dict[str, float]
        ``accuracy``; plus ``accuracy_oracle`` and ``accuracy_ratio``
        when ``Y_true`` is given.
    """
    out = {'accuracy': float(decoder.score(Y_pred, labels))}
    if Y_true is not None:
        oracle = float(decoder.score(Y_true, labels))
        out['accuracy_oracle'] = oracle
        out['accuracy_ratio'] = out['accuracy'] / max(oracle, 1e-12)
    return out


def alignment_metrics(
    Y_pred: np.ndarray,
    Y_true: np.ndarray,
    decoder: Decoder | None = None,
    labels: np.ndarray | None = None,
    ks: tuple[int, ...] = (1, 5),
) -> dict[str, Any]:
    """All available metrics for one fitted alignment.

    Parameters
    ----------
    Y_pred, Y_true : np.ndarray, shape (m, d_tgt)
        Mapped and true target latents, paired row by row.
    decoder : Decoder, optional
        The receiver's decoder; skipped when ``None``.
    labels : np.ndarray, optional
        Ground-truth labels, required alongside ``decoder``.
    ks : tuple[int, ...], default=(1, 5)
        Retrieval cut-offs.

    Returns
    -------
    dict[str, Any]
    """
    out: dict[str, Any] = {}
    out.update(reconstruction_metrics(Y_pred, Y_true))
    out.update(retrieval_metrics(Y_pred, Y_true, ks=ks))
    if decoder is not None and labels is not None:
        out.update(decoder_metrics(decoder, Y_pred, labels, Y_true=Y_true))
    return out

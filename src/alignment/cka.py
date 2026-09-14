"""CKA-based matching and alignment (Maniparambil et al., CVPR 2024).

"Do Vision and Language Encoders Represent the World Similarly?" asks
whether two encoders that never saw each other agree on the *geometry* of
what they encode, and answers it with Centred Kernel Alignment. Its
observation (their Table 1) is that CKA between a vision and a language
encoder collapses from 0.72 to 0.01 as the pairing is shuffled away, so
CKA is not merely a similarity score -- it is an objective whose maximiser
*is* the correspondence:

    sigma* = argmax_sigma CKA(K_sigma(Z), K_H).                    (Eq. 3)

Two solvers are proposed for it, and both are here: a seeded quadratic
assignment (Sec. 4.1) and a *local* CKA score used for retrieval and for
linear sum assignment (Sec. 4.2). Both are seeded by a **base set**
``B = {(z_i^b, h_i^b)}_{i=1}^M`` of aligned pairs, which is this package's
anchor set under another name -- and it is the only thing the kernels are
ever evaluated against.

What is different about this method
-----------------------------------
Relative representations (:mod:`~src.alignment.relative`) also encode a
query through a shared anchor set, and the paper's Sec. 2 says exactly
what it thinks of that: an *explicit* similarity measure "is sensitive to
the selection of anchors and noise in the original embeddings", so the
better design choice is "an implicit measure that captures the similarity
of similarities". That is the whole distinction. Both methods send the
same object -- a vector of similarities to ``M`` shared anchors -- but

- relative representations compare two such vectors **coordinatewise**,
  so anchor ``i`` on the transmitter must mean the same thing as anchor
  ``i`` on the receiver in an absolute sense;
- CKA compares them **through the base set's own kernel**, after the
  centring and re-scaling of Eq. (2), so only the *pattern* of
  similarities has to agree, not its offset or its scale.

That is why the two agree on the anchors and differ on everything else,
and why this class shares :class:`~src.anchors.Anchor` with them rather
than reimplementing anchor selection.

The anchor kernel map
---------------------
Everything below is built from one object: the row of the centred and
re-scaled kernel of Eq. (4),

    Kbar = HSIC(K, K)^{-1/2} K C,      C = I - 11'/N

restricted to the base-set columns,

    phi(z) = HSIC(G, G)^{-1/2} ( k(z, A) - mean_j k(z, a_j) ),

with ``G = k(A, A)`` the base Gram. The row-centring is the paper's
``C``; the ``HSIC(G, G)^{-1/2}`` factor is what makes the transmitter's
and the receiver's feature vectors commensurate without either agent
seeing the other's data, since it normalises each space by its own kernel
scale. Only the base-set columns are used, so ``phi`` is deployable on a
single new point -- the row mean is taken over the ``M`` anchors and not
over an ``(M+1)``-column augmented row, so that what a query encodes never
depends on which other queries happen to be in the batch.

From a matching rule to a map
-----------------------------
The paper's output is a permutation; this package's contract is a map
``T : R^{d_src} -> R^{d_tgt}`` applicable to samples that have no partner.
Two decodes bridge the gap, and they differ in what the receiver has to
believe:

``decode='ridge'`` (default)
    A ridge readout ``R : phi -> z_tgt``. ``readout='source'`` regresses
    ``phi_src(Z_src)`` on ``Z_tgt`` over the whole pilot set -- a paired
    fit, and the default because it is the stronger map. ``readout=
    'target'`` instead fits ``phi_tgt(Z)`` against ``Z`` on the receiver's
    *own latents alone* and applies it to the transmitter's ``phi_src``,
    which costs no airtime beyond the ``M`` anchor pairs and keeps the
    method training-free in the paper's sense. It is as good as the paired
    fit exactly when the two anchor feature spaces coincide, which
    ``cka_feature_mismatch`` measures directly.
``decode='match'``
    The paper's transport, literally. Score the query against a support
    set of the receiver's own latents with the local CKA of Eq. (6) and
    return the best-scoring support latent, or a softmax barycentre of
    them (``match_temperature``). Nothing is fitted; the output is always
    a real target latent, which is what makes it a *retrieval* method
    rather than a regression.

:meth:`CKAAligner.match` exposes the paper's own task unchanged -- given
``n`` source and ``n`` shuffled target latents, recover the permutation --
through either solver, and :meth:`CKAAligner.local_cka` returns the score
matrix Eq. (6) defines.

Local CKA in closed form
------------------------
Eq. (6) reads as ``p x q`` separate global CKAs of an ``(M+1) x (M+1)``
Gram pair, which is ``O(p q M^2)`` and unusable. It is not needed. With
``K = [[G, u], [u', s]]`` the augmented source Gram and ``L`` the target
one, the identity

    tr(KCLC) = tr(KL) - (2/N)(K1)'(L1) + (1/N^2)(1'K1)(1'L1)

expands into terms that are either constants of the base set, separable
per side, or the single bilinear ``U V'``. So the whole score matrix is
one ``(p, M) x (M, q)`` product, ``O(p q M)``, and is *exact* -- no
approximation of Eq. (6) is involved. See :func:`_local_cka`.

Stretching and clustering (Sec. 4.3)
------------------------------------
The paper's two preprocessing choices are both defaults here, and its
Table 6 measures what each is worth on COCO caption matching:

=========================================  =====  ==========
 configuration                               QAP   local CKA
=========================================  =====  ==========
 neither                                    48.8       48.5
 stretching only                            57.3       56.7
 clustering only                            56.2       55.1
 both                                       65.5       63.3
=========================================  =====  ==========

*Stretching* is ``S = diag(1/std(x_l))``, applied to both spaces before
the kernels are formed -- which is precisely this package's
``preprocess='standard'``, hence the default. *Clustering* is the base-set
design: "simple k-means clustering on the image embeddings works best",
taking "one closest sample to each of the S cluster centers", which is
``strategy='kmeans'`` with ``medoids=True``. Both are worth more than any
other knob in this file.
"""

from __future__ import annotations

import logging
from typing import Literal, NamedTuple

import numpy as np
from scipy.optimize import linear_sum_assignment, quadratic_assignment

from ..anchors import Anchor, AnchorStrategy
from ..kernels import Kernel, KernelName
from .base import Aligner

log = logging.getLogger(__name__)

__all__ = ['CKAAligner', 'cka_score', 'hsic']

Decode = Literal['ridge', 'match']
Readout = Literal['source', 'target']
Matcher = Literal['lsa', 'qap']

# The paper fixes its base set at 320 samples and its query set at 500,
# and every number it reports is at that size.
_BASE_SET_SIZE = 320

# Queries scored against each other for the self-matching diagnostic. The
# score matrix is dense and the assignment is O(q^3), so this is capped
# well below a realistic pilot budget.
_DIAGNOSTIC_QUERIES = 256

# Kernels with k(z, z) == 1, for which the augmented Gram's diagonal
# entry is known without seeing the point (see :meth:`CKAAligner.receive`).
_SELF_NORMALISED: tuple[str, ...] = ('rbf', 'laplacian', 'cosine')


def _double_centre(K: np.ndarray) -> np.ndarray:
    """``C K C`` with ``C = I - 11'/N``, the centring matrix of Eq. (2)."""
    row = K.mean(axis=1, keepdims=True)
    col = K.mean(axis=0, keepdims=True)
    return K - row - col + K.mean()


def hsic(K: np.ndarray, L: np.ndarray) -> float:
    """Hilbert-Schmidt Independence Criterion, Eq. (2).

    ``tr(K C L C) / (N - 1)^2``, evaluated as ``sum(CKC * CLC)`` because
    ``C`` is idempotent and both Grams are symmetric -- which is the same
    number without forming a matrix product.

    Parameters
    ----------
    K, L : np.ndarray, shape (N, N)
        Gram matrices over the *same* ``N`` samples, in the same order.

    Returns
    -------
    float
    """
    if K.shape != L.shape or K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(
            f'hsic expects two square Grams of equal size, got {K.shape} '
            f'and {L.shape}.'
        )
    n = K.shape[0]
    if n < 2:
        return 0.0
    return float(np.sum(_double_centre(K) * _double_centre(L))) / (n - 1) ** 2


def cka_score(K: np.ndarray, L: np.ndarray) -> float:
    """Centred Kernel Alignment of two Grams, Eq. (1).

    ``HSIC(K, L) / sqrt(HSIC(K, K) HSIC(L, L))``, in ``[0, 1]`` for
    positive-definite kernels. This is the paper's headline statistic: on
    5k COCO image-caption pairs it reads 0.72 on the true ordering and
    0.01 once the pairing is shuffled out (their Table 1), which is what
    licenses treating it as a matching objective.

    Parameters
    ----------
    K, L : np.ndarray, shape (N, N)
        Gram matrices over the same ``N`` samples, in the same order.

    Returns
    -------
    float
    """
    denominator = np.sqrt(hsic(K, K) * hsic(L, L))
    if denominator <= 0:
        return 0.0
    return float(hsic(K, L) / denominator)


def _rescaled_kernel(K: np.ndarray) -> np.ndarray:
    """``Kbar = HSIC(K, K)^{-1/2} K C`` of Eq. (4).

    Half-centred on purpose: the QAP objective of Eq. (5) contracts two
    of these against each other and a permutation commutes with ``C``, so
    the two half-centrings compose into the double-centring that HSIC
    asks for. Centring both here would centre twice.
    """
    n = K.shape[0]
    scale = max(n - 1, 1) / max(
        float(np.linalg.norm(_double_centre(K))), 1e-30
    )
    return scale * (K - K.mean(axis=1, keepdims=True))


def _self_kernel(kernel: Kernel, Z: np.ndarray) -> np.ndarray:
    """``k(z, z)`` per row -- the augmented Gram's new diagonal entry.

    Computed in closed form per family rather than as the diagonal of
    ``k(Z, Z)``, which would cost an ``O(p^2)`` Gram to read ``p``
    numbers off.
    """
    Z = np.asarray(Z, dtype=np.float64)
    match kernel.name:
        case 'rbf' | 'laplacian' | 'cosine':
            return np.ones(Z.shape[0])
        case 'linear':
            return np.einsum('ij,ij->i', Z, Z)
        case 'polynomial':
            return (
                np.einsum('ij,ij->i', Z, Z) + kernel.coef0
            ) ** kernel.degree
    raise ValueError(  # pragma: no cover
        f'Unknown kernel {kernel.name!r}.'
    )


class _Augmented(NamedTuple):
    """Per-query pieces of the augmented Gram ``[[G, u], [u', s]]``.

    Everything Eq. (6) needs from one side, for a whole block of queries
    at once and with the base set already contracted out.
    """

    rows: np.ndarray  # (p, M) -- u', the kernel against the base set
    diagonal: np.ndarray  # (p,)  -- s = k(z, z)
    row_sum: np.ndarray  # (p,)  -- sigma_u = 1'u
    total: np.ndarray  # (p,)  -- 1'K1 of the augmented Gram
    hsic: np.ndarray  # (p,)  -- HSIC(K, K), up to the (N-1)^-2 factor


class _AnchorKernel:
    """One space's kernel, evaluated only against the shared base set.

    Holds ``kappa``, the ``M`` anchors, the base Gram ``G = kappa(A, A)``
    and the four scalars every augmented score reuses. This is the object
    that makes the method anchor-based: no ``n x n`` Gram is ever formed,
    on either side, at fit time or at deployment.
    """

    def __init__(self, kernel: Kernel, anchors: np.ndarray) -> None:
        self.kernel = kernel
        self.anchors = np.asarray(anchors, dtype=np.float64)
        G = kernel(self.anchors, self.anchors)
        self.gram = G
        self.row_sum = G.sum(axis=1)  # G 1
        self.total = float(G.sum())  # 1' G 1
        self.sq_total = float(np.sum(G * G))  # tr(G G)
        # HSIC(G, G)^{-1/2}, the per-space scale of Kbar. It is the only
        # quantity that has to be shared in spirit between the two
        # agents, and each computes its own from local data.
        self.scale = max(self.n_anchors - 1, 1) / max(
            float(np.linalg.norm(_double_centre(G))), 1e-30
        )

    @property
    def n_anchors(self) -> int:
        """Base-set size ``M``."""
        return int(self.anchors.shape[0])

    @property
    def n_parameters(self) -> int:
        """Numbers this side must carry to evaluate its feature map."""
        return int(self.anchors.size)

    def rows(self, Z: np.ndarray) -> np.ndarray:
        """Raw kernel rows ``k(Z, A)``, shape ``(p, M)``.

        This is what crosses the channel: the centring and the scaling
        below are deterministic given the base Gram, which the receiver
        holds, so there is no point spending airtime on them.
        """
        return self.kernel(np.asarray(Z, dtype=np.float64), self.anchors)

    def centre(self, U: np.ndarray) -> np.ndarray:
        """``Kbar`` rows from raw ones: row-centre, then re-scale."""
        return self.scale * (U - U.mean(axis=1, keepdims=True))

    def features(self, Z: np.ndarray) -> np.ndarray:
        """The deployed anchor feature map ``phi(Z)``, shape ``(p, M)``."""
        return self.centre(self.rows(Z))

    def augment(self, Z: np.ndarray) -> _Augmented:
        """Augmented-Gram pieces for a block of queries."""
        return self.augment_rows(self.rows(Z), _self_kernel(self.kernel, Z))

    def augment_rows(self, U: np.ndarray, s: np.ndarray) -> _Augmented:
        """Same, from an already-evaluated kernel row (see :meth:`rows`).

        ``HSIC(K, K)`` is expanded exactly as the numerator is, so the
        constants of the base set (``tr(GG)``, ``G1``, ``1'G1``) are
        contracted once and reused for every query.
        """
        n = self.n_anchors + 1
        row_sum = U.sum(axis=1)
        total = self.total + 2.0 * row_sum + s
        trace = self.sq_total + 2.0 * np.einsum('ij,ij->i', U, U) + s**2
        norm = (
            np.sum((self.row_sum[None, :] + U) ** 2, axis=1)
            + (row_sum + s) ** 2
        )
        hsic_self = trace - (2.0 / n) * norm + total**2 / n**2
        return _Augmented(
            rows=U,
            diagonal=s,
            row_sum=row_sum,
            total=total,
            hsic=np.maximum(hsic_self, 1e-30),
        )


def _local_cka(
    base_src: _AnchorKernel,
    base_tgt: _AnchorKernel,
    left: _Augmented,
    right: _Augmented,
) -> np.ndarray:
    """Local CKA of every query pair, Eq. (6), in one matrix product.

    Expands ``tr(KCLC) = tr(KL) - (2/N)(K1)'(L1) + (1/N^2)(1'K1)(1'L1)``
    on the augmented Grams ``K = [[G, u], [u', s]]`` and
    ``L = [[H, v], [v', t]]``:

    ==============  ==================================================
     term            expansion
    ==============  ==================================================
     ``tr(KL)``      ``tr(GH) + 2 u'v + s t``
     ``(K1)'(L1)``   ``(G1)'(H1) + (G1)'v + u'(H1) + u'v``
                     ``+ (sigma_u + s)(sigma_v + t)``
     ``1'K1``        ``1'G1 + 2 sigma_u + s``
    ==============  ==================================================

    Every term is a base-set constant, separable per side, or the single
    bilinear ``u'v`` -- so the whole ``(p, q)`` matrix costs one
    ``(p, M) x (M, q)`` product. The ``(N - 1)^-2`` of Eq. (2) is common
    to numerator and denominator and is dropped throughout.

    Parameters
    ----------
    base_src, base_tgt : _AnchorKernel
        The two sides' anchor kernels, over the *same* base pairs.
    left, right : _Augmented
        Query blocks of the source and target side.

    Returns
    -------
    np.ndarray, shape (p, q)
        ``localCKA(z_i^q, h_j^q)``.
    """
    n = base_src.n_anchors + 1
    cross = left.rows @ right.rows.T  # (p, q)

    trace = (
        float(np.sum(base_src.gram * base_tgt.gram))
        + 2.0 * cross
        + np.outer(left.diagonal, right.diagonal)
    )
    ones = (
        float(base_src.row_sum @ base_tgt.row_sum)
        + (right.rows @ base_src.row_sum)[None, :]
        + (left.rows @ base_tgt.row_sum)[:, None]
        + cross
        + np.outer(
            left.row_sum + left.diagonal, right.row_sum + right.diagonal
        )
    )
    numerator = (
        trace - (2.0 / n) * ones + np.outer(left.total, right.total) / n**2
    )
    return numerator / np.sqrt(np.outer(left.hsic, right.hsic))


def _ridge_readout(F: np.ndarray, Y: np.ndarray, alpha: float) -> np.ndarray:
    """Ridge least squares ``argmin_R ||F R - Y||^2``, both centred.

    The ridge is relative to the mean eigenvalue of ``F'F``, so one
    setting behaves the same whatever scale the anchor features came out
    at -- and they are scaled by ``HSIC(G, G)^{-1/2}``, which depends on
    the kernel and the base set.
    """
    gram = F.T @ F
    ridge = alpha * max(float(np.trace(gram)) / gram.shape[0], 1e-30)
    gram[np.diag_indices_from(gram)] += ridge
    return np.linalg.solve(gram, F.T @ Y)


def _barycentre(
    scores: np.ndarray, support: np.ndarray, temperature: float
) -> np.ndarray:
    """Transport a query onto its support set by its local CKA scores.

    ``temperature <= 0`` is the paper's hard retrieval -- return the
    best-scoring support latent. Anything larger is a softmax barycentre
    of the support, which is what turns a retrieval rule into a map that
    can land between two support points.

    The scores are standardised *per query* before the softmax. They have
    to be: the base set contributes ``M`` of the ``M + 1`` points in both
    augmented Grams, so every candidate scores within a hair of every
    other and an absolute temperature would be uninterpretable. In these
    units the parameter reads as "how many standard deviations of score
    separate the candidates I am willing to mix".
    """
    if temperature <= 0:
        return support[np.argmax(scores, axis=1)]
    centred = scores - scores.mean(axis=1, keepdims=True)
    spread = np.maximum(scores.std(axis=1, keepdims=True), 1e-30)
    logits = centred / (spread * temperature)
    weights = np.exp(logits - logits.max(axis=1, keepdims=True))
    weights /= weights.sum(axis=1, keepdims=True)
    return weights @ support


class CKAAligner(Aligner):
    """CKA-based alignment on a shared base set of anchors.

    Each agent evaluates its own kernel against the ``M`` shared anchors
    and works with the centred, re-scaled rows of Eq. (4). What crosses
    the channel is that ``M``-vector; what the receiver does with it is
    either a ridge readout into its own latents or the paper's local-CKA
    retrieval against a support set of its own samples.

    Parameters
    ----------
    n_anchors : int | None, default=None
        Base-set size ``M``. ``None`` uses the paper's 320, capped at the
        pilot budget -- every number in the paper is measured at that
        size, with a 500-sample query set. The base set is the *only*
        thing both agents have to embed under the default decode, so this
        is the airtime cost of the method (see
        :attr:`paired_samples_used`) as well as its rate.
    strategy : AnchorStrategy, default='kmeans'
        Anchor selection (see :class:`src.anchors.Anchor`). Must be
        index-based: the paper's base set is a set of aligned *sample*
        pairs, so the receiver has to be able to embed the very same
        inputs. Centroid anchors that exist only in the source space
        cannot be transferred -- for those,
        :class:`~src.alignment.relative.RelativeRepresentationAligner`'s
        ``anchor_transfer='clusters'`` is the construction to use.
    medoids : bool, default=True
        For ``strategy='kmeans'``, snap each centroid to its nearest real
        sample. This is exactly the paper's base-set design: cluster the
        source embeddings and take "one closest sample to each of the S
        cluster centers". Worth +8.2 QAP points in their Table 6.
    kernel : {'rbf', 'laplacian', 'polynomial', 'linear', 'cosine'}
        Family of the source kernel ``k``. The paper uses kernel CKA with
        an RBF kernel, which is the default.
    kernel_tgt : str, optional
        Family of the target kernel ``l``. ``None`` reuses ``kernel``.
        The two are separate kernels in Eq. (1) and their bandwidths are
        fitted per space regardless -- CKA is invariant to each side's
        kernel scale by construction, which is what lets two unrelated
        encoders be compared at all.
    gamma : float, optional
        Inverse bandwidth of both kernels. ``None`` fits one per space by
        the median heuristic, on that agent's *local* cloud.
    bandwidth_scale : float, default=1.0
        Multiplier on the median heuristic (``> 1`` smooths).
    degree, coef0 : int, float
        Polynomial-kernel parameters.
    decode : {'ridge', 'match'}, default='ridge'
        How the anchor features become a target latent. ``'ridge'``
        fits a linear readout; ``'match'`` is the paper's retrieval,
        transporting each query onto a support set of real target
        latents by local CKA. See the module docstring.
    readout : {'source', 'target'}, default='source'
        Which side's anchor features the ridge readout is fitted on.
        ``'source'`` regresses the transmitter's features on the
        receiver's latents over the whole pilot set -- a genuine paired
        fit, and the stronger map. ``'target'`` is fitted by the receiver
        on its own latents alone, so it costs no airtime past the anchors
        and keeps the method zero-shot in the paper's sense; whether it
        can afford to is exactly what ``cka_feature_mismatch`` reports.
        Measured on a smooth synthetic pair at 128 anchors, held-out
        NMSE, as the two spaces are pulled apart:

        =====================  ======  ======  ======
         feature mismatch       0.16    0.27    0.60
        =====================  ======  ======  ======
         ``readout='target'``   0.069   0.119   0.661
         ``readout='source'``   0.062   0.069   0.101
        =====================  ======  ======  ======

        Near an isometry the two are indistinguishable and ``'target'``
        is free; past a mismatch of ~0.3 only the paired fit survives.
        Ignored by ``decode='match'``.
    decode_alpha : float, default=1e-2
        Ridge of that readout, relative to the mean eigenvalue of the
        feature Gram.
    match_temperature : float, default=0.25
        Softmax temperature of ``decode='match'``, in standard deviations
        of the per-query score spread. ``0`` recovers the paper's hard
        retrieval, which returns an actual target latent and is what the
        matching accuracy is measured with; anything larger mixes the top
        candidates, which is what a *map* wants -- on the same fixture
        the hard rule scores 0.50 held-out NMSE against 0.25 at this
        default, because a piecewise-constant map cannot land between two
        support points. Above ~1 the barycentre flattens towards the
        support mean and it degrades again (0.55).
    max_points : int | None, default=2000
        Cap on the local cloud each side uses for its bandwidth, its
        readout regression and its matching support set. None of these
        need pairing, so they are drawn from ``src_context`` /
        ``tgt_context`` when those are given.
    preprocess : ScalingMethod, default='standard'
        Per-space standardisation. This is the paper's *stretching*
        matrix ``S = diag(1/std(x_l))`` (Sec. 4.3), and it is not
        optional in spirit: dropping it costs 8.5 QAP points in their
        Table 6, more than any other single choice here.
    eps : float, default=1e-6
        Relative covariance ridge of the whitening step.
    seed : int, default=42
        Seed of the anchor selection, the bandwidth heuristic, the
        support subsample and the QAP solver.

    Attributes
    ----------
    anchor_ : Anchor
        The fitted base-set selector, over the source cloud.
    base_src_, base_tgt_ : _AnchorKernel
        The two sides' kernels against the shared base set.
    readout_ : np.ndarray | None
        ``(M, d_tgt)`` ridge readout, under ``decode='ridge'``.
    support_ : np.ndarray | None
        Target latents a ``decode='match'`` query is transported onto.
    """

    def __init__(
        self,
        n_anchors: int | None = None,
        strategy: AnchorStrategy = 'kmeans',
        medoids: bool = True,
        kernel: KernelName = 'rbf',
        kernel_tgt: KernelName | None = None,
        gamma: float | None = None,
        bandwidth_scale: float = 1.0,
        degree: int = 3,
        coef0: float = 1.0,
        decode: Decode = 'ridge',
        readout: Readout = 'source',
        decode_alpha: float = 1e-2,
        match_temperature: float = 0.25,
        max_points: int | None = 2000,
        preprocess: str = 'standard',
        eps: float = 1e-6,
        n_components: float | None = None,
        shrinkage: str | float | None = 'auto',
        seed: int = 42,
    ) -> None:
        super().__init__(
            preprocess=preprocess,
            eps=eps,
            n_components=n_components,
            shrinkage=shrinkage,
            seed=seed,
        )
        if decode not in ('ridge', 'match'):
            raise ValueError(
                f"Unknown decode {decode!r}; expected 'ridge' or 'match'."
            )
        if readout not in ('target', 'source'):
            raise ValueError(
                f"Unknown readout {readout!r}; expected 'target' or 'source'."
            )
        if n_anchors is not None and int(n_anchors) < 2:
            raise ValueError(
                f'n_anchors must be >= 2 or None, got {n_anchors}: CKA of a '
                'single-point base set has no spread to align.'
            )

        self.n_anchors = None if n_anchors is None else int(n_anchors)
        self.strategy: AnchorStrategy = strategy
        self.medoids = bool(medoids)
        self.kernel = kernel
        self.kernel_tgt = kernel_tgt
        self.gamma = None if gamma is None else float(gamma)
        self.bandwidth_scale = float(bandwidth_scale)
        self.degree = int(degree)
        self.coef0 = float(coef0)
        self.decode: Decode = decode
        self.readout: Readout = readout
        self.decode_alpha = float(decode_alpha)
        self.match_temperature = float(match_temperature)
        self.max_points = None if max_points is None else int(max_points)

        self.anchor_: Anchor | None = None
        self.base_src_: _AnchorKernel | None = None
        self.base_tgt_: _AnchorKernel | None = None
        self.mean_tgt_: np.ndarray | None = None
        self.readout_: np.ndarray | None = None
        self.support_: np.ndarray | None = None
        self._support_parts: _Augmented | None = None

    def hyperparameters(self) -> dict:
        params = super().hyperparameters()
        params.update(
            n_anchors=self.n_anchors,
            strategy=self.strategy,
            medoids=self.medoids,
            kernel=self.kernel,
            kernel_tgt=self.kernel_tgt or self.kernel,
            gamma=self.gamma,
            bandwidth_scale=self.bandwidth_scale,
            decode=self.decode,
            readout=self.readout,
            decode_alpha=self.decode_alpha,
            match_temperature=self.match_temperature,
            max_points=self.max_points,
        )
        if 'polynomial' in (self.kernel, self.kernel_tgt):
            params.update(degree=self.degree, coef0=self.coef0)
        return params

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def _fit(
        self,
        Z_src: np.ndarray,
        Z_tgt: np.ndarray,
        labels: np.ndarray | None = None,
    ) -> None:
        n = Z_src.shape[0]
        rng = np.random.default_rng(self.seed)

        # The base set B: aligned pairs of *real* samples, since the
        # receiver has to embed the same inputs the transmitter chose.
        indices = self._select_base_set(Z_src, n, labels)
        A_src, A_tgt = Z_src[indices], Z_tgt[indices]

        # Kernels are single-space objects and so cost no airtime: each
        # agent fits its own bandwidth on its own local cloud, which is
        # the unpaired context when there is one. CKA is invariant to
        # each side's kernel scale, so the two never have to agree.
        pool_src = self._pool(
            Z_src if self._context_src is None else self._context_src, rng
        )
        pool_tgt = self._pool(
            Z_tgt if self._context_tgt is None else self._context_tgt, rng
        )
        self.base_src_ = _AnchorKernel(
            self._make_kernel(self.kernel, pool_src), A_src
        )
        self.base_tgt_ = _AnchorKernel(
            self._make_kernel(self.kernel_tgt or self.kernel, pool_tgt), A_tgt
        )

        Phi_src = self.base_src_.features(Z_src)
        Phi_tgt = self.base_tgt_.features(Z_tgt)
        if self.decode == 'ridge':
            self._fit_readout(Phi_src, Z_tgt, pool_tgt)
        else:
            self.support_ = pool_tgt
            self._support_parts = self.base_tgt_.augment(pool_tgt)

        self._record_diagnostics(Z_src, Z_tgt, Phi_src, Phi_tgt, indices)

    def _select_base_set(
        self, Z_src: np.ndarray, n: int, labels: np.ndarray | None
    ) -> np.ndarray:
        """Choose the ``M`` anchor pairs, and return their row indices."""
        n_anchors = (
            min(n, _BASE_SET_SIZE)
            if self.n_anchors is None
            else self.n_anchors
        )
        if n_anchors > n:
            log.warning(
                'n_anchors=%d exceeds the %d available calibration samples; '
                'using %d anchors instead.',
                n_anchors,
                n,
                n,
            )
            n_anchors = n

        fit_kwargs: dict = {'n_anchors': n_anchors}
        if self.strategy == 'kmeans':
            fit_kwargs['medoids'] = self.medoids
        if self.strategy in ('stratified', 'round_robin'):
            if labels is None:
                raise ValueError(
                    f'strategy={self.strategy!r} requires per-sample labels; '
                    'pass them to fit(..., labels=...).'
                )
            fit_kwargs['labels'] = labels
        if self.strategy == 'herding':
            # An unfitted spec: `Anchor` fits the bandwidth on the pool it
            # is selecting from, which is what makes the design kernel-
            # aware in the same geometry the base Gram will live in.
            fit_kwargs['kernel'] = Kernel(
                name=self.kernel,
                gamma=self.gamma,
                degree=self.degree,
                coef0=self.coef0,
                bandwidth_scale=self.bandwidth_scale,
            )

        self.anchor_ = Anchor(Z_src, strategy=self.strategy, seed=self.seed)
        self.anchor_.fit(**fit_kwargs)
        indices = self.anchor_.indices
        if indices is None:
            raise ValueError(
                f'CKA needs an index-based base set, but {self.strategy!r}'
                + (' with medoids=False' if self.strategy == 'kmeans' else '')
                + ' produced centroids, which the receiver cannot embed. '
                'Use medoids=True, or an index-based strategy.'
            )
        return np.asarray(indices, dtype=int)

    def _make_kernel(self, name: KernelName, pool: np.ndarray) -> Kernel:
        """One space's kernel, with its bandwidth fitted on local data."""
        kernel = Kernel(
            name=name,
            gamma=self.gamma,
            degree=self.degree,
            coef0=self.coef0,
            bandwidth_scale=self.bandwidth_scale,
        )
        return kernel.fit(pool, seed=self.seed)

    def _pool(self, Z: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Sub-sample a local cloud down to ``max_points``."""
        if self.max_points is None or Z.shape[0] <= self.max_points:
            return Z
        keep = rng.choice(Z.shape[0], size=self.max_points, replace=False)
        return Z[np.sort(keep)]

    def _fit_readout(
        self,
        Phi_src: np.ndarray,
        Z_tgt: np.ndarray,
        pool_tgt: np.ndarray,
    ) -> None:
        """Fit ``R : phi -> z_tgt`` on whichever side ``readout`` names.

        Under ``'source'`` this is a paired regression and is confined to
        the pilot set by definition. Under ``'target'`` it is the
        receiver's alone -- its own anchor features against its own
        latents -- so it runs on the full local pool instead, and the
        pilots are needed only to build the anchors.
        """
        if self.readout == 'target':
            features, targets = self.base_tgt_.features(pool_tgt), pool_tgt
        else:
            features, targets = Phi_src, Z_tgt
        self.mean_tgt_ = targets.mean(axis=0)
        self.readout_ = _ridge_readout(
            features, targets - self.mean_tgt_, self.decode_alpha
        )

    def _record_diagnostics(
        self,
        Z_src: np.ndarray,
        Z_tgt: np.ndarray,
        Phi_src: np.ndarray,
        Phi_tgt: np.ndarray,
        indices: np.ndarray,
    ) -> None:
        energy = max(float(np.sum((Z_tgt - Z_tgt.mean(axis=0)) ** 2)), 1e-30)
        residual = float(np.sum((self._transform(Z_src) - Z_tgt) ** 2))

        self.diagnostics_ = {
            'cka_n_anchors': self.base_src_.n_anchors,
            # Eq. (1) on the two base Grams: the paper's own statistic,
            # and the single number that says whether the two encoders
            # are alignable at all before any decode is attempted.
            'cka_score': cka_score(self.base_src_.gram, self.base_tgt_.gram),
            # The paper's task, measured on held-in queries: local CKA
            # plus linear sum assignment, against the true pairing.
            'cka_matching_accuracy': self._self_matching_accuracy(
                Z_src, Z_tgt, indices
            ),
            'cka_gamma_src': self.base_src_.kernel.summary().get(
                'kernel_gamma'
            ),
            'cka_gamma_tgt': self.base_tgt_.kernel.summary().get(
                'kernel_gamma'
            ),
            # How close the two anchor feature spaces actually are -- the
            # quantity the method assumes to be ~0, and the direct
            # counterpart of `rr_space_mismatch`. Note that CKA needs it
            # to be small only up to the centring and re-scaling that
            # `phi` already applies, which is exactly the difference
            # between this method and relative representations.
            'cka_feature_mismatch': float(
                np.linalg.norm(Phi_src - Phi_tgt)
                / max(np.linalg.norm(Phi_tgt), 1e-12)
            ),
            # The deployed path -- the transmitter's anchor features,
            # decoded -- on the calibration pairs. Under `decode='match'`
            # read it as a floor and not as a score: the support set is
            # the receiver's local pool, which at fit time contains the
            # very targets being reconstructed.
            'cka_transfer_r2': 1.0 - residual / energy,
        }
        if self.decode == 'ridge':
            # The same readout, applied to the *receiver's own* anchor
            # features instead. Whichever side `readout` fitted it on,
            # the gap between the two is the price of the two feature
            # spaces not coinciding -- and it is signed: the in-sample
            # side is the higher one, so under the default it is this
            # number that falls below `cka_transfer_r2`.
            fitted = Phi_tgt @ self.readout_ + self.mean_tgt_
            self.diagnostics_['cka_readout_r2'] = 1.0 - float(
                np.sum((fitted - Z_tgt) ** 2) / energy
            )
        else:
            self.diagnostics_['cka_support_size'] = int(self.support_.shape[0])

    def _self_matching_accuracy(
        self, Z_src: np.ndarray, Z_tgt: np.ndarray, indices: np.ndarray
    ) -> float:
        """Local-CKA matching accuracy on the non-anchor pilots.

        The anchors are excluded because they sit in the base set of
        every augmented Gram, which makes them trivially self-matching
        and the number meaningless. Returns NaN when the budget leaves
        too few queries for the answer to mean anything.
        """
        mask = np.ones(Z_src.shape[0], dtype=bool)
        mask[indices] = False
        queries = np.flatnonzero(mask)
        if queries.size < 2:
            return float('nan')
        if queries.size > _DIAGNOSTIC_QUERIES:
            queries = queries[
                np.linspace(0, queries.size - 1, _DIAGNOSTIC_QUERIES).astype(
                    int
                )
            ]
        scores = _local_cka(
            self.base_src_,
            self.base_tgt_,
            self.base_src_.augment(Z_src[queries]),
            self.base_tgt_.augment(Z_tgt[queries]),
        )
        matched = linear_sum_assignment(-scores)[1]
        return float(np.mean(matched == np.arange(queries.size)))

    # ------------------------------------------------------------------
    # The paper's own tasks
    # ------------------------------------------------------------------

    def local_cka(self, X_src: np.ndarray, X_tgt: np.ndarray) -> np.ndarray:
        """Local CKA between two query sets, Eq. (6).

        The retrieval score of Sec. 4.2: ``scores[i, j]`` is the global
        CKA of the base set augmented with the pair
        ``(X_src[i], X_tgt[j])``, so a row ranks every candidate target
        for one source query. Unlike :meth:`transform` this needs both
        sides' latents and does not decode anything -- it is the paper's
        measurement, exposed as it stands.

        Parameters
        ----------
        X_src : np.ndarray, shape (p, d_src)
        X_tgt : np.ndarray, shape (q, d_tgt)
            Raw latents; neither set has to be paired with the other.

        Returns
        -------
        np.ndarray, shape (p, q)
        """
        self._check_fitted()
        Z_src = self.scaler_src_.transform(np.asarray(X_src, dtype=np.float64))
        Z_tgt = self.scaler_tgt_.transform(np.asarray(X_tgt, dtype=np.float64))
        return _local_cka(
            self.base_src_,
            self.base_tgt_,
            self.base_src_.augment(Z_src),
            self.base_tgt_.augment(Z_tgt),
        )

    def match(
        self,
        X_src: np.ndarray,
        X_tgt: np.ndarray,
        method: Matcher = 'lsa',
    ) -> np.ndarray:
        """Recover the pairing of two shuffled query sets (Sec. 4).

        This is the paper's caption-matching task, and the only place
        where the method is used as it was published: no map is applied,
        the answer is a permutation.

        Parameters
        ----------
        X_src : np.ndarray, shape (q, d_src)
        X_tgt : np.ndarray, shape (q, d_tgt)
            Two query sets of equal size, in unknown correspondence.
        method : {'lsa', 'qap'}, default='lsa'
            ``'lsa'`` scores every pair with the local CKA of Eq. (6) and
            solves the resulting assignment problem exactly
            (:func:`scipy.optimize.linear_sum_assignment`). ``'qap'`` is
            the seeded quadratic assignment of Eq. (5): the base set
            seeds the FAQ solver as a partial match and the query block
            is permuted to maximise ``tr((I_M + P)' Kbar_Z (I_M + P)
            Kbar_H)``. Which wins is not settled: QAP optimises the
            objective that Eq. (3) actually asks for but FAQ only finds a
            local maximum of an NP-hard problem, while LSA solves its own
            (localised) surrogate exactly and is the only one that also
            hands back per-query rankings. On COCO the paper measures
            72.3 for QAP against 71.9 for local CKA; on this repo's
            smooth synthetic pair the order reverses, 0.98 against 0.76
            at 120 queries and 64 anchors, because FAQ's barycentre start
            has no cluster structure to latch onto.

        Returns
        -------
        np.ndarray, shape (q,)
            ``perm``, with ``X_tgt[perm[i]]`` matched to ``X_src[i]``.
        """
        self._check_fitted()
        Z_src = self.scaler_src_.transform(np.asarray(X_src, dtype=np.float64))
        Z_tgt = self.scaler_tgt_.transform(np.asarray(X_tgt, dtype=np.float64))
        if Z_src.shape[0] != Z_tgt.shape[0]:
            raise ValueError(
                'match() needs two query sets of equal size, got '
                f'{Z_src.shape[0]} and {Z_tgt.shape[0]}.'
            )
        if method == 'lsa':
            scores = _local_cka(
                self.base_src_,
                self.base_tgt_,
                self.base_src_.augment(Z_src),
                self.base_tgt_.augment(Z_tgt),
            )
            return linear_sum_assignment(-scores)[1]
        if method != 'qap':
            raise ValueError(
                f"Unknown match method {method!r}; expected 'lsa' or 'qap'."
            )

        # Eq. (5): the base pairs seed the assignment as the identity
        # block, and only the query block is free to permute.
        m = self.base_src_.n_anchors
        stacked_src = np.vstack([self.base_src_.anchors, Z_src])
        stacked_tgt = np.vstack([self.base_tgt_.anchors, Z_tgt])
        result = quadratic_assignment(
            _rescaled_kernel(self.base_src_.kernel(stacked_src, stacked_src)),
            _rescaled_kernel(self.base_tgt_.kernel(stacked_tgt, stacked_tgt)),
            method='faq',
            options={
                'maximize': True,
                'partial_match': np.tile(np.arange(m)[:, None], (1, 2)),
                'rng': self.seed,
            },
        )
        return result.col_ind[m:] - m

    # ------------------------------------------------------------------
    # The channel
    # ------------------------------------------------------------------

    def transmit(self, X_src: np.ndarray) -> np.ndarray:
        """The ``M`` anchor kernel values, i.e. what crosses the channel.

        Deliberately the *raw* row ``k(z, A_src)`` rather than the
        centred and re-scaled ``phi``: the centring is over the base set
        and the scale is ``HSIC(G, G)^{-1/2}``, both of which the
        receiver can apply itself, so spending airtime on them would be
        pointless. This and :meth:`receive` are :meth:`transform` split
        at the channel, and composing them reproduces it exactly.
        """
        self._check_fitted()
        Z = self.scaler_src_.transform(np.asarray(X_src, dtype=np.float64))
        return self.base_src_.rows(Z)

    def receive(
        self,
        U: np.ndarray,
        self_similarity: np.ndarray | None = None,
    ) -> np.ndarray:
        """Decode transmitted anchor kernel values into raw target latents.

        Parameters
        ----------
        U : np.ndarray, shape (p, M)
            Anchor kernel rows, as produced by :meth:`transmit`.
        self_similarity : np.ndarray, optional
            ``k(z, z)`` per row, needed only by ``decode='match'`` to
            complete the augmented Gram. It is identically 1 for the
            self-normalised families (``rbf``, ``laplacian``, ``cosine``)
            and is filled in automatically there, so nothing extra
            crosses the channel under the defaults; for ``linear`` and
            ``polynomial`` it is one more number per sample and must be
            supplied.
        """
        self._check_fitted()
        U = np.asarray(U, dtype=np.float64)
        if self.decode == 'ridge':
            Z_hat = self.base_src_.centre(U) @ self.readout_ + self.mean_tgt_
            return self.scaler_tgt_.inverse_transform(Z_hat)

        if self_similarity is None:
            if self.kernel not in _SELF_NORMALISED:
                raise ValueError(
                    f'receive() with decode="match" and kernel='
                    f'{self.kernel!r} needs self_similarity: k(z, z) is not '
                    'constant for this family, so the augmented Gram cannot '
                    'be completed from the anchor row alone.'
                )
            self_similarity = np.ones(U.shape[0])
        scores = _local_cka(
            self.base_src_,
            self.base_tgt_,
            self.base_src_.augment_rows(
                U, np.asarray(self_similarity, dtype=np.float64)
            ),
            self._support_parts,
        )
        Z_hat = _barycentre(scores, self.support_, self.match_temperature)
        return self.scaler_tgt_.inverse_transform(Z_hat)

    # ------------------------------------------------------------------
    # What the method costs
    # ------------------------------------------------------------------

    @property
    def paired_samples_used(self) -> int:
        """Samples both agents must embed for the map to exist.

        The default readout is a paired regression, so it spends every
        pilot it is given. The other two decodes do not: under
        ``readout='target'`` and under ``decode='match'`` the receiver
        fits its readout, or draws its support set, from its own local
        latents, which need no partner -- so the base set is the whole
        airtime bill, and the method is zero-shot in the paper's sense.
        """
        self._check_fitted()
        if self.decode == 'ridge' and self.readout == 'source':
            return int(self.n_calibration_)
        return int(self.base_src_.n_anchors)

    @property
    def transmitted_symbols(self) -> int:
        """One kernel value per anchor -- see :meth:`transmit`.

        The rate is the base-set size, exactly as it is the anchor count
        for :class:`~src.alignment.relative.RelativeRepresentationAligner`
        and for the same reason: what is sent is a vector of similarities
        against a shared set. Nothing about the latent width enters.
        """
        self._check_fitted()
        return int(self.base_src_.n_anchors)

    @property
    def map_parameters(self) -> int:
        """The transmitter's anchors plus whatever the receiver decodes with.

        Kernel methods carry their reference points into deployment, and
        stating it is the honest comparison against a linear map: the
        transmitter holds ``A_src`` to evaluate its kernel rows, and the
        receiver holds either the readout or -- under ``decode='match'``
        -- its target anchors and the support latents it retrieves from.
        The base Gram and the ``HSIC^{-1/2}`` scale are derived from the
        anchors and are not counted twice.
        """
        self._check_fitted()
        total = self.base_src_.n_parameters
        if self.decode == 'ridge':
            return int(total + self.readout_.size + self.mean_tgt_.size)
        return int(total + self.base_tgt_.n_parameters + self.support_.size)

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def _transform(self, Z_src: np.ndarray) -> np.ndarray:
        if self.decode == 'ridge':
            return (
                self.base_src_.features(Z_src) @ self.readout_ + self.mean_tgt_
            )
        scores = _local_cka(
            self.base_src_,
            self.base_tgt_,
            self.base_src_.augment(Z_src),
            self._support_parts,
        )
        return _barycentre(scores, self.support_, self.match_temperature)

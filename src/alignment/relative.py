"""Relative representations as a latent-space alignment method.

Moschella et al. (2023): every agent re-expresses a latent ``z`` through
its similarities to a shared anchor set,

    r(z) = [ sim(z, a_1), ..., sim(z, a_K) ]

Because the anchors are the *same underlying inputs* for every agent, the
resulting ``K``-dimensional relative space is approximately shared, and a
source latent can be transported to the target simply by encoding it on
the source side and decoding it on the target side.

The code is decoded back into the receiver's *raw* latent space, so its
own decoder can consume it and the method stays comparable with the other
aligners here. Two inversions are available:

- ``decode='pinv'`` -- inverse relative projection, ``Z_hat = R P_tgt^+``
  (Maiorca et al., "Latent Space Translation via Inverse Relative
  Projection"), applied to a well-conditioned projector as in Fiorellino
  et al. Exact for ``similarity='inner'``, and the default here.
- ``decode='ridge'`` -- a readout ``R -> Z_tgt`` fitted on the calibration
  pairs. The pseudo-inverse only inverts a *linear* projection, so it
  cannot handle a general ``sim``; fitting the inversion instead is the
  route the semantic-channel-equalization line of work takes to reach
  arbitrary similarities. This is its cheapest (linear, closed-form)
  case, not that paper's optimization-based inversion -- and it is the
  only decode here that survives ``similarity='cosine'``.

Two defaults are load-bearing:

- ``parseval=True``. Real anchors are strongly correlated, so a raw
  ``P_tgt^+`` amplifies the two agents' relative-space mismatch rather
  than inverting it. A Parseval frame makes ``P^+ = P^T`` exactly, so
  decoding becomes an orthogonal projection (see :func:`parseval_frame`
  for the two regimes).
- ``similarity='cosine'``. The canonical choice, but it encodes only the
  direction, so the inverse projection returns a unit-norm vector while
  the target latents have norm ~``sqrt(d)``. One fitted scalar puts it
  back (:meth:`RelativeRepresentationAligner._fit_norm_scale`); without
  it the re-added mean dominates and retrieval collapses to chance.

Anchor correspondence across the two spaces is the crux, and is handled in
one of three ways (``anchor_transfer``):

- ``'indices'``    : the anchor strategy returns *sample indices*, which
  are re-embedded by the target encoder. Requires an index-based strategy
  (``random``, ``fps``, ``stratified``, or ``kmeans`` with medoids).
- ``'clusters'``   : the source's cluster structure is *injected* into the
  target space and each anchor is recomputed as the centroid of its
  cluster members there (the SEMASIA injected-prototype construction);
  works with any strategy, including plain k-means centroids.
- ``'prototypes'`` : the *support set* of each anchor -- the ``M`` sample
  indices actually averaged into it -- is shared, and each agent means its
  own embeddings of those same samples. This is Alg. 1 of Fiorellino et
  al.; it differs from ``'clusters'`` only when ``n_prototype_samples``
  subsamples the cluster, but there the difference is essential, since
  otherwise the two agents would average different subsets.

Three named methods of the literature are reachable from this one class,
and each has a config preset under ``config/hydra/alignment/``:

======================  ==================================================
 preset                  construction
======================  ==================================================
 ``relative``            Moschella et al.: k-means medoid anchors,
                         cosine, Parseval, pseudo-inverse decode.
 ``rr``                  Maiorca et al., "Inverse Relative Projection":
                         random anchors, no Parseval, anchor *pruning*
                         (``prune_threshold``) and *subspace* ensembling
                         (``n_subspaces``) as the conditioning fix.
 ``ppfe``                Fiorellino et al., Parseval Frame Equalizer with
                         prototypical anchors (:class:`PPFEAligner`).
======================  ==================================================
"""

from __future__ import annotations

import logging

import numpy as np

from ..anchors import Anchor, AnchorStrategy
from .base import Aligner

log = logging.getLogger(__name__)

__all__ = [
    'PPFEAligner',
    'RelativeRepresentationAligner',
    'parseval_frame',
    'prune_anchors',
]


def parseval_frame(P: np.ndarray) -> np.ndarray:
    """Nearest matrix to ``P`` with orthonormal rows *or* columns.

    Computes ``U V^T`` from the thin SVD of ``P``, which is the analytic
    whitening of the anchor projector: it removes the anchor set's own
    redundancy so that the frame operator becomes an identity and the
    pseudo-inverse collapses to a transpose. Which identity depends on
    the regime, and both are used here:

    ========================  ==============  ==========================
     regime                    result          interpretation
    ========================  ==============  ==========================
     ``K <= d`` (deficient)    ``P P^T = I``   compression: the decode is
                                               the orthogonal projection
                                               onto the anchor span.
     ``K >= d`` (redundant)    ``P^T P = I``   Parseval frame proper: the
                                               reconstruction formula
                                               ``x = sum_n <x, f_n> f_n``
                                               is exact.
    ========================  ==============  ==========================

    The redundant case is the one Fiorellino et al. build the Parseval
    Frame Equalizer on -- ``F~ = F (F^H F)^{-1/2}``, which is exactly
    ``U V^T`` -- and it needs ``K >= max(d_src, d_tgt)`` for the two
    latent spaces to be fully spanned. In *both* regimes ``P^+ = P^T``
    and ``cond(P) = 1``, which is the whole point.

    Parameters
    ----------
    P : np.ndarray, shape (K, d)
        Anchor projector, one anchor per row.

    Returns
    -------
    np.ndarray, shape (K, d)
    """
    P = np.asarray(P, dtype=np.float64)
    U, _, Vt = np.linalg.svd(P, full_matrices=False)
    return U @ Vt


def prune_anchors(
    P: np.ndarray,
    threshold: float,
    seed: int = 42,
) -> np.ndarray:
    """Farthest-point sampling of anchors under the ``|cosine|`` distance.

    The "anchor pruning" of Maiorca et al. (Sec. 3.3): the anchors are a
    subset of the data manifold and are therefore correlated, which makes
    the relative projector ill-conditioned and its pseudo-inverse
    unstable. Greedy FPS under

        d(a_i, a_j) = 1 - |cos(a_i, a_j)|

    keeps the most mutually orthogonal subset; the absolute value is the
    paper's, and stops two anti-parallel anchors (which carry the same
    direction) from looking maximally far apart. Selection stops as soon
    as the best remaining candidate is closer than ``threshold`` to what
    is already chosen, so ``threshold`` -- their ``delta`` -- sets the
    size of the surviving set rather than a count.

    Parameters
    ----------
    P : np.ndarray, shape (K, d)
        Anchor projector, one anchor per row.
    threshold : float
        Minimum acceptable ``|cosine|`` distance between kept anchors.
        ``0.0`` keeps everything.
    seed : int, default=42
        Seed of the (arbitrary) first pick.

    Returns
    -------
    np.ndarray
        Sorted indices of the kept anchors.
    """
    P = np.asarray(P, dtype=np.float64)
    K = P.shape[0]
    unit = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-12)
    distance = 1.0 - np.abs(unit @ unit.T)

    rng = np.random.default_rng(seed)
    first = int(rng.integers(K))
    selected = [first]
    min_distance = distance[first].copy()

    while len(selected) < K:
        candidate = int(np.argmax(min_distance))
        if min_distance[candidate] < threshold:
            break
        selected.append(candidate)
        min_distance = np.minimum(min_distance, distance[candidate])

    return np.sort(np.asarray(selected, dtype=int))


class RelativeRepresentationAligner(Aligner):
    """Anchor-based relative representations, decoded into target space.

    Parameters
    ----------
    n_anchors : int | None, default=None
        Size ``K`` of the anchor set. ``None`` takes it from the latent
        dimensions -- ``min(d_src, d_tgt)``, the ``k = d`` of the
        relative-representation papers, which is the largest anchor set
        whose projector can still be inverted on both sides.
        :class:`PPFEAligner` overrides that default the other way, since
        a Parseval frame wants to be redundant. Either way the count is
        capped at the number of calibration samples, with a warning.
    strategy : {'random', 'kmeans', 'fps', 'stratified'}, default='kmeans'
        Anchor selection strategy (see :class:`src.anchors.Anchor`).
    medoids : bool, default=True
        For ``strategy='kmeans'``, snap each centroid to its closest
        actual sample so the anchors become index-based (and therefore
        shareable). Ignored by the other strategies.
    anchor_transfer : {'indices', 'clusters', 'prototypes'}, default='indices'
        How the anchors are rendered in the target space (see the module
        docstring).
    n_prototype_samples : int | None, default=None
        For ``anchor_transfer='prototypes'``, the ``M`` cluster members
        averaged into each prototypical anchor. ``None`` averages every
        member, which makes prototypes and clusters coincide.
    similarity : {'cosine', 'inner'}, default='cosine'
        Similarity used against the anchors. Under ``'cosine'`` the code
        carries only the direction, so the decoded latent is put back on
        the target's scale by a single fitted factor (see
        :meth:`_fit_norm_scale`).
    parseval : bool, default=True
        Orthonormalise each anchor projector before use, making its
        pseudo-inverse a transpose and its condition number 1.
    prune_threshold : float | None, default=None
        Minimum ``|cosine|`` distance between anchors (Maiorca et al.'s
        ``delta``). ``None`` disables pruning and uses every anchor. This
        is an alternative conditioning fix to ``parseval``: the inverse-
        relative-projection paper prunes instead of orthonormalising.
    n_subspaces : int, default=1
        Number of independently pruned anchor subsets (their ``omega``)
        whose reconstructions are averaged, which buys back the coverage
        that aggressive pruning gives up. Only meaningful together with
        ``prune_threshold``.
    decode : {'pinv', 'ridge'}, default='pinv'
        How the relative code is brought back into the target's raw
        latent space.
    decode_alpha : float, default=1e-2
        Ridge strength of the ``'ridge'`` decoder, relative to the trace
        of the relative-space Gram matrix. Ignored by ``'pinv'``.
    preprocess : ScalingMethod, default='standard'
        Per-space standardisation. Whitening flattens the spectrum, which
        leaves a ``K``-anchor span covering far less of the latent energy;
        empirically it costs most of the retrieval accuracy here.
    eps : float, default=1e-6
        Relative covariance ridge of the whitening step.
    seed : int, default=42
        Seed of the anchor selection and of the pruning subspaces.
    """

    def __init__(
        self,
        n_anchors: int | None = None,
        strategy: AnchorStrategy = 'kmeans',
        medoids: bool = True,
        anchor_transfer: str = 'indices',
        n_prototype_samples: int | None = None,
        similarity: str = 'cosine',
        parseval: bool = True,
        prune_threshold: float | None = None,
        n_subspaces: int = 1,
        decode: str = 'pinv',
        decode_alpha: float = 1e-2,
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
        if similarity not in ('cosine', 'inner'):
            raise ValueError(f'Unknown similarity {similarity!r}.')
        if anchor_transfer not in ('indices', 'clusters', 'prototypes'):
            raise ValueError(f'Unknown anchor_transfer {anchor_transfer!r}.')
        if decode not in ('ridge', 'pinv'):
            raise ValueError(f'Unknown decode {decode!r}.')
        if int(n_subspaces) < 1:
            raise ValueError(f'n_subspaces must be >= 1, got {n_subspaces}.')
        self.n_anchors = None if n_anchors is None else int(n_anchors)
        self.strategy: AnchorStrategy = strategy
        self.medoids = bool(medoids)
        self.anchor_transfer = anchor_transfer
        self.n_prototype_samples = n_prototype_samples
        self.similarity = similarity
        self.parseval = bool(parseval)
        self.prune_threshold = prune_threshold
        self.n_subspaces = int(n_subspaces)
        self.decode = decode
        self.decode_alpha = float(decode_alpha)

        self.anchor_: Anchor | None = None
        self.P_src_: np.ndarray | None = None
        self.P_tgt_: np.ndarray | None = None
        self.readout_: np.ndarray | None = None
        self.subspaces_: list[np.ndarray] | None = None
        self.norm_scale_: float = 1.0

    def hyperparameters(self) -> dict:
        params = super().hyperparameters()
        params.update(
            n_anchors=self.n_anchors,
            strategy=self.strategy,
            medoids=self.medoids,
            anchor_transfer=self.anchor_transfer,
            n_prototype_samples=self.n_prototype_samples,
            similarity=self.similarity,
            parseval=self.parseval,
            prune_threshold=self.prune_threshold,
            n_subspaces=self.n_subspaces,
            decode=self.decode,
            decode_alpha=self.decode_alpha,
        )
        return params

    # ------------------------------------------------------------------
    # Relative space
    # ------------------------------------------------------------------

    def encode(self, X_src: np.ndarray) -> np.ndarray:
        """Relative representation of raw source latents, shape ``(m, K)``."""
        self._check_fitted()
        Z = self.scaler_src_.transform(np.asarray(X_src, dtype=np.float64))
        return self._project(Z, self.P_src_)

    def _project(self, Z: np.ndarray, P: np.ndarray) -> np.ndarray:
        """Similarities of standardised ``Z`` against projector ``P``."""
        if self.similarity == 'cosine':
            norms = np.linalg.norm(Z, axis=1, keepdims=True)
            Z = Z / np.maximum(norms, 1e-12)
        return Z @ P.T

    def _projector(self, anchors: np.ndarray) -> np.ndarray:
        """Build the anchor projector matching :attr:`similarity`."""
        P = np.asarray(anchors, dtype=np.float64)
        if self.similarity == 'cosine':
            norms = np.linalg.norm(P, axis=1, keepdims=True)
            P = P / np.maximum(norms, 1e-12)
        # Parseval comes last so that it composes predictably with the
        # cosine row-normalisation rather than replacing it.
        return parseval_frame(P) if self.parseval else P

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def _fit(
        self,
        Z_src: np.ndarray,
        Z_tgt: np.ndarray,
        labels: np.ndarray | None = None,
    ) -> None:
        n, d_src = Z_src.shape
        d_tgt = Z_tgt.shape[1]
        n_anchors = (
            self._default_n_anchors(d_src, d_tgt)
            if self.n_anchors is None
            else self.n_anchors
        )
        if n_anchors > n:
            log.warning(
                'n_anchors=%d exceeds the %d available calibration '
                'samples; using %d anchors instead.',
                n_anchors,
                n,
                n,
            )
            n_anchors = n
        if self.parseval:
            log.debug(
                'Parseval frame with K=%d anchors in (%d -> %d) dimensions: '
                '%s regime.',
                n_anchors,
                d_src,
                d_tgt,
                'redundant'
                if n_anchors >= max(d_src, d_tgt)
                else 'rank-deficient',
            )

        # Anchors are selected once, on the source cloud, and rendered in
        # the target space through the shared sample correspondence.
        fit_kwargs = {'n_anchors': n_anchors}
        if self.strategy == 'kmeans':
            fit_kwargs['medoids'] = self.medoids
            fit_kwargs['n_samples'] = self.n_prototype_samples
        if self.strategy == 'stratified':
            if labels is None:
                raise ValueError(
                    "strategy='stratified' requires per-sample labels; pass "
                    'them to fit(..., labels=...).'
                )
            fit_kwargs['labels'] = labels

        self.anchor_ = Anchor(Z_src, strategy=self.strategy, seed=self.seed)
        self.anchor_.fit(**fit_kwargs)

        anchors_src = self.anchor_.anchors
        anchors_tgt = self._render_target_anchors(Z_tgt)

        self.P_src_ = self._projector(anchors_src)
        self.P_tgt_ = self._projector(anchors_tgt)
        self.subspaces_ = self._select_subspaces()

        R_src = self._project(Z_src, self.P_src_)
        R_tgt = self._project(Z_tgt, self.P_tgt_)
        self.readout_ = self._fit_readout(R_tgt, Z_tgt)
        self.norm_scale_ = self._fit_norm_scale(Z_src, Z_tgt)

        kept = [len(s) for s in self.subspaces_]
        self.diagnostics_ = {
            'rr_n_anchors_effective': int(self.P_src_.shape[0]),
            'rr_index_based': self.anchor_.indices is not None,
            'rr_norm_scale': self.norm_scale_,
            'rr_n_anchors_pruned': float(np.mean(kept)),
            # 1.0 under `parseval`; large values mean the decode is
            # inverting an ill-conditioned projector. Measured on the
            # pruned subsets, which is what is actually inverted.
            'rr_projector_cond': float(
                np.mean(
                    [np.linalg.cond(self.P_tgt_[s]) for s in self.subspaces_]
                )
            ),
            # How close the two relative spaces actually are: this is the
            # quantity the whole method assumes to be ~0.
            'rr_space_mismatch': float(
                np.linalg.norm(R_src - R_tgt)
                / max(np.linalg.norm(R_tgt), 1e-12)
            ),
        }

    @staticmethod
    def _default_n_anchors(d_src: int, d_tgt: int) -> int:
        """Anchor count when none is configured.

        The relative-representation line of work sets ``k = d``, and with
        two spaces of different width the binding constraint is the
        narrower one: an anchor set larger than ``d_tgt`` cannot be
        inverted on the receiver's side without the pseudo-inverse
        amplifying the anchors' own correlation.
        """
        return min(d_src, d_tgt)

    def _select_subspaces(self) -> list[np.ndarray]:
        """Anchor subsets whose reconstructions are averaged."""
        K = self.P_src_.shape[0]
        if self.prune_threshold is None:
            return [np.arange(K)]
        return [
            prune_anchors(
                self.P_src_, float(self.prune_threshold), seed=self.seed + i
            )
            for i in range(self.n_subspaces)
        ]

    def _fit_norm_scale(self, Z_src: np.ndarray, Z_tgt: np.ndarray) -> float:
        """Global scale putting a decoded *direction* back on target scale.

        Cosine similarity encodes ``z / ||z||``, so the pseudo-inverse
        returns a unit-norm direction while the target latents have norm
        ~``sqrt(d)``. Left uncorrected the decoded latent is swamped by
        the mean that :meth:`Aligner.transform` adds back, and retrieval
        drops to chance. The fix is one scalar, matching total energy on
        the calibration split.

        Note this is norm *restoration*, not a least-squares fit: the
        MSE-optimal scale deliberately shrinks toward the mean, which
        lowers NMSE but costs per-sample discriminability. The fitted
        ``'ridge'`` decode already carries its own scale, so it is left
        alone.
        """
        self.norm_scale_ = 1.0  # _transform applies it; start neutral
        if self.similarity != 'cosine' or self.decode != 'pinv':
            return 1.0
        decoded = self._transform(Z_src)
        energy = float(np.sum(decoded**2))
        if energy <= 0:
            return 1.0
        return float(np.sqrt(float(np.sum(Z_tgt**2)) / energy))

    def _render_target_anchors(self, Z_tgt: np.ndarray) -> np.ndarray:
        """Express the source-selected anchors in the target space."""
        if self.anchor_transfer == 'indices':
            indices = self.anchor_.indices
            if indices is None:
                raise ValueError(
                    f"anchor_transfer='indices' needs an index-based anchor "
                    f'strategy, but {self.strategy!r}'
                    + (
                        ' with medoids=False'
                        if self.strategy == 'kmeans'
                        else ''
                    )
                    + ' produced centroids. Use medoids=True, '
                    "anchor_transfer='clusters', or "
                    "anchor_transfer='prototypes'."
                )
            return Z_tgt[indices]

        # 'clusters' averages every member of each source cluster in the
        # target space; 'prototypes' averages the same M members the
        # source anchor was built from, which is the only version that
        # still corresponds once the cluster is subsampled.
        groups = (
            self.anchor_.support_indices
            if self.anchor_transfer == 'prototypes'
            else self.anchor_.cluster_indices
        )
        anchors = np.empty((self.anchor_.n_anchors, Z_tgt.shape[1]))
        for i in range(self.anchor_.n_anchors):
            members = groups.get(i)
            if members is None or len(members) == 0:
                log.warning(
                    'Anchor %d has an empty cluster; falling back to the '
                    'target-space mean.',
                    i,
                )
                anchors[i] = Z_tgt.mean(axis=0)
            else:
                anchors[i] = Z_tgt[members].mean(axis=0)
        return anchors

    def _fit_readout(self, R_tgt: np.ndarray, Z_tgt: np.ndarray) -> np.ndarray:
        """Readout matrix ``(K, d_tgt)`` from relative to raw target space.

        Under ``'pinv'`` and ``n_subspaces > 1`` the per-subspace
        reconstructions are averaged, and because every one of them is
        linear in the relative code the ensemble collapses into a single
        matrix -- the mean of the per-subspace pseudo-inverses, scattered
        back into the full anchor layout.
        """
        if self.decode == 'ridge':
            gram = R_tgt.T @ R_tgt
            ridge = self.decode_alpha * max(
                float(np.trace(gram)) / gram.shape[0], 1e-30
            )
            gram[np.diag_indices_from(gram)] += ridge
            return np.linalg.solve(gram, R_tgt.T @ Z_tgt)  # (K, d_tgt)

        # Z P_tgt^T = R  =>  Z_hat = R (P_tgt^+)^T, and (P^+)^T = P under
        # a Parseval frame.
        readout = np.zeros((self.P_tgt_.shape[0], Z_tgt.shape[1]))
        for subset in self.subspaces_:
            readout[subset] += np.linalg.pinv(self.P_tgt_[subset]).T
        return readout / len(self.subspaces_)

    # ------------------------------------------------------------------
    # What the method costs
    # ------------------------------------------------------------------

    @property
    def paired_samples_used(self) -> int:
        """Samples both agents must embed for the map to exist.

        Under ``decode='pinv'`` the map is built entirely from the two
        anchor matrices, so the cost is whatever those anchors are made
        of -- which is *not* the anchor count once the anchors are
        prototypes. An index-based transfer needs the ``K`` anchor
        samples and nothing else. A prototype or cluster transfer needs
        every sample that enters a mean, because the receiver has to
        embed the same ones to build the matching prototype; with
        ``n_prototype_samples=None`` that is the whole pilot set, and
        with ``M`` set it is at most ``K x M``.

        The fitted ``'ridge'`` readout is a genuine regression on every
        pilot, so there the whole budget is consumed regardless.
        """
        self._check_fitted()
        if self.decode == 'ridge' or self.anchor_transfer != 'indices':
            groups = (
                self.anchor_.support_indices
                if self.anchor_transfer == 'prototypes'
                else self.anchor_.cluster_indices
            )
            if self.decode == 'ridge':
                return int(self.n_calibration_)
            used = {int(i) for members in groups.values() for i in members}
            return len(used)
        return int(self.P_src_.shape[0])

    @property
    def transmitted_symbols(self) -> int:
        """The frame coefficients, one per anchor.

        ``c = F x`` is what goes on the channel, so the anchor count *is*
        the rate -- which is why the relative-representation line of work
        sweeps it directly (PPFE's compression factor is defined from
        it). Nothing about the whitening rank enters here.
        """
        self._check_fitted()
        return int(self.P_src_.shape[0])

    @property
    def map_parameters(self) -> int:
        """Both anchor projectors: the analysis operator and the readout."""
        self._check_fitted()
        return self.P_src_.size + self.readout_.size

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def _transform(self, Z_src: np.ndarray) -> np.ndarray:
        # A pruned-away anchor carries a zero row in `readout_`, so the
        # full relative code can be used as-is: its unused coordinates
        # multiply zero and the product is exactly the subspace ensemble.
        return self.norm_scale_ * (
            self._project(Z_src, self.P_src_) @ self.readout_
        )


class PPFEAligner(RelativeRepresentationAligner):
    """Proto-anchor Parseval Frame Equalizer (Fiorellino et al.).

    The Parseval Frame Equalizer treats the anchor matrix as the
    *analysis operator* of a frame rather than as a similarity table. The
    transmitter sends frame coefficients ``c = F x``; the receiver
    synthesises its own latent from them with its own frame,
    ``y_hat = sum_n [c]_n g_n``. Whitening ``F`` into a Parseval frame --
    ``F~ = F (F^H F)^{-1/2}``, which is what :func:`parseval_frame`
    computes -- forces the frame operator to the identity, so the
    synthesis is a plain transpose with condition number 1 and the
    reconstruction never amplifies the mismatch between the two agents.

    The user-facing pipeline is therefore whiten (Parsevalise the anchor
    frame) -> align (transmit coefficients in the shared relative space)
    -> colour (synthesise with the receiver's own frame), and it is
    zero-shot: nothing but the anchors is fitted.

    Anchors are *prototypical* (Alg. 1 of the paper): the source clusters
    its latents into ``K`` groups, draws ``M`` samples from each, and
    means their embeddings. The support sets are shared, so the receiver
    builds the matching prototype from its own embeddings of the very
    same samples. Prototypes sit in dense regions of the latent space and
    are far more robust to outliers than single-sample anchors.

    Parameters
    ----------
    n_anchors : int | None, default=None
        Frame size ``N``. ``None`` resolves to ``max(d_src, d_tgt)``,
        which is what the paper asks for: below it the two latent spaces
        are not both fully spanned and the equalizer becomes a
        *compression*, projecting onto the anchor span. That regime is
        legitimate -- it is the paper's own rate/accuracy trade-off, swept
        in their Fig. 2 -- so an explicit smaller ``N`` is a valid
        setting, not a mistake. Capped at the pilot budget, which is the
        real constraint: a redundant frame needs at least
        ``max(d_src, d_tgt)`` paired samples to build the anchors from.
    n_prototype_samples : int | None, default=None
        ``M``, the samples averaged into each prototype. ``None`` uses
        every member of the cluster, i.e. the true cluster centroid.

    Other parameters are inherited from
    :class:`RelativeRepresentationAligner`; the defaults changed here are
    ``strategy='kmeans'``, ``medoids=False``,
    ``anchor_transfer='prototypes'``, ``similarity='inner'`` and
    ``parseval=True``, which together are the paper's construction.
    """

    def __init__(
        self,
        n_anchors: int | None = None,
        n_prototype_samples: int | None = None,
        strategy: AnchorStrategy = 'kmeans',
        medoids: bool = False,
        anchor_transfer: str = 'prototypes',
        similarity: str = 'inner',
        parseval: bool = True,
        decode: str = 'pinv',
        decode_alpha: float = 1e-2,
        preprocess: str = 'standard',
        eps: float = 1e-6,
        n_components: float | None = None,
        shrinkage: str | float | None = 'auto',
        seed: int = 42,
    ) -> None:
        super().__init__(
            n_anchors=n_anchors,
            strategy=strategy,
            medoids=medoids,
            anchor_transfer=anchor_transfer,
            n_prototype_samples=n_prototype_samples,
            similarity=similarity,
            parseval=parseval,
            decode=decode,
            decode_alpha=decode_alpha,
            preprocess=preprocess,
            eps=eps,
            n_components=n_components,
            shrinkage=shrinkage,
            seed=seed,
        )

    @staticmethod
    def _default_n_anchors(d_src: int, d_tgt: int) -> int:
        """A Parseval frame wants to be redundant, not minimal.

        ``N >= max(d_src, d_tgt)`` is the paper's condition for both
        latent spaces to be fully spanned by the analysis operator, and
        therefore for the reconstruction formula to be exact rather than
        a projection.
        """
        return max(d_src, d_tgt)

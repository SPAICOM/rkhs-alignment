"""Anchor / semantic-pilot selection strategies.

An anchor set is a small collection of points in a latent space used as
probes for the relative-representation projection. This module mirrors
the ``Anchor`` object of the SEMASIA reference implementation
(``SPAICOM/semasia-datasets``) and extends it with strategies that
return *sample indices*, so the very same anchor inputs can be
re-embedded by every agent of a network.

The same machinery selects *semantic pilots*: the paired calibration
samples any alignment map is fitted on. Anchors and pilots are the same
object seen from two angles -- a representative subset of the latent
distribution -- so :func:`src.alignment.select_pilots` simply reuses the
strategies here rather than duplicating them.

``'herding'`` implements the kernel-aware pilot design of the RKA paper
(Sec. IV): rather than sampling at random, it greedily minimises the
maximum mean discrepancy between the full candidate pool and the selected
subset, *in the same RKHS geometry* the aligner will later fit in.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

from ..kernels import Kernel

if TYPE_CHECKING:
    from sklearn.base import ClusterMixin

log = logging.getLogger(__name__)

AnchorStrategy = Literal[
    'random',
    'kmeans',
    'fps',
    'stratified',
    'herding',
    'round_robin',
]

# Rows of the candidate pool processed per chunk when accumulating the
# kernel mean embedding, which would otherwise need the full M x M Gram.
_HERDING_CHUNK = 512


class Anchor:
    """Anchor points derived from a point cloud.

    Parameters
    ----------
    point_cloud : np.ndarray, shape (n_points, n_features)
        Source point cloud from which anchors are extracted.
    strategy : AnchorStrategy, default='kmeans'
        Extraction strategy:

        - ``'random'``     : uniform random sample of points
          (index-based).
        - ``'kmeans'``     : cluster centroids (optionally snapped to
          medoids, which makes them index-based).
        - ``'fps'``        : farthest-point sampling — greedy maximal
          spread (index-based).
        - ``'stratified'`` : label-stratified medoids, one quota of
          anchors per class (index-based, requires ``labels``).
        - ``'herding'``    : kernel herding — greedy MMD matching to the
          full pool in an RKHS (index-based, deterministic).
        - ``'round_robin'``: nested stratified random ordering — classes
          are visited round-robin, one unused member each, so every
          prefix is as class-balanced as its size allows (index-based).
    seed : int, default=42
        Random seed for reproducible extraction.

    Notes
    -----
    After :meth:`fit`:

    - :attr:`anchors` always holds the anchor vectors ``(K, d)``.
    - :attr:`indices` holds the anchor sample indices when the strategy
      is index-based (``random``, ``fps``, ``stratified``, ``herding``,
      and ``kmeans`` with ``medoids=True``), else ``None``.
    - :attr:`cluster_indices` maps each anchor to the observation
      indices of its cluster (Voronoi cell for point-based strategies),
      enabling *injected* anchor computation on another agent's latent
      space.
    """

    def __init__(
        self,
        point_cloud: np.ndarray,
        strategy: AnchorStrategy = 'kmeans',
        seed: int = 42,
    ) -> None:
        self._point_cloud = np.asarray(point_cloud, dtype=np.float32)
        if self._point_cloud.ndim != 2:
            raise ValueError('point_cloud must be 2-dimensional.')
        self._strategy: AnchorStrategy = strategy
        self._seed = seed

        self._anchors: np.ndarray | None = None
        self._indices: np.ndarray | None = None
        self._order: np.ndarray | None = None
        self._cluster_indices: dict[int, np.ndarray] = {}
        self._support_indices: dict[int, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(
        self,
        n_anchors: int | None = None,
        labels: np.ndarray | None = None,
        n_samples: int | None = None,
        medoids: bool = False,
        clusters: np.ndarray | None = None,
        clusterer_cls: type[ClusterMixin] | None = None,
        clusterer_kwargs: dict | None = None,
        kernel: Kernel | None = None,
    ) -> Anchor:
        """Extract the anchors.

        Parameters
        ----------
        n_anchors : int, optional
            Number of anchors ``K``. Required for every strategy except
            ``'kmeans'`` with precomputed ``clusters``.
        labels : np.ndarray, optional
            Per-point class labels, shape ``(n_points,)``. Required by
            the ``'stratified'`` and ``'round_robin'`` strategies.
        n_samples : int | None, default=None
            For centroid-based strategies, number of cluster members
            averaged to estimate each centroid. ``None`` uses all
            members.
        medoids : bool, default=False
            For ``'kmeans'``: snap each centroid to the closest actual
            point, making the anchors index-based (shareable across
            agents without cluster injection).
        clusters : np.ndarray, optional
            Precomputed cluster label array for ``'kmeans'``, shape
            ``(n_points,)``. Skips the internal clustering.
        clusterer_cls : type[ClusterMixin], optional
            Clustering class for ``'kmeans'`` (default
            :class:`sklearn.cluster.KMeans`).
        clusterer_kwargs : dict, optional
            Extra kwargs for ``clusterer_cls``.
        kernel : Kernel, optional
            For ``'herding'``: the RKHS the MMD is measured in. Defaults
            to an RBF kernel with the median-pairwise-distance heuristic
            -- the same geometry :class:`src.alignment.RKHSAligner` uses
            by default. Pass a spec (family and ``bandwidth_scale``)
            rather than a fitted kernel: the bandwidth is fitted here, on
            the pool being selected from.

        Returns
        -------
        Anchor
            ``self``, fitted.
        """
        match self._strategy:
            case 'random':
                self._fit_random(n_anchors)
            case 'kmeans':
                self._fit_kmeans(
                    n_anchors=n_anchors,
                    n_samples=n_samples,
                    medoids=medoids,
                    clusters=clusters,
                    clusterer_cls=clusterer_cls,
                    clusterer_kwargs=clusterer_kwargs,
                )
            case 'fps':
                self._fit_fps(n_anchors)
            case 'stratified':
                self._fit_stratified(n_anchors, labels)
            case 'herding':
                self._fit_herding(n_anchors, kernel)
            case 'round_robin':
                self._fit_round_robin(n_anchors, labels)
            case _:
                raise ValueError(
                    f'Unknown anchor strategy {self._strategy!r}. Supported: '
                    "'random', 'kmeans', 'fps', 'stratified', 'herding', "
                    "'round_robin'."
                )
        return self

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def strategy(self) -> AnchorStrategy:
        """Extraction strategy."""
        return self._strategy

    @property
    def anchors(self) -> np.ndarray:
        """Anchor vectors, shape ``(K, n_features)``."""
        if self._anchors is None:
            raise RuntimeError('Anchor.fit() must be called first.')
        return self._anchors

    @property
    def indices(self) -> np.ndarray | None:
        """Anchor sample indices ``(K,)``, or ``None`` if centroid-based."""
        return self._indices

    @property
    def order(self) -> np.ndarray | None:
        """Selection order for the greedy strategies, else ``None``.

        ``'fps'`` and ``'herding'`` both build their set one point at a
        time and never revisit, so the first ``k`` entries here are
        exactly what the same strategy would return for a budget of
        ``k``. A budget sweep can therefore select once at the largest
        budget and slice, instead of re-running the selection per budget.
        """
        return self._order

    @property
    def cluster_indices(self) -> dict[int, np.ndarray]:
        """Anchor index -> observation indices of its cluster/cell."""
        return self._cluster_indices

    @property
    def support_indices(self) -> dict[int, np.ndarray]:
        """Anchor index -> the observations actually averaged into it.

        This is the *support set* ``S_i`` of Alg. 1 in Fiorellino et al.:
        the ``M`` samples drawn from cluster ``i`` whose embeddings are
        meaned to form the prototypical anchor. It is what a second agent
        needs in order to build the *same* anchor in its own latent space
        -- the cluster assignment alone is not enough once ``n_samples``
        subsamples it, because the two agents would then average
        different subsets and the anchors would stop corresponding.

        Falls back to :attr:`cluster_indices` for the strategies that
        average every member (or select a single point).
        """
        return self._support_indices or self._cluster_indices

    @property
    def n_anchors(self) -> int:
        """Number of anchors (0 before :meth:`fit`)."""
        return 0 if self._anchors is None else self._anchors.shape[0]

    @property
    def is_fitted(self) -> bool:
        """Whether :meth:`fit` has been called successfully."""
        return self._anchors is not None

    def __repr__(self) -> str:
        return (
            f'Anchor(strategy={self._strategy!r}, '
            f'n_anchors={self.n_anchors}, '
            f'index_based={self._indices is not None})'
        )

    # ------------------------------------------------------------------
    # Strategies
    # ------------------------------------------------------------------

    def _fit_random(self, n_anchors: int | None) -> None:
        if n_anchors is None:
            raise ValueError("'random' strategy requires n_anchors.")
        n = self._point_cloud.shape[0]
        if n_anchors > n:
            raise ValueError(
                f'n_anchors ({n_anchors}) exceeds n_points ({n}).'
            )
        rng = np.random.default_rng(self._seed)
        idx = np.sort(rng.choice(n, size=n_anchors, replace=False))
        self._indices = idx
        self._anchors = self._point_cloud[idx].copy()
        self._voronoi_partition()

    def _fit_kmeans(
        self,
        n_anchors: int | None,
        n_samples: int | None,
        medoids: bool,
        clusters: np.ndarray | None,
        clusterer_cls: type[ClusterMixin] | None,
        clusterer_kwargs: dict | None,
    ) -> None:
        kwargs = dict(clusterer_kwargs) if clusterer_kwargs else {}

        if clusters is not None:
            cluster_labels = np.asarray(clusters)
        else:
            n = self._point_cloud.shape[0]
            if n_anchors is not None and n_anchors > n:
                raise ValueError(
                    f'n_anchors ({n_anchors}) exceeds n_points ({n}).'
                )
            if clusterer_cls is None:
                clusterer_cls = KMeans
            if n_anchors is not None:
                kwargs.setdefault('n_clusters', n_anchors)
            kwargs.setdefault('random_state', self._seed)
            cluster_labels = clusterer_cls(**kwargs).fit_predict(
                self._point_cloud
            )

        unique = np.unique(cluster_labels)
        # Noise label of density-based clusterers (e.g. DBSCAN)
        unique = unique[unique != -1]
        k = unique.shape[0]
        d = self._point_cloud.shape[1]
        anchors = np.empty((k, d), dtype=np.float32)
        indices = np.empty(k, dtype=int) if medoids else None

        for i, c in enumerate(unique):
            members = np.where(cluster_labels == c)[0]
            self._cluster_indices[i] = members

            pool = members
            if n_samples is not None and n_samples < members.shape[0]:
                rng = np.random.default_rng(self._seed + i)
                pool = np.sort(
                    rng.choice(members, size=n_samples, replace=False)
                )
            self._support_indices[i] = pool
            centroid = self._point_cloud[pool].mean(axis=0)

            if medoids:
                dists = np.linalg.norm(
                    self._point_cloud[members] - centroid, axis=1
                )
                indices[i] = members[np.argmin(dists)]
                anchors[i] = self._point_cloud[indices[i]]
            else:
                anchors[i] = centroid

        self._anchors = anchors
        self._indices = indices

    def _fit_fps(self, n_anchors: int | None) -> None:
        if n_anchors is None:
            raise ValueError("'fps' strategy requires n_anchors.")
        X = self._point_cloud
        n = X.shape[0]
        if n_anchors > n:
            raise ValueError(
                f'n_anchors ({n_anchors}) exceeds n_points ({n}).'
            )
        rng = np.random.default_rng(self._seed)

        idx = np.empty(n_anchors, dtype=int)
        idx[0] = rng.integers(n)
        min_dist = np.linalg.norm(X - X[idx[0]], axis=1)
        for i in range(1, n_anchors):
            idx[i] = int(np.argmax(min_dist))
            dist = np.linalg.norm(X - X[idx[i]], axis=1)
            min_dist = np.minimum(min_dist, dist)

        self._order = idx
        self._indices = np.sort(idx)
        self._anchors = X[self._indices].copy()
        self._voronoi_partition()

    def _fit_stratified(
        self,
        n_anchors: int | None,
        labels: np.ndarray | None,
    ) -> None:
        if n_anchors is None:
            raise ValueError("'stratified' strategy requires n_anchors.")
        if labels is None:
            raise ValueError("'stratified' strategy requires labels.")
        labels = np.asarray(labels)
        if labels.shape[0] != self._point_cloud.shape[0]:
            raise ValueError('labels length must match n_points.')

        classes = np.unique(labels)
        rng = np.random.default_rng(self._seed)

        # Distribute the anchor budget as evenly as possible over the
        # classes; the remainder goes to a random subset of classes.
        base, extra = divmod(n_anchors, classes.shape[0])
        quotas = np.full(classes.shape[0], base, dtype=int)
        if extra:
            quotas[rng.choice(classes.shape[0], extra, replace=False)] += 1

        indices: list[int] = []
        for c, quota in zip(classes, quotas):
            if quota == 0:
                continue
            members = np.where(labels == c)[0]
            Xc = self._point_cloud[members]
            if quota == 1:
                centroids = Xc.mean(axis=0, keepdims=True)
            else:
                q = min(quota, members.shape[0])
                km = KMeans(n_clusters=q, random_state=self._seed)
                km.fit(Xc)
                centroids = km.cluster_centers_
            # Snap each per-class centroid to its medoid so the anchor
            # is an actual sample index, shareable across agents.
            dists = cdist(centroids, Xc)
            indices.extend(int(members[np.argmin(row)]) for row in dists)

        # Two centroids of the same class can snap to the same medoid;
        # np.unique both sorts and drops those duplicates, which would
        # otherwise become duplicated (perfectly collinear) anchors.
        self._indices = np.unique(np.asarray(indices, dtype=int))
        if self._indices.shape[0] < n_anchors:
            log.warning(
                "'stratified' produced %d anchors instead of the %d "
                'requested: some per-class centroids shared a medoid.',
                self._indices.shape[0],
                n_anchors,
            )
        self._anchors = self._point_cloud[self._indices].copy()
        self._voronoi_partition()

    def _fit_round_robin(
        self, n_anchors: int | None, labels: np.ndarray | None
    ) -> None:
        """Nested stratified random ordering, truncated to ``n_anchors``.

        Each class contributes a randomly permuted queue of its members.
        Classes are then visited round-robin -- in a fresh random order
        every pass -- each handing over its next unused member. The
        result is an ordering of the whole pool whose every prefix is as
        class-balanced as its length allows, so a budget sweep can slice
        prefixes instead of re-drawing, and larger budgets contain the
        smaller ones.

        This differs from ``'stratified'``, which allocates a fixed quota
        per class and snaps to per-class k-means medoids: that answer is
        geometric and depends on the requested size, whereas this one is
        random and nested.
        """
        if n_anchors is None:
            raise ValueError("'round_robin' strategy requires n_anchors.")
        if labels is None:
            raise ValueError("'round_robin' strategy requires labels.")
        labels = np.asarray(labels)
        n = self._point_cloud.shape[0]
        if labels.shape[0] != n:
            raise ValueError('labels length must match n_points.')
        if n_anchors > n:
            raise ValueError(
                f'n_anchors ({n_anchors}) exceeds n_points ({n}).'
            )

        rng = np.random.default_rng(self._seed)
        queues = [
            list(rng.permutation(np.where(labels == c)[0]))
            for c in np.unique(labels)
        ]
        positions = [0] * len(queues)

        order: list[int] = []
        while len(order) < n:
            for j in rng.permutation(len(queues)):
                if positions[j] < len(queues[j]):
                    order.append(int(queues[j][positions[j]]))
                    positions[j] += 1

        self._order = np.asarray(order[:n_anchors], dtype=int)
        self._indices = np.sort(self._order)
        self._anchors = self._point_cloud[self._indices].copy()
        self._voronoi_partition()

    def _fit_herding(
        self, n_anchors: int | None, kernel: Kernel | None
    ) -> None:
        """Kernel herding: greedy MMD matching to the full pool.

        Implements Sec. IV of the RKA paper. The subset that best
        represents the pool is the one whose kernel mean embedding is
        closest to the pool's,

            argmin_{|P|=N} || (1/M) sum_i phi(x_i)
                             - (1/N) sum_{i in P} phi(x_i) ||^2_{H_k},

        which kernel herding (Chen et al., 2010) approximates greedily by
        repeatedly taking

            i_{t+1} = argmax_{i not in P_t}
                          mu_i - (1/t) sum_{p in P_t} k(x_i, x_p),

        where ``mu_i = (1/M) sum_j k(x_i, x_j)``. The first term prefers
        points central to the distribution, the second penalises points
        close to what is already selected -- so the set stays
        representative without piling up on one mode.

        The running sum over ``P_t`` is accumulated one kernel column at
        a time, so the full ``M x M`` Gram matrix is never formed: only
        ``mu`` needs pairwise work, and that is chunked.
        """
        if n_anchors is None:
            raise ValueError("'herding' strategy requires n_anchors.")
        X = self._point_cloud.astype(np.float64)
        n = X.shape[0]
        if n_anchors > n:
            raise ValueError(
                f'n_anchors ({n_anchors}) exceeds n_points ({n}).'
            )

        # The bandwidth belongs to the pool being selected from, so it is
        # fitted here rather than left to the caller. `Kernel.fit` is a
        # no-op once `gamma` is set explicitly, so a fully specified
        # kernel passes through untouched, while a spec carrying only a
        # `bandwidth_scale` gets the median heuristic times that scale.
        # Before this, a kernel supplied by the caller was used unfitted
        # and raised on first evaluation.
        kernel = (Kernel('rbf') if kernel is None else kernel).fit(
            X, seed=self._seed
        )

        # mu[i] = (1/M) sum_j k(x_i, x_j), the pool's mean embedding
        # evaluated at each candidate.
        mu = np.empty(n)
        for start in range(0, n, _HERDING_CHUNK):
            stop = min(start + _HERDING_CHUNK, n)
            mu[start:stop] = kernel(X[start:stop], X).mean(axis=1)

        selected = np.empty(n_anchors, dtype=int)
        # Running sum_{p in P_t} k(x_i, x_p) for every candidate i.
        affinity = np.zeros(n)
        available = np.ones(n, dtype=bool)

        for t in range(n_anchors):
            score = mu if t == 0 else mu - affinity / t
            score = np.where(available, score, -np.inf)
            pick = int(np.argmax(score))
            selected[t] = pick
            available[pick] = False
            affinity += kernel(X[pick][None, :], X).ravel()

        self._order = selected
        self._indices = np.sort(selected)
        self._anchors = self._point_cloud[self._indices].copy()
        self._voronoi_partition()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _voronoi_partition(self) -> None:
        """Assign every point to its nearest anchor (Voronoi cells)."""
        dists = cdist(self._point_cloud, self._anchors)
        cells = dists.argmin(axis=1)
        self._cluster_indices = {
            i: np.where(cells == i)[0] for i in range(self.n_anchors)
        }


__all__ = ['Anchor', 'AnchorStrategy']

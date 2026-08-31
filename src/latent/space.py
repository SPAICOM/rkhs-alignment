"""Latent space of one encoder on one dataset, with anchor handling.

The pipeline implemented here follows the relative-representation (RR)
construction: a fixed, analytic projector ``P`` built from an anchor set
maps each latent ``x`` to its relative representation ``r = P x``
(cosine or inner-product similarities against the anchors). Anchors can
be derived on this space with any :class:`src.anchors.Anchor` strategy,
imported from shared sample indices, or *injected* from another agent's
cluster structure (same observations, different encoder).
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np

from ..alignment import LatentScaler, parseval_frame
from ..anchors import Anchor, AnchorStrategy
from ..decoder import Decoder, MLPDecoder, TimmDecoder
from .semasia import (
    DEFAULT_ORG,
    DEFAULT_PREFIX,
    EMBEDDING_COLUMN,
    load_semasia_split,
)

log = logging.getLogger(__name__)

NormalizeMethod = Literal['standard', 'center', 'l2', 'whiten']
SimilarityMode = Literal['cosine', 'inner']
DecoderKind = Literal['linear', 'mlp']


class LatentSpace:
    """Latent representations of one model for one dataset split.

    Parameters
    ----------
    latent : np.ndarray, shape (n_points, n_features)
        The latent point cloud.
    extras : dict[str, np.ndarray], optional
        Per-sample side information (e.g. ``'label'``, ``'id'``), each
        of shape ``(n_points, ...)``.
    model_name : str, optional
        Identifier of the encoder that produced the latents.
    dataset : str, optional
        Name of the source dataset.
    split : str, optional
        Source split name.
    seed : int, default=42
        Random seed for reproducible subsampling / anchor extraction.
    """

    def __init__(
        self,
        latent: np.ndarray,
        extras: dict[str, np.ndarray] | None = None,
        model_name: str | None = None,
        dataset: str | None = None,
        split: str | None = None,
        seed: int = 42,
    ) -> None:
        self._latent = np.asarray(latent, dtype=np.float32)
        if self._latent.ndim != 2:
            raise ValueError('latent must be 2-dimensional.')

        self._extras: dict[str, np.ndarray] = {}
        if extras is not None:
            for name, arr in extras.items():
                arr = np.asarray(arr)
                if arr.shape[0] != self._latent.shape[0]:
                    raise ValueError(
                        f'extras[{name!r}] has {arr.shape[0]} rows, '
                        f'expected {self._latent.shape[0]}.'
                    )
                self._extras[name] = arr

        self.model_name = model_name
        self.dataset = dataset
        self.split = split
        self._seed = seed

        self._anchor_obj: Anchor | None = None
        self._anchors: np.ndarray | None = None

        self._decoder: Decoder | None = None

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_semasia(
        cls,
        dataset: str,
        model: str,
        split: str = 'train',
        org: str = DEFAULT_ORG,
        prefix: str = DEFAULT_PREFIX,
        embedding_column: str = EMBEDDING_COLUMN,
        cache_dir: str | None = None,
        seed: int = 42,
    ) -> LatentSpace:
        """Load one model's latent space from the SEMASIA Hub collection.

        Parameters
        ----------
        dataset : str
            SEMASIA benchmark name (e.g. ``'cifar10'``).
        model : str
            Model config name (timm identifier).
        split : str, default='train'
            Dataset split.
        org, prefix, embedding_column, cache_dir
            Forwarded to :func:`src.latent.semasia.load_semasia_split`.
        seed : int, default=42
            Random seed of the resulting instance.

        Returns
        -------
        LatentSpace
        """
        latent, extras = load_semasia_split(
            dataset=dataset,
            model=model,
            split=split,
            org=org,
            prefix=prefix,
            embedding_column=embedding_column,
            cache_dir=cache_dir,
        )
        return cls(
            latent=latent,
            extras=extras,
            model_name=model,
            dataset=dataset,
            split=split,
            seed=seed,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def latent(self) -> np.ndarray:
        """Latent array, shape ``(n_points, n_features)``."""
        return self._latent

    @property
    def extras(self) -> dict[str, np.ndarray]:
        """Per-sample side information."""
        return self._extras

    @property
    def labels(self) -> np.ndarray | None:
        """Convenience accessor for ``extras['label']`` (or ``None``)."""
        return self._extras.get('label')

    @property
    def n_points(self) -> int:
        """Number of points."""
        return self._latent.shape[0]

    @property
    def dim(self) -> int:
        """Latent dimensionality."""
        return self._latent.shape[1]

    @property
    def seed(self) -> int:
        """Random seed."""
        return self._seed

    @property
    def anchor(self) -> Anchor | None:
        """The fitted :class:`Anchor` object, when anchors were derived
        here (``None`` if anchors were set from a raw matrix)."""
        return self._anchor_obj

    @property
    def anchors(self) -> np.ndarray | None:
        """Anchor matrix ``(K, n_features)`` or ``None`` if unset."""
        return self._anchors

    @property
    def n_anchors(self) -> int:
        """Number of anchors (0 if unset)."""
        return 0 if self._anchors is None else self._anchors.shape[0]

    def __repr__(self) -> str:
        name = self.model_name or 'unnamed'
        return (
            f'LatentSpace(model={name!r}, dataset={self.dataset!r}, '
            f'split={self.split!r}, n_points={self.n_points}, '
            f'dim={self.dim}, n_anchors={self.n_anchors})'
        )

    # ------------------------------------------------------------------
    # Subsetting / preprocessing
    # ------------------------------------------------------------------

    def select(self, indices: np.ndarray) -> LatentSpace:
        """Return a new instance restricted to the given row indices.

        Using the *same* index array on every agent of a network keeps
        the point clouds aligned sample-by-sample.

        Parameters
        ----------
        indices : np.ndarray
            Row indices into the latent array.

        Returns
        -------
        LatentSpace
            New instance (anchors are not carried over).
        """
        indices = np.asarray(indices)
        return LatentSpace(
            latent=self._latent[indices],
            extras={k: v[indices] for k, v in self._extras.items()},
            model_name=self.model_name,
            dataset=self.dataset,
            split=self.split,
            seed=self._seed,
        )

    def subsample(self, n_points: int, seed: int | None = None) -> LatentSpace:
        """Randomly subsample ``n_points`` rows.

        With the same ``seed`` and the same original ``n_points`` this
        draws identical indices for every agent, preserving cross-agent
        alignment.

        Parameters
        ----------
        n_points : int
            Number of rows to keep.
        seed : int, optional
            Random seed (instance seed when ``None``).

        Returns
        -------
        LatentSpace
        """
        if n_points >= self.n_points:
            return self.select(np.arange(self.n_points))
        rng = np.random.default_rng(self._seed if seed is None else seed)
        idx = np.sort(rng.choice(self.n_points, size=n_points, replace=False))
        return self.select(idx)

    def biased_subsample(
        self,
        n_points: int,
        target_classes: list | np.ndarray | None = None,
        shift_strength: float = 0.0,
        seed: int | None = None,
    ) -> LatentSpace:
        """Subsample with a class-distribution shift toward ``target_classes``.

        Each label ``y`` is drawn i.i.d. from a mixture of two uniform
        distributions -- uniform over ``target_classes`` and uniform over
        all classes present in this space::

            P(y) = shift_strength * Unif(target_classes)
                   + (1 - shift_strength) * Unif(all_classes)

        so target classes are over-represented (by ``shift_strength``)
        relative to their natural frequency, while every class remains
        reachable. For each label drawn this way, one row with that label
        is drawn uniformly at random, without replacement within a class
        (so target classes cannot repeat rows just because they are drawn
        more often).

        ``target_classes=None``/empty or ``shift_strength<=0`` falls back
        to plain uniform :meth:`subsample`.

        Parameters
        ----------
        n_points : int
            Number of rows to keep.
        target_classes : list | np.ndarray, optional
            Labels to bias sampling toward. Must be a subset of the
            labels present in ``self.labels``.
        shift_strength : float, default=0.0
            Mixture weight ``s`` in ``[0, 1]``; ``0`` is uniform, ``1``
            draws exclusively from ``target_classes``.
        seed : int, optional
            Random seed (instance seed when ``None``).

        Returns
        -------
        LatentSpace
        """
        if (
            target_classes is None
            or len(target_classes) == 0
            or shift_strength <= 0
        ):
            return self.subsample(n_points, seed=seed)
        if self.labels is None:
            raise ValueError(
                'biased_subsample() requires labels '
                "(this LatentSpace has no 'label' extra)."
            )
        if n_points >= self.n_points:
            return self.select(np.arange(self.n_points))

        labels = self.labels
        all_classes = np.unique(labels)
        target_classes = np.asarray(list(target_classes))
        missing = np.setdiff1d(target_classes, all_classes)
        if missing.size:
            raise ValueError(
                f'target_classes {missing.tolist()} not present in this '
                f"space's labels (available: {all_classes.tolist()})."
            )

        s = float(shift_strength)
        n_all = all_classes.shape[0]
        n_target = target_classes.shape[0]
        probs = np.full(n_all, (1.0 - s) / n_all)
        target_mask = np.isin(all_classes, target_classes)
        probs[target_mask] += s / n_target
        probs /= probs.sum()  # defensive renormalization

        rng = np.random.default_rng(self._seed if seed is None else seed)
        drawn_classes = rng.choice(all_classes, size=n_points, p=probs)
        unique_drawn, counts = np.unique(drawn_classes, return_counts=True)

        selected = []
        for c, cnt in zip(unique_drawn, counts):
            pool = np.where(labels == c)[0]
            if cnt > pool.shape[0]:
                log.warning(
                    'biased_subsample: class %r requested %d rows but '
                    'only %d are available; capping.',
                    c,
                    int(cnt),
                    pool.shape[0],
                )
                cnt = pool.shape[0]
            selected.append(rng.choice(pool, size=cnt, replace=False))

        idx = np.sort(np.concatenate(selected))
        return self.select(idx)

    def normalize(
        self,
        method: NormalizeMethod = 'standard',
        inplace: bool = False,
        eps: float = 1e-6,
    ) -> np.ndarray | LatentSpace:
        """Normalize the point cloud.

        Parameters
        ----------
        method : {'standard', 'center', 'l2', 'whiten'}
            Feature z-scoring, mean-centering, row L2 normalization, or
            whitening (mean-centered, unit feature covariance).
        inplace : bool, default=False
            Modify ``self`` and return it, instead of returning the
            normalized array.
        eps : float, default=1e-6
            Covariance ridge of ``method='whiten'``, relative to the mean
            feature variance.

        Returns
        -------
        np.ndarray | LatentSpace

        Notes
        -----
        This is a one-shot transform: it estimates its statistics on
        *this* point cloud and forgets them. To standardize a train split
        and reuse the very same transform on a test split -- which is
        what any alignment method needs -- fit a
        :class:`src.alignment.LatentScaler` instead.
        """
        X = self._latent
        if method == 'l2':
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            out = X / norms
        elif method in ('standard', 'center', 'pca', 'pga', 'whiten'):
            out = LatentScaler(method, eps=eps).fit_transform(X)
        else:
            raise ValueError(f'Unknown normalize method {method!r}')
        out = out.astype(np.float32)
        if inplace:
            self._latent = out
            return self
        return out

    # ------------------------------------------------------------------
    # Anchors
    # ------------------------------------------------------------------

    def derive_anchors(
        self,
        strategy: AnchorStrategy = 'kmeans',
        n_anchors: int | None = None,
        set_as_anchors: bool = True,
        **fit_kwargs,
    ) -> Anchor:
        """Derive anchors from this latent space with a given strategy.

        Parameters
        ----------
        strategy : AnchorStrategy, default='kmeans'
            Extraction strategy (see :class:`src.anchors.Anchor`).
        n_anchors : int, optional
            Number of anchors.
        set_as_anchors : bool, default=True
            Store the result as this space's anchor set.
        **fit_kwargs
            Forwarded to :meth:`src.anchors.Anchor.fit`. For the
            ``'stratified'`` strategy, ``labels`` defaults to
            ``self.labels``.

        Returns
        -------
        Anchor
            The fitted anchor object (exposes ``indices`` /
            ``cluster_indices`` for sharing with other agents).
        """
        if strategy == 'stratified':
            fit_kwargs.setdefault('labels', self.labels)

        anchor = Anchor(self._latent, strategy=strategy, seed=self._seed)
        anchor.fit(n_anchors=n_anchors, **fit_kwargs)

        if set_as_anchors:
            self._anchor_obj = anchor
            self._anchors = anchor.anchors
        return anchor

    def set_anchors(self, anchors: Anchor | np.ndarray) -> LatentSpace:
        """Set the anchor set of this space.

        Parameters
        ----------
        anchors : Anchor | np.ndarray
            A fitted :class:`Anchor` object or a raw anchor matrix of
            shape ``(K, n_features)``.

        Returns
        -------
        LatentSpace
            ``self``.
        """
        if isinstance(anchors, Anchor):
            self._anchor_obj = anchors
            self._anchors = anchors.anchors
        else:
            anchors = np.asarray(anchors, dtype=np.float32)
            if anchors.ndim != 2 or anchors.shape[1] != self.dim:
                raise ValueError(
                    f'Anchor matrix must have shape (K, {self.dim}), '
                    f'got {anchors.shape}.'
                )
            self._anchor_obj = None
            self._anchors = anchors
        return self

    def anchors_from_indices(self, indices: np.ndarray) -> np.ndarray:
        """Embed shared anchor *inputs* with this agent's encoder.

        The indices refer to rows of the (aligned) dataset, typically
        produced by an index-based anchor strategy on another agent.

        Parameters
        ----------
        indices : np.ndarray
            Anchor sample indices, shape ``(K,)``.

        Returns
        -------
        np.ndarray
            The anchor matrix ``(K, n_features)``, also stored on the
            instance.
        """
        indices = np.asarray(indices)
        anchors = self._latent[indices].copy()
        self._anchor_obj = None
        self._anchors = anchors
        return anchors

    def anchors_from_clusters(
        self,
        cluster_indices: dict[int, np.ndarray],
        n_samples: int | None = None,
        seed: int | None = None,
    ) -> np.ndarray:
        """Compute *injected* anchors from another agent's clusters.

        Given the cluster structure found on another agent's latent
        space (same observations, different encoder), each anchor is
        the centroid of the corresponding cluster members in *this*
        space — the injected-prototype construction of SEMASIA.

        Parameters
        ----------
        cluster_indices : dict[int, np.ndarray]
            Anchor index -> observation indices, e.g.
            ``Anchor.cluster_indices``.
        n_samples : int | None, default=None
            Members averaged per cluster (all when ``None``).
        seed : int, optional
            Seed for member subsampling.

        Returns
        -------
        np.ndarray
            The anchor matrix ``(K, n_features)``, also stored on the
            instance.
        """
        seed = self._seed if seed is None else seed
        n_proto = len(cluster_indices)
        anchors = np.empty((n_proto, self.dim), dtype=np.float32)

        for i, obs in cluster_indices.items():
            members = self._latent[np.asarray(obs)]
            if n_samples is not None and n_samples < members.shape[0]:
                rng = np.random.default_rng(seed + i)
                members = members[
                    rng.choice(members.shape[0], size=n_samples, replace=False)
                ]
            anchors[i] = members.mean(axis=0)

        self._anchor_obj = None
        self._anchors = anchors
        return anchors

    # ------------------------------------------------------------------
    # Decoder (private classifier on raw latents)
    # ------------------------------------------------------------------

    @property
    def decoder(self) -> Decoder | None:
        """Decoder fitted on raw train latents (or ``None``)."""
        return self._decoder

    def set_decoder(self, decoder: Decoder) -> LatentSpace:
        """Attach an already fitted decoder (e.g. from a checkpoint).

        Parameters
        ----------
        decoder : Decoder
            A fitted decoder whose ``input_dim`` matches this space.

        Returns
        -------
        LatentSpace
            ``self``.
        """
        input_dim = getattr(decoder, 'input_dim', None)
        if input_dim is not None and input_dim != self.dim:
            raise ValueError(
                f'Decoder input_dim={input_dim} does not match this '
                f'latent space (dim={self.dim}).'
            )
        self._decoder = decoder
        return self

    def fit_decoder(
        self,
        labels: np.ndarray | None = None,
        kind: DecoderKind = 'linear',
        n_classes: int = 10,
        l2: float = 1e-6,
        input_dim: int | None = None,
        **mlp_kwargs,
    ) -> Decoder:
        """Fit this agent's private decoder on its raw latents.

        Both decoder kinds operate on the full-dimensional raw latent
        space — the same space the original timm classifier head sees —
        so that semantic communication accuracy is measured with the
        actual latent geometry the model was designed for:

        - ``kind='linear'`` : :class:`~src.decoder.TimmDecoder`, a
          closed-form linear probe (ridge regression) mirroring the
          timm classifier-head contract;
        - ``kind='mlp'``    : :class:`~src.decoder.MLPDecoder`, a
          two-layer MLP trained with Adam and early stopping (the
          dual-sim neural evaluation pipeline).

        Parameters
        ----------
        labels : np.ndarray, optional
            Training labels.  Defaults to ``self.labels``.
        kind : {'linear', 'mlp'}, default='linear'
            Decoder family.
        n_classes : int, default=10
            Number of output classes (linear only; the MLP infers it
            from the labels).
        l2 : float, default=1e-6
            Ridge regularisation strength (linear only).
        input_dim : int, optional
            Latent dimensionality (defaults to this space's ``dim``).
        **mlp_kwargs
            Forwarded to :class:`~src.decoder.MLPDecoder` (e.g.
            ``hidden_dim``, ``lr``, ``max_epochs``, ``patience``,
            ``batch_size``, ``val_fraction``, ``device``, ``seed``).

        Returns
        -------
        Decoder
            The fitted decoder, also stored as ``self._decoder``.
        """
        if labels is None:
            labels = self.labels
        if labels is None:
            raise ValueError(
                'No labels available. Pass labels or ensure the '
                'LatentSpace has a "label" extra.'
            )

        if kind == 'linear':
            self._decoder = TimmDecoder(
                model_name=self.model_name,
                input_dim=input_dim or self.dim,
                n_classes=n_classes,
                l2=l2,
            )
        elif kind == 'mlp':
            mlp_kwargs.setdefault('seed', self._seed)
            self._decoder = MLPDecoder(
                model_name=self.model_name,
                input_dim=input_dim or self.dim,
                **mlp_kwargs,
            )
        else:
            raise ValueError(f'Unknown decoder kind {kind!r}')
        self._decoder.fit(self._latent, labels)
        return self._decoder

    # ------------------------------------------------------------------
    # Relative representation
    # ------------------------------------------------------------------

    def projector(
        self,
        mode: SimilarityMode = 'cosine',
        whiten: bool = False,
    ) -> np.ndarray:
        """Analytic anchor projector ``P``, shape ``(K, n_features)``.

        - ``mode='inner'``  : raw anchor matrix, ``r = A x``.
        - ``mode='cosine'`` : row-normalized anchors; the projection is
          applied to L2-normalized latents, so ``r_k`` is the cosine
          similarity to anchor ``k``.
        - ``whiten=True``   : additionally replace ``P`` by its Parseval
          frame (orthonormal rows), removing the anchor set's own
          redundancy. It is applied *after* the mode-specific
          normalization, so the two options compose; the default keeps
          the raw projector and lets the downstream map absorb the
          anchor geometry instead. Requires ``n_anchors <= dim``.

        Returns
        -------
        np.ndarray
        """
        if self._anchors is None:
            raise ValueError(
                'No anchors set. Call derive_anchors(), set_anchors(), '
                'anchors_from_indices() or anchors_from_clusters() first.'
            )
        if mode not in ('cosine', 'inner'):
            raise ValueError(f'Unknown similarity mode {mode!r}')

        P = self._anchors.astype(np.float32)
        if mode == 'cosine':
            norms = np.linalg.norm(P, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            P = P / norms
        if whiten:
            P = parseval_frame(P).astype(np.float32)
        return P

    def relative(
        self,
        X: np.ndarray | None = None,
        mode: SimilarityMode = 'cosine',
        whiten: bool = False,
    ) -> np.ndarray:
        """Project latents onto the anchors (relative representation).

        Parameters
        ----------
        X : np.ndarray, optional
            Latents to project, shape ``(n, n_features)``. Defaults to
            this space's own latents. Passing another split's latents
            (e.g. test) reuses the anchors fitted here.
        mode : {'cosine', 'inner'}, default='cosine'
            Similarity used against the anchors.
        whiten : bool, default=False
            Use the Parseval-whitened projector.

        Returns
        -------
        np.ndarray
            Relative representations ``r = P x``, shape ``(n, K)``.
        """
        P = self.projector(mode=mode, whiten=whiten)
        X = self._latent if X is None else np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != self.dim:
            raise ValueError(
                f'X must have shape (n, {self.dim}), got {X.shape}.'
            )
        if mode == 'cosine':
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            X = X / norms
        return (X @ P.T).astype(np.float32)


__all__ = ['DecoderKind', 'LatentSpace', 'NormalizeMethod', 'SimilarityMode']

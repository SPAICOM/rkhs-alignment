"""Synthetic multi-agent latent spaces with a known shared ground truth.

Generates the regime the alignment methods are designed for: one shared,
class-clustered "ground" latent space observed by each agent through its
own map. Each agent applies, in order,

1. a per-agent anisotropic stretch of the ground coordinates -- so anchors
   are rendered *imperfectly* across agents;
2. a random orthogonal embedding into its own (larger) latent dimension
   and a per-agent global scale -- the part orthogonal Procrustes can
   undo exactly;
3. a monotone element-wise power distortion ``sign(x)|x|^p`` -- a smooth
   non-linearity that no orthogonal or affine map can absorb, and which
   the RKHS residual stage is meant to pick up;
4. isotropic Gaussian noise.

Because every agent observes the *same* ground points, row ``i`` is the
same underlying sample in every agent's space, exactly like the SEMASIA
benchmark. That makes this a drop-in, offline stand-in for it -- handy for
checking a pipeline end to end without downloading gigabytes of latents.
"""

from __future__ import annotations

import numpy as np

from .space import LatentSpace

__all__ = ['make_synthetic_agents']


def _agent_name(index: int, n_agents: int) -> str:
    width = max(2, len(str(n_agents - 1)))
    return f'agent_{index:0{width}d}'


def make_synthetic_agents(
    n_agents: int = 8,
    n_train: int = 2000,
    n_test: int = 1000,
    n_classes: int = 10,
    ground_dim: int = 48,
    agent_dims: list[int] | None = None,
    scale_range: tuple[float, float] = (0.5, 2.0),
    noise_std: float = 0.5,
    class_sep: float = 1.0,
    anisotropy: float = 0.0,
    power_range: tuple[float, float] = (1.0, 1.0),
    seed: int = 42,
    dataset: str = 'synthetic',
) -> dict[str, dict[str, LatentSpace]]:
    """Build ``n_agents`` paired train/test latent spaces.

    Parameters
    ----------
    n_agents : int, default=8
        Number of agents (encoders) to simulate.
    n_train, n_test : int
        Samples per split. Row ``i`` is the same ground point for every
        agent within a split.
    n_classes : int, default=10
        Number of classes in the shared ground space.
    ground_dim : int, default=48
        Dimensionality of the shared ground latent space.
    agent_dims : list[int], optional
        Per-agent latent dimensionality; each entry must be ``>=
        ground_dim`` so the embedding can stay isometric. Cycled if
        shorter than ``n_agents``; defaults to all ``ground_dim``.
    scale_range : tuple[float, float], default=(0.5, 2.0)
        Uniform range of the per-agent global scale.
    noise_std : float, default=0.5
        Standard deviation of the additive observation noise.
    class_sep : float, default=1.0
        Standard deviation of the class centroids (relative to the unit
        within-class spread), i.e. how separable the ground space is.
    anisotropy : float, default=0.0
        Log-normal spread of the per-agent, per-feature stretch. ``0``
        makes every agent's rendering of the ground space isometric.
    power_range : tuple[float, float], default=(1.0, 1.0)
        Uniform range of the per-agent monotone exponent ``p`` in
        ``sign(x)|x|^p``. ``(1.0, 1.0)`` disables the non-linearity.
    seed : int, default=42
        Master seed; every agent derives its parameters from it.
    dataset : str, default='synthetic'
        Name recorded on the produced :class:`LatentSpace` objects.

    Returns
    -------
    dict[str, dict[str, LatentSpace]]
        ``{agent_name: {'train': LatentSpace, 'test': LatentSpace}}``.
    """
    rng = np.random.default_rng(seed)

    if agent_dims is None:
        agent_dims = [ground_dim] * n_agents
    agent_dims = [
        int(agent_dims[i % len(agent_dims)]) for i in range(n_agents)
    ]
    too_small = [d for d in agent_dims if d < ground_dim]
    if too_small:
        raise ValueError(
            f'agent_dims must all be >= ground_dim={ground_dim}, got '
            f'{sorted(set(too_small))}.'
        )

    # --- shared ground space -----------------------------------------
    centroids = class_sep * rng.normal(size=(n_classes, ground_dim))
    splits: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split, n in (('train', n_train), ('test', n_test)):
        y = rng.integers(0, n_classes, size=n)
        splits[split] = (centroids[y] + rng.normal(size=(n, ground_dim)), y)

    # --- per-agent observation models --------------------------------
    agents: dict[str, dict[str, LatentSpace]] = {}
    for i, dim in enumerate(agent_dims):
        stretch = np.exp(anisotropy * rng.normal(size=ground_dim))
        basis = np.linalg.qr(rng.normal(size=(dim, dim)))[0][:, :ground_dim]
        scale = rng.uniform(*scale_range)
        power = rng.uniform(*power_range)

        name = _agent_name(i, n_agents)
        agents[name] = {}
        for split, (ground, y) in splits.items():
            X = scale * ((ground * stretch) @ basis.T)
            X = np.sign(X) * np.abs(X) ** power
            X = X + noise_std * rng.normal(size=X.shape)
            agents[name][split] = LatentSpace(
                latent=X,
                extras={'label': y, 'id': np.arange(y.shape[0])},
                model_name=name,
                dataset=dataset,
                split=split,
                seed=seed,
            )

    return agents

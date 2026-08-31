"""One entry point for loading a set of paired agent latent spaces.

Every alignment experiment needs the same thing: for each agent (encoder),
a train split used as *calibration* data and a test split used for
evaluation, with rows aligned sample-by-sample **across agents** so that
row ``i`` is the same underlying input everywhere. That property is what
makes paired alignment possible at all, and it is enforced here rather
than left to each caller:

- SEMASIA splits are sorted by their ``id`` column on load, so a single
  index array drawn once per split and applied to every agent keeps the
  correspondence exact.
- Synthetic agents observe the same ground points by construction.

Sources
-------
``'semasia'``    latents pulled from the Hugging Face Hub collection.
``'synthetic'``  offline agents from :func:`make_synthetic_agents`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

from .semasia import DEFAULT_ORG, DEFAULT_PREFIX
from .space import LatentSpace
from .synthetic import make_synthetic_agents

log = logging.getLogger(__name__)

__all__ = ['load_agents']

AgentSpaces = dict[str, dict[str, LatentSpace]]


def load_agents(
    source: str,
    models: Sequence[str] | None = None,
    seed: int = 42,
    **kwargs: Any,
) -> AgentSpaces:
    """Load one train/test :class:`LatentSpace` pair per agent.

    Parameters
    ----------
    source : {'semasia', 'synthetic'}
        Where the latents come from.
    models : Sequence[str], optional
        Agents to keep. For ``'semasia'`` these are timm config names and
        only those configs are downloaded; for ``'synthetic'`` they
        filter the generated agents. ``None`` keeps everything the source
        offers.
    seed : int, default=42
        Seed of the aligned sub-sampling (and of the synthetic
        generator).
    **kwargs
        Source-specific options: see :func:`_load_semasia_agents` and
        :func:`~src.latent.synthetic.make_synthetic_agents`.

    Returns
    -------
    dict[str, dict[str, LatentSpace]]
        ``{agent_name: {'train': LatentSpace, 'test': LatentSpace}}``.
    """
    match source:
        case 'semasia':
            return _load_semasia_agents(models=models, seed=seed, **kwargs)
        case 'synthetic':
            return _load_synthetic_agents(models=models, seed=seed, **kwargs)
        case _:
            raise ValueError(
                f"Unknown data source {source!r}. Supported: 'semasia', "
                "'synthetic'."
            )


def _load_semasia_agents(
    dataset: str = 'cifar10',
    models: Sequence[str] | None = None,
    org: str = DEFAULT_ORG,
    prefix: str = DEFAULT_PREFIX,
    train_split: str = 'train',
    test_split: str = 'test',
    cache_dir: str | None = None,
    n_train: int | None = None,
    n_test: int | None = None,
    seed: int = 42,
) -> AgentSpaces:
    """Load SEMASIA latents for the requested model configs.

    Parameters
    ----------
    dataset : str, default='cifar10'
        SEMASIA benchmark name.
    models : Sequence[str], optional
        Model configs to load. Required: downloading *every* config of a
        benchmark is never what an alignment experiment wants.
    org, prefix, cache_dir
        Forwarded to :meth:`LatentSpace.from_semasia`.
    train_split, test_split : str
        Split names.
    n_train, n_test : int, optional
        Aligned random sub-sample sizes (``None`` keeps every row).
    seed : int, default=42
        Sub-sampling seed.

    Returns
    -------
    dict[str, dict[str, LatentSpace]]
    """
    if not models:
        raise ValueError(
            "source='semasia' requires an explicit `models` list (the "
            'benchmark holds hundreds of configs).'
        )

    agents: AgentSpaces = {}
    for model in models:
        log.info('Loading SEMASIA latents for %s (%s).', model, dataset)
        agents[model] = {
            'train': LatentSpace.from_semasia(
                dataset=dataset,
                model=model,
                split=train_split,
                org=org,
                prefix=prefix,
                cache_dir=cache_dir,
                seed=seed,
            ),
            'test': LatentSpace.from_semasia(
                dataset=dataset,
                model=model,
                split=test_split,
                org=org,
                prefix=prefix,
                cache_dir=cache_dir,
                seed=seed,
            ),
        }

    return _subsample_aligned(
        agents, {'train': n_train, 'test': n_test}, seed=seed
    )


def _load_synthetic_agents(
    models: Sequence[str] | None = None,
    seed: int = 42,
    **kwargs: Any,
) -> AgentSpaces:
    """Generate synthetic agents, optionally keeping only some of them."""
    kwargs.pop('source', None)
    agents = make_synthetic_agents(seed=seed, **kwargs)
    if models:
        missing = [m for m in models if m not in agents]
        if missing:
            raise ValueError(
                f'Unknown synthetic agents {missing}; generated agents are '
                f'{list(agents)[:5]}... ({len(agents)} total).'
            )
        agents = {m: agents[m] for m in models}
    return agents


def _subsample_aligned(
    agents: AgentSpaces,
    sizes: dict[str, int | None],
    seed: int = 42,
) -> AgentSpaces:
    """Apply one shared random index array per split to every agent.

    Drawing the indices once -- rather than letting each agent subsample
    on its own -- is what keeps the cross-agent correspondence exact.
    """
    for split, n_keep in sizes.items():
        counts = {
            name: spaces[split].n_points for name, spaces in agents.items()
        }
        n_available = min(counts.values())
        if len(set(counts.values())) > 1:
            raise ValueError(
                f'Agents disagree on the size of the {split!r} split '
                f'({counts}); the rows cannot be paired.'
            )
        if n_keep is None or n_keep >= n_available:
            continue
        idx = np.sort(
            np.random.default_rng(seed).choice(
                n_available, size=n_keep, replace=False
            )
        )
        for spaces in agents.values():
            spaces[split] = spaces[split].select(idx)
        log.info(
            'Sub-sampled the %r split to %d of %d aligned rows.',
            split,
            n_keep,
            n_available,
        )
    return agents

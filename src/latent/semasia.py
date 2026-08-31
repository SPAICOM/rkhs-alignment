"""Loading helpers for the SEMASIA collection on the Hugging Face Hub.

SEMASIA (``SPAICOM/semasia-datasets``) hosts precomputed latent
representations of ~1700 timm vision models over standard image
benchmarks. Each Hub repository is one benchmark, each *config* is one
model, and each parquet row carries::

    id          : row index aligned with the source dataset split
    model_name  : timm identifier of the encoder (constant per config)
    embedding   : the latent vector
    <extras>    : original dataset fields (e.g. ``label``)

Usage
-----
>>> emb, extras = load_semasia_split('cifar10', 'resnet50.a1_in1k')
>>> emb.shape, extras['label'].shape
((50000, 2048), (50000,))
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from datasets import (
    DownloadMode,
    get_dataset_config_names,
    load_dataset,
)
from datasets.config import HF_DATASETS_CACHE

DEFAULT_ORG = 'spaicom-lab'
DEFAULT_PREFIX = 'semasia-'
EMBEDDING_COLUMN = 'embedding'

# Columns that are metadata rather than per-sample payload.
_NON_EXTRA_COLUMNS = (EMBEDDING_COLUMN, 'model_name')


def semasia_repo_id(
    dataset: str,
    org: str = DEFAULT_ORG,
    prefix: str = DEFAULT_PREFIX,
) -> str:
    """Build the Hub repo id of a SEMASIA benchmark.

    Parameters
    ----------
    dataset : str
        Benchmark name (e.g. ``'cifar10'``, ``'mnist'``,
        ``'tiny-imagenet'``).
    org : str
        Hub organisation hosting the collection.
    prefix : str
        Repository name prefix.

    Returns
    -------
    str
        Full repo id, e.g. ``'spaicom-lab/semasia-cifar10'``.
    """
    return f'{org}/{prefix}{dataset}'


def is_semasia_cached(
    dataset: str,
    model: str,
    split: str = 'train',
    org: str = DEFAULT_ORG,
    prefix: str = DEFAULT_PREFIX,
    cache_dir: str | None = None,
) -> bool:
    """Check if a SEMASIA split is already cached locally (no network call).

    Parameters
    ----------
    dataset : str
        Benchmark name (e.g. ``'cifar10'``).
    model : str
        Model config name (timm identifier).
    split : str, default='train'
        Dataset split (``'train'`` or ``'test'``).
    org, prefix : str
        Hub organisation and repo-name prefix.
    cache_dir : str, optional
        Hugging Face datasets cache directory.

    Returns
    -------
    bool
        ``True`` if the arrow file for this split exists in the local cache.
    """
    repo_id = semasia_repo_id(dataset, org=org, prefix=prefix)
    repo_path = repo_id.replace('/', '___')
    base = Path(cache_dir) if cache_dir else Path(HF_DATASETS_CACHE)
    model_dir = base / repo_path / model
    if not model_dir.exists():
        return False
    arrow_name = f'{prefix}{dataset}-{split}.arrow'
    for version_dir in model_dir.iterdir():
        if not version_dir.is_dir():
            continue
        for hash_dir in version_dir.iterdir():
            if not hash_dir.is_dir():
                continue
            if (hash_dir / arrow_name).exists():
                return True
    return False


def load_semasia_split(
    dataset: str,
    model: str,
    split: str = 'train',
    org: str = DEFAULT_ORG,
    prefix: str = DEFAULT_PREFIX,
    embedding_column: str = EMBEDDING_COLUMN,
    cache_dir: str | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Load one model's latents for one split of a SEMASIA benchmark.

    Only the parquet files of the requested model config are downloaded.
    Rows are sorted by the ``id`` column (when present) so that the same
    row index refers to the same input sample across *all* models of the
    benchmark — the property that makes shared anchors possible.

    Parameters
    ----------
    dataset : str
        Benchmark name (e.g. ``'cifar10'``).
    model : str
        Model config name (timm identifier, e.g.
        ``'resnet50.a1_in1k'``).
    split : str, default='train'
        Dataset split (``'train'`` or ``'test'``).
    org, prefix : str
        Hub organisation and repo-name prefix.
    embedding_column : str, default='embedding'
        Name of the latent column.
    cache_dir : str, optional
        Optional Hugging Face datasets cache directory.

    Returns
    -------
    embeddings : np.ndarray, shape (n_points, n_features)
        Latent representations, float32.
    extras : dict[str, np.ndarray]
        All remaining per-sample columns (e.g. ``'id'``, ``'label'``).
    """
    repo_id = semasia_repo_id(dataset, org=org, prefix=prefix)
    cached = is_semasia_cached(
        dataset,
        model,
        split=split,
        org=org,
        prefix=prefix,
        cache_dir=cache_dir,
    )
    download_mode = (
        DownloadMode.REUSE_CACHE_IF_EXISTS
        if cached
        else DownloadMode.REUSE_DATASET_IF_EXISTS
    )
    ds = load_dataset(
        repo_id,
        model,
        split=split,
        cache_dir=cache_dir,
        download_mode=download_mode,
    )

    # `datasets` >= 4 returns a lazy Column; np.asarray materialises it.
    embeddings = np.asarray(ds.with_format('numpy')[embedding_column])
    if embeddings.dtype == object:  # ragged fallback
        embeddings = np.stack(list(embeddings))
    embeddings = embeddings.astype(np.float32, copy=False)

    extras = {
        name: np.asarray(ds.with_format('numpy')[name])
        for name in ds.column_names
        if name not in (embedding_column, *_NON_EXTRA_COLUMNS)
    }

    if 'id' in extras:
        order = np.argsort(extras['id'], kind='stable')
        embeddings = embeddings[order]
        extras = {name: arr[order] for name, arr in extras.items()}

    return embeddings, extras


def available_models(
    dataset: str,
    org: str = DEFAULT_ORG,
    prefix: str = DEFAULT_PREFIX,
) -> list[str]:
    """List the model configs available for a SEMASIA benchmark.

    Parameters
    ----------
    dataset : str
        Benchmark name (e.g. ``'cifar10'``).
    org, prefix : str
        Hub organisation and repo-name prefix.

    Returns
    -------
    list[str]
        Model config names (timm identifiers).
    """
    return get_dataset_config_names(
        semasia_repo_id(dataset, org=org, prefix=prefix)
    )


__all__ = [
    'DEFAULT_ORG',
    'DEFAULT_PREFIX',
    'EMBEDDING_COLUMN',
    'available_models',
    'is_semasia_cached',
    'load_semasia_split',
    'semasia_repo_id',
]

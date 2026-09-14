"""Browse the SEMASIA encoder pool with polars.

SEMASIA publishes one parquet config per timm encoder, and the sibling
``spaicom-lab/model-registry`` publishes one metadata row per encoder --
``latent_dim``, ``family``, ``num_parameters``, the pre-training tag
split into fields. This script joins the two, so the pool a benchmark
offers can be read as a table instead of guessed from model names.

The ``cached`` column says whether that config's arrow file is already in
the local Hugging Face cache, i.e. whether listing it in
``config/hydra/data/*.yaml`` costs a download. Nothing is downloaded
here: only the registry parquet and the benchmark's config list are
fetched, both of which are tiny.

Examples
--------
    uv run scripts/explore_semasia.py
    uv run scripts/explore_semasia.py --dataset mnist --cached-only
    uv run scripts/explore_semasia.py --family ViT --min-dim 384
    uv run scripts/explore_semasia.py --sort num_parameters --desc --top 20
    uv run scripts/explore_semasia.py --summary
    uv run scripts/explore_semasia.py --csv /tmp/semasia_cifar10.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.latent.semasia import (
    DEFAULT_ORG,
    DEFAULT_PREFIX,
    available_models,
    is_semasia_cached,
)

REGISTRY_REPO = 'spaicom-lab/semasia-model-registry'
REGISTRY_FILE = 'semasia_model_registry.parquet'

# The registry has 42 columns; these are the ones worth a terminal.
COLUMNS = (
    'model_name',
    'latent_dim',
    'family',
    'macro_family',
    'size',
    'num_parameters',
    'patch_size',
    'input_resolution',
    'pretrain_dataset',
    'pretrain_method',
    'pretrain_ft',
    'cached',
)


def load_registry() -> pl.DataFrame:
    """Read the timm metadata registry as a polars frame.

    Returns
    -------
    pl.DataFrame
        One row per timm model, keyed by ``model_name``.
    """
    path = hf_hub_download(REGISTRY_REPO, REGISTRY_FILE, repo_type='dataset')
    return pl.read_parquet(path)


def benchmark_frame(
    dataset: str,
    split: str = 'train',
    org: str = DEFAULT_ORG,
    prefix: str = DEFAULT_PREFIX,
    cache_dir: str | None = None,
) -> pl.DataFrame:
    """Join a benchmark's config list with the registry metadata.

    Parameters
    ----------
    dataset : str
        Benchmark name (e.g. ``'cifar10'``, ``'mnist'``).
    split : str, default='train'
        Split whose local cache decides the ``cached`` column.
    org, prefix : str
        Hub organisation and repo-name prefix.
    cache_dir : str, optional
        Hugging Face datasets cache directory.

    Returns
    -------
    pl.DataFrame
        One row per encoder published for the benchmark. Encoders absent
        from the registry keep null metadata rather than being dropped.
    """
    models = available_models(dataset, org=org, prefix=prefix)
    configs = pl.DataFrame(
        {
            'model_name': models,
            'cached': [
                is_semasia_cached(
                    dataset,
                    model,
                    split=split,
                    org=org,
                    prefix=prefix,
                    cache_dir=cache_dir,
                )
                for model in models
            ],
        }
    )
    return configs.join(load_registry(), on='model_name', how='left')


def _filtered(df: pl.DataFrame, args: argparse.Namespace) -> pl.DataFrame:
    """Apply the command-line filters to the joined frame."""
    if args.cached_only:
        df = df.filter(pl.col('cached'))
    if args.family:
        df = df.filter(pl.col('family').str.contains(f'(?i){args.family}'))
    if args.macro_family:
        df = df.filter(
            pl.col('macro_family').str.contains(f'(?i){args.macro_family}')
        )
    if args.contains:
        df = df.filter(
            pl.col('model_name').str.contains(f'(?i){args.contains}')
        )
    if args.min_dim is not None:
        df = df.filter(pl.col('latent_dim') >= args.min_dim)
    if args.max_dim is not None:
        df = df.filter(pl.col('latent_dim') <= args.max_dim)
    if args.max_params is not None:
        df = df.filter(pl.col('num_parameters') <= args.max_params)
    return df


def _summary(df: pl.DataFrame) -> pl.DataFrame:
    """Count encoders per latent dimension."""
    return (
        df.group_by('latent_dim')
        .agg(
            pl.len().alias('n_models'),
            pl.col('cached').sum().alias('n_cached'),
            pl.col('family').n_unique().alias('n_families'),
            pl.col('family').unique().sort().str.join(', ').alias('families'),
        )
        .sort('latent_dim', nulls_last=True)
    )


def parse_args() -> argparse.Namespace:
    """Parse the command line."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--dataset', default='cifar10')
    p.add_argument('--split', default='train')
    p.add_argument('--sort', default='latent_dim')
    p.add_argument('--desc', action='store_true')
    p.add_argument('--top', type=int, default=30, help='0 shows all rows')
    p.add_argument('--family', help='substring of `family`, e.g. ViT')
    p.add_argument('--macro-family', help='substring of `macro_family`')
    p.add_argument('--contains', help='substring of the model name')
    p.add_argument('--min-dim', type=int)
    p.add_argument('--max-dim', type=int)
    p.add_argument(
        '--max-params',
        type=int,
        help='cap on `num_parameters`, e.g. 10_000_000',
    )
    p.add_argument('--cached-only', action='store_true')
    p.add_argument(
        '--summary',
        action='store_true',
        help='counts per latent dimension instead of the model list',
    )
    p.add_argument('--csv', help='write the (filtered) table here')
    return p.parse_args()


def main() -> None:
    """Print one view of a benchmark's encoder pool."""
    args = parse_args()
    df = _filtered(benchmark_frame(args.dataset, split=args.split), args)

    table = (
        _summary(df)
        if args.summary
        else df.select(COLUMNS).sort(
            args.sort, descending=args.desc, nulls_last=True
        )
    )

    if args.csv:
        table.write_csv(args.csv)
        print(f'wrote {table.height} rows to {args.csv}')

    shown = table if args.top <= 0 else table.head(args.top)
    with pl.Config(
        tbl_rows=-1, tbl_cols=-1, tbl_width_chars=200, fmt_str_lengths=60
    ):
        print(shown)
    print(
        f'{df.height} encoders in semasia-{args.dataset} after filters '
        f'({df["cached"].sum()} cached locally); showing {shown.height}.'
    )


if __name__ == '__main__':
    main()

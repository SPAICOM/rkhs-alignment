"""Latent space toolkit.

``LatentSpace`` wraps the latent representations produced by one model
on one dataset — typically loaded from the SEMASIA collection on the
Hugging Face Hub (``spaicom-lab/semasia-<dataset>``), or generated
synthetically by :func:`make_synthetic_agents` — and provides anchor
handling plus the relative-representation projection.

:func:`load_agents` is the single entry point used by the experiment
scripts: it returns one train/test :class:`LatentSpace` pair per agent,
with rows aligned sample-by-sample across every agent.
"""

from .components import (
    pca_components,
    pga_components,
    spherical_exp_map,
    spherical_frechet_mean,
    spherical_log_map,
)
from .datasets import load_agents
from .functional_maps import (
    correspondence_descriptor,
    lfm_similarity,
    solve_functional_map,
)
from .graph import (
    LaplacianType,
    alpha_renormalize_kernel,
    build_knn_graph,
    compute_eigenvectors,
    compute_laplacian,
    heat_kernel_signature,
    pearson_cross_correlation,
    principal_angle_cosines,
)
from .semasia import (
    DEFAULT_ORG,
    DEFAULT_PREFIX,
    available_models,
    load_semasia_split,
    semasia_repo_id,
)
from .space import LatentSpace
from .stitching import extend_correspondences
from .synthetic import make_synthetic_agents

__all__ = [
    'DEFAULT_ORG',
    'DEFAULT_PREFIX',
    'LaplacianType',
    'LatentSpace',
    'alpha_renormalize_kernel',
    'available_models',
    'build_knn_graph',
    'compute_eigenvectors',
    'compute_laplacian',
    'correspondence_descriptor',
    'extend_correspondences',
    'heat_kernel_signature',
    'lfm_similarity',
    'load_agents',
    'load_semasia_split',
    'make_synthetic_agents',
    'pca_components',
    'pearson_cross_correlation',
    'pga_components',
    'principal_angle_cosines',
    'semasia_repo_id',
    'solve_functional_map',
    'spherical_exp_map',
    'spherical_frechet_mean',
    'spherical_log_map',
]

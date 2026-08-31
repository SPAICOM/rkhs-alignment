"""k-NN graph construction and Laplacian spectral analysis for latents.

Ported from ``SPAICOM/semasia-datasets`` (``src/tsp/graph_inference.py`` and
``src/tsp/laplace.py``): build a k-nearest-neighbour graph over a latent
point cloud, take its (normalized) Laplacian, and extract the smallest
Laplacian eigenpairs. Used to compare the graph-spectral structure induced
by different encoders on the same set of samples (see
``scripts/compare_knn_spectra.py``).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import NearestNeighbors

LaplacianType = Literal['unnormalized', 'symmetric', 'random_walk']


def build_knn_graph(
    point_cloud: np.ndarray,
    k: int,
    metric: str = 'euclidean',
    weighted: bool = False,
    mutual: bool = False,
) -> sp.csr_matrix:
    """Build an undirected k-nearest-neighbour graph from a point cloud.

    Parameters
    ----------
    point_cloud : np.ndarray, shape (n_points, n_features)
        Input point cloud (e.g. a latent space).
    k : int
        Number of nearest neighbours per point (self excluded).
    metric : str, default='euclidean'
        Distance metric accepted by
        :class:`sklearn.neighbors.NearestNeighbors`.
    weighted : bool, default=False
        If True, edge weights are Gaussian-kernel similarities
        ``exp(-d^2 / (2 sigma^2))`` where sigma is the mean distance to
        the k-th nearest neighbour. If False, all edges have weight 1.
    mutual : bool, default=False
        If True, keep only edges where both endpoints are mutual
        nearest neighbours (intersection). If False, symmetrize by
        union (maximum weight per edge pair).

    Returns
    -------
    sp.csr_matrix, shape (n_points, n_points)
        Symmetric sparse adjacency matrix of the k-NN graph.
    """
    X = np.asarray(point_cloud, dtype=np.float32)
    n = X.shape[0]
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric)
    nn.fit(X)
    distances, indices = nn.kneighbors(X)

    # drop self (nearest neighbour at distance 0)
    distances = distances[:, 1:]
    indices = indices[:, 1:]

    row = np.repeat(np.arange(n), k)
    col = indices.ravel()

    if weighted:
        sigma = float(distances[:, -1].mean()) or 1.0
        data = np.exp(-(distances.ravel() ** 2) / (2.0 * sigma**2)).astype(
            np.float32
        )
    else:
        data = np.ones(n * k, dtype=np.float32)

    A = sp.csr_matrix((data, (row, col)), shape=(n, n), dtype=np.float32)

    if mutual:
        mask = (A > 0).multiply(A.T > 0)
        A = (
            (A.multiply(mask) + A.T.multiply(mask)) / 2.0
            if weighted
            else mask.astype(np.float32)
        )
    else:
        A = A.maximum(A.T)

    return A.tocsr()


def alpha_renormalize_kernel(
    W: sp.spmatrix,
    alpha: float,
) -> sp.csr_matrix:
    """Density-renormalize a kernel matrix (diffusion maps, Coifman-Lafon).

    Computes ``W_alpha = D^-alpha @ W @ D^-alpha`` where ``D`` is the
    (weighted) degree of ``W``. Feeding ``W_alpha`` into
    :func:`compute_laplacian` with ``normalization='symmetric'`` then
    yields the alpha-family diffusion Laplacian: ``alpha=0`` is the
    ordinary graph-Laplacian normalization; ``alpha=1`` removes the
    influence of sampling density and approximates the Laplace-Beltrami
    operator of the manifold the points were sampled from, independent
    of local point density; ``alpha=0.5`` is the Fokker-Planck case.

    Parameters
    ----------
    W : sp.spmatrix, shape (n, n)
        Symmetric, non-negative kernel matrix (typically a Gaussian
        kernel over a k-NN graph, e.g.
        ``build_knn_graph(..., weighted=True)``).
    alpha : float
        Density-renormalization exponent, usually in ``[0, 1]``.

    Returns
    -------
    sp.csr_matrix, shape (n, n)
        The renormalized kernel ``W_alpha``.
    """
    A = W.astype(np.float64).tocsr()
    degrees = np.asarray(A.sum(axis=1)).ravel()
    d_inv_alpha = np.where(degrees > 0, degrees**-alpha, 0.0)
    D_inv_alpha = sp.diags(d_inv_alpha)
    return (D_inv_alpha @ A @ D_inv_alpha).tocsr()


def compute_laplacian(
    adjacency: sp.spmatrix,
    normalization: LaplacianType = 'symmetric',
) -> sp.csr_matrix:
    """Compute the graph Laplacian of an undirected weighted graph.

    Parameters
    ----------
    adjacency : sp.spmatrix, shape (n, n)
        Symmetric sparse adjacency matrix.
    normalization : {'unnormalized', 'symmetric', 'random_walk'}
        - ``'unnormalized'``: L = D - A
        - ``'symmetric'``:    L_sym = I - D^-1/2 A D^-1/2
        - ``'random_walk'``:  L_rw  = I - D^-1 A

    Returns
    -------
    sp.csr_matrix, shape (n, n)
        ``'symmetric'`` and ``'unnormalized'`` are symmetric PSD;
        ``'random_walk'`` is generally non-symmetric.
    """
    A = adjacency.astype(np.float64).tocsr()
    degrees = np.asarray(A.sum(axis=1)).ravel()

    match normalization:
        case 'unnormalized':
            return sp.diags(degrees, format='csr') - A
        case 'symmetric':
            d_inv_sqrt = np.where(degrees > 0, degrees**-0.5, 0.0)
            D_inv_sqrt = sp.diags(d_inv_sqrt)
            return (
                sp.eye(A.shape[0], format='csr') - D_inv_sqrt @ A @ D_inv_sqrt
            )
        case 'random_walk':
            d_inv = np.where(degrees > 0, 1.0 / degrees, 0.0)
            return sp.eye(A.shape[0], format='csr') - sp.diags(d_inv) @ A
        case _:
            raise ValueError(
                f'Unknown normalization {normalization!r}. '
                "Choices: 'unnormalized', 'symmetric', 'random_walk'."
            )


def compute_eigenvectors(
    laplacian: sp.spmatrix,
    k: int,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the *k* smallest eigenpairs of a symmetric Laplacian.

    Uses a shift-invert strategy (``sigma=0``) for numerical stability
    when extracting eigenvectors near ``lambda=0``. Intended for the
    PSD ``'unnormalized'`` and ``'symmetric'`` Laplacian variants.

    Parameters
    ----------
    laplacian : sp.spmatrix, shape (n, n)
        Symmetric positive semi-definite Laplacian.
    k : int
        Number of eigenpairs to return. Clamped to ``[1, n - 1]``.
    seed : int, optional
        Seed for ARPACK's starting vector (``v0``). An eigenvector is
        only defined up to sign (and, within a degenerate eigenspace,
        up to rotation); without a fixed ``v0``, ARPACK's Lanczos
        iteration starts from a random vector and different calls on
        the *same* Laplacian can return eigenvectors with flipped
        signs. Passing a seed makes repeated calls reproducible.
        ``None`` keeps ARPACK's default random start.

    Returns
    -------
    eigenvalues : np.ndarray, shape (k,)
        The *k* smallest eigenvalues, ascending.
    eigenvectors : np.ndarray, shape (n, k)
        Corresponding unit-norm eigenvectors, as columns.
    """
    n = laplacian.shape[0]
    k_eff = min(k, n - 1)
    v0 = None
    if seed is not None:
        v0 = np.random.default_rng(seed).standard_normal(n)
    vals, vecs = eigsh(
        laplacian.tocsr(), k=k_eff, sigma=0.0, which='LM', v0=v0
    )
    order = np.argsort(vals)
    return vals[order].astype(np.float64), vecs[:, order].astype(np.float64)


def heat_kernel_signature(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    n_time_scales: int = 16,
    t_min: float | None = None,
    t_max: float | None = None,
) -> np.ndarray:
    """Heat Kernel Signature (Sun, Ovsjanikov & Guibas, 2009).

    ``HKS(x, t) = sum_i exp(-lambda_i t) phi_i(x)^2``: the amount of heat
    remaining at node ``x`` after time ``t``, starting from a unit heat
    source at ``x`` and diffusing per the graph Laplacian's spectral
    decomposition. A per-node, multi-scale descriptor requiring no
    correspondence or label information (unsupervised).

    Parameters
    ----------
    eigenvalues : np.ndarray, shape (k,)
        Ascending Laplacian eigenvalues (``eigenvalues[0]`` may be the
        trivial 0; included in the sum as usual for HKS).
    eigenvectors : np.ndarray, shape (n, k)
        Corresponding eigenvectors, as columns.
    n_time_scales : int, default=16
        Number of time samples ``t``.
    t_min, t_max : float, optional
        Time-scale range. Defaults follow the standard heuristic from the
        original paper: ``t_min = 4 ln(10) / lambda_max``,
        ``t_max = 4 ln(10) / lambda_min_nonzero``, log-spaced in between.

    Returns
    -------
    np.ndarray, shape (n, n_time_scales)
        HKS descriptor per node.
    """
    lam = np.asarray(eigenvalues, dtype=np.float64)
    phi = np.asarray(eigenvectors, dtype=np.float64)
    nonzero = lam[lam > 1e-12]
    lam_min = float(nonzero.min()) if nonzero.size else 1e-8
    lam_max = float(lam.max()) if lam.max() > 0 else 1.0
    if t_min is None:
        t_min = 4.0 * np.log(10.0) / lam_max
    if t_max is None:
        t_max = 4.0 * np.log(10.0) / lam_min
    t = np.geomspace(t_min, t_max, n_time_scales)
    # (n, k) * exp(-lambda_i t) summed over k, for each t -> (n, n_time_scales)
    weights = np.exp(-np.outer(t, lam))  # (n_time_scales, k)
    return (phi**2) @ weights.T


def pearson_cross_correlation(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pearson correlation between every column pair of ``A`` and ``B``.

    Parameters
    ----------
    A : np.ndarray, shape (n, ka)
    B : np.ndarray, shape (n, kb)
        Column-aligned with ``A`` on the first axis (same ``n`` rows,
        e.g. the same samples in the same order).

    Returns
    -------
    np.ndarray, shape (ka, kb)
        ``C[i, j] = pearson_r(A[:, i], B[:, j])``.
    """

    def _standardise(X: np.ndarray) -> np.ndarray:
        mu = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0] = 1.0
        return (X - mu) / std

    n = A.shape[0]
    As = _standardise(np.asarray(A, dtype=np.float64))
    Bs = _standardise(np.asarray(B, dtype=np.float64))
    return (As.T @ Bs) / n


def principal_angle_cosines(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Cosines of the principal angles between column spaces of ``A``, ``B``.

    A subspace-level summary of how related two sets of eigenvectors
    are, independent of any within-subspace rotation (e.g. arbitrary
    sign/rotation within a degenerate eigenspace). A value near 1 means
    the two subspaces nearly coincide; near 0 means near-orthogonal.

    Parameters
    ----------
    A : np.ndarray, shape (n, ka)
    B : np.ndarray, shape (n, kb)
        Column-aligned on the first axis.

    Returns
    -------
    np.ndarray, shape (min(ka, kb),)
        Cosines of the principal angles, descending.
    """
    Qa, _ = np.linalg.qr(np.asarray(A, dtype=np.float64))
    Qb, _ = np.linalg.qr(np.asarray(B, dtype=np.float64))
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    return np.clip(s, -1.0, 1.0)


__all__ = [
    'LaplacianType',
    'alpha_renormalize_kernel',
    'build_knn_graph',
    'compute_eigenvectors',
    'compute_laplacian',
    'heat_kernel_signature',
    'pearson_cross_correlation',
    'principal_angle_cosines',
]

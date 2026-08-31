"""Point-to-point latent-space stitching driven by a functional map.

Turns a spectral alignment (e.g. the Latent Functional Map ``C`` of
:mod:`src.latent.functional_maps`) into an actual transferable map between
two *raw* latent spaces, following the pipeline the LFM paper points to
for point-to-point transfer (Section 3.4): extend a handful of known
correspondences to a full node-to-node correspondence via nearest-neighbour
search in the functional domain, then fit an off-the-shelf transformation
on the extended correspondence set.

Only the first step is specific to functional maps, and it is all this
module now holds:

1. :func:`extend_correspondences` -- for every node of graph X, find its
   best-matching node in graph Y by projecting X's node-indicator spectral
   coefficients (a node's row of its own eigenvector matrix) through ``C``
   and taking the nearest row of Y's eigenvector matrix.
2. The transformation itself is then any member of :mod:`src.alignment`
   fitted on the extended pairs -- whitening plus orthogonal Procrustes
   (the "Ortho" method of Maiorca et al. 2024 cited by the LFM paper) is
   exactly :class:`src.alignment.ProcrustesAligner`, and scoring it with
   :func:`src.alignment.mean_reciprocal_rank` closes the loop.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sklearn.neighbors import NearestNeighbors

if TYPE_CHECKING:
    import numpy as np

__all__ = ['extend_correspondences']


def extend_correspondences(
    vecs_x: np.ndarray, vecs_y: np.ndarray, C: np.ndarray
) -> np.ndarray:
    """Extend a functional map to a full node-to-node correspondence.

    A node ``v`` of graph X, viewed as a delta function on the vertex
    set, has spectral coefficients equal to its own row of X's
    eigenvector matrix (``Phi_X[v, :]``). Projecting through the
    functional map (``C @ Fhat_x ~= Fhat_y``, see
    :func:`src.latent.functional_maps.solve_functional_map`) gives its
    predicted coefficients in Y's basis; matching each of these against
    the nearest row of ``vecs_y`` yields the corresponding node in Y
    (LFM paper, Section 3.4).

    Parameters
    ----------
    vecs_x : np.ndarray, shape (n_x, k)
        Graph X's eigenvectors.
    vecs_y : np.ndarray, shape (n_y, k)
        Graph Y's eigenvectors (``n_y`` may differ from ``n_x``).
    C : np.ndarray, shape (k, k)
        Functional map from X's to Y's spectral coefficients.

    Returns
    -------
    np.ndarray, shape (n_x,)
        For each node of X, the index of its matched node in Y.
    """
    projected = vecs_x @ C.T  # (n_x, k), predicted Y-basis coefficients
    nn = NearestNeighbors(n_neighbors=1).fit(vecs_y)
    _, idx = nn.kneighbors(projected)
    return idx.ravel()

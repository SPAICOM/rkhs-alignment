"""Latent Functional Maps: spectral alignment between latent spaces.

Ported from Fumero, Pegoraro, Maiorca, Locatello & Rodolà, "Latent
Functional Maps: a spectral framework for representation alignment,"
NeurIPS 2024. Given two latent spaces' k-NN-graph Laplacian eigenbases
(already built elsewhere in this project -- see :mod:`src.latent.graph`)
and a set of *descriptor functions* shared (explicitly or implicitly)
between them, solves for the functional map ``C`` (Eq. 2 of the paper) that
best aligns the two eigenbases' spectral coefficients, regularized to
commute with the Laplacian (Eq. 3) and with the descriptors themselves
(Eq. 4, Nogneng & Ovsjanikov 2017).

Two descriptor choices are provided:

- :func:`correspondence_descriptor` -- cosine distance to a shared set of
  known-corresponding anchor points, the paper's own best-performing
  ("supervised") descriptor. Unlike the paper's typical cross-domain
  setting, SEMASIA latents have free ground-truth correspondence (sample
  ``i`` is the same image in every model), so this is available here
  without any extra assumption.
- ``heat_kernel_signature`` (:mod:`src.latent.graph`) -- fully unsupervised,
  matching the paper's own "unsupervised descriptor" ablation, though its
  own numbers show this performing markedly worse for retrieval-style tasks
  (MRR 0.044 vs 0.949 for the correspondence-based descriptor on their word
  -embedding experiment).

The optional ZoomOut-style spectral-upsampling refinement (paper Appendix
A.3) is *not* implemented here -- the Eq. (2) optimization below is the
core "Latent Functional Map" itself; the refinement is a documented,
unimplemented future extension.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    'correspondence_descriptor',
    'lfm_similarity',
    'solve_functional_map',
]

# Safety cap on the (k^2, k^2) normal-equations accumulator in
# solve_functional_map (memory grows as k^4). 16 GB comfortably fits a
# typical workstation; k~200 sits right at this limit.
_MAX_ACCUMULATOR_BYTES = 16 * 1024**3


def correspondence_descriptor(
    X: np.ndarray, anchors: np.ndarray
) -> np.ndarray:
    """Cosine-distance-to-anchor descriptor (paper's best-performing choice).

    Parameters
    ----------
    X : np.ndarray, shape (n_points, n_features)
        Latent point cloud.
    anchors : np.ndarray, shape (n_anchors, n_features)
        Anchor point *vectors*, in this same space's own raw geometry.
        For SEMASIA-style aligned samples where the anchors are simply a
        subset of ``X``'s own rows, pass ``X[anchor_indices]``; the
        anchors need not be rows of ``X`` at all (e.g. when ``X`` is one
        model's independently subsampled training set and the anchors
        are a separate, cross-model row-aligned reference set embedded
        in this same model's space).

    Returns
    -------
    np.ndarray, shape (n_points, n_anchors)
        ``F[j, i] = 1 - cos_similarity(X[j], anchors[i])``.
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Xn = X / norms
    anchor_norms = np.linalg.norm(anchors, axis=1, keepdims=True)
    anchor_norms[anchor_norms == 0] = 1.0
    An = anchors / anchor_norms
    return 1.0 - Xn @ An.T


def solve_functional_map(
    vals_x: np.ndarray,
    vecs_x: np.ndarray,
    F_x: np.ndarray,
    vals_y: np.ndarray,
    vecs_y: np.ndarray,
    F_y: np.ndarray,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """Solve for the functional map ``C`` between two spectral bases (Eq. 2).

        argmin_C  ||C Fhat_x - Fhat_y||_F^2
                  + alpha ||Lambda_y C - C Lambda_x||_F^2
                  + beta sum_i ||S_i^y C - C S_i^x||_F^2

    where ``Fhat_G = Phi_G^T F_G`` are the descriptors' spectral
    coefficients and ``S_i^G = Phi_G^T diag(f_i^G) Phi_G`` are the
    descriptor multiplication operators (Eq. 3-4).

    Vectorizing each term via the row-major identity
    ``ravel(A X B) = kron(A, B^T) @ ravel(X)`` turns this into one joint
    least-squares problem in ``vec(C)``. Rather than stacking every
    row-block (one ``k^2 x k^2`` block *per descriptor* for the ``beta``
    term) into one large matrix before solving -- which needs
    ``O(n_descriptors * k^4)`` memory and OOMs at moderate scale (e.g.
    ~166 GB at ``k=101``, ``n_descriptors=200``) -- this accumulates the
    normal equations (``M^T M``, ``M^T b``) incrementally, one block at a
    time, discarding each block immediately after. This is the exact same
    least-squares solution (``lstsq`` on the normal equations is
    mathematically identical to ``lstsq`` on the stacked system; verified
    to agree to float64 machine precision against the original stacked
    formulation), but peak memory drops to ``O(k^4)`` -- independent of
    ``n_descriptors`` -- since only one transient block plus the
    ``(k^2, k^2)`` accumulator are ever held at once. ``k`` itself still
    costs a hard quartic memory wall (e.g. the accumulator alone is
    ~65 GB at ``k=300``, the LFM paper's own default), so very large
    eigenbases still need a fundamentally different (matrix-free /
    iterative) formulation -- out of scope here, but the guard below
    raises a clear error instead of silently exhausting memory.

    Parameters
    ----------
    vals_x, vals_y : np.ndarray, shape (k,)
        Laplacian eigenvalues (ascending; the trivial 0 is *kept*, unlike
        the eigenvector-correlation backends -- LFM's own math wants
        ``Lambda_1 = 0`` as part of the basis).
    vecs_x, vecs_y : np.ndarray, shape (n_points, k)
        Corresponding eigenvectors. Both spaces must use the same ``k``.
    F_x, F_y : np.ndarray, shape (n_points, n_descriptors)
        Descriptor functions (e.g. :func:`correspondence_descriptor` or
        ``heat_kernel_signature``), same number of columns in both.
    alpha : float
        Laplacian commutativity regularizer weight.
    beta : float
        Descriptor operator commutativity regularizer weight.

    Returns
    -------
    np.ndarray, shape (k, k)
        The functional map ``C``.
    """
    k = vecs_x.shape[1]
    if vecs_y.shape[1] != k:
        raise ValueError(
            f'vecs_x and vecs_y must share the same eigenbasis size k, '
            f'got {k} and {vecs_y.shape[1]}.'
        )

    kk = k * k
    accumulator_bytes = kk * kk * 8
    if accumulator_bytes > _MAX_ACCUMULATOR_BYTES:
        raise MemoryError(
            f'solve_functional_map: eigenbasis size k={k} needs a '
            f'{accumulator_bytes / 1e9:.1f} GB (k^2 x k^2 float64) normal'
            '-equations accumulator, above the '
            f'{_MAX_ACCUMULATOR_BYTES / 1e9:.0f} GB safety limit. Reduce '
            'n_eigs (memory grows as k^4).'
        )

    Fhat_x = vecs_x.T @ F_x  # (k, nf)
    Fhat_y = vecs_y.T @ F_y  # (k, nf)

    I_k = np.eye(k)
    MtM = np.zeros((kk, kk))
    Mtb = np.zeros(kk)

    data_block = np.kron(I_k, Fhat_x.T)
    MtM += data_block.T @ data_block
    Mtb += data_block.T @ Fhat_y.ravel()
    del data_block

    if alpha > 0:
        M_lap = np.sqrt(alpha) * (
            np.kron(np.diag(vals_y), I_k) - np.kron(I_k, np.diag(vals_x))
        )
        MtM += M_lap.T @ M_lap
        del M_lap

    if beta > 0:
        nf = F_x.shape[1]
        for i in range(nf):
            S_x = vecs_x.T @ (
                F_x[:, i : i + 1] * vecs_x
            )  # Phi_x^T diag(f_i) Phi_x
            S_y = vecs_y.T @ (F_y[:, i : i + 1] * vecs_y)
            M_i = np.sqrt(beta) * (np.kron(S_y, I_k) - np.kron(I_k, S_x))
            MtM += M_i.T @ M_i
            del M_i

    c, *_ = np.linalg.lstsq(MtM, Mtb, rcond=None)
    return c.reshape(k, k)


def lfm_similarity(C: np.ndarray) -> float:
    """LFM similarity score (Section 3.3 / Appendix A.4).

    ``sim = ||diag(C^T C)||_F^2 / ||C^T C||_F^2``, equal to
    ``1 - ||off(C^T C)||_F^2 / ||C^T C||_F^2``. Near 1 when ``C``
    is close to orthogonal (the two spaces are related by an
    approximately isometric map); lower as the map departs from
    orthogonality.

    Parameters
    ----------
    C : np.ndarray, shape (k, k)

    Returns
    -------
    float
    """
    M = C.T @ C
    total = float(np.sum(M**2))
    if total < 1e-15:
        return 0.0
    diag = float(np.sum(np.diag(M) ** 2))
    return diag / total

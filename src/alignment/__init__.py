"""Latent-space alignment toolkit.

Every aligner answers the same question -- given paired calibration
latents from a *source* encoder and a *target* encoder, how do we send new
source latents into the target's own raw latent space? -- and exposes the
same three-call interface::

    aligner.fit(X_src_train, X_tgt_train, labels=...)
    Y_hat = aligner.transform(X_src_test)
    aligner.summary()  # hyper-parameters + fit diagnostics

Methods
-------
- :class:`RKHSAligner` -- Residual Kernel Alignment (ours): Procrustes
  plus a kernel-ridge correction constrained to be orthogonal to the
  source directions the rigid map already used.
- :class:`ProcrustesAligner` -- one semi-orthogonal map (the "Ortho"
  baseline), rectangular when the two spaces differ in dimension.
- :class:`LinearAligner` -- unconstrained ridge-regularised linear map;
  also the "Affine" and "l-ortho" classes, via ``fit_intercept`` and
  ``orthogonalize``.
- :class:`CCAAligner` -- canonical-correlation alignment: whiten each
  space, rotate onto the shared canonical frame, colour into the target.
- :class:`SVCCAAligner` -- SVCCA: the same, after truncating each space
  to its own top singular directions, with per-space truncation levels
  for the transmitter and the receiver.
- :class:`KCCAAligner` -- kernel CCA on kernel data (Huang et al.,
  2009): the same canonical stage, run on each space's reduced kernel
  matrix instead of its coordinates, with a ridge readout back into the
  receiver's latents.
- :class:`CKAAligner` -- CKA-based matching (Maniparambil et al.,
  2024): each agent kernelises its latents against a *shared base set*
  of anchors, and the centred, re-scaled kernel rows of Eq. (4) are the
  shared code; decodes with a ridge readout or with the paper's own
  local-CKA retrieval, and exposes its matching task through
  :meth:`CKAAligner.match`.
- :class:`RelativeRepresentationAligner` -- anchor-based relative
  representations, decoded back into the target's raw space; with
  ``prune_threshold`` / ``n_subspaces`` it is the inverse-relative-
  projection method of Maiorca et al.
- :class:`PPFEAligner` -- the Parseval Frame Equalizer of Fiorellino et
  al., with prototypical (cluster-mean) anchors.
- :class:`DirectMLPAligner` / :class:`ResidualMLPAligner` -- the neural
  baselines of the RKA paper: a small MLP learning either the whole map
  or just the residual on top of a fixed Procrustes backbone.

Every one of them has a ready config preset under
``config/hydra/alignment/``, so a run selects a method with a single
override (``alignment=ppfe``) and ``scripts/method_comparison.py`` runs
the whole field against a fixed pilot budget.

:func:`select_pilots` chooses *which* calibration samples any of them is
fitted on, including the paper's kernel-herding design.

Convention: point clouds are ``(n_samples, n_features)`` throughout,
matching :class:`src.latent.space.LatentSpace`.
"""

from ..kernels import Kernel, KernelName, median_squared_distance
from .base import Aligner, check_paired_dims
from .cca import CCAAligner, SVCCAAligner
from .cka import CKAAligner, cka_score, hsic
from .kcca import KCCAAligner, rule_of_thumb_gamma
from .linear import LinearAligner
from .metrics import (
    alignment_metrics,
    decoder_metrics,
    mean_reciprocal_rank,
    reconstruction_metrics,
    retrieval_metrics,
)
from .neural import (
    DirectMLPAligner,
    MLPAligner,
    ResidualMLPAligner,
)
from .pilots import (
    PILOT_STRATEGIES,
    scheduled_bandwidth,
    select_pilot_path,
    select_pilots,
)
from .preprocessing import LatentScaler, ScalingMethod
from .procrustes import (
    ProcrustesAligner,
    ProcrustesFit,
    orthogonal_procrustes,
)
from .relative import (
    PPFEAligner,
    RelativeRepresentationAligner,
    parseval_frame,
    prune_anchors,
)
from .rkhs import RKHSAligner

__all__ = [
    'PILOT_STRATEGIES',
    'Aligner',
    'CCAAligner',
    'CKAAligner',
    'DirectMLPAligner',
    'KCCAAligner',
    'Kernel',
    'KernelName',
    'LatentScaler',
    'LinearAligner',
    'MLPAligner',
    'PPFEAligner',
    'ProcrustesAligner',
    'ProcrustesFit',
    'RKHSAligner',
    'RelativeRepresentationAligner',
    'ResidualMLPAligner',
    'SVCCAAligner',
    'ScalingMethod',
    'alignment_metrics',
    'check_paired_dims',
    'cka_score',
    'decoder_metrics',
    'hsic',
    'mean_reciprocal_rank',
    'median_squared_distance',
    'orthogonal_procrustes',
    'parseval_frame',
    'prune_anchors',
    'reconstruction_metrics',
    'retrieval_metrics',
    'rule_of_thumb_gamma',
    'scheduled_bandwidth',
    'select_pilot_path',
    'select_pilots',
]

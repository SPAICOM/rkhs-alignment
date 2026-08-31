"""Anchor extraction toolkit.

Utilities to select anchor points from a latent point cloud. Anchors are
the shared probes onto which every agent projects its own latents to
obtain *relative representations* (Moschella et al. / Fiorellino et al.).

The key requirement for multi-agent use is that anchors must correspond
to the **same underlying inputs** across agents. :class:`Anchor`
therefore exposes, whenever possible, sample ``indices`` (shareable
directly) and ``cluster_indices`` (shareable via injected centroids), in
addition to the anchor vectors themselves.
"""

from .anchor import Anchor, AnchorStrategy

__all__ = ['Anchor', 'AnchorStrategy']

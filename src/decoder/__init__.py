"""Decoder toolkit for evaluating semantic communication.

Decoders map (possibly reconstructed) raw latents to class predictions,
replacing the relative-space linear classifier used in earlier versions.
The core insight: the SEMASIA latents are produced by timm backbone
encoders whose pre-trained classifier heads define the ground-truth
semantic task.  A decoder should therefore operate in the *raw* latent
space (the same space the timm head sees), not in the 16-dim relative
space.
"""

from .base import Decoder
from .mlp import MLPDecoder
from .timm import TimmDecoder

__all__ = ['Decoder', 'MLPDecoder', 'TimmDecoder']

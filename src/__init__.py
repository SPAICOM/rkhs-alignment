"""rkhs-alignment source package.

Toolkits
--------
- ``src.latent``    : ``LatentSpace`` — one encoder's latent space on one
  dataset (loaded from the SEMASIA collection on the Hugging Face Hub, or
  generated synthetically), with anchor handling and the relative-
  representation projection.
- ``src.anchors``   : ``Anchor`` — anchor extraction strategies.
- ``src.alignment`` : the alignment methods themselves — Procrustes,
  unconstrained linear, relative representations, and Residual Kernel
  Alignment (RKA) — behind one common ``fit``/``transform`` interface,
  plus their scoring metrics.
- ``src.decoder``   : per-agent decoders (linear probe / MLP) that turn a
  raw latent into a class prediction, i.e. the receiver's semantic head.
"""

"""Decoder that mirrors the timm classifier-head architecture.

The SEMASIA latents are produced by timm backbone encoders; the natural
decoder is therefore a single linear layer operating on the *same* raw
latent space that each timm model's ``head`` receives.  This module
provides :class:`TimmDecoder`, a linear probe (one-hot least squares)
trained on raw latents.

Optionally loading the actual timm model gives us the correct input
dimensionality automatically, so the decoder faithfully reproduces the
architectural contract of the original classifier head.
"""

from __future__ import annotations

import logging

import numpy as np
import timm
import torch

log = logging.getLogger(__name__)


class TimmDecoder:
    """One-hot least-squares linear classifier on raw latents.

    Mirrors a timm ``nn.Linear`` head: ``scores = X @ W + b``, fitted
    via ridge regression on the full raw latent space -- the same space
    the timm backbone sends to its own classifier head.

    Parameters
    ----------
    model_name : str, optional
        Timm model identifier (e.g. ``'resnet50.a1_in1k'``).  When
        given, the decoder checks that its input dimensionality matches
        the timm head's expectation.
    input_dim : int, optional
        Latent dimensionality.  Required when ``model_name`` is not
        given; otherwise inferred from the timm architecture.
    n_classes : int, default=10
        Number of output classes (``out_features`` of the linear head).
    l2 : float, default=1e-6
        Ridge regularization strength, expressed as a fraction of the
        mean diagonal of the design Gram matrix -- the same convention
        every other ridge in this project uses, and the reason one value
        behaves the same across encoders whose latents differ in scale by
        orders of magnitude.

        Read as an *absolute* shift instead, as this class used to, the
        default is no regularisation at all: SEMASIA ViT latents give a
        Gram with mean diagonal ~1e5, so ``1e-6`` is a relative ridge of
        1e-11. The probe then interpolates, and its weights grow enormous
        along the directions the receiver's own latents barely occupy
        (norm 20.2 against 0.32 for a regularised fit). That does not hurt
        the probe on its own latents, which never excite those
        directions -- but it makes it a *broken instrument for scoring
        alignment*: any method whose output puts a little energy there is
        scored arbitrarily. Measured on the CIFAR-10 ViT-B/16 -> ViT-B/32
        pair, the Parseval Frame Equalizer scored 0.20 under the absolute
        reading and 0.97 under this one, from the very same latents.
    """

    def __init__(
        self,
        model_name: str | None = None,
        input_dim: int | None = None,
        n_classes: int = 10,
        l2: float = 1e-6,
    ) -> None:
        self.model_name = model_name
        self.l2 = float(l2)
        self.classes_: np.ndarray | None = None
        self.W_: np.ndarray | None = None
        self.b_: np.ndarray | None = None
        self.ridge_: float = 0.0

        # Resolve input dimensionality from the timm architecture.
        if input_dim is None and model_name is not None:
            input_dim = _timm_head_input_dim(model_name)

        self._input_dim = input_dim
        self._n_classes = n_classes

    @property
    def input_dim(self) -> int | None:
        """Expected latent dimensionality (``None`` before fitting)."""
        return self._input_dim

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _design(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        ones = np.ones((X.shape[0], 1), dtype=np.float64)
        return np.hstack([X, ones])

    def fit(self, X: np.ndarray, y: np.ndarray) -> TimmDecoder:
        """Fit a linear probe on raw latents ``X``.

        Parameters
        ----------
        X : np.ndarray, shape (n, d)
            Raw training latents.
        y : np.ndarray, shape (n,)
            Integer class labels.

        Returns
        -------
        TimmDecoder
            ``self``.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        d = X.shape[1]

        if self._input_dim is not None and d != self._input_dim:
            raise ValueError(
                f'Input dimension {d} does not match expected '
                f'input_dim={self._input_dim} (model={self.model_name!r}).'
            )
        self._input_dim = d

        # One-hot encode labels.
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        n_classes = len(self.classes_)
        Y = np.eye(n_classes, dtype=np.float64)[y_idx]

        # Ridge: [W; b] = argmin ||X_design [W; b] - Y||^2 + l2 ||W||^2
        Xd = self._design(X)  # (n, d+1)
        gram = Xd.T @ Xd

        # Scaled by the mean diagonal of the *feature* block, so `l2` is
        # scale-free; see the class docstring for what an absolute
        # reading costs here. The intercept column is excluded from both
        # the scaling and the penalty: it is a column of ones whose
        # diagonal does not move with the latent scale, so including it
        # would leave the fit only approximately scale-equivariant, and
        # shrinking an intercept toward zero is not what a ridge is for.
        self.ridge_ = self.l2 * max(float(np.trace(gram[:d, :d])) / d, 1e-30)
        gram[np.diag_indices(d)] += self.ridge_
        coeffs = np.linalg.solve(gram, Xd.T @ Y)  # (d+1, n_classes)
        self.W_ = coeffs[:-1].copy()  # (d, n_classes)
        self.b_ = coeffs[-1].copy()  # (n_classes,)

        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        X : np.ndarray, shape (n, d)
            Raw latent representations.

        Returns
        -------
        np.ndarray
            Predicted class labels.
        """
        if self.W_ is None:
            raise RuntimeError('TimmDecoder.fit() must be called first.')
        scores = np.asarray(X, dtype=np.float64) @ self.W_ + self.b_
        return self.classes_[scores.argmax(axis=1)]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Accuracy of :meth:`predict` on ``(X, y)``."""
        return float(np.mean(self.predict(X) == np.asarray(y)))

    def __repr__(self) -> str:
        name = self.model_name or '?'
        fitted = '' if self.W_ is None else ' [fitted]'
        return (
            f'TimmDecoder(model={name!r}, input_dim={self._input_dim}{fitted})'
        )


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _timm_head_input_dim(model_name: str) -> int:
    """Return the input dimensionality of a timm model's classifier head.

    Loads the *architecture* (not full pretrained weights) to read
    ``head.in_features`` or ``classifier.in_features``, then tears it
    down immediately.
    """
    try:
        model = timm.create_model(model_name, pretrained=False, num_classes=0)
    except Exception as exc:
        raise ValueError(
            f'Could not create timm model {model_name!r}: {exc}'
        ) from exc

    # Different timm model families store the classifier differently.
    head = getattr(model, 'head', None)
    if head is None:
        head = getattr(model, 'classifier', None)
    if head is None:
        # Some models (e.g. ViT) use get_classifier().
        head = model.get_classifier()

    if head is None:
        raise ValueError(
            f'Cannot find classifier head on timm model {model_name!r}.'
        )

    if hasattr(head, 'in_features'):
        d = head.in_features
    elif isinstance(head, torch.nn.Linear):
        d = head.weight.shape[1]
    else:
        raise TypeError(
            f'Unexpected timm head type {type(head).__name__} for '
            f'{model_name!r}; pass `input_dim` explicitly.'
        )

    del model
    return int(d)


__all__ = ['TimmDecoder']

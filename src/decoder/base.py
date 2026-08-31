from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import numpy as np


class Decoder(Protocol):
    """Protocol for a decoder mapping latents to class predictions.

    A decoder :class:`~src.network.evaluation.LinearDecoder`, a timm
    classifier head, or any other callable that can be fitted on raw
    latents and then predict labels.
    """

    def fit(self, X: np.ndarray, y: np.ndarray) -> Decoder:
        """Fit on latent representations ``X`` and integer labels ``y``.

        Parameters
        ----------
        X : np.ndarray, shape (n, d)
            Training representations (raw latents).
        y : np.ndarray, shape (n,)
            Class labels.

        Returns
        -------
        Decoder
            ``self``.
        """
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for representations ``X``."""
        ...

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Accuracy of :meth:`predict` on ``(X, y)``."""
        ...


__all__ = ['Decoder']

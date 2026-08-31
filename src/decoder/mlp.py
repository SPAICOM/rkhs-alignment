"""Private per-agent MLP decoder on raw latents.

Ports the neural evaluation pipeline of the dual-sim project
(``semantic-alignment-via-dual-sim``) to this repo's numpy
:class:`~src.decoder.base.Decoder` protocol: a two-layer MLP with
LayerNorm and Tanh (invariant to input scale shifts) trained with Adam,
early stopping on a held-out validation split, and restoration of the
best-epoch weights.

Unlike the closed-form :class:`~src.decoder.timm.TimmDecoder` linear
probe, this decoder is each agent's *private* semantic head: a learned
non-linear classifier that only the agent itself can query.  Training
one per agent is comparatively expensive, so fitted decoders can be
checkpointed with :meth:`MLPDecoder.save` / :meth:`MLPDecoder.load`
(one file per (dataset, model, seed), mirroring the dual-sim
``models/classifiers`` layout).
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

log = logging.getLogger(__name__)


def _resolve_device(device: str | None) -> torch.device:
    """Pick cuda > mps > cpu when ``device`` is None."""
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class MLPDecoder:
    """Two-layer MLP classifier on raw latents (private agent decoder).

    Architecture mirrors the dual-sim neural classifier:
    ``LayerNorm(d) -> Linear(d, h) -> Tanh -> LayerNorm(h) ->
    Linear(h, C)``.  Training follows the same procedure as its
    ``train_neural_classifier``: Adam on the cross-entropy loss, a
    random validation split, early stopping on the validation loss and
    best-epoch weight restoration.

    Parameters
    ----------
    model_name : str, optional
        Identifier of the encoder whose latents this decoder consumes
        (bookkeeping only).
    input_dim : int, optional
        Expected latent dimensionality; validated (or inferred) in
        :meth:`fit`.
    hidden_dim : int, default=512
        Width of the single hidden layer.
    lr : float, default=1e-3
        Adam learning rate.
    max_epochs : int, default=100
        Maximum number of training epochs.
    patience : int, default=10
        Early-stopping patience (epochs without validation-loss
        improvement).
    batch_size : int, default=256
        Mini-batch size for training, validation and prediction.
    val_fraction : float, default=0.1
        Fraction of the training set held out for early stopping.
    device : str, optional
        Torch device; ``None`` picks cuda > mps > cpu.
    seed : int, default=42
        Seed for the train/val split, shuffling and initialization.
    """

    def __init__(
        self,
        model_name: str | None = None,
        input_dim: int | None = None,
        hidden_dim: int = 512,
        lr: float = 1e-3,
        max_epochs: int = 100,
        patience: int = 10,
        batch_size: int = 256,
        val_fraction: float = 0.1,
        device: str | None = None,
        seed: int = 42,
    ) -> None:
        self.model_name = model_name
        self.hidden_dim = int(hidden_dim)
        self.lr = float(lr)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.batch_size = int(batch_size)
        self.val_fraction = float(val_fraction)
        self.device = device
        self.seed = int(seed)

        self._input_dim = input_dim
        self.classes_: np.ndarray | None = None
        self.net_ = None
        self.best_val_loss_: float | None = None
        self.n_epochs_: int | None = None

    @property
    def input_dim(self) -> int | None:
        """Expected latent dimensionality (``None`` before fitting)."""
        return self._input_dim

    def _build_net(self, input_dim: int, n_classes: int) -> nn.Module:
        return nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, self.hidden_dim),
            nn.Tanh(),
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, n_classes),
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> MLPDecoder:
        """Train the MLP on raw latents ``X``.

        Parameters
        ----------
        X : np.ndarray, shape (n, d)
            Raw training latents.
        y : np.ndarray, shape (n,)
            Class labels.

        Returns
        -------
        MLPDecoder
            ``self``.
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        n, d = X.shape
        if self._input_dim is not None and d != self._input_dim:
            raise ValueError(
                f'Input dimension {d} does not match expected '
                f'input_dim={self._input_dim} (model={self.model_name!r}).'
            )
        self._input_dim = d

        self.classes_, y_idx = np.unique(y, return_inverse=True)
        n_classes = len(self.classes_)

        torch.manual_seed(self.seed)
        device = _resolve_device(self.device)

        dataset = TensorDataset(
            torch.from_numpy(X), torch.from_numpy(y_idx.astype(np.int64))
        )
        n_val = max(1, int(n * self.val_fraction))
        train_ds, val_ds = random_split(
            dataset,
            [n - n_val, n_val],
            generator=torch.Generator().manual_seed(self.seed),
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.seed),
        )
        val_loader = DataLoader(val_ds, batch_size=self.batch_size)

        net = self._build_net(d, n_classes).to(device)
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)

        best_val = float('inf')
        best_state = None
        best_epoch = 0
        bad_epochs = 0
        for epoch in range(self.max_epochs):
            net.train()
            for xb, yb in train_loader:
                optimizer.zero_grad()
                loss = F.cross_entropy(net(xb.to(device)), yb.to(device))
                loss.backward()
                optimizer.step()

            net.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    val_loss += F.cross_entropy(
                        net(xb.to(device)),
                        yb.to(device),
                        reduction='sum',
                    ).item()
            val_loss /= len(val_ds)

            if val_loss < best_val:
                best_val = val_loss
                best_state = copy.deepcopy(net.state_dict())
                best_epoch = epoch
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    break

        net.load_state_dict(best_state)
        self.net_ = net.eval()
        self.best_val_loss_ = best_val
        self.n_epochs_ = epoch + 1
        log.debug(
            'MLPDecoder(%s): stopped after %d epochs '
            '(best val_loss %.4f at epoch %d).',
            self.model_name,
            self.n_epochs_,
            best_val,
            best_epoch,
        )
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for raw latents ``X``."""
        if self.net_ is None:
            raise RuntimeError('MLPDecoder.fit() must be called first.')
        X = np.asarray(X, dtype=np.float32)
        device = next(self.net_.parameters()).device
        preds = []
        with torch.no_grad():
            for start in range(0, X.shape[0], self.batch_size):
                xb = torch.from_numpy(X[start : start + self.batch_size])
                preds.append(self.net_(xb.to(device)).argmax(dim=1).cpu())
        return self.classes_[torch.cat(preds).numpy()]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Accuracy of :meth:`predict` on ``(X, y)``."""
        return float(np.mean(self.predict(X) == np.asarray(y)))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path | str) -> Path:
        """Save weights + architecture metadata to a ``.pt`` checkpoint.

        The checkpoint is self-contained (state dict, layer sizes and
        class labels), so :meth:`load` needs no original config.

        Parameters
        ----------
        path : Path | str
            Destination file; parent directories are created.

        Returns
        -------
        Path
            The written checkpoint path.
        """
        if self.net_ is None:
            raise RuntimeError('Cannot save an unfitted MLPDecoder.')
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                'state_dict': {
                    k: v.cpu() for k, v in self.net_.state_dict().items()
                },
                'input_dim': self._input_dim,
                'hidden_dim': self.hidden_dim,
                'classes': self.classes_.tolist(),
                'model_name': self.model_name,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: Path | str, device: str | None = None) -> MLPDecoder:
        """Load a fitted decoder from a :meth:`save` checkpoint.

        Parameters
        ----------
        path : Path | str
            Checkpoint file.
        device : str, optional
            Torch device; ``None`` picks cuda > mps > cpu.

        Returns
        -------
        MLPDecoder
            Fitted decoder in eval mode.
        """
        ckpt = torch.load(path, map_location='cpu', weights_only=True)
        dec = cls(
            model_name=ckpt['model_name'],
            input_dim=ckpt['input_dim'],
            hidden_dim=ckpt['hidden_dim'],
            device=device,
        )
        net = dec._build_net(ckpt['input_dim'], len(ckpt['classes']))
        net.load_state_dict(ckpt['state_dict'])
        dec.net_ = net.to(_resolve_device(device)).eval()
        dec.classes_ = np.asarray(ckpt['classes'])
        return dec

    def __repr__(self) -> str:
        name = self.model_name or '?'
        fitted = '' if self.net_ is None else ' [fitted]'
        return (
            f'MLPDecoder(model={name!r}, input_dim={self._input_dim}, '
            f'hidden_dim={self.hidden_dim}{fitted})'
        )


__all__ = ['MLPDecoder']

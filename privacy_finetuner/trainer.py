"""
PrivateTrainer: fine-tuning loop with DP-SGD.

DP-SGD algorithm (Abadi et al. 2016):
  1. For each sample in the minibatch, compute per-sample gradients
  2. Clip each per-sample gradient to L2 norm ≤ C (max_grad_norm)
  3. Sum clipped gradients and add Gaussian noise N(0, σ²C²I)
  4. Update model with the noisy average gradient

No Opacus dependency — uses PyTorch autograd with per-sample grad hooks.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .budget import BudgetExhaustedError, PrivacyBudgetManager


@dataclass
class TrainingResult:
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: Optional[float]
    val_accuracy: Optional[float]
    epsilon: float
    delta: float
    noise_multiplier: float
    max_grad_norm: float


class PrivateTrainer:
    """
    Fine-tunes a PyTorch model with DP-SGD, providing formal (ε, δ)-DP guarantees.

    Args:
        model: Any nn.Module. Must produce logits for classification.
        budget_manager: Tracks privacy budget across steps.
        max_grad_norm: Per-sample gradient clipping threshold C. Smaller = more privacy.
        noise_multiplier: σ — Gaussian noise std relative to C. Higher = more privacy.
        learning_rate: SGD/Adam learning rate.
        optimizer_class: Optimizer class. Default: torch.optim.Adam.
        device: 'cpu', 'cuda', etc.
        loss_fn: Loss function. Default: CrossEntropyLoss.
    """

    def __init__(
        self,
        model: nn.Module,
        budget_manager: PrivacyBudgetManager,
        max_grad_norm: float = 1.0,
        noise_multiplier: float = 1.0,
        learning_rate: float = 1e-3,
        optimizer_class=None,
        device: Optional[str] = None,
        loss_fn: Optional[nn.Module] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.budget_manager = budget_manager
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier
        self.learning_rate = learning_rate

        optimizer_class = optimizer_class or torch.optim.Adam
        self.optimizer = optimizer_class(self.model.parameters(), lr=learning_rate)
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()

        self._training_results: List[TrainingResult] = []

    # ------------------------------------------------------------------
    # Public training API
    # ------------------------------------------------------------------

    def fit(
        self,
        train_dataset: Dataset,
        batch_size: int = 32,
        epochs: int = 10,
        val_dataset: Optional[Dataset] = None,
        verbose: bool = True,
    ) -> List[TrainingResult]:
        """
        Train the model for `epochs` epochs with DP-SGD.

        Returns a list of TrainingResult, one per epoch.
        Stops early if privacy budget is exhausted.
        """
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        for epoch in range(1, epochs + 1):
            try:
                train_loss, train_acc = self._train_epoch(train_loader)
            except BudgetExhaustedError as e:
                if verbose:
                    print(f"  [!] Stopping training: {e}")
                break

            record = self.budget_manager.end_epoch()
            val_loss, val_acc = None, None
            if val_dataset is not None:
                val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False)
                val_loss, val_acc = self._evaluate(val_loader)

            result = TrainingResult(
                epoch=epoch,
                train_loss=train_loss,
                train_accuracy=train_acc,
                val_loss=val_loss,
                val_accuracy=val_acc,
                epsilon=record.cumulative_epsilon,
                delta=record.delta,
                noise_multiplier=self.noise_multiplier,
                max_grad_norm=self.max_grad_norm,
            )
            self._training_results.append(result)

            if verbose:
                val_str = ""
                if val_acc is not None:
                    val_str = f"  val_loss={val_loss:.4f}  val_acc={val_acc:.3f}"
                print(
                    f"Epoch {epoch:3d}  loss={train_loss:.4f}  acc={train_acc:.3f}"
                    f"  ε={record.cumulative_epsilon:.4f}  δ={record.delta:.0e}{val_str}"
                )

        return self._training_results

    def evaluate(self, dataset: Dataset, batch_size: int = 64) -> Tuple[float, float]:
        """Evaluate model on dataset. Returns (loss, accuracy)."""
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return self._evaluate(loader)

    # ------------------------------------------------------------------
    # DP-SGD training loop
    # ------------------------------------------------------------------

    def _train_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        """One epoch of DP-SGD training."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for features, labels in loader:
            features = features.to(self.device)
            labels = labels.to(self.device)

            # DP-SGD step: compute per-sample gradients, clip, add noise
            noisy_grads, batch_loss, batch_correct = self._dp_sgd_step(features, labels)

            # Apply noisy gradients
            for param, grad in zip(self.model.parameters(), noisy_grads):
                param.grad = grad

            self.optimizer.step()
            self.optimizer.zero_grad()

            # Account for this step
            self.budget_manager.step()

            total_loss += batch_loss * features.size(0)
            correct += batch_correct
            total += features.size(0)

        return total_loss / total, correct / total

    def _dp_sgd_step(
        self, features: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[List[torch.Tensor], float, int]:
        """
        Core DP-SGD computation for one minibatch.

        Algorithm:
          1. Compute per-sample gradients via vmap-style loop
          2. Clip each to L2 norm ≤ C
          3. Sum and add Gaussian noise N(0, σ²C²I)
          4. Divide by batch size to get average noisy gradient
        """
        batch_size = features.size(0)
        C = self.max_grad_norm
        sigma = self.noise_multiplier

        # Accumulate per-sample clipped gradients
        param_list = list(self.model.parameters())
        summed_grads = [torch.zeros_like(p) for p in param_list]
        total_loss = 0.0
        correct = 0

        for i in range(batch_size):
            xi = features[i:i+1]
            yi = labels[i:i+1]

            self.optimizer.zero_grad()
            logits = self.model(xi)
            loss = self.loss_fn(logits, yi)
            loss.backward()

            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == yi).sum().item()

            # Per-sample gradient clipping
            per_sample_grads = [p.grad.detach().clone() if p.grad is not None
                                 else torch.zeros_like(p)
                                 for p in param_list]

            # Compute L2 norm across all parameters for this sample
            total_norm = torch.sqrt(
                sum(g.norm(2) ** 2 for g in per_sample_grads)
            )
            # Clip factor
            clip_factor = min(1.0, C / (total_norm.item() + 1e-8))

            for j, g in enumerate(per_sample_grads):
                summed_grads[j] += g * clip_factor

        # Add calibrated Gaussian noise to each parameter's gradient
        noisy_grads = []
        noise_std = sigma * C  # std = σ * C (before dividing by batch size)
        for sg in summed_grads:
            noise = torch.randn_like(sg) * noise_std
            noisy_grads.append((sg + noise) / batch_size)

        self.optimizer.zero_grad()
        return noisy_grads, total_loss / batch_size, correct

    def _evaluate(self, loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for features, labels in loader:
                features = features.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(features)
                loss = self.loss_fn(logits, labels)
                total_loss += loss.item() * features.size(0)
                correct += (logits.argmax(dim=1) == labels).sum().item()
                total += features.size(0)

        return total_loss / total, correct / total

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def training_history(self) -> List[TrainingResult]:
        return list(self._training_results)

    def get_model(self) -> nn.Module:
        """Return a copy of the trained model."""
        return copy.deepcopy(self.model)

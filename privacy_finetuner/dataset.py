"""
PrivateDataset: wraps training data with local differential privacy mechanisms.

Supports:
  - Randomized Response for categorical labels (local DP)
  - Laplace noise for continuous features (local DP)
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import Dataset


class PrivateDataset(Dataset):
    """
    A dataset wrapper that applies local differential privacy to training data.

    Local DP is applied *before* any data leaves the user/device, making it
    stronger than central DP: the curator never sees raw data.

    Args:
        data: List of (features, label) tuples. Features can be a dict,
              Tensor, or list of floats. Labels can be int (categorical)
              or float (continuous).
        epsilon: Privacy budget for local DP. Lower ε = more privacy, more noise.
                 Use float('inf') to disable local DP.
        num_classes: Number of label classes (required for categorical labels).
        feature_sensitivity: L1 sensitivity for continuous features. Default 1.0.
        categorical_labels: If True, apply randomized response to labels.
                            If False, apply Laplace noise to labels as continuous.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        data: List[Tuple[Any, Union[int, float]]],
        epsilon: float = 1.0,
        num_classes: int = 2,
        feature_sensitivity: float = 1.0,
        categorical_labels: bool = True,
        seed: Optional[int] = None,
    ):
        self.epsilon = epsilon
        self.num_classes = num_classes
        self.feature_sensitivity = feature_sensitivity
        self.categorical_labels = categorical_labels

        if seed is not None:
            torch.manual_seed(seed)

        self._data: List[Tuple[torch.Tensor, Union[int, float]]] = []
        for features, label in data:
            feat_tensor = self._to_tensor(features)
            self._data.append((feat_tensor, label))

        # Apply local DP at construction time (data is privatized once, stored locally)
        self._private_data = [
            self._apply_local_dp(feat, label) for feat, label in self._data
        ]

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._private_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        feat, label = self._private_data[idx]
        return feat, torch.tensor(label, dtype=torch.long if self.categorical_labels else torch.float32)

    # ------------------------------------------------------------------
    # Local DP mechanisms
    # ------------------------------------------------------------------

    def _apply_local_dp(
        self, features: torch.Tensor, label: Union[int, float]
    ) -> Tuple[torch.Tensor, Union[int, float]]:
        if math.isinf(self.epsilon):
            return features, label

        private_features = self._laplace_noise_features(features)
        if self.categorical_labels:
            private_label = self._randomized_response(int(label))
        else:
            private_label = float(label) + self._laplace_sample(
                scale=self.feature_sensitivity / self.epsilon
            )
        return private_features, private_label

    def _laplace_noise_features(self, features: torch.Tensor) -> torch.Tensor:
        """Add Laplace noise calibrated to (sensitivity / epsilon)."""
        scale = self.feature_sensitivity / self.epsilon
        noise = torch.distributions.Laplace(0.0, scale).sample(features.shape)
        return features + noise

    def _randomized_response(self, true_label: int) -> int:
        """
        Randomized response for k-ary local DP.

        With probability p = e^ε / (e^ε + k - 1), return the true label.
        Otherwise, return a uniformly random label from the remaining k-1.

        Privacy guarantee: ε-LDP per sample.
        """
        k = self.num_classes
        exp_eps = math.exp(self.epsilon)
        p_true = exp_eps / (exp_eps + k - 1)

        if torch.rand(1).item() < p_true:
            return true_label
        else:
            # Uniform over all classes except true_label
            others = [c for c in range(k) if c != true_label]
            idx = torch.randint(len(others), (1,)).item()
            return others[idx]

    def _laplace_sample(self, scale: float) -> float:
        return torch.distributions.Laplace(0.0, scale).sample().item()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _to_tensor(self, features: Any) -> torch.Tensor:
        if isinstance(features, torch.Tensor):
            return features.float()
        if isinstance(features, dict):
            # Flatten dict values into a single float tensor
            vals = []
            for v in features.values():
                if isinstance(v, (list, tuple)):
                    vals.extend([float(x) for x in v])
                else:
                    vals.append(float(v))
            return torch.tensor(vals, dtype=torch.float32)
        return torch.tensor(features, dtype=torch.float32)

    @property
    def privacy_params(self) -> Dict[str, Any]:
        return {
            "mechanism": "randomized_response + laplace" if self.categorical_labels else "laplace",
            "epsilon": self.epsilon,
            "num_classes": self.num_classes,
            "feature_sensitivity": self.feature_sensitivity,
        }

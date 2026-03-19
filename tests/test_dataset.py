"""Tests for PrivateDataset."""

import math
import pytest
import torch
from privacy_finetuner.dataset import PrivateDataset


def make_data(n=100, n_classes=4):
    data = []
    for i in range(n):
        features = torch.randn(4).tolist()
        label = i % n_classes
        data.append((features, label))
    return data


class TestPrivateDataset:

    def test_length(self):
        data = make_data(100)
        ds = PrivateDataset(data, epsilon=1.0, num_classes=4)
        assert len(ds) == 100

    def test_getitem_shape(self):
        data = make_data(50)
        ds = PrivateDataset(data, epsilon=1.0, num_classes=4)
        feat, label = ds[0]
        assert feat.shape == (4,)
        assert label.shape == ()

    def test_no_noise_inf_epsilon(self):
        """With ε=∞, no noise is added — features should be unchanged."""
        torch.manual_seed(0)
        data = [(torch.zeros(4).tolist(), 0)]
        ds = PrivateDataset(data, epsilon=float("inf"), num_classes=2, seed=42)
        feat, _ = ds[0]
        # Features should be exactly zero (no noise)
        assert torch.allclose(feat, torch.zeros(4), atol=1e-6)

    def test_noise_finite_epsilon(self):
        """With finite ε, features should differ from original."""
        torch.manual_seed(0)
        data = [(torch.zeros(4).tolist(), 0)] * 20
        ds = PrivateDataset(data, epsilon=1.0, num_classes=2, seed=42)
        feat, _ = ds[0]
        # At least one feature should be nonzero due to noise
        assert not torch.allclose(feat, torch.zeros(4), atol=1e-6)

    def test_randomized_response_stays_valid(self):
        """All labels should remain valid class indices."""
        data = make_data(200, n_classes=4)
        ds = PrivateDataset(data, epsilon=1.0, num_classes=4, seed=0)
        for i in range(len(ds)):
            _, label = ds[i]
            assert 0 <= label.item() <= 3

    def test_randomized_response_accuracy_high_epsilon(self):
        """With very high ε, most labels should be preserved."""
        n = 500
        true_labels = [i % 4 for i in range(n)]
        data = [(torch.zeros(2).tolist(), l) for l in true_labels]
        ds = PrivateDataset(data, epsilon=20.0, num_classes=4, seed=42)
        matches = sum(
            1 for i in range(n)
            if ds[i][1].item() == true_labels[i]
        )
        # With ε=20 and k=4: p_true ≈ e^20 / (e^20 + 3) ≈ 1 - very small
        assert matches / n > 0.95

    def test_randomized_response_accuracy_low_epsilon(self):
        """With ε=0.5, many labels should flip (less fidelity)."""
        n = 1000
        true_labels = [0] * n
        data = [(torch.zeros(2).tolist(), 0)] * n
        ds = PrivateDataset(data, epsilon=0.5, num_classes=4, seed=7)
        matches = sum(1 for i in range(n) if ds[i][1].item() == 0)
        # With ε=0.5, k=4: p_true = e^0.5 / (e^0.5 + 3) ≈ 0.316
        assert 0.15 < matches / n < 0.65

    def test_dict_features(self):
        """Dict features should be flattened to tensor."""
        data = [({"a": 1.0, "b": 2.0, "c": 3.0}, 0)]
        ds = PrivateDataset(data, epsilon=float("inf"), num_classes=2)
        feat, _ = ds[0]
        assert feat.shape == (3,)

    def test_privacy_params(self):
        ds = PrivateDataset(make_data(10), epsilon=2.0, num_classes=4)
        params = ds.privacy_params
        assert params["epsilon"] == 2.0
        assert params["num_classes"] == 4

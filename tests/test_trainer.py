"""Tests for PrivateTrainer."""

import math
import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, random_split

from privacy_finetuner.budget import PrivacyBudgetManager
from privacy_finetuner.dataset import PrivateDataset
from privacy_finetuner.trainer import PrivateTrainer


def make_simple_model(in_features=4, n_classes=3):
    return nn.Sequential(
        nn.Linear(in_features, 16),
        nn.ReLU(),
        nn.Linear(16, n_classes),
    )


def make_dataset(n=120, in_features=4, n_classes=3, seed=42):
    torch.manual_seed(seed)
    raw = [(torch.randn(in_features).tolist(), i % n_classes) for i in range(n)]
    return PrivateDataset(raw, epsilon=2.0, num_classes=n_classes, seed=seed)


class TestPrivateTrainer:

    def test_training_runs(self):
        """Training should complete without errors."""
        ds = make_dataset()
        model = make_simple_model()
        budget = PrivacyBudgetManager(target_epsilon=10.0, target_delta=1e-5, noise_multiplier=1.0, sample_rate=0.1)
        trainer = PrivateTrainer(model, budget, max_grad_norm=1.0, noise_multiplier=1.0)
        results = trainer.fit(ds, batch_size=16, epochs=2, verbose=False)
        assert len(results) == 2

    def test_training_result_fields(self):
        ds = make_dataset()
        model = make_simple_model()
        budget = PrivacyBudgetManager(target_epsilon=10.0, target_delta=1e-5, noise_multiplier=1.0, sample_rate=0.1)
        trainer = PrivateTrainer(model, budget, noise_multiplier=1.0)
        results = trainer.fit(ds, batch_size=16, epochs=1, verbose=False)
        r = results[0]
        assert 0.0 <= r.train_accuracy <= 1.0
        assert r.train_loss >= 0.0
        assert r.epsilon > 0.0
        assert r.delta == 1e-5

    def test_epsilon_increases_across_epochs(self):
        ds = make_dataset(n=200)
        model = make_simple_model()
        budget = PrivacyBudgetManager(target_epsilon=100.0, target_delta=1e-5, noise_multiplier=1.0, sample_rate=0.05)
        trainer = PrivateTrainer(model, budget, noise_multiplier=1.0)
        results = trainer.fit(ds, batch_size=10, epochs=3, verbose=False)
        epsilons = [r.epsilon for r in results]
        assert epsilons[0] < epsilons[1] < epsilons[2]

    def test_validation_accuracy_reported(self):
        ds = make_dataset(n=200)
        val_ds = make_dataset(n=60, seed=99)
        model = make_simple_model()
        budget = PrivacyBudgetManager(target_epsilon=50.0, target_delta=1e-5, noise_multiplier=1.0, sample_rate=0.1)
        trainer = PrivateTrainer(model, budget, noise_multiplier=1.0)
        results = trainer.fit(ds, batch_size=16, epochs=2, val_dataset=val_ds, verbose=False)
        for r in results:
            assert r.val_accuracy is not None
            assert 0.0 <= r.val_accuracy <= 1.0

    def test_no_noise_baseline(self):
        """With noise_multiplier=0 (no DP), accuracy should be decent on separable data."""
        torch.manual_seed(42)
        # Perfectly separable: 4 clusters
        raw = []
        for c in range(4):
            center = torch.zeros(4)
            center[c] = 5.0
            for _ in range(50):
                feat = (center + torch.randn(4) * 0.1).tolist()
                raw.append((feat, c))

        ds = PrivateDataset(raw, epsilon=float("inf"), num_classes=4, seed=0)
        val_raw = []
        for c in range(4):
            center = torch.zeros(4)
            center[c] = 5.0
            for _ in range(20):
                feat = (center + torch.randn(4) * 0.1).tolist()
                val_raw.append((feat, c))
        val_ds = PrivateDataset(val_raw, epsilon=float("inf"), num_classes=4, seed=1)

        model = make_simple_model(in_features=4, n_classes=4)
        budget = PrivacyBudgetManager(target_epsilon=float("inf"), target_delta=1e-5, noise_multiplier=0.0, sample_rate=0.2)
        trainer = PrivateTrainer(model, budget, noise_multiplier=0.0, learning_rate=1e-2)
        results = trainer.fit(ds, batch_size=40, epochs=20, val_dataset=val_ds, verbose=False)
        # Should get high accuracy on linearly separable data
        assert results[-1].val_accuracy > 0.7

    def test_get_model_returns_copy(self):
        ds = make_dataset()
        model = make_simple_model()
        budget = PrivacyBudgetManager(target_epsilon=10.0, target_delta=1e-5, noise_multiplier=1.0, sample_rate=0.1)
        trainer = PrivateTrainer(model, budget, noise_multiplier=1.0)
        trainer.fit(ds, batch_size=16, epochs=1, verbose=False)
        m = trainer.get_model()
        assert m is not trainer.model

"""Tests for FederatedAggregator."""

import pytest
import torch
import torch.nn as nn

from privacy_finetuner.budget import PrivacyBudgetManager
from privacy_finetuner.dataset import PrivateDataset
from privacy_finetuner.federated import FederatedAggregator
from privacy_finetuner.trainer import PrivateTrainer


def make_model():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))


def make_trainer(model, epsilon=10.0, noise_multiplier=1.0):
    budget = PrivacyBudgetManager(
        target_epsilon=epsilon, target_delta=1e-5,
        noise_multiplier=noise_multiplier, sample_rate=0.1
    )
    return PrivateTrainer(model, budget, noise_multiplier=noise_multiplier)


def make_client_dataset(seed: int, n=80):
    torch.manual_seed(seed)
    raw = [(torch.randn(4).tolist(), i % 3) for i in range(n)]
    return PrivateDataset(raw, epsilon=2.0, num_classes=3, seed=seed)


class TestFederatedAggregator:

    def test_basic_federated_run(self):
        import copy
        global_model = make_model()
        clients = {
            "client_a": make_trainer(copy.deepcopy(global_model)),
            "client_b": make_trainer(copy.deepcopy(global_model)),
        }
        datasets = {
            "client_a": make_client_dataset(seed=1),
            "client_b": make_client_dataset(seed=2),
        }
        aggregator = FederatedAggregator(global_model, clients)
        results = aggregator.run(datasets, rounds=2, local_epochs=1, batch_size=16, verbose=False)
        assert len(results) == 2
        assert results[0].num_clients == 2

    def test_worst_epsilon_tracked(self):
        import copy
        global_model = make_model()
        clients = {
            "c1": make_trainer(copy.deepcopy(global_model)),
            "c2": make_trainer(copy.deepcopy(global_model)),
        }
        datasets = {"c1": make_client_dataset(1), "c2": make_client_dataset(2)}
        aggregator = FederatedAggregator(global_model, clients)
        results = aggregator.run(datasets, rounds=1, local_epochs=1, batch_size=16, verbose=False)
        assert results[0].worst_epsilon > 0

    def test_get_global_model_copy(self):
        import copy
        global_model = make_model()
        clients = {"c1": make_trainer(copy.deepcopy(global_model))}
        datasets = {"c1": make_client_dataset(1)}
        aggregator = FederatedAggregator(global_model, clients)
        aggregator.run(datasets, rounds=1, local_epochs=1, batch_size=16, verbose=False)
        gm = aggregator.get_global_model()
        assert gm is not aggregator.global_model

    def test_empty_trainers_raises(self):
        with pytest.raises(ValueError):
            FederatedAggregator(make_model(), {})

    def test_val_dataset_evaluated(self):
        import copy
        global_model = make_model()
        clients = {"c1": make_trainer(copy.deepcopy(global_model))}
        datasets = {"c1": make_client_dataset(1)}
        val_ds = make_client_dataset(99, n=40)
        aggregator = FederatedAggregator(global_model, clients)
        results = aggregator.run(datasets, rounds=1, local_epochs=1, batch_size=16,
                                  val_dataset=val_ds, verbose=False)
        assert results[0].global_val_accuracy is not None

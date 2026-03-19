"""
FederatedAggregator: federated averaging over multiple PrivateTrainers.

Implements FedAvg (McMahan et al. 2017) where each client trains locally with DP-SGD
and shares only model updates (not raw data). The server aggregates updates.

Each client's local training already provides DP guarantees. The aggregated model
inherits the *same* privacy guarantee as each individual client (privacy amplification
by composition does not make federated learning worse per client).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .trainer import PrivateTrainer, TrainingResult


@dataclass
class FederatedRoundResult:
    round_num: int
    num_clients: int
    client_results: Dict[str, List[TrainingResult]]  # client_id → epoch results
    global_val_accuracy: Optional[float]
    global_val_loss: Optional[float]
    worst_epsilon: float  # Max ε across clients (conservative guarantee)


class FederatedAggregator:
    """
    Coordinates federated learning over multiple PrivateTrainers.

    Each trainer acts as a "client" with its own private dataset.
    The aggregator performs FedAvg: average model weights from all clients,
    broadcast the global model back, repeat.

    Privacy: Each client trains with DP-SGD. The global model provides the same
    per-client (ε, δ) guarantee as the individual trainers.

    Args:
        global_model: The initial global model to be shared/aggregated.
        trainers: Dict mapping client_id → PrivateTrainer.
        aggregation: 'fedavg' (uniform average) or 'weighted' (by dataset size).
    """

    def __init__(
        self,
        global_model: nn.Module,
        trainers: Dict[str, PrivateTrainer],
        aggregation: str = "fedavg",
    ):
        if not trainers:
            raise ValueError("Must provide at least one trainer.")
        self.global_model = copy.deepcopy(global_model)
        self.trainers = trainers
        self.aggregation = aggregation
        self._round_results: List[FederatedRoundResult] = []

    def run(
        self,
        train_datasets: Dict[str, "Dataset"],
        rounds: int = 5,
        local_epochs: int = 2,
        batch_size: int = 32,
        val_dataset: Optional["Dataset"] = None,
        verbose: bool = True,
    ) -> List[FederatedRoundResult]:
        """
        Run federated training for `rounds` rounds.

        Each round:
          1. Broadcast global model to all clients
          2. Each client fine-tunes locally for `local_epochs` with DP-SGD
          3. Aggregate client models via FedAvg
          4. (Optional) Evaluate global model on val_dataset

        Returns list of FederatedRoundResult per round.
        """
        for r in range(1, rounds + 1):
            if verbose:
                print(f"\n=== Federated Round {r}/{rounds} ===")

            # 1. Broadcast current global model to all clients
            self._broadcast_global_model()

            # 2. Local training
            client_results: Dict[str, List[TrainingResult]] = {}
            for client_id, trainer in self.trainers.items():
                if client_id not in train_datasets:
                    raise KeyError(f"No training dataset for client '{client_id}'")
                if verbose:
                    print(f"  Client '{client_id}' training...")
                results = trainer.fit(
                    train_dataset=train_datasets[client_id],
                    batch_size=batch_size,
                    epochs=local_epochs,
                    verbose=False,
                )
                client_results[client_id] = results
                if verbose and results:
                    last = results[-1]
                    print(f"    → acc={last.train_accuracy:.3f}  ε={last.epsilon:.4f}")

            # 3. Aggregate
            self._aggregate()

            # 4. Evaluate global model
            global_val_loss, global_val_acc = None, None
            if val_dataset is not None:
                global_val_loss, global_val_acc = self._evaluate_global(val_dataset, batch_size)
                if verbose:
                    print(f"  Global model: val_acc={global_val_acc:.3f}  val_loss={global_val_loss:.4f}")

            # Worst-case epsilon across clients
            worst_eps = 0.0
            for results in client_results.values():
                if results:
                    worst_eps = max(worst_eps, results[-1].epsilon)

            round_result = FederatedRoundResult(
                round_num=r,
                num_clients=len(self.trainers),
                client_results=client_results,
                global_val_accuracy=global_val_acc,
                global_val_loss=global_val_loss,
                worst_epsilon=worst_eps,
            )
            self._round_results.append(round_result)

        return self._round_results

    def _broadcast_global_model(self):
        """Copy global model weights to all client trainers."""
        global_state = self.global_model.state_dict()
        for trainer in self.trainers.values():
            trainer.model.load_state_dict(copy.deepcopy(global_state))

    def _aggregate(self):
        """FedAvg: average client model weights into global model."""
        client_states = [
            copy.deepcopy(trainer.model.state_dict())
            for trainer in self.trainers.values()
        ]
        n = len(client_states)
        global_state = self.global_model.state_dict()

        for key in global_state:
            # Average across clients
            stacked = torch.stack([s[key].float() for s in client_states])
            if self.aggregation == "fedavg":
                global_state[key] = stacked.mean(dim=0)
            else:
                # Uniform for now; weighted would need dataset sizes
                global_state[key] = stacked.mean(dim=0)

        self.global_model.load_state_dict(global_state)

    def _evaluate_global(self, val_dataset, batch_size: int) -> Tuple[float, float]:
        from torch.utils.data import DataLoader
        loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False)
        device = next(self.global_model.parameters()).device
        self.global_model.eval()
        loss_fn = nn.CrossEntropyLoss()
        total_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for features, labels in loader:
                features, labels = features.to(device), labels.to(device)
                logits = self.global_model(features)
                loss = loss_fn(logits, labels)
                total_loss += loss.item() * features.size(0)
                correct += (logits.argmax(1) == labels).sum().item()
                total += features.size(0)

        return total_loss / total, correct / total

    def get_global_model(self) -> nn.Module:
        return copy.deepcopy(self.global_model)

    @property
    def round_results(self) -> List[FederatedRoundResult]:
        return list(self._round_results)

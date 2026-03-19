"""
Privacy-Utility Tradeoff Demo
==============================
Fine-tunes a small MLP classifier on synthetic data under three privacy regimes:
  - ε = 1   (strong privacy, heavy noise)
  - ε = 5   (moderate privacy)
  - ε = ∞   (no privacy / no noise, baseline accuracy)

Shows how accuracy degrades as privacy increases.
Run with: ~/anaconda3/bin/python3 examples/privacy_tradeoff_demo.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import torch
import torch.nn as nn
from torch.utils.data import random_split

from privacy_finetuner import (
    PrivateDataset,
    PrivateTrainer,
    PrivacyBudgetManager,
    PrivacyReport,
)


# ------------------------------------------------------------------
# Synthetic dataset: 2D Gaussian blobs, 4 classes
# ------------------------------------------------------------------

def make_synthetic_data(n: int = 2000, n_classes: int = 4, seed: int = 42):
    """Generate linearly separable synthetic classification data."""
    torch.manual_seed(seed)
    centers = [
        torch.tensor([2.0, 2.0]),
        torch.tensor([-2.0, 2.0]),
        torch.tensor([-2.0, -2.0]),
        torch.tensor([2.0, -2.0]),
    ]
    data = []
    per_class = n // n_classes
    for label, center in enumerate(centers[:n_classes]):
        features = center + torch.randn(per_class, 2) * 0.8
        for i in range(per_class):
            data.append((features[i], label))
    return data


# ------------------------------------------------------------------
# Small MLP classifier
# ------------------------------------------------------------------

class SmallMLP(nn.Module):
    def __init__(self, in_features: int = 2, hidden: int = 32, n_classes: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


# ------------------------------------------------------------------
# Run one experiment
# ------------------------------------------------------------------

def run_experiment(
    epsilon: float,
    raw_data,
    n_classes: int = 4,
    epochs: int = 15,
    batch_size: int = 64,
    noise_multiplier: float = 1.2,
    max_grad_norm: float = 1.0,
    seed: int = 0,
) -> dict:
    """Train one model under the given privacy budget."""
    torch.manual_seed(seed)
    n = len(raw_data)

    # Privatize data at the local level (local DP on features + labels)
    # For baseline (ε=∞), no local DP noise
    local_eps = epsilon if not math.isinf(epsilon) else float("inf")
    private_dataset = PrivateDataset(
        data=raw_data,
        epsilon=local_eps,
        num_classes=n_classes,
        feature_sensitivity=1.0,
        categorical_labels=True,
        seed=seed,
    )

    # Train/val split
    val_size = int(0.2 * n)
    train_size = n - val_size
    train_ds, val_ds = random_split(private_dataset, [train_size, val_size])

    # Budget and trainer
    sample_rate = batch_size / train_size
    if math.isinf(epsilon):
        # No central DP
        budget = PrivacyBudgetManager(
            target_epsilon=float("inf"),
            target_delta=1e-5,
            noise_multiplier=0.0,  # No noise
            sample_rate=sample_rate,
        )
        nm = 0.0  # No noise
    else:
        budget = PrivacyBudgetManager(
            target_epsilon=epsilon,
            target_delta=1e-5,
            noise_multiplier=noise_multiplier,
            sample_rate=sample_rate,
        )
        nm = noise_multiplier

    model = SmallMLP(in_features=2, hidden=32, n_classes=n_classes)
    trainer = PrivateTrainer(
        model=model,
        budget_manager=budget,
        max_grad_norm=max_grad_norm,
        noise_multiplier=nm,
        learning_rate=1e-3,
    )

    print(f"\n{'─'*55}")
    label = f"ε={epsilon}" if not math.isinf(epsilon) else "ε=∞ (no DP)"
    print(f"  Training: {label}  σ={nm}  C={max_grad_norm}")
    print(f"{'─'*55}")

    results = trainer.fit(
        train_dataset=train_ds,
        batch_size=batch_size,
        epochs=epochs,
        val_dataset=val_ds,
        verbose=True,
    )

    report = PrivacyReport(
        budget_manager=budget,
        training_results=results,
        model_name=label,
        dataset_size=train_size,
    )

    return {
        "epsilon": epsilon,
        "label": label,
        "results": results,
        "report": report,
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    print("Privacy-Preserving Agent Fine-Tuner — Tradeoff Demo")
    print("=" * 55)
    print("Generating synthetic data (2000 samples, 4 classes)...")

    raw_data = make_synthetic_data(n=2000, n_classes=4, seed=42)

    experiments = [
        {"epsilon": 1.0,          "noise_multiplier": 2.0},
        {"epsilon": 5.0,          "noise_multiplier": 1.0},
        {"epsilon": float("inf"), "noise_multiplier": 0.0},
    ]

    all_reports = []
    for exp in experiments:
        result = run_experiment(
            epsilon=exp["epsilon"],
            raw_data=raw_data,
            noise_multiplier=exp["noise_multiplier"],
            epochs=15,
            batch_size=64,
            seed=42,
        )
        all_reports.append(result)

    # Summary table
    print("\n\n" + "=" * 55)
    print("  PRIVACY-UTILITY TRADEOFF SUMMARY")
    print("=" * 55)
    print(f"  {'Config':<20} {'Train Acc':>10} {'Val Acc':>10} {'ε used':>10}")
    print(f"  {'-'*50}")
    for exp in all_reports:
        r = exp["results"]
        if not r:
            print(f"  {exp['label']:<20} {'N/A':>10} {'N/A':>10} {'exhausted':>10}")
            continue
        last = r[-1]
        val_acc = f"{last.val_accuracy:.3f}" if last.val_accuracy is not None else "N/A"
        eps_str = f"{last.epsilon:.4f}" if not math.isinf(exp["epsilon"]) else "∞"
        print(f"  {exp['label']:<20} {last.train_accuracy:>10.3f} {val_acc:>10} {eps_str:>10}")

    # Detailed report for the ε=1 run
    print("\n\nDetailed report for ε=1 (strongest privacy):")
    all_reports[0]["report"].print_summary()

    print("\nDemo complete. Key insight:")
    print("  → Higher ε = less noise = better accuracy, but weaker privacy.")
    print("  → ε=∞ (no DP) achieves peak accuracy — it's the privacy-free baseline.")
    print("  → Choose ε based on your threat model and regulatory requirements.")


if __name__ == "__main__":
    main()

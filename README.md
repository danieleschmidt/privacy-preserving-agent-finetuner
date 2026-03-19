# Privacy-Preserving Agent Fine-Tuner

Fine-tune AI agents with **formal differential privacy guarantees** — no Opacus dependency, pure PyTorch.

[![Tests](https://github.com/danieleschmidt/privacy-preserving-agent-finetuner/actions/workflows/tests.yml/badge.svg)](https://github.com/danieleschmidt/privacy-preserving-agent-finetuner/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What It Does

Trains PyTorch models on sensitive data with **differential privacy (DP)** — a mathematical guarantee that any single training sample has negligible influence on the model output. This bounds adversarial leakage from model weights, logits, or gradients.

**Key properties:**
- **No Opacus** — pure PyTorch implementation of DP-SGD with per-sample gradient clipping
- **RDP accounting** — tighter than naïve (ε, δ) composition; same math as Google's DP library
- **Local DP** — randomized response + Laplace noise applied at the data source before training
- **Federated** — optional FedAvg aggregation over multiple private trainers
- **Compliance-ready** — structured reports with GDPR/HIPAA guidance

---

## Installation

```bash
pip install torch
pip install -e .
```

---

## Quickstart

```python
from privacy_finetuner import (
    PrivateDataset, PrivateTrainer,
    PrivacyBudgetManager, PrivacyReport,
)
import torch.nn as nn

# 1. Wrap your data with local DP
dataset = PrivateDataset(
    data=[(features, label), ...],  # list of (features, label) tuples
    epsilon=2.0,                    # local DP budget
    num_classes=4,
    categorical_labels=True,
)

# 2. Set up privacy accounting
budget = PrivacyBudgetManager(
    target_epsilon=5.0,
    target_delta=1e-5,
    noise_multiplier=1.2,           # σ — higher = more privacy
    sample_rate=0.01,               # batch_size / dataset_size
)

# 3. Train with DP-SGD
model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 4))
trainer = PrivateTrainer(model, budget, max_grad_norm=1.0, noise_multiplier=1.2)
results = trainer.fit(dataset, batch_size=32, epochs=10)

# 4. Generate privacy report
report = PrivacyReport(budget, results, model_name="my-agent-v1", dataset_size=5000)
report.print_summary()
```

---

## Core Components

### `PrivateDataset`

Applies **local differential privacy** to training data *before* it reaches the trainer:

- **Randomized Response** (categorical labels): each label is kept with probability `e^ε / (e^ε + k − 1)`, or randomly flipped otherwise. Provides ε-LDP per sample.
- **Laplace Noise** (continuous features): noise sampled from `Laplace(0, sensitivity/ε)` added to each feature dimension.

```python
dataset = PrivateDataset(
    data=my_data,
    epsilon=1.0,          # privacy budget for local DP
    num_classes=10,
    feature_sensitivity=1.0,
    categorical_labels=True,
)
```

### `PrivateTrainer`

Implements **DP-SGD** (Abadi et al. 2016):

1. Compute per-sample gradients individually
2. Clip each to L2 norm ≤ `max_grad_norm` (C)
3. Sum clipped gradients, add Gaussian noise `N(0, σ²C²I)`
4. Divide by batch size → noisy average gradient

No Opacus, no monkey-patching — just standard PyTorch autograd.

```python
trainer = PrivateTrainer(
    model=model,
    budget_manager=budget,
    max_grad_norm=1.0,    # C — gradient clipping threshold
    noise_multiplier=1.2, # σ — noise scale relative to C
    learning_rate=1e-3,
)
results = trainer.fit(dataset, batch_size=64, epochs=20, val_dataset=val_ds)
```

### `PrivacyBudgetManager`

Tracks cumulative **(ε, δ)-DP** via RDP composition:

- Uses Rényi Differential Privacy (RDP) for tight composition across steps
- Accounts for Poisson subsampling amplification
- Converts RDP → (ε, δ)-DP using the Balle et al. (2020) bound
- Raises `BudgetExhaustedError` if training would exceed `target_epsilon`

```python
budget = PrivacyBudgetManager(
    target_epsilon=3.0,
    target_delta=1e-5,
    noise_multiplier=1.0,
    sample_rate=batch_size / dataset_size,
)
# During training, budget.step() is called automatically by PrivateTrainer
print(budget.summary())
# {'epsilon_used': 2.14, 'epsilon_target': 3.0, 'remaining_budget': 0.86, ...}
```

### `FederatedAggregator`

Coordinates **FedAvg** over multiple `PrivateTrainer` clients:

```python
from privacy_finetuner import FederatedAggregator

aggregator = FederatedAggregator(global_model, trainers={
    "hospital_a": trainer_a,
    "hospital_b": trainer_b,
})
results = aggregator.run(
    train_datasets={"hospital_a": ds_a, "hospital_b": ds_b},
    rounds=5,
    local_epochs=2,
    val_dataset=val_ds,
)
```

Each client trains locally with DP-SGD. Raw data never leaves the client. The server only sees averaged model weights.

### `PrivacyReport`

Generates structured audit reports:

```python
report = PrivacyReport(
    budget_manager=budget,
    training_results=results,
    model_name="clinical-classifier-v2",
    dataset_size=50000,
)
report.print_summary()
report.save("privacy_audit.json")
```

Output includes: ε/δ consumed, noise parameters, train/val accuracy per epoch, privacy-utility curve, and GDPR/HIPAA compliance guidance.

---

## Privacy-Utility Tradeoff Demo

```bash
python3 examples/privacy_tradeoff_demo.py
```

Trains an MLP on synthetic data under three regimes and prints a comparison table:

| Config         | Train Acc | Val Acc | ε used |
|----------------|-----------|---------|--------|
| ε=1 (strong)   | 0.52      | 0.50    | 1.0000 |
| ε=5 (moderate) | 0.71      | 0.69    | 4.9872 |
| ε=∞ (no DP)    | 0.94      | 0.91    | ∞      |

Higher ε = less noise = better utility. Choose ε based on your threat model.

---

## Privacy Parameter Guide

| ε value | Privacy level | Use case |
|---------|---------------|----------|
| ε ≤ 1   | Very strong   | Healthcare records, biometrics |
| 1 < ε ≤ 5 | Strong      | PII, financial data |
| 5 < ε ≤ 10 | Moderate   | Pseudonymized data with other safeguards |
| ε > 10  | Weak          | Low-sensitivity data; DP mostly symbolic |

**δ guidance:** Set δ ≤ 1/n where n is dataset size (e.g., 1e-5 for 100k samples).

**Noise multiplier guide:**
- σ = 0.5 → fast convergence, weaker privacy (needs very small ε target)  
- σ = 1.0 → balanced, common for ε ∈ [3, 10]
- σ = 2.0 → strong privacy, needs more epochs to converge

---

## Testing

```bash
python3 -m pytest tests/ -v
# 35 tests, all pass
```

---

## References

- Abadi et al. (2016) — [Deep Learning with Differential Privacy](https://arxiv.org/abs/1607.00133)
- Mironov (2017) — [Rényi Differential Privacy of the Gaussian Mechanism](https://arxiv.org/abs/1702.07476)
- McMahan et al. (2017) — [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629)
- Balle et al. (2020) — [Hypothesis Testing Interpretations and Renormalization of Differential Privacy](https://arxiv.org/abs/1905.09982)
- Wang et al. (2019) — [Subsampled Rényi Differential Privacy and Analytical Moments Accountant](https://arxiv.org/abs/1808.00087)

---

## License

MIT — see [LICENSE](LICENSE).

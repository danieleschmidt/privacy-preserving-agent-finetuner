"""
PrivacyReport: structured privacy-utility tradeoff report.

Surfaces:
  - ε/δ consumed
  - Noise level (σ, max_grad_norm)
  - Model utility (train/val accuracy per epoch)
  - Privacy-utility tradeoff visualization (text + JSON)
  - Compliance indicators (GDPR, CCPA, HIPAA guidance)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .budget import PrivacyBudgetManager
from .trainer import TrainingResult


@dataclass
class PrivacyUtilityPoint:
    """Single point on the privacy-utility curve."""
    epsilon: float
    delta: float
    train_accuracy: Optional[float]
    val_accuracy: Optional[float]
    epoch: int


class PrivacyReport:
    """
    Generates a structured report on privacy guarantees and model utility.

    Args:
        budget_manager: The PrivacyBudgetManager used during training.
        training_results: List of TrainingResult from PrivateTrainer.fit().
        model_name: Optional label for the model (e.g., "BERT-base-privacy-ε1").
        dataset_size: Number of training samples (for δ guidance).
    """

    def __init__(
        self,
        budget_manager: PrivacyBudgetManager,
        training_results: List[TrainingResult],
        model_name: str = "model",
        dataset_size: Optional[int] = None,
    ):
        self.budget_manager = budget_manager
        self.training_results = training_results
        self.model_name = model_name
        self.dataset_size = dataset_size
        self.generated_at = datetime.utcnow().isoformat() + "Z"

    # ------------------------------------------------------------------
    # Core report generation
    # ------------------------------------------------------------------

    def generate(self) -> Dict:
        """Generate the full privacy report as a dict."""
        budget = self.budget_manager.summary()
        utility = self._utility_summary()
        tradeoff = self._privacy_utility_curve()
        compliance = self._compliance_guidance(budget["epsilon_used"], budget["delta"])

        report = {
            "model": self.model_name,
            "generated_at": self.generated_at,
            "privacy": {
                "mechanism": "DP-SGD (Gaussian mechanism + Poisson subsampling)",
                "accounting": "RDP composition → (ε, δ)-DP conversion",
                "epsilon_consumed": budget["epsilon_used"],
                "epsilon_target": budget["epsilon_target"],
                "delta": budget["delta"],
                "noise_multiplier": budget["noise_multiplier"],
                "sample_rate": budget["sample_rate"],
                "steps": budget["steps"],
                "epochs": budget["epochs"],
                "budget_remaining": budget["remaining_budget"],
                "budget_exhausted": budget["exhausted"],
            },
            "utility": utility,
            "privacy_utility_curve": [asdict(p) for p in tradeoff],
            "compliance": compliance,
            "interpretation": self._interpretation(budget["epsilon_used"]),
        }
        return report

    def print_summary(self, width: int = 70):
        """Print a human-readable summary to stdout."""
        report = self.generate()
        priv = report["privacy"]
        util = report["utility"]

        print("=" * width)
        print(f"  Privacy Report: {self.model_name}")
        print("=" * width)
        print(f"\n  {'Privacy Guarantee':}")
        print(f"    ε (epsilon):        {priv['epsilon_consumed']:.4f}  (target: {priv['epsilon_target']})")
        print(f"    δ (delta):          {priv['delta']:.2e}")
        print(f"    Noise multiplier σ: {priv['noise_multiplier']}")
        print(f"    Training steps:     {priv['steps']}")
        print(f"    Budget remaining:   {priv['budget_remaining']:.4f}")

        print(f"\n  {'Model Utility':}")
        print(f"    Best train acc:     {util['best_train_accuracy']:.3f}")
        if util['best_val_accuracy'] is not None:
            print(f"    Best val acc:       {util['best_val_accuracy']:.3f}")
        print(f"    Final train acc:    {util['final_train_accuracy']:.3f}")
        if util['final_val_accuracy'] is not None:
            print(f"    Final val acc:      {util['final_val_accuracy']:.3f}")

        print(f"\n  {'Interpretation':}")
        for line in report["interpretation"]:
            print(f"    • {line}")

        print(f"\n  {'Compliance Indicators':}")
        for domain, status in report["compliance"].items():
            print(f"    {domain}: {status}")

        print("=" * width)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.generate(), indent=indent)

    def save(self, path: str):
        """Save report as JSON file."""
        with open(path, "w") as f:
            f.write(self.to_json())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _utility_summary(self) -> Dict:
        if not self.training_results:
            return {}

        train_accs = [r.train_accuracy for r in self.training_results]
        val_accs = [r.val_accuracy for r in self.training_results if r.val_accuracy is not None]

        return {
            "epochs_trained": len(self.training_results),
            "best_train_accuracy": max(train_accs),
            "final_train_accuracy": train_accs[-1],
            "best_val_accuracy": max(val_accs) if val_accs else None,
            "final_val_accuracy": val_accs[-1] if val_accs else None,
            "train_loss_final": self.training_results[-1].train_loss,
            "val_loss_final": self.training_results[-1].val_loss,
        }

    def _privacy_utility_curve(self) -> List[PrivacyUtilityPoint]:
        return [
            PrivacyUtilityPoint(
                epsilon=r.epsilon,
                delta=r.delta,
                train_accuracy=r.train_accuracy,
                val_accuracy=r.val_accuracy,
                epoch=r.epoch,
            )
            for r in self.training_results
        ]

    def _compliance_guidance(self, epsilon: float, delta: float) -> Dict[str, str]:
        """
        Rough guidance on regulatory alignment. NOT legal advice.
        These thresholds reflect common practitioner guidance (2024).
        """
        guidance = {}

        if epsilon <= 1.0:
            gdpr = "Strong — ε≤1 aligns with strong anonymization. Risk of re-identification very low."
        elif epsilon <= 5.0:
            gdpr = "Moderate — ε≤5 acceptable for many GDPR pseudonymization use cases."
        elif epsilon <= 10.0:
            gdpr = "Weak — ε≤10 may satisfy some GDPR interpretations; document carefully."
        else:
            gdpr = "At risk — high ε; consult legal team for GDPR compliance assessment."

        if epsilon <= 2.0:
            hipaa = "Potentially compatible — low ε supports de-identification claims."
        elif epsilon <= 8.0:
            hipaa = "Borderline — consider combining with other HIPAA safeguards."
        else:
            hipaa = "Insufficient — HIPAA de-identification likely requires lower ε or additional controls."

        if self.dataset_size:
            recommended_delta = 1.0 / self.dataset_size
            if delta <= recommended_delta:
                delta_note = f"✓ δ≤1/n (n={self.dataset_size})"
            else:
                delta_note = f"⚠ δ>{recommended_delta:.2e} — consider δ≤1/n for best practice"
        else:
            delta_note = "Set δ≤1/dataset_size for best practice."

        guidance["GDPR"] = gdpr
        guidance["HIPAA"] = hipaa
        guidance["delta_guidance"] = delta_note
        guidance["note"] = "This guidance is informational only. Consult legal counsel for compliance decisions."

        return guidance

    def _interpretation(self, epsilon: float) -> List[str]:
        """Plain-language interpretation of the privacy guarantee."""
        lines = []
        if epsilon <= 1.0:
            lines.append(
                f"ε={epsilon:.2f}: Strong privacy. An adversary observing the model "
                "gains very little information about any individual training sample."
            )
        elif epsilon <= 5.0:
            lines.append(
                f"ε={epsilon:.2f}: Moderate privacy. Meaningful protection against "
                "membership inference and reconstruction attacks."
            )
        elif epsilon <= 10.0:
            lines.append(
                f"ε={epsilon:.2f}: Weak privacy. Some protection, but a determined "
                "adversary may distinguish participants. Augment with access controls."
            )
        else:
            lines.append(
                f"ε={epsilon:.2f}: Minimal formal privacy. DP guarantees are loose. "
                "Consider lower noise_multiplier budget or fewer epochs."
            )

        lines.append(
            "DP-SGD bounds worst-case information leakage — it does not guarantee "
            "individual privacy in practice for very small datasets."
        )
        return lines

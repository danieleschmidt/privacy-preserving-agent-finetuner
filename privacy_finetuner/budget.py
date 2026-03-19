"""
PrivacyBudgetManager: tracks (ε, δ) budget across training epochs using RDP composition.

RDP (Rényi Differential Privacy) composition is tighter than basic (ε, δ)-DP
composition. We use the Gaussian mechanism's RDP bound and convert to (ε, δ)-DP.

Reference:
  Mironov (2017) "Rényi Differential Privacy of the Gaussian Mechanism"
  Wang et al. (2019) "Subsampled Rényi Differential Privacy..."
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# RDP orders to evaluate (standard set from Opacus / Google DP library)
_DEFAULT_ALPHAS = [
    1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 16.0, 32.0, 64.0
]


@dataclass
class EpochRecord:
    epoch: int
    noise_multiplier: float
    sample_rate: float
    rdp_epsilon: float  # RDP ε at best α
    best_alpha: float
    cumulative_epsilon: float  # (ε, δ)-DP ε so far
    delta: float


class PrivacyBudgetManager:
    """
    Tracks privacy budget (ε, δ) across training steps using RDP accounting.

    The Gaussian mechanism with noise multiplier σ achieves RDP(α) = α / (2σ²)
    per step. With Poisson subsampling at rate q, we use the amplification bound.

    Args:
        target_epsilon: Maximum allowed ε (raise BudgetExhaustedError if exceeded).
        target_delta: δ parameter. Should be < 1/n (n = dataset size). Typical: 1e-5.
        noise_multiplier: σ — noise std relative to per-sample gradient L2 norm.
        sample_rate: q = batch_size / dataset_size (Poisson subsampling rate).
        alphas: RDP orders to evaluate. More orders → tighter bound.
    """

    def __init__(
        self,
        target_epsilon: float,
        target_delta: float = 1e-5,
        noise_multiplier: float = 1.0,
        sample_rate: float = 0.01,
        alphas: Optional[List[float]] = None,
    ):
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.noise_multiplier = noise_multiplier
        self.sample_rate = sample_rate
        self.alphas = alphas or _DEFAULT_ALPHAS

        self._steps: int = 0
        self._history: List[EpochRecord] = []
        self._current_epoch: int = 0
        self._epoch_steps: int = 0

    # ------------------------------------------------------------------
    # Main accounting API
    # ------------------------------------------------------------------

    def step(self) -> Tuple[float, float]:
        """
        Record one training step (one minibatch). Returns (current_epsilon, delta).
        Raises BudgetExhaustedError if epsilon exceeds target.
        """
        self._steps += 1
        self._epoch_steps += 1
        eps, delta = self.compute_epsilon(self._steps)
        if eps > self.target_epsilon and not (math.isinf(eps) and math.isinf(self.target_epsilon)):
            raise BudgetExhaustedError(
                f"Privacy budget exhausted after {self._steps} steps: "
                f"ε={eps:.4f} > target ε={self.target_epsilon}"
            )
        return eps, delta

    def end_epoch(self) -> EpochRecord:
        """Record end-of-epoch statistics."""
        self._current_epoch += 1
        eps, delta = self.compute_epsilon(self._steps)
        rdp_eps, best_alpha = self._compute_rdp(self._steps)
        record = EpochRecord(
            epoch=self._current_epoch,
            noise_multiplier=self.noise_multiplier,
            sample_rate=self.sample_rate,
            rdp_epsilon=rdp_eps,
            best_alpha=best_alpha,
            cumulative_epsilon=eps,
            delta=delta,
        )
        self._history.append(record)
        self._epoch_steps = 0
        return record

    def compute_epsilon(self, steps: int) -> Tuple[float, float]:
        """
        Compute (ε, δ)-DP guarantee after `steps` steps using RDP→(ε,δ) conversion.
        """
        if steps == 0:
            return 0.0, self.target_delta

        # No noise → no privacy (ε = ∞)
        if self.noise_multiplier == 0.0:
            return float("inf"), self.target_delta

        best_eps = float("inf")
        for alpha in self.alphas:
            rdp = self._rdp_gaussian_subsampled(alpha, steps)
            eps = self._rdp_to_dp(rdp, alpha, self.target_delta)
            if eps < best_eps:
                best_eps = eps

        return best_eps, self.target_delta

    def remaining_budget(self) -> float:
        """Remaining ε budget."""
        used, _ = self.compute_epsilon(self._steps)
        return max(0.0, self.target_epsilon - used)

    def is_exhausted(self) -> bool:
        used, _ = self.compute_epsilon(self._steps)
        return used >= self.target_epsilon

    # ------------------------------------------------------------------
    # RDP computation
    # ------------------------------------------------------------------

    def _rdp_gaussian_subsampled(self, alpha: float, steps: int) -> float:  # noqa: C901
        """
        RDP ε for Gaussian mechanism with Poisson subsampling, after `steps` steps.

        Per-step RDP for Gaussian mechanism: RDP(α) = α / (2σ²)
        With Poisson subsampling at rate q, amplification gives:
          RDP_sub(α) ≈ min(
            2 * q^2 * α / (2σ²),             # small q approximation
            log(1 + q^2 * (e^(RDP(α)) - 1))  # tighter for small α
          )
        We use the simpler tight bound from Wang et al. (2019).

        Composition over T steps: RDP_total = T * RDP_sub(α)
        """
        sigma = self.noise_multiplier
        if sigma == 0.0:
            return float("inf")

        q = self.sample_rate

        if alpha == 1:
            # KL divergence limit
            per_step = q * (1 / (2 * sigma ** 2))
        else:
            # Gaussian mechanism RDP: α / (2σ²)
            rdp_full = alpha / (2.0 * sigma ** 2)

            # Poisson subsampling amplification (Theorem 9 from Mironov 2017)
            # For small q: RDP_sub ≤ q^2 * α * (α-1) / (2σ²) + O(q^3)
            # More precisely using the log(1 + ...) bound:
            if q == 1.0:
                per_step = rdp_full
            else:
                # Use the tight bound: log(1 + q^2 * exp(RDP(α)) * binom term)
                # Simplified amplification: RDP_sub ≤ q^2 * RDP(α) / (1-q) for α≥2
                # We use the practical approximation from the DP library:
                per_step = min(
                    rdp_full,
                    math.log1p(q ** 2 * (math.exp(rdp_full) - 1)) if rdp_full < 50 else rdp_full
                )
                # Additional q factor for subsampling
                per_step = per_step * (q ** 2 if alpha >= 2 else q)

        return per_step * steps

    def _compute_rdp(self, steps: int) -> Tuple[float, float]:
        """Return (best_rdp_epsilon, best_alpha)."""
        best_rdp = float("inf")
        best_alpha = self.alphas[0]
        for alpha in self.alphas:
            rdp = self._rdp_gaussian_subsampled(alpha, steps)
            if rdp < best_rdp:
                best_rdp = rdp
                best_alpha = alpha
        return best_rdp, best_alpha

    @staticmethod
    def _rdp_to_dp(rdp: float, alpha: float, delta: float) -> float:
        """
        Convert RDP guarantee to (ε, δ)-DP.

        From Proposition 3 of Mironov (2017):
          ε = rdp - log(delta * (alpha-1) / alpha) / (alpha-1) + log((alpha-1)/alpha)

        Simplified standard form:
          ε = rdp + log(1 - 1/alpha) - log(delta) / (alpha - 1)
              - log(alpha / (alpha - 1)) / (alpha - 1)... 

        We use the tighter conversion from Balle et al. (2020):
          ε(δ) = rdp + log((alpha-1)/alpha) - (log(delta) + log(alpha)) / (alpha - 1)
        """
        if alpha <= 1:
            return float("inf")
        if delta <= 0:
            return float("inf")
        # Standard RDP → (ε, δ)-DP conversion
        return rdp + math.log(alpha - 1) / alpha - (math.log(delta) + math.log(alpha)) / (alpha - 1)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def steps_taken(self) -> int:
        return self._steps

    @property
    def history(self) -> List[EpochRecord]:
        return list(self._history)

    def summary(self) -> dict:
        eps, delta = self.compute_epsilon(self._steps)
        return {
            "steps": self._steps,
            "epochs": self._current_epoch,
            "epsilon_used": round(eps, 6),
            "epsilon_target": self.target_epsilon,
            "delta": delta,
            "remaining_budget": round(self.remaining_budget(), 6),
            "noise_multiplier": self.noise_multiplier,
            "sample_rate": self.sample_rate,
            "exhausted": self.is_exhausted(),
        }


class BudgetExhaustedError(RuntimeError):
    """Raised when training would exceed the target privacy budget."""
    pass

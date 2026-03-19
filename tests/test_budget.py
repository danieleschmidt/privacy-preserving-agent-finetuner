"""Tests for PrivacyBudgetManager."""

import math
import pytest
from privacy_finetuner.budget import PrivacyBudgetManager, BudgetExhaustedError


class TestPrivacyBudgetManager:

    def test_zero_steps(self):
        mgr = PrivacyBudgetManager(target_epsilon=10.0, target_delta=1e-5, noise_multiplier=1.0, sample_rate=0.01)
        eps, delta = mgr.compute_epsilon(0)
        assert eps == 0.0
        assert delta == 1e-5

    def test_epsilon_increases_with_steps(self):
        mgr = PrivacyBudgetManager(target_epsilon=100.0, target_delta=1e-5, noise_multiplier=1.0, sample_rate=0.01)
        eps1, _ = mgr.compute_epsilon(10)
        eps2, _ = mgr.compute_epsilon(100)
        assert eps2 > eps1

    def test_epsilon_decreases_with_higher_noise(self):
        """More noise → better privacy → lower ε for same steps."""
        mgr1 = PrivacyBudgetManager(target_epsilon=100.0, target_delta=1e-5, noise_multiplier=1.0, sample_rate=0.01)
        mgr2 = PrivacyBudgetManager(target_epsilon=100.0, target_delta=1e-5, noise_multiplier=3.0, sample_rate=0.01)
        eps1, _ = mgr1.compute_epsilon(100)
        eps2, _ = mgr2.compute_epsilon(100)
        assert eps2 < eps1

    def test_step_raises_on_exhaustion(self):
        mgr = PrivacyBudgetManager(target_epsilon=0.001, target_delta=1e-5, noise_multiplier=0.5, sample_rate=0.1)
        with pytest.raises(BudgetExhaustedError):
            for _ in range(10000):
                mgr.step()

    def test_end_epoch_records_history(self):
        mgr = PrivacyBudgetManager(target_epsilon=100.0, target_delta=1e-5, noise_multiplier=1.0, sample_rate=0.01)
        for _ in range(10):
            mgr.step()
        record = mgr.end_epoch()
        assert record.epoch == 1
        assert record.cumulative_epsilon > 0
        assert record.delta == 1e-5
        assert len(mgr.history) == 1

    def test_remaining_budget(self):
        mgr = PrivacyBudgetManager(target_epsilon=10.0, target_delta=1e-5, noise_multiplier=1.0, sample_rate=0.01)
        remaining_initial = mgr.remaining_budget()
        for _ in range(50):
            mgr.step()
        remaining_after = mgr.remaining_budget()
        assert remaining_after < remaining_initial

    def test_summary_keys(self):
        mgr = PrivacyBudgetManager(target_epsilon=5.0, target_delta=1e-5, noise_multiplier=1.0, sample_rate=0.01)
        summary = mgr.summary()
        for key in ["steps", "epochs", "epsilon_used", "epsilon_target", "delta", "remaining_budget", "exhausted"]:
            assert key in summary

    def test_not_exhausted_initially(self):
        mgr = PrivacyBudgetManager(target_epsilon=10.0, target_delta=1e-5, noise_multiplier=1.0, sample_rate=0.01)
        assert not mgr.is_exhausted()

    def test_delta_preserved(self):
        mgr = PrivacyBudgetManager(target_epsilon=10.0, target_delta=1e-6, noise_multiplier=1.0, sample_rate=0.01)
        _, delta = mgr.compute_epsilon(100)
        assert delta == 1e-6

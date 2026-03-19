"""Tests for PrivacyReport."""

import pytest
import torch
import torch.nn as nn

from privacy_finetuner.budget import PrivacyBudgetManager
from privacy_finetuner.dataset import PrivateDataset
from privacy_finetuner.trainer import PrivateTrainer, TrainingResult
from privacy_finetuner.report import PrivacyReport


def make_mock_results(n=3):
    return [
        TrainingResult(
            epoch=i+1,
            train_loss=1.0 - 0.1*i,
            train_accuracy=0.5 + 0.05*i,
            val_loss=1.1 - 0.1*i,
            val_accuracy=0.45 + 0.05*i,
            epsilon=0.5*(i+1),
            delta=1e-5,
            noise_multiplier=1.0,
            max_grad_norm=1.0,
        )
        for i in range(n)
    ]


class TestPrivacyReport:

    def test_generate_returns_dict(self):
        mgr = PrivacyBudgetManager(target_epsilon=5.0, target_delta=1e-5, noise_multiplier=1.0, sample_rate=0.01)
        for _ in range(30):
            mgr.step()
        mgr.end_epoch()
        report = PrivacyReport(mgr, make_mock_results(), model_name="test-model")
        result = report.generate()
        assert isinstance(result, dict)
        assert "privacy" in result
        assert "utility" in result
        assert "compliance" in result

    def test_utility_best_accuracy(self):
        mgr = PrivacyBudgetManager(target_epsilon=5.0, target_delta=1e-5, noise_multiplier=1.0, sample_rate=0.01)
        results = make_mock_results(3)
        report = PrivacyReport(mgr, results)
        d = report.generate()
        assert d["utility"]["best_train_accuracy"] == pytest.approx(0.60, abs=0.01)
        assert d["utility"]["final_train_accuracy"] == pytest.approx(0.60, abs=0.01)

    def test_to_json(self):
        import json
        mgr = PrivacyBudgetManager(target_epsilon=5.0, target_delta=1e-5, noise_multiplier=1.0, sample_rate=0.01)
        report = PrivacyReport(mgr, make_mock_results())
        j = report.to_json()
        parsed = json.loads(j)
        assert "privacy" in parsed

    def test_compliance_keys(self):
        mgr = PrivacyBudgetManager(target_epsilon=5.0, target_delta=1e-5, noise_multiplier=1.0, sample_rate=0.01)
        report = PrivacyReport(mgr, make_mock_results())
        d = report.generate()
        assert "GDPR" in d["compliance"]
        assert "HIPAA" in d["compliance"]

    def test_strong_privacy_interpretation(self):
        mgr = PrivacyBudgetManager(target_epsilon=1.0, target_delta=1e-5, noise_multiplier=2.0, sample_rate=0.01)
        # Simulate very few steps so ε stays < 1
        report = PrivacyReport(mgr, make_mock_results())
        d = report.generate()
        assert len(d["interpretation"]) >= 1

    def test_print_summary_runs(self, capsys):
        mgr = PrivacyBudgetManager(target_epsilon=5.0, target_delta=1e-5, noise_multiplier=1.0, sample_rate=0.01)
        for _ in range(10):
            mgr.step()
        mgr.end_epoch()
        report = PrivacyReport(mgr, make_mock_results(), model_name="my-model")
        report.print_summary()
        captured = capsys.readouterr()
        assert "my-model" in captured.out
        assert "epsilon" in captured.out.lower()

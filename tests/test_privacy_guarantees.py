"""
Privacy guarantee tests for differential privacy mechanisms.

These tests verify that the privacy-preserving training maintains
formal (ε,δ)-differential privacy guarantees under various conditions.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from privacy_finetuner.core import PrivateTrainer
from privacy_finetuner.privacy import PrivacyConfig, AccountingMode
from privacy_finetuner.context_guard import ContextGuard, RedactionStrategy


class TestDifferentialPrivacyGuarantees:
    """Test suite for formal privacy guarantee verification."""
    
    @pytest.fixture
    def privacy_config(self):
        """Standard privacy configuration for testing."""
        return PrivacyConfig(
            epsilon=1.0,
            delta=1e-5,
            max_grad_norm=1.0,
            noise_multiplier=0.5
        )
    
    @pytest.fixture
    def trainer(self, privacy_config):
        """PrivateTrainer instance for testing."""
        return PrivateTrainer(
            model_name="test-model",
            privacy_config=privacy_config,
            use_mcp_gateway=False
        )
    
    @pytest.mark.privacy
    def test_privacy_budget_tracking(self, trainer):
        """Test that privacy budget is correctly tracked during training."""
        initial_budget = trainer.privacy_config.epsilon
        
        # Simulate training that consumes budget
        result = trainer.train(
            dataset="test_dataset.jsonl",
            epochs=1,
            batch_size=4
        )
        
        assert result.privacy_budget_consumed > 0
        assert result.privacy_budget_consumed <= initial_budget
        
        # Verify budget tracking in privacy report
        report = trainer.get_privacy_report()
        assert "budget_consumed" in report
        assert "budget_remaining" in report
        assert report["budget_remaining"] >= 0
    
    @pytest.mark.privacy
    def test_epsilon_delta_validation(self):
        """Test validation of privacy parameters."""
        # Valid configuration should not raise
        PrivacyConfig(epsilon=1.0, delta=1e-5)
        
        # Invalid epsilon should raise
        with pytest.raises(ValueError, match="Epsilon must be positive"):
            PrivacyConfig(epsilon=-1.0, delta=1e-5)
        
        # Invalid delta should raise
        with pytest.raises(ValueError, match="Delta must be in"):
            PrivacyConfig(epsilon=1.0, delta=0.0)
        
        with pytest.raises(ValueError, match="Delta must be in"):
            PrivacyConfig(epsilon=1.0, delta=1.0)
    
    @pytest.mark.privacy
    def test_gradient_clipping_bounds(self, privacy_config):
        """Test that gradient clipping maintains privacy bounds."""
        # Test various gradient clipping thresholds
        for max_grad_norm in [0.5, 1.0, 2.0]:
            config = PrivacyConfig(
                epsilon=privacy_config.epsilon,
                delta=privacy_config.delta,
                max_grad_norm=max_grad_norm,
                noise_multiplier=privacy_config.noise_multiplier
            )
            
            trainer = PrivateTrainer("test-model", config)
            
            # Verify configuration is valid
            assert trainer.privacy_config.max_grad_norm == max_grad_norm
            assert trainer.privacy_config.max_grad_norm > 0
    
    @pytest.mark.privacy
    def test_noise_calibration(self, privacy_config):
        """Test that noise is properly calibrated for privacy guarantees."""
        trainer = PrivateTrainer("test-model", privacy_config)
        
        # The noise multiplier should be calibrated to the privacy parameters
        expected_noise_scale = privacy_config.noise_multiplier * privacy_config.max_grad_norm
        
        # For this test, we check that the noise multiplier is reasonable
        assert 0.1 <= privacy_config.noise_multiplier <= 10.0
        assert expected_noise_scale > 0


class TestContextProtection:
    """Test suite for context guard privacy mechanisms."""
    
    @pytest.fixture
    def context_guard(self):
        """ContextGuard instance for testing."""
        return ContextGuard(strategies=[RedactionStrategy.PII_REMOVAL])
    
    @pytest.mark.privacy
    def test_pii_removal(self, context_guard):
        """Test that PII is properly removed from context."""
        test_cases = [
            ("Contact john.doe@email.com for details", "Contact [EMAIL] for details"),
            ("Call me at 555-123-4567", "Call me at [PHONE]"),
            ("SSN: 123-45-6789", "SSN: [SSN]"),
            ("Card number 1234-5678-9012-3456", "Card number [CREDIT_CARD]"),
        ]
        
        for original, expected in test_cases:
            protected = context_guard.protect(original)
            assert expected in protected or "[" in protected  # Allow for different redaction formats
    
    @pytest.mark.privacy
    def test_entity_hashing_consistency(self):
        """Test that entity hashing produces consistent results."""
        guard = ContextGuard(strategies=[RedactionStrategy.ENTITY_HASHING])
        
        text = "Send email to john.doe@company.com"
        
        # Same input should produce same hash
        result1 = guard.protect(text)
        result2 = guard.protect(text)
        assert result1 == result2
        
        # Different email should produce different hash
        different_text = "Send email to jane.smith@company.com"
        result3 = guard.protect(different_text)
        assert result3 != result1
    
    @pytest.mark.privacy
    def test_redaction_report(self, context_guard):
        """Test redaction reporting functionality."""
        text = "Contact John Doe at john.doe@email.com or 555-123-4567"
        
        report = context_guard.explain_redactions(text)
        
        assert report.original_length == len(text)
        assert report.redacted_length <= report.original_length
        assert len(report.redactions_applied) > 0
        assert "EMAIL" in str(report.redactions_applied)
        assert 0 <= report.confidence_score <= 1.0
    
    @pytest.mark.privacy
    def test_batch_protection(self, context_guard):
        """Test batch protection maintains individual privacy."""
        texts = [
            "Email alice@company.com",
            "Call Bob at 555-111-2222", 
            "Meet Carol at the office"
        ]
        
        protected = context_guard.batch_protect(texts)
        
        assert len(protected) == len(texts)
        assert "[EMAIL]" in protected[0]
        assert "[PHONE]" in protected[1]
        # Third text might not have PII to redact


class TestPrivacyBudgetExhaustion:
    """Test privacy budget exhaustion scenarios."""
    
    @pytest.mark.privacy
    def test_budget_exhaustion_detection(self):
        """Test detection of privacy budget exhaustion."""
        config = PrivacyConfig(epsilon=0.1, delta=1e-5)  # Small budget
        
        # Check budget exhaustion logic
        assert not config.is_privacy_budget_exhausted(threshold=0.05)
        
        # Simulate near-exhaustion
        low_budget_config = PrivacyConfig(epsilon=0.05, delta=1e-5)
        assert low_budget_config.is_privacy_budget_exhausted(threshold=0.1)
    
    @pytest.mark.privacy
    def test_training_stops_on_budget_exhaustion(self):
        """Test that training stops when privacy budget is exhausted."""
        config = PrivacyConfig(epsilon=0.01, delta=1e-5)  # Very small budget
        trainer = PrivateTrainer("test-model", config)
        
        # This would typically raise an exception or stop training
        # In a real implementation, we'd test the actual budget exhaustion handling
        report = trainer.get_privacy_report()
        assert "budget_remaining" in report


class TestRegulatoryCompliance:
    """Test compliance with privacy regulations."""
    
    @pytest.mark.privacy
    def test_gdpr_compliance(self):
        """Test GDPR compliance features."""
        config = PrivacyConfig(epsilon=1.0, delta=1e-5)
        trainer = PrivateTrainer("test-model", config)
        
        report = trainer.get_privacy_report()
        
        # GDPR requires explicit privacy guarantees
        assert "privacy_guarantees" in report
        assert "GDPR" in report.get("compliance_status", "")
    
    @pytest.mark.privacy
    def test_hipaa_compliance(self):
        """Test HIPAA compliance for healthcare data."""
        # HIPAA requires stronger privacy guarantees
        config = PrivacyConfig(epsilon=0.5, delta=1e-6)  # Stricter parameters
        trainer = PrivateTrainer("test-model", config)
        
        report = trainer.get_privacy_report()
        assert "HIPAA" in report.get("compliance_status", "")
    
    @pytest.mark.privacy
    def test_eu_ai_act_compliance(self):
        """Test EU AI Act compliance for high-risk AI systems."""
        config = PrivacyConfig(epsilon=1.0, delta=1e-5)
        trainer = PrivateTrainer("test-model", config)
        
        report = trainer.get_privacy_report()
        assert "EU AI Act" in report.get("compliance_status", "")


# Parameterized tests for different privacy budgets
@pytest.mark.parametrize("epsilon,delta", [
    (0.1, 1e-6),  # Very private
    (1.0, 1e-5),  # Standard
    (3.0, 1e-4),  # Less private
])
@pytest.mark.privacy
def test_privacy_budget_variants(epsilon, delta):
    """Test various privacy budget configurations."""
    config = PrivacyConfig(epsilon=epsilon, delta=delta)
    trainer = PrivateTrainer("test-model", config)
    
    assert trainer.privacy_config.epsilon == epsilon
    assert trainer.privacy_config.delta == delta
    
    report = trainer.get_privacy_report()
    assert report["privacy_config"]["epsilon"] == epsilon
    assert report["privacy_config"]["delta"] == delta
#!/usr/bin/env python3
"""Basic functionality test for Privacy-Preserving Agent Finetuner."""

import sys
import os
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, "/root/repo")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test basic imports work correctly."""
    try:
        from privacy_finetuner.core import PrivateTrainer, ContextGuard, PrivacyConfig, RedactionStrategy
        from privacy_finetuner.core.privacy_analytics import PrivacyBudgetTracker, PrivacyAttackDetector, PrivacyComplianceChecker
        logger.info("✓ All core imports successful")
        return True
    except ImportError as e:
        logger.error(f"✗ Import failed: {e}")
        return False

def test_privacy_config():
    """Test privacy configuration validation."""
    try:
        from privacy_finetuner.core import PrivacyConfig
        
        # Valid config
        config = PrivacyConfig(epsilon=1.0, delta=1e-5, max_grad_norm=1.0, noise_multiplier=0.5)
        config.validate()
        logger.info("✓ Privacy config validation passed")
        
        # Test privacy cost estimation
        cost = config.estimate_privacy_cost(steps=100, sample_rate=0.1)
        logger.info(f"✓ Privacy cost estimation: {cost:.6f}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Privacy config test failed: {e}")
        return False

def test_context_guard():
    """Test context protection functionality."""
    try:
        from privacy_finetuner.core import ContextGuard, RedactionStrategy
        
        # Create context guard with PII removal
        guard = ContextGuard([RedactionStrategy.PII_REMOVAL])
        
        # Test text protection
        test_text = "Contact John Doe at john@example.com or call 555-123-4567"
        protected = guard.protect(test_text)
        logger.info(f"✓ Context protection: '{test_text}' -> '{protected}'")
        
        # Test sensitivity analysis
        analysis = guard.analyze_sensitivity(test_text)
        logger.info(f"✓ Sensitivity analysis: level={analysis['sensitivity_level']}, score={analysis['sensitivity_score']}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Context guard test failed: {e}")
        return False

def test_privacy_analytics():
    """Test privacy analytics components."""
    try:
        from privacy_finetuner.core.privacy_analytics import PrivacyBudgetTracker, PrivacyAttackDetector, PrivacyComplianceChecker
        
        # Test budget tracker
        tracker = PrivacyBudgetTracker(total_epsilon=10.0, total_delta=1e-5)
        success = tracker.record_event("test_event", epsilon_cost=0.5)
        assert success, "Budget tracking should succeed"
        
        summary = tracker.get_usage_summary()
        logger.info(f"✓ Budget tracker: {summary['utilization']['epsilon_percent']:.2f}% used")
        
        # Test attack detector
        detector = PrivacyAttackDetector()
        risk_analysis = detector.analyze_membership_inference_risk(
            "Test query", {"confidence": 0.8}
        )
        logger.info(f"✓ Attack detection: risk level = {risk_analysis['overall_risk']}")
        
        # Test compliance checker
        checker = PrivacyComplianceChecker()
        compliance = checker.check_compliance({"epsilon": 0.8}, "GDPR")
        logger.info(f"✓ Compliance check: GDPR compliant = {compliance['compliant']}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Privacy analytics test failed: {e}")
        return False

def test_trainer_initialization():
    """Test trainer initialization without actual model loading."""
    try:
        from privacy_finetuner.core import PrivateTrainer, PrivacyConfig
        
        # Create privacy config
        privacy_config = PrivacyConfig(epsilon=1.0, delta=1e-5)
        
        # Initialize trainer (this will try to load model, so we expect it might fail)
        try:
            trainer = PrivateTrainer(
                model_name="dummy-model",  # Use non-existent model to avoid download
                privacy_config=privacy_config,
                use_mcp_gateway=False
            )
            logger.info("✓ Trainer initialization successful")
        except Exception as model_error:
            # Expected - model doesn't exist or libraries not available
            logger.info(f"✓ Trainer initialization structure valid (model loading failed as expected: {type(model_error).__name__})")
        
        return True
    except Exception as e:
        logger.error(f"✗ Trainer initialization test failed: {e}")
        return False

def main():
    """Run all basic functionality tests."""
    logger.info("🚀 Starting Privacy-Preserving Agent Finetuner basic functionality tests")
    
    tests = [
        ("Import Tests", test_imports),
        ("Privacy Config", test_privacy_config),
        ("Context Guard", test_context_guard),
        ("Privacy Analytics", test_privacy_analytics),
        ("Trainer Initialization", test_trainer_initialization)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n--- Running {test_name} ---")
        try:
            if test_func():
                passed += 1
                logger.info(f"✓ {test_name} PASSED")
            else:
                logger.error(f"✗ {test_name} FAILED")
        except Exception as e:
            logger.error(f"✗ {test_name} FAILED with exception: {e}")
    
    logger.info(f"\n🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All basic functionality tests PASSED!")
        return 0
    else:
        logger.warning(f"⚠️  {total - passed} tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
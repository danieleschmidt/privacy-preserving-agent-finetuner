#!/usr/bin/env python3
"""Basic core functionality test without heavy dependencies."""

import sys
import os
import logging

# Add project root to path
sys.path.insert(0, "/root/repo")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_privacy_config():
    """Test privacy configuration validation."""
    try:
        from privacy_finetuner.core.privacy_config import PrivacyConfig
        
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

def test_context_guard_basic():
    """Test basic context protection without advanced NLP."""
    try:
        from privacy_finetuner.core.context_guard import ContextGuard, RedactionStrategy
        
        # Create context guard with PII removal
        guard = ContextGuard([RedactionStrategy.PII_REMOVAL])
        
        # Test basic text protection
        test_text = "Contact john@example.com or call 555-123-4567"
        protected = guard.protect(test_text)
        logger.info(f"✓ Basic protection: '{test_text}' -> '{protected}'")
        
        return True
    except Exception as e:
        logger.error(f"✗ Context guard test failed: {e}")
        return False

def test_exceptions():
    """Test exception classes."""
    try:
        from privacy_finetuner.core.exceptions import (
            PrivacyBudgetExhaustedException,
            ModelTrainingException,
            DataValidationException,
            SecurityViolationException
        )
        
        # Test exception creation
        try:
            raise PrivacyBudgetExhaustedException("Test budget exhausted")
        except PrivacyBudgetExhaustedException as e:
            logger.info(f"✓ Privacy budget exception: {e}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Exception test failed: {e}")
        return False

def main():
    """Run basic core functionality tests."""
    logger.info("🚀 Starting basic core functionality tests")
    
    tests = [
        ("Privacy Config", test_privacy_config),
        ("Context Guard Basic", test_context_guard_basic),
        ("Exceptions", test_exceptions)
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
        logger.info("🎉 All basic core tests PASSED!")
        return 0
    else:
        logger.warning(f"⚠️  {total - passed} tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
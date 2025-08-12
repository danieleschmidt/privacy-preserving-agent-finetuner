#!/usr/bin/env python3
"""
Simple core functionality tests that work without external dependencies.
"""

import logging
import sys
import os
sys.path.append('/root/repo')

def test_basic_imports():
    """Test basic module imports."""
    print("Testing basic imports...")
    try:
        from privacy_finetuner.core.exceptions import (
            PrivacyBudgetExhaustedException, ModelTrainingException, 
            DataValidationException, SecurityViolationException
        )
        print("✅ Exception imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_privacy_config_basic():
    """Test privacy configuration basic functionality."""
    print("Testing privacy config...")
    try:
        from privacy_finetuner.core.privacy_config import PrivacyConfig
        
        config = PrivacyConfig(
            epsilon=1.0,
            delta=1e-5,
            max_grad_norm=1.0,
            noise_multiplier=0.5
        )
        config.validate()
        print("✅ Privacy config validation successful")
        return True
    except Exception as e:
        print(f"❌ Privacy config failed: {e}")
        return False

def test_exception_creation():
    """Test exception creation."""
    print("Testing exception creation...")
    try:
        from privacy_finetuner.core.exceptions import PrivacyBudgetExhaustedException
        
        exc = PrivacyBudgetExhaustedException(spent_epsilon=1.5, total_epsilon=1.0)
        print(f"✅ Exception created: {exc}")
        return True
    except Exception as e:
        print(f"❌ Exception creation failed: {e}")
        return False

def test_context_guard_basic():
    """Test context guard basic functionality."""
    print("Testing context guard basics...")
    try:
        from privacy_finetuner.core.context_guard import ContextGuard
        
        guard = ContextGuard()
        test_text = "Hello John Doe, your SSN is 123-45-6789"
        
        # This should work even with limited dependencies
        protected = guard.protect(test_text)
        print(f"✅ Context guard protection: '{protected}'")
        return True
    except Exception as e:
        print(f"❌ Context guard failed: {e}")
        return False

def main():
    """Run simple tests."""
    print("🚀 Starting simple core functionality tests")
    print("=" * 60)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Privacy Config", test_privacy_config_basic),
        ("Exception Creation", test_exception_creation),
        ("Context Guard Basic", test_context_guard_basic)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print(f"\n{'='*60}")
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests PASSED!")
        return 0
    else:
        print(f"⚠️ {total - passed} tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
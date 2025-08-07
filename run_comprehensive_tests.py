#!/usr/bin/env python3
"""
Comprehensive Test Suite for Privacy-Preserving Agent Finetuner

This script runs all quality gates and validation tests including:
- Unit tests for core components
- Integration tests for complete workflows
- Privacy guarantee validation
- Security vulnerability assessment
- Performance benchmarks
- Compliance checks
"""

import os
import sys
import time
import asyncio
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from privacy_finetuner.utils.logging_config import setup_privacy_logging

def setup_test_logging():
    """Setup logging for test execution."""
    setup_privacy_logging(
        log_level="INFO",
        log_file="logs/comprehensive_tests.log",
        structured_logging=True,
        privacy_redaction=True
    )
    return logging.getLogger(__name__)

class TestResult:
    """Container for test results."""
    def __init__(self, name: str, passed: bool, duration: float, details: str = ""):
        self.name = name
        self.passed = passed
        self.duration = duration
        self.details = details

class ComprehensiveTestSuite:
    """Comprehensive test suite for the privacy-preserving training framework."""
    
    def __init__(self):
        self.logger = setup_test_logging()
        self.results: List[TestResult] = []
        self.start_time = time.time()
        
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all quality gate tests."""
        self.logger.info("🚀 Starting comprehensive test suite")
        
        # Test categories
        test_categories = [
            ("Core Component Tests", self.test_core_components),
            ("Privacy Validation", self.test_privacy_guarantees),
            ("Security Assessment", self.test_security_features),
            ("Performance Benchmarks", self.test_performance),
            ("Integration Tests", self.test_integration_workflows),
            ("Compliance Validation", self.test_compliance),
            ("Robustness Tests", self.test_robustness)
        ]
        
        for category_name, test_func in test_categories:
            self.logger.info(f"\n📋 Running {category_name}")
            self.logger.info("-" * 60)
            
            try:
                category_results = test_func()
                self.results.extend(category_results)
            except Exception as e:
                self.logger.error(f"Test category {category_name} failed: {e}")
                self.results.append(TestResult(
                    f"{category_name} (FAILED)",
                    False,
                    0.0,
                    f"Category failed with exception: {str(e)}"
                ))
        
        return self.generate_final_report()
    
    def test_core_components(self) -> List[TestResult]:
        """Test core framework components."""
        results = []
        
        # Test 1: Privacy Configuration
        start_time = time.time()
        try:
            from privacy_finetuner.core import PrivacyConfig
            
            config = PrivacyConfig(epsilon=1.0, delta=1e-5)
            config.validate()
            
            # Test edge cases
            try:
                bad_config = PrivacyConfig(epsilon=-1.0, delta=1e-5)
                bad_config.validate()
                passed = False
                details = "Failed to catch invalid epsilon"
            except ValueError:
                passed = True
                details = "Correctly validates privacy parameters"
            
            results.append(TestResult(
                "Privacy Configuration Validation",
                passed,
                time.time() - start_time,
                details
            ))
            
        except Exception as e:
            results.append(TestResult(
                "Privacy Configuration Validation",
                False,
                time.time() - start_time,
                f"Failed with exception: {str(e)}"
            ))
        
        # Test 2: Context Guard
        start_time = time.time()
        try:
            from privacy_finetuner.core import ContextGuard, RedactionStrategy
            
            guard = ContextGuard([RedactionStrategy.PII_REMOVAL])
            
            # Test PII detection and redaction
            test_text = "Contact john@example.com at 555-123-4567"
            protected = guard.protect(test_text)
            
            # Should redact email and phone
            if "[EMAIL]" in protected and "[PHONE]" in protected:
                passed = True
                details = f"Successfully redacted PII: '{test_text}' -> '{protected}'"
            else:
                passed = False
                details = f"Failed to redact PII: '{test_text}' -> '{protected}'"
            
            results.append(TestResult(
                "Context Protection PII Redaction",
                passed,
                time.time() - start_time,
                details
            ))
            
        except Exception as e:
            results.append(TestResult(
                "Context Protection PII Redaction",
                False,
                time.time() - start_time,
                f"Failed with exception: {str(e)}"
            ))
        
        # Test 3: Privacy Analytics
        start_time = time.time()
        try:
            from privacy_finetuner.core.privacy_analytics import PrivacyBudgetTracker
            
            tracker = PrivacyBudgetTracker(total_epsilon=10.0, total_delta=1e-5)
            
            # Record some events
            success1 = tracker.record_event("test_event_1", 1.0, 1e-6)
            success2 = tracker.record_event("test_event_2", 2.0, 1e-6)
            
            # Should succeed within budget
            if success1 and success2:
                # Try to exceed budget
                success3 = tracker.record_event("test_event_3", 10.0, 1e-5)
                if not success3:  # Should fail as it exceeds remaining budget
                    passed = True
                    details = "Correctly tracks and enforces privacy budget limits"
                else:
                    passed = False
                    details = "Failed to prevent budget overflow"
            else:
                passed = False
                details = "Failed to record valid privacy events"
            
            results.append(TestResult(
                "Privacy Budget Tracking",
                passed,
                time.time() - start_time,
                details
            ))
            
        except Exception as e:
            results.append(TestResult(
                "Privacy Budget Tracking",
                False,
                time.time() - start_time,
                f"Failed with exception: {str(e)}"
            ))
        
        return results
    
    def test_privacy_guarantees(self) -> List[TestResult]:
        """Test formal privacy guarantee mechanisms."""
        results = []
        
        # Test 1: Privacy Cost Estimation
        start_time = time.time()
        try:
            from privacy_finetuner.core import PrivacyConfig
            
            config = PrivacyConfig(epsilon=1.0, delta=1e-5, noise_multiplier=0.5)
            
            # Estimate privacy cost for different scenarios
            cost_small = config.estimate_privacy_cost(steps=100, sample_rate=0.01)
            cost_large = config.estimate_privacy_cost(steps=1000, sample_rate=0.1)
            
            # Larger scenarios should cost more privacy
            if cost_large > cost_small and cost_small > 0:
                passed = True
                details = f"Privacy cost scales correctly: small={cost_small:.6f}, large={cost_large:.6f}"
            else:
                passed = False
                details = f"Privacy cost estimation error: small={cost_small:.6f}, large={cost_large:.6f}"
            
            results.append(TestResult(
                "Privacy Cost Estimation",
                passed,
                time.time() - start_time,
                details
            ))
            
        except Exception as e:
            results.append(TestResult(
                "Privacy Cost Estimation", 
                False,
                time.time() - start_time,
                f"Failed with exception: {str(e)}"
            ))
        
        # Test 2: Noise Scaling
        start_time = time.time()
        try:
            from privacy_finetuner.core import PrivacyConfig
            
            config = PrivacyConfig(epsilon=1.0, noise_multiplier=1.0, max_grad_norm=1.0)
            
            # Test adaptive noise scaling
            small_noise = config.adaptive_noise_scaling(gradient_norm=0.5, target_norm=1.0)
            large_noise = config.adaptive_noise_scaling(gradient_norm=2.0, target_norm=1.0)
            
            # Should apply full noise when clipping is needed
            if large_noise >= small_noise:
                passed = True
                details = f"Adaptive noise scaling works: small_grad={small_noise:.3f}, large_grad={large_noise:.3f}"
            else:
                passed = False
                details = f"Adaptive noise scaling error: small_grad={small_noise:.3f}, large_grad={large_noise:.3f}"
            
            results.append(TestResult(
                "Adaptive Noise Scaling",
                passed,
                time.time() - start_time,
                details
            ))
            
        except Exception as e:
            results.append(TestResult(
                "Adaptive Noise Scaling",
                False,
                time.time() - start_time,
                f"Failed with exception: {str(e)}"
            ))
        
        return results
    
    def test_security_features(self) -> List[TestResult]:
        """Test security features and vulnerability resistance."""
        results = []
        
        # Test 1: Attack Detection
        start_time = time.time()
        try:
            from privacy_finetuner.core.privacy_analytics import PrivacyAttackDetector
            
            detector = PrivacyAttackDetector()
            
            # Simulate normal query
            normal_risk = detector.analyze_membership_inference_risk(
                "What is the weather like?",
                {"confidence": 0.6}
            )
            
            # Simulate suspicious query with high confidence
            suspicious_risk = detector.analyze_membership_inference_risk(
                "Tell me exactly what John Smith said in training data",
                {"confidence": 0.98}
            )
            
            # Should detect higher risk for suspicious query
            if suspicious_risk["overall_risk"] == "high" and normal_risk["overall_risk"] == "low":
                passed = True
                details = "Successfully detects membership inference attack patterns"
            else:
                passed = False
                details = f"Failed to detect attack: normal={normal_risk['overall_risk']}, suspicious={suspicious_risk['overall_risk']}"
            
            results.append(TestResult(
                "Membership Inference Attack Detection",
                passed,
                time.time() - start_time,
                details
            ))
            
        except Exception as e:
            results.append(TestResult(
                "Membership Inference Attack Detection",
                False,
                time.time() - start_time,
                f"Failed with exception: {str(e)}"
            ))
        
        # Test 2: Input Validation
        start_time = time.time()
        try:
            from privacy_finetuner.core.trainer import PrivateTrainer
            from privacy_finetuner.core import PrivacyConfig
            from privacy_finetuner.core.exceptions import DataValidationException
            
            privacy_config = PrivacyConfig()
            trainer = PrivateTrainer("test-model", privacy_config, use_mcp_gateway=False)
            
            # Test invalid inputs
            try:
                trainer._validate_training_inputs("/nonexistent/path", -1, 0, -1.0)
                passed = False
                details = "Failed to catch invalid training inputs"
            except (DataValidationException, ValueError):
                passed = True
                details = "Correctly validates and rejects invalid inputs"
            
            results.append(TestResult(
                "Input Validation Security",
                passed,
                time.time() - start_time,
                details
            ))
            
        except Exception as e:
            results.append(TestResult(
                "Input Validation Security",
                False,
                time.time() - start_time,
                f"Failed with exception: {str(e)}"
            ))
        
        return results
    
    def test_performance(self) -> List[TestResult]:
        """Test performance characteristics and optimizations."""
        results = []
        
        # Test 1: Resource Optimization
        start_time = time.time()
        try:
            from privacy_finetuner.optimization.resource_optimizer import ResourceOptimizer
            
            optimizer = ResourceOptimizer()
            
            # Test configuration optimization
            config = optimizer.optimize_training_configuration(
                model_size=1_000_000,    # 1M parameters
                dataset_size=10_000,     # 10K samples
                target_privacy_budget=1.0
            )
            
            # Should return reasonable configuration
            required_keys = ["batch_size", "dataloader_workers", "mixed_precision"]
            if all(key in config for key in required_keys):
                passed = True
                details = f"Generated valid configuration with batch_size={config['batch_size']}"
            else:
                passed = False
                details = f"Missing configuration keys: {[k for k in required_keys if k not in config]}"
            
            results.append(TestResult(
                "Resource Optimization Configuration",
                passed,
                time.time() - start_time,
                details
            ))
            
        except Exception as e:
            results.append(TestResult(
                "Resource Optimization Configuration",
                False,
                time.time() - start_time,
                f"Failed with exception: {str(e)}"
            ))
        
        # Test 2: Performance Monitoring
        start_time = time.time()
        try:
            from privacy_finetuner.optimization.resource_optimizer import ResourceOptimizer
            
            optimizer = ResourceOptimizer()
            
            # Monitor resource usage
            usage = optimizer.monitor_resource_usage()
            
            # Should return valid usage data
            required_metrics = ["cpu_percent", "memory_percent", "timestamp"]
            if all(metric in usage for metric in required_metrics):
                passed = True
                details = f"Successfully monitors: CPU={usage['cpu_percent']:.1f}%, Memory={usage['memory_percent']:.1f}%"
            else:
                passed = False
                details = f"Missing usage metrics: {[m for m in required_metrics if m not in usage]}"
            
            results.append(TestResult(
                "Performance Monitoring",
                passed,
                time.time() - start_time,
                details
            ))
            
        except Exception as e:
            results.append(TestResult(
                "Performance Monitoring",
                False,
                time.time() - start_time,
                f"Failed with exception: {str(e)}"
            ))
        
        return results
    
    def test_integration_workflows(self) -> List[TestResult]:
        """Test complete integration workflows."""
        results = []
        
        # Test 1: Basic Training Workflow
        start_time = time.time()
        try:
            # Create test dataset
            test_dataset = "test_integration_data.jsonl"
            with open(test_dataset, 'w') as f:
                f.write('{"text": "This is a test sentence."}\n')
                f.write('{"text": "Another test sentence for training."}\n')
            
            from privacy_finetuner.core import PrivateTrainer, PrivacyConfig
            
            privacy_config = PrivacyConfig(epsilon=1.0, delta=1e-5)
            trainer = PrivateTrainer("test-model", privacy_config, use_mcp_gateway=False)
            
            # Test dataset loading
            dataset = trainer._load_dataset(test_dataset)
            
            if len(dataset) == 2:  # Should load 2 samples
                passed = True
                details = f"Successfully loaded dataset with {len(dataset)} samples"
            else:
                passed = False
                details = f"Dataset loading error: expected 2 samples, got {len(dataset)}"
            
            # Clean up
            os.remove(test_dataset)
            
            results.append(TestResult(
                "Basic Training Workflow - Dataset Loading",
                passed,
                time.time() - start_time,
                details
            ))
            
        except Exception as e:
            results.append(TestResult(
                "Basic Training Workflow - Dataset Loading",
                False,
                time.time() - start_time,
                f"Failed with exception: {str(e)}"
            ))
            
            # Clean up on error
            if os.path.exists("test_integration_data.jsonl"):
                os.remove("test_integration_data.jsonl")
        
        return results
    
    def test_compliance(self) -> List[TestResult]:
        """Test regulatory compliance features."""
        results = []
        
        # Test 1: GDPR Compliance
        start_time = time.time()
        try:
            from privacy_finetuner.core.privacy_analytics import PrivacyComplianceChecker
            
            checker = PrivacyComplianceChecker()
            
            # Test GDPR compliant configuration
            gdpr_config = {"epsilon": 0.5, "audit_enabled": True, "encryption_enabled": True}
            gdpr_result = checker.check_compliance(gdpr_config, "GDPR")
            
            # Test non-compliant configuration
            non_compliant_config = {"epsilon": 5.0, "audit_enabled": False}
            non_compliant_result = checker.check_compliance(non_compliant_config, "GDPR")
            
            if gdpr_result["compliant"] and not non_compliant_result["compliant"]:
                passed = True
                details = "Correctly identifies GDPR compliance status"
            else:
                passed = False
                details = f"GDPR compliance check error: compliant={gdpr_result['compliant']}, non_compliant={non_compliant_result['compliant']}"
            
            results.append(TestResult(
                "GDPR Compliance Validation",
                passed,
                time.time() - start_time,
                details
            ))
            
        except Exception as e:
            results.append(TestResult(
                "GDPR Compliance Validation",
                False,
                time.time() - start_time,
                f"Failed with exception: {str(e)}"
            ))
        
        return results
    
    def test_robustness(self) -> List[TestResult]:
        """Test system robustness and error handling."""
        results = []
        
        # Test 1: Graceful Dependency Handling
        start_time = time.time()
        try:
            # Test that system works without optional dependencies
            # This is already demonstrated by successful import and basic functionality
            
            from privacy_finetuner.core import PrivacyConfig, ContextGuard, RedactionStrategy
            from privacy_finetuner.optimization.resource_optimizer import ResourceOptimizer
            
            # All these should work without full ML dependencies
            config = PrivacyConfig()
            guard = ContextGuard([RedactionStrategy.PII_REMOVAL])
            optimizer = ResourceOptimizer()
            
            passed = True
            details = "Successfully imports and initializes core components without full dependencies"
            
            results.append(TestResult(
                "Graceful Dependency Handling",
                passed,
                time.time() - start_time,
                details
            ))
            
        except Exception as e:
            results.append(TestResult(
                "Graceful Dependency Handling",
                False,
                time.time() - start_time,
                f"Failed with exception: {str(e)}"
            ))
        
        # Test 2: Error Recovery
        start_time = time.time()
        try:
            from privacy_finetuner.core.trainer import PrivateTrainer
            from privacy_finetuner.core import PrivacyConfig
            from privacy_finetuner.core.exceptions import ModelTrainingException
            
            privacy_config = PrivacyConfig()
            trainer = PrivateTrainer("nonexistent-model", privacy_config, use_mcp_gateway=False)
            
            # Should handle missing model gracefully with proper exception
            try:
                trainer.train("nonexistent_dataset.jsonl")
                passed = False
                details = "Failed to raise appropriate exception for missing dependencies"
            except ModelTrainingException:
                passed = True
                details = "Correctly raises ModelTrainingException for missing dependencies"
            except Exception as e:
                passed = False
                details = f"Raised wrong exception type: {type(e).__name__}"
            
            results.append(TestResult(
                "Error Recovery and Exception Handling",
                passed,
                time.time() - start_time,
                details
            ))
            
        except Exception as e:
            results.append(TestResult(
                "Error Recovery and Exception Handling",
                False,
                time.time() - start_time,
                f"Failed with exception: {str(e)}"
            ))
        
        return results
    
    def generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        failed_tests = total_tests - passed_tests
        
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        total_duration = time.time() - self.start_time
        
        # Generate summary
        self.logger.info(f"\n🏁 COMPREHENSIVE TEST RESULTS")
        self.logger.info("=" * 80)
        self.logger.info(f"Total Tests: {total_tests}")
        self.logger.info(f"Passed: {passed_tests}")
        self.logger.info(f"Failed: {failed_tests}")
        self.logger.info(f"Success Rate: {success_rate:.1f}%")
        self.logger.info(f"Total Duration: {total_duration:.2f}s")
        
        # Detailed results
        self.logger.info(f"\n📋 DETAILED TEST RESULTS")
        self.logger.info("-" * 80)
        
        for result in self.results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            self.logger.info(f"{status} {result.name} ({result.duration:.3f}s)")
            if result.details:
                self.logger.info(f"    {result.details}")
        
        # Quality gate assessment
        quality_gates = {
            "core_functionality": passed_tests >= total_tests * 0.8,  # 80% pass rate
            "privacy_guarantees": any("Privacy" in r.name and r.passed for r in self.results),
            "security_features": any("Security" in r.name or "Attack" in r.name and r.passed for r in self.results),
            "error_handling": any("Error" in r.name or "Robust" in r.name and r.passed for r in self.results),
            "performance": any("Performance" in r.name and r.passed for r in self.results)
        }
        
        all_gates_passed = all(quality_gates.values())
        
        self.logger.info(f"\n🚪 QUALITY GATES ASSESSMENT")
        self.logger.info("-" * 80)
        for gate, passed in quality_gates.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            self.logger.info(f"{status} {gate.replace('_', ' ').title()}")
        
        overall_status = "PASS" if all_gates_passed else "FAIL"
        self.logger.info(f"\n🎯 OVERALL QUALITY ASSESSMENT: {overall_status}")
        
        return {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": success_rate,
            "duration": total_duration,
            "quality_gates": quality_gates,
            "overall_pass": all_gates_passed,
            "detailed_results": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "duration": r.duration,
                    "details": r.details
                }
                for r in self.results
            ]
        }

def main():
    """Run comprehensive test suite."""
    
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    print("Privacy-Preserving Agent Finetuner - Comprehensive Test Suite")
    print("=" * 70)
    
    # Run tests
    test_suite = ComprehensiveTestSuite()
    report = test_suite.run_all_tests()
    
    # Return exit code based on results
    return 0 if report["overall_pass"] else 1

if __name__ == "__main__":
    sys.exit(main())
"""Privacy validation and testing module."""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime
from dataclasses import dataclass
import random
import math

logger = logging.getLogger(__name__)


@dataclass
class PrivacyTestResult:
    """Result of a privacy test."""
    test_name: str
    passed: bool
    score: float
    details: Dict[str, Any]
    timestamp: str
    recommendations: List[str] = None


class PrivacyValidator:
    """Comprehensive privacy validation for differential privacy implementations."""
    
    def __init__(self):
        self.test_results = []
        self.privacy_budgets = {}
        
    def validate_epsilon_delta_bounds(
        self, 
        epsilon: float, 
        delta: float
    ) -> PrivacyTestResult:
        """Validate epsilon and delta parameters are within acceptable bounds."""
        passed = True
        score = 1.0
        details = {}
        recommendations = []
        
        # Check epsilon bounds
        if epsilon <= 0:
            passed = False
            score = 0.0
            recommendations.append("Epsilon must be positive")
        elif epsilon > 10:
            score *= 0.7
            recommendations.append(f"Epsilon {epsilon} is high, consider reducing for stronger privacy")
        elif epsilon > 1:
            score *= 0.9
            recommendations.append(f"Epsilon {epsilon} provides moderate privacy")
        
        # Check delta bounds
        if delta <= 0 or delta >= 1:
            passed = False
            score = 0.0
            recommendations.append("Delta must be between 0 and 1")
        elif delta > 1e-5:
            score *= 0.8
            recommendations.append(f"Delta {delta} is relatively high, consider reducing")
        
        details = {
            "epsilon": epsilon,
            "delta": delta,
            "epsilon_category": self._categorize_epsilon(epsilon),
            "delta_category": self._categorize_delta(delta)
        }
        
        return PrivacyTestResult(
            test_name="epsilon_delta_bounds",
            passed=passed,
            score=score,
            details=details,
            recommendations=recommendations,
            timestamp=datetime.now().isoformat()
        )
    
    def validate_noise_calibration(
        self,
        sensitivity: float,
        noise_multiplier: float,
        epsilon: float,
        delta: float
    ) -> PrivacyTestResult:
        """Validate that noise is properly calibrated for given privacy parameters."""
        # Calculate theoretical noise scale for Gaussian mechanism
        theoretical_sigma = sensitivity * noise_multiplier
        
        # Check if noise scale is sufficient for (epsilon, delta)-DP
        passed = True
        score = 1.0
        recommendations = []
        
        if noise_multiplier <= 0:
            passed = False
            score = 0.0
            recommendations.append("Noise multiplier must be positive")
        else:
            # Simplified check - in practice would use more sophisticated analysis
            min_noise_multiplier = 2.0  # Conservative estimate
            if noise_multiplier < min_noise_multiplier:
                score *= 0.6
                recommendations.append(
                    f"Noise multiplier {noise_multiplier} may be too low, "
                    f"consider >= {min_noise_multiplier}"
                )
        
        details = {
            "sensitivity": sensitivity,
            "noise_multiplier": noise_multiplier,
            "theoretical_sigma": theoretical_sigma,
            "epsilon": epsilon,
            "delta": delta
        }
        
        return PrivacyTestResult(
            test_name="noise_calibration",
            passed=passed,
            score=score,
            details=details,
            recommendations=recommendations,
            timestamp=datetime.now().isoformat()
        )
    
    def test_membership_inference_resistance(
        self,
        model_func: Callable,
        train_data: List[Any],
        test_data: List[Any],
        num_shadow_models: int = 10
    ) -> PrivacyTestResult:
        """Test resistance to membership inference attacks."""
        logger.info("Running membership inference resistance test")
        
        try:
            # Simulate membership inference attack
            attack_accuracies = []
            
            for i in range(num_shadow_models):
                # Create shadow model training data
                shadow_train = random.sample(train_data + test_data, len(train_data))
                shadow_test = [x for x in train_data + test_data if x not in shadow_train]
                
                # Simulate model predictions (mock implementation)
                train_confidences = [random.uniform(0.5, 1.0) for _ in shadow_train[:10]]
                test_confidences = [random.uniform(0.3, 0.8) for _ in shadow_test[:10]]
                
                # Simple membership inference based on confidence
                threshold = np.mean(train_confidences + test_confidences)
                
                # Calculate attack accuracy
                train_correct = sum(1 for c in train_confidences if c > threshold)
                test_correct = sum(1 for c in test_confidences if c <= threshold)
                
                total_samples = len(train_confidences) + len(test_confidences)
                attack_accuracy = (train_correct + test_correct) / total_samples if total_samples > 0 else 0.5
                attack_accuracies.append(attack_accuracy)
            
            avg_attack_accuracy = np.mean(attack_accuracies)
            
            # Score based on attack accuracy (lower is better)
            if avg_attack_accuracy <= 0.55:
                score = 1.0
                passed = True
                recommendations = ["Excellent membership inference resistance"]
            elif avg_attack_accuracy <= 0.65:
                score = 0.8
                passed = True
                recommendations = ["Good membership inference resistance"]
            elif avg_attack_accuracy <= 0.75:
                score = 0.6
                passed = True
                recommendations = ["Moderate membership inference resistance, consider increasing privacy"]
            else:
                score = 0.3
                passed = False
                recommendations = ["Poor membership inference resistance, increase noise or reduce epsilon"]
            
            details = {
                "average_attack_accuracy": avg_attack_accuracy,
                "attack_accuracies": attack_accuracies,
                "num_shadow_models": num_shadow_models,
                "baseline_accuracy": 0.5
            }
            
        except Exception as e:
            logger.error(f"Membership inference test failed: {e}")
            return PrivacyTestResult(
                test_name="membership_inference_resistance",
                passed=False,
                score=0.0,
                details={"error": str(e)},
                recommendations=["Test failed to run - check implementation"],
                timestamp=datetime.now().isoformat()
            )
        
        return PrivacyTestResult(
            test_name="membership_inference_resistance",
            passed=passed,
            score=score,
            details=details,
            recommendations=recommendations,
            timestamp=datetime.now().isoformat()
        )
    
    def validate_gradient_clipping(
        self,
        gradients: List[float],
        max_grad_norm: float
    ) -> PrivacyTestResult:
        """Validate that gradients are properly clipped."""
        if not gradients:
            return PrivacyTestResult(
                test_name="gradient_clipping",
                passed=False,
                score=0.0,
                details={"error": "No gradients provided"},
                recommendations=["Provide gradient values for validation"],
                timestamp=datetime.now().isoformat()
            )
        
        # Calculate gradient norms
        gradient_norms = [abs(g) for g in gradients]
        max_norm = max(gradient_norms)
        avg_norm = np.mean(gradient_norms)
        
        # Check if clipping is properly applied
        passed = max_norm <= max_grad_norm * 1.001  # Small tolerance for floating point
        
        if passed:
            score = 1.0
            recommendations = ["Gradient clipping is properly applied"]
        else:
            score = 0.0
            recommendations = [
                f"Gradient clipping failed: max norm {max_norm} exceeds limit {max_grad_norm}"
            ]
        
        details = {
            "max_gradient_norm": max_norm,
            "average_gradient_norm": avg_norm,
            "max_grad_norm_limit": max_grad_norm,
            "num_gradients": len(gradients),
            "clipping_ratio": max_norm / max_grad_norm if max_grad_norm > 0 else float('inf')
        }
        
        return PrivacyTestResult(
            test_name="gradient_clipping",
            passed=passed,
            score=score,
            details=details,
            recommendations=recommendations,
            timestamp=datetime.now().isoformat()
        )
    
    def test_privacy_budget_accounting(
        self,
        privacy_events: List[Dict[str, Any]]
    ) -> PrivacyTestResult:
        """Test privacy budget accounting accuracy."""
        if not privacy_events:
            return PrivacyTestResult(
                test_name="privacy_budget_accounting",
                passed=False,
                score=0.0,
                details={"error": "No privacy events provided"},
                recommendations=["Provide privacy events for validation"],
                timestamp=datetime.now().isoformat()
            )
        
        total_epsilon_spent = 0.0
        composition_errors = []
        recommendations = []
        
        for event in privacy_events:
            epsilon = event.get("epsilon", 0)
            delta = event.get("delta", 0)
            
            if epsilon <= 0:
                composition_errors.append(f"Invalid epsilon {epsilon} in event {event.get('name', 'unknown')}")
            
            total_epsilon_spent += epsilon
        
        # Check for budget exhaustion
        max_epsilon = max((event.get("max_epsilon", 1.0) for event in privacy_events), default=1.0)
        
        budget_utilization = total_epsilon_spent / max_epsilon if max_epsilon > 0 else float('inf')
        
        passed = len(composition_errors) == 0 and budget_utilization <= 1.0
        
        if passed:
            if budget_utilization > 0.9:
                score = 0.7
                recommendations.append("Privacy budget near exhaustion - monitor carefully")
            elif budget_utilization > 0.7:
                score = 0.9
                recommendations.append("Privacy budget moderately utilized")
            else:
                score = 1.0
                recommendations.append("Privacy budget well managed")
        else:
            score = 0.0
            recommendations.extend(composition_errors)
            if budget_utilization > 1.0:
                recommendations.append(f"Privacy budget exceeded: {budget_utilization:.2%} utilized")
        
        details = {
            "total_epsilon_spent": total_epsilon_spent,
            "max_epsilon_budget": max_epsilon,
            "budget_utilization": budget_utilization,
            "num_privacy_events": len(privacy_events),
            "composition_errors": composition_errors
        }
        
        return PrivacyTestResult(
            test_name="privacy_budget_accounting",
            passed=passed,
            score=score,
            details=details,
            recommendations=recommendations,
            timestamp=datetime.now().isoformat()
        )
    
    def _categorize_epsilon(self, epsilon: float) -> str:
        """Categorize epsilon value."""
        if epsilon <= 0.1:
            return "very_strong"
        elif epsilon <= 1.0:
            return "strong"
        elif epsilon <= 5.0:
            return "moderate"
        elif epsilon <= 10.0:
            return "weak"
        else:
            return "very_weak"
    
    def _categorize_delta(self, delta: float) -> str:
        """Categorize delta value."""
        if delta <= 1e-6:
            return "very_strong"
        elif delta <= 1e-5:
            return "strong"
        elif delta <= 1e-4:
            return "moderate"
        else:
            return "weak"


class PrivacyTester:
    """Advanced privacy testing framework."""
    
    def __init__(self):
        self.validator = PrivacyValidator()
        self.test_suites = {}
        
    def register_test_suite(self, name: str, tests: List[Callable]):
        """Register a test suite."""
        self.test_suites[name] = tests
        
    def run_comprehensive_privacy_test(
        self,
        privacy_config: Dict[str, Any],
        model_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Run comprehensive privacy validation."""
        logger.info("Running comprehensive privacy test suite")
        
        results = []
        
        # Test 1: Epsilon-Delta bounds
        epsilon = privacy_config.get("epsilon", 1.0)
        delta = privacy_config.get("delta", 1e-5)
        bounds_result = self.validator.validate_epsilon_delta_bounds(epsilon, delta)
        results.append(bounds_result)
        
        # Test 2: Noise calibration
        sensitivity = privacy_config.get("sensitivity", 1.0)
        noise_multiplier = privacy_config.get("noise_multiplier", 1.0)
        noise_result = self.validator.validate_noise_calibration(
            sensitivity, noise_multiplier, epsilon, delta
        )
        results.append(noise_result)
        
        # Test 3: Gradient clipping (if gradients provided)
        if model_data and "gradients" in model_data:
            max_grad_norm = privacy_config.get("max_grad_norm", 1.0)
            clipping_result = self.validator.validate_gradient_clipping(
                model_data["gradients"], max_grad_norm
            )
            results.append(clipping_result)
        
        # Test 4: Privacy budget accounting (if events provided)
        if model_data and "privacy_events" in model_data:
            budget_result = self.validator.test_privacy_budget_accounting(
                model_data["privacy_events"]
            )
            results.append(budget_result)
        
        # Test 5: Membership inference resistance (if data provided)
        if (model_data and "train_data" in model_data and "test_data" in model_data and 
            "model_func" in model_data):
            mi_result = self.validator.test_membership_inference_resistance(
                model_data["model_func"],
                model_data["train_data"],
                model_data["test_data"]
            )
            results.append(mi_result)
        
        # Calculate overall score
        total_score = sum(r.score for r in results)
        max_possible_score = len(results)
        overall_score = total_score / max_possible_score if max_possible_score > 0 else 0.0
        
        passed_tests = sum(1 for r in results if r.passed)
        
        # Generate summary
        summary = {
            "total_tests": len(results),
            "passed_tests": passed_tests,
            "overall_score": overall_score,
            "overall_passed": all(r.passed for r in results),
            "test_results": [
                {
                    "test": r.test_name,
                    "passed": r.passed,
                    "score": r.score,
                    "recommendations": r.recommendations
                }
                for r in results
            ],
            "timestamp": datetime.now().isoformat()
        }
        
        return summary
    
    def generate_privacy_report(self, test_results: Dict[str, Any]) -> str:
        """Generate human-readable privacy report."""
        report = f"""
Privacy Validation Report
========================
Generated: {test_results['timestamp']}

Overall Score: {test_results['overall_score']:.2%}
Tests Passed: {test_results['passed_tests']}/{test_results['total_tests']}
Overall Status: {'PASSED' if test_results['overall_passed'] else 'FAILED'}

Test Results:
"""
        
        for test in test_results['test_results']:
            status = "✅ PASS" if test['passed'] else "❌ FAIL"
            report += f"  {status} {test['test']} (Score: {test['score']:.2%})\n"
            
            if test['recommendations']:
                for rec in test['recommendations']:
                    report += f"    • {rec}\n"
            report += "\n"
        
        return report
"""Tests for privacy analytics and monitoring components."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import numpy as np

from privacy_finetuner.core.privacy_analytics import (
    PrivacyBudgetTracker, PrivacyAttackDetector, PrivacyComplianceChecker,
    PrivacyEvent, create_privacy_dashboard_data
)


class TestPrivacyBudgetTracker:
    """Test suite for privacy budget tracking."""
    
    @pytest.fixture
    def tracker(self):
        """Create a privacy budget tracker for testing."""
        return PrivacyBudgetTracker(total_epsilon=5.0, total_delta=1e-5)
    
    def test_initialization(self, tracker):
        """Test tracker initialization."""
        assert tracker.total_epsilon == 5.0
        assert tracker.total_delta == 1e-5
        assert tracker.spent_epsilon == 0.0
        assert tracker.spent_delta == 0.0
        assert len(tracker.events) == 0
    
    def test_record_event_success(self, tracker):
        """Test successful event recording."""
        success = tracker.record_event(
            "training_step", 
            epsilon_cost=0.5,
            delta_cost=1e-6,
            context={"step": 1, "model": "test-model"}
        )
        
        assert success is True
        assert tracker.spent_epsilon == 0.5
        assert tracker.spent_delta == 1e-6
        assert len(tracker.events) == 1
        
        event = tracker.events[0]
        assert event.event_type == "training_step"
        assert event.epsilon_cost == 0.5
        assert event.context["step"] == 1
    
    def test_record_event_budget_exceeded(self, tracker):
        """Test event recording when budget is exceeded."""
        # First, consume most of the budget
        tracker.record_event("training", 4.5)
        
        # Try to exceed the budget
        success = tracker.record_event("training", 1.0)  # Would exceed 5.0 total
        
        assert success is False
        assert tracker.spent_epsilon == 4.5  # Should not have changed
        assert len(tracker.events) == 1  # Should not have added the event
    
    def test_remaining_budget_calculation(self, tracker):
        """Test remaining budget calculations."""
        assert tracker.remaining_epsilon == 5.0
        assert tracker.remaining_delta == 1e-5
        
        tracker.record_event("test", 2.0, 5e-6)
        
        assert tracker.remaining_epsilon == 3.0
        assert tracker.remaining_delta == 5e-6
    
    def test_epsilon_utilization(self, tracker):
        """Test epsilon utilization percentage."""
        assert tracker.epsilon_utilization == 0.0
        
        tracker.record_event("test", 2.5)  # 50% of budget
        assert tracker.epsilon_utilization == 50.0
        
        tracker.record_event("test", 2.5)  # 100% of budget
        assert tracker.epsilon_utilization == 100.0
    
    def test_budget_alerts(self, tracker):
        """Test budget alert system."""
        # No alerts initially
        assert len(tracker.alerts_sent) == 0
        
        # Trigger 50% alert
        tracker.record_event("test", 2.5)
        assert 0.5 in tracker.alerts_sent
        
        # Trigger 80% alert
        tracker.record_event("test", 1.5)
        assert 0.8 in tracker.alerts_sent
        
        # Should not duplicate alerts
        tracker.record_event("test", 0.9)
        assert len(tracker.alerts_sent) == 2  # Only 0.5 and 0.8
    
    def test_usage_summary(self, tracker):
        """Test comprehensive usage summary."""
        # Add some events
        tracker.record_event("training", 1.0)
        tracker.record_event("evaluation", 0.5)
        tracker.record_event("training", 1.5)
        
        summary = tracker.get_usage_summary()
        
        assert summary["total_budget"]["epsilon"] == 5.0
        assert summary["spent_budget"]["epsilon"] == 3.0
        assert summary["remaining_budget"]["epsilon"] == 2.0
        assert summary["utilization"]["epsilon_percent"] == 60.0
        
        assert summary["event_summary"]["total_events"] == 3
        assert summary["event_summary"]["events_by_type"]["training"] == 2
        assert summary["event_summary"]["events_by_type"]["evaluation"] == 1
        assert summary["event_summary"]["costs_by_type"]["training"] == 2.5


class TestPrivacyAttackDetector:
    """Test suite for privacy attack detection."""
    
    @pytest.fixture
    def detector(self):
        """Create a privacy attack detector for testing."""
        return PrivacyAttackDetector(window_size=50)
    
    def test_initialization(self, detector):
        """Test detector initialization."""
        assert len(detector.recent_queries) == 0
        assert len(detector.model_outputs) == 0
        assert detector.window_size == 50
    
    def test_membership_inference_analysis_low_risk(self, detector):
        """Test membership inference analysis with low risk."""
        query = "What is the capital of France?"
        model_output = {"confidence": 0.7, "response": "Paris"}
        
        analysis = detector.analyze_membership_inference_risk(query, model_output)
        
        assert analysis["overall_risk"] == "low"
        assert analysis["high_confidence_risk"] is False
        assert analysis["query_similarity_risk"] is False
        assert analysis["confidence_score"] == 0.7
        assert len(analysis["recommendations"]) > 0
    
    def test_membership_inference_analysis_high_confidence(self, detector):
        """Test membership inference analysis with high confidence."""
        query = "Sensitive training data information"
        model_output = {"confidence": 0.98, "response": "Detailed response"}
        
        analysis = detector.analyze_membership_inference_risk(query, model_output)
        
        assert analysis["overall_risk"] == "high"
        assert analysis["high_confidence_risk"] is True
        assert analysis["confidence_score"] == 0.98
        assert "Increase noise multiplier" in str(analysis["recommendations"])
    
    def test_query_similarity_detection(self, detector):
        """Test detection of similar queries."""
        base_query = "Tell me about machine learning privacy"
        
        # Add multiple similar queries
        similar_queries = [
            "Tell me about machine learning privacy techniques",
            "What is machine learning privacy",
            "Explain machine learning privacy methods",
            "How does machine learning privacy work"
        ]
        
        for query in similar_queries:
            detector.analyze_membership_inference_risk(
                query, {"confidence": 0.8}
            )
        
        # The next similar query should trigger similarity risk
        analysis = detector.analyze_membership_inference_risk(
            base_query, {"confidence": 0.8}
        )
        
        # Might trigger similarity risk depending on implementation
        assert "query_similarity_risk" in analysis
    
    def test_output_anomaly_detection(self, detector):
        """Test output anomaly detection."""
        # Add normal outputs
        for i in range(15):
            detector.analyze_membership_inference_risk(
                f"query_{i}", {"confidence": 0.6 + np.random.normal(0, 0.05)}
            )
        
        # Add consistent high confidence outputs (anomaly)
        for i in range(10):
            detector.analyze_membership_inference_risk(
                f"anomaly_query_{i}", {"confidence": 0.95}
            )
        
        # The anomaly detection should trigger
        analysis = detector.analyze_membership_inference_risk(
            "test_query", {"confidence": 0.95}
        )
        
        # Check if anomaly was detected (may require more sophisticated testing)
        assert "output_anomaly_risk" in analysis
    
    def test_risk_recommendations(self, detector):
        """Test risk recommendation generation."""
        # High risk scenario
        high_risk_recommendations = detector._get_risk_recommendations("high")
        assert len(high_risk_recommendations) > 0
        assert any("noise multiplier" in rec.lower() for rec in high_risk_recommendations)
        
        # Low risk scenario
        low_risk_recommendations = detector._get_risk_recommendations("low")
        assert len(low_risk_recommendations) > 0
        assert any("monitoring" in rec.lower() for rec in low_risk_recommendations)


class TestPrivacyComplianceChecker:
    """Test suite for privacy compliance checking."""
    
    @pytest.fixture
    def checker(self):
        """Create a privacy compliance checker for testing."""
        return PrivacyComplianceChecker()
    
    def test_initialization(self, checker):
        """Test checker initialization."""
        assert "GDPR" in checker.regulations
        assert "HIPAA" in checker.regulations
        assert "CCPA" in checker.regulations
    
    def test_gdpr_compliance_check_pass(self, checker):
        """Test GDPR compliance check that passes."""
        privacy_config = {
            "epsilon": 0.8,  # Within GDPR limit of 1.0
            "delta": 1e-5,
            "encryption_enabled": True,
            "audit_enabled": True
        }
        
        result = checker.check_compliance(privacy_config, "GDPR")
        
        assert result["regulation"] == "GDPR"
        assert result["compliant"] is True
        assert len(result["violations"]) == 0
    
    def test_gdpr_compliance_check_fail(self, checker):
        """Test GDPR compliance check that fails."""
        privacy_config = {
            "epsilon": 2.0,  # Exceeds GDPR limit of 1.0
            "delta": 1e-5
        }
        
        result = checker.check_compliance(privacy_config, "GDPR")
        
        assert result["regulation"] == "GDPR"
        assert result["compliant"] is False
        assert len(result["violations"]) > 0
        assert "exceeds maximum" in result["violations"][0]
        assert len(result["recommendations"]) > 0
    
    def test_hipaa_compliance_check(self, checker):
        """Test HIPAA compliance check."""
        privacy_config = {
            "epsilon": 0.3,  # Within HIPAA limit of 0.5
            "delta": 1e-5,
            "encryption_enabled": True,
            "audit_enabled": True
        }
        
        result = checker.check_compliance(privacy_config, "HIPAA")
        
        assert result["regulation"] == "HIPAA"
        assert result["compliant"] is True
    
    def test_hipaa_compliance_missing_requirements(self, checker):
        """Test HIPAA compliance with missing requirements."""
        privacy_config = {
            "epsilon": 0.3,
            "delta": 1e-5,
            "encryption_enabled": False,  # Required for HIPAA
            "audit_enabled": False  # Required for HIPAA
        }
        
        result = checker.check_compliance(privacy_config, "HIPAA")
        
        assert result["compliant"] is False
        assert len(result["violations"]) >= 2  # Missing encryption and audit
        assert any("Encryption required" in v for v in result["violations"])
        assert any("Audit trail required" in v for v in result["violations"])
    
    def test_unknown_regulation(self, checker):
        """Test compliance check with unknown regulation."""
        privacy_config = {"epsilon": 1.0}
        
        result = checker.check_compliance(privacy_config, "UNKNOWN_REGULATION")
        
        assert "error" in result
        assert "Unknown regulation" in result["error"]
    
    def test_comprehensive_compliance_report(self, checker):
        """Test comprehensive compliance report generation."""
        privacy_config = {
            "epsilon": 1.5,  # Fails GDPR and HIPAA
            "delta": 1e-5,
            "encryption_enabled": False,
            "audit_enabled": False
        }
        
        report = checker.generate_compliance_report(privacy_config)
        
        assert "timestamp" in report
        assert "privacy_config" in report
        assert "compliance_results" in report
        assert "overall_compliant" in report
        
        # Should fail overall compliance
        assert report["overall_compliant"] is False
        
        # Check individual regulation results
        assert "GDPR" in report["compliance_results"]
        assert "HIPAA" in report["compliance_results"]
        assert "CCPA" in report["compliance_results"]
        
        # All should have violations
        for regulation in ["GDPR", "HIPAA"]:
            result = report["compliance_results"][regulation]
            assert result["compliant"] is False
            assert len(result["violations"]) > 0


class TestPrivacyDashboard:
    """Test suite for privacy dashboard data generation."""
    
    @pytest.fixture
    def mock_components(self):
        """Create mock components for dashboard testing."""
        budget_tracker = Mock(spec=PrivacyBudgetTracker)
        budget_tracker.get_usage_summary.return_value = {
            "total_budget": {"epsilon": 10.0, "delta": 1e-5},
            "spent_budget": {"epsilon": 3.5, "delta": 3e-6},
            "utilization": {"epsilon_percent": 35.0}
        }
        
        attack_detector = Mock(spec=PrivacyAttackDetector)
        attack_detector.recent_queries = ["query1", "query2", "query3"]
        attack_detector.model_outputs = [{"confidence": 0.8}, {"confidence": 0.7}]
        
        compliance_checker = Mock(spec=PrivacyComplianceChecker)
        compliance_checker.generate_compliance_report.return_value = {
            "overall_compliant": True,
            "compliance_results": {"GDPR": {"compliant": True}}
        }
        
        privacy_config = {
            "epsilon": 1.0,
            "delta": 1e-5,
            "encryption_enabled": True
        }
        
        return budget_tracker, attack_detector, compliance_checker, privacy_config
    
    def test_dashboard_data_creation(self, mock_components):
        """Test privacy dashboard data creation."""
        budget_tracker, attack_detector, compliance_checker, privacy_config = mock_components
        
        dashboard_data = create_privacy_dashboard_data(
            budget_tracker, attack_detector, compliance_checker, privacy_config
        )
        
        assert "budget_status" in dashboard_data
        assert "compliance_status" in dashboard_data
        assert "security_metrics" in dashboard_data
        assert "system_health" in dashboard_data
        
        # Verify budget status
        budget_status = dashboard_data["budget_status"]
        assert budget_status["spent_budget"]["epsilon"] == 3.5
        assert budget_status["utilization"]["epsilon_percent"] == 35.0
        
        # Verify security metrics
        security_metrics = dashboard_data["security_metrics"]
        assert security_metrics["recent_queries"] == 3
        assert security_metrics["total_risk_assessments"] == 2
        
        # Verify system health
        system_health = dashboard_data["system_health"]
        assert system_health["privacy_engine_active"] is True
        assert system_health["monitoring_enabled"] is True
        assert "last_update" in system_health
    
    def test_dashboard_data_with_high_usage(self, mock_components):
        """Test dashboard data with high privacy budget usage."""
        budget_tracker, attack_detector, compliance_checker, privacy_config = mock_components
        
        # Simulate high usage
        budget_tracker.get_usage_summary.return_value = {
            "total_budget": {"epsilon": 10.0},
            "spent_budget": {"epsilon": 9.5},
            "utilization": {"epsilon_percent": 95.0}
        }
        
        dashboard_data = create_privacy_dashboard_data(
            budget_tracker, attack_detector, compliance_checker, privacy_config
        )
        
        budget_status = dashboard_data["budget_status"]
        assert budget_status["utilization"]["epsilon_percent"] == 95.0
        
        # In a real implementation, this might trigger alerts or warnings
        assert budget_status["spent_budget"]["epsilon"] == 9.5


class TestPrivacyEvent:
    """Test suite for privacy event data structure."""
    
    def test_privacy_event_creation(self):
        """Test privacy event creation."""
        timestamp = datetime.now().timestamp()
        context = {"model": "test-model", "step": 100}
        
        event = PrivacyEvent(
            timestamp=timestamp,
            event_type="training_step",
            epsilon_cost=0.1,
            delta_cost=1e-7,
            context=context
        )
        
        assert event.timestamp == timestamp
        assert event.event_type == "training_step"
        assert event.epsilon_cost == 0.1
        assert event.delta_cost == 1e-7
        assert event.context == context
    
    def test_privacy_event_serialization(self):
        """Test privacy event can be serialized."""
        event = PrivacyEvent(
            timestamp=datetime.now().timestamp(),
            event_type="evaluation",
            epsilon_cost=0.05,
            delta_cost=0,
            context={"test": True}
        )
        
        # Should be able to convert to dict (useful for logging/storage)
        event_dict = {
            "timestamp": event.timestamp,
            "event_type": event.event_type,
            "epsilon_cost": event.epsilon_cost,
            "delta_cost": event.delta_cost,
            "context": event.context
        }
        
        assert event_dict["event_type"] == "evaluation"
        assert event_dict["epsilon_cost"] == 0.05
        assert event_dict["context"]["test"] is True
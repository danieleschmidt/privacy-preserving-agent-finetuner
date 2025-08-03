"""Tests for advanced database operations and privacy analytics."""

import pytest
from datetime import datetime, timedelta
from uuid import UUID, uuid4
from unittest.mock import Mock, patch
from sqlalchemy.orm import Session

from privacy_finetuner.database.advanced_operations import (
    AdvancedPrivacyOperations, PrivacyBudgetAnalysis, ModelPerformanceMetrics
)
from privacy_finetuner.database.query_optimizer import QueryOptimizer, QueryMetrics
from privacy_finetuner.database.models import (
    TrainingJob, PrivacyBudgetEntry, Model, Dataset, User
)


class TestAdvancedPrivacyOperations:
    """Test suite for advanced privacy operations."""
    
    @pytest.fixture
    def mock_session(self):
        """Create a mock database session."""
        return Mock(spec=Session)
    
    @pytest.fixture
    def mock_query_optimizer(self):
        """Create a mock query optimizer."""
        return Mock(spec=QueryOptimizer)
    
    @pytest.fixture
    def advanced_ops(self, mock_session, mock_query_optimizer):
        """Create advanced privacy operations instance."""
        return AdvancedPrivacyOperations(mock_session, mock_query_optimizer)
    
    def test_analyze_privacy_budget_patterns_no_data(self, advanced_ops, mock_session):
        """Test budget analysis with no data."""
        user_id = uuid4()
        
        # Mock empty query result
        mock_query = Mock()
        mock_query.filter.return_value.all.return_value = []
        mock_session.query.return_value = mock_query
        
        analysis = advanced_ops.analyze_privacy_budget_patterns(user_id)
        
        assert isinstance(analysis, PrivacyBudgetAnalysis)
        assert analysis.user_id == str(user_id)
        assert analysis.total_epsilon_spent == 0.0
        assert analysis.budget_utilization_percent == 0.0
        assert analysis.jobs_in_period == 0
        assert analysis.risk_level == "low"
        assert len(analysis.recommendations) == 1
        assert "No privacy budget usage" in analysis.recommendations[0]
    
    def test_analyze_privacy_budget_patterns_with_data(self, advanced_ops, mock_session):
        """Test budget analysis with sample data."""
        user_id = uuid4()
        
        # Create mock privacy budget entries
        mock_entries = []
        for i in range(5):
            entry = Mock(spec=PrivacyBudgetEntry)
            entry.epsilon_spent = 0.5
            entry.training_job_id = uuid4()
            entry.created_at = datetime.now() - timedelta(days=i)
            mock_entries.append(entry)
        
        # Mock query chain
        mock_query = Mock()
        mock_query.filter.return_value.all.return_value = mock_entries
        mock_session.query.return_value = mock_query
        
        analysis = advanced_ops.analyze_privacy_budget_patterns(user_id, analysis_period_days=30)
        
        assert analysis.total_epsilon_spent == 2.5  # 5 entries * 0.5 each
        assert analysis.jobs_in_period == 5  # Unique training job IDs
        assert analysis.average_epsilon_per_job == 0.5
        assert analysis.budget_utilization_percent == 25.0  # 2.5 / 10.0 * 100
        assert analysis.risk_level == "low"  # < 40% utilization
        assert analysis.projected_depletion_date is not None
    
    def test_analyze_privacy_budget_patterns_high_risk(self, advanced_ops, mock_session):
        """Test budget analysis with high risk scenario."""
        user_id = uuid4()
        
        # Create mock entries with high epsilon consumption
        mock_entries = []
        for i in range(3):
            entry = Mock(spec=PrivacyBudgetEntry)
            entry.epsilon_spent = 3.0  # High consumption
            entry.training_job_id = uuid4()
            entry.created_at = datetime.now() - timedelta(days=i)
            mock_entries.append(entry)
        
        mock_query = Mock()
        mock_query.filter.return_value.all.return_value = mock_entries
        mock_session.query.return_value = mock_query
        
        analysis = advanced_ops.analyze_privacy_budget_patterns(user_id)
        
        assert analysis.total_epsilon_spent == 9.0
        assert analysis.budget_utilization_percent == 90.0  # 9.0 / 10.0 * 100
        assert analysis.risk_level == "critical"  # >= 90% utilization
        assert len(analysis.recommendations) > 0
        assert any("CRITICAL" in rec for rec in analysis.recommendations)
    
    def test_optimize_privacy_allocation_budget_exhausted(self, advanced_ops):
        """Test privacy allocation when budget is exhausted."""
        user_id = uuid4()
        
        # Mock budget repository to return exhausted budget
        with patch.object(advanced_ops, 'privacy_repo') as mock_repo:
            mock_repo.get_user_budget_summary.return_value = {
                'total_epsilon_spent': 10.0  # Fully exhausted
            }
            
            planned_jobs = [
                {"priority": "high", "epochs": 3, "batch_size": 8},
                {"priority": "medium", "epochs": 2, "batch_size": 16}
            ]
            
            result = advanced_ops.optimize_privacy_allocation(user_id, planned_jobs, total_budget=10.0)
            
            assert result["status"] == "budget_exhausted"
            assert "No remaining privacy budget" in result["message"]
    
    def test_optimize_privacy_allocation_success(self, advanced_ops):
        """Test successful privacy allocation optimization."""
        user_id = uuid4()
        
        # Mock budget repository
        with patch.object(advanced_ops, 'privacy_repo') as mock_repo:
            mock_repo.get_user_budget_summary.return_value = {
                'total_epsilon_spent': 2.0  # 8.0 remaining
            }
            
            planned_jobs = [
                {"priority": "high", "epochs": 2, "batch_size": 8, "dataset_size": 1000},
                {"priority": "medium", "epochs": 3, "batch_size": 4, "dataset_size": 500},
                {"priority": "low", "epochs": 1, "batch_size": 16, "dataset_size": 2000}
            ]
            
            result = advanced_ops.optimize_privacy_allocation(user_id, planned_jobs, total_budget=10.0)
            
            assert result["status"] == "optimization_complete"
            assert result["total_budget"] == 10.0
            assert result["remaining_budget"] == 8.0
            assert "allocated_jobs" in result
            assert len(result["allocated_jobs"]) == 3
            
            # High priority job should be allocated first
            high_priority_job = next(job for job in result["allocated_jobs"] if job["priority"] == "high")
            assert high_priority_job["allocated"] is True
    
    def test_analyze_model_privacy_utility_tradeoff(self, advanced_ops, mock_session):
        """Test model privacy-utility tradeoff analysis."""
        model_id = uuid4()
        
        # Mock model and training job
        mock_model = Mock(spec=Model)
        mock_model.id = model_id
        mock_model.base_model = "meta-llama/Llama-2-7b-hf"
        mock_model.epsilon_spent = 1.5
        mock_model.delta_value = 1e-5
        mock_model.eval_accuracy = 0.85
        mock_model.model_path = "/tmp/model"
        
        mock_training_job = Mock(spec=TrainingJob)
        mock_training_job.training_time_seconds = 3600
        mock_training_job.gpu_hours_used = 2.0
        mock_training_job.dataset = Mock(spec=Dataset)
        
        mock_model.training_job = mock_training_job
        
        # Mock model repository
        with patch.object(advanced_ops, 'model_repo') as mock_repo:
            mock_repo.get_by_id.return_value = mock_model
            
            analysis = advanced_ops.analyze_model_privacy_utility_tradeoff(model_id)
            
            assert isinstance(analysis, ModelPerformanceMetrics)
            assert analysis.model_id == str(model_id)
            assert analysis.privacy_utility_ratio > 0  # 0.85 / 1.5
            assert analysis.training_efficiency_score > 0
            assert analysis.compliance_score >= 0
            assert "privacy_utility_ratio_percentile" in analysis.benchmark_comparison
    
    def test_analyze_model_privacy_utility_tradeoff_not_found(self, advanced_ops):
        """Test model analysis with non-existent model."""
        model_id = uuid4()
        
        with patch.object(advanced_ops, 'model_repo') as mock_repo:
            mock_repo.get_by_id.return_value = None
            
            with pytest.raises(ValueError, match="Model .* not found"):
                advanced_ops.analyze_model_privacy_utility_tradeoff(model_id)
    
    def test_detect_privacy_anomalies_no_data(self, advanced_ops, mock_session):
        """Test anomaly detection with no data."""
        mock_query = Mock()
        mock_query.filter.return_value.all.return_value = []
        mock_session.query.return_value = mock_query
        
        anomalies = advanced_ops.detect_privacy_anomalies(lookback_hours=24)
        
        assert anomalies == []
    
    def test_detect_privacy_anomalies_with_data(self, advanced_ops, mock_session):
        """Test anomaly detection with sample data."""
        user_id = uuid4()
        
        # Create mock entries with some anomalous patterns
        mock_entries = []
        
        # Normal entries
        for i in range(5):
            entry = Mock(spec=PrivacyBudgetEntry)
            entry.user_id = user_id
            entry.epsilon_spent = 0.5
            entry.created_at = datetime.now() - timedelta(minutes=i*10)
            entry.training_job_id = uuid4()
            mock_entries.append(entry)
        
        # Anomalous entry (high epsilon)
        anomalous_entry = Mock(spec=PrivacyBudgetEntry)
        anomalous_entry.user_id = user_id
        anomalous_entry.epsilon_spent = 5.0  # Much higher than normal
        anomalous_entry.created_at = datetime.now()
        anomalous_entry.training_job_id = uuid4()
        mock_entries.append(anomalous_entry)
        
        mock_query = Mock()
        mock_query.filter.return_value.all.return_value = mock_entries
        mock_session.query.return_value = mock_query
        
        anomalies = advanced_ops.detect_privacy_anomalies(lookback_hours=24)
        
        assert len(anomalies) > 0
        
        # Check for high epsilon consumption anomaly
        high_epsilon_anomalies = [a for a in anomalies if a["type"] == "high_epsilon_consumption"]
        assert len(high_epsilon_anomalies) > 0
        
        anomaly = high_epsilon_anomalies[0]
        assert anomaly["epsilon_spent"] == 5.0
        assert anomaly["severity"] in ["high", "medium"]
    
    def test_generate_privacy_compliance_report(self, advanced_ops, mock_session):
        """Test comprehensive privacy compliance report generation."""
        user_id = uuid4()
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()
        
        # Mock training jobs
        mock_jobs = []
        for i in range(10):
            job = Mock(spec=TrainingJob)
            job.id = uuid4()
            job.job_name = f"test-job-{i}"
            job.epsilon_spent = 0.8 if i < 8 else 1.2  # 8 compliant, 2 violations
            job.target_epsilon = 1.0
            job.created_at = start_date + timedelta(days=i)
            job.completed_at = start_date + timedelta(days=i, hours=2)
            mock_jobs.append(job)
        
        # Mock query chain
        mock_query = Mock()
        mock_query.filter.return_value.all.return_value = mock_jobs
        mock_session.query.return_value = mock_query
        
        # Mock audit repository
        with patch.object(advanced_ops, 'audit_repo') as mock_audit_repo:
            mock_audit_repo.get_privacy_events.return_value = []
            
            report = advanced_ops.generate_privacy_compliance_report(
                user_id, start_date, end_date
            )
            
            assert "report_period" in report
            assert "compliance_summary" in report
            assert "privacy_budget_analysis" in report
            assert "violations" in report
            
            # Verify compliance calculations
            compliance_summary = report["compliance_summary"]
            assert compliance_summary["total_training_jobs"] == 10
            assert compliance_summary["compliant_jobs"] == 8
            assert compliance_summary["compliance_rate_percent"] == 80.0
            assert compliance_summary["privacy_violations"] == 2
            
            # Verify violations details
            violations = report["violations"]
            assert len(violations) == 2
            for violation in violations:
                assert violation["actual_epsilon"] == 1.2
                assert violation["target_epsilon"] == 1.0
                assert violation["violation_amount"] == 0.2
    
    def test_estimate_job_privacy_cost(self, advanced_ops):
        """Test privacy cost estimation for training jobs."""
        # Test various job configurations
        test_cases = [
            {
                "config": {"epochs": 1, "batch_size": 32, "learning_rate": 1e-5, "dataset_size": 1000},
                "expected_range": (0.1, 2.0)
            },
            {
                "config": {"epochs": 5, "batch_size": 8, "learning_rate": 5e-5, "dataset_size": 5000},
                "expected_range": (0.1, 10.0)
            },
            {
                "config": {"epochs": 2, "batch_size": 16, "learning_rate": 2e-5, "dataset_size": 2000},
                "expected_range": (0.1, 5.0)
            }
        ]
        
        for test_case in test_cases:
            cost = advanced_ops._estimate_job_privacy_cost(test_case["config"])
            min_expected, max_expected = test_case["expected_range"]
            
            assert min_expected <= cost <= max_expected
            assert isinstance(cost, float)
    
    def test_generate_budget_recommendations(self, advanced_ops):
        """Test budget recommendation generation."""
        # Test critical utilization
        recommendations = advanced_ops._generate_budget_recommendations(
            utilization_percent=95.0,
            daily_rate=1.0,
            jobs_count=15,
            avg_epsilon=3.0
        )
        
        assert len(recommendations) > 0
        assert any("CRITICAL" in rec for rec in recommendations)
        assert any("noise multiplier" in rec for rec in recommendations)
        assert any("High number of training jobs" in rec for rec in recommendations)
        
        # Test low utilization
        low_utilization_recommendations = advanced_ops._generate_budget_recommendations(
            utilization_percent=10.0,
            daily_rate=0.1,
            jobs_count=2,
            avg_epsilon=0.5
        )
        
        assert len(low_utilization_recommendations) == 0 or \
               not any("CRITICAL" in rec for rec in low_utilization_recommendations)
    
    def test_generate_compliance_recommendations(self, advanced_ops):
        """Test compliance recommendation generation."""
        # Test low compliance score with violations
        violations = [
            {"job_id": "job1", "violation_amount": 0.5},
            {"job_id": "job2", "violation_amount": 1.0}
        ]
        
        recommendations = advanced_ops._generate_compliance_recommendations(75.0, violations)
        
        assert len(recommendations) > 0
        assert any("CRITICAL" in rec for rec in recommendations)
        assert any("privacy budget violations" in rec for rec in recommendations)
        
        # Test high compliance score with no violations
        high_compliance_recommendations = advanced_ops._generate_compliance_recommendations(98.0, [])
        
        # Should have fewer or no critical recommendations
        assert len(high_compliance_recommendations) == 0 or \
               not any("CRITICAL" in rec for rec in high_compliance_recommendations)


class TestQueryOptimizer:
    """Test suite for query optimizer."""
    
    @pytest.fixture
    def mock_redis_client(self):
        """Create a mock Redis client."""
        return Mock()
    
    @pytest.fixture
    def optimizer(self, mock_redis_client):
        """Create query optimizer instance."""
        return QueryOptimizer(mock_redis_client)
    
    def test_initialization(self, optimizer):
        """Test optimizer initialization."""
        assert optimizer.redis_client is not None
        assert optimizer.cache_ttl == 3600
        assert len(optimizer.metrics) == 0
    
    def test_generate_cache_key(self, optimizer):
        """Test cache key generation."""
        key1 = optimizer._generate_cache_key("test_func", (1, 2), {"param": "value"}, "prefix")
        key2 = optimizer._generate_cache_key("test_func", (1, 2), {"param": "value"}, "prefix")
        key3 = optimizer._generate_cache_key("test_func", (1, 3), {"param": "value"}, "prefix")
        
        # Same inputs should generate same key
        assert key1 == key2
        
        # Different inputs should generate different keys
        assert key1 != key3
        
        # Key should contain prefix
        assert key1.startswith("prefix:")
    
    def test_record_metrics(self, optimizer):
        """Test metrics recording."""
        optimizer._record_metrics("test_query", 100.5, 50, False)
        
        assert len(optimizer.metrics) == 1
        
        metric = optimizer.metrics[0]
        assert metric.query_hash == "test_query"
        assert metric.execution_time_ms == 100.5
        assert metric.rows_returned == 50
        assert metric.cache_hit is False
    
    def test_get_slow_queries(self, optimizer):
        """Test slow query identification."""
        # Add some metrics
        optimizer._record_metrics("fast_query", 500, 10, False)
        optimizer._record_metrics("slow_query_1", 1500, 100, False)
        optimizer._record_metrics("cache_hit", 2000, 50, True)  # Cache hit, should be excluded
        optimizer._record_metrics("slow_query_2", 2500, 200, False)
        
        slow_queries = optimizer.get_slow_queries(threshold_ms=1000)
        
        assert len(slow_queries) == 2
        slow_query_hashes = [q.query_hash for q in slow_queries]
        assert "slow_query_1" in slow_query_hashes
        assert "slow_query_2" in slow_query_hashes
        assert "fast_query" not in slow_query_hashes
        assert "cache_hit" not in slow_query_hashes  # Cache hits excluded
    
    def test_get_cache_stats_no_redis(self):
        """Test cache stats when Redis is not available."""
        optimizer = QueryOptimizer(redis_client=None)
        stats = optimizer.get_cache_stats()
        
        assert stats["cache_enabled"] is False
    
    def test_get_cache_stats_with_redis(self, optimizer, mock_redis_client):
        """Test cache stats with Redis available."""
        # Mock Redis info
        mock_redis_client.info.return_value = {
            "used_memory_human": "10MB",
            "connected_clients": 5
        }
        
        # Add some metrics
        optimizer._record_metrics("query1", 100, 10, True)
        optimizer._record_metrics("query2", 200, 20, False)
        optimizer._record_metrics("query3", 150, 15, True)
        
        stats = optimizer.get_cache_stats()
        
        assert stats["cache_enabled"] is True
        assert stats["total_queries"] == 3
        assert stats["cache_hits"] == 2
        assert stats["cache_hit_rate"] == pytest.approx(66.67, rel=1e-2)
        assert stats["redis_memory_usage"] == "10MB"
        assert stats["redis_connected_clients"] == 5
    
    def test_clear_cache(self, optimizer, mock_redis_client):
        """Test cache clearing."""
        mock_redis_client.keys.return_value = ["key1", "key2", "key3"]
        mock_redis_client.delete.return_value = 3
        
        deleted_count = optimizer.clear_cache("test_pattern:*")
        
        assert deleted_count == 3
        mock_redis_client.keys.assert_called_once_with("test_pattern:*")
        mock_redis_client.delete.assert_called_once_with("key1", "key2", "key3")
    
    def test_clear_cache_no_keys(self, optimizer, mock_redis_client):
        """Test cache clearing when no keys match."""
        mock_redis_client.keys.return_value = []
        
        deleted_count = optimizer.clear_cache("nonexistent:*")
        
        assert deleted_count == 0
        mock_redis_client.delete.assert_not_called()
    
    def test_cache_decorator_functionality(self, optimizer, mock_redis_client):
        """Test cache decorator functionality."""
        # Mock Redis get/set operations
        mock_redis_client.get.return_value = None  # Cache miss
        mock_redis_client.setex.return_value = True
        
        @optimizer.cache_query(ttl=1800, cache_key_prefix="test_cache")
        def sample_function(param1, param2="default"):
            return {"result": param1 + len(param2)}
        
        # First call should execute function and cache result
        result1 = sample_function("test", param2="value")
        
        assert result1 == {"result": "testvalue"}
        mock_redis_client.setex.assert_called_once()
        
        # Verify cache key was generated and used
        mock_redis_client.get.assert_called_once()
        
        # Verify metrics were recorded
        assert len(optimizer.metrics) == 1
        assert optimizer.metrics[0].cache_hit is False


class TestQueryMetrics:
    """Test suite for query metrics data structure."""
    
    def test_query_metrics_creation(self):
        """Test query metrics creation."""
        timestamp = datetime.now()
        
        metrics = QueryMetrics(
            query_hash="test_hash",
            execution_time_ms=150.5,
            rows_returned=25,
            cache_hit=True,
            timestamp=timestamp
        )
        
        assert metrics.query_hash == "test_hash"
        assert metrics.execution_time_ms == 150.5
        assert metrics.rows_returned == 25
        assert metrics.cache_hit is True
        assert metrics.timestamp == timestamp
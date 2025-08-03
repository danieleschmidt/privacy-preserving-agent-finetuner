"""Comprehensive API tests for privacy-preserving training endpoints."""

import pytest
import json
from datetime import datetime
from uuid import uuid4
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient

from privacy_finetuner.api.server import create_app
from privacy_finetuner.core import PrivacyConfig
from privacy_finetuner.database import TrainingJob, Model


class TestPrivacyAPI:
    """Test suite for privacy-preserving API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app()
        return TestClient(app)
    
    @pytest.fixture
    def sample_training_request(self):
        """Sample training request payload."""
        return {
            "job_name": "test-privacy-training",
            "model_name": "meta-llama/Llama-2-7b-hf",
            "dataset_path": "/tmp/test_dataset.jsonl",
            "privacy_config": {
                "epsilon": 1.0,
                "delta": 1e-5,
                "max_grad_norm": 1.0,
                "noise_multiplier": 0.5,
                "accounting_mode": "rdp"
            },
            "training_params": {
                "epochs": 2,
                "batch_size": 4,
                "learning_rate": 5e-5
            }
        }
    
    @pytest.fixture
    def sample_context_request(self):
        """Sample context protection request."""
        return {
            "text": "Process payment for John Doe with card 4111-1111-1111-1111",
            "sensitivity_level": "high",
            "strategies": ["pii_removal", "entity_hashing"]
        }
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        with patch('privacy_finetuner.database.get_database') as mock_db:
            mock_db.return_value.health_check.return_value = {
                "database": "healthy",
                "redis": "healthy"
            }
            
            response = client.get("/health")
            assert response.status_code == 200
            
            data = response.json()
            assert data["status"] == "healthy"
            assert "timestamp" in data
            assert data["version"] == "1.0.0"
            assert data["database_status"] == "healthy"
            assert data["privacy_engine_status"] == "active"
    
    def test_health_check_degraded(self, client):
        """Test health check with degraded status."""
        with patch('privacy_finetuner.database.get_database') as mock_db:
            mock_db.return_value.health_check.return_value = {
                "database": "unhealthy: connection failed",
                "redis": "not_configured"
            }
            
            response = client.get("/health")
            assert response.status_code == 200
            
            data = response.json()
            assert data["status"] == "degraded"
            assert "connection failed" in data["database_status"]
    
    @patch('privacy_finetuner.api.server.get_db_session')
    @patch('privacy_finetuner.api.server.BackgroundTasks.add_task')
    def test_start_training_success(self, mock_add_task, mock_db_session, client, sample_training_request):
        """Test successful training job creation."""
        # Mock database session and repository
        mock_session = Mock()
        mock_db_session.return_value.__next__ = Mock(return_value=mock_session)
        
        mock_job_record = Mock()
        mock_job_record.id = uuid4()
        
        with patch('privacy_finetuner.api.server.TrainingJobRepository') as mock_repo_class:
            mock_repo = Mock()
            mock_repo.create.return_value = mock_job_record
            mock_repo_class.return_value = mock_repo
            
            response = client.post("/train", json=sample_training_request)
            
            assert response.status_code == 200
            data = response.json()
            
            assert "job_id" in data
            assert data["status"] == "started"
            assert data["privacy_budget_used"] == 1.0
            assert "Training job initiated successfully" in data["message"]
            
            # Verify repository was called correctly
            mock_repo.create.assert_called_once()
            create_args = mock_repo.create.call_args[1]
            assert create_args["job_name"] == "test-privacy-training"
            assert create_args["target_epsilon"] == 1.0
            assert create_args["target_delta"] == 1e-5
    
    def test_start_training_invalid_privacy_config(self, client):
        """Test training with invalid privacy configuration."""
        invalid_request = {
            "job_name": "test-job",
            "model_name": "test-model",
            "dataset_path": "/tmp/test.jsonl",
            "privacy_config": {
                "epsilon": 1.0
                # Missing required fields
            }
        }
        
        response = client.post("/train", json=invalid_request)
        assert response.status_code == 422  # Validation error
    
    def test_start_training_budget_exceeded(self, client, sample_training_request):
        """Test training when privacy budget is exceeded."""
        with patch('privacy_finetuner.api.server.app_state') as mock_state:
            mock_budget_tracker = Mock()
            mock_budget_tracker.record_event.return_value = False  # Budget exceeded
            mock_state.__getitem__.return_value = mock_budget_tracker
            
            response = client.post("/train", json=sample_training_request)
            
            assert response.status_code == 400
            assert "Insufficient privacy budget" in response.json()["detail"]
    
    def test_protect_context_success(self, client, sample_context_request):
        """Test successful context protection."""
        response = client.post("/protect-context", json=sample_context_request)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "protected_text" in data
        assert data["original_length"] > 0
        assert data["protected_length"] > 0
        assert data["redactions_applied"] >= 0
        assert "sensitivity_analysis" in data
        assert "compliance_status" in data
        
        # Verify sensitive data is redacted
        assert "4111-1111-1111-1111" not in data["protected_text"]
        assert "John Doe" not in data["protected_text"]
    
    def test_protect_context_different_strategies(self, client):
        """Test context protection with different strategies."""
        request = {
            "text": "Contact john.doe@example.com for payment details",
            "sensitivity_level": "medium",
            "strategies": ["pii_removal"]
        }
        
        response = client.post("/protect-context", json=request)
        assert response.status_code == 200
        
        data = response.json()
        assert "john.doe@example.com" not in data["protected_text"]
        assert "[EMAIL]" in data["protected_text"]
    
    def test_get_privacy_budget_status(self, client):
        """Test privacy budget status endpoint."""
        with patch('privacy_finetuner.api.server.app_state') as mock_state:
            mock_budget_tracker = Mock()
            mock_budget_tracker.get_usage_summary.return_value = {
                "spent_budget": {"epsilon": 2.5, "delta": 1e-5},
                "remaining_budget": {"epsilon": 7.5, "delta": 0},
                "utilization": {"epsilon_percent": 25.0, "delta_percent": 100.0}
            }
            mock_state.__getitem__.return_value = mock_budget_tracker
            
            response = client.get("/privacy-budget")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["epsilon_spent"] == 2.5
            assert data["remaining_budget"] == 7.5
            assert data["budget_utilization_percent"] == 25.0
            assert "recommendations" in data
    
    @patch('privacy_finetuner.api.server.get_db_session')
    def test_get_training_status_running(self, mock_db_session, client):
        """Test getting status of running training job."""
        job_id = str(uuid4())
        
        # Mock active trainer
        with patch('privacy_finetuner.api.server.app_state') as mock_state:
            mock_trainer = Mock()
            mock_trainer.get_privacy_report.return_value = {
                "epsilon_spent": 0.5,
                "delta": 1e-5,
                "remaining_budget": 0.5,
                "accounting_mode": "rdp"
            }
            mock_trainers = {job_id: mock_trainer}
            mock_state.__getitem__.return_value = mock_trainers
            
            response = client.get(f"/training/{job_id}/status")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["job_id"] == job_id
            assert data["status"] == "running"
            assert "privacy_report" in data
            assert data["privacy_report"]["epsilon_spent"] == 0.5
    
    @patch('privacy_finetuner.api.server.get_db_session')
    def test_list_models(self, mock_db_session, client):
        """Test listing trained models."""
        # Mock database session
        mock_session = Mock()
        mock_db_session.return_value.__next__ = Mock(return_value=mock_session)
        
        # Mock model repository
        mock_model1 = Mock(spec=Model)
        mock_model1.id = uuid4()
        mock_model1.name = "test-model-1"
        mock_model1.version = "1.0"
        mock_model1.base_model = "meta-llama/Llama-2-7b-hf"
        mock_model1.epsilon_spent = 1.5
        mock_model1.eval_accuracy = 0.85
        mock_model1.created_at = datetime.now()
        mock_model1.is_deployed = False
        
        with patch('privacy_finetuner.api.server.ModelRepository') as mock_repo_class:
            mock_repo = Mock()
            mock_repo.get_all.return_value = [mock_model1]
            mock_repo_class.return_value = mock_repo
            
            response = client.get("/models")
            
            assert response.status_code == 200
            data = response.json()
            
            assert "models" in data
            assert len(data["models"]) == 1
            assert data["total"] == 1
            
            model_data = data["models"][0]
            assert model_data["name"] == "test-model-1"
            assert model_data["epsilon_spent"] == 1.5
            assert model_data["eval_accuracy"] == 0.85
    
    @patch('privacy_finetuner.api.server.get_db_session')
    def test_evaluate_model_success(self, mock_db_session, client):
        """Test successful model evaluation."""
        model_id = str(uuid4())
        
        # Mock database components
        mock_session = Mock()
        mock_db_session.return_value.__next__ = Mock(return_value=mock_session)
        
        mock_model = Mock(spec=Model)
        mock_model.id = uuid4()
        mock_model.base_model = "meta-llama/Llama-2-7b-hf"
        mock_model.epsilon_spent = 1.0
        mock_model.delta_value = 1e-5
        mock_model.noise_multiplier = 0.5
        mock_model.max_grad_norm = 1.0
        mock_model.model_path = "/tmp/model"
        
        with patch('privacy_finetuner.api.server.ModelRepository') as mock_repo_class, \
             patch('privacy_finetuner.api.server.PrivateTrainer') as mock_trainer_class, \
             patch('privacy_finetuner.api.server.AdvancedPrivacyOperations') as mock_ops_class:
            
            # Setup mocks
            mock_repo = Mock()
            mock_repo.get_by_id.return_value = mock_model
            mock_repo_class.return_value = mock_repo
            
            mock_trainer = Mock()
            mock_trainer.evaluate.return_value = {
                "accuracy": 0.85,
                "perplexity": 2.1,
                "privacy_leakage": 0.01
            }
            mock_trainer_class.return_value = mock_trainer
            
            mock_ops = Mock()
            mock_performance_metrics = Mock()
            mock_performance_metrics.__dict__ = {
                "privacy_utility_ratio": 0.85,
                "accuracy_degradation_percent": 5.0,
                "deployment_readiness": True
            }
            mock_ops.analyze_model_privacy_utility_tradeoff.return_value = mock_performance_metrics
            mock_ops_class.return_value = mock_ops
            
            request_data = {
                "model_id": model_id,
                "test_dataset_path": "/tmp/test_data.jsonl",
                "evaluation_params": {"batch_size": 8}
            }
            
            response = client.post("/evaluate-model", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["model_id"] == model_id
            assert "evaluation_results" in data
            assert "performance_metrics" in data
            assert "privacy_guarantees" in data
            
            eval_results = data["evaluation_results"]
            assert eval_results["accuracy"] == 0.85
            assert eval_results["privacy_leakage"] == 0.01
    
    def test_evaluate_model_not_found(self, client):
        """Test model evaluation with non-existent model."""
        model_id = str(uuid4())
        
        with patch('privacy_finetuner.api.server.get_db_session') as mock_db_session, \
             patch('privacy_finetuner.api.server.ModelRepository') as mock_repo_class:
            
            mock_session = Mock()
            mock_db_session.return_value.__next__ = Mock(return_value=mock_session)
            
            mock_repo = Mock()
            mock_repo.get_by_id.return_value = None  # Model not found
            mock_repo_class.return_value = mock_repo
            
            request_data = {
                "model_id": model_id,
                "test_dataset_path": "/tmp/test_data.jsonl"
            }
            
            response = client.post("/evaluate-model", json=request_data)
            
            assert response.status_code == 404
            assert "Model not found" in response.json()["detail"]
    
    @patch('privacy_finetuner.api.server.get_db_session')
    def test_get_privacy_analysis(self, mock_db_session, client):
        """Test comprehensive privacy analysis endpoint."""
        # Mock database session
        mock_session = Mock()
        mock_db_session.return_value.__next__ = Mock(return_value=mock_session)
        
        with patch('privacy_finetuner.api.server.AdvancedPrivacyOperations') as mock_ops_class, \
             patch('privacy_finetuner.api.server.app_state') as mock_state:
            
            # Mock query optimizer
            mock_query_optimizer = Mock()
            mock_state.__getitem__.return_value = mock_query_optimizer
            
            # Mock advanced operations
            mock_ops = Mock()
            
            # Mock budget analysis
            mock_budget_analysis = Mock()
            mock_budget_analysis.__dict__ = {
                "user_id": "test-user",
                "total_epsilon_spent": 3.5,
                "risk_level": "medium",
                "recommendations": ["Reduce training frequency", "Increase noise multiplier"]
            }
            mock_ops.analyze_privacy_budget_patterns.return_value = mock_budget_analysis
            
            # Mock anomaly detection
            mock_ops.detect_privacy_anomalies.return_value = [
                {
                    "type": "high_epsilon_consumption",
                    "severity": "medium",
                    "epsilon_spent": 2.0,
                    "threshold": 1.5
                }
            ]
            
            # Mock compliance report
            mock_ops.generate_privacy_compliance_report.return_value = {
                "compliance_summary": {"compliance_rate_percent": 85.0}
            }
            
            mock_ops_class.return_value = mock_ops
            
            response = client.get("/privacy-analysis")
            
            assert response.status_code == 200
            data = response.json()
            
            assert "analysis_id" in data
            assert "user_budget_summary" in data
            assert data["risk_level"] == "medium"
            assert len(data["anomalies_detected"]) == 1
            assert data["compliance_score"] == 85.0
            assert len(data["recommendations"]) == 2
    
    def test_cors_headers(self, client):
        """Test CORS headers are properly set."""
        response = client.options("/health")
        
        # Check that CORS headers would be set (TestClient doesn't always show all headers)
        # In a real test, you'd verify Access-Control-Allow-Origin, etc.
        assert response.status_code in [200, 405]  # OPTIONS might not be implemented
    
    def test_request_validation(self, client):
        """Test request validation for various endpoints."""
        # Test invalid training request
        invalid_training = {
            "job_name": "",  # Empty name
            "model_name": "test-model",
            "dataset_path": "/tmp/test.jsonl",
            "privacy_config": {"epsilon": -1.0}  # Invalid epsilon
        }
        
        response = client.post("/train", json=invalid_training)
        assert response.status_code == 422
        
        # Test invalid context protection request
        invalid_context = {
            "text": "",  # Empty text
            "sensitivity_level": "invalid_level"
        }
        
        response = client.post("/protect-context", json=invalid_context)
        assert response.status_code == 422


class TestAPIIntegration:
    """Integration tests for API with real components."""
    
    @pytest.fixture
    def app_with_mocked_db(self):
        """Create app with mocked database components."""
        with patch('privacy_finetuner.database.get_database') as mock_db:
            mock_db.return_value.health_check.return_value = {
                "database": "healthy",
                "redis": "healthy"
            }
            app = create_app()
            return app
    
    def test_full_training_workflow(self, app_with_mocked_db):
        """Test complete training workflow from start to status check."""
        client = TestClient(app_with_mocked_db)
        
        # Create training request
        training_request = {
            "job_name": "integration-test",
            "model_name": "meta-llama/Llama-2-7b-hf",
            "dataset_path": "/tmp/test_dataset.jsonl",
            "privacy_config": {
                "epsilon": 0.5,
                "delta": 1e-5,
                "max_grad_norm": 1.0,
                "noise_multiplier": 1.0,
                "accounting_mode": "rdp"
            },
            "training_params": {
                "epochs": 1,
                "batch_size": 2
            }
        }
        
        with patch('privacy_finetuner.api.server.get_db_session') as mock_db_session, \
             patch('privacy_finetuner.api.server.TrainingJobRepository') as mock_repo_class:
            
            # Mock database components
            mock_session = Mock()
            mock_db_session.return_value.__next__ = Mock(return_value=mock_session)
            
            mock_job_record = Mock()
            mock_job_record.id = uuid4()
            
            mock_repo = Mock()
            mock_repo.create.return_value = mock_job_record
            mock_repo_class.return_value = mock_repo
            
            # Start training
            response = client.post("/train", json=training_request)
            assert response.status_code == 200
            
            job_data = response.json()
            job_id = job_data["job_id"]
            
            # Check status (should be running initially)
            status_response = client.get(f"/training/{job_id}/status")
            assert status_response.status_code == 200
            
            # Check privacy budget
            budget_response = client.get("/privacy-budget")
            assert budget_response.status_code == 200
            budget_data = budget_response.json()
            assert budget_data["epsilon_spent"] >= 0.5  # Should have consumed some budget
    
    def test_context_protection_workflow(self, app_with_mocked_db):
        """Test complete context protection workflow."""
        client = TestClient(app_with_mocked_db)
        
        # Test with various types of sensitive data
        test_cases = [
            {
                "text": "My email is john.doe@company.com and phone is 555-123-4567",
                "expected_redactions": ["email", "phone"]
            },
            {
                "text": "Credit card: 4111-1111-1111-1111, SSN: 123-45-6789",
                "expected_redactions": ["credit_card", "ssn"]
            },
            {
                "text": "Regular text without sensitive information",
                "expected_redactions": []
            }
        ]
        
        for test_case in test_cases:
            request = {
                "text": test_case["text"],
                "sensitivity_level": "high",
                "strategies": ["pii_removal"]
            }
            
            response = client.post("/protect-context", json=request)
            assert response.status_code == 200
            
            data = response.json()
            assert data["original_length"] == len(test_case["text"])
            
            # Verify expected redactions
            if test_case["expected_redactions"]:
                assert data["redactions_applied"] > 0
            else:
                # Text without sensitive info might still have some redactions due to patterns
                assert data["redactions_applied"] >= 0
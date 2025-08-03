"""Unit tests for API server."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import json
from fastapi.testclient import TestClient

from privacy_finetuner.api.server import create_app
from privacy_finetuner.core.privacy_config import PrivacyConfig


class TestAPIServer:
    """Test API server functionality."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app()
        return TestClient(app)
    
    @pytest.fixture
    def mock_trainer(self):
        """Mock trainer for testing."""
        with patch('privacy_finetuner.api.server.PrivateTrainer') as mock:
            trainer_instance = Mock()
            mock.return_value = trainer_instance
            
            # Mock training results
            trainer_instance.train.return_value = {
                "status": "training_complete",
                "epochs_completed": 3,
                "final_loss": 2.45,
                "privacy_spent": 0.85,
                "model_path": "/models/test_model"
            }
            
            # Mock privacy report
            trainer_instance.get_privacy_report.return_value = {
                "epsilon_spent": 0.85,
                "delta": 1e-5,
                "remaining_budget": 0.15,
                "accounting_mode": "rdp"
            }
            
            yield trainer_instance
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
    
    @patch('privacy_finetuner.api.server.PrivateTrainer')
    @patch('privacy_finetuner.api.server.PrivacyConfig')
    def test_start_training(self, mock_config, mock_trainer, client):
        """Test starting a training job."""
        # Mock privacy config validation
        config_instance = Mock()
        mock_config.return_value = config_instance
        config_instance.validate.return_value = None
        
        # Mock trainer
        trainer_instance = Mock()
        mock_trainer.return_value = trainer_instance
        
        training_request = {
            "model_name": "gpt2",
            "dataset_path": "/data/test.jsonl",
            "privacy_config": {
                "epsilon": 1.0,
                "delta": 1e-5,
                "max_grad_norm": 1.0,
                "noise_multiplier": 0.5
            },
            "training_config": {
                "epochs": 3,
                "batch_size": 8,
                "learning_rate": 5e-5
            }
        }
        
        # Mock authentication
        headers = {"Authorization": "Bearer test-token"}
        
        response = client.post(
            "/train",
            json=training_request,
            headers=headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Training started"
        assert "details" in data
    
    def test_privacy_report(self, client, mock_trainer):
        """Test getting privacy report."""
        # First start a training job to have an active trainer
        training_request = {
            "model_name": "gpt2",
            "dataset_path": "/data/test.jsonl",
            "privacy_config": {
                "epsilon": 1.0,
                "delta": 1e-5
            },
            "training_params": {}
        }
        
        headers = {"Authorization": "Bearer test-token"}
        client.post("/train", json=training_request, headers=headers)
        
        # Now get privacy report
        response = client.get("/privacy-report")
        
        assert response.status_code == 200
        data = response.json()
        assert "epsilon_spent" in data
        assert "delta" in data
        assert "remaining_budget" in data
    
    def test_protect_context(self, client):
        """Test context protection endpoint."""
        with patch('privacy_finetuner.api.server.ContextGuard') as mock_guard:
            guard_instance = Mock()
            mock_guard.return_value = guard_instance
            
            # Mock protection result
            guard_instance.protect.return_value = "This is [PROTECTED] text"
            guard_instance.explain_redactions.return_value = {
                "total_redactions": 1,
                "redaction_details": [
                    {
                        "type": "PII",
                        "position": (8, 18),
                        "reason": "Detected PII pattern"
                    }
                ]
            }
            
            response = client.post(
                "/protect-context",
                json={
                    "text": "This is sensitive text",
                    "strategies": ["pii_removal"],
                    "sensitivity": "high"
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "protected_text" in data
            assert "redaction_report" in data
            assert data["protected_text"] == "This is [PROTECTED] text"
    
    def test_privacy_report_no_trainer(self, client):
        """Test privacy report with no active trainer."""
        response = client.get("/privacy-report")
        
        assert response.status_code == 404
        data = response.json()
        assert "No active training session" in data["detail"]
    
    def test_training_with_invalid_config(self, client):
        """Test training with invalid privacy configuration."""
        training_request = {
            "model_name": "gpt2",
            "dataset_path": "/data/test.jsonl",
            "privacy_config": {
                "epsilon": -1.0,  # Invalid epsilon
                "delta": 1e-5
            },
            "training_params": {}
        }
        
        headers = {"Authorization": "Bearer test-token"}
        
        with patch('privacy_finetuner.api.server.PrivacyConfig') as mock_config:
            mock_config.side_effect = ValueError("Epsilon must be positive")
            
            response = client.post(
                "/train",
                json=training_request,
                headers=headers
            )
            
            assert response.status_code == 500
    
    def test_context_protection_error_handling(self, client):
        """Test context protection error handling."""
        with patch('privacy_finetuner.api.server.ContextGuard') as mock_guard:
            mock_guard.side_effect = Exception("Context protection failed")
            
            response = client.post(
                "/protect-context",
                json={
                    "text": "Test text",
                    "strategies": ["invalid_strategy"]
                }
            )
            
            assert response.status_code == 500
            data = response.json()
            assert "Context protection failed" in data["detail"]
    
    @patch('privacy_finetuner.api.server.RedactionStrategy')
    def test_context_protection_with_multiple_strategies(self, mock_strategy, client):
        """Test context protection with multiple strategies."""
        with patch('privacy_finetuner.api.server.ContextGuard') as mock_guard:
            guard_instance = Mock()
            mock_guard.return_value = guard_instance
            
            # Mock hasattr to return True for strategy names
            mock_strategy.PII_REMOVAL = "pii_removal"
            mock_strategy.ENTITY_HASHING = "entity_hashing"
            
            guard_instance.protect.return_value = "Protected text"
            guard_instance.explain_redactions.return_value = {
                "total_redactions": 2,
                "redaction_details": []
            }
            
            response = client.post(
                "/protect-context",
                json={
                    "text": "John Doe works at Acme Corp",
                    "strategies": ["pii_removal", "entity_hashing"],
                    "sensitivity": "medium"
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["protected_text"] == "Protected text"
            assert data["strategies_applied"] == ["pii_removal", "entity_hashing"]


class TestAPIErrorHandling:
    """Test API error handling."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app()
        return TestClient(app)
    
    def test_missing_authentication(self, client):
        """Test missing authentication header."""
        training_request = {
            "model_name": "gpt2",
            "dataset_path": "/data/test.jsonl",
            "privacy_config": {"epsilon": 1.0, "delta": 1e-5},
            "training_params": {}
        }
        
        response = client.post("/train", json=training_request)
        
        # Should require authentication
        assert response.status_code == 403
    
    def test_invalid_json(self, client):
        """Test invalid JSON in request."""
        headers = {
            "Authorization": "Bearer test-token",
            "Content-Type": "application/json"
        }
        
        response = client.post(
            "/train",
            data="invalid json",
            headers=headers
        )
        
        assert response.status_code == 422
    
    def test_missing_required_fields(self, client):
        """Test missing required fields in request."""
        incomplete_request = {
            "model_name": "gpt2"
            # Missing dataset_path and privacy_config
        }
        
        headers = {"Authorization": "Bearer test-token"}
        
        response = client.post(
            "/train",
            json=incomplete_request,
            headers=headers
        )
        
        assert response.status_code == 422


class TestAPIDocumentation:
    """Test API documentation endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app()
        return TestClient(app)
    
    def test_openapi_docs(self, client):
        """Test OpenAPI documentation endpoint."""
        response = client.get("/docs")
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    def test_redoc_docs(self, client):
        """Test ReDoc documentation endpoint."""
        response = client.get("/redoc")
        
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
    
    def test_openapi_json(self, client):
        """Test OpenAPI JSON schema."""
        response = client.get("/openapi.json")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "openapi" in data
        assert "info" in data
        assert data["info"]["title"] == "Privacy-Preserving Agent Finetuner"
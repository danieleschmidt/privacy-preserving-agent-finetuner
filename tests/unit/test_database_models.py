"""Unit tests for database models."""

import pytest
from datetime import datetime, timedelta
import uuid

from privacy_finetuner.database.models import (
    User, Dataset, TrainingJob, Model, PrivacyBudgetEntry, AuditLog
)


class TestUserModel:
    """Test User model."""
    
    def test_user_creation(self, test_database):
        """Test user creation with required fields."""
        user = User(
            username="testuser",
            email="test@example.com",
            hashed_password="hashed_password_123",
            full_name="Test User",
            is_active=True
        )
        
        test_database.add(user)
        test_database.commit()
        
        assert user.id is not None
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.is_active is True
        assert user.is_superuser is False
        assert user.created_at is not None
        assert user.updated_at is not None
    
    def test_user_privacy_preferences(self, test_database):
        """Test user privacy preferences."""
        user = User(
            username="privacyuser",
            email="privacy@example.com",
            hashed_password="hashed_password_123",
            privacy_preferences={
                "data_retention_days": 90,
                "consent_required": True
            },
            consent_given=True,
            consent_date=datetime.utcnow()
        )
        
        test_database.add(user)
        test_database.commit()
        
        assert user.privacy_preferences["data_retention_days"] == 90
        assert user.privacy_preferences["consent_required"] is True
        assert user.consent_given is True
        assert user.consent_date is not None


class TestDatasetModel:
    """Test Dataset model."""
    
    def test_dataset_creation(self, test_database):
        """Test dataset creation."""
        # Create owner first
        user = User(
            username="dataowner",
            email="owner@example.com",
            hashed_password="hashed_password_123"
        )
        test_database.add(user)
        test_database.flush()
        
        dataset = Dataset(
            name="Test Dataset",
            description="A test dataset for unit testing",
            file_path="/data/test_dataset.jsonl",
            format="jsonl",
            size_bytes=1024*1024,
            record_count=1000,
            contains_pii=True,
            privacy_level="high",
            data_classification="confidential",
            owner_id=user.id,
            file_hash="abc123def456",
            metadata_hash="xyz789"
        )
        
        test_database.add(dataset)
        test_database.commit()
        
        assert dataset.id is not None
        assert dataset.name == "Test Dataset"
        assert dataset.format == "jsonl"
        assert dataset.contains_pii is True
        assert dataset.privacy_level == "high"
        assert dataset.owner_id == user.id
    
    def test_dataset_privacy_levels(self, test_database):
        """Test dataset privacy level validation."""
        user = User(
            username="dataowner2",
            email="owner2@example.com",
            hashed_password="hashed_password_123"
        )
        test_database.add(user)
        test_database.flush()
        
        # Test valid privacy levels
        for level in ["low", "medium", "high"]:
            dataset = Dataset(
                name=f"Dataset {level}",
                file_path=f"/data/dataset_{level}.jsonl",
                format="jsonl",
                privacy_level=level,
                owner_id=user.id
            )
            test_database.add(dataset)
        
        test_database.commit()


class TestTrainingJobModel:
    """Test TrainingJob model."""
    
    def test_training_job_creation(self, test_database):
        """Test training job creation."""
        # Create dependencies
        user = User(
            username="trainer",
            email="trainer@example.com",
            hashed_password="hashed_password_123"
        )
        test_database.add(user)
        test_database.flush()
        
        dataset = Dataset(
            name="Training Dataset",
            file_path="/data/training.jsonl",
            format="jsonl",
            owner_id=user.id
        )
        test_database.add(dataset)
        test_database.flush()
        
        training_job = TrainingJob(
            job_name="Test Training Job",
            status="queued",
            model_name="gpt2",
            dataset_id=dataset.id,
            epochs=3,
            batch_size=8,
            learning_rate=5e-5,
            target_epsilon=1.0,
            target_delta=1e-5,
            noise_multiplier=0.5,
            max_grad_norm=1.0,
            accounting_mode="rdp",
            user_id=user.id
        )
        
        test_database.add(training_job)
        test_database.commit()
        
        assert training_job.id is not None
        assert training_job.job_name == "Test Training Job"
        assert training_job.status == "queued"
        assert training_job.target_epsilon == 1.0
        assert training_job.epsilon_spent == 0.0
        assert training_job.progress == 0.0
    
    def test_training_job_status_updates(self, test_database):
        """Test training job status updates."""
        user = User(
            username="trainer2",
            email="trainer2@example.com",
            hashed_password="hashed_password_123"
        )
        test_database.add(user)
        test_database.flush()
        
        dataset = Dataset(
            name="Training Dataset 2",
            file_path="/data/training2.jsonl",
            format="jsonl",
            owner_id=user.id
        )
        test_database.add(dataset)
        test_database.flush()
        
        training_job = TrainingJob(
            job_name="Test Training Job 2",
            status="queued",
            model_name="gpt2",
            dataset_id=dataset.id,
            epochs=3,
            batch_size=8,
            learning_rate=5e-5,
            target_epsilon=1.0,
            target_delta=1e-5,
            noise_multiplier=0.5,
            max_grad_norm=1.0,
            accounting_mode="rdp",
            user_id=user.id
        )
        test_database.add(training_job)
        test_database.commit()
        
        # Update job status
        training_job.status = "running"
        training_job.started_at = datetime.utcnow()
        training_job.progress = 0.5
        training_job.epsilon_spent = 0.3
        
        test_database.commit()
        
        assert training_job.status == "running"
        assert training_job.started_at is not None
        assert training_job.progress == 0.5
        assert training_job.epsilon_spent == 0.3


class TestModelModel:
    """Test Model model."""
    
    def test_model_creation(self, test_database):
        """Test model creation."""
        user = User(
            username="modelowner",
            email="model@example.com",
            hashed_password="hashed_password_123"
        )
        test_database.add(user)
        test_database.flush()
        
        model = Model(
            name="Test Model",
            version="1.0.0",
            description="A test model",
            base_model="gpt2",
            model_type="causal_lm",
            model_path="/models/test_model/",
            epsilon_spent=0.85,
            delta_value=1e-5,
            eval_loss=2.45,
            eval_accuracy=0.92,
            model_size_mb=500.0,
            owner_id=user.id
        )
        
        test_database.add(model)
        test_database.commit()
        
        assert model.id is not None
        assert model.name == "Test Model"
        assert model.version == "1.0.0"
        assert model.epsilon_spent == 0.85
        assert model.delta_value == 1e-5
        assert model.is_deployed is False


class TestPrivacyBudgetEntryModel:
    """Test PrivacyBudgetEntry model."""
    
    def test_privacy_budget_entry_creation(self, test_database):
        """Test privacy budget entry creation."""
        user = User(
            username="budgetuser",
            email="budget@example.com",
            hashed_password="hashed_password_123"
        )
        test_database.add(user)
        test_database.flush()
        
        entry = PrivacyBudgetEntry(
            epsilon_spent=0.1,
            delta_value=1e-5,
            operation="training_step",
            user_id=user.id,
            step_number=100,
            epoch_number=1,
            noise_multiplier=0.5,
            sample_rate=0.01,
            accounting_mode="rdp",
            metadata={"batch_size": 8, "learning_rate": 5e-5}
        )
        
        test_database.add(entry)
        test_database.commit()
        
        assert entry.id is not None
        assert entry.epsilon_spent == 0.1
        assert entry.operation == "training_step"
        assert entry.step_number == 100
        assert entry.metadata["batch_size"] == 8


class TestAuditLogModel:
    """Test AuditLog model."""
    
    def test_audit_log_creation(self, test_database):
        """Test audit log creation."""
        user = User(
            username="audituser",
            email="audit@example.com",
            hashed_password="hashed_password_123"
        )
        test_database.add(user)
        test_database.flush()
        
        audit_log = AuditLog(
            event_type="user_login",
            resource_type="user",
            resource_id=str(user.id),
            action="login",
            user_id=user.id,
            actor_ip="192.168.1.100",
            user_agent="Test Browser",
            session_id="session_123",
            status_code=200,
            contains_pii=False,
            metadata={"login_method": "password"},
            tags=["authentication", "login"]
        )
        
        test_database.add(audit_log)
        test_database.commit()
        
        assert audit_log.id is not None
        assert audit_log.event_type == "user_login"
        assert audit_log.action == "login"
        assert audit_log.status_code == 200
        assert audit_log.contains_pii is False
        assert "authentication" in audit_log.tags
    
    def test_audit_log_with_pii(self, test_database):
        """Test audit log with PII data."""
        user = User(
            username="piiuser",
            email="pii@example.com",
            hashed_password="hashed_password_123"
        )
        test_database.add(user)
        test_database.flush()
        
        retention_date = datetime.utcnow() + timedelta(days=90)
        
        audit_log = AuditLog(
            event_type="data_access",
            resource_type="dataset",
            action="read",
            user_id=user.id,
            contains_pii=True,
            data_classification="confidential",
            retention_until=retention_date,
            metadata={"access_reason": "training_preparation"}
        )
        
        test_database.add(audit_log)
        test_database.commit()
        
        assert audit_log.contains_pii is True
        assert audit_log.data_classification == "confidential"
        assert audit_log.retention_until == retention_date
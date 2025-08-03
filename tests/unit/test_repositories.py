"""Unit tests for repository layer."""

import pytest
from datetime import datetime, timedelta
import uuid

from privacy_finetuner.database.models import (
    User, Dataset, TrainingJob, Model, PrivacyBudgetEntry, AuditLog
)
from privacy_finetuner.database.repositories import (
    UserRepository, DatasetRepository, TrainingJobRepository,
    ModelRepository, PrivacyBudgetRepository, AuditLogRepository
)


class TestUserRepository:
    """Test UserRepository functionality."""
    
    def test_create_user(self, test_database):
        """Test user creation via repository."""
        repo = UserRepository(test_database)
        
        user = repo.create(
            username="testuser",
            email="test@example.com",
            hashed_password="hashed123",
            full_name="Test User",
            is_active=True
        )
        
        assert user.id is not None
        assert user.username == "testuser"
        assert user.email == "test@example.com"
    
    def test_get_by_username(self, test_database):
        """Test getting user by username."""
        repo = UserRepository(test_database)
        
        # Create user
        user = repo.create(
            username="uniqueuser",
            email="unique@example.com",
            hashed_password="hashed123"
        )
        
        # Retrieve by username
        found_user = repo.get_by_username("uniqueuser")
        
        assert found_user is not None
        assert found_user.id == user.id
        assert found_user.username == "uniqueuser"
    
    def test_get_by_email(self, test_database):
        """Test getting user by email."""
        repo = UserRepository(test_database)
        
        user = repo.create(
            username="emailuser",
            email="emailtest@example.com",
            hashed_password="hashed123"
        )
        
        found_user = repo.get_by_email("emailtest@example.com")
        
        assert found_user is not None
        assert found_user.id == user.id
        assert found_user.email == "emailtest@example.com"
    
    def test_search_users(self, test_database):
        """Test user search functionality."""
        repo = UserRepository(test_database)
        
        # Create multiple users
        repo.create(username="john_doe", email="john@example.com", hashed_password="123")
        repo.create(username="jane_smith", email="jane@example.com", hashed_password="123")
        repo.create(username="bob_jones", email="bob@example.com", hashed_password="123")
        
        # Search by username
        results = repo.search_users("john")
        assert len(results) == 1
        assert results[0].username == "john_doe"
        
        # Search by email domain
        results = repo.search_users("example.com")
        assert len(results) == 3


class TestDatasetRepository:
    """Test DatasetRepository functionality."""
    
    def setup_test_user(self, test_database):
        """Helper to create test user."""
        user_repo = UserRepository(test_database)
        return user_repo.create(
            username="datasetowner",
            email="owner@example.com",
            hashed_password="hashed123"
        )
    
    def test_create_dataset(self, test_database):
        """Test dataset creation."""
        user = self.setup_test_user(test_database)
        repo = DatasetRepository(test_database)
        
        dataset = repo.create(
            name="Test Dataset",
            file_path="/data/test.jsonl",
            format="jsonl",
            size_bytes=1024,
            record_count=100,
            owner_id=user.id,
            privacy_level="medium"
        )
        
        assert dataset.id is not None
        assert dataset.name == "Test Dataset"
        assert dataset.owner_id == user.id
    
    def test_get_by_owner(self, test_database):
        """Test getting datasets by owner."""
        user = self.setup_test_user(test_database)
        repo = DatasetRepository(test_database)
        
        # Create multiple datasets
        dataset1 = repo.create(
            name="Dataset 1",
            file_path="/data/1.jsonl",
            format="jsonl",
            owner_id=user.id
        )
        dataset2 = repo.create(
            name="Dataset 2",
            file_path="/data/2.jsonl",
            format="jsonl",
            owner_id=user.id
        )
        
        datasets = repo.get_by_owner(user.id)
        
        assert len(datasets) == 2
        dataset_names = [d.name for d in datasets]
        assert "Dataset 1" in dataset_names
        assert "Dataset 2" in dataset_names
    
    def test_get_by_privacy_level(self, test_database):
        """Test getting datasets by privacy level."""
        user = self.setup_test_user(test_database)
        repo = DatasetRepository(test_database)
        
        # Create datasets with different privacy levels
        repo.create(
            name="High Privacy Dataset",
            file_path="/data/high.jsonl",
            format="jsonl",
            privacy_level="high",
            owner_id=user.id
        )
        repo.create(
            name="Medium Privacy Dataset",
            file_path="/data/medium.jsonl",
            format="jsonl",
            privacy_level="medium",
            owner_id=user.id
        )
        
        high_privacy_datasets = repo.get_by_privacy_level("high")
        assert len(high_privacy_datasets) == 1
        assert high_privacy_datasets[0].name == "High Privacy Dataset"
    
    def test_get_datasets_with_pii(self, test_database):
        """Test getting datasets containing PII."""
        user = self.setup_test_user(test_database)
        repo = DatasetRepository(test_database)
        
        # Create dataset with PII
        repo.create(
            name="PII Dataset",
            file_path="/data/pii.jsonl",
            format="jsonl",
            contains_pii=True,
            owner_id=user.id
        )
        
        # Create dataset without PII
        repo.create(
            name="Non-PII Dataset",
            file_path="/data/no_pii.jsonl",
            format="jsonl",
            contains_pii=False,
            owner_id=user.id
        )
        
        pii_datasets = repo.get_datasets_with_pii()
        assert len(pii_datasets) == 1
        assert pii_datasets[0].name == "PII Dataset"


class TestTrainingJobRepository:
    """Test TrainingJobRepository functionality."""
    
    def setup_test_data(self, test_database):
        """Helper to create test user and dataset."""
        user_repo = UserRepository(test_database)
        dataset_repo = DatasetRepository(test_database)
        
        user = user_repo.create(
            username="trainer",
            email="trainer@example.com",
            hashed_password="hashed123"
        )
        
        dataset = dataset_repo.create(
            name="Training Dataset",
            file_path="/data/train.jsonl",
            format="jsonl",
            owner_id=user.id
        )
        
        return user, dataset
    
    def test_create_training_job(self, test_database):
        """Test training job creation."""
        user, dataset = self.setup_test_data(test_database)
        repo = TrainingJobRepository(test_database)
        
        job = repo.create(
            job_name="Test Training",
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
        
        assert job.id is not None
        assert job.job_name == "Test Training"
        assert job.status == "queued"
    
    def test_get_by_status(self, test_database):
        """Test getting jobs by status."""
        user, dataset = self.setup_test_data(test_database)
        repo = TrainingJobRepository(test_database)
        
        # Create jobs with different statuses
        job1 = repo.create(
            job_name="Running Job",
            status="running",
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
        
        job2 = repo.create(
            job_name="Completed Job",
            status="completed",
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
        
        running_jobs = repo.get_by_status("running")
        completed_jobs = repo.get_by_status("completed")
        
        assert len(running_jobs) == 1
        assert len(completed_jobs) == 1
        assert running_jobs[0].job_name == "Running Job"
        assert completed_jobs[0].job_name == "Completed Job"
    
    def test_get_privacy_budget_summary(self, test_database):
        """Test privacy budget summary."""
        user, dataset = self.setup_test_data(test_database)
        repo = TrainingJobRepository(test_database)
        
        # Create jobs with different epsilon values
        repo.create(
            job_name="Job 1",
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
            epsilon_spent=0.5,
            user_id=user.id
        )
        
        repo.create(
            job_name="Job 2",
            model_name="gpt2",
            dataset_id=dataset.id,
            epochs=3,
            batch_size=8,
            learning_rate=5e-5,
            target_epsilon=2.0,
            target_delta=1e-5,
            noise_multiplier=0.5,
            max_grad_norm=1.0,
            accounting_mode="rdp",
            epsilon_spent=1.0,
            user_id=user.id
        )
        
        summary = repo.get_privacy_budget_summary(user.id)
        
        assert summary["total_epsilon_spent"] == 1.5
        assert summary["total_jobs"] == 2
        assert summary["average_epsilon_per_job"] == 0.75
        assert summary["max_epsilon_single_job"] == 1.0
    
    def test_update_job_progress(self, test_database):
        """Test updating job progress."""
        user, dataset = self.setup_test_data(test_database)
        repo = TrainingJobRepository(test_database)
        
        job = repo.create(
            job_name="Progress Job",
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
        
        # Update progress
        updated_job = repo.update_job_progress(
            job.id,
            progress=0.75,
            status="running",
            epsilon_spent=0.3
        )
        
        assert updated_job.progress == 0.75
        assert updated_job.status == "running"
        assert updated_job.epsilon_spent == 0.3


class TestPrivacyBudgetRepository:
    """Test PrivacyBudgetRepository functionality."""
    
    def setup_test_data(self, test_database):
        """Helper to create test data."""
        user_repo = UserRepository(test_database)
        user = user_repo.create(
            username="budgetuser",
            email="budget@example.com",
            hashed_password="hashed123"
        )
        return user
    
    def test_record_budget_entry(self, test_database):
        """Test recording privacy budget entry."""
        user = self.setup_test_data(test_database)
        repo = PrivacyBudgetRepository(test_database)
        
        entry = repo.create(
            epsilon_spent=0.1,
            delta_value=1e-5,
            operation="training_step",
            user_id=user.id,
            step_number=100,
            noise_multiplier=0.5,
            accounting_mode="rdp"
        )
        
        assert entry.id is not None
        assert entry.epsilon_spent == 0.1
        assert entry.operation == "training_step"
    
    def test_get_user_budget_summary(self, test_database):
        """Test getting user budget summary."""
        user = self.setup_test_data(test_database)
        repo = PrivacyBudgetRepository(test_database)
        
        # Create multiple budget entries
        repo.create(
            epsilon_spent=0.1,
            delta_value=1e-5,
            operation="training_step",
            user_id=user.id
        )
        repo.create(
            epsilon_spent=0.2,
            delta_value=1e-5,
            operation="evaluation_step",
            user_id=user.id
        )
        
        summary = repo.get_user_budget_summary(user.id)
        
        assert summary["total_epsilon_spent"] == 0.3
        assert summary["total_entries"] == 2
    
    def test_get_budget_by_operation(self, test_database):
        """Test getting budget grouped by operation."""
        user = self.setup_test_data(test_database)
        repo = PrivacyBudgetRepository(test_database)
        
        # Create entries with different operations
        repo.create(
            epsilon_spent=0.15,
            delta_value=1e-5,
            operation="training_step",
            user_id=user.id
        )
        repo.create(
            epsilon_spent=0.25,
            delta_value=1e-5,
            operation="training_step",
            user_id=user.id
        )
        repo.create(
            epsilon_spent=0.1,
            delta_value=1e-5,
            operation="evaluation_step",
            user_id=user.id
        )
        
        budget_by_operation = repo.get_budget_by_operation(user.id)
        
        # Should be sorted by total epsilon descending
        assert len(budget_by_operation) == 2
        assert budget_by_operation[0][0] == "training_step"  # operation name
        assert budget_by_operation[0][1] == 0.4  # total epsilon
        assert budget_by_operation[0][2] == 2  # count


class TestAuditLogRepository:
    """Test AuditLogRepository functionality."""
    
    def setup_test_data(self, test_database):
        """Helper to create test user."""
        user_repo = UserRepository(test_database)
        return user_repo.create(
            username="audituser",
            email="audit@example.com",
            hashed_password="hashed123"
        )
    
    def test_log_event(self, test_database):
        """Test logging an audit event."""
        user = self.setup_test_data(test_database)
        repo = AuditLogRepository(test_database)
        
        entry = repo.log_event(
            event_type="user_login",
            resource_type="user",
            action="login",
            user_id=user.id,
            resource_id=str(user.id),
            actor_ip="192.168.1.100",
            status_code=200
        )
        
        assert entry.id is not None
        assert entry.event_type == "user_login"
        assert entry.action == "login"
        assert entry.user_id == user.id
        assert entry.status_code == 200
    
    def test_get_security_events(self, test_database):
        """Test getting security events."""
        user = self.setup_test_data(test_database)
        repo = AuditLogRepository(test_database)
        
        # Create security-related events
        repo.log_event(
            event_type="login_failed",
            resource_type="user",
            action="login",
            user_id=user.id,
            status_code=401
        )
        
        repo.log_event(
            event_type="permission_denied",
            resource_type="dataset",
            action="read",
            user_id=user.id,
            status_code=403
        )
        
        # Create non-security event
        repo.log_event(
            event_type="data_export",
            resource_type="model",
            action="export",
            user_id=user.id,
            status_code=200
        )
        
        security_events = repo.get_security_events(hours=24)
        
        assert len(security_events) == 2
        event_types = [event.event_type for event in security_events]
        assert "login_failed" in event_types
        assert "permission_denied" in event_types
        assert "data_export" not in event_types
    
    def test_get_privacy_events(self, test_database):
        """Test getting privacy-related events."""
        user = self.setup_test_data(test_database)
        repo = AuditLogRepository(test_database)
        
        # Create event with PII
        repo.log_event(
            event_type="data_access",
            resource_type="dataset",
            action="read",
            user_id=user.id,
            contains_pii=True,
            data_classification="confidential"
        )
        
        # Create event without PII
        repo.log_event(
            event_type="model_training",
            resource_type="model",
            action="create",
            user_id=user.id,
            contains_pii=False
        )
        
        privacy_events = repo.get_privacy_events(days=7)
        
        assert len(privacy_events) == 1
        assert privacy_events[0].event_type == "data_access"
        assert privacy_events[0].contains_pii is True
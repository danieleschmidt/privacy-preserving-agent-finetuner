"""Database seeder for initial data and test data."""

import logging
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from passlib.context import CryptContext

from ..models import User, Dataset, TrainingJob, Model, PrivacyBudgetEntry, AuditLog
from ..connection import get_database

logger = logging.getLogger(__name__)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class DatabaseSeeder:
    """Database seeder for creating initial and test data."""
    
    def __init__(self, session: Session):
        self.session = session
    
    def seed_all(self, include_test_data: bool = False):
        """Seed all data."""
        logger.info("Starting database seeding...")
        
        # Create users
        users = self.create_users()
        
        # Create datasets
        datasets = self.create_datasets(users)
        
        if include_test_data:
            # Create training jobs
            training_jobs = self.create_training_jobs(users, datasets)
            
            # Create models
            models = self.create_models(users, training_jobs)
            
            # Create privacy budget entries
            self.create_privacy_budget_entries(users, training_jobs)
            
            # Create audit logs
            self.create_audit_logs(users)
        
        self.session.commit()
        logger.info("Database seeding completed")
    
    def create_users(self) -> List[User]:
        """Create initial users."""
        users_data = [
            {
                "username": "admin",
                "email": "admin@privacy-finetuner.com",
                "password": "admin123",
                "full_name": "System Administrator",
                "is_superuser": True,
                "privacy_preferences": {
                    "data_retention_days": 90,
                    "consent_required": True,
                    "audit_level": "full"
                },
                "consent_given": True,
                "consent_date": datetime.utcnow()
            },
            {
                "username": "researcher",
                "email": "researcher@privacy-finetuner.com",
                "password": "research123",
                "full_name": "Privacy Researcher",
                "is_superuser": False,
                "privacy_preferences": {
                    "data_retention_days": 365,
                    "consent_required": True,
                    "audit_level": "standard"
                },
                "consent_given": True,
                "consent_date": datetime.utcnow()
            },
            {
                "username": "demo_user",
                "email": "demo@privacy-finetuner.com",
                "password": "demo123",
                "full_name": "Demo User",
                "is_superuser": False,
                "privacy_preferences": {
                    "data_retention_days": 30,
                    "consent_required": True,
                    "audit_level": "basic"
                },
                "consent_given": True,
                "consent_date": datetime.utcnow()
            }
        ]
        
        users = []
        for user_data in users_data:
            # Check if user already exists
            existing_user = self.session.query(User).filter(
                User.username == user_data["username"]
            ).first()
            
            if existing_user:
                logger.info(f"User {user_data['username']} already exists, skipping")
                users.append(existing_user)
                continue
            
            # Hash password
            password = user_data.pop("password")
            hashed_password = pwd_context.hash(password)
            
            user = User(
                id=uuid.uuid4(),
                hashed_password=hashed_password,
                **user_data
            )
            
            self.session.add(user)
            users.append(user)
            logger.info(f"Created user: {user.username}")
        
        return users
    
    def create_datasets(self, users: List[User]) -> List[Dataset]:
        """Create sample datasets."""
        datasets_data = [
            {
                "name": "Sample Medical Records",
                "description": "Synthetic medical records for privacy research",
                "file_path": "/data/medical_records_synthetic.jsonl",
                "format": "jsonl",
                "size_bytes": 1024*1024*50,  # 50MB
                "record_count": 10000,
                "contains_pii": True,
                "privacy_level": "high",
                "data_classification": "confidential",
                "retention_policy": "HIPAA_compliant",
                "is_public": False,
                "file_hash": "a1b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef123456",
                "metadata_hash": "123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
            },
            {
                "name": "Financial Transactions",
                "description": "Anonymized financial transaction data",
                "file_path": "/data/financial_transactions.csv",
                "format": "csv",
                "size_bytes": 1024*1024*25,  # 25MB
                "record_count": 50000,
                "contains_pii": False,
                "privacy_level": "medium",
                "data_classification": "internal",
                "retention_policy": "5_years",
                "is_public": False,
                "file_hash": "b2c3d4e5f6789012345678901234567890abcdef1234567890abcdef1234567",
                "metadata_hash": "23456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef1"
            },
            {
                "name": "Public Reviews Dataset",
                "description": "Public product reviews for sentiment analysis",
                "file_path": "/data/public_reviews.jsonl",
                "format": "jsonl",
                "size_bytes": 1024*1024*100,  # 100MB
                "record_count": 100000,
                "contains_pii": False,
                "privacy_level": "low",
                "data_classification": "public",
                "retention_policy": "indefinite",
                "is_public": True,
                "file_hash": "c3d4e5f6789012345678901234567890abcdef1234567890abcdef12345678",
                "metadata_hash": "3456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef12"
            }
        ]
        
        datasets = []
        for i, dataset_data in enumerate(datasets_data):
            # Assign to different users
            owner = users[i % len(users)]
            
            dataset = Dataset(
                id=uuid.uuid4(),
                owner_id=owner.id,
                **dataset_data
            )
            
            self.session.add(dataset)
            datasets.append(dataset)
            logger.info(f"Created dataset: {dataset.name}")
        
        return datasets
    
    def create_training_jobs(self, users: List[User], datasets: List[Dataset]) -> List[TrainingJob]:
        """Create sample training jobs."""
        training_jobs_data = [
            {
                "job_name": "Medical Records Privacy Training",
                "status": "completed",
                "model_name": "microsoft/DialoGPT-medium",
                "epochs": 3,
                "batch_size": 8,
                "learning_rate": 5e-5,
                "max_steps": 1000,
                "warmup_steps": 100,
                "target_epsilon": 1.0,
                "target_delta": 1e-5,
                "noise_multiplier": 0.5,
                "max_grad_norm": 1.0,
                "accounting_mode": "rdp",
                "epsilon_spent": 0.85,
                "actual_steps": 900,
                "sample_rate": 0.01,
                "final_loss": 2.45,
                "best_eval_loss": 2.32,
                "training_time_seconds": 3600,
                "gpu_hours_used": 1.0,
                "progress": 1.0
            },
            {
                "job_name": "Financial Data Fine-tuning",
                "status": "running",
                "model_name": "distilbert-base-uncased",
                "epochs": 5,
                "batch_size": 16,
                "learning_rate": 3e-5,
                "max_steps": 2000,
                "warmup_steps": 200,
                "target_epsilon": 2.0,
                "target_delta": 1e-6,
                "noise_multiplier": 0.3,
                "max_grad_norm": 1.5,
                "accounting_mode": "gdp",
                "epsilon_spent": 1.2,
                "actual_steps": 1200,
                "sample_rate": 0.015,
                "progress": 0.6
            },
            {
                "job_name": "Public Reviews Classification",
                "status": "queued",
                "model_name": "bert-base-uncased",
                "epochs": 2,
                "batch_size": 32,
                "learning_rate": 2e-5,
                "max_steps": 500,
                "warmup_steps": 50,
                "target_epsilon": 3.0,
                "target_delta": 1e-4,
                "noise_multiplier": 0.8,
                "max_grad_norm": 0.5,
                "accounting_mode": "rdp",
                "epsilon_spent": 0.0,
                "actual_steps": 0,
                "sample_rate": 0.02,
                "progress": 0.0
            }
        ]
        
        training_jobs = []
        base_time = datetime.utcnow() - timedelta(days=7)
        
        for i, job_data in enumerate(training_jobs_data):
            user = users[i % len(users)]
            dataset = datasets[i % len(datasets)]
            
            # Set timestamps based on status
            created_at = base_time + timedelta(days=i)
            started_at = None
            completed_at = None
            
            if job_data["status"] in ["running", "completed"]:
                started_at = created_at + timedelta(hours=1)
            
            if job_data["status"] == "completed":
                completed_at = started_at + timedelta(hours=2)
            
            training_job = TrainingJob(
                id=uuid.uuid4(),
                user_id=user.id,
                dataset_id=dataset.id,
                created_at=created_at,
                started_at=started_at,
                completed_at=completed_at,
                **job_data
            )
            
            self.session.add(training_job)
            training_jobs.append(training_job)
            logger.info(f"Created training job: {training_job.job_name}")
        
        return training_jobs
    
    def create_models(self, users: List[User], training_jobs: List[TrainingJob]) -> List[Model]:
        """Create sample models."""
        models = []
        
        for i, training_job in enumerate(training_jobs):
            if training_job.status != "completed":
                continue
            
            model = Model(
                id=uuid.uuid4(),
                name=f"privacy-model-{i+1}",
                version="1.0.0",
                description=f"Privacy-preserving model from {training_job.job_name}",
                base_model=training_job.model_name,
                model_type="causal_lm",
                model_path=f"/models/privacy-model-{i+1}/",
                config_path=f"/models/privacy-model-{i+1}/config.json",
                tokenizer_path=f"/models/privacy-model-{i+1}/tokenizer/",
                training_job_id=training_job.id,
                epsilon_spent=training_job.epsilon_spent,
                delta_value=training_job.target_delta,
                noise_multiplier=training_job.noise_multiplier,
                max_grad_norm=training_job.max_grad_norm,
                eval_loss=training_job.final_loss,
                eval_accuracy=0.92 - (i * 0.05),
                eval_perplexity=15.0 + (i * 2.0),
                model_size_mb=500.0 + (i * 50.0),
                is_deployed=False,
                deployment_status="not_deployed",
                owner_id=training_job.user_id
            )
            
            self.session.add(model)
            models.append(model)
            logger.info(f"Created model: {model.name}")
        
        return models
    
    def create_privacy_budget_entries(self, users: List[User], training_jobs: List[TrainingJob]):
        """Create sample privacy budget entries."""
        for training_job in training_jobs:
            if training_job.actual_steps == 0:
                continue
            
            # Create entries for each training step
            epsilon_per_step = training_job.epsilon_spent / training_job.actual_steps
            
            for step in range(0, training_job.actual_steps, 100):  # Every 100 steps
                entry = PrivacyBudgetEntry(
                    id=uuid.uuid4(),
                    epsilon_spent=epsilon_per_step * 100,
                    delta_value=training_job.target_delta,
                    operation="training_step",
                    training_job_id=training_job.id,
                    user_id=training_job.user_id,
                    step_number=step,
                    epoch_number=step // (training_job.actual_steps // training_job.epochs),
                    batch_number=step % 100,
                    metadata={
                        "batch_size": training_job.batch_size,
                        "learning_rate": training_job.learning_rate
                    },
                    noise_multiplier=training_job.noise_multiplier,
                    sample_rate=training_job.sample_rate,
                    accounting_mode=training_job.accounting_mode
                )
                
                self.session.add(entry)
        
        logger.info("Created privacy budget entries")
    
    def create_audit_logs(self, users: List[User]):
        """Create sample audit logs."""
        audit_events = [
            {
                "event_type": "user_login",
                "resource_type": "user",
                "action": "login",
                "status_code": 200,
                "contains_pii": False,
                "metadata": {"login_method": "password"}
            },
            {
                "event_type": "dataset_access",
                "resource_type": "dataset",
                "action": "read",
                "status_code": 200,
                "contains_pii": True,
                "data_classification": "confidential",
                "metadata": {"access_reason": "training_preparation"}
            },
            {
                "event_type": "training_start",
                "resource_type": "training_job",
                "action": "create",
                "status_code": 201,
                "contains_pii": False,
                "metadata": {"privacy_budget": "1.0"}
            },
            {
                "event_type": "model_export",
                "resource_type": "model",
                "action": "export",
                "status_code": 200,
                "contains_pii": False,
                "metadata": {"export_format": "pytorch"}
            }
        ]
        
        base_time = datetime.utcnow() - timedelta(days=3)
        
        for i, event_data in enumerate(audit_events):
            for j, user in enumerate(users):
                audit_log = AuditLog(
                    id=uuid.uuid4(),
                    user_id=user.id,
                    resource_id=str(uuid.uuid4()),
                    actor_ip="192.168.1.100",
                    user_agent="Mozilla/5.0 (Privacy Finetuner Client)",
                    session_id=f"session_{j}_{i}",
                    created_at=base_time + timedelta(hours=i*6 + j),
                    **event_data
                )
                
                self.session.add(audit_log)
        
        logger.info("Created audit logs")


def seed_database(include_test_data: bool = False):
    """Seed the database with initial data."""
    db = get_database()
    
    with db.session_scope() as session:
        seeder = DatabaseSeeder(session)
        seeder.seed_all(include_test_data=include_test_data)
    
    logger.info("Database seeding completed successfully")


if __name__ == "__main__":
    import sys
    include_test = "--test-data" in sys.argv
    seed_database(include_test_data=include_test)
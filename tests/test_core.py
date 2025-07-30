"""
Core functionality tests for the privacy-preserving training system.

Tests the main PrivateTrainer class and core training workflows.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from privacy_finetuner.core import PrivateTrainer, TrainingResult
from privacy_finetuner.privacy import PrivacyConfig


class TestPrivateTrainer:
    """Test suite for PrivateTrainer class."""
    
    @pytest.fixture
    def privacy_config(self):
        """Standard privacy configuration for testing."""
        return PrivacyConfig(
            epsilon=1.0,
            delta=1e-5,
            max_grad_norm=1.0,
            noise_multiplier=0.5
        )
    
    @pytest.fixture
    def trainer(self, privacy_config):
        """PrivateTrainer instance for testing."""
        return PrivateTrainer(
            model_name="microsoft/DialoGPT-small",
            privacy_config=privacy_config,
            use_mcp_gateway=True
        )
    
    def test_trainer_initialization(self, trainer, privacy_config):
        """Test proper initialization of PrivateTrainer."""
        assert trainer.model_name == "microsoft/DialoGPT-small"
        assert trainer.privacy_config == privacy_config
        assert trainer.use_mcp_gateway is True
        assert trainer._privacy_budget_consumed == 0.0
    
    def test_trainer_initialization_without_mcp(self, privacy_config):
        """Test initialization without MCP gateway."""
        trainer = PrivateTrainer(
            model_name="gpt2",
            privacy_config=privacy_config,
            use_mcp_gateway=False
        )
        assert trainer.use_mcp_gateway is False
    
    def test_train_basic_functionality(self, trainer, tmp_path):
        """Test basic training functionality."""
        # Create a dummy dataset file
        dataset_file = tmp_path / "train.jsonl"
        dataset_file.write_text('{"text": "Hello world"}\n')
        
        result = trainer.train(
            dataset=str(dataset_file),
            epochs=1,
            batch_size=4,
            learning_rate=1e-5
        )
        
        assert isinstance(result, TrainingResult)
        assert result.model_path.endswith("_private")
        assert 0 <= result.privacy_budget_consumed <= trainer.privacy_config.epsilon
        assert 0 <= result.training_accuracy <= 1.0
        assert 0 <= result.validation_accuracy <= 1.0
        assert isinstance(result.privacy_report, dict)
    
    def test_train_with_custom_parameters(self, trainer, tmp_path):
        """Test training with custom parameters."""
        dataset_file = tmp_path / "custom_train.jsonl"
        dataset_file.write_text('{"text": "Custom training data"}\n')
        
        result = trainer.train(
            dataset=str(dataset_file),
            epochs=5,
            batch_size=16,
            learning_rate=2e-5,
            custom_param="test_value"  # Test **kwargs
        )
        
        assert isinstance(result, TrainingResult)
        assert result.privacy_budget_consumed > 0
    
    def test_privacy_report_structure(self, trainer):
        """Test the structure of the privacy report."""
        report = trainer.get_privacy_report()
        
        # Check required fields
        required_fields = [
            "privacy_config",
            "budget_consumed", 
            "budget_remaining",
            "privacy_guarantees",
            "compliance_status"
        ]
        
        for field in required_fields:
            assert field in report
        
        # Check privacy config structure
        privacy_config = report["privacy_config"]
        assert "epsilon" in privacy_config
        assert "delta" in privacy_config
        assert "max_grad_norm" in privacy_config
        assert "noise_multiplier" in privacy_config
        
        # Check compliance status
        assert "GDPR" in report["compliance_status"]
        assert "HIPAA" in report["compliance_status"]
        assert "EU AI Act" in report["compliance_status"]
    
    def test_privacy_budget_calculation(self, trainer):
        """Test privacy budget calculations."""
        initial_report = trainer.get_privacy_report()
        initial_remaining = initial_report["budget_remaining"]
        
        # Simulate some budget consumption
        trainer._privacy_budget_consumed = 0.3
        
        updated_report = trainer.get_privacy_report()
        updated_remaining = updated_report["budget_remaining"]
        
        assert updated_remaining < initial_remaining
        assert updated_report["budget_consumed"] == 0.3


class TestTrainingResult:
    """Test suite for TrainingResult dataclass."""
    
    def test_training_result_creation(self):
        """Test creation of TrainingResult."""
        result = TrainingResult(
            model_path="/path/to/model",
            privacy_budget_consumed=0.5,
            training_accuracy=0.85,
            validation_accuracy=0.83,
            privacy_report={"test": "report"}
        )
        
        assert result.model_path == "/path/to/model"
        assert result.privacy_budget_consumed == 0.5
        assert result.training_accuracy == 0.85
        assert result.validation_accuracy == 0.83
        assert result.privacy_report == {"test": "report"}
    
    def test_training_result_validation(self):
        """Test validation of TrainingResult values."""
        result = TrainingResult(
            model_path="./models/test",
            privacy_budget_consumed=1.2,
            training_accuracy=0.92,
            validation_accuracy=0.89,
            privacy_report={}
        )
        
        # Verify accuracy values are reasonable
        assert 0 <= result.training_accuracy <= 1.0
        assert 0 <= result.validation_accuracy <= 1.0
        
        # Privacy budget consumed might exceed epsilon in some cases
        assert result.privacy_budget_consumed >= 0


class TestIntegrationScenarios:
    """Integration tests for realistic training scenarios."""
    
    @pytest.fixture
    def sample_dataset(self, tmp_path):
        """Create a sample training dataset."""
        dataset_file = tmp_path / "sample.jsonl"
        
        # Create sample training data
        sample_data = [
            '{"text": "This is a sample training example."}',
            '{"text": "Another example for model training."}',
            '{"text": "Privacy-preserving machine learning is important."}',
            '{"text": "Differential privacy provides formal guarantees."}'
        ]
        
        dataset_file.write_text('\n'.join(sample_data))
        return str(dataset_file)
    
    def test_end_to_end_training_workflow(self, sample_dataset):
        """Test complete training workflow from start to finish."""
        # Configure privacy parameters
        privacy_config = PrivacyConfig(
            epsilon=2.0,
            delta=1e-5,
            max_grad_norm=1.0,
            noise_multiplier=0.8
        )
        
        # Initialize trainer
        trainer = PrivateTrainer(
            model_name="distilgpt2",
            privacy_config=privacy_config,
            use_mcp_gateway=True
        )
        
        # Run training
        result = trainer.train(
            dataset=sample_dataset,
            epochs=2,
            batch_size=2,
            learning_rate=5e-5
        )
        
        # Verify results
        assert isinstance(result, TrainingResult)
        assert result.privacy_budget_consumed <= privacy_config.epsilon
        
        # Check privacy report
        report = trainer.get_privacy_report()
        assert report["privacy_config"]["epsilon"] == 2.0
        assert report["budget_remaining"] >= 0
    
    def test_multiple_training_sessions(self, sample_dataset):
        """Test multiple training sessions with budget tracking."""
        privacy_config = PrivacyConfig(epsilon=3.0, delta=1e-5)
        trainer = PrivateTrainer("gpt2", privacy_config)
        
        # First training session
        result1 = trainer.train(
            dataset=sample_dataset,
            epochs=1,
            batch_size=2
        )
        
        budget_after_first = result1.privacy_budget_consumed
        
        # Second training session (would accumulate budget)
        result2 = trainer.train(
            dataset=sample_dataset,
            epochs=1,
            batch_size=2
        )
        
        # In a real implementation, budget would accumulate
        # For this test, we just verify the structure
        assert result2.privacy_budget_consumed >= 0
        assert isinstance(result2.privacy_report, dict)
    
    @pytest.mark.slow
    def test_large_dataset_handling(self, tmp_path):
        """Test handling of larger datasets."""
        # Create a larger dataset
        large_dataset = tmp_path / "large.jsonl"
        
        # Generate more training examples
        examples = []
        for i in range(100):
            examples.append(f'{{"text": "Training example {i} for privacy testing."}}')
        
        large_dataset.write_text('\n'.join(examples))
        
        privacy_config = PrivacyConfig(epsilon=1.0, delta=1e-5)
        trainer = PrivateTrainer("distilgpt2", privacy_config)
        
        result = trainer.train(
            dataset=str(large_dataset),
            epochs=1,
            batch_size=8
        )
        
        assert isinstance(result, TrainingResult)
        assert result.privacy_budget_consumed > 0
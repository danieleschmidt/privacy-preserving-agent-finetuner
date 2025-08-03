"""Integration tests for the complete training pipeline."""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, Mock

from privacy_finetuner.core.trainer import PrivateTrainer
from privacy_finetuner.core.privacy_config import PrivacyConfig
from privacy_finetuner.core.context_guard import ContextGuard, RedactionStrategy
from privacy_finetuner.utils.monitoring import PrivacyBudgetMonitor


class TestTrainingPipeline:
    """Test complete training pipeline integration."""
    
    @pytest.fixture
    def sample_dataset(self):
        """Create a sample dataset file."""
        data = [
            {"text": "This is a sample training sentence."},
            {"text": "Another training example for testing."},
            {"text": "Privacy-preserving machine learning is important."},
            {"text": "Differential privacy provides formal guarantees."},
            {"text": "Test data for integration testing purposes."}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
            return Path(f.name)
    
    @pytest.fixture
    def privacy_config(self):
        """Create test privacy configuration."""
        return PrivacyConfig(
            epsilon=2.0,
            delta=1e-5,
            max_grad_norm=1.0,
            noise_multiplier=0.5,
            accounting_mode="rdp"
        )
    
    @patch('privacy_finetuner.core.trainer.AutoModelForCausalLM')
    @patch('privacy_finetuner.core.trainer.AutoTokenizer')
    @patch('privacy_finetuner.core.trainer.torch')
    def test_end_to_end_training(
        self, 
        mock_torch, 
        mock_tokenizer, 
        mock_model,
        sample_dataset, 
        privacy_config
    ):
        """Test end-to-end training pipeline."""
        # Mock torch components
        mock_torch.optim.AdamW.return_value = Mock()
        mock_torch.nn.utils.clip_grad_norm_ = Mock()
        mock_torch.tensor = lambda x: x
        
        # Mock tokenizer
        tokenizer_instance = Mock()
        tokenizer_instance.pad_token = None
        tokenizer_instance.eos_token = "<eos>"
        tokenizer_instance.pad_token_id = 0
        tokenizer_instance.return_value = {
            "input_ids": [1, 2, 3, 4, 5],
            "attention_mask": [1, 1, 1, 1, 1]
        }
        mock_tokenizer.from_pretrained.return_value = tokenizer_instance
        
        # Mock model
        model_instance = Mock()
        mock_outputs = Mock()
        mock_outputs.loss.item.return_value = 2.5
        mock_outputs.loss.backward = Mock()
        model_instance.return_value = mock_outputs
        model_instance.train = Mock()
        model_instance.save_pretrained = Mock()
        mock_model.from_pretrained.return_value = model_instance
        
        # Mock DataLoader
        with patch('privacy_finetuner.core.trainer.DataLoader') as mock_dataloader:
            mock_dataloader_instance = Mock()
            # Simulate one batch
            mock_dataloader_instance.__iter__.return_value = iter([
                {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}
            ])
            mock_dataloader.return_value = mock_dataloader_instance
            
            # Create trainer
            trainer = PrivateTrainer(
                model_name="microsoft/DialoGPT-small",
                privacy_config=privacy_config
            )
            
            # Run training
            results = trainer.train(
                dataset=str(sample_dataset),
                epochs=1,
                batch_size=2,
                learning_rate=1e-5
            )
            
            # Verify results
            assert results["status"] == "training_complete"
            assert results["epochs_completed"] == 1
            assert "total_steps" in results
            assert "model_path" in results
            
            # Verify privacy report
            privacy_report = trainer.get_privacy_report()
            assert "epsilon_spent" in privacy_report
            assert "remaining_budget" in privacy_report
        
        # Cleanup
        sample_dataset.unlink()
    
    def test_privacy_config_validation_pipeline(self):
        """Test privacy configuration validation in pipeline."""
        # Test valid configuration
        valid_config = PrivacyConfig(
            epsilon=1.0,
            delta=1e-5,
            max_grad_norm=1.0,
            noise_multiplier=0.5
        )
        valid_config.validate()  # Should not raise
        
        # Test invalid configurations
        with pytest.raises(ValueError, match="Epsilon must be positive"):
            invalid_config = PrivacyConfig(epsilon=-1.0)
            invalid_config.validate()
        
        with pytest.raises(ValueError, match="Delta must be between 0 and 1"):
            invalid_config = PrivacyConfig(delta=1.5)
            invalid_config.validate()
    
    def test_context_protection_integration(self):
        """Test context protection integration."""
        guard = ContextGuard([RedactionStrategy.PII_REMOVAL])
        
        sensitive_text = "John Doe's phone is 555-123-4567 and email is john@example.com"
        protected_text = guard.protect(sensitive_text, sensitivity_level="high")
        
        # Should have redacted PII
        assert "555-123-4567" not in protected_text
        assert "john@example.com" not in protected_text
        assert "[PHONE]" in protected_text
        assert "[EMAIL]" in protected_text
        
        # Test explanation
        explanation = guard.explain_redactions(sensitive_text)
        assert explanation["total_redactions"] > 0
        assert len(explanation["redaction_details"]) > 0
    
    def test_privacy_budget_monitoring_integration(self):
        """Test privacy budget monitoring integration."""
        monitor = PrivacyBudgetMonitor(
            total_epsilon=2.0,
            total_delta=1e-5
        )
        
        # Record some privacy events
        monitor.record_event(
            epsilon_spent=0.5,
            delta=1e-5,
            operation="training_step",
            metadata={"step": 100}
        )
        
        monitor.record_event(
            epsilon_spent=0.3,
            delta=1e-5,
            operation="evaluation_step",
            metadata={"step": 200}
        )
        
        # Check budget status
        status = monitor.generate_compliance_report()
        
        assert status["total_spent"] == 0.8
        assert status["remaining_budget"] == 1.2
        assert status["budget_utilization"] == 0.4
        assert status["total_operations"] == 2
        assert status["compliance_status"] == "compliant"
    
    def test_privacy_cost_estimation(self, privacy_config):
        """Test privacy cost estimation."""
        # Test basic estimation
        estimated_cost = privacy_config.estimate_privacy_cost(
            steps=1000,
            sample_rate=0.01
        )
        
        assert estimated_cost > 0
        assert isinstance(estimated_cost, float)
        
        # Test with different parameters
        higher_cost = privacy_config.estimate_privacy_cost(
            steps=2000,  # More steps
            sample_rate=0.02  # Higher sample rate
        )
        
        assert higher_cost > estimated_cost
    
    @patch('privacy_finetuner.core.trainer.torch')
    @patch('privacy_finetuner.core.trainer.AutoTokenizer')
    @patch('privacy_finetuner.core.trainer.AutoModelForCausalLM')
    def test_trainer_error_recovery(
        self, 
        mock_model, 
        mock_tokenizer, 
        mock_torch,
        sample_dataset, 
        privacy_config
    ):
        """Test trainer error recovery mechanisms."""
        # Mock setup that will fail during training
        mock_tokenizer.from_pretrained.side_effect = Exception("Model loading failed")
        
        trainer = PrivateTrainer(
            model_name="nonexistent/model",
            privacy_config=privacy_config
        )
        
        # Training should handle the error gracefully
        with pytest.raises(Exception, match="Model loading failed"):
            trainer.train(
                dataset=str(sample_dataset),
                epochs=1,
                batch_size=2,
                learning_rate=1e-5
            )
        
        # Cleanup
        sample_dataset.unlink()


class TestDataPipeline:
    """Test data processing pipeline."""
    
    def test_dataset_loading_formats(self):
        """Test loading different dataset formats."""
        # Test JSONL format
        jsonl_data = [
            {"text": "First example"},
            {"text": "Second example"},
            {"prompt": "Third example as prompt"}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for item in jsonl_data:
                f.write(json.dumps(item) + '\n')
            jsonl_path = Path(f.name)
        
        try:
            # Mock trainer for testing dataset loading
            privacy_config = PrivacyConfig()
            trainer = PrivateTrainer("gpt2", privacy_config)
            
            # Mock tokenizer
            trainer.tokenizer = Mock()
            trainer.tokenizer.return_value = {
                "input_ids": [1, 2, 3],
                "attention_mask": [1, 1, 1]
            }
            
            # Load dataset
            dataset = trainer._load_dataset(str(jsonl_path))
            
            assert dataset is not None
            assert len(dataset) == 3
            
        finally:
            jsonl_path.unlink()
    
    def test_data_collation_edge_cases(self):
        """Test data collation with edge cases."""
        privacy_config = PrivacyConfig()
        trainer = PrivateTrainer("gpt2", privacy_config)
        trainer.tokenizer = Mock()
        trainer.tokenizer.pad_token_id = 0
        
        # Test empty batch
        empty_batch = []
        with pytest.raises((IndexError, ValueError)):
            trainer._data_collator(empty_batch)
        
        # Test batch with varying lengths
        varied_batch = [
            {"input_ids": [1], "attention_mask": [1]},
            {"input_ids": [2, 3, 4, 5], "attention_mask": [1, 1, 1, 1]},
            {"input_ids": [6, 7], "attention_mask": [1, 1]}
        ]
        
        # Mock torch.tensor
        with patch('privacy_finetuner.core.trainer.torch') as mock_torch:
            mock_torch.tensor = lambda x: x
            
            result = trainer._data_collator(varied_batch)
            
            # All sequences should be padded to same length (4)
            assert len(result["input_ids"][0]) == 4
            assert len(result["input_ids"][1]) == 4
            assert len(result["input_ids"][2]) == 4


class TestPrivacyGuarantees:
    """Test privacy guarantee integration."""
    
    def test_differential_privacy_bounds(self):
        """Test differential privacy bound calculations."""
        config = PrivacyConfig(
            epsilon=1.0,
            delta=1e-5,
            noise_multiplier=1.0,
            accounting_mode="rdp"
        )
        
        # Test privacy cost estimation for different scenarios
        low_steps_cost = config.estimate_privacy_cost(steps=100, sample_rate=0.001)
        high_steps_cost = config.estimate_privacy_cost(steps=1000, sample_rate=0.001)
        
        # More steps should cost more privacy
        assert high_steps_cost > low_steps_cost
        
        # Test with different sample rates
        low_rate_cost = config.estimate_privacy_cost(steps=500, sample_rate=0.001)
        high_rate_cost = config.estimate_privacy_cost(steps=500, sample_rate=0.01)
        
        # Higher sample rate should cost more privacy
        assert high_rate_cost > low_rate_cost
    
    def test_privacy_accounting_modes(self):
        """Test different privacy accounting modes."""
        rdp_config = PrivacyConfig(accounting_mode="rdp")
        gdp_config = PrivacyConfig(accounting_mode="gdp")
        
        # Both should produce valid estimates
        rdp_cost = rdp_config.estimate_privacy_cost(steps=1000, sample_rate=0.01)
        gdp_cost = gdp_config.estimate_privacy_cost(steps=1000, sample_rate=0.01)
        
        assert rdp_cost > 0
        assert gdp_cost > 0
        
        # The costs may differ between accounting modes
        # (exact relationship depends on parameters)
        assert isinstance(rdp_cost, float)
        assert isinstance(gdp_cost, float)
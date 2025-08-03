"""Unit tests for PrivateTrainer."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json
from pathlib import Path

from privacy_finetuner.core.trainer import PrivateTrainer
from privacy_finetuner.core.privacy_config import PrivacyConfig


class TestPrivateTrainer:
    """Test PrivateTrainer functionality."""
    
    def test_trainer_initialization(self, privacy_config):
        """Test trainer initialization."""
        trainer = PrivateTrainer(
            model_name="microsoft/DialoGPT-small",
            privacy_config=privacy_config
        )
        
        assert trainer.model_name == "microsoft/DialoGPT-small"
        assert trainer.privacy_config == privacy_config
        assert trainer.use_mcp_gateway is True
        assert trainer._model is None
        assert trainer._privacy_accountant is None
    
    def test_trainer_with_custom_settings(self, privacy_config):
        """Test trainer with custom settings."""
        trainer = PrivateTrainer(
            model_name="gpt2",
            privacy_config=privacy_config,
            use_mcp_gateway=False
        )
        
        assert trainer.model_name == "gpt2"
        assert trainer.use_mcp_gateway is False
    
    @patch('privacy_finetuner.core.trainer.AutoModelForCausalLM')
    @patch('privacy_finetuner.core.trainer.AutoTokenizer')
    def test_setup_model_and_privacy(self, mock_tokenizer, mock_model, privacy_config):
        """Test model and privacy setup."""
        # Mock tokenizer
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token = None
        mock_tokenizer_instance.eos_token = "<eos>"
        mock_tokenizer.from_pretrained.return_value = mock_tokenizer_instance
        
        # Mock model
        mock_model_instance = Mock()
        mock_model.from_pretrained.return_value = mock_model_instance
        
        trainer = PrivateTrainer(
            model_name="gpt2",
            privacy_config=privacy_config
        )
        
        trainer._setup_model_and_privacy()
        
        # Verify model loading
        mock_tokenizer.from_pretrained.assert_called_once_with("gpt2")
        mock_model.from_pretrained.assert_called_once()
        
        # Verify padding token was set
        assert trainer.tokenizer.pad_token == "<eos>"
        assert trainer._model == mock_model_instance
    
    def test_load_dataset(self, privacy_config):
        """Test dataset loading."""
        trainer = PrivateTrainer(
            model_name="gpt2",
            privacy_config=privacy_config
        )
        
        # Create test dataset
        test_data = [
            {"text": "This is test data 1"},
            {"text": "This is test data 2"},
            {"prompt": "This is a prompt"}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')
            dataset_path = f.name
        
        try:
            # Mock tokenizer
            mock_tokenizer = Mock()
            mock_tokenizer.return_value = {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}
            trainer.tokenizer = mock_tokenizer
            
            dataset = trainer._load_dataset(dataset_path)
            
            assert dataset is not None
            assert len(dataset) == 3
            
        finally:
            Path(dataset_path).unlink()
    
    @patch('privacy_finetuner.core.trainer.torch')
    def test_data_collator(self, mock_torch, privacy_config):
        """Test data collator functionality."""
        trainer = PrivateTrainer(
            model_name="gpt2",
            privacy_config=privacy_config
        )
        
        # Mock tokenizer
        trainer.tokenizer = Mock()
        trainer.tokenizer.pad_token_id = 0
        
        # Test batch data
        batch = [
            {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]},
            {"input_ids": [4, 5], "attention_mask": [1, 1]}
        ]
        
        # Mock torch.tensor
        mock_torch.tensor.side_effect = lambda x: x
        
        result = trainer._data_collator(batch)
        
        assert "input_ids" in result
        assert "attention_mask" in result
        assert len(result["input_ids"]) == 2  # Two items in batch
        assert len(result["input_ids"][0]) == 3  # Padded to max length
        assert len(result["input_ids"][1]) == 3  # Padded to max length
    
    @patch('privacy_finetuner.core.trainer.torch')
    @patch('privacy_finetuner.core.trainer.DataLoader')
    def test_train_with_dp_sgd_basic(self, mock_dataloader, mock_torch, privacy_config):
        """Test basic DP-SGD training functionality."""
        trainer = PrivateTrainer(
            model_name="gpt2",
            privacy_config=privacy_config
        )
        
        # Setup mocks
        mock_dataset = Mock()
        mock_model = Mock()
        trainer._model = mock_model
        
        # Mock training loop components
        mock_optimizer = Mock()
        mock_torch.optim.AdamW.return_value = mock_optimizer
        
        mock_dataloader_instance = Mock()
        mock_dataloader_instance.__iter__.return_value = [
            {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}
        ]
        mock_dataloader.return_value = mock_dataloader_instance
        
        # Mock model outputs
        mock_outputs = Mock()
        mock_outputs.loss.item.return_value = 1.5
        mock_model.return_value = mock_outputs
        
        # Mock loss backward
        mock_outputs.loss.backward = Mock()
        
        # Mock save_pretrained
        mock_model.save_pretrained = Mock()
        trainer.tokenizer = Mock()
        trainer.tokenizer.save_pretrained = Mock()
        
        result = trainer._train_with_dp_sgd(
            dataset=mock_dataset,
            epochs=1,
            batch_size=2,
            learning_rate=1e-5
        )
        
        assert result["status"] == "training_complete"
        assert result["epochs_completed"] == 1
        assert "total_steps" in result
        assert "model_path" in result
    
    def test_get_privacy_spent_no_accountant(self, privacy_config):
        """Test privacy spent calculation without accountant."""
        trainer = PrivateTrainer(
            model_name="gpt2",
            privacy_config=privacy_config
        )
        
        assert trainer._get_privacy_spent() == 0.0
    
    def test_get_privacy_report(self, privacy_config):
        """Test privacy report generation."""
        trainer = PrivateTrainer(
            model_name="gpt2",
            privacy_config=privacy_config
        )
        
        report = trainer.get_privacy_report()
        
        assert "epsilon_spent" in report
        assert "delta" in report
        assert "remaining_budget" in report
        assert "accounting_mode" in report
        assert report["delta"] == privacy_config.delta
        assert report["accounting_mode"] == privacy_config.accounting_mode
    
    @patch('privacy_finetuner.core.trainer.logger')
    def test_train_error_handling(self, mock_logger, privacy_config):
        """Test error handling in training."""
        trainer = PrivateTrainer(
            model_name="gpt2",
            privacy_config=privacy_config
        )
        
        # Mock setup to raise an error
        with patch.object(trainer, '_setup_model_and_privacy', side_effect=Exception("Test error")):
            with pytest.raises(Exception, match="Test error"):
                trainer.train("dummy_dataset.jsonl")
            
            # Verify error was logged
            mock_logger.error.assert_called()
    
    @patch('privacy_finetuner.core.trainer.AutoModelForCausalLM')
    @patch('privacy_finetuner.core.trainer.AutoTokenizer')
    def test_opacus_integration_available(self, mock_tokenizer, mock_model, privacy_config):
        """Test Opacus integration when available."""
        with patch('privacy_finetuner.core.trainer.RDPAccountant') as mock_accountant:
            mock_accountant_instance = Mock()
            mock_accountant.return_value = mock_accountant_instance
            
            trainer = PrivateTrainer(
                model_name="gpt2",
                privacy_config=privacy_config
            )
            
            trainer._setup_model_and_privacy()
            
            assert trainer._privacy_accountant == mock_accountant_instance
    
    @patch('privacy_finetuner.core.trainer.AutoModelForCausalLM')
    @patch('privacy_finetuner.core.trainer.AutoTokenizer')
    def test_opacus_integration_unavailable(self, mock_tokenizer, mock_model, privacy_config):
        """Test graceful fallback when Opacus is unavailable."""
        with patch('privacy_finetuner.core.trainer.RDPAccountant', side_effect=ImportError("Opacus not available")):
            trainer = PrivateTrainer(
                model_name="gpt2",
                privacy_config=privacy_config
            )
            
            trainer._setup_model_and_privacy()
            
            assert trainer._privacy_accountant is None
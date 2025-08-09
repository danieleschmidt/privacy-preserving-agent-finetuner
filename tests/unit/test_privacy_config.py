"""Unit tests for privacy configuration."""

import pytest
from pathlib import Path
import tempfile
import yaml

from privacy_finetuner.core.privacy_config import PrivacyConfig


class TestPrivacyConfig:
    """Test privacy configuration management."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = PrivacyConfig()
        
        assert config.epsilon == 1.0
        assert config.delta == 1e-5
        assert config.max_grad_norm == 1.0
        assert config.noise_multiplier == 0.5
        assert config.accounting_mode == "rdp"
        assert not config.federated_enabled
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = PrivacyConfig(
            epsilon=3.0,
            delta=1e-4,
            max_grad_norm=2.0,
            noise_multiplier=1.0
        )
        
        assert config.epsilon == 3.0
        assert config.delta == 1e-4
        assert config.max_grad_norm == 2.0
        assert config.noise_multiplier == 1.0
    
    def test_validation_valid_config(self):
        """Test validation with valid parameters."""
        config = PrivacyConfig(epsilon=1.0, delta=1e-5)
        config.validate()  # Should not raise
    
    def test_validation_invalid_epsilon(self):
        """Test validation with invalid epsilon."""
        config = PrivacyConfig(epsilon=-1.0)
        
        with pytest.raises(ValueError, match="Epsilon must be positive"):
            config.validate()
    
    def test_validation_invalid_delta(self):
        """Test validation with invalid delta."""
        config = PrivacyConfig(delta=0.0)
        
        with pytest.raises(ValueError, match="Privacy configuration validation failed"):
            config.validate()
        
        config = PrivacyConfig(delta=1.5)
        with pytest.raises(ValueError, match="Privacy configuration validation failed"):
            config.validate()
    
    def test_validation_invalid_grad_norm(self):
        """Test validation with invalid gradient norm."""
        config = PrivacyConfig(max_grad_norm=-1.0)
        
        with pytest.raises(ValueError, match="Max gradient norm must be positive"):
            config.validate()
    
    def test_from_yaml(self):
        """Test loading configuration from YAML file."""
        config_data = {
            'privacy': {
                'epsilon': 2.0,
                'delta': 1e-4,
                'max_grad_norm': 1.5,
                'noise_multiplier': 0.8,
                'accounting_mode': 'gdp',
                'federated_enabled': True
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = Path(f.name)
        
        try:
            config = PrivacyConfig.from_yaml(config_path)
            
            assert config.epsilon == 2.0
            assert config.delta == 1e-4
            assert config.max_grad_norm == 1.5
            assert config.noise_multiplier == 0.8
            assert config.accounting_mode == 'gdp'
            assert config.federated_enabled is True
        finally:
            config_path.unlink()
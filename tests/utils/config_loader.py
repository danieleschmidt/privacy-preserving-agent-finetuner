"""Test configuration loader utilities."""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional


def load_test_config(config_path: str, section: Optional[str] = None) -> Dict[str, Any]:
    """Load test configuration from YAML file.
    
    Args:
        config_path: Path to config within test_configs.yaml (e.g., "privacy_configs.moderate")
        section: Optional specific section to return
        
    Returns:
        Configuration dictionary
        
    Example:
        >>> config = load_test_config("privacy_configs.moderate")
        >>> epsilon = config["epsilon"]
    """
    config_file = Path(__file__).parent.parent / "config" / "test_configs.yaml"
    
    if not config_file.exists():
        raise FileNotFoundError(f"Test config file not found: {config_file}")
    
    with open(config_file, "r") as f:
        all_configs = yaml.safe_load(f)
    
    # Navigate nested config path (e.g., "privacy_configs.moderate")
    config = all_configs
    for key in config_path.split("."):
        if key not in config:
            raise KeyError(f"Config path '{config_path}' not found in test configs")
        config = config[key]
    
    if section:
        if section not in config:
            raise KeyError(f"Section '{section}' not found in config '{config_path}'")
        return config[section]
    
    return config


def get_privacy_config(profile: str = "moderate") -> Dict[str, Any]:
    """Get privacy configuration for testing.
    
    Args:
        profile: Privacy profile (minimal, moderate, strict)
        
    Returns:
        Privacy configuration dictionary
    """
    return load_test_config(f"privacy_configs.{profile}")


def get_model_config(size: str = "tiny") -> Dict[str, Any]:
    """Get model configuration for testing.
    
    Args:
        size: Model size (tiny, small, medium)
        
    Returns:
        Model configuration dictionary
    """
    return load_test_config(f"model_configs.{size}")


def get_federated_config(setup: str = "local_test") -> Dict[str, Any]:
    """Get federated learning configuration for testing.
    
    Args:
        setup: Federated setup (local_test, distributed_test)
        
    Returns:
        Federated configuration dictionary
    """
    return load_test_config(f"federated_configs.{setup}")


def get_compliance_config(regulation: str = "gdpr_test") -> Dict[str, Any]:
    """Get compliance configuration for testing.
    
    Args:
        regulation: Regulation type (gdpr_test, hipaa_test, ccpa_test)
        
    Returns:
        Compliance configuration dictionary
    """
    return load_test_config(f"compliance_configs.{regulation}")


def get_performance_config(test_type: str = "speed_test") -> Dict[str, Any]:
    """Get performance testing configuration.
    
    Args:
        test_type: Performance test type (speed_test, scalability_test, accuracy_test)
        
    Returns:
        Performance configuration dictionary
    """
    return load_test_config(f"performance_configs.{test_type}")


def get_hardware_config(setup: str = "cpu_only") -> Dict[str, Any]:
    """Get hardware configuration for testing.
    
    Args:
        setup: Hardware setup (cpu_only, single_gpu, multi_gpu)
        
    Returns:
        Hardware configuration dictionary
    """
    return load_test_config(f"hardware_configs.{setup}")


def list_available_configs() -> Dict[str, list]:
    """List all available test configurations.
    
    Returns:
        Dictionary mapping config categories to available profiles
    """
    config_file = Path(__file__).parent.parent / "config" / "test_configs.yaml"
    
    with open(config_file, "r") as f:
        all_configs = yaml.safe_load(f)
    
    return {
        category: list(configs.keys())
        for category, configs in all_configs.items()
        if isinstance(configs, dict)
    }
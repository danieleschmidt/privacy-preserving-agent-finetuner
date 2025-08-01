"""Test utilities and helper functions for privacy-preserving ML testing."""

import torch
import numpy as np
import pytest
from typing import Dict, List, Any, Tuple, Optional
from unittest.mock import Mock
import tempfile
import os
import json


def create_mock_model(input_size: int = 10, hidden_size: int = 64, num_classes: int = 2):
    """Create a simple mock model for testing."""
    import torch.nn as nn
    
    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(input_size, hidden_size)
            self.relu = nn.ReLU()
            self.linear2 = nn.Linear(hidden_size, num_classes)
            
        def forward(self, x):
            x = self.linear1(x)
            x = self.relu(x)
            x = self.linear2(x)
            return x
    
    return MockModel()


def generate_synthetic_gradients(
    shape: Tuple[int, ...], 
    norm_range: Tuple[float, float] = (0.1, 3.0),
    num_samples: int = 32
) -> torch.Tensor:
    """Generate synthetic gradients with specified norm range."""
    gradients = []
    
    for _ in range(num_samples):
        # Generate random gradient
        grad = torch.randn(shape)
        
        # Scale to desired norm range
        current_norm = torch.norm(grad)
        target_norm = np.random.uniform(*norm_range)
        scaled_grad = grad * (target_norm / current_norm)
        
        gradients.append(scaled_grad)
    
    return torch.stack(gradients)


def assert_privacy_bounds(
    epsilon_consumed: float,
    delta_consumed: float,
    epsilon_budget: float,
    delta_budget: float,
    tolerance: float = 1e-6
):
    """Assert that privacy bounds are respected."""
    assert epsilon_consumed <= epsilon_budget + tolerance, \
        f"Epsilon budget exceeded: {epsilon_consumed} > {epsilon_budget}"
    
    assert delta_consumed <= delta_budget + tolerance, \
        f"Delta budget exceeded: {delta_consumed} > {delta_budget}"
    
    assert epsilon_consumed >= 0, "Epsilon consumed cannot be negative"
    assert delta_consumed >= 0, "Delta consumed cannot be negative"


def assert_gradient_clipping(
    original_gradients: torch.Tensor,
    clipped_gradients: torch.Tensor,
    max_norm: float,
    tolerance: float = 1e-6
):
    """Assert that gradient clipping was applied correctly."""
    assert original_gradients.shape == clipped_gradients.shape, \
        "Gradient shapes should match after clipping"
    
    # Check that all clipped gradients respect the norm constraint
    for i in range(clipped_gradients.size(0)):
        clipped_norm = torch.norm(clipped_gradients[i])
        assert clipped_norm <= max_norm + tolerance, \
            f"Clipped gradient norm {clipped_norm} exceeds max {max_norm}"
        
        # If original was within bounds, should be unchanged
        original_norm = torch.norm(original_gradients[i])
        if original_norm <= max_norm:
            assert torch.allclose(original_gradients[i], clipped_gradients[i], atol=tolerance), \
                "Gradient within bounds should not be modified"


def assert_noise_properties(
    noise: torch.Tensor,
    expected_std: float,
    tolerance: float = 0.1
):
    """Assert that noise has expected statistical properties."""
    actual_mean = torch.mean(noise).item()
    actual_std = torch.std(noise).item()
    
    # Mean should be close to zero
    assert abs(actual_mean) < tolerance, \
        f"Noise mean {actual_mean} not close to zero"
    
    # Standard deviation should match expected
    assert abs(actual_std - expected_std) < tolerance, \
        f"Noise std {actual_std} doesn't match expected {expected_std}"


def create_privacy_test_config(
    epsilon: float = 1.0,
    delta: float = 1e-5,
    noise_multiplier: float = 0.5,
    max_grad_norm: float = 1.0,
    **kwargs
) -> Dict[str, Any]:
    """Create a standard privacy configuration for testing."""
    config = {
        'epsilon': epsilon,
        'delta': delta,
        'noise_multiplier': noise_multiplier,
        'max_grad_norm': max_grad_norm,
        'accounting_mode': 'rdp',
        'target_delta': delta,
        **kwargs
    }
    return config


def simulate_training_step(
    model: torch.nn.Module,
    batch: torch.Tensor,
    labels: torch.Tensor,
    privacy_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Simulate a single training step with privacy."""
    # Forward pass
    outputs = model(batch)
    loss = torch.nn.functional.cross_entropy(outputs, labels)
    
    # Backward pass
    loss.backward()
    
    # Get gradients
    gradients = []
    for param in model.parameters():
        if param.grad is not None:
            gradients.append(param.grad.clone())
    
    # Apply gradient clipping
    max_norm = privacy_config['max_grad_norm']
    clipped_gradients = []
    
    for grad in gradients:
        grad_norm = torch.norm(grad)
        if grad_norm > max_norm:
            clipped_grad = grad * (max_norm / grad_norm)
        else:
            clipped_grad = grad
        clipped_gradients.append(clipped_grad)
    
    # Add noise
    noise_multiplier = privacy_config['noise_multiplier']
    noisy_gradients = []
    
    for grad in clipped_gradients:
        noise = torch.normal(0, noise_multiplier, grad.shape)
        noisy_grad = grad + noise
        noisy_gradients.append(noisy_grad)
    
    # Clear gradients
    model.zero_grad()
    
    return {
        'loss': loss.item(),
        'original_gradients': gradients,
        'clipped_gradients': clipped_gradients,
        'noisy_gradients': noisy_gradients,
        'privacy_consumed': {
            'epsilon': privacy_config.get('step_epsilon', 0.01),
            'delta': privacy_config.get('step_delta', 1e-7)
        }
    }


def create_temporary_model_file(model_state: Dict[str, Any]) -> str:
    """Create a temporary file with model state for testing."""
    temp_file = tempfile.NamedTemporaryFile(suffix='.pt', delete=False)
    torch.save(model_state, temp_file.name)
    temp_file.close()
    return temp_file.name


def cleanup_temporary_files(file_paths: List[str]):
    """Clean up temporary files created during testing."""
    for file_path in file_paths:
        try:
            os.remove(file_path)
        except FileNotFoundError:
            pass  # File already deleted


def assert_model_privacy_metadata(model_state: Dict[str, Any]):
    """Assert that model state contains required privacy metadata."""
    required_fields = [
        'epsilon_consumed',
        'delta_consumed', 
        'noise_multiplier',
        'max_grad_norm',
        'training_steps'
    ]
    
    assert 'privacy_metadata' in model_state, "Model state missing privacy metadata"
    
    metadata = model_state['privacy_metadata']
    for field in required_fields:
        assert field in metadata, f"Privacy metadata missing field: {field}"
    
    # Validate field types and ranges
    assert isinstance(metadata['epsilon_consumed'], (int, float)), "Epsilon should be numeric"
    assert isinstance(metadata['delta_consumed'], (int, float)), "Delta should be numeric"
    assert metadata['epsilon_consumed'] >= 0, "Epsilon consumed should be non-negative"
    assert metadata['delta_consumed'] >= 0, "Delta consumed should be non-negative"


def generate_pii_test_samples() -> List[Dict[str, Any]]:
    """Generate test samples containing various types of PII."""
    return [
        {
            'text': "My name is John Smith and my SSN is 123-45-6789",
            'expected_pii': ['John Smith', '123-45-6789'],
            'pii_types': ['PERSON', 'SSN']
        },
        {
            'text': "Contact me at john.doe@example.com or call (555) 123-4567",
            'expected_pii': ['john.doe@example.com', '(555) 123-4567'],
            'pii_types': ['EMAIL', 'PHONE']
        },
        {
            'text': "My address is 123 Main Street, Anytown, CA 90210",
            'expected_pii': ['123 Main Street, Anytown, CA 90210'],
            'pii_types': ['ADDRESS']
        },
        {
            'text': "Credit card number: 4111-1111-1111-1111, exp: 12/25",
            'expected_pii': ['4111-1111-1111-1111', '12/25'],
            'pii_types': ['CREDIT_CARD', 'DATE']
        }
    ]


def assert_pii_protection(
    original_text: str,
    protected_text: str,
    expected_pii: List[str],
    protection_method: str = 'removal'
):
    """Assert that PII protection was applied correctly."""
    assert len(protected_text) > 0, "Protected text should not be empty"
    
    if protection_method == 'removal':
        for pii in expected_pii:
            assert pii not in protected_text, f"PII '{pii}' not removed from text"
    
    elif protection_method == 'masking':
        for pii in expected_pii:
            assert pii not in protected_text, f"PII '{pii}' not masked in text"
            # Should contain mask tokens
            assert '[REDACTED]' in protected_text or '[MASK]' in protected_text
    
    elif protection_method == 'hashing':
        for pii in expected_pii:
            assert pii not in protected_text, f"PII '{pii}' not hashed in text"
            # Should contain hash-like strings
            import re
            hash_pattern = r'[a-f0-9]{8,}'
            assert re.search(hash_pattern, protected_text), "No hash patterns found"


def create_mock_federated_clients(
    num_clients: int = 5,
    samples_per_client: int = 100,
    features: int = 10
) -> List[Dict[str, Any]]:
    """Create mock federated learning clients."""
    clients = []
    
    for client_id in range(num_clients):
        # Create non-IID data distribution
        np.random.seed(client_id * 42)
        torch.manual_seed(client_id * 42)
        
        # Add bias to simulate non-IID data
        bias = client_id * 0.3
        data = torch.randn(samples_per_client, features) + bias
        labels = torch.randint(0, 2, (samples_per_client,))
        
        client = {
            'client_id': client_id,
            'data': data,
            'labels': labels,
            'privacy_budget': {'epsilon': 0.0, 'delta': 0.0},
            'participation_history': []
        }
        clients.append(client)
    
    return clients


def assert_federated_privacy_bounds(
    clients: List[Dict[str, Any]],
    max_epsilon: float,
    max_delta: float
):
    """Assert that all federated clients respect privacy bounds."""
    for client in clients:
        budget = client['privacy_budget']
        assert budget['epsilon'] <= max_epsilon, \
            f"Client {client['client_id']} epsilon {budget['epsilon']} exceeds {max_epsilon}"
        assert budget['delta'] <= max_delta, \
            f"Client {client['client_id']} delta {budget['delta']} exceeds {max_delta}"


def benchmark_privacy_overhead(
    baseline_func,
    private_func,
    *args,
    **kwargs
) -> Dict[str, float]:
    """Benchmark the overhead of privacy mechanisms."""
    import time
    
    # Benchmark baseline
    start_time = time.time()
    baseline_result = baseline_func(*args, **kwargs)
    baseline_time = time.time() - start_time
    
    # Benchmark private version
    start_time = time.time()
    private_result = private_func(*args, **kwargs)
    private_time = time.time() - start_time
    
    # Calculate overhead
    overhead_ratio = private_time / baseline_time if baseline_time > 0 else 1
    overhead_percentage = (overhead_ratio - 1) * 100
    
    return {
        'baseline_time': baseline_time,
        'private_time': private_time,
        'overhead_ratio': overhead_ratio,
        'overhead_percentage': overhead_percentage,
        'baseline_result': baseline_result,
        'private_result': private_result
    }


class PrivacyTestContext:
    """Context manager for privacy testing with automatic cleanup."""
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        self.epsilon = epsilon
        self.delta = delta
        self.consumed_epsilon = 0.0
        self.consumed_delta = 0.0
        self.temp_files = []
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cleanup temporary files
        cleanup_temporary_files(self.temp_files)
        
        # Reset random seeds
        torch.manual_seed(42)
        np.random.seed(42)
    
    def consume_privacy_budget(self, epsilon: float, delta: float):
        """Consume privacy budget and check bounds."""
        self.consumed_epsilon += epsilon
        self.consumed_delta += delta
        
        assert_privacy_bounds(
            self.consumed_epsilon,
            self.consumed_delta,
            self.epsilon,
            self.delta
        )
    
    def add_temp_file(self, file_path: str):
        """Add a temporary file for cleanup."""
        self.temp_files.append(file_path)
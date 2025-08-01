"""Test data fixtures for privacy-preserving training tests."""

import pytest
import torch
import numpy as np
from typing import List, Dict, Any
from torch.utils.data import Dataset, DataLoader


class MockPrivateDataset(Dataset):
    """Mock dataset with privacy-sensitive data for testing."""
    
    def __init__(self, size: int = 100, features: int = 10, num_classes: int = 2):
        self.size = size
        self.features = features
        self.num_classes = num_classes
        
        # Generate synthetic sensitive data
        self.data = torch.randn(size, features)
        self.labels = torch.randint(0, num_classes, (size,))
        
        # Add some "PII-like" patterns for context protection testing
        self.contexts = [
            f"User {i} with ID {1000 + i} has sensitive data",
            f"Patient record #{i}: diagnosis confidential",
            f"Financial record for account {5000 + i}",
        ]
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.data[idx],
            'labels': self.labels[idx],
            'context': self.contexts[idx % len(self.contexts)]
        }


class MockFederatedDataset(Dataset):
    """Mock federated dataset for testing distributed training."""
    
    def __init__(self, client_id: int, size: int = 50, features: int = 10):
        self.client_id = client_id
        self.size = size
        self.features = features
        
        # Generate client-specific data (simulating non-IID distribution)
        seed = client_id * 42
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Add bias based on client_id to simulate non-IID data
        bias = client_id * 0.5
        self.data = torch.randn(size, features) + bias
        self.labels = torch.randint(0, 2, (size,))
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.data[idx],
            'labels': self.labels[idx],
            'client_id': self.client_id
        }


@pytest.fixture
def mock_private_dataset():
    """Basic mock dataset with sensitive data."""
    return MockPrivateDataset(size=100, features=10, num_classes=2)


@pytest.fixture
def small_private_dataset():
    """Small dataset for quick tests."""
    return MockPrivateDataset(size=20, features=5, num_classes=2)


@pytest.fixture
def large_private_dataset():
    """Large dataset for performance tests."""
    return MockPrivateDataset(size=1000, features=50, num_classes=10)


@pytest.fixture
def federated_datasets():
    """Multiple federated datasets for distributed testing."""
    return [
        MockFederatedDataset(client_id=i, size=50, features=10)
        for i in range(5)
    ]


@pytest.fixture
def privacy_dataloader(mock_private_dataset):
    """DataLoader with privacy-aware batching."""
    return DataLoader(
        mock_private_dataset,
        batch_size=8,
        shuffle=True,
        drop_last=True,  # Important for DP-SGD
        num_workers=0    # Avoid multiprocessing in tests
    )


@pytest.fixture
def sensitive_text_samples():
    """Sample texts with PII for context protection testing."""
    return [
        "John Doe with SSN 123-45-6789 applied for loan",
        "Patient Sarah Smith, DOB 1990-05-15, diagnosed with condition X",
        "Credit card 4111-1111-1111-1111 charged $500.00",
        "Email john.doe@example.com sent sensitive document",
        "Phone number (555) 123-4567 called about account",
        "Address: 123 Main St, Anytown, CA 90210",
        "IP address 192.168.1.100 accessed secure system",
        "Medical record #MR123456 shows treatment history",
    ]


@pytest.fixture
def gdpr_test_data():
    """Test data for GDPR compliance validation."""
    return {
        "personal_data": {
            "name": "Maria García",
            "email": "maria.garcia@example.eu",
            "phone": "+34-123-456-789",
            "address": "Calle Mayor 1, Madrid, Spain",
        },
        "processing_purpose": "ML model training",
        "legal_basis": "consent",
        "retention_period": "2 years",
        "data_subject_rights": [
            "access", "rectification", "erasure", 
            "portability", "restriction", "objection"
        ]
    }


@pytest.fixture
def hipaa_test_data():
    """Test data for HIPAA compliance validation."""
    return {
        "phi_elements": [
            "Patient Name: Robert Johnson",
            "DOB: 1985-03-22",
            "SSN: 987-65-4321", 
            "Medical Record Number: 12345",
            "Account Number: ACC-789",
            "Certificate/License Number: LIC-456",
            "Device Identifiers: DEV-123",
            "Biometric Identifiers: BIO-789",
            "Full Face Photos: IMG-001.jpg",
            "Web URLs: https://hospital.com/patient/12345",
            "IP Addresses: 10.0.0.1",
            "Fingerprints: FP-456789",
            "Voice Prints: VP-123456"
        ],
        "covered_entities": ["Healthcare Provider", "Health Plan", "Healthcare Clearinghouse"],
        "minimum_necessary": True,
        "breach_notification_required": True
    }


@pytest.fixture
def mock_model_outputs():
    """Mock model outputs for privacy analysis."""
    return {
        "logits": torch.randn(32, 10),  # Batch size 32, 10 classes
        "hidden_states": torch.randn(32, 128),  # Hidden representations
        "attention_weights": torch.randn(32, 8, 64, 64),  # Attention patterns
        "gradients": torch.randn(1000),  # Flattened gradients
        "privacy_budget_consumed": 0.1,
        "noise_added": torch.randn(1000),
    }


@pytest.fixture
def privacy_attack_data():
    """Data for testing privacy attacks."""
    return {
        "membership_inference": {
            "train_data": torch.randn(100, 10),
            "test_data": torch.randn(100, 10),
            "train_labels": torch.randint(0, 2, (100,)),
            "test_labels": torch.randint(0, 2, (100,)),
        },
        "model_inversion": {
            "target_features": torch.randn(50, 10),
            "auxiliary_data": torch.randn(200, 10),
        },
        "attribute_inference": {
            "public_attributes": torch.randn(100, 5),
            "private_attributes": torch.randn(100, 3),
        }
    }
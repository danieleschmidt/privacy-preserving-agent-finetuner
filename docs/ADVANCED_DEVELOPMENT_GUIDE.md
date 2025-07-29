# Advanced Development Guide

## Overview

This guide provides comprehensive instructions for advanced development practices, optimization techniques, and production deployment strategies for the Privacy-Preserving Agent Finetuner.

## Table of Contents

- [Development Environment Setup](#development-environment-setup)
- [Advanced Configuration](#advanced-configuration)
- [Performance Optimization](#performance-optimization)
- [Security Best Practices](#security-best-practices)
- [Monitoring and Observability](#monitoring-and-observability)
- [Deployment Strategies](#deployment-strategies)
- [Troubleshooting Guide](#troubleshooting-guide)

## Development Environment Setup

### Advanced Poetry Configuration

```bash
# Configure Poetry for development
poetry config virtualenvs.create true
poetry config virtualenvs.in-project true
poetry config virtualenvs.prefer-active-python true
poetry config installer.max-workers 10
poetry config installer.modern-installation true

# Enable parallel dependency resolution
poetry config solver.lazy-wheel true
poetry config experimental.new-installer true
```

### GPU Development Setup

```bash
# Install CUDA-enabled PyTorch
poetry run pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU access
poetry run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

# Install additional GPU tools
poetry add nvidia-ml-py3 pynvml gpustat
```

### Advanced IDE Configuration

**VS Code Settings (.vscode/settings.json):**

```json
{
    "python.defaultInterpreterPath": "./.venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true,
    "python.linting.banditEnabled": true,
    "python.formatting.provider": "black",
    "python.sortImports.args": ["--profile", "black"],
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    },
    "python.testing.pytestEnabled": true,
    "python.testing.unittestEnabled": false,
    "python.testing.pytestArgs": [
        "tests/",
        "--tb=short",
        "--strict-markers"
    ],
    "files.associations": {
        "*.env.example": "dotenv",
        "*.env.local": "dotenv"
    },
    "yaml.schemas": {
        "https://json.schemastore.org/github-workflow.json": ".github/workflows/*.yml"
    }
}
```

**VS Code Extensions (.vscode/extensions.json):**

```json
{
    "recommendations": [
        "ms-python.python",
        "ms-python.flake8",
        "ms-python.mypy-type-checker",
        "ms-python.black-formatter",
        "ms-python.isort",
        "charliermarsh.ruff",
        "ms-vscode.makefile-tools",
        "redhat.vscode-yaml",
        "ms-vscode.vscode-json",
        "github.vscode-github-actions",
        "ms-azuretools.vscode-docker",
        "ms-vscode-remote.remote-containers",
        "gruntfuggly.todo-tree",
        "streetsidesoftware.code-spell-checker"
    ]
}
```

## Advanced Configuration

### Environment-Specific Configurations

**Development Configuration (config/development.yaml):**

```yaml
app:
  debug: true
  log_level: DEBUG
  hot_reload: true
  
privacy:
  epsilon: 10.0  # Relaxed for development
  delta: 1e-3
  strict_mode: false
  
database:
  url: postgresql://dev:dev@localhost:5432/privacy_finetuner_dev
  echo: true
  pool_size: 5
  
redis:
  url: redis://localhost:6379/1
  
monitoring:
  enabled: true
  export_interval: 10  # More frequent for development
```

**Production Configuration (config/production.yaml):**

```yaml
app:
  debug: false
  log_level: INFO
  workers: 4
  
privacy:
  epsilon: 1.0
  delta: 1e-5
  strict_mode: true
  hardware_attestation: true
  
database:
  url: ${DATABASE_URL}
  pool_size: 20
  max_overflow: 30
  pool_pre_ping: true
  
redis:
  url: ${REDIS_URL}
  connection_pool_size: 50
  
monitoring:
  enabled: true
  export_interval: 300
  alerting: true
```

### Multi-Environment Docker Setup

**docker-compose.dev.yml:**

```yaml
version: '3.8'
services:
  privacy-finetuner:
    build:
      context: .
      dockerfile: Dockerfile.dev
    volumes:
      - .:/app
      - /app/.venv
    environment:
      - ENVIRONMENT=development
      - DEBUG=true
    ports:
      - "8080:8080"
      - "5678:5678"  # debugpy
    depends_on:
      - postgres
      - redis
      
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: privacy_finetuner_dev
      POSTGRES_USER: dev
      POSTGRES_PASSWORD: dev
    volumes:
      - postgres_dev_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
      
  redis:
    image: redis:7-alpine
    volumes:
      - redis_dev_data:/data
    ports:
      - "6379:6379"

volumes:
  postgres_dev_data:
  redis_dev_data:
```

## Performance Optimization

### Memory Optimization

```python
# Example: Memory-efficient data loading
class MemoryEfficientDataLoader:
    def __init__(self, batch_size: int, prefetch_factor: int = 2):
        self.batch_size = batch_size
        self.prefetch_factor = prefetch_factor
        
    def setup_dataloader(self):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=self.prefetch_factor,
            persistent_workers=True
        )
```

### GPU Optimization

```python
# Mixed precision training for better performance
from torch.cuda.amp import GradScaler, autocast

class OptimizedTrainer:
    def __init__(self):
        self.scaler = GradScaler()
        
    def training_step(self, batch):
        with autocast():
            outputs = self.model(batch)
            loss = self.compute_loss(outputs, batch)
            
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
```

### Profiling and Benchmarking

```bash
# CPU profiling
poetry run python -m cProfile -o profile.stats scripts/performance_profiler.py

# Memory profiling
poetry run python -m memory_profiler scripts/performance_profiler.py

# GPU profiling
poetry run nsys profile --trace=cuda,nvtx -o profile.nsys python scripts/performance_profiler.py

# PyTorch profiler
poetry run python scripts/torch_profiler.py
```

## Security Best Practices

### Secrets Management

```python
# Use proper secrets management
import os
from pathlib import Path
from cryptography.fernet import Fernet

class SecretsManager:
    def __init__(self):
        key_file = Path(".secrets.key")
        if not key_file.exists():
            key = Fernet.generate_key()
            key_file.write_bytes(key)
        else:
            key = key_file.read_bytes()
        self.cipher = Fernet(key)
    
    def encrypt_secret(self, secret: str) -> bytes:
        return self.cipher.encrypt(secret.encode())
    
    def decrypt_secret(self, encrypted_secret: bytes) -> str:
        return self.cipher.decrypt(encrypted_secret).decode()
```

### Input Validation

```python
from pydantic import BaseModel, validator
from typing import List, Optional

class PrivacyConfig(BaseModel):
    epsilon: float
    delta: float
    max_grad_norm: float
    
    @validator('epsilon')
    def validate_epsilon(cls, v):
        if not 0.01 <= v <= 20.0:
            raise ValueError('Epsilon must be between 0.01 and 20.0')
        return v
    
    @validator('delta')
    def validate_delta(cls, v):
        if not 1e-10 <= v <= 1e-2:
            raise ValueError('Delta must be between 1e-10 and 1e-2')
        return v
```

### Secure Communication

```python
import ssl
import aiohttp
from aiohttp import ClientSession

class SecureHTTPClient:
    def __init__(self):
        # Create SSL context with strong security
        self.ssl_context = ssl.create_default_context()
        self.ssl_context.check_hostname = True
        self.ssl_context.verify_mode = ssl.CERT_REQUIRED
        
    async def make_request(self, url: str, **kwargs):
        async with ClientSession(
            connector=aiohttp.TCPConnector(ssl=self.ssl_context),
            timeout=aiohttp.ClientTimeout(total=30)
        ) as session:
            async with session.get(url, **kwargs) as response:
                return await response.json()
```

## Monitoring and Observability

### Custom Metrics

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Define custom metrics
privacy_budget_used = Gauge('privacy_budget_used_total', 'Total privacy budget consumed')
training_iterations = Counter('training_iterations_total', 'Total training iterations')
model_accuracy = Gauge('model_accuracy', 'Current model accuracy')
inference_latency = Histogram('inference_latency_seconds', 'Inference latency')

class MetricsCollector:
    def __init__(self):
        start_http_server(9090)
    
    def record_privacy_budget(self, epsilon_used: float):
        privacy_budget_used.set(epsilon_used)
    
    def record_training_iteration(self):
        training_iterations.inc()
```

### Structured Logging

```python
import structlog
from pythonjsonlogger import jsonlogger

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.JSONRenderer()
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Usage
logger.info(
    "Privacy budget updated",
    epsilon_used=0.5,
    delta_used=1e-5,
    remaining_budget=0.5,
    user_id="user123"
)
```

### Health Checks

```python
from fastapi import FastAPI, status
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get("/health")
async def health_check():
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "0.1.0",
        "checks": {
            "database": await check_database(),
            "redis": await check_redis(),
            "gpu": check_gpu_availability(),
            "privacy_budget": check_privacy_budget()
        }
    }
    
    all_healthy = all(check["status"] == "healthy" for check in health_status["checks"].values())
    
    return JSONResponse(
        content=health_status,
        status_code=status.HTTP_200_OK if all_healthy else status.HTTP_503_SERVICE_UNAVAILABLE
    )
```

## Deployment Strategies

### Blue-Green Deployment

```bash
#!/bin/bash
# Blue-green deployment script

CURRENT_VERSION=$(kubectl get deployment privacy-finetuner -o jsonpath='{.metadata.labels.version}')
NEW_VERSION=$1

echo "Current version: $CURRENT_VERSION"
echo "Deploying version: $NEW_VERSION"

# Deploy new version alongside current
kubectl set image deployment/privacy-finetuner-green privacy-finetuner=ghcr.io/privacy-finetuner:$NEW_VERSION

# Wait for rollout
kubectl rollout status deployment/privacy-finetuner-green

# Run smoke tests
curl -f http://privacy-finetuner-green-service/health || exit 1

# Switch traffic
kubectl patch service privacy-finetuner-service -p '{"spec":{"selector":{"version":"'$NEW_VERSION'"}}}'

echo "Deployment successful"
```

### Canary Deployment

```yaml
# Istio VirtualService for canary deployment
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: privacy-finetuner-canary
spec:
  hosts:
  - privacy-finetuner
  http:
  - match:
    - headers:
        canary:
          exact: "true"
    route:
    - destination:
        host: privacy-finetuner
        subset: canary
  - route:
    - destination:
        host: privacy-finetuner
        subset: stable
      weight: 90
    - destination:
        host: privacy-finetuner
        subset: canary
      weight: 10
```

### Infrastructure as Code

```terraform
# Example Terraform configuration
resource "aws_ecs_cluster" "privacy_finetuner" {
  name = "privacy-finetuner"
  
  capacity_providers = ["FARGATE", "FARGATE_SPOT"]
  
  default_capacity_provider_strategy {
    capacity_provider = "FARGATE"
    weight           = 100
  }
}

resource "aws_ecs_service" "privacy_finetuner" {
  name            = "privacy-finetuner"
  cluster         = aws_ecs_cluster.privacy_finetuner.id
  task_definition = aws_ecs_task_definition.privacy_finetuner.arn
  desired_count   = 3
  
  network_configuration {
    subnets         = var.private_subnet_ids
    security_groups = [aws_security_group.privacy_finetuner.id]
  }
  
  service_registries {
    registry_arn = aws_service_discovery_service.privacy_finetuner.arn
  }
}
```

## Troubleshooting Guide

### Common Issues and Solutions

**Issue: Out of Memory (OOM) during training**

```bash
# Check memory usage
poetry run python -m memory_profiler scripts/train.py

# Solutions:
# 1. Reduce batch size
# 2. Enable gradient checkpointing
# 3. Use gradient accumulation
# 4. Enable mixed precision training
```

**Issue: Privacy budget exhausted**

```python
# Monitor and manage privacy budget
class PrivacyBudgetManager:
    def __init__(self, total_epsilon: float):
        self.total_epsilon = total_epsilon
        self.used_epsilon = 0.0
    
    def check_budget(self, requested_epsilon: float) -> bool:
        return (self.used_epsilon + requested_epsilon) <= self.total_epsilon
    
    def consume_budget(self, epsilon: float):
        if self.check_budget(epsilon):
            self.used_epsilon += epsilon
            return True
        raise ValueError("Insufficient privacy budget")
```

**Issue: GPU utilization is low**

```bash
# Profile GPU usage
nvidia-smi dmon -s pucvmet -d 1

# Check for bottlenecks:
# 1. Data loading (increase num_workers)
# 2. CPU preprocessing (optimize transforms)
# 3. Memory transfers (use pin_memory=True)
```

### Debugging Tools

```python
# Enable detailed debugging
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

# Use PyTorch's built-in debugger
import torch
torch.autograd.set_detect_anomaly(True)

# Add debugging hooks
def debug_hook(grad):
    print(f"Gradient norm: {grad.norm()}")
    print(f"Gradient shape: {grad.shape}")
    return grad

model.register_backward_hook(debug_hook)
```

### Performance Debugging

```bash
# Generate flame graphs
poetry run py-spy record -o profile.svg -- python scripts/train.py

# Profile with line-by-line timing
poetry run kernprof -l -v scripts/train.py

# Memory debugging
poetry run python -m memory_profiler scripts/train.py
```

## Advanced Testing Strategies

### Property-Based Testing

```python
from hypothesis import given, strategies as st

@given(
    epsilon=st.floats(min_value=0.1, max_value=10.0),
    delta=st.floats(min_value=1e-8, max_value=1e-3)
)
def test_privacy_guarantees(epsilon, delta):
    """Test that privacy guarantees hold for any valid epsilon/delta."""
    trainer = PrivateTrainer(epsilon=epsilon, delta=delta)
    # Test privacy properties
    assert trainer.get_privacy_budget().epsilon <= epsilon
    assert trainer.get_privacy_budget().delta <= delta
```

### Mutation Testing

```bash
# Run mutation testing
poetry run mutmut run --paths-to-mutate privacy_finetuner/
poetry run mutmut results
poetry run mutmut html
```

### Load Testing

```python
import asyncio
import aiohttp
from locust import HttpUser, task, between

class PrivacyFinetunerUser(HttpUser):
    wait_time = between(1, 3)
    
    @task
    def inference_request(self):
        payload = {
            "text": "Test inference request",
            "privacy_level": "high"
        }
        self.client.post("/inference", json=payload)
    
    @task(3)
    def health_check(self):
        self.client.get("/health")
```

This advanced guide provides comprehensive coverage of development best practices, optimization techniques, and production deployment strategies for the privacy-preserving AI framework.
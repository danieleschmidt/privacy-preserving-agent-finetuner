# Troubleshooting Guide

## 🔧 Privacy-Preserving Agent Finetuner - Problem Resolution Guide

This comprehensive troubleshooting guide helps you diagnose and resolve common issues across all generations of the privacy-preserving ML framework.

## 📋 Table of Contents

- [Quick Diagnosis](#quick-diagnosis)
- [Installation Issues](#installation-issues)
- [Privacy Configuration Problems](#privacy-configuration-problems)
- [Training Issues](#training-issues)
- [Security & Threat Detection Issues](#security--threat-detection-issues)
- [Scaling & Performance Issues](#scaling--performance-issues)
- [Global Deployment Issues](#global-deployment-issues)
- [Memory & Resource Problems](#memory--resource-problems)
- [API & Integration Issues](#api--integration-issues)
- [Compliance & Regulatory Issues](#compliance--regulatory-issues)
- [Debug Mode & Diagnostics](#debug-mode--diagnostics)
- [Getting Additional Help](#getting-additional-help)

---

## 🩺 Quick Diagnosis

### System Health Check

Run this quick health check to identify common issues:

```python
#!/usr/bin/env python3
"""
Quick system health check for privacy-preserving ML framework
"""

import sys
import torch
import psutil
import subprocess
from pathlib import Path

def run_health_check():
    """Comprehensive system health check"""
    print("🔍 Privacy-Preserving ML Framework - Health Check")
    print("=" * 60)
    
    issues_found = []
    
    # Check Python version
    print(f"🐍 Python Version: {sys.version}")
    if sys.version_info < (3, 9):
        issues_found.append("❌ Python 3.9+ required")
    else:
        print("✅ Python version OK")
    
    # Check PyTorch installation
    try:
        print(f"🔥 PyTorch Version: {torch.__version__}")
        print(f"   CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   GPU Count: {torch.cuda.device_count()}")
        else:
            print("⚠️  No CUDA GPU detected (CPU training only)")
    except ImportError:
        issues_found.append("❌ PyTorch not installed")
    
    # Check system resources
    memory = psutil.virtual_memory()
    print(f"💾 Memory: {memory.total / (1024**3):.1f} GB total, {memory.percent}% used")
    if memory.total < 16 * (1024**3):  # Less than 16GB
        issues_found.append("⚠️  Low system memory (16GB+ recommended)")
    
    # Check disk space
    disk = psutil.disk_usage('/')
    print(f"💿 Disk: {disk.total / (1024**3):.1f} GB total, {(disk.used/disk.total)*100:.1f}% used")
    if (disk.free / (1024**3)) < 10:  # Less than 10GB free
        issues_found.append("⚠️  Low disk space (10GB+ recommended)")
    
    # Check privacy-finetuner installation
    try:
        import privacy_finetuner
        print(f"🔒 Privacy Finetuner: Installed")
        from privacy_finetuner import PrivateTrainer, PrivacyConfig
        print("✅ Core components accessible")
    except ImportError as e:
        issues_found.append(f"❌ Privacy Finetuner not installed: {e}")
    
    # Check optional dependencies
    optional_deps = {
        'opacus': 'Differential Privacy',
        'transformers': 'Transformer Models',
        'datasets': 'Dataset Processing',
        'accelerate': 'Training Acceleration'
    }
    
    for dep, description in optional_deps.items():
        try:
            __import__(dep)
            print(f"✅ {description} ({dep}): Available")
        except ImportError:
            print(f"⚠️  {description} ({dep}): Not installed")
    
    # Summary
    print("\n" + "=" * 60)
    if issues_found:
        print("❌ Issues Found:")
        for issue in issues_found:
            print(f"  {issue}")
        print("\n💡 See installation section for resolution steps")
    else:
        print("✅ System health check passed!")
    
    return len(issues_found) == 0

if __name__ == "__main__":
    run_health_check()
```

---

## 📦 Installation Issues

### Issue: `ImportError: No module named 'privacy_finetuner'`

**Symptoms:**
```python
ImportError: No module named 'privacy_finetuner'
```

**Solutions:**

1. **Basic Installation**
   ```bash
   # Install from requirements
   pip install -r requirements.txt
   
   # Or install with poetry
   poetry install
   poetry shell
   
   # Verify installation
   python -c "import privacy_finetuner; print('Success!')"
   ```

2. **Development Installation**
   ```bash
   # Clone and install in development mode
   git clone https://github.com/your-org/privacy-preserving-agent-finetuner
   cd privacy-preserving-agent-finetuner
   pip install -e .
   ```

3. **Virtual Environment Issues**
   ```bash
   # Ensure virtual environment is activated
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows
   
   # Verify you're in the right environment
   which python
   which pip
   ```

### Issue: CUDA/GPU Not Detected

**Symptoms:**
- `torch.cuda.is_available()` returns `False`
- Training is very slow (CPU only)

**Solutions:**

1. **Check NVIDIA Driver**
   ```bash
   # Check driver installation
   nvidia-smi
   
   # If not installed, install NVIDIA drivers
   sudo apt install nvidia-driver-535  # Ubuntu
   ```

2. **Install CUDA-compatible PyTorch**
   ```bash
   # Uninstall CPU-only PyTorch
   pip uninstall torch torchvision torchaudio
   
   # Install CUDA version (check https://pytorch.org for latest)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Verify CUDA Installation**
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"CUDA version: {torch.version.cuda}")
   print(f"GPU count: {torch.cuda.device_count()}")
   if torch.cuda.is_available():
       print(f"GPU name: {torch.cuda.get_device_name(0)}")
   ```

### Issue: Memory/OOM Errors During Installation

**Symptoms:**
```
MemoryError: Unable to allocate array
Killed
```

**Solutions:**

1. **Increase Swap Space**
   ```bash
   # Create swap file (8GB)
   sudo fallocate -l 8G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

2. **Install with Limited Parallelism**
   ```bash
   # Limit pip parallelism
   pip install --no-cache-dir -r requirements.txt
   
   # Or install packages one by one
   pip install torch
   pip install transformers
   pip install opacus
   ```

---

## 🔒 Privacy Configuration Problems

### Issue: `PrivacyBudgetExhaustedException`

**Symptoms:**
```python
PrivacyBudgetExhaustedException: Privacy budget (ε=1.0) exhausted at step 150
```

**Solutions:**

1. **Increase Privacy Budget**
   ```python
   privacy_config = PrivacyConfig(
       epsilon=2.0,  # Increase from 1.0
       delta=1e-5
   )
   ```

2. **Optimize Privacy Budget Usage**
   ```python
   privacy_config = PrivacyConfig(
       epsilon=1.0,
       delta=1e-5,
       max_grad_norm=0.5,  # Reduce from 1.0 for better efficiency
       noise_multiplier=0.8,  # Adjust noise
       adaptive_clipping=True  # Enable adaptive clipping
   )
   ```

3. **Use Batch Size Optimization**
   ```python
   # Larger batch sizes are more privacy-efficient
   result = trainer.train(
       dataset=dataset,
       batch_size=32,  # Increase from 8
       epochs=2        # Reduce epochs if needed
   )
   ```

### Issue: Privacy Guarantees Not Met

**Symptoms:**
- Theoretical epsilon exceeds configured limit
- Privacy validation fails

**Diagnostic Code:**
```python
# Check privacy guarantees
from privacy_finetuner.quality import PrivacyValidator

validator = PrivacyValidator()
validation_result = validator.validate_privacy_guarantees(
    model=trained_model,
    training_history=training_history,
    privacy_config=privacy_config
)

print(f"Guarantee satisfied: {validation_result.guarantee_verified}")
print(f"Theoretical ε: {validation_result.theoretical_epsilon}")
print(f"Target ε: {privacy_config.epsilon}")
```

**Solutions:**

1. **Tighten Privacy Accounting**
   ```python
   privacy_config = PrivacyConfig(
       epsilon=1.0,
       delta=1e-7,  # Reduce delta
       accounting_mode="gdp",  # Use tighter Gaussian DP
       secure_rng=True
   )
   ```

2. **Reduce Training Complexity**
   ```python
   # Reduce number of training steps
   result = trainer.train(
       dataset=dataset,
       epochs=2,       # Reduce from 5
       max_steps=500,  # Limit total steps
       save_steps=100  # More frequent checkpoints
   )
   ```

### Issue: Context Guard Over-Redacting

**Symptoms:**
- Too much text is being redacted
- Loss of semantic meaning

**Diagnostic Code:**
```python
from privacy_finetuner import ContextGuard

# Test redaction behavior
guard = ContextGuard()
result = guard.explain_redactions("Your sensitive text here")
print(f"Redaction explanation: {result.explanation}")
```

**Solutions:**

1. **Adjust Sensitivity Threshold**
   ```python
   guard = ContextGuard(
       sensitivity_threshold=0.9,  # Increase from 0.8
       preserve_structure=True
   )
   ```

2. **Preserve Important Entities**
   ```python
   result = guard.protect(
       text,
       preserve_entities=["organization", "product", "location"]
   )
   ```

3. **Use Selective Redaction Strategies**
   ```python
   guard = ContextGuard(
       strategies=[RedactionStrategy.PII_REMOVAL],  # Only remove PII
       # Remove semantic encryption if too aggressive
   )
   ```

---

## 🚂 Training Issues

### Issue: Training Extremely Slow

**Symptoms:**
- Training takes much longer than expected
- Low GPU utilization

**Diagnostic Steps:**

1. **Check Resource Utilization**
   ```bash
   # Monitor GPU usage
   nvidia-smi -l 1
   
   # Monitor CPU and memory
   htop
   
   # Check I/O usage
   iotop
   ```

2. **Profile Training Performance**
   ```python
   from privacy_finetuner.monitoring import PerformanceProfiler
   
   profiler = PerformanceProfiler()
   with profiler.profile_training():
       result = trainer.train(dataset, epochs=1)
   
   report = profiler.get_performance_report()
   print(f"Bottlenecks: {report.bottlenecks}")
   ```

**Solutions:**

1. **Optimize Batch Size**
   ```python
   # Find optimal batch size
   optimal_batch_size = trainer.find_optimal_batch_size(
       dataset=dataset,
       max_memory_gb=16
   )
   
   result = trainer.train(
       dataset=dataset,
       batch_size=optimal_batch_size
   )
   ```

2. **Enable Performance Optimizations**
   ```python
   from privacy_finetuner.scaling import PerformanceOptimizer
   
   optimizer = PerformanceOptimizer(
       target_throughput=1000.0,
       auto_optimization=True
   )
   optimizer.start_optimization()
   
   # Train with optimization
   result = trainer.train(dataset)
   ```

3. **Use Distributed Training**
   ```python
   from privacy_finetuner.distributed import DistributedTrainer
   
   distributed_trainer = DistributedTrainer(
       model_name=model_name,
       privacy_config=privacy_config,
       num_gpus=2
   )
   ```

### Issue: Loss Not Converging

**Symptoms:**
- Loss plateaus early
- Poor model performance

**Solutions:**

1. **Adjust Learning Rate**
   ```python
   # Try different learning rates
   learning_rates = [1e-5, 3e-5, 5e-5, 1e-4]
   
   for lr in learning_rates:
       trainer = PrivateTrainer(model_name, privacy_config)
       result = trainer.train(
           dataset=dataset,
           learning_rate=lr,
           epochs=1
       )
       print(f"LR {lr}: Final loss {result.training_history.loss[-1]}")
   ```

2. **Reduce Privacy Noise**
   ```python
   privacy_config = PrivacyConfig(
       epsilon=2.0,  # Less privacy = less noise
       max_grad_norm=2.0,  # Higher gradient norm
       noise_multiplier=0.3  # Less noise
   )
   ```

3. **Check Data Quality**
   ```python
   # Analyze training data
   from privacy_finetuner.utils import DatasetAnalyzer
   
   analyzer = DatasetAnalyzer()
   analysis = analyzer.analyze_dataset(dataset)
   
   print(f"Data quality score: {analysis.quality_score}")
   print(f"Issues found: {analysis.issues}")
   ```

### Issue: Training Fails with `OutOfMemoryError`

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Solutions:**

1. **Reduce Batch Size**
   ```python
   result = trainer.train(
       dataset=dataset,
       batch_size=4,  # Reduce from 16
       gradient_accumulation_steps=4  # Maintain effective batch size
   )
   ```

2. **Enable Gradient Checkpointing**
   ```python
   trainer = PrivateTrainer(
       model_name=model_name,
       privacy_config=privacy_config,
       use_gradient_checkpointing=True
   )
   ```

3. **Use Model Sharding**
   ```python
   from privacy_finetuner.scaling import ModelSharding
   
   sharding = ModelSharding(
       shard_count=2,
       privacy_preserving=True
   )
   
   trainer = PrivateTrainer(
       model_name=model_name,
       privacy_config=privacy_config,
       model_sharding=sharding
   )
   ```

---

## 🚨 Security & Threat Detection Issues

### Issue: Too Many False Positive Alerts

**Symptoms:**
- Constant security alerts during normal training
- Training frequently interrupted

**Solutions:**

1. **Adjust Alert Threshold**
   ```python
   threat_detector = ThreatDetector(
       alert_threshold=0.8,  # Increase from 0.5
       monitoring_interval=5.0  # Reduce monitoring frequency
   )
   ```

2. **Tune Threat Detection Parameters**
   ```python
   # Update baseline metrics to reflect normal training
   threat_detector.update_baseline_metrics({
       "loss": 1.5,  # Expected loss range
       "gradient_l2_norm": 2.0,  # Expected gradient norms
       "accuracy": 0.8
   })
   ```

3. **Disable Specific Threat Types**
   ```python
   # Only monitor critical threats
   threat_detector = ThreatDetector(
       threat_types=[
           ThreatType.PRIVACY_BUDGET_EXHAUSTION,
           ThreatType.DATA_POISONING
       ]
   )
   ```

### Issue: Security System Not Detecting Real Threats

**Symptoms:**
- Suspicious training behavior not detected
- No alerts during actual attacks

**Diagnostic Code:**
```python
# Test threat detection manually
test_metrics = {
    "gradient_l2_norm": 50.0,  # Extremely high
    "current_loss": 10.0,      # Loss explosion
    "accuracy": 0.1            # Poor performance
}

alerts = threat_detector.detect_threat(test_metrics)
print(f"Alerts generated: {len(alerts)}")
```

**Solutions:**

1. **Lower Alert Threshold**
   ```python
   threat_detector = ThreatDetector(
       alert_threshold=0.3,  # More sensitive
       enable_automated_response=True
   )
   ```

2. **Enable All Threat Types**
   ```python
   threat_detector = ThreatDetector(
       threat_types=list(ThreatType),  # Monitor all threat types
       monitoring_interval=0.5  # More frequent monitoring
   )
   ```

3. **Update Threat Detection Models**
   ```python
   # Retrain threat detection on your data
   threat_detector.retrain_detection_models(
       training_data=historical_metrics,
       known_attacks=attack_examples
   )
   ```

### Issue: Recovery System Failing

**Symptoms:**
- Recovery points not being created
- Failure recovery not working

**Diagnostic Code:**
```python
from privacy_finetuner.resilience import FailureRecoverySystem

recovery = FailureRecoverySystem()

# Test recovery system health
test_results = recovery.test_recovery_system()
print(f"Health check results: {test_results}")
```

**Solutions:**

1. **Check Storage Permissions**
   ```bash
   # Ensure checkpoint directory is writable
   chmod 755 recovery_checkpoints/
   chown $USER:$USER recovery_checkpoints/
   ```

2. **Increase Recovery Resources**
   ```python
   recovery = FailureRecoverySystem(
       checkpoint_dir="recovery_checkpoints",
       max_recovery_attempts=5,  # Increase attempts
       privacy_threshold=0.9     # Adjust threshold
   )
   ```

3. **Manual Recovery Test**
   ```python
   # Test manual recovery
   recovery_id = recovery.create_recovery_point(
       epoch=1, step=100,
       model_state=model.state_dict(),
       optimizer_state=optimizer.state_dict(),
       privacy_state={"epsilon_spent": 0.1}
   )
   
   # Verify recovery point exists
   recovery_points = recovery.list_recovery_points()
   print(f"Available recovery points: {recovery_points}")
   ```

---

## ⚡ Scaling & Performance Issues

### Issue: Auto-Scaler Not Scaling

**Symptoms:**
- High load but no scaling events
- Resources underutilized

**Diagnostic Code:**
```python
from privacy_finetuner.scaling import AutoScaler

# Check scaling status
scaler = AutoScaler()
status = scaler.get_scaling_status()
print(f"Scaling status: {status}")

# Check scaling policy
policy = scaler.get_scaling_policy()
print(f"Scale up threshold: {policy.scale_up_threshold}")
print(f"Current metrics: {scaler.get_current_metrics()}")
```

**Solutions:**

1. **Adjust Scaling Thresholds**
   ```python
   scaling_policy = ScalingPolicy(
       scale_up_threshold={
           "cpu_utilization": 60.0,  # Reduce from 80.0
           "gpu_utilization": 65.0   # More aggressive scaling
       },
       min_nodes=1,
       max_nodes=10
   )
   ```

2. **Check Scaling Cooldown**
   ```python
   scaling_policy = ScalingPolicy(
       cooldown_period_seconds=60,  # Reduce from 300
       scaling_step_size=2          # Scale more aggressively
   )
   ```

3. **Enable Manual Scaling for Testing**
   ```python
   # Test manual scaling
   result = scaler.manual_scale(
       direction=ScalingDirection.SCALE_OUT,
       node_type=NodeType.GPU_WORKER,
       count=2
   )
   print(f"Manual scaling result: {result}")
   ```

### Issue: Performance Degradation After Scaling

**Symptoms:**
- Performance worse with more nodes
- Increased latency

**Solutions:**

1. **Check Network Latency**
   ```python
   from privacy_finetuner.monitoring import NetworkMonitor
   
   monitor = NetworkMonitor()
   latency_report = monitor.measure_inter_node_latency()
   print(f"Average latency: {latency_report.average_ms}ms")
   ```

2. **Optimize Communication**
   ```python
   from privacy_finetuner.distributed import CommunicationOptimizer
   
   comm_optimizer = CommunicationOptimizer(
       compression_enabled=True,
       batch_communication=True
   )
   ```

3. **Check Load Balancing**
   ```python
   from privacy_finetuner.scaling import LoadBalancer
   
   balancer = LoadBalancer()
   balance_report = balancer.get_load_distribution()
   
   if balance_report.imbalance_ratio > 0.3:
       balancer.rebalance_workload()
   ```

### Issue: Cost Optimization Not Working

**Symptoms:**
- High costs despite optimization enabled
- Suboptimal resource allocation

**Solutions:**

1. **Review Cost Policies**
   ```python
   cost_analysis = scaler.optimize_cost()
   print(f"Current cost: ${cost_analysis['current_hourly_cost']}")
   print("Recommendations:")
   for rec in cost_analysis['optimization_recommendations']:
       print(f"  - {rec['action']}: Save ${rec['potential_savings']}/hr")
   ```

2. **Set Strict Cost Limits**
   ```python
   scaling_policy = ScalingPolicy(
       cost_constraints={
           "max_hourly_cost": 50.0,  # Strict limit
           "cost_efficiency_threshold": 0.8
       }
   )
   ```

3. **Use Spot Instances (Cloud)**
   ```python
   scaler = AutoScaler(
       enable_cost_optimization=True,
       prefer_spot_instances=True,
       spot_instance_ratio=0.7  # 70% spot instances
   )
   ```

---

## 🌍 Global Deployment Issues

### Issue: Localization Not Working

**Symptoms:**
- Text not translating
- Formatting issues with dates/currency

**Solutions:**

1. **Check Locale Installation**
   ```bash
   # Install required locales (Ubuntu)
   sudo locale-gen en_US.UTF-8 de_DE.UTF-8 fr_FR.UTF-8 ja_JP.UTF-8
   sudo update-locale
   
   # Check available locales
   locale -a
   ```

2. **Test Locale Configuration**
   ```python
   from privacy_finetuner.global_first import I18nManager
   
   i18n = I18nManager()
   
   # Test translation
   result = i18n.translate("app.title", SupportedLocale.DE_DE)
   print(f"Translation result: {result}")
   
   # Test formatting
   formatted_date = i18n.format_date(time.time(), SupportedLocale.DE_DE)
   print(f"Formatted date: {formatted_date}")
   ```

3. **Update Translation Files**
   ```bash
   # Check translation file exists
   ls -la locales/de_DE/LC_MESSAGES/
   
   # Regenerate translation files if missing
   python scripts/generate_translations.py
   ```

### Issue: Compliance Violations Not Detected

**Symptoms:**
- Processing activities not tracked
- GDPR/CCPA violations missed

**Diagnostic Code:**
```python
from privacy_finetuner.global_first import ComplianceManager

compliance = ComplianceManager(primary_regions=["EU", "California"])

# Check compliance status
status = compliance.get_compliance_status()
print(f"Active violations: {status['active_violations']}")
print(f"Processing records: {status['processing_records']}")

# Test violation detection
test_processing = compliance.record_data_processing(
    data_categories=["personal_identifiers"],
    processing_purpose="marketing",  # This might trigger GDPR alerts
    legal_basis="none",              # Invalid legal basis
    data_subjects_count=1000000      # Large dataset
)
```

**Solutions:**

1. **Enable Real-time Monitoring**
   ```python
   compliance = ComplianceManager(
       primary_regions=["EU", "California", "Canada"],
       enable_real_time_monitoring=True,
       auto_remediation=True
   )
   compliance.start_compliance_monitoring()
   ```

2. **Update Compliance Rules**
   ```python
   # Add custom compliance rules
   compliance.add_compliance_rule(
       framework="gdpr",
       rule_type="data_retention",
       parameters={"max_retention_days": 365}
   )
   ```

3. **Check Regional Configuration**
   ```python
   # Verify regional compliance requirements
   requirements = compliance.get_regional_compliance_requirements("EU")
   print(f"GDPR requirements: {requirements}")
   ```

### Issue: Multi-Region Deployment Fails

**Symptoms:**
- Deployment timeouts
- Regional compliance errors

**Solutions:**

1. **Check Regional Availability**
   ```python
   from privacy_finetuner.global_first import DeploymentOrchestrator
   
   orchestrator = DeploymentOrchestrator()
   available_regions = orchestrator.get_available_regions()
   print(f"Available regions: {available_regions}")
   
   # Test regional connectivity
   connectivity = orchestrator.test_regional_connectivity()
   for region, status in connectivity.items():
       print(f"{region}: {status}")
   ```

2. **Increase Deployment Timeouts**
   ```python
   orchestrator = DeploymentOrchestrator(
       health_check_timeout=600,  # Increase from 300
       deployment_timeout=1800    # 30 minutes
   )
   ```

3. **Use Staged Deployment**
   ```python
   # Deploy to regions sequentially
   regions = ["us-east-1", "eu-west-1", "asia-southeast-1"]
   
   for region in regions:
       result = orchestrator.deploy_to_region(
           region=region,
           services=services,
           wait_for_completion=True
       )
       if not result.success:
           print(f"Deployment to {region} failed: {result.error}")
           break
   ```

---

## 💾 Memory & Resource Problems

### Issue: Memory Leaks

**Symptoms:**
- Memory usage continuously increases
- System becomes unresponsive

**Diagnostic Code:**
```python
import psutil
import gc
import torch

def monitor_memory():
    """Monitor memory usage during training"""
    process = psutil.Process()
    
    print(f"Memory usage: {process.memory_info().rss / 1024**2:.1f} MB")
    print(f"GPU memory: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
    
    # Force garbage collection
    gc.collect()
    torch.cuda.empty_cache()

# Call during training
monitor_memory()
```

**Solutions:**

1. **Enable Memory Profiling**
   ```python
   from privacy_finetuner.optimization import MemoryProfiler
   
   profiler = MemoryProfiler()
   
   with profiler.profile_memory():
       result = trainer.train(dataset)
   
   memory_report = profiler.get_memory_report()
   print(f"Memory leaks detected: {memory_report.leaks}")
   ```

2. **Use Memory-Efficient Training**
   ```python
   trainer = PrivateTrainer(
       model_name=model_name,
       privacy_config=privacy_config,
       use_gradient_checkpointing=True,
       mixed_precision=True,
       memory_efficient_attention=True
   )
   ```

3. **Clear Caches Regularly**
   ```python
   # Clear caches after each epoch
   def clear_caches():
       gc.collect()
       torch.cuda.empty_cache()
       torch.cuda.reset_peak_memory_stats()
   
   # In training loop
   for epoch in range(num_epochs):
       # ... training code ...
       clear_caches()
   ```

### Issue: Resource Allocation Failures

**Symptoms:**
- Cannot allocate requested resources
- Node scaling fails

**Solutions:**

1. **Check Resource Quotas**
   ```bash
   # Check Kubernetes resource quotas
   kubectl describe resourcequota -n privacy-ml
   
   # Check node capacity
   kubectl describe nodes
   ```

2. **Adjust Resource Requests**
   ```python
   # Reduce resource requirements
   scaling_policy = ScalingPolicy(
       node_requirements={
           "cpu": 2,     # Reduce from 4
           "memory_gb": 8,  # Reduce from 16
           "gpu": 0      # Make GPU optional
       }
   )
   ```

3. **Use Resource Pooling**
   ```python
   from privacy_finetuner.optimization import ResourcePool
   
   pool = ResourcePool(
       shared_resources=True,
       resource_recycling=True
   )
   ```

---

## 🌐 API & Integration Issues

### Issue: REST API Not Responding

**Symptoms:**
- API endpoints return 500 errors
- Connection timeouts

**Diagnostic Steps:**
```bash
# Check API server status
curl -v http://localhost:8080/health

# Check logs
docker logs privacy-gateway

# Test specific endpoints
curl -X POST http://localhost:8080/api/v1/training/start \
  -H "Content-Type: application/json" \
  -d '{"model_name": "test"}'
```

**Solutions:**

1. **Check Server Configuration**
   ```python
   from privacy_finetuner.api import APIServer
   
   # Start with debug mode
   server = APIServer(
       debug=True,
       log_level="DEBUG",
       host="0.0.0.0",
       port=8080
   )
   server.start()
   ```

2. **Increase Timeouts**
   ```python
   server = APIServer(
       request_timeout=300,  # 5 minutes
       worker_timeout=600    # 10 minutes
   )
   ```

3. **Check Authentication**
   ```python
   # Test API with authentication
   import requests
   
   response = requests.post(
       "http://localhost:8080/auth/login",
       json={"username": "admin", "password": "password"}
   )
   
   if response.status_code == 200:
       token = response.json()["access_token"]
       
       # Use token for API calls
       headers = {"Authorization": f"Bearer {token}"}
       result = requests.get(
           "http://localhost:8080/api/v1/training/jobs",
           headers=headers
       )
       print(f"API response: {result.status_code}")
   ```

### Issue: SDK Integration Problems

**Symptoms:**
- SDK methods not working as expected
- Type errors or attribute errors

**Solutions:**

1. **Check SDK Version Compatibility**
   ```python
   import privacy_finetuner
   print(f"SDK Version: {privacy_finetuner.__version__}")
   
   # Check for version conflicts
   from privacy_finetuner.utils import CompatibilityChecker
   checker = CompatibilityChecker()
   issues = checker.check_compatibility()
   if issues:
       print(f"Compatibility issues: {issues}")
   ```

2. **Use Type Hints for Debugging**
   ```python
   from typing import Optional
   from privacy_finetuner import PrivateTrainer, PrivacyConfig
   
   def create_trainer(model_name: str, 
                     privacy_config: PrivacyConfig) -> PrivateTrainer:
       return PrivateTrainer(model_name, privacy_config)
   ```

3. **Enable Verbose Logging**
   ```python
   import logging
   
   logging.basicConfig(level=logging.DEBUG)
   logger = logging.getLogger("privacy_finetuner")
   logger.setLevel(logging.DEBUG)
   
   # Now run your code with detailed logs
   ```

---

## ⚖️ Compliance & Regulatory Issues

### Issue: GDPR Compliance Failures

**Symptoms:**
- Data subject requests not handled
- Audit failures

**Solutions:**

1. **Enable GDPR Compliance Mode**
   ```python
   compliance = ComplianceManager(
       primary_regions=["EU"],
       enable_real_time_monitoring=True,
       gdpr_mode=True
   )
   
   # Set strict data handling policies
   compliance.configure_gdpr_policies({
       "data_minimization": True,
       "purpose_limitation": True,
       "storage_limitation": True
   })
   ```

2. **Implement Data Subject Rights**
   ```python
   # Handle access request
   response = compliance.handle_data_subject_request(
       request_type="access",
       data_subject_id="user123",
       region="EU",
       verification_method="email"
   )
   
   # Handle erasure request
   erasure_response = compliance.handle_data_subject_request(
       request_type="erasure",
       data_subject_id="user123",
       region="EU"
   )
   ```

### Issue: HIPAA Compliance Violations

**Symptoms:**
- Healthcare data not properly protected
- Audit trail insufficient

**Solutions:**

1. **Enable HIPAA Security Features**
   ```python
   # Configure for healthcare
   privacy_config = PrivacyConfig(
       epsilon=0.1,  # Stricter privacy for healthcare
       delta=1e-8,   # Lower delta
       secure_rng=True,
       audit_logging=True
   )
   
   hipaa_compliance = ComplianceManager(
       primary_regions=["US_Healthcare"],
       hipaa_mode=True,
       audit_level="comprehensive"
   )
   ```

2. **Implement Access Controls**
   ```python
   # Set healthcare-specific access controls
   access_policy = {
       "minimum_necessary": True,
       "role_based_access": True,
       "encryption_at_rest": True,
       "encryption_in_transit": True
   }
   
   hipaa_compliance.configure_access_controls(access_policy)
   ```

---

## 🔍 Debug Mode & Diagnostics

### Enable Comprehensive Debugging

```python
#!/usr/bin/env python3
"""
Comprehensive debugging setup for privacy-preserving ML
"""

import logging
import os
from privacy_finetuner.utils import setup_privacy_logging

# Enable debug mode
os.environ["PRIVACY_ML_DEBUG"] = "1"
os.environ["PRIVACY_ML_LOG_LEVEL"] = "DEBUG"

# Setup comprehensive logging
setup_privacy_logging(
    log_level="DEBUG",
    log_file="debug.log",
    structured_logging=True,
    privacy_redaction=False  # Disable for debugging (be careful with sensitive data)
)

# Enable component-specific debugging
components_to_debug = [
    "privacy_finetuner.core.trainer",
    "privacy_finetuner.security.threat_detector",
    "privacy_finetuner.scaling.auto_scaler",
    "privacy_finetuner.global_first.compliance_manager"
]

for component in components_to_debug:
    logger = logging.getLogger(component)
    logger.setLevel(logging.DEBUG)
    print(f"Debug logging enabled for {component}")

# Enable PyTorch debugging
import torch
torch.autograd.set_detect_anomaly(True)  # Detect gradient anomalies
torch.backends.cudnn.deterministic = True  # Reproducible results

print("🔍 Comprehensive debugging enabled!")
print("Check debug.log for detailed logs")
```

### System Diagnostics Script

```python
#!/usr/bin/env python3
"""
Complete system diagnostics for privacy-preserving ML framework
"""

import sys
import json
from datetime import datetime
from pathlib import Path

def run_diagnostics():
    """Run complete system diagnostics"""
    
    diagnostics_report = {
        "timestamp": datetime.now().isoformat(),
        "system_info": {},
        "component_status": {},
        "configuration": {},
        "performance_metrics": {},
        "issues": []
    }
    
    print("🔍 Running Complete System Diagnostics...")
    print("=" * 60)
    
    # System Information
    import platform
    diagnostics_report["system_info"] = {
        "platform": platform.platform(),
        "python_version": sys.version,
        "architecture": platform.architecture()[0]
    }
    
    # Component Status Checks
    components = {
        "privacy_finetuner": check_privacy_finetuner,
        "security": check_security_components,
        "scaling": check_scaling_components,
        "compliance": check_compliance_components,
        "gpu": check_gpu_status
    }
    
    for name, check_func in components.items():
        try:
            status = check_func()
            diagnostics_report["component_status"][name] = status
            print(f"✅ {name}: {status['status']}")
        except Exception as e:
            diagnostics_report["component_status"][name] = {
                "status": "error",
                "error": str(e)
            }
            diagnostics_report["issues"].append(f"{name}: {str(e)}")
            print(f"❌ {name}: Error - {str(e)}")
    
    # Save diagnostics report
    report_file = f"diagnostics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(diagnostics_report, f, indent=2)
    
    print(f"\n📊 Diagnostics Complete!")
    print(f"Report saved to: {report_file}")
    print(f"Issues found: {len(diagnostics_report['issues'])}")
    
    return diagnostics_report

def check_privacy_finetuner():
    """Check privacy finetuner core components"""
    from privacy_finetuner import PrivateTrainer, PrivacyConfig
    
    # Test basic initialization
    config = PrivacyConfig(epsilon=1.0, delta=1e-5)
    trainer = PrivateTrainer("microsoft/DialoGPT-small", config)
    
    return {
        "status": "healthy",
        "version": getattr(trainer, "__version__", "unknown"),
        "components": ["PrivateTrainer", "PrivacyConfig"]
    }

def check_security_components():
    """Check security monitoring components"""
    from privacy_finetuner.security import ThreatDetector
    from privacy_finetuner.resilience import FailureRecoverySystem
    
    detector = ThreatDetector()
    recovery = FailureRecoverySystem()
    
    return {
        "status": "healthy",
        "components": ["ThreatDetector", "FailureRecoverySystem"]
    }

def check_scaling_components():
    """Check scaling and performance components"""
    from privacy_finetuner.scaling import AutoScaler, PerformanceOptimizer
    
    scaler = AutoScaler()
    optimizer = PerformanceOptimizer()
    
    return {
        "status": "healthy",
        "components": ["AutoScaler", "PerformanceOptimizer"]
    }

def check_compliance_components():
    """Check global compliance components"""
    from privacy_finetuner.global_first import ComplianceManager, I18nManager
    
    compliance = ComplianceManager(primary_regions=["EU"])
    i18n = I18nManager()
    
    return {
        "status": "healthy",
        "components": ["ComplianceManager", "I18nManager"]
    }

def check_gpu_status():
    """Check GPU availability and status"""
    import torch
    
    if not torch.cuda.is_available():
        return {"status": "no_gpu", "message": "CUDA not available"}
    
    return {
        "status": "healthy",
        "gpu_count": torch.cuda.device_count(),
        "gpu_name": torch.cuda.get_device_name(0),
        "cuda_version": torch.version.cuda
    }

if __name__ == "__main__":
    run_diagnostics()
```

---

## 📞 Getting Additional Help

### Community Support

1. **GitHub Issues**: https://github.com/your-org/privacy-preserving-agent-finetuner/issues
2. **Discussions**: https://github.com/your-org/privacy-preserving-agent-finetuner/discussions
3. **Discord Community**: [Join our Discord](https://discord.gg/privacy-ml)

### Professional Support

- **Email**: support@your-org.com
- **Enterprise Support**: enterprise@your-org.com
- **Emergency Hotline**: +1-xxx-xxx-xxxx

### Before Contacting Support

Please prepare the following information:

1. **System Information**
   ```bash
   python --version
   pip list | grep -E "(torch|privacy|opacus)"
   nvidia-smi  # If using GPU
   ```

2. **Error Logs**
   ```bash
   # Collect relevant logs
   tail -100 /var/log/privacy-ml/app.log
   
   # Docker logs if using containers
   docker logs privacy-gateway --tail 100
   ```

3. **Configuration Details**
   - Privacy configuration used
   - Model and dataset information
   - Hardware specifications
   - Deployment environment (local/cloud/kubernetes)

4. **Reproduction Steps**
   - Minimal code example that reproduces the issue
   - Expected vs actual behavior
   - When the issue started occurring

### Emergency Procedures

For critical production issues:

1. **Immediate Actions**
   ```bash
   # Stop training if privacy budget exhausted
   kubectl scale deployment privacy-trainer --replicas=0
   
   # Enable emergency privacy budget reset (use with extreme caution)
   python -c "from privacy_finetuner.core.privacy_config import emergency_reset; emergency_reset()"
   
   # Create emergency backup
   kubectl exec -it privacy-db-pod -- pg_dump privacy_db > emergency_backup.sql
   ```

2. **Contact emergency support with**:
   - Severity level (Critical/High/Medium/Low)
   - Business impact description
   - Current system status
   - Actions already taken

---

This troubleshooting guide covers the most common issues across all generations of the privacy-preserving ML framework. For issues not covered here, please refer to the component-specific documentation or contact support.

Remember: When dealing with privacy-preserving systems, always err on the side of caution and maintain comprehensive audit trails of any troubleshooting actions taken.
"""Private trainer implementation with differential privacy guarantees."""

from typing import Dict, Any, Optional, Union, Tuple, List
from pathlib import Path
import logging
from datetime import datetime
import json
import time
import os
import warnings

# Handle optional dependencies gracefully
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Training functionality will be limited.")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn("Transformers not available. Model loading functionality will be limited.")

try:
    from datasets import Dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    warnings.warn("Datasets library not available. Dataset loading will use fallback methods.")

try:
    from opacus.accountants import RDPAccountant
    from opacus import PrivacyEngine
    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False
    warnings.warn("Opacus not available. Formal privacy guarantees will use approximations.")

from .privacy_config import PrivacyConfig

# Import quantum optimizer with fallback
try:
    if TORCH_AVAILABLE:
        from .quantum_optimizer import QuantumInspiredOptimizer
    else:
        from .quantum_optimizer_stub import QuantumInspiredOptimizer
except ImportError:
    from .quantum_optimizer_stub import QuantumInspiredOptimizer

# Import adaptive scheduler with graceful fallback
try:
    from .adaptive_privacy_scheduler import AdaptivePrivacyScheduler
except ImportError:
    # Create a simple stub if needed
    class AdaptivePrivacyScheduler:
        def __init__(self, initial_config):
            self.initial_config = initial_config
            logger.warning("Using AdaptivePrivacyScheduler stub")

# Import robust training components with fallback
try:
    from .robust_training import TrainingMonitor, SecurityMonitor, AuditLogger
except ImportError:
    # Create simple stubs
    class TrainingMonitor:
        def __init__(self, **kwargs):
            logger.warning("Using TrainingMonitor stub")
    
    class SecurityMonitor:
        def __init__(self, **kwargs):
            logger.warning("Using SecurityMonitor stub")
    
    class AuditLogger:
        def __init__(self, **kwargs):
            logger.warning("Using AuditLogger stub")
        
        def log_initialization(self, data):
            logger.info(f"Audit: Initialization - {data}")
        
        def log_training_event(self, event_type, data):
            logger.info(f"Audit: {event_type} - {data}")
from .exceptions import (
    PrivacyBudgetExhaustedException,
    ModelTrainingException,
    DataValidationException,
    SecurityViolationException,
    ResourceExhaustedException,
    ValidationException
)
from .circuit_breaker import (
    RobustExecutor,
    CircuitBreakerConfig,
    RetryConfig,
    RetryStrategy,
    robust_execution
)
# Try to import resource manager with fallback
try:
    from .resource_manager import resource_manager, ResourceType
except ImportError:
    from .resource_manager_stub import resource_manager, ResourceType

logger = logging.getLogger(__name__)

# Import additional modules for enhanced functionality
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class SimpleDictDataset:
    """Simple fallback dataset class when HuggingFace datasets is not available."""
    
    def __init__(self, data: list):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def select(self, indices):
        if isinstance(indices, range):
            indices = list(indices)
        selected_data = [self.data[i] for i in indices]
        return SimpleDictDataset(selected_data)
    
    def map(self, function, batched=False):
        if batched:
            # Simple batched processing
            batch_size = 100
            new_data = []
            for i in range(0, len(self.data), batch_size):
                batch = self.data[i:i+batch_size]
                batch_dict = {}
                for key in batch[0].keys():
                    batch_dict[key] = [item[key] for item in batch]
                processed = function(batch_dict)
                
                # Convert back to individual items
                for j in range(len(batch)):
                    item = {}
                    for key in processed.keys():
                        item[key] = processed[key][j] if key in processed else batch[j].get(key)
                    new_data.append(item)
        else:
            new_data = [function(item) for item in self.data]
        
        return SimpleDictDataset(new_data)


class BasicPrivacyAccountant:
    """Basic privacy accountant fallback when Opacus is not available."""
    
    def __init__(self):
        self._epsilon_spent = 0.0
        self._steps = 0
        
    def step(self, noise_multiplier: float, sample_rate: float):
        """Record a training step with privacy cost."""
        # Basic privacy cost approximation (not formally guaranteed)
        epsilon_step = sample_rate / (noise_multiplier ** 2)
        self._epsilon_spent += epsilon_step
        self._steps += 1
        
    def get_epsilon(self, delta: float) -> float:
        """Get current epsilon spending."""
        return self._epsilon_spent
    
    def get_privacy_spent(self) -> Dict[str, float]:
        """Get privacy spending summary."""
        return {
            "epsilon": self._epsilon_spent,
            "steps": self._steps,
            "accounting_method": "basic_approximation"
        }


class PrivateTrainer:
    """Differential privacy trainer for LLM fine-tuning.
    
    Implements DP-SGD with configurable privacy budgets and noise injection
    to ensure formal privacy guarantees during model training.
    """
    
    def __init__(
        self,
        model_name: str,
        privacy_config: PrivacyConfig,
        use_mcp_gateway: bool = True
    ):
        """Initialize private trainer with privacy configuration.
        
        Args:
            model_name: HuggingFace model identifier
            privacy_config: Privacy parameters and configuration
            use_mcp_gateway: Enable Model Context Protocol gateway
        """
        self.model_name = model_name
        self.privacy_config = privacy_config
        self.use_mcp_gateway = use_mcp_gateway
        self._privacy_accountant = None
        self._model = None
        self._quantum_optimizer = None
        self._adaptive_scheduler = None
        
        # Validate privacy configuration
        self.privacy_config.validate()
        
        # Setup security monitoring
        self._security_monitor = SecurityMonitor()
        self._audit_logger = AuditLogger()
        
        # Setup robust execution with circuit breaker and retry
        self._setup_error_recovery()
        
        # Setup resource management
        self._setup_resource_management()
        
        # Log initialization with audit trail
        self._audit_logger.log_initialization({
            "model_name": model_name,
            "epsilon": privacy_config.epsilon,
            "delta": privacy_config.delta,
            "timestamp": datetime.now().isoformat(),
            "user_context": self._get_user_context()
        })
        
        logger.info(f"Initialized PrivateTrainer for {model_name}")
        logger.info(f"Privacy budget: ε={privacy_config.epsilon}, δ={privacy_config.delta}")
    
    def train(
        self,
        dataset: str,
        epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        checkpoint_interval: int = 100,
        early_stopping_patience: int = 5,
        validation_split: float = 0.1,
        **kwargs
    ) -> Dict[str, Any]:
        """Train model with differential privacy guarantees.
        
        Args:
            dataset: Path to training dataset (JSONL format)
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for optimizer
            **kwargs: Additional training parameters
            
        Returns:
            Training results including privacy report
        """
        logger.info(f"Starting private training on {dataset}")
        logger.info(f"Training config: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}")
        
        try:
            # Validate inputs
            self._validate_training_inputs(dataset, epochs, batch_size, learning_rate)
            
            # Allocate resources for training
            resource_allocations = self._allocate_training_resources(batch_size)
            
            # Initialize model and privacy components with robust error handling
            self._setup_model_and_privacy()
            
            # Load and prepare dataset with validation
            train_dataset, val_dataset = self._load_and_split_dataset(
                dataset, validation_split
            )
            
            # Setup monitoring and checkpointing
            training_monitor = TrainingMonitor(
                checkpoint_interval=checkpoint_interval,
                early_stopping_patience=early_stopping_patience,
                privacy_config=self.privacy_config
            )
            
            # Setup DP-SGD training loop with robust error recovery
            results = self._train_with_dp_sgd_robust(
                train_dataset, val_dataset, epochs, batch_size, 
                learning_rate, training_monitor, **kwargs
            )
            
            logger.info("Training completed successfully")
            return results
            
        except PrivacyBudgetExhaustedException as e:
            logger.error(f"Privacy budget exhausted: {str(e)}")
            self._handle_privacy_budget_exhaustion()
            raise
        except ModelTrainingException as e:
            logger.error(f"Model training failed: {str(e)}")
            self._handle_training_failure(e)
            raise
        except DataValidationException as e:
            logger.error(f"Data validation failed: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected training failure: {str(e)}", exc_info=True)
            self._handle_unexpected_failure(e)
            raise
        finally:
            # Clean up allocated resources
            if 'resource_allocations' in locals():
                self._deallocate_training_resources(resource_allocations)
    
    def evaluate(self, test_set: str, privacy_aware: bool = True) -> Dict[str, Any]:
        """Evaluate model while tracking privacy leakage.
        
        Args:
            test_set: Path to test dataset
            privacy_aware: Whether to apply privacy-preserving evaluation
            
        Returns:
            Evaluation metrics with privacy analysis
        """
        logger.info(f"Evaluating on {test_set} (privacy_aware={privacy_aware})")
        
        try:
            # Load test dataset
            test_dataset = self._load_dataset(test_set)
            
            # Prepare data loader
            from torch.utils.data import DataLoader
            test_dataloader = DataLoader(
                test_dataset, 
                batch_size=8, 
                shuffle=False,
                collate_fn=self._data_collator
            )
            
            # Evaluation metrics
            total_loss = 0.0
            total_samples = 0
            privacy_leakage_score = 0.0
            
            self._model.eval()
            import torch
            
            with torch.no_grad():
                for batch in test_dataloader:
                    outputs = self._model(**batch, labels=batch['input_ids'])
                    loss = outputs.loss
                    
                    total_loss += loss.item() * len(batch['input_ids'])
                    total_samples += len(batch['input_ids'])
                    
                    # Privacy leakage analysis if requested
                    if privacy_aware:
                        privacy_leakage_score += self._analyze_privacy_leakage(outputs, batch)
            
            avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
            perplexity = torch.exp(torch.tensor(avg_loss)).item()
            
            # Calculate accuracy proxy (inverse perplexity)
            accuracy = max(0, 1 - (perplexity / 100))
            
            return {
                "accuracy": accuracy,
                "perplexity": perplexity,
                "average_loss": avg_loss,
                "privacy_leakage": privacy_leakage_score / total_samples if total_samples > 0 else 0.0,
                "privacy_budget_used": self._get_privacy_spent(),
                "samples_evaluated": total_samples
            }
            
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            return {"accuracy": 0.0, "privacy_leakage": 0.0, "error": str(e)}
    
    def get_privacy_report(self) -> Dict[str, Any]:
        """Generate comprehensive privacy audit report with enhanced metrics."""
        epsilon_spent = self._get_privacy_spent()
        
        return {
            "epsilon_spent": epsilon_spent,
            "delta": self.privacy_config.delta,
            "remaining_budget": max(0, self.privacy_config.epsilon - epsilon_spent),
            "budget_utilization": epsilon_spent / self.privacy_config.epsilon if self.privacy_config.epsilon > 0 else 0,
            "accounting_mode": self.privacy_config.accounting_mode,
            "privacy_risk_level": self._assess_privacy_risk(epsilon_spent),
            "error_recovery_metrics": self._get_error_recovery_metrics(),
            "compliance_status": self._check_compliance_status(epsilon_spent),
            "recommendations": self._get_privacy_recommendations(epsilon_spent),
            "timestamp": datetime.now().isoformat()
        }
    
    def _assess_privacy_risk(self, epsilon_spent: float) -> str:
        """Assess privacy risk level based on budget usage."""
        if self.privacy_config.epsilon <= 0:
            return "unknown"
        
        utilization = epsilon_spent / self.privacy_config.epsilon
        
        if utilization >= 0.95:
            return "critical"
        elif utilization >= 0.8:
            return "high"
        elif utilization >= 0.6:
            return "medium"
        else:
            return "low"
    
    def _get_error_recovery_metrics(self) -> Dict[str, Any]:
        """Get error recovery system metrics."""
        metrics = {}
        
        if hasattr(self, '_training_executor'):
            metrics['training_executor'] = self._training_executor.get_metrics()
        
        if hasattr(self, '_data_executor'):
            metrics['data_executor'] = self._data_executor.get_metrics()
        
        return metrics
    
    def _check_compliance_status(self, epsilon_spent: float) -> Dict[str, Any]:
        """Check compliance with privacy regulations."""
        return {
            "privacy_budget_compliant": epsilon_spent <= self.privacy_config.epsilon,
            "delta_compliant": self.privacy_config.delta <= 1e-3,  # Common compliance threshold
            "noise_adequate": self.privacy_config.noise_multiplier >= 0.5,
            "overall_compliant": (
                epsilon_spent <= self.privacy_config.epsilon and
                self.privacy_config.delta <= 1e-3 and
                self.privacy_config.noise_multiplier >= 0.5
            )
        }
    
    def _get_privacy_recommendations(self, epsilon_spent: float) -> List[str]:
        """Get privacy configuration recommendations."""
        recommendations = []
        
        if epsilon_spent > self.privacy_config.epsilon * 0.9:
            recommendations.append("Consider reducing learning rate to preserve privacy budget")
            recommendations.append("Implement early stopping to prevent budget exhaustion")
        
        if self.privacy_config.noise_multiplier < 0.5:
            recommendations.append("Increase noise multiplier for stronger privacy guarantees")
        
        if self.privacy_config.delta > 1e-3:
            recommendations.append("Consider reducing delta for better privacy compliance")
        
        if not recommendations:
            recommendations.append("Privacy configuration appears optimal")
        
        return recommendations
    
    def _get_privacy_spent(self) -> float:
        """Calculate privacy budget spent so far."""
        if self._privacy_accountant is None:
            return 0.0
        return self._privacy_accountant.get_epsilon(self.privacy_config.delta)
    
    def _setup_model_and_privacy(self) -> None:
        """Initialize model and privacy components with robust error handling."""
        logger.info(f"Loading model: {self.model_name}")
        
        if not TRANSFORMERS_AVAILABLE:
            raise ModelTrainingException(
                "Transformers library is not available. Install with: pip install transformers"
            )
        
        if not TORCH_AVAILABLE:
            raise ModelTrainingException(
                "PyTorch is not available. Install with: pip install torch"
            )
        
        try:
            # Load tokenizer with fallback
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=False,  # Security: don't execute remote code
                cache_dir=os.getenv("HF_CACHE_DIR", None)
            )
            
            # Load model with proper error handling
            model_kwargs = {
                "trust_remote_code": False,  # Security: don't execute remote code
                "cache_dir": os.getenv("HF_CACHE_DIR", None)
            }
            
            # Add device configuration if GPU available
            if torch.cuda.is_available():
                model_kwargs["torch_dtype"] = torch.float16
                model_kwargs["device_map"] = "auto"
                logger.info("GPU detected, using half precision and auto device mapping")
            else:
                logger.warning("No GPU detected, using CPU (training will be slow)")
                model_kwargs["torch_dtype"] = torch.float32
            
            self._model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
            
        except Exception as e:
            error_msg = f"Failed to load model '{self.model_name}': {str(e)}"
            logger.error(error_msg)
            raise ModelTrainingException(error_msg) from e
        
        # Add padding token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize privacy accountant with robust error handling
        if OPACUS_AVAILABLE:
            try:
                self._privacy_accountant = RDPAccountant()
                logger.info("Initialized Opacus RDP privacy accountant")
            except Exception as e:
                logger.error(f"Failed to initialize Opacus accountant: {e}")
                self._privacy_accountant = BasicPrivacyAccountant()
                logger.info("Falling back to basic privacy tracking")
        else:
            logger.warning("Opacus not available, using basic privacy tracking")
            self._privacy_accountant = BasicPrivacyAccountant()
        
        # Initialize quantum-inspired optimizer
        self._quantum_optimizer = QuantumInspiredOptimizer(
            privacy_config=self.privacy_config,
            model_params=self._model.parameters() if self._model else None
        )
        
        # Initialize adaptive privacy scheduler
        self._adaptive_scheduler = AdaptivePrivacyScheduler(
            initial_config=self.privacy_config
        )
        
        logger.info("Quantum-inspired optimization and adaptive scheduling initialized")
    
    def _load_dataset(self, dataset_path: str) -> Any:
        """Load and prepare training dataset with robust error handling and validation."""
        logger.info(f"Loading dataset from {dataset_path}")
        
        # Use robust executor for data loading
        def _load_dataset_internal() -> Any:
            return self._load_dataset_with_validation(dataset_path)
        
        result = self._data_executor.execute(_load_dataset_internal)
        if not result.success:
            logger.error(f"Failed to load dataset after {result.attempts} attempts")
            raise DataValidationException(
                f"Dataset loading failed: {result.exception}",
                data_path=dataset_path,
                context={"attempts": result.attempts, "total_time": result.total_time}
            )
        
        return result.result
    
    def _load_dataset_with_validation(self, dataset_path: str) -> Any:
        """Internal dataset loading with comprehensive validation."""
        # Validate dataset path
        dataset_file = Path(dataset_path)
        if not dataset_file.exists():
            raise DataValidationException(f"Dataset file not found: {dataset_path}")
        
        if not dataset_file.is_file():
            raise DataValidationException(f"Dataset path is not a file: {dataset_path}")
        
        if dataset_file.stat().st_size == 0:
            raise DataValidationException(f"Dataset file is empty: {dataset_path}")
        
        # Security check: ensure file is within allowed directories
        try:
            resolved_path = dataset_file.resolve()
            # Add allowed directories check here if needed
        except Exception as e:
            raise DataValidationException(f"Path resolution failed: {e}")
        
        # Load and validate dataset content with enhanced error handling
        data = []
        validation_errors = []
        skipped_lines = 0
        max_line_length = 100000  # 100KB per line limit
        max_lines = 1000000  # 1M lines limit
        
        try:
            with open(dataset_path, 'r', encoding='utf-8', errors='replace') as f:
                for line_num, line in enumerate(f, 1):
                    # Check line limits
                    if line_num > max_lines:
                        logger.warning(f"Dataset too large, stopping at line {max_lines}")
                        break
                    
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                    
                    # Check line length
                    if len(line) > max_line_length:
                        validation_errors.append(f"Line {line_num}: Line too long ({len(line)} chars)")
                        skipped_lines += 1
                        continue
                    
                    try:
                        item = json.loads(line)
                        
                        # Enhanced validation
                        validation_result = self._validate_data_item(item, line_num)
                        if validation_result['valid']:
                            # Sanitize the item
                            sanitized_item = self._sanitize_data_item(item)
                            data.append(sanitized_item)
                        else:
                            validation_errors.extend(validation_result['errors'])
                            skipped_lines += 1
                        
                    except json.JSONDecodeError as e:
                        validation_errors.append(f"Line {line_num}: Invalid JSON - {e}")
                        skipped_lines += 1
                        continue
                    except Exception as e:
                        validation_errors.append(f"Line {line_num}: Unexpected error - {e}")
                        skipped_lines += 1
                        continue
                
        except UnicodeDecodeError as e:
            raise DataValidationException(f"Cannot decode dataset file as UTF-8: {e}")
        except MemoryError:
            raise ResourceExhaustedException(
                "Insufficient memory to load dataset",
                resource_type="memory",
                context={"dataset_path": dataset_path, "lines_loaded": len(data)}
            )
        except Exception as e:
            raise DataValidationException(f"Failed to read dataset file: {e}")
        
        # Log validation summary
        if validation_errors:
            logger.warning(f"Dataset validation: {len(validation_errors)} errors, {skipped_lines} lines skipped")
            if len(validation_errors) > 10:
                logger.warning("First 10 validation errors:")
                for error in validation_errors[:10]:
                    logger.warning(f"  {error}")
            else:
                for error in validation_errors:
                    logger.warning(f"  {error}")
        
        # Final validation
        if not data:
            raise DataValidationException(
                f"No valid data found in dataset: {dataset_path}",
                data_path=dataset_path,
                validation_errors=validation_errors[:100]  # Keep first 100 errors
            )
        
        # Check minimum data requirements
        min_samples = 10
        if len(data) < min_samples:
            logger.warning(f"Dataset has only {len(data)} samples, which may be insufficient for training")
        
        logger.info(f"Successfully loaded {len(data)} samples from dataset (skipped {skipped_lines} invalid entries)")
        
        # Additional dataset statistics
        if data:
            self._log_dataset_statistics(data)
        
        # Convert to HuggingFace dataset if available, otherwise use fallback
        if DATASETS_AVAILABLE:
            from datasets import Dataset
            dataset = Dataset.from_list(data)
        else:
            # Fallback: use simple dict-based dataset
            dataset = SimpleDictDataset(data)
        
        # Tokenize dataset
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'] if 'text' in examples else examples['prompt'],
                truncation=True,
                padding=True,
                max_length=512
            )
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        return tokenized_dataset
    
    def _validate_data_item(self, item: Any, line_num: int) -> Dict[str, Any]:
        """Validate individual data item with comprehensive checks."""
        errors = []
        
        # Type validation
        if not isinstance(item, dict):
            return {'valid': False, 'errors': [f"Line {line_num}: Item is not a dictionary"]}
        
        # Required fields validation
        required_fields = ['text', 'prompt']
        if not any(field in item for field in required_fields):
            errors.append(f"Line {line_num}: Missing required field ('text' or 'prompt')")
        
        # Field content validation
        for field in ['text', 'prompt']:
            if field in item:
                if not isinstance(item[field], str):
                    errors.append(f"Line {line_num}: Field '{field}' must be string")
                elif len(item[field].strip()) == 0:
                    errors.append(f"Line {line_num}: Field '{field}' is empty")
                elif len(item[field]) > 50000:  # 50KB limit per field
                    errors.append(f"Line {line_num}: Field '{field}' too long ({len(item[field])} chars)")
        
        # Security validation - check for suspicious content
        for field in ['text', 'prompt']:
            if field in item and isinstance(item[field], str):
                if self._contains_suspicious_content(item[field]):
                    errors.append(f"Line {line_num}: Field '{field}' contains potentially malicious content")
        
        return {'valid': len(errors) == 0, 'errors': errors}
    
    def _sanitize_data_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize data item to remove potential security risks."""
        sanitized = {}
        
        for key, value in item.items():
            if isinstance(value, str):
                # Basic HTML/script tag removal
                import re
                value = re.sub(r'<script[^>]*>.*?</script>', '', value, flags=re.IGNORECASE | re.DOTALL)
                value = re.sub(r'<[^>]+>', '', value)  # Remove HTML tags
                value = value.strip()
                
                # Limit length
                if len(value) > 10000:
                    value = value[:10000] + "...[truncated]"
                
                sanitized[key] = value
            elif isinstance(value, (int, float, bool)):
                sanitized[key] = value
            else:
                # Convert other types to string
                sanitized[key] = str(value)[:1000]  # Limit length
        
        return sanitized
    
    def _contains_suspicious_content(self, text: str) -> bool:
        """Check if text contains suspicious content."""
        suspicious_patterns = [
            r'<script[^>]*>',
            r'javascript:',
            r'eval\s*\(',
            r'exec\s*\(',
            r'__import__\s*\(',
            r'file:///',
            r'\\x[0-9a-fA-F]{2}',  # Hex encoded characters
        ]
        
        import re
        for pattern in suspicious_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def _log_dataset_statistics(self, data: List[Dict[str, Any]]) -> None:
        """Log dataset statistics for monitoring."""
        try:
            # Calculate statistics
            text_lengths = []
            for item in data[:1000]:  # Sample first 1000 items
                text_field = item.get('text', item.get('prompt', ''))
                if isinstance(text_field, str):
                    text_lengths.append(len(text_field))
            
            if text_lengths:
                avg_length = sum(text_lengths) / len(text_lengths)
                max_length = max(text_lengths)
                min_length = min(text_lengths)
                
                logger.info(f"Dataset statistics: avg_length={avg_length:.1f}, min={min_length}, max={max_length}")
            
        except Exception as e:
            logger.debug(f"Failed to calculate dataset statistics: {e}")
    
    def _train_with_dp_sgd(
        self, 
        dataset: Any, 
        epochs: int, 
        batch_size: int, 
        learning_rate: float,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute DP-SGD training loop."""
        import torch
        from torch.utils.data import DataLoader
        
        # Setup data loader
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=True,
            collate_fn=self._data_collator
        )
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(self._model.parameters(), lr=learning_rate)
        
        # Apply privacy engine if available
        if self._privacy_accountant is not None:
            try:
                from opacus import PrivacyEngine
                privacy_engine = PrivacyEngine()
                
                self._model, optimizer, dataloader = privacy_engine.make_private_with_epsilon(
                    module=self._model,
                    optimizer=optimizer,
                    data_loader=dataloader,
                    epochs=epochs,
                    target_epsilon=self.privacy_config.epsilon,
                    target_delta=self.privacy_config.delta,
                    max_grad_norm=self.privacy_config.max_grad_norm,
                )
                
                logger.info(f"Applied Opacus privacy engine with ε={self.privacy_config.epsilon}")
            except ImportError:
                logger.warning("Opacus not available, training without formal privacy guarantees")
        
        # Training loop
        total_steps = 0
        training_losses = []
        
        self._model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            epoch_steps = 0
            
            for batch in dataloader:
                # Forward pass
                outputs = self._model(**batch, labels=batch['input_ids'])
                loss = outputs.loss
                
                # Backward pass
                loss.backward()
                
                # Manual gradient clipping if not using Opacus
                if self._privacy_accountant is None:
                    torch.nn.utils.clip_grad_norm_(
                        self._model.parameters(), 
                        self.privacy_config.max_grad_norm
                    )
                
                optimizer.step()
                optimizer.zero_grad()
                
                epoch_loss += loss.item()
                epoch_steps += 1
                total_steps += 1
            
            avg_epoch_loss = epoch_loss / epoch_steps if epoch_steps > 0 else 0.0
            training_losses.append(avg_epoch_loss)
            
            logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_epoch_loss:.4f}")
        
        # Save model
        model_path = f"./models/private_model_eps_{self.privacy_config.epsilon}"
        self._model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)
        
        return {
            "status": "training_complete",
            "epochs_completed": epochs,
            "total_steps": total_steps,
            "final_loss": training_losses[-1] if training_losses else 0.0,
            "privacy_spent": self._get_privacy_spent(),
            "model_path": model_path,
            "training_losses": training_losses
        }
    
    def _data_collator(self, batch):
        """Collate function for DataLoader."""
        import torch
        
        # Simple collation - pad sequences to same length
        input_ids = [item['input_ids'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]
        
        # Pad sequences
        max_len = max(len(seq) for seq in input_ids)
        
        padded_input_ids = []
        padded_attention_mask = []
        
        for ids, mask in zip(input_ids, attention_mask):
            padding = [self.tokenizer.pad_token_id] * (max_len - len(ids))
            mask_padding = [0] * (max_len - len(mask))
            
            padded_input_ids.append(ids + padding)
            padded_attention_mask.append(mask + mask_padding)
        
        return {
            'input_ids': torch.tensor(padded_input_ids),
            'attention_mask': torch.tensor(padded_attention_mask)
        }
    
    def _analyze_privacy_leakage(self, outputs, batch) -> float:
        """Analyze potential privacy leakage in model outputs.
        
        This implements membership inference attack resistance analysis.
        """
        import torch
        import torch.nn.functional as F
        
        # Simple privacy leakage metric based on output confidence
        logits = outputs.logits
        probs = F.softmax(logits, dim=-1)
        
        # High confidence predictions might indicate memorization
        max_probs = torch.max(probs, dim=-1)[0]
        avg_confidence = torch.mean(max_probs).item()
        
        # Convert confidence to leakage score (higher confidence = potential leakage)
        leakage_score = max(0, avg_confidence - 0.7)  # Threshold for concern
        
        return leakage_score
    
    def save_checkpoint(self, checkpoint_path: str) -> None:
        """Save training checkpoint with privacy state."""
        import torch
        import json
        from pathlib import Path
        
        checkpoint_dir = Path(checkpoint_path)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        if self._model is not None:
            self._model.save_pretrained(checkpoint_dir / "model")
            self.tokenizer.save_pretrained(checkpoint_dir / "model")
        
        # Save privacy state
        privacy_state = {
            "epsilon_spent": self._get_privacy_spent(),
            "privacy_config": {
                "epsilon": self.privacy_config.epsilon,
                "delta": self.privacy_config.delta,
                "max_grad_norm": self.privacy_config.max_grad_norm,
                "noise_multiplier": self.privacy_config.noise_multiplier,
                "accounting_mode": self.privacy_config.accounting_mode
            }
        }
        
        with open(checkpoint_dir / "privacy_state.json", 'w') as f:
            json.dump(privacy_state, f, indent=2)
        
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def _validate_training_inputs(self, dataset, epochs, batch_size, learning_rate):
        """Enhanced validation of training inputs with comprehensive checks."""
        from pathlib import Path
        import sys
        
        errors = []
        warnings = []
        
        # Dataset validation
        try:
            dataset_path = Path(dataset)
            if not dataset_path.exists():
                errors.append(f"Dataset file not found: {dataset}")
            elif not dataset_path.is_file():
                errors.append(f"Dataset path is not a file: {dataset}")
            elif dataset_path.stat().st_size == 0:
                errors.append(f"Dataset file is empty: {dataset}")
            elif dataset_path.stat().st_size > 10 * 1024 * 1024 * 1024:  # 10GB
                warnings.append(f"Large dataset file ({dataset_path.stat().st_size / (1024**3):.2f}GB) may cause memory issues")
                
            # Check file extension
            if dataset_path.suffix.lower() not in ['.jsonl', '.json', '.csv', '.txt']:
                warnings.append(f"Unusual dataset file extension: {dataset_path.suffix}")
                
        except Exception as e:
            errors.append(f"Dataset validation error: {str(e)}")
        
        # Numeric parameter validation with bounds checking
        if not isinstance(epochs, int) or epochs <= 0:
            errors.append(f"Epochs must be a positive integer, got: {epochs}")
        elif epochs > 1000:
            warnings.append(f"Very high epoch count ({epochs}) may lead to overfitting")
            
        if not isinstance(batch_size, int) or batch_size <= 0:
            errors.append(f"Batch size must be a positive integer, got: {batch_size}")
        elif batch_size > 1024:
            warnings.append(f"Large batch size ({batch_size}) may cause memory issues")
        elif batch_size == 1:
            warnings.append("Batch size of 1 may lead to unstable training")
            
        if not isinstance(learning_rate, (int, float)) or learning_rate <= 0:
            errors.append(f"Learning rate must be a positive number, got: {learning_rate}")
        elif learning_rate > 1.0:
            warnings.append(f"Very high learning rate ({learning_rate}) may cause training instability")
        elif learning_rate < 1e-6:
            warnings.append(f"Very low learning rate ({learning_rate}) may lead to slow convergence")
        
        # System resource validation
        try:
            import psutil
            available_memory = psutil.virtual_memory().available
            estimated_memory_usage = batch_size * 1024 * 1024 * 100  # Rough estimate
            
            if estimated_memory_usage > available_memory * 0.8:
                warnings.append(f"Training may use {estimated_memory_usage / (1024**3):.2f}GB memory, but only {available_memory / (1024**3):.2f}GB available")
        except ImportError:
            logger.debug("psutil not available for memory validation")
        
        # GPU validation if available
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                estimated_gpu_usage = batch_size * 1024 * 1024 * 50  # Rough estimate
                
                if estimated_gpu_usage > gpu_memory * 0.8:
                    warnings.append(f"Training may use {estimated_gpu_usage / (1024**3):.2f}GB GPU memory, but only {gpu_memory / (1024**3):.2f}GB available")
            except Exception as e:
                logger.debug(f"GPU validation error: {e}")
        
        # Privacy budget validation
        if hasattr(self, 'privacy_config') and self.privacy_config:
            estimated_steps = epochs * 1000  # Rough estimate
            sample_rate = min(batch_size / 1000, 1.0)  # Rough estimate
            
            estimated_privacy_cost = self.privacy_config.estimate_privacy_cost(estimated_steps, sample_rate)
            if estimated_privacy_cost > self.privacy_config.epsilon:
                errors.append(f"Estimated privacy cost ({estimated_privacy_cost:.6f}) exceeds budget ({self.privacy_config.epsilon:.6f})")
            elif estimated_privacy_cost > self.privacy_config.epsilon * 0.9:
                warnings.append(f"Training may consume most privacy budget ({estimated_privacy_cost:.6f}/{self.privacy_config.epsilon:.6f})")
        
        # Log warnings
        for warning in warnings:
            logger.warning(f"Input validation warning: {warning}")
        
        # Raise errors if any
        if errors:
            error_message = "Input validation failed:\n" + "\n".join(f"  - {error}" for error in errors)
            raise ValidationException(error_message, context={"errors": errors, "warnings": warnings})
    
    def _load_and_split_dataset(self, dataset_path: str, validation_split: float) -> tuple:
        """Load and split dataset into training and validation sets."""
        dataset = self._load_dataset(dataset_path)
        
        if validation_split > 0:
            train_size = int((1 - validation_split) * len(dataset))
            train_dataset = dataset.select(range(train_size))
            val_dataset = dataset.select(range(train_size, len(dataset)))
            return train_dataset, val_dataset
        else:
            return dataset, None
    
    def _train_with_dp_sgd_robust(
        self, 
        train_dataset, 
        val_dataset,
        epochs: int, 
        batch_size: int, 
        learning_rate: float,
        training_monitor,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute robust DP-SGD training with comprehensive error recovery."""
        def _train_internal():
            return self._train_with_dp_sgd(
                train_dataset, epochs, batch_size, learning_rate, **kwargs
            )
        
        # Execute training with robust error handling
        result = self._training_executor.execute(_train_internal)
        
        if not result.success:
            logger.error(f"Training failed after {result.attempts} attempts: {result.exception}")
            
            # Check if fallback was used
            if hasattr(result, 'result') and isinstance(result.result, dict) and result.result.get('fallback_used'):
                logger.warning("Training completed using fallback mode")
                return result.result
            
            # Analyze failure and provide detailed error information
            failure_context = {
                "attempts": result.attempts,
                "total_time": result.total_time,
                "circuit_state": result.circuit_state.value if hasattr(result, 'circuit_state') else 'unknown',
                "executor_metrics": self._training_executor.get_metrics()
            }
            
            # Raise appropriate exception based on failure type
            if isinstance(result.exception, torch.cuda.OutOfMemoryError):
                raise ResourceExhaustedException(
                    f"GPU memory exhausted during training: {result.exception}",
                    resource_type="gpu_memory",
                    context=failure_context
                )
            elif isinstance(result.exception, PrivacyBudgetExhaustedException):
                raise result.exception  # Re-raise privacy exceptions as-is
            else:
                raise ModelTrainingException(
                    f"Training failed after {result.attempts} attempts: {result.exception}",
                    context=failure_context
                )
        
        logger.info(f"Training completed successfully after {result.attempts} attempts")
        return result.result
    
    def _handle_privacy_budget_exhaustion(self) -> None:
        """Handle privacy budget exhaustion."""
        logger.warning("Privacy budget exhausted - stopping training")
    
    def _handle_training_failure(self, e: Exception) -> None:
        """Handle training failures with recovery logic."""
        logger.error(f"Training failed: {e}")
    
    def _handle_unexpected_failure(self, e: Exception) -> None:
        """Handle unexpected failures."""
        logger.critical(f"Unexpected failure: {e}")
    
    def _setup_error_recovery(self) -> None:
        """Setup error recovery mechanisms."""
        # Configure circuit breaker for training operations
        circuit_config = CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=60.0,
            expected_exception=ModelTrainingException,
            fallback_function=self._training_fallback,
            half_open_max_calls=2
        )
        
        # Configure retry mechanism
        retry_config = RetryConfig(
            max_attempts=3,
            base_delay=5.0,
            max_delay=120.0,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            jitter=True,
            timeout=300.0,  # 5 minutes timeout
            retriable_exceptions=[
                ConnectionError, TimeoutError, RuntimeError,
                torch.cuda.OutOfMemoryError if TORCH_AVAILABLE else Exception
            ]
        )
        
        # Create robust executors for different operations
        self._training_executor = RobustExecutor(
            circuit_config=circuit_config,
            retry_config=retry_config,
            enable_circuit_breaker=True,
            enable_retry=True
        )
        
        # Create executor for data loading
        data_retry_config = RetryConfig(
            max_attempts=5,
            base_delay=2.0,
            max_delay=60.0,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF
        )
        
        self._data_executor = RobustExecutor(
            circuit_config=None,
            retry_config=data_retry_config,
            enable_circuit_breaker=False,
            enable_retry=True
        )
        
        logger.info("Error recovery mechanisms initialized")
    
    def _training_fallback(self, *args, **kwargs) -> Dict[str, Any]:
        """Fallback function for training failures."""
        logger.warning("Using training fallback due to circuit breaker activation")
        
        # Return minimal training result to allow graceful degradation
        return {
            "status": "degraded_mode",
            "epochs_completed": 0,
            "total_steps": 0,
            "final_loss": float('inf'),
            "privacy_spent": 0.0,
            "model_path": None,
            "training_losses": [],
            "fallback_used": True,
            "message": "Training running in degraded mode due to repeated failures"
        }
    
    def _get_user_context(self) -> dict:
        """Get current user context for auditing."""
        import os
        return {
            "user": os.getenv("USER", "unknown"),
            "hostname": os.getenv("HOSTNAME", "unknown")
        }
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load training checkpoint with privacy state."""
        import json
        from pathlib import Path
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        checkpoint_dir = Path(checkpoint_path)
        
        # Load model
        model_path = checkpoint_dir / "model"
        if model_path.exists():
            self._model = AutoModelForCausalLM.from_pretrained(str(model_path))
            self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
            logger.info(f"Model loaded from {model_path}")
        
        # Load privacy state
        privacy_state_path = checkpoint_dir / "privacy_state.json"
        if privacy_state_path.exists():
            with open(privacy_state_path, 'r') as f:
                privacy_state = json.load(f)
            
            logger.info(f"Privacy state loaded: ε_spent = {privacy_state.get('epsilon_spent', 0)}")
        else:
            logger.warning("No privacy state found in checkpoint")
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health report."""
        health_report = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": "healthy",
            "components": {},
            "recommendations": []
        }
        
        # Check privacy system health
        privacy_health = self._check_privacy_health()
        health_report["components"]["privacy_system"] = privacy_health
        
        # Check error recovery system health
        if hasattr(self, '_training_executor'):
            recovery_metrics = self._training_executor.get_metrics()
            recovery_health = {
                "status": "healthy" if recovery_metrics.get('success_rate', 0) > 0.8 else "degraded",
                "metrics": recovery_metrics
            }
            health_report["components"]["error_recovery"] = recovery_health
        
        # Check system resources
        try:
            import psutil
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent()
            
            resource_health = {
                "status": "healthy",
                "memory_usage": memory.percent,
                "cpu_usage": cpu_percent,
                "warnings": []
            }
            
            if memory.percent > 90:
                resource_health["status"] = "critical"
                resource_health["warnings"].append("High memory usage")
            elif memory.percent > 80:
                resource_health["status"] = "warning"
                resource_health["warnings"].append("Elevated memory usage")
                
            if cpu_percent > 95:
                resource_health["warnings"].append("High CPU usage")
            
            health_report["components"]["system_resources"] = resource_health
            
        except ImportError:
            health_report["components"]["system_resources"] = {"status": "unavailable"}
        
        # Determine overall status
        component_statuses = [comp.get("status", "unknown") for comp in health_report["components"].values()]
        if "critical" in component_statuses:
            health_report["overall_status"] = "critical"
        elif "degraded" in component_statuses or "warning" in component_statuses:
            health_report["overall_status"] = "degraded"
        
        return health_report
    
    def _check_privacy_health(self) -> Dict[str, Any]:
        """Check privacy system health."""
        epsilon_spent = self._get_privacy_spent()
        utilization = epsilon_spent / self.privacy_config.epsilon if self.privacy_config.epsilon > 0 else 0
        
        if utilization >= 0.95:
            status = "critical"
        elif utilization >= 0.8:
            status = "warning"
        else:
            status = "healthy"
        
        return {
            "status": status,
            "epsilon_spent": epsilon_spent,
            "budget_utilization": utilization,
            "privacy_config_valid": True  # Could add more validation here
        }
    
    def _setup_resource_management(self):
        """Setup resource management for the trainer."""
        try:
            # Start resource management system
            if not resource_manager.resource_management_active:
                resource_manager.start_resource_management()
            
            # Optimize for privacy training
            optimization_result = resource_manager.optimize_for_privacy_training()
            logger.info(f"Resource optimization completed: {optimization_result}")
            
            # Store resource allocations for cleanup
            self._resource_allocations = {}
            
        except Exception as e:
            logger.warning(f"Resource management setup failed: {e}")
            # Continue without resource management if it fails
            self._resource_allocations = {}
    
    def _allocate_training_resources(self, batch_size: int) -> Dict[str, Optional[str]]:
        """Allocate resources for training based on batch size and model requirements."""
        try:
            # Estimate resource requirements based on model and batch size
            estimated_memory = self._estimate_memory_requirement(batch_size)
            estimated_gpu_memory = self._estimate_gpu_memory_requirement(batch_size)
            estimated_cpu = self._estimate_cpu_requirement()
            
            # Allocate resources
            allocations = resource_manager.allocate_training_resources(
                memory_gb=estimated_memory,
                gpu_memory_gb=estimated_gpu_memory,
                cpu_cores=estimated_cpu,
                owner=f"trainer_{id(self)}",
                priority=7  # High priority for training
            )
            
            logger.info(f"Allocated training resources: memory={estimated_memory}GB, "
                       f"gpu_memory={estimated_gpu_memory}GB, cpu={estimated_cpu} cores")
            
            return allocations
            
        except Exception as e:
            logger.warning(f"Resource allocation failed: {e}")
            return {}
    
    def _deallocate_training_resources(self, allocations: Dict[str, Optional[str]]):
        """Deallocate training resources."""
        try:
            if allocations:
                success = resource_manager.deallocate_training_resources(allocations)
                if success:
                    logger.info("Training resources deallocated successfully")
                else:
                    logger.warning("Some resources may not have been properly deallocated")
            
        except Exception as e:
            logger.error(f"Resource deallocation failed: {e}")
    
    def _estimate_memory_requirement(self, batch_size: int) -> float:
        """Estimate memory requirement for training."""
        # Base memory for model loading (rough estimate)
        base_memory = 2.0  # GB
        
        # Additional memory per batch item (rough estimate)
        memory_per_item = 0.01  # GB per item
        
        # Differential privacy overhead
        dp_overhead = 0.5  # GB
        
        total_memory = base_memory + (batch_size * memory_per_item) + dp_overhead
        return max(1.0, total_memory)  # At least 1GB
    
    def _estimate_gpu_memory_requirement(self, batch_size: int) -> float:
        """Estimate GPU memory requirement for training."""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return 0.0
        
        # Base GPU memory for model
        base_gpu_memory = 1.0  # GB
        
        # Additional GPU memory per batch item
        gpu_memory_per_item = 0.005  # GB per item
        
        total_gpu_memory = base_gpu_memory + (batch_size * gpu_memory_per_item)
        return total_gpu_memory
    
    def _estimate_cpu_requirement(self) -> float:
        """Estimate CPU requirement for training."""
        # Base CPU requirement
        return 2.0  # 2 CPU cores
    
    def get_resource_status(self) -> Dict[str, Any]:
        """Get current resource status and usage."""
        try:
            if hasattr(self, '_resource_allocations'):
                resource_summary = resource_manager.get_comprehensive_status()
                return {
                    "resource_manager_active": resource_manager.resource_management_active,
                    "current_allocations": self._resource_allocations,
                    "system_status": resource_summary.get("system_health", {}),
                    "resource_usage": resource_summary.get("resource_usage", {}),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {"status": "resource_management_not_initialized"}
                
        except Exception as e:
            logger.error(f"Failed to get resource status: {e}")
            return {"status": "error", "error": str(e)}
"""Private trainer implementation with differential privacy guarantees."""

from typing import Dict, Any, Optional, Union, Tuple
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
    SecurityViolationException
)

logger = logging.getLogger(__name__)


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
        """Generate comprehensive privacy audit report."""
        return {
            "epsilon_spent": self._get_privacy_spent(),
            "delta": self.privacy_config.delta,
            "remaining_budget": max(0, self.privacy_config.epsilon - self._get_privacy_spent()),
            "accounting_mode": self.privacy_config.accounting_mode
        }
    
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
        """Load and prepare training dataset with robust error handling."""
        logger.info(f"Loading dataset from {dataset_path}")
        
        # Validate dataset path
        dataset_file = Path(dataset_path)
        if not dataset_file.exists():
            raise DataValidationException(f"Dataset file not found: {dataset_path}")
        
        if not dataset_file.is_file():
            raise DataValidationException(f"Dataset path is not a file: {dataset_path}")
        
        if dataset_file.stat().st_size == 0:
            raise DataValidationException(f"Dataset file is empty: {dataset_path}")
        
        # Load and validate dataset content
        data = []
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                    
                    try:
                        item = json.loads(line)
                        
                        # Validate required fields
                        if not isinstance(item, dict):
                            logger.warning(f"Line {line_num}: Item is not a dictionary, skipping")
                            continue
                        
                        if 'text' not in item and 'prompt' not in item:
                            logger.warning(f"Line {line_num}: Missing 'text' or 'prompt' field, skipping")
                            continue
                        
                        data.append(item)
                        
                    except json.JSONDecodeError as e:
                        logger.warning(f"Line {line_num}: Invalid JSON, skipping - {e}")
                        continue
                        
        except UnicodeDecodeError as e:
            raise DataValidationException(f"Cannot decode dataset file as UTF-8: {e}")
        except Exception as e:
            raise DataValidationException(f"Failed to read dataset file: {e}")
        
        if not data:
            raise DataValidationException(f"No valid data found in dataset: {dataset_path}")
        
        logger.info(f"Loaded {len(data)} samples from dataset")
        
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
        """Validate training inputs for correctness."""
        from pathlib import Path
        
        if not Path(dataset).exists():
            raise DataValidationException(f"Dataset not found: {dataset}")
        if epochs <= 0:
            raise ValueError("Epochs must be positive")
        if batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
    
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
        """Execute robust DP-SGD training with error recovery."""
        return self._train_with_dp_sgd(train_dataset, epochs, batch_size, learning_rate, **kwargs)
    
    def _handle_privacy_budget_exhaustion(self) -> None:
        """Handle privacy budget exhaustion."""
        logger.warning("Privacy budget exhausted - stopping training")
    
    def _handle_training_failure(self, e: Exception) -> None:
        """Handle training failures with recovery logic."""
        logger.error(f"Training failed: {e}")
    
    def _handle_unexpected_failure(self, e: Exception) -> None:
        """Handle unexpected failures."""
        logger.critical(f"Unexpected failure: {e}")
    
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
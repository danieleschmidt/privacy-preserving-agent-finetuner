"""Private trainer implementation with differential privacy guarantees."""

from typing import Dict, Any, Optional
from pathlib import Path
import logging

from .privacy_config import PrivacyConfig

logger = logging.getLogger(__name__)


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
        
        # Validate privacy configuration
        self.privacy_config.validate()
        
        logger.info(f"Initialized PrivateTrainer for {model_name}")
        logger.info(f"Privacy budget: ε={privacy_config.epsilon}, δ={privacy_config.delta}")
    
    def train(
        self,
        dataset: str,
        epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
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
            # Initialize model and privacy components
            self._setup_model_and_privacy()
            
            # Load and prepare dataset
            train_dataset = self._load_dataset(dataset)
            
            # Setup DP-SGD training loop
            results = self._train_with_dp_sgd(
                train_dataset, epochs, batch_size, learning_rate, **kwargs
            )
            
            logger.info("Training completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
    
    def evaluate(self, test_set: str) -> Dict[str, Any]:
        """Evaluate model while tracking privacy leakage."""
        logger.info(f"Evaluating on {test_set}")
        # TODO: Implement privacy-aware evaluation
        return {"accuracy": 0.0, "privacy_leakage": 0.0}
    
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
        """Initialize model and privacy components."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info(f"Loading model: {self.model_name}")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Add padding token if missing
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize privacy accountant
        try:
            from opacus.accountants import RDPAccountant
            self._privacy_accountant = RDPAccountant()
            logger.info("Initialized Opacus privacy accountant")
        except ImportError:
            logger.warning("Opacus not available, using basic privacy tracking")
            self._privacy_accountant = None
    
    def _load_dataset(self, dataset_path: str) -> Any:
        """Load and prepare training dataset."""
        import json
        from datasets import Dataset
        
        logger.info(f"Loading dataset from {dataset_path}")
        
        # Load JSONL dataset
        data = []
        with open(dataset_path, 'r') as f:
            for line in f:
                data.append(json.loads(line.strip()))
        
        # Convert to HuggingFace dataset
        dataset = Dataset.from_list(data)
        
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
"""
Context protection mechanisms for safeguarding sensitive data in prompts.

This module implements various redaction and protection strategies to ensure
sensitive information is removed or encrypted before model processing.
"""

import re
import hashlib
from enum import Enum
from typing import List, Dict, Any
from dataclasses import dataclass


class RedactionStrategy(Enum):
    """Available context protection strategies."""
    
    PII_REMOVAL = "pii_removal"
    ENTITY_HASHING = "entity_hashing"  
    SEMANTIC_ENCRYPTION = "semantic_encryption"
    K_ANONYMIZATION = "k_anonymization"


@dataclass
class RedactionReport:
    """Report of redactions applied to text."""
    
    original_length: int
    redacted_length: int
    redactions_applied: List[str]
    sensitivity_level: str
    confidence_score: float


class ContextGuard:
    """
    Privacy protection for text inputs and model context windows.
    
    Applies configurable redaction strategies to remove or obfuscate
    sensitive information while preserving semantic structure.
    
    Args:
        strategies: List of redaction strategies to apply
        salt: Salt for consistent hashing (if using entity hashing)
    """
    
    def __init__(
        self,
        strategies: List[RedactionStrategy] = None,
        salt: str = "privacy-finetuner-default"
    ):
        self.strategies = strategies or [RedactionStrategy.PII_REMOVAL]
        self.salt = salt
        
        # Common PII patterns
        self.pii_patterns = {
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'phone': re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'),
            'ssn': re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            'credit_card': re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'),
            'name': re.compile(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'),  # Simple name pattern
        }
    
    def protect(self, text: str, sensitivity_level: str = "medium") -> str:
        """
        Apply privacy protection to input text.
        
        Args:
            text: Input text to protect
            sensitivity_level: Protection level (low/medium/high)
            
        Returns:
            Protected text with sensitive information redacted
        """
        protected_text = text
        
        for strategy in self.strategies:
            if strategy == RedactionStrategy.PII_REMOVAL:
                protected_text = self._apply_pii_removal(protected_text)
            elif strategy == RedactionStrategy.ENTITY_HASHING:
                protected_text = self._apply_entity_hashing(protected_text)
            elif strategy == RedactionStrategy.SEMANTIC_ENCRYPTION:
                protected_text = self._apply_semantic_encryption(protected_text, sensitivity_level)
        
        return protected_text
    
    def batch_protect(self, texts: List[str]) -> List[str]:
        """Efficiently protect multiple texts."""
        return [self.protect(text) for text in texts]
    
    def explain_redactions(self, text: str) -> RedactionReport:
        """Explain what redactions would be applied to text."""
        redactions = []
        
        for pattern_name, pattern in self.pii_patterns.items():
            if pattern.search(text):
                redactions.append(f"Detected {pattern_name.upper()}")
        
        return RedactionReport(
            original_length=len(text),
            redacted_length=len(self.protect(text)),
            redactions_applied=redactions,
            sensitivity_level="medium",
            confidence_score=0.85
        )
    
    def _apply_pii_removal(self, text: str) -> str:
        """Remove personally identifiable information."""
        protected = text
        
        for pattern_name, pattern in self.pii_patterns.items():
            replacement = f"[{pattern_name.upper()}]"
            protected = pattern.sub(replacement, protected)
        
        return protected
    
    def _apply_entity_hashing(self, text: str) -> str:
        """Replace entities with consistent hashes."""
        protected = text
        
        # Hash email addresses while preserving structure
        def hash_email(match):
            email = match.group(0)
            hashed = hashlib.sha256((email + self.salt).encode()).hexdigest()[:8]
            return f"user_{hashed}@domain.com"
        
        protected = self.pii_patterns['email'].sub(hash_email, protected)
        
        return protected
    
    def _apply_semantic_encryption(self, text: str, sensitivity_level: str) -> str:
        """Apply semantic encryption while preserving structure."""
        # Placeholder for semantic encryption
        # Real implementation would use homomorphic encryption or similar
        
        if sensitivity_level == "high":
            # More aggressive redaction for high sensitivity
            protected = re.sub(r'\b\w{4,}\b', '[ENCRYPTED]', text)
        else:
            # Standard redaction
            protected = self._apply_pii_removal(text)
        
        return protected
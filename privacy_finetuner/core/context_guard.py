"""Context protection for sensitive data in prompts and responses."""

from typing import List, Dict, Any
from enum import Enum
import re
import logging

logger = logging.getLogger(__name__)


class RedactionStrategy(Enum):
    """Supported redaction strategies for context protection."""
    PII_REMOVAL = "pii_removal"
    ENTITY_HASHING = "entity_hashing" 
    SEMANTIC_ENCRYPTION = "semantic_encryption"
    K_ANONYMIZATION = "k_anonymization"


class ContextGuard:
    """Privacy protection for context windows and user prompts.
    
    Implements multiple redaction strategies to protect sensitive information
    while preserving semantic meaning for model training and inference.
    """
    
    def __init__(self, strategies: List[RedactionStrategy]):
        """Initialize context guard with protection strategies.
        
        Args:
            strategies: List of redaction strategies to apply
        """
        self.strategies = strategies
        self._pii_patterns = self._load_pii_patterns()
        
        logger.info(f"Initialized ContextGuard with strategies: {[s.value for s in strategies]}")
    
    def protect(self, text: str, sensitivity_level: str = "medium") -> str:
        """Apply privacy protection to text input.
        
        Args:
            text: Input text to protect
            sensitivity_level: Protection level (low/medium/high)
            
        Returns:
            Protected text with redacted sensitive information
        """
        protected_text = text
        
        for strategy in self.strategies:
            if strategy == RedactionStrategy.PII_REMOVAL:
                protected_text = self._remove_pii(protected_text)
            elif strategy == RedactionStrategy.ENTITY_HASHING:
                protected_text = self._hash_entities(protected_text)
            elif strategy == RedactionStrategy.SEMANTIC_ENCRYPTION:
                protected_text = self._semantic_encrypt(protected_text, sensitivity_level)
        
        logger.debug(f"Protected text: {len(text)} -> {len(protected_text)} chars")
        return protected_text
    
    def batch_protect(self, texts: List[str]) -> List[str]:
        """Efficiently protect multiple texts."""
        return [self.protect(text) for text in texts]
    
    def explain_redactions(self, text: str) -> Dict[str, Any]:
        """Explain what was redacted and why."""
        redactions = []
        
        # Detect PII patterns
        for pattern_name, pattern in self._pii_patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                redactions.append({
                    "type": pattern_name,
                    "position": (match.start(), match.end()),
                    "reason": f"Detected {pattern_name} pattern"
                })
        
        return {
            "total_redactions": len(redactions),
            "redaction_details": redactions,
            "privacy_level": "high" if len(redactions) > 5 else "medium"
        }
    
    def _load_pii_patterns(self) -> Dict[str, str]:
        """Load PII detection patterns."""
        return {
            "credit_card": r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',
            "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phone": r'\b\d{3}[- ]?\d{3}[- ]?\d{4}\b'
        }
    
    def _remove_pii(self, text: str) -> str:
        """Remove personally identifiable information."""
        protected = text
        for pattern_name, pattern in self._pii_patterns.items():
            protected = re.sub(pattern, f'[{pattern_name.upper()}]', protected)
        return protected
    
    def _hash_entities(self, text: str) -> str:
        """Replace entities with consistent hashes."""
        import hashlib
        import spacy
        
        # Load spacy model for NER (in production, cache this)
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found, skipping entity hashing")
            return text
        
        doc = nlp(text)
        protected_text = text
        
        # Replace entities with consistent hashes
        for ent in reversed(doc.ents):  # Reverse to maintain positions
            if ent.label_ in ["PERSON", "ORG", "GPE", "LOC"]:
                # Create consistent hash for entity
                entity_hash = hashlib.sha256(
                    (ent.text + ent.label_).encode()
                ).hexdigest()[:8]
                
                replacement = f"[{ent.label_}_{entity_hash}]"
                protected_text = (
                    protected_text[:ent.start_char] + 
                    replacement + 
                    protected_text[ent.end_char:]
                )
        
        return protected_text
    
    def _semantic_encrypt(self, text: str, sensitivity_level: str) -> str:
        """Apply semantic encryption while preserving structure."""
        import hashlib
        from collections import defaultdict
        
        # Simple semantic encryption using word-level replacement
        words = text.split()
        word_mapping = defaultdict(str)
        
        # Sensitivity-based encryption strength
        hash_length = {
            "low": 4,
            "medium": 6, 
            "high": 8
        }.get(sensitivity_level, 6)
        
        encrypted_words = []
        for word in words:
            # Skip common words for structure preservation
            if word.lower() in ["the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for"]:
                encrypted_words.append(word)
                continue
            
            # Create consistent encryption for each unique word
            if word not in word_mapping:
                word_hash = hashlib.sha256(word.encode()).hexdigest()[:hash_length]
                word_mapping[word] = f"ENC_{word_hash}"
            
            encrypted_words.append(word_mapping[word])
        
        return " ".join(encrypted_words)
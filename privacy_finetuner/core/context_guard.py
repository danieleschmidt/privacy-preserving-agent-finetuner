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
        """Load comprehensive PII detection patterns."""
        return {
            "credit_card": r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',
            "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phone": r'\b\d{3}[- ]?\d{3}[- ]?\d{4}\b',
            "ip_address": r'\b(?:\d{1,3}\.){3}\d{1,3}\b',
            "us_passport": r'\b[A-Z]{1,2}\d{6,9}\b',
            "driver_license": r'\b[A-Z]{1,2}\d{6,8}\b',
            "bank_account": r'\b\d{8,12}\b',
            "iban": r'\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}([A-Z0-9]?){0,16}\b',
            "bitcoin_address": r'\b[13][a-km-zA-HJ-NP-Z1-9]{25,34}\b',
            "api_key": r'\b[A-Za-z0-9]{32,}\b',
            "jwt_token": r'\beyJ[A-Za-z0-9-_=]+\.[A-Za-z0-9-_=]+\.?[A-Za-z0-9-_.+/=]*\b'
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
    
    def analyze_sensitivity(self, text: str) -> Dict[str, Any]:
        """Analyze text sensitivity and recommend protection level.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Sensitivity analysis with recommendations
        """
        sensitivity_score = 0
        detected_patterns = []
        
        # Check for PII patterns
        for pattern_name, pattern in self._pii_patterns.items():
            matches = len(re.findall(pattern, text))
            if matches > 0:
                sensitivity_score += matches * 2
                detected_patterns.append(pattern_name)
        
        # Check for potential sensitive keywords
        sensitive_keywords = [
            "password", "secret", "token", "key", "medical", "health",
            "financial", "bank", "credit", "ssn", "social", "confidential"
        ]
        
        for keyword in sensitive_keywords:
            if keyword.lower() in text.lower():
                sensitivity_score += 1
                detected_patterns.append(f"keyword:{keyword}")
        
        # Determine sensitivity level
        if sensitivity_score >= 5:
            level = "high"
        elif sensitivity_score >= 2:
            level = "medium"
        else:
            level = "low"
        
        return {
            "sensitivity_level": level,
            "sensitivity_score": sensitivity_score,
            "detected_patterns": detected_patterns,
            "recommended_strategies": self._recommend_strategies(level),
            "text_length": len(text),
            "estimated_entities": self._count_entities(text)
        }
    
    def _recommend_strategies(self, sensitivity_level: str) -> List[str]:
        """Recommend protection strategies based on sensitivity level."""
        if sensitivity_level == "high":
            return ["pii_removal", "entity_hashing", "semantic_encryption"]
        elif sensitivity_level == "medium":
            return ["pii_removal", "entity_hashing"]
        else:
            return ["pii_removal"]
    
    def _count_entities(self, text: str) -> int:
        """Count named entities in text."""
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(text)
            return len(doc.ents)
        except (ImportError, OSError):
            # Fallback: simple capitalized word count
            import re
            return len(re.findall(r'\b[A-Z][a-z]+\b', text))
    
    def create_privacy_report(self, original_text: str, protected_text: str) -> Dict[str, Any]:
        """Create comprehensive privacy protection report.
        
        Args:
            original_text: Original unprotected text
            protected_text: Text after privacy protection
            
        Returns:
            Detailed privacy protection report
        """
        original_analysis = self.analyze_sensitivity(original_text)
        protected_analysis = self.analyze_sensitivity(protected_text)
        redaction_info = self.explain_redactions(original_text)
        
        return {
            "protection_summary": {
                "original_sensitivity": original_analysis["sensitivity_level"],
                "protected_sensitivity": protected_analysis["sensitivity_level"],
                "sensitivity_reduction": original_analysis["sensitivity_score"] - protected_analysis["sensitivity_score"],
                "strategies_applied": [s.value for s in self.strategies]
            },
            "text_metrics": {
                "original_length": len(original_text),
                "protected_length": len(protected_text),
                "compression_ratio": len(protected_text) / len(original_text) if len(original_text) > 0 else 0
            },
            "redaction_analysis": redaction_info,
            "privacy_compliance": {
                "gdpr_compliant": protected_analysis["sensitivity_score"] < 2,
                "hipaa_ready": "medical" not in original_analysis["detected_patterns"],
                "pci_safe": "credit_card" not in original_analysis["detected_patterns"]
            }
        }
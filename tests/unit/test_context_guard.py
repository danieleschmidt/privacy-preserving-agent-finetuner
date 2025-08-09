"""Unit tests for context guard privacy protection."""

import pytest
from privacy_finetuner.core.context_guard import ContextGuard, RedactionStrategy


class TestContextGuard:
    """Test context protection functionality."""
    
    def test_pii_removal(self):
        """Test PII removal redaction strategy."""
        guard = ContextGuard([RedactionStrategy.PII_REMOVAL])
        
        text = "Contact John at john@example.com or call 555-123-4567"
        protected = guard.protect(text)
        
        assert "john@example.com" not in protected
        assert "555-123-4567" not in protected
        assert "[EMAIL]" in protected
        assert "[PHONE]" in protected
    
    def test_credit_card_redaction(self):
        """Test credit card number redaction."""
        guard = ContextGuard([RedactionStrategy.PII_REMOVAL])
        
        text = "Process payment for card 4111-1111-1111-1111"
        protected = guard.protect(text)
        
        assert "4111-1111-1111-1111" not in protected
        assert "[CREDIT_CARD]" in protected
    
    def test_multiple_strategies(self):
        """Test applying multiple redaction strategies."""
        guard = ContextGuard([
            RedactionStrategy.PII_REMOVAL,
            RedactionStrategy.ENTITY_HASHING
        ])
        
        text = "Send email to john@example.com about account 123-45-6789"
        protected = guard.protect(text)
        
        # Should apply PII removal
        assert "[EMAIL]" in protected
        assert "[SSN]" in protected
    
    def test_batch_protection(self):
        """Test batch text protection."""
        guard = ContextGuard([RedactionStrategy.PII_REMOVAL])
        
        texts = [
            "Email: john@example.com",
            "Phone: 555-123-4567",
            "Regular text without PII"
        ]
        
        protected = guard.batch_protect(texts)
        
        assert len(protected) == 3
        assert "[EMAIL]" in protected[0]
        assert "[PHONE]" in protected[1]
        assert protected[2] == "Regular text without PII"  # Unchanged
    
    def test_redaction_explanation(self):
        """Test redaction explanation functionality."""
        guard = ContextGuard([RedactionStrategy.PII_REMOVAL])
        
        text = "Contact john@example.com or 555-123-4567"
        explanation = guard.explain_redactions(text)
        
        assert explanation["total_redactions"] == 2
        assert len(explanation["redaction_details"]) == 2
        
        # Check that email and phone redactions are detected
        redaction_types = [r["type"] for r in explanation["redaction_details"]]
        assert "email" in redaction_types
        assert "phone" in redaction_types
    
    def test_no_redactions_needed(self):
        """Test text with no sensitive information."""
        guard = ContextGuard([RedactionStrategy.PII_REMOVAL])
        
        text = "This is a normal text without any sensitive information."
        protected = guard.protect(text)
        explanation = guard.explain_redactions(text)
        
        assert protected == text  # Should be unchanged
        assert explanation["total_redactions"] == 0
    
    def test_sensitivity_levels(self):
        """Test different sensitivity levels."""
        guard = ContextGuard([RedactionStrategy.SEMANTIC_ENCRYPTION])
        
        text = "Sensitive business information"
        
        # Test different sensitivity levels (implementation pending)
        low_protection = guard.protect(text, "low")
        high_protection = guard.protect(text, "high")
        
        # The implementation may apply semantic encryption, so both may differ
        assert len(low_protection) > 0
        assert len(high_protection) > 0
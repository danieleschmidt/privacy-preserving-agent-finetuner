"""Custom exceptions for privacy-preserving training."""

from typing import Dict, Any, Optional


class PrivacyPreservingException(Exception):
    """Base exception for privacy-preserving training framework."""
    
    def __init__(self, message: str, error_code: str = None, context: Dict[str, Any] = None):
        super().__init__(message)
        self.error_code = error_code
        self.context = context or {}
        self.timestamp = None
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging."""
        return {
            "error_type": self.__class__.__name__,
            "message": str(self),
            "error_code": self.error_code,
            "context": self.context,
            "timestamp": self.timestamp
        }


class PrivacyBudgetExhaustedException(PrivacyPreservingException):
    """Raised when privacy budget is exhausted."""
    
    def __init__(
        self, 
        spent_epsilon: float, 
        total_epsilon: float,
        context: Dict[str, Any] = None
    ):
        message = f"Privacy budget exhausted: spent {spent_epsilon:.6f} of {total_epsilon:.6f} epsilon"
        super().__init__(message, "PRIVACY_BUDGET_EXHAUSTED", context)
        self.spent_epsilon = spent_epsilon
        self.total_epsilon = total_epsilon


class ModelTrainingException(PrivacyPreservingException):
    """Raised when model training fails."""
    
    def __init__(
        self, 
        message: str, 
        training_step: int = None,
        loss_value: float = None,
        context: Dict[str, Any] = None
    ):
        super().__init__(message, "MODEL_TRAINING_FAILED", context)
        self.training_step = training_step
        self.loss_value = loss_value


class DataValidationException(PrivacyPreservingException):
    """Raised when data validation fails."""
    
    def __init__(
        self, 
        message: str, 
        data_path: str = None,
        validation_errors: list = None,
        context: Dict[str, Any] = None
    ):
        super().__init__(message, "DATA_VALIDATION_FAILED", context)
        self.data_path = data_path
        self.validation_errors = validation_errors or []


class SecurityViolationException(PrivacyPreservingException):
    """Raised when security violation is detected."""
    
    def __init__(
        self, 
        message: str, 
        violation_type: str = None,
        severity: str = "HIGH",
        context: Dict[str, Any] = None
    ):
        super().__init__(message, "SECURITY_VIOLATION", context)
        self.violation_type = violation_type
        self.severity = severity


class PrivacyAnalysisException(PrivacyPreservingException):
    """Raised when privacy analysis fails."""
    
    def __init__(
        self, 
        message: str, 
        analysis_type: str = None,
        context: Dict[str, Any] = None
    ):
        super().__init__(message, "PRIVACY_ANALYSIS_FAILED", context)
        self.analysis_type = analysis_type


class QuantumOptimizationException(PrivacyPreservingException):
    """Raised when quantum optimization fails."""
    
    def __init__(
        self, 
        message: str, 
        quantum_state: str = None,
        context: Dict[str, Any] = None
    ):
        super().__init__(message, "QUANTUM_OPTIMIZATION_FAILED", context)
        self.quantum_state = quantum_state


class ConfigurationException(PrivacyPreservingException):
    """Raised when configuration is invalid."""
    
    def __init__(
        self, 
        message: str, 
        config_field: str = None,
        config_value: Any = None,
        context: Dict[str, Any] = None
    ):
        super().__init__(message, "CONFIGURATION_INVALID", context)
        self.config_field = config_field
        self.config_value = config_value


class DatabaseException(PrivacyPreservingException):
    """Raised when database operations fail."""
    
    def __init__(
        self, 
        message: str, 
        operation: str = None,
        table: str = None,
        context: Dict[str, Any] = None
    ):
        super().__init__(message, "DATABASE_OPERATION_FAILED", context)
        self.operation = operation
        self.table = table


class APIException(PrivacyPreservingException):
    """Raised when API operations fail."""
    
    def __init__(
        self, 
        message: str, 
        endpoint: str = None,
        status_code: int = None,
        context: Dict[str, Any] = None
    ):
        super().__init__(message, "API_OPERATION_FAILED", context)
        self.endpoint = endpoint
        self.status_code = status_code


class CacheException(PrivacyPreservingException):
    """Raised when cache operations fail."""
    
    def __init__(
        self, 
        message: str, 
        cache_key: str = None,
        operation: str = None,
        context: Dict[str, Any] = None
    ):
        super().__init__(message, "CACHE_OPERATION_FAILED", context)
        self.cache_key = cache_key
        self.operation = operation


class ComplianceException(PrivacyPreservingException):
    """Raised when compliance checks fail."""
    
    def __init__(
        self, 
        message: str, 
        regulation: str = None,
        compliance_rule: str = None,
        context: Dict[str, Any] = None
    ):
        super().__init__(message, "COMPLIANCE_CHECK_FAILED", context)
        self.regulation = regulation
        self.compliance_rule = compliance_rule


class ResourceExhaustedException(PrivacyPreservingException):
    """Raised when system resources are exhausted."""
    
    def __init__(
        self, 
        message: str, 
        resource_type: str = None,
        current_usage: float = None,
        limit: float = None,
        context: Dict[str, Any] = None
    ):
        super().__init__(message, "RESOURCE_EXHAUSTED", context)
        self.resource_type = resource_type
        self.current_usage = current_usage
        self.limit = limit


class IntegrationException(PrivacyPreservingException):
    """Raised when external integration fails."""
    
    def __init__(
        self, 
        message: str, 
        service: str = None,
        integration_type: str = None,
        context: Dict[str, Any] = None
    ):
        super().__init__(message, "INTEGRATION_FAILED", context)
        self.service = service
        self.integration_type = integration_type


class ValidationException(PrivacyPreservingException):
    """Raised when validation fails."""
    
    def __init__(
        self, 
        message: str, 
        field: str = None,
        value: Any = None,
        expected_type: str = None,
        context: Dict[str, Any] = None
    ):
        super().__init__(message, "VALIDATION_FAILED", context)
        self.field = field
        self.value = value
        self.expected_type = expected_type


# Enhanced exception hierarchy for robustness
class RecoverableException(PrivacyPreservingException):
    """Base class for recoverable exceptions."""
    
    def __init__(self, message: str, recovery_strategy: str = None, **kwargs):
        super().__init__(message, **kwargs)
        self.recovery_strategy = recovery_strategy
        self.is_recoverable = True


class TransientException(RecoverableException):
    """Exceptions that are temporary and should be retried."""
    
    def __init__(self, message: str, max_retries: int = 3, **kwargs):
        super().__init__(message, recovery_strategy="retry", **kwargs)
        self.max_retries = max_retries


class CriticalException(PrivacyPreservingException):
    """Critical exceptions that require immediate attention."""
    
    def __init__(self, message: str, component: str = None, **kwargs):
        super().__init__(message, **kwargs)
        self.component = component
        self.is_critical = True
        self.requires_immediate_action = True


# Memory optimization exception
class MemoryOptimizationException(PrivacyPreservingException):
    """Raised when memory optimization fails."""
    
    def __init__(
        self, 
        message: str, 
        current_usage: float = None,
        target_reduction: float = None,
        context: Dict[str, Any] = None
    ):
        super().__init__(message, "MEMORY_OPTIMIZATION_FAILED", context)
        self.current_usage = current_usage
        self.target_reduction = target_reduction


# Exception registry for enhanced error handling
class ExceptionRegistry:
    """Registry for exception handlers and recovery strategies."""
    
    def __init__(self):
        self._handlers = {}
        self._recovery_strategies = {}
    
    def register_handler(self, exception_type: type, handler_func):
        """Register an exception handler."""
        self._handlers[exception_type] = handler_func
    
    def register_recovery_strategy(self, exception_type: type, strategy_func):
        """Register a recovery strategy."""
        self._recovery_strategies[exception_type] = strategy_func
    
    def handle_exception(self, exception: Exception) -> bool:
        """Handle an exception using registered handlers."""
        handler = self._handlers.get(type(exception))
        if handler:
            return handler(exception)
        return False
    
    def recover_from_exception(self, exception: Exception) -> bool:
        """Attempt recovery from an exception."""
        strategy = self._recovery_strategies.get(type(exception))
        if strategy:
            return strategy(exception)
        return False


# Global exception registry
exception_registry = ExceptionRegistry()
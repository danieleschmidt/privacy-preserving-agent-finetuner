"""Comprehensive input validation and sanitization for privacy-preserving ML training.

This module provides robust validation, sanitization, and security checks for all inputs
to the privacy-finetuner system, including data, configurations, and API requests.
"""

import re
import json
import logging
import hashlib
import ipaddress
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, date
from decimal import Decimal, InvalidOperation
import uuid

from .exceptions import ValidationException, SecurityViolationException

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of validation with details about any issues."""
    is_valid: bool
    value: Any = None
    errors: List[str] = None
    warnings: List[str] = None
    sanitized: bool = False
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []


class SecurityValidator:
    """Security-focused validator for potentially malicious inputs."""
    
    # Patterns for detecting malicious content
    MALICIOUS_PATTERNS = [
        # Code injection patterns
        r'<script[^>]*>.*?</script>',
        r'javascript\s*:',
        r'vbscript\s*:',
        r'on\w+\s*=',
        r'eval\s*\(',
        r'exec\s*\(',
        r'__import__\s*\(',
        r'system\s*\(',
        r'subprocess\s*\.',
        
        # Path traversal patterns
        r'\.\./',
        r'\.\.\\',
        r'/etc/passwd',
        r'/windows/system32',
        
        # SQL injection patterns
        r'(union|select|insert|delete|update|drop)\s+',
        r';.*--',
        r"'.*or.*'.*=.*'",
        
        # Command injection patterns
        r'[;&|]+\s*(ls|cat|wget|curl|nc|rm|cp|mv)',
        r'\$\([^)]+\)',
        r'`[^`]+`',
        
        # Protocol handlers
        r'(file|ftp|ldap|dict)://',
        
        # Hex encoding
        r'\\x[0-9a-fA-F]{2}',
        r'%[0-9a-fA-F]{2}',
        
        # Base64 suspicious patterns
        r'[A-Za-z0-9+/]{50,}={0,2}',  # Long base64 strings
    ]
    
    # Compiled patterns for performance
    _compiled_patterns = None
    
    @classmethod
    def _get_compiled_patterns(cls):
        if cls._compiled_patterns is None:
            cls._compiled_patterns = [
                re.compile(pattern, re.IGNORECASE | re.MULTILINE | re.DOTALL)
                for pattern in cls.MALICIOUS_PATTERNS
            ]
        return cls._compiled_patterns
    
    @staticmethod
    def scan_for_malicious_content(text: str) -> List[str]:
        """Scan text for malicious patterns."""
        if not isinstance(text, str):
            return []
        
        threats_found = []
        patterns = SecurityValidator._get_compiled_patterns()
        
        for i, pattern in enumerate(patterns):
            if pattern.search(text):
                threats_found.append(f"Malicious pattern #{i+1}: {SecurityValidator.MALICIOUS_PATTERNS[i][:50]}...")
        
        return threats_found
    
    @staticmethod
    def sanitize_string(text: str, max_length: int = 10000, allow_html: bool = False) -> str:
        """Sanitize string by removing malicious content."""
        if not isinstance(text, str):
            return str(text)[:max_length]
        
        # Limit length first
        if len(text) > max_length:
            text = text[:max_length] + "...[truncated]"
        
        # Remove null bytes
        text = text.replace('\x00', '')
        
        if not allow_html:
            # Remove HTML/XML tags
            text = re.sub(r'<[^>]+>', '', text)
            
            # Remove script content
            text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
            text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove control characters except newlines and tabs
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text


class TypeValidator:
    """Comprehensive type validation with coercion support."""
    
    @staticmethod
    def validate_integer(
        value: Any,
        min_value: Optional[int] = None,
        max_value: Optional[int] = None,
        allow_coercion: bool = True
    ) -> ValidationResult:
        """Validate integer with optional bounds checking."""
        errors = []
        warnings = []
        
        # Type checking and coercion
        if isinstance(value, int):
            validated_value = value
        elif allow_coercion:
            try:
                if isinstance(value, str):
                    # Handle hex, binary, octal representations
                    value = value.strip()
                    if value.startswith('0x'):
                        validated_value = int(value, 16)
                    elif value.startswith('0b'):
                        validated_value = int(value, 2)
                    elif value.startswith('0o'):
                        validated_value = int(value, 8)
                    else:
                        validated_value = int(float(value))  # Handle "123.0" strings
                elif isinstance(value, float):
                    if value != int(value):
                        warnings.append(f"Float {value} truncated to integer {int(value)}")
                    validated_value = int(value)
                elif isinstance(value, bool):
                    validated_value = int(value)
                    warnings.append(f"Boolean {value} converted to integer {validated_value}")
                else:
                    validated_value = int(value)
            except (ValueError, TypeError, OverflowError) as e:
                errors.append(f"Cannot convert to integer: {e}")
                return ValidationResult(is_valid=False, errors=errors, warnings=warnings)
        else:
            errors.append(f"Expected integer, got {type(value).__name__}")
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)
        
        # Bounds checking
        if min_value is not None and validated_value < min_value:
            errors.append(f"Value {validated_value} is below minimum {min_value}")
        
        if max_value is not None and validated_value > max_value:
            errors.append(f"Value {validated_value} exceeds maximum {max_value}")
        
        # Security checks for extremely large values
        if abs(validated_value) > 10**15:
            warnings.append(f"Very large integer value: {validated_value}")
        
        is_valid = len(errors) == 0
        return ValidationResult(
            is_valid=is_valid,
            value=validated_value if is_valid else None,
            errors=errors,
            warnings=warnings,
            sanitized=allow_coercion and not isinstance(value, int)
        )
    
    @staticmethod
    def validate_float(
        value: Any,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        allow_nan: bool = False,
        allow_inf: bool = False,
        allow_coercion: bool = True
    ) -> ValidationResult:
        """Validate float with comprehensive checks."""
        errors = []
        warnings = []
        
        # Type checking and coercion
        if isinstance(value, float):
            validated_value = value
        elif allow_coercion:
            try:
                if isinstance(value, str):
                    value = value.strip().lower()
                    if value in ['nan', 'inf', '-inf', '+inf']:
                        if value == 'nan':
                            validated_value = float('nan')
                        elif value in ['inf', '+inf']:
                            validated_value = float('inf')
                        else:
                            validated_value = float('-inf')
                    else:
                        validated_value = float(value)
                else:
                    validated_value = float(value)
            except (ValueError, TypeError, OverflowError) as e:
                errors.append(f"Cannot convert to float: {e}")
                return ValidationResult(is_valid=False, errors=errors, warnings=warnings)
        else:
            errors.append(f"Expected float, got {type(value).__name__}")
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)
        
        # Special value checks
        import math
        if math.isnan(validated_value):
            if not allow_nan:
                errors.append("NaN values are not allowed")
        elif math.isinf(validated_value):
            if not allow_inf:
                errors.append("Infinite values are not allowed")
        else:
            # Bounds checking for finite values
            if min_value is not None and validated_value < min_value:
                errors.append(f"Value {validated_value} is below minimum {min_value}")
            
            if max_value is not None and validated_value > max_value:
                errors.append(f"Value {validated_value} exceeds maximum {max_value}")
        
        # Precision warnings
        if abs(validated_value) > 1e100:
            warnings.append(f"Very large float value: {validated_value}")
        elif abs(validated_value) < 1e-100 and validated_value != 0:
            warnings.append(f"Very small float value: {validated_value}")
        
        is_valid = len(errors) == 0
        return ValidationResult(
            is_valid=is_valid,
            value=validated_value if is_valid else None,
            errors=errors,
            warnings=warnings,
            sanitized=allow_coercion and not isinstance(value, float)
        )
    
    @staticmethod
    def validate_string(
        value: Any,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        pattern: Optional[str] = None,
        allowed_chars: Optional[str] = None,
        forbidden_chars: Optional[str] = None,
        allow_coercion: bool = True,
        sanitize: bool = True,
        check_security: bool = True
    ) -> ValidationResult:
        """Comprehensive string validation with security checks."""
        errors = []
        warnings = []
        sanitized = False
        
        # Type checking and coercion
        if isinstance(value, str):
            validated_value = value
        elif allow_coercion:
            validated_value = str(value)
            warnings.append(f"Converted {type(value).__name__} to string")
            sanitized = True
        else:
            errors.append(f"Expected string, got {type(value).__name__}")
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)
        
        # Security scanning
        if check_security:
            threats = SecurityValidator.scan_for_malicious_content(validated_value)
            if threats:
                errors.extend(threats)
                # Don't continue validation if security threats found
                return ValidationResult(is_valid=False, errors=errors, warnings=warnings)
        
        # Sanitization
        if sanitize:
            original_value = validated_value
            validated_value = SecurityValidator.sanitize_string(validated_value, max_length or 10000)
            if validated_value != original_value:
                sanitized = True
                warnings.append("String was sanitized for security")
        
        # Length validation
        if min_length is not None and len(validated_value) < min_length:
            errors.append(f"String length {len(validated_value)} is below minimum {min_length}")
        
        if max_length is not None and len(validated_value) > max_length:
            errors.append(f"String length {len(validated_value)} exceeds maximum {max_length}")
        
        # Pattern validation
        if pattern is not None:
            try:
                if not re.match(pattern, validated_value):
                    errors.append(f"String does not match required pattern: {pattern}")
            except re.error as e:
                errors.append(f"Invalid regex pattern: {e}")
        
        # Character validation
        if allowed_chars is not None:
            invalid_chars = set(validated_value) - set(allowed_chars)
            if invalid_chars:
                errors.append(f"String contains forbidden characters: {sorted(invalid_chars)}")
        
        if forbidden_chars is not None:
            found_chars = set(validated_value) & set(forbidden_chars)
            if found_chars:
                errors.append(f"String contains forbidden characters: {sorted(found_chars)}")
        
        # Encoding validation
        try:
            validated_value.encode('utf-8')
        except UnicodeEncodeError as e:
            errors.append(f"String contains invalid Unicode characters: {e}")
        
        is_valid = len(errors) == 0
        return ValidationResult(
            is_valid=is_valid,
            value=validated_value if is_valid else None,
            errors=errors,
            warnings=warnings,
            sanitized=sanitized
        )
    
    @staticmethod
    def validate_path(
        value: Any,
        must_exist: bool = False,
        must_be_file: bool = False,
        must_be_dir: bool = False,
        allowed_extensions: Optional[List[str]] = None,
        max_size: Optional[int] = None,
        check_security: bool = True
    ) -> ValidationResult:
        """Validate file system paths with security checks."""
        errors = []
        warnings = []
        
        # Convert to Path object
        try:
            if isinstance(value, Path):
                path_obj = value
            else:
                path_obj = Path(str(value))
        except Exception as e:
            errors.append(f"Invalid path format: {e}")
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)
        
        # Security checks
        if check_security:
            path_str = str(path_obj)
            
            # Check for path traversal
            if '..' in path_obj.parts:
                errors.append("Path contains directory traversal sequences (..)") 
            
            # Check for suspicious paths
            suspicious_paths = [
                '/etc/', '/sys/', '/proc/', '/dev/',
                'c:\\windows\\', 'c:\\program files\\',
                '/root/', '/home/*/.ssh/'
            ]
            
            path_lower = path_str.lower()
            for suspicious in suspicious_paths:
                if suspicious.replace('*', '') in path_lower:
                    errors.append(f"Path accesses sensitive system directory: {suspicious}")
            
            # Check for unusual characters
            unusual_chars = set(path_str) & set('\x00\x01\x02\x03\x04\x05\x06\x07\x08\x0b\x0c\x0e\x0f')
            if unusual_chars:
                errors.append(f"Path contains unusual characters: {unusual_chars}")
        
        # Existence checks
        if must_exist and not path_obj.exists():
            errors.append(f"Path does not exist: {path_obj}")
        
        if path_obj.exists():
            if must_be_file and not path_obj.is_file():
                errors.append(f"Path is not a file: {path_obj}")
            
            if must_be_dir and not path_obj.is_dir():
                errors.append(f"Path is not a directory: {path_obj}")
            
            # Size checks
            if max_size is not None and path_obj.is_file():
                try:
                    file_size = path_obj.stat().st_size
                    if file_size > max_size:
                        errors.append(f"File size {file_size} exceeds maximum {max_size}")
                except OSError as e:
                    warnings.append(f"Could not check file size: {e}")
        
        # Extension validation
        if allowed_extensions is not None:
            extension = path_obj.suffix.lower()
            if extension not in [ext.lower() for ext in allowed_extensions]:
                errors.append(f"File extension '{extension}' not in allowed list: {allowed_extensions}")
        
        is_valid = len(errors) == 0
        return ValidationResult(
            is_valid=is_valid,
            value=path_obj if is_valid else None,
            errors=errors,
            warnings=warnings
        )
    
    @staticmethod
    def validate_email(value: Any) -> ValidationResult:
        """Validate email address format."""
        errors = []
        warnings = []
        
        # Convert to string
        if not isinstance(value, str):
            value = str(value)
        
        value = value.strip().lower()
        
        # Basic regex validation
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_pattern, value):
            errors.append("Invalid email format")
        
        # Length checks
        if len(value) > 254:  # RFC 5321 limit
            errors.append("Email address too long")
        
        # Domain validation
        if '@' in value:
            local, domain = value.rsplit('@', 1)
            
            if len(local) > 64:  # RFC 5321 limit
                errors.append("Email local part too long")
            
            if len(domain) > 253:  # RFC 1035 limit
                errors.append("Email domain too long")
            
            # Check for suspicious domains
            suspicious_domains = ['temp-mail', 'guerrillamail', '10minutemail']
            if any(susp in domain for susp in suspicious_domains):
                warnings.append("Email appears to use temporary/disposable service")
        
        is_valid = len(errors) == 0
        return ValidationResult(
            is_valid=is_valid,
            value=value if is_valid else None,
            errors=errors,
            warnings=warnings,
            sanitized=True
        )
    
    @staticmethod
    def validate_url(value: Any, allowed_schemes: Optional[List[str]] = None) -> ValidationResult:
        """Validate URL format and security."""
        errors = []
        warnings = []
        
        if not isinstance(value, str):
            value = str(value)
        
        value = value.strip()
        
        # Basic URL pattern
        url_pattern = r'^https?://[^\s/$.?#].[^\s]*$'
        if not re.match(url_pattern, value):
            errors.append("Invalid URL format")
        
        # Parse URL components
        try:
            from urllib.parse import urlparse
            parsed = urlparse(value)
            
            # Scheme validation
            if allowed_schemes is not None and parsed.scheme not in allowed_schemes:
                errors.append(f"URL scheme '{parsed.scheme}' not in allowed list: {allowed_schemes}")
            
            # Security checks
            if parsed.hostname:
                # Check for localhost/internal IPs
                if parsed.hostname in ['localhost', '127.0.0.1', '::1']:
                    warnings.append("URL points to localhost")
                
                # Check for private IP ranges
                try:
                    ip = ipaddress.ip_address(parsed.hostname)
                    if ip.is_private:
                        warnings.append("URL points to private IP address")
                except ValueError:
                    pass  # Not an IP address
                
                # Check for suspicious domains
                suspicious_domains = ['bit.ly', 'tinyurl', 'shortened']
                if any(susp in parsed.hostname for susp in suspicious_domains):
                    warnings.append("URL uses URL shortening service")
        
        except Exception as e:
            errors.append(f"URL parsing failed: {e}")
        
        is_valid = len(errors) == 0
        return ValidationResult(
            is_valid=is_valid,
            value=value if is_valid else None,
            errors=errors,
            warnings=warnings
        )


class ConfigurationValidator:
    """Validator for system configuration objects."""
    
    @staticmethod
    def validate_privacy_config(config: Dict[str, Any]) -> ValidationResult:
        """Validate privacy configuration parameters."""
        errors = []
        warnings = []
        sanitized_config = {}
        
        # Required fields
        required_fields = ['epsilon', 'delta']
        for field in required_fields:
            if field not in config:
                errors.append(f"Missing required field: {field}")
        
        # Epsilon validation
        if 'epsilon' in config:
            epsilon_result = TypeValidator.validate_float(
                config['epsilon'],
                min_value=0.001,
                max_value=100.0,
                allow_nan=False,
                allow_inf=False
            )
            if not epsilon_result.is_valid:
                errors.extend([f"epsilon: {error}" for error in epsilon_result.errors])
            else:
                sanitized_config['epsilon'] = epsilon_result.value
                if epsilon_result.warnings:
                    warnings.extend([f"epsilon: {warn}" for warn in epsilon_result.warnings])
                
                # Privacy-specific warnings
                if epsilon_result.value > 10:
                    warnings.append("epsilon: Very high value provides weak privacy guarantees")
                elif epsilon_result.value < 0.1:
                    warnings.append("epsilon: Very low value may severely impact model utility")
        
        # Delta validation
        if 'delta' in config:
            delta_result = TypeValidator.validate_float(
                config['delta'],
                min_value=1e-10,
                max_value=0.1,
                allow_nan=False,
                allow_inf=False
            )
            if not delta_result.is_valid:
                errors.extend([f"delta: {error}" for error in delta_result.errors])
            else:
                sanitized_config['delta'] = delta_result.value
                if delta_result.warnings:
                    warnings.extend([f"delta: {warn}" for warn in delta_result.warnings])
                
                # Privacy-specific warnings
                if delta_result.value > 1e-3:
                    warnings.append("delta: Large value may not provide strong privacy guarantees")
        
        # Optional fields with validation
        optional_fields = {
            'max_grad_norm': (1e-6, 100.0),
            'noise_multiplier': (0.01, 10.0),
            'learning_rate': (1e-8, 1.0)
        }
        
        for field, (min_val, max_val) in optional_fields.items():
            if field in config:
                result = TypeValidator.validate_float(
                    config[field],
                    min_value=min_val,
                    max_value=max_val,
                    allow_nan=False,
                    allow_inf=False
                )
                if not result.is_valid:
                    errors.extend([f"{field}: {error}" for error in result.errors])
                else:
                    sanitized_config[field] = result.value
                    if result.warnings:
                        warnings.extend([f"{field}: {warn}" for warn in result.warnings])
        
        # Cross-field validation
        if 'epsilon' in sanitized_config and 'delta' in sanitized_config:
            # Check if epsilon-delta combination makes sense
            if sanitized_config['epsilon'] < 1.0 and sanitized_config['delta'] > 1e-4:
                warnings.append("Low epsilon with high delta may not provide expected privacy level")
        
        is_valid = len(errors) == 0
        return ValidationResult(
            is_valid=is_valid,
            value=sanitized_config if is_valid else None,
            errors=errors,
            warnings=warnings,
            sanitized=True
        )
    
    @staticmethod
    def validate_training_config(config: Dict[str, Any]) -> ValidationResult:
        """Validate training configuration parameters."""
        errors = []
        warnings = []
        sanitized_config = {}
        
        # Validate required training parameters
        required_params = {
            'epochs': (1, 10000),
            'batch_size': (1, 10000),
            'learning_rate': (1e-8, 1.0)
        }
        
        for param, (min_val, max_val) in required_params.items():
            if param not in config:
                errors.append(f"Missing required parameter: {param}")
                continue
            
            if param in ['epochs', 'batch_size']:
                result = TypeValidator.validate_integer(
                    config[param],
                    min_value=min_val,
                    max_value=max_val
                )
            else:
                result = TypeValidator.validate_float(
                    config[param],
                    min_value=min_val,
                    max_value=max_val,
                    allow_nan=False,
                    allow_inf=False
                )
            
            if not result.is_valid:
                errors.extend([f"{param}: {error}" for error in result.errors])
            else:
                sanitized_config[param] = result.value
                if result.warnings:
                    warnings.extend([f"{param}: {warn}" for warn in result.warnings])
        
        # Validate optional parameters
        optional_params = {
            'weight_decay': (0.0, 1.0),
            'warmup_steps': (0, 10000),
            'max_grad_norm': (0.1, 100.0)
        }
        
        for param, (min_val, max_val) in optional_params.items():
            if param in config:
                if param == 'warmup_steps':
                    result = TypeValidator.validate_integer(
                        config[param],
                        min_value=min_val,
                        max_value=max_val
                    )
                else:
                    result = TypeValidator.validate_float(
                        config[param],
                        min_value=min_val,
                        max_value=max_val,
                        allow_nan=False,
                        allow_inf=False
                    )
                
                if not result.is_valid:
                    errors.extend([f"{param}: {error}" for error in result.errors])
                else:
                    sanitized_config[param] = result.value
                    if result.warnings:
                        warnings.extend([f"{param}: {warn}" for warn in result.warnings])
        
        # Validate dataset path if provided
        if 'dataset_path' in config:
            path_result = TypeValidator.validate_path(
                config['dataset_path'],
                must_exist=True,
                must_be_file=True,
                allowed_extensions=['.jsonl', '.json', '.csv', '.txt'],
                max_size=10 * 1024 * 1024 * 1024  # 10GB limit
            )
            
            if not path_result.is_valid:
                errors.extend([f"dataset_path: {error}" for error in path_result.errors])
            else:
                sanitized_config['dataset_path'] = str(path_result.value)
                if path_result.warnings:
                    warnings.extend([f"dataset_path: {warn}" for warn in path_result.warnings])
        
        # Cross-parameter validation
        if 'batch_size' in sanitized_config and 'epochs' in sanitized_config:
            total_updates = sanitized_config['epochs'] * 1000  # Rough estimate
            if total_updates > 100000:
                warnings.append("Very long training may lead to overfitting")
        
        is_valid = len(errors) == 0
        return ValidationResult(
            is_valid=is_valid,
            value=sanitized_config if is_valid else None,
            errors=errors,
            warnings=warnings,
            sanitized=True
        )


class DataValidator:
    """Validator for training data and datasets."""
    
    @staticmethod
    def validate_data_sample(sample: Dict[str, Any]) -> ValidationResult:
        """Validate individual training data sample."""
        errors = []
        warnings = []
        sanitized_sample = {}
        
        # Check if sample is a dictionary
        if not isinstance(sample, dict):
            errors.append(f"Sample must be a dictionary, got {type(sample).__name__}")
            return ValidationResult(is_valid=False, errors=errors)
        
        # Check required fields
        required_fields = ['text', 'prompt']
        has_required_field = any(field in sample for field in required_fields)
        
        if not has_required_field:
            errors.append(f"Sample must contain at least one of: {required_fields}")
        
        # Validate text fields
        text_fields = ['text', 'prompt', 'response', 'instruction']
        for field in text_fields:
            if field in sample:
                text_result = TypeValidator.validate_string(
                    sample[field],
                    min_length=1,
                    max_length=100000,  # 100KB limit
                    sanitize=True,
                    check_security=True
                )
                
                if not text_result.is_valid:
                    errors.extend([f"{field}: {error}" for error in text_result.errors])
                else:
                    sanitized_sample[field] = text_result.value
                    if text_result.warnings:
                        warnings.extend([f"{field}: {warn}" for warn in text_result.warnings])
        
        # Validate metadata fields
        metadata_fields = ['label', 'category', 'source']
        for field in metadata_fields:
            if field in sample:
                if isinstance(sample[field], str):
                    meta_result = TypeValidator.validate_string(
                        sample[field],
                        max_length=1000,
                        sanitize=True
                    )
                    if meta_result.is_valid:
                        sanitized_sample[field] = meta_result.value
                    else:
                        warnings.extend([f"{field}: {error}" for error in meta_result.errors])
                else:
                    sanitized_sample[field] = str(sample[field])[:1000]
        
        # Copy other fields with basic sanitization
        for key, value in sample.items():
            if key not in sanitized_sample:
                if isinstance(value, (str, int, float, bool)):
                    sanitized_sample[key] = value
                else:
                    sanitized_sample[key] = str(value)[:1000]
        
        is_valid = len(errors) == 0
        return ValidationResult(
            is_valid=is_valid,
            value=sanitized_sample if is_valid else None,
            errors=errors,
            warnings=warnings,
            sanitized=True
        )
    
    @staticmethod
    def validate_dataset_file(file_path: str) -> ValidationResult:
        """Validate dataset file format and content."""
        errors = []
        warnings = []
        
        # Validate path
        path_result = TypeValidator.validate_path(
            file_path,
            must_exist=True,
            must_be_file=True,
            allowed_extensions=['.jsonl', '.json', '.csv', '.txt'],
            max_size=50 * 1024 * 1024 * 1024  # 50GB limit
        )
        
        if not path_result.is_valid:
            return path_result
        
        file_path_obj = path_result.value
        
        # Check file content
        try:
            sample_count = 0
            valid_samples = 0
            
            with open(file_path_obj, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    if line_num > 1000:  # Sample first 1000 lines
                        break
                    
                    line = line.strip()
                    if not line:
                        continue
                    
                    sample_count += 1
                    
                    try:
                        if file_path_obj.suffix.lower() == '.jsonl':
                            sample = json.loads(line)
                        else:
                            # For other formats, treat as simple text
                            sample = {'text': line}
                        
                        sample_result = DataValidator.validate_data_sample(sample)
                        if sample_result.is_valid:
                            valid_samples += 1
                        elif line_num <= 10:  # Report errors for first 10 samples
                            warnings.extend([f"Line {line_num}: {error}" for error in sample_result.errors])
                    
                    except json.JSONDecodeError as e:
                        if line_num <= 10:
                            warnings.append(f"Line {line_num}: JSON decode error - {e}")
            
            # Validate sample statistics
            if sample_count == 0:
                errors.append("Dataset file contains no samples")
            elif valid_samples == 0:
                errors.append("Dataset file contains no valid samples")
            elif valid_samples < sample_count * 0.8:  # Less than 80% valid
                warnings.append(f"Dataset has low validity rate: {valid_samples}/{sample_count} samples valid")
            
            if sample_count < 10:
                warnings.append(f"Dataset has very few samples: {sample_count}")
        
        except Exception as e:
            errors.append(f"Error reading dataset file: {e}")
        
        is_valid = len(errors) == 0
        return ValidationResult(
            is_valid=is_valid,
            value=str(file_path_obj) if is_valid else None,
            errors=errors,
            warnings=warnings
        )


class ComprehensiveValidator:
    """Main validator class that combines all validation capabilities."""
    
    def __init__(self, strict_mode: bool = False, security_enabled: bool = True):
        """Initialize validator with configuration options.
        
        Args:
            strict_mode: If True, warnings are treated as errors
            security_enabled: If True, security checks are performed
        """
        self.strict_mode = strict_mode
        self.security_enabled = security_enabled
        self.validation_history = []
    
    def validate(self, value: Any, validator_type: str, **kwargs) -> ValidationResult:
        """Main validation entry point.
        
        Args:
            value: Value to validate
            validator_type: Type of validation to perform
            **kwargs: Additional validation parameters
            
        Returns:
            ValidationResult with validation outcome
        """
        start_time = datetime.now()
        
        try:
            # Route to appropriate validator
            if validator_type == 'integer':
                result = TypeValidator.validate_integer(value, **kwargs)
            elif validator_type == 'float':
                result = TypeValidator.validate_float(value, **kwargs)
            elif validator_type == 'string':
                kwargs['check_security'] = kwargs.get('check_security', self.security_enabled)
                result = TypeValidator.validate_string(value, **kwargs)
            elif validator_type == 'path':
                kwargs['check_security'] = kwargs.get('check_security', self.security_enabled)
                result = TypeValidator.validate_path(value, **kwargs)
            elif validator_type == 'email':
                result = TypeValidator.validate_email(value)
            elif validator_type == 'url':
                result = TypeValidator.validate_url(value, **kwargs)
            elif validator_type == 'privacy_config':
                result = ConfigurationValidator.validate_privacy_config(value)
            elif validator_type == 'training_config':
                result = ConfigurationValidator.validate_training_config(value)
            elif validator_type == 'data_sample':
                result = DataValidator.validate_data_sample(value)
            elif validator_type == 'dataset_file':
                result = DataValidator.validate_dataset_file(value)
            else:
                result = ValidationResult(
                    is_valid=False,
                    errors=[f"Unknown validator type: {validator_type}"]
                )
            
            # Apply strict mode
            if self.strict_mode and result.warnings:
                result.errors.extend(result.warnings)
                result.warnings = []
                result.is_valid = False
            
            # Record validation history
            validation_record = {
                'timestamp': start_time,
                'validator_type': validator_type,
                'value_type': type(value).__name__,
                'is_valid': result.is_valid,
                'errors_count': len(result.errors),
                'warnings_count': len(result.warnings),
                'duration': (datetime.now() - start_time).total_seconds()
            }
            
            self.validation_history.append(validation_record)
            
            # Keep history manageable
            if len(self.validation_history) > 1000:
                self.validation_history.pop(0)
            
            return result
        
        except Exception as e:
            logger.error(f"Validation error for type {validator_type}: {e}")
            return ValidationResult(
                is_valid=False,
                errors=[f"Validation failed with exception: {e}"]
            )
    
    def validate_batch(self, items: List[Tuple[Any, str]], **kwargs) -> Dict[str, ValidationResult]:
        """Validate multiple items at once.
        
        Args:
            items: List of (value, validator_type) tuples
            **kwargs: Common validation parameters
            
        Returns:
            Dictionary mapping item indices to validation results
        """
        results = {}
        
        for i, (value, validator_type) in enumerate(items):
            results[i] = self.validate(value, validator_type, **kwargs)
        
        return results
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        if not self.validation_history:
            return {'status': 'no_validations_performed'}
        
        total_validations = len(self.validation_history)
        successful_validations = sum(1 for v in self.validation_history if v['is_valid'])
        
        validation_types = {}
        for v in self.validation_history:
            vtype = v['validator_type']
            if vtype not in validation_types:
                validation_types[vtype] = {'total': 0, 'successful': 0}
            validation_types[vtype]['total'] += 1
            if v['is_valid']:
                validation_types[vtype]['successful'] += 1
        
        avg_duration = sum(v['duration'] for v in self.validation_history) / total_validations
        
        return {
            'total_validations': total_validations,
            'successful_validations': successful_validations,
            'success_rate': successful_validations / total_validations,
            'validation_types': validation_types,
            'average_duration': avg_duration,
            'strict_mode': self.strict_mode,
            'security_enabled': self.security_enabled
        }


# Global validator instance
default_validator = ComprehensiveValidator(
    strict_mode=False,
    security_enabled=True
)


def validate(value: Any, validator_type: str, **kwargs) -> ValidationResult:
    """Convenience function for validation using default validator."""
    return default_validator.validate(value, validator_type, **kwargs)


def validate_and_raise(value: Any, validator_type: str, **kwargs) -> Any:
    """Validate and raise exception if invalid, otherwise return sanitized value."""
    result = default_validator.validate(value, validator_type, **kwargs)
    
    if not result.is_valid:
        raise ValidationException(
            f"Validation failed for {validator_type}: " + "; ".join(result.errors),
            field=validator_type,
            value=value,
            context={'errors': result.errors, 'warnings': result.warnings}
        )
    
    # Log warnings
    for warning in result.warnings:
        logger.warning(f"Validation warning for {validator_type}: {warning}")
    
    return result.value
"""Cache strategies for different use cases."""

import time
import logging
from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, List, Tuple
from collections import OrderedDict
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class CacheStrategy(ABC):
    """Abstract base class for cache strategies."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass
    
    @abstractmethod
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all cached values."""
        pass
    
    @abstractmethod
    def size(self) -> int:
        """Get cache size."""
        pass


class TTLCache(CacheStrategy):
    """Time-To-Live cache implementation."""
    
    def __init__(self, default_ttl: int = 3600):
        """Initialize TTL cache.
        
        Args:
            default_ttl: Default time to live in seconds
        """
        self.default_ttl = default_ttl
        self.cache: Dict[str, Dict[str, Any]] = {}
    
    def _is_expired(self, entry: Dict[str, Any]) -> bool:
        """Check if cache entry is expired."""
        return entry["expires_at"] <= time.time()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        if key not in self.cache:
            return None
        
        entry = self.cache[key]
        if self._is_expired(entry):
            del self.cache[key]
            return None
        
        entry["last_accessed"] = time.time()
        return entry["value"]
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with TTL."""
        ttl = ttl or self.default_ttl
        current_time = time.time()
        
        self.cache[key] = {
            "value": value,
            "created_at": current_time,
            "last_accessed": current_time,
            "expires_at": current_time + ttl,
            "ttl": ttl
        }
        return True
    
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        if key in self.cache:
            del self.cache[key]
            return True
        return False
    
    def clear(self) -> None:
        """Clear all cached values."""
        self.cache.clear()
    
    def size(self) -> int:
        """Get cache size after cleaning expired entries."""
        self._cleanup_expired()
        return len(self.cache)
    
    def _cleanup_expired(self) -> int:
        """Remove expired entries from cache."""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self.cache.items()
            if entry["expires_at"] <= current_time
        ]
        
        for key in expired_keys:
            del self.cache[key]
        
        return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        self._cleanup_expired()
        
        current_time = time.time()
        total_entries = len(self.cache)
        
        if total_entries == 0:
            return {
                "total_entries": 0,
                "average_age": 0,
                "oldest_entry": None,
                "newest_entry": None
            }
        
        ages = [current_time - entry["created_at"] for entry in self.cache.values()]
        last_accessed = [entry["last_accessed"] for entry in self.cache.values()]
        
        return {
            "total_entries": total_entries,
            "average_age": sum(ages) / total_entries,
            "oldest_entry": current_time - min(entry["created_at"] for entry in self.cache.values()),
            "newest_entry": current_time - max(entry["created_at"] for entry in self.cache.values()),
            "last_cleanup": current_time
        }


class LRUCache(CacheStrategy):
    """Least Recently Used cache implementation."""
    
    def __init__(self, max_size: int = 1000):
        """Initialize LRU cache.
        
        Args:
            max_size: Maximum number of entries to keep
        """
        self.max_size = max_size
        self.cache = OrderedDict()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value and move to end (most recently used)."""
        if key not in self.cache:
            return None
        
        # Move to end (most recently used)
        value = self.cache.pop(key)
        self.cache[key] = value
        return value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache, evicting LRU if necessary."""
        if key in self.cache:
            # Update existing key
            self.cache.pop(key)
        elif len(self.cache) >= self.max_size:
            # Remove least recently used
            self.cache.popitem(last=False)
        
        self.cache[key] = value
        return True
    
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        if key in self.cache:
            del self.cache[key]
            return True
        return False
    
    def clear(self) -> None:
        """Clear all cached values."""
        self.cache.clear()
    
    def size(self) -> int:
        """Get cache size."""
        return len(self.cache)
    
    def get_lru_keys(self, count: int = 10) -> List[str]:
        """Get least recently used keys."""
        return list(self.cache.keys())[:count]
    
    def get_mru_keys(self, count: int = 10) -> List[str]:
        """Get most recently used keys."""
        return list(reversed(list(self.cache.keys())))[:count]


class ModelCache(CacheStrategy):
    """Specialized cache for ML models and related data."""
    
    def __init__(self, max_models: int = 5, model_ttl: int = 7200):
        """Initialize model cache.
        
        Args:
            max_models: Maximum number of models to keep in memory
            model_ttl: Time to live for models in seconds
        """
        self.max_models = max_models
        self.model_ttl = model_ttl
        self.models: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.metadata_cache = TTLCache(default_ttl=3600)  # Faster TTL for metadata
    
    def get(self, key: str) -> Optional[Any]:
        """Get model or metadata from cache."""
        # Check if it's a model
        if key.startswith("model:"):
            return self._get_model(key)
        else:
            return self.metadata_cache.get(key)
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set model or metadata in cache."""
        if key.startswith("model:"):
            return self._set_model(key, value)
        else:
            return self.metadata_cache.set(key, value, ttl)
    
    def delete(self, key: str) -> bool:
        """Delete model or metadata from cache."""
        if key.startswith("model:"):
            return self._delete_model(key)
        else:
            return self.metadata_cache.delete(key)
    
    def clear(self) -> None:
        """Clear all cached data."""
        self.models.clear()
        self.metadata_cache.clear()
    
    def size(self) -> int:
        """Get total cache size."""
        return len(self.models) + self.metadata_cache.size()
    
    def _get_model(self, key: str) -> Optional[Any]:
        """Get model from cache."""
        if key not in self.models:
            return None
        
        entry = self.models[key]
        current_time = time.time()
        
        # Check if expired
        if entry["expires_at"] <= current_time:
            del self.models[key]
            logger.info(f"Model cache entry expired and removed: {key}")
            return None
        
        # Move to end (most recently used)
        self.models.move_to_end(key)
        entry["last_accessed"] = current_time
        entry["access_count"] += 1
        
        logger.debug(f"Model cache hit: {key}")
        return entry["model"]
    
    def _set_model(self, key: str, model: Any) -> bool:
        """Set model in cache with LRU eviction."""
        current_time = time.time()
        
        # Remove existing entry if present
        if key in self.models:
            del self.models[key]
        
        # Evict least recently used if at capacity
        while len(self.models) >= self.max_models:
            evicted_key, evicted_entry = self.models.popitem(last=False)
            logger.info(f"Evicted model from cache: {evicted_key}")
        
        # Add new model
        self.models[key] = {
            "model": model,
            "created_at": current_time,
            "last_accessed": current_time,
            "expires_at": current_time + self.model_ttl,
            "access_count": 0,
            "size_estimate": self._estimate_model_size(model)
        }
        
        logger.info(f"Model cached: {key}")
        return True
    
    def _delete_model(self, key: str) -> bool:
        """Delete model from cache."""
        if key in self.models:
            del self.models[key]
            logger.info(f"Model removed from cache: {key}")
            return True
        return False
    
    def _estimate_model_size(self, model: Any) -> int:
        """Estimate model size in bytes."""
        try:
            import sys
            return sys.getsizeof(model)
        except Exception:
            return 0
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get model cache statistics."""
        current_time = time.time()
        
        if not self.models:
            return {
                "total_models": 0,
                "total_size_estimate": 0,
                "average_age": 0,
                "total_accesses": 0
            }
        
        total_accesses = sum(entry["access_count"] for entry in self.models.values())
        total_size = sum(entry["size_estimate"] for entry in self.models.values())
        ages = [current_time - entry["created_at"] for entry in self.models.values()]
        
        return {
            "total_models": len(self.models),
            "total_size_estimate": total_size,
            "average_age": sum(ages) / len(ages),
            "total_accesses": total_accesses,
            "most_accessed": max(
                self.models.items(),
                key=lambda x: x[1]["access_count"],
                default=(None, {"access_count": 0})
            )[0],
            "cache_utilization": len(self.models) / self.max_models
        }
    
    def preload_model(self, key: str, model_loader: callable) -> bool:
        """Preload a model using a loader function."""
        try:
            model = model_loader()
            return self._set_model(key, model)
        except Exception as e:
            logger.error(f"Failed to preload model {key}: {e}")
            return False
    
    def evict_expired(self) -> int:
        """Manually evict expired models."""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self.models.items()
            if entry["expires_at"] <= current_time
        ]
        
        for key in expired_keys:
            del self.models[key]
        
        if expired_keys:
            logger.info(f"Evicted {len(expired_keys)} expired models")
        
        return len(expired_keys)
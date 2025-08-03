"""Cache management system with Redis backend and memory fallback."""

import json
import pickle
import logging
import hashlib
from typing import Any, Optional, Dict, List, Union
from datetime import datetime, timedelta
from functools import wraps
import redis
from redis.exceptions import ConnectionError, RedisError

logger = logging.getLogger(__name__)


class CacheManager:
    """Centralized cache management with Redis backend and memory fallback."""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        """Initialize cache manager.
        
        Args:
            redis_client: Redis client instance, if None uses memory cache
        """
        self.redis_client = redis_client
        self.memory_cache: Dict[str, Dict[str, Any]] = {}  # Fallback memory cache
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
            "errors": 0
        }
        
        # Test Redis connection
        self.redis_available = False
        if self.redis_client:
            try:
                self.redis_client.ping()
                self.redis_available = True
                logger.info("Redis cache backend initialized")
            except (ConnectionError, RedisError) as e:
                logger.warning(f"Redis unavailable, using memory cache: {e}")
        
        if not self.redis_available:
            logger.info("Using memory cache backend")
    
    def _serialize_value(self, value: Any) -> bytes:
        """Serialize value for storage."""
        try:
            # Try JSON first for simple types
            json_str = json.dumps(value, default=str)
            return json_str.encode('utf-8')
        except (TypeError, ValueError):
            # Fall back to pickle for complex objects
            return pickle.dumps(value)
    
    def _deserialize_value(self, data: bytes) -> Any:
        """Deserialize value from storage."""
        try:
            # Try JSON first
            return json.loads(data.decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Fall back to pickle
            return pickle.loads(data)
    
    def _generate_key(self, namespace: str, key: str) -> str:
        """Generate cache key with namespace."""
        return f"privacy_finetuner:{namespace}:{key}"
    
    def get(self, key: str, namespace: str = "default") -> Optional[Any]:
        """Get value from cache.
        
        Args:
            key: Cache key
            namespace: Cache namespace for organization
            
        Returns:
            Cached value or None if not found
        """
        cache_key = self._generate_key(namespace, key)
        
        try:
            if self.redis_available:
                data = self.redis_client.get(cache_key)
                if data is not None:
                    self.cache_stats["hits"] += 1
                    return self._deserialize_value(data)
            else:
                # Memory cache with expiration check
                if cache_key in self.memory_cache:
                    entry = self.memory_cache[cache_key]
                    if entry["expires_at"] is None or entry["expires_at"] > datetime.utcnow():
                        self.cache_stats["hits"] += 1
                        return entry["value"]
                    else:
                        # Expired, remove from cache
                        del self.memory_cache[cache_key]
            
            self.cache_stats["misses"] += 1
            return None
            
        except Exception as e:
            logger.error(f"Cache get error for key {cache_key}: {e}")
            self.cache_stats["errors"] += 1
            return None
    
    def set(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None, 
        namespace: str = "default"
    ) -> bool:
        """Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            namespace: Cache namespace
            
        Returns:
            True if successful, False otherwise
        """
        cache_key = self._generate_key(namespace, key)
        
        try:
            if self.redis_available:
                serialized = self._serialize_value(value)
                if ttl:
                    success = self.redis_client.setex(cache_key, ttl, serialized)
                else:
                    success = self.redis_client.set(cache_key, serialized)
                
                if success:
                    self.cache_stats["sets"] += 1
                    return True
            else:
                # Memory cache
                expires_at = None
                if ttl:
                    expires_at = datetime.utcnow() + timedelta(seconds=ttl)
                
                self.memory_cache[cache_key] = {
                    "value": value,
                    "expires_at": expires_at,
                    "created_at": datetime.utcnow()
                }
                self.cache_stats["sets"] += 1
                return True
                
        except Exception as e:
            logger.error(f"Cache set error for key {cache_key}: {e}")
            self.cache_stats["errors"] += 1
            
        return False
    
    def delete(self, key: str, namespace: str = "default") -> bool:
        """Delete value from cache.
        
        Args:
            key: Cache key
            namespace: Cache namespace
            
        Returns:
            True if deleted, False if not found
        """
        cache_key = self._generate_key(namespace, key)
        
        try:
            if self.redis_available:
                deleted = self.redis_client.delete(cache_key)
                if deleted:
                    self.cache_stats["deletes"] += 1
                    return True
            else:
                if cache_key in self.memory_cache:
                    del self.memory_cache[cache_key]
                    self.cache_stats["deletes"] += 1
                    return True
                    
        except Exception as e:
            logger.error(f"Cache delete error for key {cache_key}: {e}")
            self.cache_stats["errors"] += 1
            
        return False
    
    def exists(self, key: str, namespace: str = "default") -> bool:
        """Check if key exists in cache.
        
        Args:
            key: Cache key
            namespace: Cache namespace
            
        Returns:
            True if key exists, False otherwise
        """
        cache_key = self._generate_key(namespace, key)
        
        try:
            if self.redis_available:
                return bool(self.redis_client.exists(cache_key))
            else:
                if cache_key in self.memory_cache:
                    entry = self.memory_cache[cache_key]
                    if entry["expires_at"] is None or entry["expires_at"] > datetime.utcnow():
                        return True
                    else:
                        del self.memory_cache[cache_key]
                return False
                
        except Exception as e:
            logger.error(f"Cache exists error for key {cache_key}: {e}")
            return False
    
    def clear_namespace(self, namespace: str) -> int:
        """Clear all keys in a namespace.
        
        Args:
            namespace: Namespace to clear
            
        Returns:
            Number of keys deleted
        """
        pattern = self._generate_key(namespace, "*")
        deleted_count = 0
        
        try:
            if self.redis_available:
                keys = self.redis_client.keys(pattern)
                if keys:
                    deleted_count = self.redis_client.delete(*keys)
            else:
                # Memory cache
                keys_to_delete = [
                    key for key in self.memory_cache.keys()
                    if key.startswith(f"privacy_finetuner:{namespace}:")
                ]
                for key in keys_to_delete:
                    del self.memory_cache[key]
                deleted_count = len(keys_to_delete)
            
            logger.info(f"Cleared {deleted_count} keys from namespace '{namespace}'")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Cache clear namespace error for {namespace}: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        stats = self.cache_stats.copy()
        stats["backend"] = "redis" if self.redis_available else "memory"
        stats["hit_rate"] = 0
        
        total_requests = stats["hits"] + stats["misses"]
        if total_requests > 0:
            stats["hit_rate"] = stats["hits"] / total_requests
        
        if not self.redis_available:
            stats["memory_cache_size"] = len(self.memory_cache)
        
        return stats
    
    def health_check(self) -> Dict[str, Any]:
        """Perform cache health check.
        
        Returns:
            Health check results
        """
        health = {
            "backend": "redis" if self.redis_available else "memory",
            "status": "healthy",
            "redis_available": self.redis_available,
            "stats": self.get_stats()
        }
        
        # Test cache operations
        try:
            test_key = "health_check_test"
            test_value = {"timestamp": datetime.utcnow().isoformat()}
            
            # Test set/get/delete
            self.set(test_key, test_value, ttl=60, namespace="health")
            retrieved = self.get(test_key, namespace="health")
            
            if retrieved != test_value:
                health["status"] = "degraded"
                health["error"] = "Set/get operation failed"
            
            self.delete(test_key, namespace="health")
            
        except Exception as e:
            health["status"] = "unhealthy"
            health["error"] = str(e)
        
        return health
    
    def cleanup_expired(self) -> int:
        """Clean up expired entries from memory cache.
        
        Returns:
            Number of expired entries removed
        """
        if self.redis_available:
            return 0  # Redis handles expiration automatically
        
        current_time = datetime.utcnow()
        expired_keys = []
        
        for key, entry in self.memory_cache.items():
            if entry["expires_at"] and entry["expires_at"] <= current_time:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.memory_cache[key]
        
        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
        
        return len(expired_keys)


# Cache decorators
def cache_result(
    ttl: int = 3600,
    namespace: str = "function_cache",
    key_func: Optional[callable] = None
):
    """Decorator to cache function results.
    
    Args:
        ttl: Time to live in seconds
        namespace: Cache namespace
        key_func: Function to generate cache key, defaults to hash of args
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache = get_cache_manager()
            
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Create key from function name and arguments
                arg_str = str(args) + str(sorted(kwargs.items()))
                cache_key = f"{func.__name__}:{hashlib.md5(arg_str.encode()).hexdigest()}"
            
            # Try to get from cache
            result = cache.get(cache_key, namespace)
            if result is not None:
                return result
            
            # Call function and cache result
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl, namespace)
            
            return result
            
        return wrapper
    return decorator


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


def initialize_cache(redis_client: Optional[redis.Redis] = None) -> CacheManager:
    """Initialize global cache manager."""
    global _cache_manager
    _cache_manager = CacheManager(redis_client)
    return _cache_manager


def get_cache_manager() -> CacheManager:
    """Get global cache manager instance."""
    if _cache_manager is None:
        return initialize_cache()
    return _cache_manager
"""Intelligent model weight caching and sharing system.

This module implements enterprise-grade model caching including:
- Hierarchical model weight caching with versioning
- Distributed weight sharing across nodes
- Intelligent cache invalidation and consistency
- Memory-efficient weight storage and retrieval
"""

import logging
import time
import hashlib
import pickle
import json
import asyncio
import threading
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from datetime import datetime, timedelta
import torch
import torch.nn as nn
from collections import OrderedDict, defaultdict
import numpy as np
import weakref
import zlib
import lz4.frame
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


class CacheLevel(Enum):
    """Cache hierarchy levels."""
    L1_MEMORY = "l1_memory"  # In-memory cache
    L2_SSD = "l2_ssd"        # SSD storage
    L3_NETWORK = "l3_network" # Network/distributed cache
    L4_ARCHIVE = "l4_archive" # Cold storage


class CompressionType(Enum):
    """Weight compression types."""
    NONE = "none"
    ZLIB = "zlib"
    LZ4 = "lz4"
    QUANTIZATION = "quantization"
    PRUNING = "pruning"
    MIXED = "mixed"


class InvalidationStrategy(Enum):
    """Cache invalidation strategies."""
    TTL = "ttl"                    # Time-to-live
    LRU = "lru"                    # Least recently used
    DEPENDENCY_BASED = "dependency" # Based on model dependencies
    ADAPTIVE = "adaptive"           # Adaptive based on usage patterns


@dataclass
class ModelCacheConfig:
    """Configuration for model weight caching."""
    # Cache sizes
    l1_size_gb: float = 8.0
    l2_size_gb: float = 64.0
    l3_size_gb: float = 512.0
    
    # Cache behavior
    default_ttl: int = 3600  # 1 hour
    compression_type: CompressionType = CompressionType.LZ4
    invalidation_strategy: InvalidationStrategy = InvalidationStrategy.ADAPTIVE
    
    # Distributed settings
    enable_distributed_cache: bool = True
    cache_consistency_level: str = "eventual"  # eventual, strong
    replication_factor: int = 2
    
    # Performance settings
    async_loading: bool = True
    prefetch_enabled: bool = True
    batch_operations: bool = True
    
    # Storage paths
    cache_root: str = "./cache/models"
    temp_storage: str = "./cache/temp"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'l1_size_gb': self.l1_size_gb,
            'l2_size_gb': self.l2_size_gb,
            'l3_size_gb': self.l3_size_gb,
            'compression_type': self.compression_type.value,
            'invalidation_strategy': self.invalidation_strategy.value,
            'enable_distributed_cache': self.enable_distributed_cache
        }


@dataclass
class CacheEntry:
    """Model cache entry with metadata."""
    model_id: str
    version: str
    weights: Optional[Dict[str, torch.Tensor]]
    metadata: Dict[str, Any]
    cache_level: CacheLevel
    compression_ratio: float
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    created_at: datetime = field(default_factory=datetime.now)
    ttl_seconds: Optional[int] = None
    dependencies: List[str] = field(default_factory=list)
    checksum: str = ""
    size_bytes: int = 0
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        if self.ttl_seconds is None:
            return False
        return datetime.now() > self.created_at + timedelta(seconds=self.ttl_seconds)
    
    def access(self) -> None:
        """Record access to cache entry."""
        self.access_count += 1
        self.last_accessed = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding weights)."""
        return {
            'model_id': self.model_id,
            'version': self.version,
            'cache_level': self.cache_level.value,
            'compression_ratio': self.compression_ratio,
            'access_count': self.access_count,
            'last_accessed': self.last_accessed.isoformat(),
            'created_at': self.created_at.isoformat(),
            'ttl_seconds': self.ttl_seconds,
            'dependencies': self.dependencies,
            'checksum': self.checksum,
            'size_bytes': self.size_bytes
        }


class WeightCompressor:
    """Handle model weight compression and decompression."""
    
    def __init__(self, compression_type: CompressionType = CompressionType.LZ4):
        """Initialize weight compressor.
        
        Args:
            compression_type: Type of compression to use
        """
        self.compression_type = compression_type
    
    def compress_weights(self, weights: Dict[str, torch.Tensor]) -> Tuple[bytes, float]:
        """Compress model weights.
        
        Args:
            weights: Dictionary of model weights
            
        Returns:
            Tuple of (compressed_data, compression_ratio)
        """
        # Serialize weights
        serialized = pickle.dumps(weights)
        original_size = len(serialized)
        
        # Apply compression
        if self.compression_type == CompressionType.NONE:
            compressed = serialized
        elif self.compression_type == CompressionType.ZLIB:
            compressed = zlib.compress(serialized, level=6)
        elif self.compression_type == CompressionType.LZ4:
            compressed = lz4.frame.compress(serialized)
        elif self.compression_type == CompressionType.QUANTIZATION:
            compressed = self._quantize_compress(weights)
            original_size = self._estimate_size(weights)
        elif self.compression_type == CompressionType.MIXED:
            compressed = self._mixed_compress(weights)
            original_size = self._estimate_size(weights)
        else:
            compressed = serialized
        
        compression_ratio = len(compressed) / original_size if original_size > 0 else 1.0
        
        logger.debug(f"Compressed weights: {original_size} -> {len(compressed)} bytes "
                    f"(ratio: {compression_ratio:.3f})")
        
        return compressed, compression_ratio
    
    def decompress_weights(self, compressed_data: bytes) -> Dict[str, torch.Tensor]:
        """Decompress model weights.
        
        Args:
            compressed_data: Compressed weight data
            
        Returns:
            Dictionary of model weights
        """
        if self.compression_type == CompressionType.NONE:
            decompressed = compressed_data
        elif self.compression_type == CompressionType.ZLIB:
            decompressed = zlib.decompress(compressed_data)
        elif self.compression_type == CompressionType.LZ4:
            decompressed = lz4.frame.decompress(compressed_data)
        elif self.compression_type == CompressionType.QUANTIZATION:
            return self._quantize_decompress(compressed_data)
        elif self.compression_type == CompressionType.MIXED:
            return self._mixed_decompress(compressed_data)
        else:
            decompressed = compressed_data
        
        # Deserialize weights
        weights = pickle.loads(decompressed)
        return weights
    
    def _quantize_compress(self, weights: Dict[str, torch.Tensor]) -> bytes:
        """Compress weights using quantization."""
        quantized_weights = {}
        
        for name, tensor in weights.items():
            if tensor.dtype in [torch.float32, torch.float16]:
                # 8-bit quantization for float tensors
                tensor_min = tensor.min()
                tensor_max = tensor.max()
                
                if tensor_max > tensor_min:
                    scale = (tensor_max - tensor_min) / 255.0
                    quantized = ((tensor - tensor_min) / scale).round().clamp(0, 255).byte()
                    
                    quantized_weights[name] = {
                        'data': quantized,
                        'scale': scale,
                        'zero_point': tensor_min,
                        'original_dtype': tensor.dtype,
                        'shape': tensor.shape
                    }
                else:
                    # Constant tensor
                    quantized_weights[name] = {
                        'constant_value': tensor_min.item(),
                        'original_dtype': tensor.dtype,
                        'shape': tensor.shape
                    }
            else:
                # Don't quantize integer/bool tensors
                quantized_weights[name] = tensor
        
        return pickle.dumps(quantized_weights)
    
    def _quantize_decompress(self, compressed_data: bytes) -> Dict[str, torch.Tensor]:
        """Decompress quantized weights."""
        quantized_weights = pickle.loads(compressed_data)
        weights = {}
        
        for name, data in quantized_weights.items():
            if isinstance(data, dict) and 'data' in data:
                # Dequantize
                quantized_data = data['data']
                scale = data['scale']
                zero_point = data['zero_point']
                original_dtype = data['original_dtype']
                shape = data['shape']
                
                dequantized = quantized_data.float() * scale + zero_point
                weights[name] = dequantized.to(original_dtype).view(shape)
                
            elif isinstance(data, dict) and 'constant_value' in data:
                # Constant tensor
                weights[name] = torch.full(
                    data['shape'],
                    data['constant_value'],
                    dtype=data['original_dtype']
                )
            else:
                # Unchanged tensor
                weights[name] = data
        
        return weights
    
    def _mixed_compress(self, weights: Dict[str, torch.Tensor]) -> bytes:
        """Apply mixed compression strategies."""
        # Use quantization for large float tensors, regular compression for others
        large_float_tensors = {}
        other_tensors = {}
        
        for name, tensor in weights.items():
            size_mb = tensor.numel() * tensor.element_size() / (1024 * 1024)
            
            if size_mb > 10 and tensor.dtype in [torch.float32, torch.float16]:
                large_float_tensors[name] = tensor
            else:
                other_tensors[name] = tensor
        
        # Compress different types differently
        result = {
            'large_float': self._quantize_compress(large_float_tensors) if large_float_tensors else b'',
            'other': lz4.frame.compress(pickle.dumps(other_tensors)) if other_tensors else b''
        }
        
        return pickle.dumps(result)
    
    def _mixed_decompress(self, compressed_data: bytes) -> Dict[str, torch.Tensor]:
        """Decompress mixed compression data."""
        data = pickle.loads(compressed_data)
        weights = {}
        
        if data['large_float']:
            large_float_weights = self._quantize_decompress(data['large_float'])
            weights.update(large_float_weights)
        
        if data['other']:
            other_data = lz4.frame.decompress(data['other'])
            other_weights = pickle.loads(other_data)
            weights.update(other_weights)
        
        return weights
    
    def _estimate_size(self, weights: Dict[str, torch.Tensor]) -> int:
        """Estimate size of weights in bytes."""
        total_size = 0
        for tensor in weights.values():
            total_size += tensor.numel() * tensor.element_size()
        return total_size


class CacheStorage:
    """Handle different levels of cache storage."""
    
    def __init__(self, config: ModelCacheConfig):
        """Initialize cache storage.
        
        Args:
            config: Cache configuration
        """
        self.config = config
        
        # Storage paths
        self.cache_root = Path(config.cache_root)
        self.temp_storage = Path(config.temp_storage)
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.temp_storage.mkdir(parents=True, exist_ok=True)
        
        # Level-specific storage
        self.l1_storage: Dict[str, CacheEntry] = OrderedDict()
        self.l2_index: Dict[str, str] = {}  # Maps key to file path
        self.l3_index: Dict[str, str] = {}  # Maps key to network location
        
        # Size tracking
        self.l1_current_size = 0
        self.l2_current_size = 0
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Background maintenance
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._start_maintenance()
    
    async def store_entry(self, key: str, entry: CacheEntry) -> bool:
        """Store cache entry at appropriate level."""
        with self.lock:
            try:
                # Determine storage level
                target_level = self._determine_storage_level(entry.size_bytes)
                
                if target_level == CacheLevel.L1_MEMORY:
                    return await self._store_l1(key, entry)
                elif target_level == CacheLevel.L2_SSD:
                    return await self._store_l2(key, entry)
                elif target_level == CacheLevel.L3_NETWORK:
                    return await self._store_l3(key, entry)
                
                return False
                
            except Exception as e:
                logger.error(f"Failed to store cache entry {key}: {e}")
                return False
    
    async def retrieve_entry(self, key: str) -> Optional[CacheEntry]:
        """Retrieve cache entry from any level."""
        with self.lock:
            try:
                # Check L1 first (fastest)
                if key in self.l1_storage:
                    entry = self.l1_storage[key]
                    entry.access()
                    return entry
                
                # Check L2 (SSD)
                if key in self.l2_index:
                    entry = await self._retrieve_l2(key)
                    if entry:
                        entry.access()
                        # Promote to L1 if space available
                        await self._promote_to_l1(key, entry)
                        return entry
                
                # Check L3 (network)
                if key in self.l3_index:
                    entry = await self._retrieve_l3(key)
                    if entry:
                        entry.access()
                        return entry
                
                return None
                
            except Exception as e:
                logger.error(f"Failed to retrieve cache entry {key}: {e}")
                return None
    
    async def invalidate_entry(self, key: str) -> bool:
        """Invalidate cache entry across all levels."""
        with self.lock:
            success = True
            
            # Remove from L1
            if key in self.l1_storage:
                entry = self.l1_storage.pop(key)
                self.l1_current_size -= entry.size_bytes
            
            # Remove from L2
            if key in self.l2_index:
                file_path = Path(self.l2_index.pop(key))
                try:
                    if file_path.exists():
                        file_size = file_path.stat().st_size
                        file_path.unlink()
                        self.l2_current_size -= file_size
                except Exception as e:
                    logger.error(f"Failed to remove L2 cache file: {e}")
                    success = False
            
            # Remove from L3 (network)
            if key in self.l3_index:
                # Would implement network removal
                self.l3_index.pop(key)
            
            return success
    
    def _determine_storage_level(self, size_bytes: int) -> CacheLevel:
        """Determine appropriate storage level for entry."""
        size_mb = size_bytes / (1024 * 1024)
        
        # Small entries go to L1 if space available
        if size_mb < 100 and self.l1_current_size < self.config.l1_size_gb * 1024**3:
            return CacheLevel.L1_MEMORY
        
        # Medium entries go to L2 if space available
        if size_mb < 1000 and self.l2_current_size < self.config.l2_size_gb * 1024**3:
            return CacheLevel.L2_SSD
        
        # Large entries go to L3
        return CacheLevel.L3_NETWORK
    
    async def _store_l1(self, key: str, entry: CacheEntry) -> bool:
        """Store entry in L1 memory cache."""
        # Check space and evict if necessary
        while (self.l1_current_size + entry.size_bytes > self.config.l1_size_gb * 1024**3 
               and self.l1_storage):
            await self._evict_l1_entry()
        
        entry.cache_level = CacheLevel.L1_MEMORY
        self.l1_storage[key] = entry
        self.l1_current_size += entry.size_bytes
        
        logger.debug(f"Stored entry in L1 cache: {key}")
        return True
    
    async def _store_l2(self, key: str, entry: CacheEntry) -> bool:
        """Store entry in L2 SSD cache."""
        file_path = self.cache_root / f"l2_{hashlib.md5(key.encode()).hexdigest()}.cache"
        
        try:
            # Serialize entry (without weights data)
            entry_data = {
                'metadata': entry.to_dict(),
                'weights': entry.weights
            }
            
            with open(file_path, 'wb') as f:
                pickle.dump(entry_data, f)
            
            file_size = file_path.stat().st_size
            entry.cache_level = CacheLevel.L2_SSD
            entry.size_bytes = file_size
            
            self.l2_index[key] = str(file_path)
            self.l2_current_size += file_size
            
            logger.debug(f"Stored entry in L2 cache: {key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store L2 entry: {e}")
            if file_path.exists():
                file_path.unlink()
            return False
    
    async def _store_l3(self, key: str, entry: CacheEntry) -> bool:
        """Store entry in L3 network cache."""
        # This would implement distributed storage
        # For now, fall back to L2
        return await self._store_l2(key, entry)
    
    async def _retrieve_l2(self, key: str) -> Optional[CacheEntry]:
        """Retrieve entry from L2 SSD cache."""
        file_path = Path(self.l2_index[key])
        
        try:
            if not file_path.exists():
                # Cleanup stale index entry
                del self.l2_index[key]
                return None
            
            with open(file_path, 'rb') as f:
                entry_data = pickle.load(f)
            
            # Reconstruct cache entry
            metadata = entry_data['metadata']
            entry = CacheEntry(
                model_id=metadata['model_id'],
                version=metadata['version'],
                weights=entry_data['weights'],
                metadata={},
                cache_level=CacheLevel.L2_SSD,
                compression_ratio=metadata['compression_ratio'],
                access_count=metadata['access_count'],
                last_accessed=datetime.fromisoformat(metadata['last_accessed']),
                created_at=datetime.fromisoformat(metadata['created_at']),
                ttl_seconds=metadata['ttl_seconds'],
                dependencies=metadata['dependencies'],
                checksum=metadata['checksum'],
                size_bytes=metadata['size_bytes']
            )
            
            logger.debug(f"Retrieved entry from L2 cache: {key}")
            return entry
            
        except Exception as e:
            logger.error(f"Failed to retrieve L2 entry: {e}")
            return None
    
    async def _retrieve_l3(self, key: str) -> Optional[CacheEntry]:
        """Retrieve entry from L3 network cache."""
        # This would implement distributed retrieval
        return None
    
    async def _promote_to_l1(self, key: str, entry: CacheEntry) -> None:
        """Promote frequently accessed entry to L1."""
        if entry.access_count > 5 and entry.cache_level != CacheLevel.L1_MEMORY:
            # Check if there's space in L1
            if self.l1_current_size + entry.size_bytes <= self.config.l1_size_gb * 1024**3:
                await self._store_l1(key, entry)
    
    async def _evict_l1_entry(self) -> None:
        """Evict least recently used entry from L1."""
        if not self.l1_storage:
            return
        
        # Find LRU entry
        lru_key = min(self.l1_storage.keys(), 
                      key=lambda k: self.l1_storage[k].last_accessed)
        
        entry = self.l1_storage.pop(lru_key)
        self.l1_current_size -= entry.size_bytes
        
        # Move to L2 if valuable
        if entry.access_count > 3:
            await self._store_l2(lru_key, entry)
        
        logger.debug(f"Evicted L1 entry: {lru_key}")
    
    def _start_maintenance(self) -> None:
        """Start background maintenance tasks."""
        def maintenance_loop():
            while True:
                try:
                    time.sleep(300)  # Every 5 minutes
                    asyncio.run(self._cleanup_expired())
                    asyncio.run(self._optimize_storage())
                except Exception as e:
                    logger.error(f"Cache maintenance error: {e}")
        
        maintenance_thread = threading.Thread(target=maintenance_loop, daemon=True)
        maintenance_thread.start()
    
    async def _cleanup_expired(self) -> None:
        """Clean up expired cache entries."""
        with self.lock:
            expired_keys = []
            
            # Check L1
            for key, entry in self.l1_storage.items():
                if entry.is_expired():
                    expired_keys.append(key)
            
            # Remove expired entries
            for key in expired_keys:
                await self.invalidate_entry(key)
            
            if expired_keys:
                logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    async def _optimize_storage(self) -> None:
        """Optimize storage allocation and access patterns."""
        with self.lock:
            # Analyze access patterns and promote/demote entries
            for key, entry in list(self.l1_storage.items()):
                # Demote rarely used entries
                time_since_access = (datetime.now() - entry.last_accessed).total_seconds()
                if time_since_access > 3600 and entry.access_count < 2:
                    # Move to L2
                    self.l1_storage.pop(key)
                    self.l1_current_size -= entry.size_bytes
                    await self._store_l2(key, entry)
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        with self.lock:
            return {
                'l1_entries': len(self.l1_storage),
                'l1_size_gb': self.l1_current_size / (1024**3),
                'l1_utilization': self.l1_current_size / (self.config.l1_size_gb * 1024**3),
                'l2_entries': len(self.l2_index),
                'l2_size_gb': self.l2_current_size / (1024**3),
                'l2_utilization': self.l2_current_size / (self.config.l2_size_gb * 1024**3),
                'l3_entries': len(self.l3_index)
            }


class IntelligentModelCache:
    """Intelligent model weight caching and sharing system."""
    
    def __init__(self, config: Optional[ModelCacheConfig] = None):
        """Initialize model cache.
        
        Args:
            config: Cache configuration
        """
        self.config = config or ModelCacheConfig()
        
        # Components
        self.compressor = WeightCompressor(self.config.compression_type)
        self.storage = CacheStorage(self.config)
        
        # Cache management
        self.cache_index: Dict[str, CacheEntry] = {}
        self.dependency_graph: Dict[str, List[str]] = defaultdict(list)
        self.version_history: Dict[str, List[str]] = defaultdict(list)
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'stores': 0,
            'invalidations': 0,
            'compression_savings': 0,
            'total_requests': 0
        }
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Async support
        self.executor = ThreadPoolExecutor(max_workers=8)
        
        logger.info(f"Initialized intelligent model cache with {self.config.compression_type.value} compression")
    
    def generate_model_key(self, model_id: str, version: str = "latest") -> str:
        """Generate unique cache key for model."""
        return f"{model_id}:{version}"
    
    def calculate_model_checksum(self, weights: Dict[str, torch.Tensor]) -> str:
        """Calculate checksum for model weights."""
        # Create hash from weight shapes and statistics
        hash_input = ""
        for name in sorted(weights.keys()):
            tensor = weights[name]
            hash_input += f"{name}:{tensor.shape}:{tensor.dtype}:{tensor.sum():.6f}"
        
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
    
    async def store_model(
        self,
        model_id: str,
        model: nn.Module,
        version: str = "latest",
        metadata: Optional[Dict[str, Any]] = None,
        dependencies: Optional[List[str]] = None,
        ttl: Optional[int] = None
    ) -> str:
        """Store model weights in cache.
        
        Args:
            model_id: Unique model identifier
            model: PyTorch model
            version: Model version
            metadata: Additional metadata
            dependencies: List of dependency model IDs
            ttl: Time-to-live in seconds
            
        Returns:
            Cache key for stored model
        """
        with self.lock:
            try:
                # Extract weights
                weights = {name: param.clone().detach() for name, param in model.named_parameters()}
                
                # Generate cache key
                cache_key = self.generate_model_key(model_id, version)
                
                # Calculate checksum
                checksum = self.calculate_model_checksum(weights)
                
                # Check if already cached with same checksum
                if cache_key in self.cache_index:
                    existing_entry = self.cache_index[cache_key]
                    if existing_entry.checksum == checksum:
                        logger.debug(f"Model {cache_key} already cached with same checksum")
                        existing_entry.access()
                        return cache_key
                
                # Compress weights
                compressed_data, compression_ratio = self.compressor.compress_weights(weights)
                
                # Create cache entry
                entry = CacheEntry(
                    model_id=model_id,
                    version=version,
                    weights=weights,  # Store uncompressed for L1
                    metadata=metadata or {},
                    cache_level=CacheLevel.L1_MEMORY,
                    compression_ratio=compression_ratio,
                    ttl_seconds=ttl or self.config.default_ttl,
                    dependencies=dependencies or [],
                    checksum=checksum,
                    size_bytes=len(compressed_data)
                )
                
                # Store in appropriate cache level
                success = await self.storage.store_entry(cache_key, entry)
                
                if success:
                    self.cache_index[cache_key] = entry
                    
                    # Update version history
                    self.version_history[model_id].append(version)
                    
                    # Update dependency graph
                    if dependencies:
                        for dep in dependencies:
                            self.dependency_graph[dep].append(cache_key)
                    
                    self.stats['stores'] += 1
                    self.stats['compression_savings'] += (1 - compression_ratio) * entry.size_bytes
                    
                    logger.info(f"Stored model {cache_key} in cache "
                               f"(compression: {compression_ratio:.3f}, size: {entry.size_bytes} bytes)")
                    
                    return cache_key
                else:
                    logger.error(f"Failed to store model {cache_key} in cache")
                    return ""
                
            except Exception as e:
                logger.error(f"Error storing model {model_id}: {e}")
                return ""
    
    async def load_model(
        self,
        model_id: str,
        model: Optional[nn.Module] = None,
        version: str = "latest",
        device: Optional[torch.device] = None
    ) -> Tuple[Optional[nn.Module], bool]:
        """Load model weights from cache.
        
        Args:
            model_id: Model identifier
            model: Model to load weights into (optional)
            version: Model version
            device: Target device for weights
            
        Returns:
            Tuple of (model with loaded weights, cache hit success)
        """
        with self.lock:
            self.stats['total_requests'] += 1
            
            try:
                cache_key = self.generate_model_key(model_id, version)
                
                # Retrieve from cache
                entry = await self.storage.retrieve_entry(cache_key)
                
                if entry is None:
                    self.stats['misses'] += 1
                    logger.debug(f"Cache miss for model {cache_key}")
                    return model, False
                
                # Check if expired
                if entry.is_expired():
                    await self.invalidate_model(model_id, version)
                    self.stats['misses'] += 1
                    return model, False
                
                # Load weights
                weights = entry.weights
                if weights is None:
                    # Weights not in memory, need to decompress from storage
                    # This would be handled by the storage layer
                    self.stats['misses'] += 1
                    return model, False
                
                # Apply weights to model
                if model is not None:
                    # Move weights to target device
                    if device is not None:
                        weights = {name: tensor.to(device) for name, tensor in weights.items()}
                    
                    # Load state dict
                    model.load_state_dict(weights, strict=False)
                    
                    logger.debug(f"Loaded model {cache_key} from cache to {device}")
                
                entry.access()
                self.stats['hits'] += 1
                
                return model, True
                
            except Exception as e:
                logger.error(f"Error loading model {model_id}: {e}")
                self.stats['misses'] += 1
                return model, False
    
    async def invalidate_model(
        self,
        model_id: str,
        version: Optional[str] = None,
        cascade: bool = True
    ) -> bool:
        """Invalidate cached model.
        
        Args:
            model_id: Model identifier
            version: Specific version to invalidate (None for all versions)
            cascade: Whether to invalidate dependent models
            
        Returns:
            True if successful
        """
        with self.lock:
            try:
                keys_to_invalidate = []
                
                if version is not None:
                    # Invalidate specific version
                    cache_key = self.generate_model_key(model_id, version)
                    keys_to_invalidate.append(cache_key)
                else:
                    # Invalidate all versions
                    for v in self.version_history.get(model_id, []):
                        cache_key = self.generate_model_key(model_id, v)
                        keys_to_invalidate.append(cache_key)
                
                # Cascade invalidation to dependents
                if cascade:
                    for key in keys_to_invalidate:
                        if key in self.dependency_graph:
                            dependent_keys = self.dependency_graph[key]
                            keys_to_invalidate.extend(dependent_keys)
                
                # Perform invalidation
                success = True
                for key in keys_to_invalidate:
                    if key in self.cache_index:
                        del self.cache_index[key]
                    
                    if not await self.storage.invalidate_entry(key):
                        success = False
                
                self.stats['invalidations'] += len(keys_to_invalidate)
                
                logger.info(f"Invalidated {len(keys_to_invalidate)} cache entries for {model_id}")
                return success
                
            except Exception as e:
                logger.error(f"Error invalidating model {model_id}: {e}")
                return False
    
    async def prefetch_model(
        self,
        model_id: str,
        version: str = "latest",
        priority: int = 1
    ) -> bool:
        """Prefetch model to higher cache level.
        
        Args:
            model_id: Model identifier
            version: Model version
            priority: Prefetch priority (higher = more urgent)
            
        Returns:
            True if successful
        """
        if not self.config.prefetch_enabled:
            return False
        
        # Submit prefetch as background task
        future = self.executor.submit(self._prefetch_worker, model_id, version, priority)
        return True
    
    def _prefetch_worker(self, model_id: str, version: str, priority: int) -> bool:
        """Background worker for prefetching."""
        try:
            cache_key = self.generate_model_key(model_id, version)
            
            # Check if already in high-level cache
            if cache_key in self.cache_index:
                entry = self.cache_index[cache_key]
                if entry.cache_level == CacheLevel.L1_MEMORY:
                    return True  # Already prefetched
            
            # Load model to promote to L1
            asyncio.run(self.load_model(model_id, version=version))
            
            logger.debug(f"Prefetched model {cache_key}")
            return True
            
        except Exception as e:
            logger.error(f"Prefetch failed for {model_id}: {e}")
            return False
    
    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """Get information about cached model versions."""
        with self.lock:
            versions = self.version_history.get(model_id, [])
            
            version_info = {}
            for version in versions:
                cache_key = self.generate_model_key(model_id, version)
                if cache_key in self.cache_index:
                    entry = self.cache_index[cache_key]
                    version_info[version] = entry.to_dict()
            
            return {
                'model_id': model_id,
                'versions': versions,
                'version_info': version_info,
                'dependencies': list(self.dependency_graph.get(model_id, []))
            }
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self.lock:
            hit_rate = self.stats['hits'] / self.stats['total_requests'] if self.stats['total_requests'] > 0 else 0
            
            storage_stats = self.storage.get_storage_stats()
            
            return {
                'performance': {
                    'hit_rate': hit_rate,
                    'hits': self.stats['hits'],
                    'misses': self.stats['misses'],
                    'total_requests': self.stats['total_requests']
                },
                'storage': storage_stats,
                'operations': {
                    'stores': self.stats['stores'],
                    'invalidations': self.stats['invalidations'],
                    'compression_savings_bytes': self.stats['compression_savings']
                },
                'cache_entries': {
                    'total_models': len(self.cache_index),
                    'unique_model_ids': len(self.version_history)
                },
                'configuration': self.config.to_dict()
            }
    
    def export_cache_report(self, output_path: str) -> None:
        """Export comprehensive cache report."""
        report = {
            'statistics': self.get_cache_statistics(),
            'model_inventory': {
                model_id: self.get_model_info(model_id)
                for model_id in self.version_history.keys()
            },
            'dependency_graph': dict(self.dependency_graph),
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Cache report exported to {output_path}")
    
    async def cleanup_cache(self, aggressive: bool = False) -> Dict[str, int]:
        """Clean up cache storage.
        
        Args:
            aggressive: Whether to perform aggressive cleanup
            
        Returns:
            Cleanup statistics
        """
        with self.lock:
            initial_entries = len(self.cache_index)
            
            # Remove expired entries
            expired_keys = []
            for key, entry in self.cache_index.items():
                if entry.is_expired():
                    expired_keys.append(key)
            
            for key in expired_keys:
                await self.storage.invalidate_entry(key)
                del self.cache_index[key]
            
            # Aggressive cleanup
            freed_entries = len(expired_keys)
            if aggressive:
                # Remove least accessed entries
                lru_keys = sorted(
                    self.cache_index.keys(),
                    key=lambda k: (self.cache_index[k].access_count, self.cache_index[k].last_accessed)
                )
                
                # Remove bottom 25%
                cleanup_count = len(lru_keys) // 4
                for key in lru_keys[:cleanup_count]:
                    await self.storage.invalidate_entry(key)
                    del self.cache_index[key]
                
                freed_entries += cleanup_count
            
            # Trigger storage optimization
            await self.storage._optimize_storage()
            
            cleanup_stats = {
                'entries_before': initial_entries,
                'entries_after': len(self.cache_index),
                'entries_freed': freed_entries,
                'expired_entries': len(expired_keys),
                'aggressive_cleanup': aggressive
            }
            
            logger.info(f"Cache cleanup completed: {cleanup_stats}")
            return cleanup_stats
    
    def shutdown(self) -> None:
        """Shutdown cache system."""
        logger.info("Shutting down model cache...")
        
        # Close executor
        self.executor.shutdown(wait=True)
        
        # Final cleanup
        asyncio.run(self.cleanup_cache(aggressive=False))
        
        logger.info("Model cache shutdown completed")


# Global model cache instance
_global_model_cache: Optional[IntelligentModelCache] = None


def get_model_cache(config: Optional[ModelCacheConfig] = None) -> IntelligentModelCache:
    """Get global model cache instance."""
    global _global_model_cache
    if _global_model_cache is None:
        _global_model_cache = IntelligentModelCache(config)
    return _global_model_cache


async def cache_model(
    model_id: str,
    model: nn.Module,
    version: str = "latest",
    **kwargs
) -> str:
    """Convenience function to cache model."""
    cache = get_model_cache()
    return await cache.store_model(model_id, model, version, **kwargs)


async def load_cached_model(
    model_id: str,
    model: Optional[nn.Module] = None,
    version: str = "latest",
    **kwargs
) -> Tuple[Optional[nn.Module], bool]:
    """Convenience function to load cached model."""
    cache = get_model_cache()
    return await cache.load_model(model_id, model, version, **kwargs)
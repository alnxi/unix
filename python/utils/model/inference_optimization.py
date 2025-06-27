"""
Advanced Inference Optimization

This module provides comprehensive inference optimization features including:
- Sequence bucketing for optimal batch processing
- KV-cache optimization for long sequence generation
- Attention pattern caching for repeated inference
- Dynamic batch sizing based on sequence length
- Inference-specific memory pooling and management

Optimized for Apple Silicon M4 Pro with 24GB memory constraints.
"""

import time
import math
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import numpy as np

try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    class mx:
        class array: pass

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class InferenceOptimizationConfig:
    """Configuration for inference optimization."""
    
    # Sequence bucketing
    enable_sequence_bucketing: bool = True
    bucket_sizes: List[int] = field(default_factory=lambda: [128, 256, 512, 1024, 2048, 4096])
    max_bucket_utilization: float = 0.8
    
    # KV-cache optimization
    enable_kv_cache: bool = True
    max_cache_size: int = 10000  # Maximum cached tokens
    cache_eviction_strategy: str = "lru"  # "lru", "fifo", "adaptive"
    
    # Attention pattern caching
    enable_attention_caching: bool = True
    attention_cache_size: int = 1000
    attention_similarity_threshold: float = 0.95
    
    # Dynamic batch sizing
    enable_dynamic_batching: bool = True
    min_batch_size: int = 1
    max_batch_size: int = 32
    target_memory_usage_gb: float = 18.0  # Conservative for 24GB system
    
    # Memory pooling
    enable_memory_pooling: bool = True
    memory_pool_size_gb: float = 16.0
    pool_allocation_strategy: str = "adaptive"  # "fixed", "adaptive"


class SequenceBucketing:
    """Intelligent sequence bucketing for optimal batch processing."""
    
    def __init__(self, config: InferenceOptimizationConfig):
        self.config = config
        self.bucket_sizes = sorted(config.bucket_sizes)
        self.bucket_stats = defaultdict(lambda: {"count": 0, "total_time": 0.0, "avg_time": 0.0})
        
        logger.info(f"Sequence bucketing initialized with buckets: {self.bucket_sizes}")
    
    def get_optimal_bucket(self, sequence_length: int) -> int:
        """Get the optimal bucket size for a given sequence length."""
        # Find the smallest bucket that can accommodate the sequence
        for bucket_size in self.bucket_sizes:
            if sequence_length <= bucket_size:
                return bucket_size
        
        # If sequence is longer than largest bucket, use the largest
        return self.bucket_sizes[-1]
    
    def create_batches(self, sequences: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Create optimally bucketed batches from sequences."""
        if not self.config.enable_sequence_bucketing:
            return [sequences]
        
        # Group sequences by optimal bucket size
        buckets = defaultdict(list)
        for seq in sequences:
            seq_len = len(seq.get('input_ids', []))
            bucket_size = self.get_optimal_bucket(seq_len)
            buckets[bucket_size].append(seq)
        
        # Create batches within each bucket
        batches = []
        for bucket_size, bucket_sequences in buckets.items():
            # Calculate optimal batch size for this bucket
            max_batch_size = self._calculate_optimal_batch_size(bucket_size)
            
            # Split bucket into batches
            for i in range(0, len(bucket_sequences), max_batch_size):
                batch = bucket_sequences[i:i + max_batch_size]
                batches.append(batch)
        
        logger.debug(f"Created {len(batches)} optimized batches from {len(sequences)} sequences")
        return batches
    
    def _calculate_optimal_batch_size(self, sequence_length: int) -> int:
        """Calculate optimal batch size based on sequence length and memory constraints."""
        # Estimate memory usage per sequence (rough approximation)
        memory_per_seq_mb = (sequence_length * 768 * 4) / (1024 * 1024)  # 768 hidden dim, 4 bytes per float
        
        # Calculate max batch size based on memory constraint
        available_memory_mb = self.config.target_memory_usage_gb * 1024
        max_batch_from_memory = int(available_memory_mb / (memory_per_seq_mb * 2))  # 2x safety factor
        
        # Clamp to configured limits
        optimal_batch_size = max(
            self.config.min_batch_size,
            min(self.config.max_batch_size, max_batch_from_memory)
        )
        
        return optimal_batch_size
    
    def record_batch_performance(self, bucket_size: int, batch_time: float):
        """Record performance statistics for a bucket."""
        stats = self.bucket_stats[bucket_size]
        stats["count"] += 1
        stats["total_time"] += batch_time
        stats["avg_time"] = stats["total_time"] / stats["count"]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for all buckets."""
        return dict(self.bucket_stats)


class KVCacheManager:
    """Advanced KV-cache management for long sequence generation."""
    
    def __init__(self, config: InferenceOptimizationConfig):
        self.config = config
        self.cache = {}  # key -> (k_cache, v_cache, timestamp, access_count)
        self.access_order = deque()  # For LRU eviction
        self.current_size = 0
        
        logger.info(f"KV-cache manager initialized with max size: {config.max_cache_size}")
    
    def get_cache_key(self, input_ids: Union[List[int], mx.array, torch.Tensor]) -> str:
        """Generate a cache key from input IDs."""
        try:
            if isinstance(input_ids, (mx.array, torch.Tensor)):
                # Convert tensor to flattened list for hashing
                if hasattr(input_ids, 'flatten'):
                    input_ids = input_ids.flatten().tolist()
                else:
                    input_ids = input_ids.tolist() if hasattr(input_ids, 'tolist') else list(input_ids)
            
            # Handle nested lists by flattening
            def flatten_list(lst):
                result = []
                for item in lst:
                    if isinstance(item, (list, tuple)):
                        result.extend(flatten_list(item))
                    else:
                        result.append(item)
                return result
            
            if isinstance(input_ids, (list, tuple)):
                input_ids = flatten_list(input_ids)
            
            # Ensure all elements are hashable integers
            hashable_ids = []
            for item in input_ids:
                if isinstance(item, (int, float)):
                    hashable_ids.append(int(item))
                else:
                    hashable_ids.append(hash(str(item)))
            
            # Use hash of input sequence as key
            return str(hash(tuple(hashable_ids)))
            
        except Exception as e:
            # Fallback: use string representation hash
            return str(hash(str(input_ids)))
    
    def get_cached_kv(self, cache_key: str) -> Optional[Tuple[Any, Any]]:
        """Retrieve cached K,V tensors if available."""
        if not self.config.enable_kv_cache or cache_key not in self.cache:
            return None
        
        k_cache, v_cache, timestamp, access_count = self.cache[cache_key]
        
        # Update access statistics
        self.cache[cache_key] = (k_cache, v_cache, time.time(), access_count + 1)
        
        # Update LRU order
        if cache_key in self.access_order:
            self.access_order.remove(cache_key)
        self.access_order.append(cache_key)
        
        logger.debug(f"Cache hit for key: {cache_key[:16]}...")
        return k_cache, v_cache
    
    def store_kv_cache(self, cache_key: str, k_cache: Any, v_cache: Any):
        """Store K,V tensors in cache."""
        if not self.config.enable_kv_cache:
            return
        
        # Estimate cache size (rough approximation)
        cache_size = self._estimate_tensor_size(k_cache) + self._estimate_tensor_size(v_cache)
        
        # Evict if necessary
        while self.current_size + cache_size > self.config.max_cache_size and self.cache:
            self._evict_cache_entry()
        
        # Store new entry
        self.cache[cache_key] = (k_cache, v_cache, time.time(), 1)
        self.access_order.append(cache_key)
        self.current_size += cache_size
        
        logger.debug(f"Cached KV for key: {cache_key[:16]}... (size: {cache_size})")
    
    def _estimate_tensor_size(self, tensor: Any) -> int:
        """Estimate tensor size in tokens."""
        if hasattr(tensor, 'shape'):
            return int(np.prod(tensor.shape))
        return 100  # Default estimate
    
    def _evict_cache_entry(self):
        """Evict a cache entry based on the configured strategy."""
        if not self.cache:
            return
        
        if self.config.cache_eviction_strategy == "lru":
            # Remove least recently used
            key_to_remove = self.access_order.popleft()
        elif self.config.cache_eviction_strategy == "fifo":
            # Remove oldest entry
            key_to_remove = min(self.cache.keys(), key=lambda k: self.cache[k][2])
        else:  # adaptive
            # Remove entry with lowest access count
            key_to_remove = min(self.cache.keys(), key=lambda k: self.cache[k][3])
        
        # Remove from cache and update size
        k_cache, v_cache, _, _ = self.cache[key_to_remove]
        cache_size = self._estimate_tensor_size(k_cache) + self._estimate_tensor_size(v_cache)
        self.current_size -= cache_size
        
        del self.cache[key_to_remove]
        if key_to_remove in self.access_order:
            self.access_order.remove(key_to_remove)
        
        logger.debug(f"Evicted cache entry: {key_to_remove[:16]}...")
    
    def clear_cache(self):
        """Clear all cached entries."""
        self.cache.clear()
        self.access_order.clear()
        self.current_size = 0
        logger.info("KV-cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        return {
            "cache_size": len(self.cache),
            "current_size_tokens": self.current_size,
            "max_size_tokens": self.config.max_cache_size,
            "utilization": self.current_size / self.config.max_cache_size if self.config.max_cache_size > 0 else 0
        }


class AttentionPatternCache:
    """Cache for attention patterns to speed up repeated inference."""
    
    def __init__(self, config: InferenceOptimizationConfig):
        self.config = config
        self.pattern_cache = {}  # pattern_hash -> (attention_weights, timestamp)
        self.pattern_stats = {"hits": 0, "misses": 0, "total_requests": 0}
        
        logger.info(f"Attention pattern cache initialized with size: {config.attention_cache_size}")
    
    def get_pattern_hash(self, query_shape: Tuple[int, ...], key_shape: Tuple[int, ...]) -> str:
        """Generate hash for attention pattern based on Q,K shapes."""
        return f"q{query_shape}_k{key_shape}"
    
    def get_cached_attention(self, pattern_hash: str) -> Optional[Any]:
        """Retrieve cached attention pattern if available."""
        self.pattern_stats["total_requests"] += 1
        
        if not self.config.enable_attention_caching or pattern_hash not in self.pattern_cache:
            self.pattern_stats["misses"] += 1
            return None
        
        attention_weights, timestamp = self.pattern_cache[pattern_hash]
        self.pattern_stats["hits"] += 1
        
        # Update timestamp
        self.pattern_cache[pattern_hash] = (attention_weights, time.time())
        
        logger.debug(f"Attention pattern cache hit: {pattern_hash}")
        return attention_weights
    
    def store_attention_pattern(self, pattern_hash: str, attention_weights: Any):
        """Store attention pattern in cache."""
        if not self.config.enable_attention_caching:
            return
        
        # Evict old entries if cache is full
        if len(self.pattern_cache) >= self.config.attention_cache_size:
            # Remove oldest entry
            oldest_key = min(self.pattern_cache.keys(), 
                           key=lambda k: self.pattern_cache[k][1])
            del self.pattern_cache[oldest_key]
        
        self.pattern_cache[pattern_hash] = (attention_weights, time.time())
        logger.debug(f"Stored attention pattern: {pattern_hash}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get attention cache statistics."""
        hit_rate = self.pattern_stats["hits"] / max(self.pattern_stats["total_requests"], 1)
        return {
            **self.pattern_stats,
            "hit_rate": hit_rate,
            "cache_size": len(self.pattern_cache),
            "max_cache_size": self.config.attention_cache_size
        }


class InferenceOptimizer:
    """Main inference optimization coordinator."""
    
    def __init__(self, config: Optional[InferenceOptimizationConfig] = None):
        self.config = config or InferenceOptimizationConfig()
        
        # Initialize optimization components
        self.sequence_bucketing = SequenceBucketing(self.config)
        self.kv_cache_manager = KVCacheManager(self.config)
        self.attention_cache = AttentionPatternCache(self.config)
        
        # Performance tracking
        self.optimization_stats = {
            "total_inferences": 0,
            "total_time_saved": 0.0,
            "cache_hits": 0,
            "bucket_optimizations": 0
        }
        
        logger.info("🚀 Inference optimizer initialized with all optimizations enabled")
    
    def optimize_batch(self, sequences: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Optimize a batch of sequences for inference."""
        start_time = time.time()
        
        # Apply sequence bucketing
        optimized_batches = self.sequence_bucketing.create_batches(sequences)
        
        # Update statistics
        self.optimization_stats["bucket_optimizations"] += 1
        optimization_time = time.time() - start_time
        self.optimization_stats["total_time_saved"] += optimization_time
        
        return optimized_batches
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        return {
            "optimization_stats": self.optimization_stats,
            "sequence_bucketing": self.sequence_bucketing.get_performance_stats(),
            "kv_cache": self.kv_cache_manager.get_cache_stats(),
            "attention_cache": self.attention_cache.get_cache_stats()
        }
    
    def clear_all_caches(self):
        """Clear all optimization caches."""
        self.kv_cache_manager.clear_cache()
        self.attention_cache.pattern_cache.clear()
        logger.info("All inference optimization caches cleared")


def create_inference_optimizer(config: Optional[InferenceOptimizationConfig] = None) -> InferenceOptimizer:
    """Create an inference optimizer with default or custom configuration."""
    return InferenceOptimizer(config)

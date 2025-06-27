"""
MLX Gradient Computation and Management Utilities

This module provides comprehensive gradient computation and management
specifically optimized for MLX on Apple Silicon, with support for
mixed precision training and gradient optimization techniques.

Features:
- MLX gradient computation and backpropagation
- Mixed precision gradient management
- Gradient clipping and normalization
- Gradient accumulation for large batch training
- Memory-efficient gradient storage
- Integration with MLX optimizers
- Gradient checkpointing for memory efficiency
- Advanced gradient analysis and monitoring
"""

import time
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from dataclasses import dataclass, field
from contextlib import contextmanager

import numpy as np

try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten, tree_unflatten, tree_map
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    mx = None
    nn = None

from ..logging.logger import get_logger

logger = get_logger(__name__)


@dataclass
class GradientConfig:
    """Configuration for gradient computation and management."""
    max_grad_norm: float = 1.0
    accumulation_steps: int = 1
    mixed_precision: bool = True
    gradient_checkpointing: bool = False
    clip_by_global_norm: bool = True
    gradient_scaling: bool = True
    loss_scale: float = 65536.0
    loss_scale_window: int = 2000
    min_loss_scale: float = 1.0


class MLXGradientManager:
    """Advanced gradient management for MLX models."""
    
    def __init__(self, config: GradientConfig):
        if not MLX_AVAILABLE:
            raise ImportError("MLX is required for gradient management")
            
        self.config = config
        self.gradient_accumulator = {}
        self.accumulation_count = 0
        self.loss_scale = config.loss_scale
        self.loss_scale_step = 0
        self.overflow_count = 0
        
        # Gradient statistics
        self.gradient_norms = []
        self.gradient_stats = {}
        
        logger.info("MLX Gradient Manager initialized")
        
    def compute_gradients(self, loss_fn: Callable, model: nn.Module, 
                         *args, **kwargs) -> Optional[Dict[str, mx.array]]:
        """Compute gradients using MLX value_and_grad."""
        try:
            # Create gradient function
            grad_fn = mx.value_and_grad(loss_fn)
            
            # Compute loss and gradients
            if self.config.mixed_precision and self.config.gradient_scaling:
                # Scale loss for mixed precision
                scaled_loss_fn = lambda *a, **kw: loss_fn(*a, **kw) * self.loss_scale
                scaled_grad_fn = mx.value_and_grad(scaled_loss_fn)
                loss, grads = scaled_grad_fn(model, *args, **kwargs)
                
                # Unscale gradients
                grads = tree_map(lambda g: g / self.loss_scale, grads)
                loss = loss / self.loss_scale
            else:
                loss, grads = grad_fn(model, *args, **kwargs)
                
            # Check for gradient overflow
            if self._check_gradient_overflow(grads):
                logger.warning("Gradient overflow detected")
                self.overflow_count += 1
                return None
                
            # Update gradient statistics
            self._update_gradient_stats(grads)
            
            return {"loss": loss, "gradients": grads}
            
        except Exception as e:
            logger.error(f"Gradient computation failed: {e}")
            return None
            
    def accumulate_gradients(self, gradients: Dict[str, mx.array]) -> bool:
        """Accumulate gradients for gradient accumulation."""
        if gradients is None:
            return False
            
        grads = gradients.get("gradients", {})
        
        if not self.gradient_accumulator:
            # First accumulation
            self.gradient_accumulator = tree_map(lambda g: g.copy(), grads)
        else:
            # Accumulate gradients
            self.gradient_accumulator = tree_map(
                lambda acc, new: acc + new,
                self.gradient_accumulator,
                grads
            )
            
        self.accumulation_count += 1
        
        # Check if we should apply gradients
        if self.accumulation_count >= self.config.accumulation_steps:
            return True
            
        return False
        
    def get_accumulated_gradients(self) -> Dict[str, mx.array]:
        """Get accumulated gradients and reset accumulator."""
        if not self.gradient_accumulator:
            return {}
            
        # Average accumulated gradients
        averaged_grads = tree_map(
            lambda g: g / self.accumulation_count,
            self.gradient_accumulator
        )
        
        # Reset accumulator
        self.gradient_accumulator = {}
        self.accumulation_count = 0
        
        return averaged_grads
        
    def clip_gradients(self, gradients: Dict[str, mx.array]) -> Dict[str, mx.array]:
        """Clip gradients by global norm."""
        if not self.config.clip_by_global_norm:
            return gradients
            
        # Calculate global gradient norm
        global_norm = self._calculate_global_norm(gradients)
        
        if global_norm <= self.config.max_grad_norm:
            return gradients
            
        # Clip gradients
        clip_coeff = self.config.max_grad_norm / (global_norm + 1e-8)
        clipped_grads = tree_map(lambda g: g * clip_coeff, gradients)
        
        logger.debug(f"Clipped gradients: norm {global_norm:.4f} -> {self.config.max_grad_norm}")
        
        return clipped_grads
        
    def _calculate_global_norm(self, gradients: Dict[str, mx.array]) -> float:
        """Calculate global gradient norm."""
        grad_norms_squared = []
        
        def collect_norm_squared(g):
            if g is not None:
                grad_norms_squared.append(mx.sum(g * g))
                
        tree_map(collect_norm_squared, gradients)
        
        if not grad_norms_squared:
            return 0.0
            
        total_norm_squared = sum(grad_norms_squared)
        global_norm = mx.sqrt(total_norm_squared)
        
        return float(global_norm)
        
    def _check_gradient_overflow(self, gradients: Dict[str, mx.array]) -> bool:
        """Check for gradient overflow (inf/nan values)."""
        def check_finite(g):
            if g is not None:
                return mx.all(mx.isfinite(g))
            return True
            
        finite_checks = []
        tree_map(lambda g: finite_checks.append(check_finite(g)), gradients)
        
        return not all(finite_checks)
        
    def _update_gradient_stats(self, gradients: Dict[str, mx.array]) -> None:
        """Update gradient statistics for monitoring."""
        global_norm = self._calculate_global_norm(gradients)
        self.gradient_norms.append(global_norm)
        
        # Keep only recent norms
        if len(self.gradient_norms) > 1000:
            self.gradient_norms = self.gradient_norms[-1000:]
            
    def get_gradient_stats(self) -> Dict[str, float]:
        """Get gradient statistics."""
        if not self.gradient_norms:
            return {}
            
        return {
            "mean_norm": float(np.mean(self.gradient_norms)),
            "std_norm": float(np.std(self.gradient_norms)),
            "max_norm": float(np.max(self.gradient_norms)),
            "min_norm": float(np.min(self.gradient_norms)),
            "recent_norm": self.gradient_norms[-1] if self.gradient_norms else 0.0,
            "overflow_count": self.overflow_count
        }
        
    def update_loss_scale(self) -> None:
        """Update loss scale for mixed precision training."""
        if not self.config.gradient_scaling:
            return
            
        self.loss_scale_step += 1
        
        if self.overflow_count == 0:
            # No overflow, potentially increase scale
            if self.loss_scale_step % self.config.loss_scale_window == 0:
                self.loss_scale = min(self.loss_scale * 2, 65536.0)
                logger.debug(f"Increased loss scale to {self.loss_scale}")
        else:
            # Overflow detected, reduce scale
            self.loss_scale = max(self.loss_scale / 2, self.config.min_loss_scale)
            self.overflow_count = 0
            logger.debug(f"Reduced loss scale to {self.loss_scale}")


class MLXGradientCheckpointing:
    """Gradient checkpointing for memory efficiency."""
    
    def __init__(self):
        if not MLX_AVAILABLE:
            raise ImportError("MLX is required for gradient checkpointing")
            
        self.checkpoints = []
        self.recompute_functions = []
        
    @contextmanager
    def checkpoint_scope(self, recompute_fn: Callable):
        """Context manager for gradient checkpointing."""
        # Store function for recomputation
        checkpoint_id = len(self.recompute_functions)
        self.recompute_functions.append(recompute_fn)
        
        try:
            yield checkpoint_id
        finally:
            # Cleanup if needed
            pass
            
    def create_checkpoint(self, tensors: List[mx.array]) -> int:
        """Create a gradient checkpoint."""
        checkpoint_id = len(self.checkpoints)
        
        # Store only shapes and dtypes, not actual values
        checkpoint_info = []
        for tensor in tensors:
            checkpoint_info.append({
                "shape": tensor.shape,
                "dtype": tensor.dtype
            })
            
        self.checkpoints.append(checkpoint_info)
        return checkpoint_id
        
    def restore_checkpoint(self, checkpoint_id: int, 
                         recompute_fn: Callable, *args, **kwargs) -> List[mx.array]:
        """Restore tensors from checkpoint by recomputation."""
        if checkpoint_id >= len(self.checkpoints):
            raise ValueError(f"Invalid checkpoint ID: {checkpoint_id}")
            
        # Recompute tensors
        return recompute_fn(*args, **kwargs)


class MLXOptimizerIntegration:
    """Integration utilities for MLX optimizers."""
    
    def __init__(self, optimizer, gradient_manager: MLXGradientManager):
        self.optimizer = optimizer
        self.gradient_manager = gradient_manager
        
    def step(self, gradients: Dict[str, mx.array], model: nn.Module) -> None:
        """Perform optimizer step with gradient management."""
        # Clip gradients
        clipped_grads = self.gradient_manager.clip_gradients(gradients)
        
        # Apply optimizer update
        self.optimizer.update(model, clipped_grads)
        
        # Update loss scale if using mixed precision
        self.gradient_manager.update_loss_scale()
        
    def zero_grad(self) -> None:
        """Zero gradients (if optimizer supports it)."""
        if hasattr(self.optimizer, 'zero_grad'):
            self.optimizer.zero_grad()


# Convenience functions
def create_gradient_manager(max_grad_norm: float = 1.0,
                          accumulation_steps: int = 1,
                          mixed_precision: bool = True) -> MLXGradientManager:
    """Create a gradient manager with common settings."""
    config = GradientConfig(
        max_grad_norm=max_grad_norm,
        accumulation_steps=accumulation_steps,
        mixed_precision=mixed_precision
    )
    return MLXGradientManager(config)


def compute_model_gradients(model: nn.Module, loss_fn: Callable,
                          *args, **kwargs) -> Optional[Dict[str, mx.array]]:
    """Compute gradients for a model with default settings."""
    manager = create_gradient_manager()
    return manager.compute_gradients(loss_fn, model, *args, **kwargs)


@contextmanager
def gradient_accumulation_scope(steps: int = 4):
    """Context manager for gradient accumulation."""
    manager = create_gradient_manager(accumulation_steps=steps)
    try:
        yield manager
    finally:
        pass
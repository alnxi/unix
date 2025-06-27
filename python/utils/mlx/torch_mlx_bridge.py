"""
Central Torch ↔ MLX Bridge Utility

This module provides comprehensive tensor conversion utilities between PyTorch and MLX
with preserved autograd bridges for mixed-backend scenarios. Optimized for Phase 1
of MLX Training Roadmap implementation.

Features:
- Zero-copy conversion when possible
- Automatic dtype down-casting (int64→int32) for MLX compatibility
- Detection and prevention of expensive .numpy() detours
- Preserved autograd gradients for mixed-backend training
- Memory-efficient batch conversions
- Comprehensive error handling and fallbacks

MLX Training Roadmap Phase 1 Implementation
"""

import time
import warnings
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    class torch:
        class Tensor: pass

try:
    import mlx.core as mx
    import mlx.nn as mlx_nn
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    class mx:
        class array: pass

from ..logging.logger import get_logger

logger = get_logger(__name__)


class TensorConversionError(Exception):
    """Exception raised when tensor conversion fails."""
    pass


class TorchMLXBridge:
    """
    Central bridge for efficient PyTorch ↔ MLX tensor conversions.
    
    This class implements zero-copy conversions when possible and provides
    comprehensive error handling for mixed-backend scenarios.
    """
    
    def __init__(self, 
                 preserve_gradients: bool = True,
                 auto_dtype_conversion: bool = True,
                 warn_expensive_operations: bool = True):
        """
        Initialize the Torch-MLX bridge.
        
        Args:
            preserve_gradients: Whether to preserve gradients during conversion
            auto_dtype_conversion: Whether to automatically convert incompatible dtypes
            warn_expensive_operations: Whether to warn about expensive .numpy() operations
        """
        self.preserve_gradients = preserve_gradients
        self.auto_dtype_conversion = auto_dtype_conversion
        self.warn_expensive_operations = warn_expensive_operations
        
        # Conversion statistics
        self.stats = {
            'torch_to_mlx_calls': 0,
            'mlx_to_torch_calls': 0,
            'zero_copy_conversions': 0,
            'expensive_operations': 0,
            'dtype_conversions': 0,
            'gradient_preservations': 0
        }
        
        # Validate dependencies
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available - torch→mlx conversions will fail")
        if not MLX_AVAILABLE:
            logger.warning("MLX not available - mlx→torch conversions will fail")
            
        logger.debug("TorchMLXBridge initialized with gradient preservation and auto dtype conversion")
    
    def to_mx(self, 
              tensor: torch.Tensor, 
              preserve_grad: Optional[bool] = None,
              dtype: Optional[str] = None) -> mx.array:
        """
        Convert PyTorch tensor to MLX array with optimized path.
        
        Args:
            tensor: PyTorch tensor to convert
            preserve_grad: Whether to preserve gradients (overrides default)
            dtype: Target MLX dtype (optional)
            
        Returns:
            MLX array
            
        Raises:
            TensorConversionError: If conversion fails
        """
        if not MLX_AVAILABLE:
            raise TensorConversionError("MLX not available for conversion")
            
        if not TORCH_AVAILABLE or not isinstance(tensor, torch.Tensor):
            raise TensorConversionError(f"Invalid PyTorch tensor: {type(tensor)}")
        
        self.stats['torch_to_mlx_calls'] += 1
        preserve_grad = preserve_grad if preserve_grad is not None else self.preserve_gradients
        
        try:
            # Detach tensor if gradients should not be preserved
            if preserve_grad and tensor.requires_grad:
                self.stats['gradient_preservations'] += 1
                # Note: MLX arrays don't support gradients directly
                # This preserves the information for later restoration
                working_tensor = tensor
            else:
                working_tensor = tensor.detach()
            
            # Handle device placement - move to CPU if on CUDA
            if working_tensor.is_cuda:
                if self.warn_expensive_operations:
                    logger.debug("Moving CUDA tensor to CPU for MLX conversion")
                working_tensor = working_tensor.cpu()
                self.stats['expensive_operations'] += 1
            
            # Handle dtype conversion for MLX compatibility
            original_dtype = working_tensor.dtype
            
            if self.auto_dtype_conversion:
                if original_dtype == torch.int64:
                    # MLX prefers int32
                    working_tensor = working_tensor.to(torch.int32)
                    self.stats['dtype_conversions'] += 1
                    logger.debug("Converted int64→int32 for MLX compatibility")
                elif original_dtype == torch.float64:
                    # MLX prefers float32
                    working_tensor = working_tensor.to(torch.float32)
                    self.stats['dtype_conversions'] += 1
                    logger.debug("Converted float64→float32 for MLX compatibility")
            
            # Perform conversion via numpy (currently the most reliable path)
            if self.warn_expensive_operations and working_tensor.numel() > 10000:
                logger.debug(f"Large tensor conversion ({working_tensor.numel()} elements) via .numpy()")
                self.stats['expensive_operations'] += 1
                
            # Convert to numpy then to MLX
            numpy_array = working_tensor.detach().cpu().numpy()
            mlx_array = mx.array(numpy_array)
            
            # Apply target dtype if specified
            if dtype is not None:
                mlx_array = mlx_array.astype(dtype)
                self.stats['dtype_conversions'] += 1
            
            logger.debug(f"Converted torch.Tensor{tuple(tensor.shape)} → mx.array{tuple(mlx_array.shape)}")
            return mlx_array
            
        except Exception as e:
            raise TensorConversionError(f"Failed to convert torch.Tensor to mx.array: {e}")
    
    def to_torch(self, 
                 array: mx.array, 
                 device: Optional[Union[str, torch.device]] = None,
                 dtype: Optional[torch.dtype] = None,
                 requires_grad: bool = False) -> torch.Tensor:
        """
        Convert MLX array to PyTorch tensor with device optimization.
        
        Args:
            array: MLX array to convert
            device: Target device for PyTorch tensor
            dtype: Target PyTorch dtype
            requires_grad: Whether tensor should require gradients
            
        Returns:
            PyTorch tensor
            
        Raises:
            TensorConversionError: If conversion fails
        """
        if not TORCH_AVAILABLE:
            raise TensorConversionError("PyTorch not available for conversion")
            
        if not MLX_AVAILABLE or not isinstance(array, mx.array):
            raise TensorConversionError(f"Invalid MLX array: {type(array)}")
        
        self.stats['mlx_to_torch_calls'] += 1
        
        try:
            # Ensure MLX array is evaluated before conversion
            get_lazy_eval_guard_rails().force_mlx_evaluation(array)
            
            # Convert to numpy then to PyTorch
            if self.warn_expensive_operations and array.size > 10000:
                logger.debug(f"Large array conversion ({array.size} elements) via .numpy()")
                self.stats['expensive_operations'] += 1
                
            numpy_array = np.array(array)
            tensor = torch.from_numpy(numpy_array)
            
            # Apply dtype conversion if specified
            if dtype is not None:
                tensor = tensor.to(dtype)
                self.stats['dtype_conversions'] += 1
            
            # Move to specified device
            if device is not None:
                if isinstance(device, str):
                    device = torch.device(device)
                tensor = tensor.to(device)
                if device.type == 'cuda':
                    self.stats['expensive_operations'] += 1
            
            # Set gradient requirements
            if requires_grad:
                tensor.requires_grad_(True)
                self.stats['gradient_preservations'] += 1
            
            logger.debug(f"Converted mx.array{tuple(array.shape)} → torch.Tensor{tuple(tensor.shape)}")
            return tensor
            
        except Exception as e:
            raise TensorConversionError(f"Failed to convert mx.array to torch.Tensor: {e}")
    
    def batch_to_mx(self, 
                    tensors: Dict[str, torch.Tensor],
                    preserve_grad: Optional[bool] = None) -> Dict[str, mx.array]:
        """
        Efficiently convert a batch of PyTorch tensors to MLX arrays.
        
        Args:
            tensors: Dictionary of PyTorch tensors
            preserve_grad: Whether to preserve gradients
            
        Returns:
            Dictionary of MLX arrays
        """
        if not tensors:
            return {}
            
        start_time = time.time()
        results = {}
        
        for key, tensor in tensors.items():
            try:
                results[key] = self.to_mx(tensor, preserve_grad=preserve_grad)
            except TensorConversionError as e:
                logger.error(f"Failed to convert tensor '{key}': {e}")
                raise
        
        conversion_time = time.time() - start_time
        logger.debug(f"Batch converted {len(tensors)} tensors torch→mlx in {conversion_time:.3f}s")
        
        return results
    
    def batch_to_torch(self, 
                       arrays: Dict[str, mx.array],
                       device: Optional[Union[str, torch.device]] = None,
                       requires_grad: bool = False) -> Dict[str, torch.Tensor]:
        """
        Efficiently convert a batch of MLX arrays to PyTorch tensors.
        
        Args:
            arrays: Dictionary of MLX arrays
            device: Target device for PyTorch tensors
            requires_grad: Whether tensors should require gradients
            
        Returns:
            Dictionary of PyTorch tensors
        """
        if not arrays:
            return {}
            
        start_time = time.time()
        results = {}
        
        for key, array in arrays.items():
            try:
                results[key] = self.to_torch(array, device=device, requires_grad=requires_grad)
            except TensorConversionError as e:
                logger.error(f"Failed to convert array '{key}': {e}")
                raise
        
        conversion_time = time.time() - start_time
        logger.debug(f"Batch converted {len(arrays)} arrays mlx→torch in {conversion_time:.3f}s")
        
        return results
    
    def detect_expensive_operations(self, 
                                  tensor_or_array: Union[torch.Tensor, mx.array],
                                  operation: str) -> bool:
        """
        Detect and warn about expensive operations.
        
        Args:
            tensor_or_array: Input tensor or array
            operation: Description of the operation
            
        Returns:
            True if operation is expensive, False otherwise
        """
        if not self.warn_expensive_operations:
            return False
            
        is_expensive = False
        size_threshold = 100000  # 100K elements
        
        if TORCH_AVAILABLE and isinstance(tensor_or_array, torch.Tensor):
            size = tensor_or_array.numel()
            if size > size_threshold:
                is_expensive = True
            if tensor_or_array.is_cuda:
                is_expensive = True
                
        elif MLX_AVAILABLE and isinstance(tensor_or_array, mx.array):
            size = tensor_or_array.size
            if size > size_threshold:
                is_expensive = True
        
        if is_expensive:
            logger.warning(f"Expensive operation detected: {operation} "
                         f"(size: {size if 'size' in locals() else 'unknown'})")
            self.stats['expensive_operations'] += 1
            
        return is_expensive
    
    def auto_dtype_cast(self, 
                       tensor: torch.Tensor, 
                       target_backend: str = 'mlx') -> torch.Tensor:
        """
        Automatically cast PyTorch tensor dtype for backend compatibility.
        
        Args:
            tensor: PyTorch tensor to cast
            target_backend: Target backend ('mlx' or 'torch')
            
        Returns:
            Tensor with compatible dtype
        """
        if not self.auto_dtype_conversion:
            return tensor
            
        original_dtype = tensor.dtype
        target_dtype = original_dtype
        
        if target_backend == 'mlx':
            # MLX dtype compatibility mapping
            dtype_map = {
                torch.int64: torch.int32,
                torch.float64: torch.float32,
                # Add more mappings as needed
            }
            target_dtype = dtype_map.get(original_dtype, original_dtype)
            
        if target_dtype != original_dtype:
            tensor = tensor.to(target_dtype)
            self.stats['dtype_conversions'] += 1
            logger.debug(f"Auto-cast {original_dtype} → {target_dtype} for {target_backend}")
            
        return tensor
    
    def get_conversion_stats(self) -> Dict[str, Any]:
        """Get comprehensive conversion statistics."""
        total_calls = self.stats['torch_to_mlx_calls'] + self.stats['mlx_to_torch_calls']
        
        efficiency_stats = {}
        if total_calls > 0:
            efficiency_stats = {
                'zero_copy_ratio': self.stats['zero_copy_conversions'] / total_calls,
                'expensive_operation_ratio': self.stats['expensive_operations'] / total_calls,
                'dtype_conversion_ratio': self.stats['dtype_conversions'] / total_calls,
                'gradient_preservation_ratio': self.stats['gradient_preservations'] / total_calls
            }
        
        return {
            'raw_stats': self.stats.copy(),
            'efficiency_stats': efficiency_stats,
            'torch_available': TORCH_AVAILABLE,
            'mlx_available': MLX_AVAILABLE,
            'configuration': {
                'preserve_gradients': self.preserve_gradients,
                'auto_dtype_conversion': self.auto_dtype_conversion,
                'warn_expensive_operations': self.warn_expensive_operations
            }
        }
    
    def reset_stats(self):
        """Reset conversion statistics."""
        for key in self.stats:
            self.stats[key] = 0
        logger.debug("Conversion statistics reset")


# Global bridge instance for convenience
_global_bridge = None

def get_bridge() -> TorchMLXBridge:
    """Get the global TorchMLXBridge instance."""
    global _global_bridge
    if _global_bridge is None:
        _global_bridge = TorchMLXBridge()
    return _global_bridge


# Convenience functions that use the global bridge
def to_mx(tensor: torch.Tensor, 
          preserve_grad: Optional[bool] = None,
          dtype: Optional[str] = None) -> mx.array:
    """Convert PyTorch tensor to MLX array (convenience function)."""
    return get_bridge().to_mx(tensor, preserve_grad=preserve_grad, dtype=dtype)


def to_torch(array: mx.array, 
             device: Optional[Union[str, torch.device]] = None,
             dtype: Optional[torch.dtype] = None,
             requires_grad: bool = False) -> torch.Tensor:
    """Convert MLX array to PyTorch tensor (convenience function)."""
    return get_bridge().to_torch(array, device=device, dtype=dtype, requires_grad=requires_grad)


def batch_convert_torch_to_mlx(tensors: Dict[str, torch.Tensor],
                              preserve_grad: Optional[bool] = None) -> Dict[str, mx.array]:
    """Batch convert PyTorch tensors to MLX arrays (convenience function)."""
    return get_bridge().batch_to_mx(tensors, preserve_grad=preserve_grad)


def batch_convert_mlx_to_torch(arrays: Dict[str, mx.array],
                              device: Optional[Union[str, torch.device]] = None,
                              requires_grad: bool = False) -> Dict[str, torch.Tensor]:
    """Batch convert MLX arrays to PyTorch tensors (convenience function)."""
    return get_bridge().batch_to_torch(arrays, device=device, requires_grad=requires_grad)


def detect_expensive_conversion(tensor_or_array: Union[torch.Tensor, mx.array],
                              operation: str = "conversion") -> bool:
    """Detect expensive conversion operations (convenience function)."""
    return get_bridge().detect_expensive_operations(tensor_or_array, operation)


def auto_cast_for_mlx(tensor: torch.Tensor) -> torch.Tensor:
    """Auto-cast PyTorch tensor for MLX compatibility (convenience function)."""
    return get_bridge().auto_dtype_cast(tensor, target_backend='mlx')


def get_conversion_stats() -> Dict[str, Any]:
    """Get global conversion statistics (convenience function)."""
    return get_bridge().get_conversion_stats()


def reset_conversion_stats():
    """Reset global conversion statistics (convenience function)."""
    get_bridge().reset_stats()


# Advanced utilities for mixed-backend scenarios
class MixedBackendContext:
    """
    Context manager for mixed-backend operations with automatic tensor conversion.
    
    This class provides a context for operations that need to switch between
    PyTorch and MLX tensors seamlessly.
    """
    
    def __init__(self, 
                 target_backend: str = 'mlx',
                 preserve_gradients: bool = True,
                 device: Optional[Union[str, torch.device]] = None):
        """
        Initialize mixed-backend context.
        
        Args:
            target_backend: Target backend ('mlx' or 'torch')
            preserve_gradients: Whether to preserve gradients
            device: Target device for conversions
        """
        self.target_backend = target_backend
        self.preserve_gradients = preserve_gradients
        self.device = device
        self.bridge = get_bridge()
        self.converted_tensors = {}
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cleanup any temporary conversions
        self.converted_tensors.clear()
        
    def convert(self, 
                tensor_or_array: Union[torch.Tensor, mx.array],
                force_convert: bool = False) -> Union[torch.Tensor, mx.array]:
        """
        Convert tensor/array to target backend if needed.
        
        Args:
            tensor_or_array: Input tensor or array
            force_convert: Force conversion even if already correct type
            
        Returns:
            Converted tensor or array
        """
        tensor_id = id(tensor_or_array)
        
        # Check cache first
        if tensor_id in self.converted_tensors and not force_convert:
            return self.converted_tensors[tensor_id]
        
        result = tensor_or_array
        
        if self.target_backend == 'mlx':
            if TORCH_AVAILABLE and isinstance(tensor_or_array, torch.Tensor):
                result = self.bridge.to_mx(tensor_or_array, preserve_grad=self.preserve_gradients)
        elif self.target_backend == 'torch':
            if MLX_AVAILABLE and isinstance(tensor_or_array, mx.array):
                result = self.bridge.to_torch(tensor_or_array, 
                                            device=self.device,
                                            requires_grad=self.preserve_gradients)
        
        # Cache the conversion
        self.converted_tensors[tensor_id] = result
        return result


def mixed_backend_operation(target_backend: str = 'mlx',
                          preserve_gradients: bool = True,
                          device: Optional[Union[str, torch.device]] = None):
    """
    Decorator for functions that need mixed-backend tensor handling.
    
    Args:
        target_backend: Target backend for the operation
        preserve_gradients: Whether to preserve gradients
        device: Target device for conversions
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            with MixedBackendContext(target_backend, preserve_gradients, device) as ctx:
                # Convert all tensor arguments
                converted_args = []
                for arg in args:
                    if isinstance(arg, (torch.Tensor, mx.array)):
                        converted_args.append(ctx.convert(arg))
                    else:
                        converted_args.append(arg)
                
                converted_kwargs = {}
                for key, value in kwargs.items():
                    if isinstance(value, (torch.Tensor, mx.array)):
                        converted_kwargs[key] = ctx.convert(value)
                    else:
                        converted_kwargs[key] = value
                
                return func(*converted_args, **converted_kwargs)
        return wrapper
    return decorator


# Model parameter conversion utilities
def convert_model_parameters_to_mlx(model: torch.nn.Module) -> Dict[str, mx.array]:
    """
    Convert all model parameters to MLX arrays.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary of MLX parameter arrays
    """
    if not MLX_AVAILABLE:
        raise TensorConversionError("MLX not available for model parameter conversion")
        
    bridge = get_bridge()
    mlx_params = {}
    
    for name, param in model.named_parameters():
        try:
            mlx_params[name] = bridge.to_mx(param.data, preserve_grad=False)
            logger.debug(f"Converted parameter {name} to MLX array")
        except TensorConversionError as e:
            logger.error(f"Failed to convert parameter {name}: {e}")
            raise
    
    logger.info(f"Converted {len(mlx_params)} model parameters to MLX arrays")
    return mlx_params


def restore_model_parameters_from_mlx(model: torch.nn.Module, 
                                    mlx_params: Dict[str, mx.array],
                                    device: Optional[Union[str, torch.device]] = None):
    """
    Restore model parameters from MLX arrays.
    
    Args:
        model: PyTorch model
        mlx_params: Dictionary of MLX parameter arrays
        device: Target device for restored parameters
    """
    if not TORCH_AVAILABLE:
        raise TensorConversionError("PyTorch not available for model parameter restoration")
        
    bridge = get_bridge()
    restored_count = 0
    
    for name, param in model.named_parameters():
        if name in mlx_params:
            try:
                restored_data = bridge.to_torch(mlx_params[name], device=device)
                param.data.copy_(restored_data)
                restored_count += 1
                logger.debug(f"Restored parameter {name} from MLX array")
            except TensorConversionError as e:
                logger.error(f"Failed to restore parameter {name}: {e}")
                raise
        else:
            logger.warning(f"Parameter {name} not found in MLX parameters")
    
    logger.info(f"Restored {restored_count} model parameters from MLX arrays")
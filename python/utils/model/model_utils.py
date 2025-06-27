"""
UnifiedTransformer Model Utilities

This module provides utilities for model analysis, debugging, and optimization
specifically designed for the UnifiedTransformer architecture.

Features:
- Model parameter analysis and counting
- Architecture visualization and inspection
- Gradient flow analysis
- Memory usage profiling per layer
- Model compression utilities
- Expert routing analysis for MoE models
- Attention pattern visualization
"""

import math
import time
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from collections import defaultdict, OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

# Optional imports
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

try:
    import mlx.core as mx
    import mlx.nn as mlx_nn
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

from ..logging.logger import get_logger

logger = get_logger(__name__)


class ModelAnalyzer:
    """Comprehensive model analysis and debugging utilities."""
    
    def __init__(self, model: nn.Module):
        """Initialize model analyzer."""
        self.model = model
        self.memory_monitor = None  # Lazy import to avoid circular dependencies
        self.layer_stats = {}
        self.gradient_stats = {}
        
    def _get_memory_monitor(self):
        """Lazy import of memory monitor to avoid circular dependencies."""
        if self.memory_monitor is None:
            from .profiling import get_memory_monitor
            self.memory_monitor = get_memory_monitor()
        return self.memory_monitor
        
    def analyze_parameters(self) -> Dict[str, Any]:
        """Analyze model parameters in detail."""
        stats = {
            "total_params": 0,
            "trainable_params": 0,
            "frozen_params": 0,
            "layer_breakdown": {},
            "parameter_types": defaultdict(int),
            "memory_usage_mb": 0
        }
        
        for name, param in self.model.named_parameters():
            param_count = get_tensor_element_count(param)
            stats["total_params"] += param_count

            if getattr(param, 'requires_grad', True):
                stats["trainable_params"] += param_count
            else:
                stats["frozen_params"] += param_count

            # Layer breakdown
            layer_name = name.split('.')[0] if '.' in name else name
            if layer_name not in stats["layer_breakdown"]:
                stats["layer_breakdown"][layer_name] = {
                    "params": 0,
                    "trainable": 0,
                    "frozen": 0
                }

            stats["layer_breakdown"][layer_name]["params"] += param_count
            if getattr(param, 'requires_grad', True):
                stats["layer_breakdown"][layer_name]["trainable"] += param_count
            else:
                stats["layer_breakdown"][layer_name]["frozen"] += param_count
                
            # Parameter type analysis
            param_type = str(param.dtype).split('.')[-1]
            stats["parameter_types"][param_type] += param_count
            
            # Memory usage (approximate)
            element_size = param.element_size()
            stats["memory_usage_mb"] += (param_count * element_size) / (1024 * 1024)
            
        return stats
        
    def analyze_model_structure(self) -> Dict[str, Any]:
        """Analyze model architecture structure."""
        structure = {
            "module_count": 0,
            "module_types": defaultdict(int),
            "depth": 0,
            "layer_sequence": [],
            "attention_heads": {},
            "hidden_sizes": {}
        }
        
        def analyze_module(module, prefix="", depth=0):
            structure["module_count"] += 1
            structure["depth"] = max(structure["depth"], depth)
            
            module_type = type(module).__name__
            structure["module_types"][module_type] += 1
            structure["layer_sequence"].append(f"{prefix}.{module_type}" if prefix else module_type)
            
            # Analyze specific layer types
            if hasattr(module, 'num_attention_heads'):
                structure["attention_heads"][prefix or "root"] = module.num_attention_heads
                
            if hasattr(module, 'hidden_size'):
                structure["hidden_sizes"][prefix or "root"] = module.hidden_size
                
            # Recursively analyze children
            for name, child in module.named_children():
                child_prefix = f"{prefix}.{name}" if prefix else name
                analyze_module(child, child_prefix, depth + 1)
                
        analyze_module(self.model)
        return structure
        
    def profile_forward_pass(self, 
                           input_data: torch.Tensor,
                           num_runs: int = 10) -> Dict[str, Any]:
        """Profile forward pass performance."""
        self.model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = self.model(input_data)
                
        # Profile
        times = []
        memory_usage = []
        
        for i in range(num_runs):
            memory_monitor = self._get_memory_monitor()
            memory_monitor.get_memory_snapshot(f"forward_pass_start_{i}")
            
            start_time = time.perf_counter()
            with torch.no_grad():
                output = self.model(input_data)
            end_time = time.perf_counter()
            
            times.append((end_time - start_time) * 1000)  # ms
            
            snapshot = memory_monitor.get_memory_snapshot(f"forward_pass_end_{i}")
            memory_usage.append(snapshot.process_memory_gb)
            
        return {
            "mean_time_ms": np.mean(times),
            "std_time_ms": np.std(times),
            "min_time_ms": np.min(times),
            "max_time_ms": np.max(times),
            "mean_memory_gb": np.mean(memory_usage),
            "peak_memory_gb": np.max(memory_usage),
            "throughput_samples_per_sec": 1000 / np.mean(times),
            "input_shape": list(input_data.shape),
            "output_shape": list(output.shape) if hasattr(output, 'shape') else "unknown"
        }
        
    def analyze_gradients(self) -> Dict[str, Any]:
        """Analyze gradient statistics after backward pass."""
        if not any(p.grad is not None for p in self.model.parameters()):
            logger.warning("No gradients found. Run backward pass first.")
            return {}
            
        grad_stats = {
            "layer_gradients": {},
            "gradient_norms": {},
            "zero_gradients": [],
            "large_gradients": [],
            "gradient_flow": {}
        }
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad = param.grad
                grad_norm = grad.norm().item()
                grad_mean = grad.mean().item()
                grad_std = grad.std().item()
                
                layer_name = name.split('.')[0] if '.' in name else name
                
                if layer_name not in grad_stats["layer_gradients"]:
                    grad_stats["layer_gradients"][layer_name] = {
                        "norms": [],
                        "means": [],
                        "stds": []
                    }
                    
                grad_stats["layer_gradients"][layer_name]["norms"].append(grad_norm)
                grad_stats["layer_gradients"][layer_name]["means"].append(grad_mean)
                grad_stats["layer_gradients"][layer_name]["stds"].append(grad_std)
                
                grad_stats["gradient_norms"][name] = grad_norm
                
                # Check for problematic gradients
                if grad_norm < 1e-7:
                    grad_stats["zero_gradients"].append(name)
                elif grad_norm > 10.0:
                    grad_stats["large_gradients"].append(name)
                    
        # Compute layer-wise statistics
        for layer_name, stats in grad_stats["layer_gradients"].items():
            grad_stats["gradient_flow"][layer_name] = {
                "mean_norm": np.mean(stats["norms"]),
                "total_norm": np.sum(stats["norms"]),
                "param_count": len(stats["norms"])
            }
            
        return grad_stats
        
    def check_model_health(self) -> Dict[str, Any]:
        """Comprehensive model health check."""
        health_report = {
            "parameter_health": {},
            "gradient_health": {},
            "warnings": [],
            "errors": [],
            "overall_status": "unknown"
        }
        
        # Check parameters
        param_stats = self.analyze_parameters()
        
        if param_stats["total_params"] == 0:
            health_report["errors"].append("Model has no parameters")
        elif param_stats["trainable_params"] == 0:
            health_report["warnings"].append("Model has no trainable parameters")
            
        # Check for NaN/Inf parameters
        nan_params = []
        inf_params = []
        
        for name, param in self.model.named_parameters():
            if torch.isnan(param).any():
                nan_params.append(name)
            if torch.isinf(param).any():
                inf_params.append(name)
                
        if nan_params:
            health_report["errors"].extend([f"NaN parameters in {name}" for name in nan_params])
        if inf_params:
            health_report["errors"].extend([f"Inf parameters in {name}" for name in inf_params])
            
        health_report["parameter_health"] = {
            "total_params": param_stats["total_params"],
            "trainable_ratio": param_stats["trainable_params"] / max(param_stats["total_params"], 1),
            "memory_usage_mb": param_stats["memory_usage_mb"],
            "nan_parameters": len(nan_params),
            "inf_parameters": len(inf_params)
        }
        
        # Check gradients if available
        grad_stats = self.analyze_gradients()
        if grad_stats:
            zero_grad_ratio = len(grad_stats["zero_gradients"]) / len(grad_stats["gradient_norms"])
            large_grad_ratio = len(grad_stats["large_gradients"]) / len(grad_stats["gradient_norms"])
            
            if zero_grad_ratio > 0.5:
                health_report["warnings"].append(f"High ratio of zero gradients: {zero_grad_ratio:.2%}")
            if large_grad_ratio > 0.1:
                health_report["warnings"].append(f"High ratio of large gradients: {large_grad_ratio:.2%}")
                
            health_report["gradient_health"] = {
                "zero_gradient_ratio": zero_grad_ratio,
                "large_gradient_ratio": large_grad_ratio,
                "mean_gradient_norm": np.mean(list(grad_stats["gradient_norms"].values()))
            }
            
        # Overall status
        if health_report["errors"]:
            health_report["overall_status"] = "error"
        elif health_report["warnings"]:
            health_report["overall_status"] = "warning"
        else:
            health_report["overall_status"] = "healthy"
            
        return health_report
        
    def generate_model_report(self, save_path: Optional[str] = None) -> str:
        """Generate comprehensive model analysis report."""
        param_stats = self.analyze_parameters()
        structure_stats = self.analyze_model_structure()
        health_report = self.check_model_health()
        
        report_lines = [
            "UnifiedTransformer Model Analysis Report",
            "=" * 50,
            "",
            "Parameter Analysis:",
            f"  Total parameters: {param_stats['total_params']:,}",
            f"  Trainable parameters: {param_stats['trainable_params']:,}",
            f"  Frozen parameters: {param_stats['frozen_params']:,}",
            f"  Memory usage: {param_stats['memory_usage_mb']:.1f} MB",
            "",
            "Architecture Structure:",
            f"  Total modules: {structure_stats['module_count']}",
            f"  Model depth: {structure_stats['depth']}",
            f"  Module types: {dict(structure_stats['module_types'])}",
            "",
            "Layer Breakdown:"
        ]
        
        for layer_name, layer_stats in param_stats["layer_breakdown"].items():
            report_lines.append(f"  {layer_name}: {layer_stats['params']:,} params "
                              f"({layer_stats['trainable']:,} trainable)")
                              
        report_lines.extend([
            "",
            "Health Status:",
            f"  Overall: {health_report['overall_status'].upper()}",
            f"  Warnings: {len(health_report['warnings'])}",
            f"  Errors: {len(health_report['errors'])}"
        ])
        
        if health_report["warnings"]:
            report_lines.append("  Warning details:")
            for warning in health_report["warnings"]:
                report_lines.append(f"    - {warning}")
                
        if health_report["errors"]:
            report_lines.append("  Error details:")
            for error in health_report["errors"]:
                report_lines.append(f"    - {error}")
                
        report = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            logger.info(f"Model report saved to {save_path}")
            
        return report


def get_tensor_element_count(tensor) -> int:
    """Get element count for both PyTorch tensors and MLX arrays.

    This is a backend-agnostic replacement for PyTorch's .numel() method.

    Args:
        tensor: PyTorch tensor, MLX array, or any tensor-like object

    Returns:
        int: Number of elements in the tensor
    """
    if hasattr(tensor, 'numel'):
        # PyTorch tensor
        return tensor.numel()
    elif hasattr(tensor, 'size'):
        # MLX array
        return tensor.size
    elif hasattr(tensor, 'shape'):
        # Fallback: calculate from shape
        import numpy as np
        return int(np.prod(tensor.shape))
    else:
        return 0


def is_scalar_tensor(tensor) -> bool:
    """Check if tensor is scalar for both PyTorch and MLX.

    Args:
        tensor: PyTorch tensor, MLX array, or any tensor-like object

    Returns:
        bool: True if tensor contains exactly one element
    """
    return get_tensor_element_count(tensor) == 1


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters in a model."""
    total_params = sum(get_tensor_element_count(p) for p in model.parameters())
    trainable_params = sum(get_tensor_element_count(p) for p in model.parameters()
                          if getattr(p, 'requires_grad', True))
    return total_params, trainable_params


def estimate_model_size(model: nn.Module, precision: str = "float32") -> float:
    """Estimate model size in MB."""
    total_params = sum(get_tensor_element_count(p) for p in model.parameters())

    bytes_per_param = {
        "float32": 4,
        "float16": 2,
        "int8": 1,
        "int4": 0.5
    }

    if precision not in bytes_per_param:
        logger.warning(f"Unknown precision {precision}, using float32")
        precision = "float32"

    size_mb = total_params * bytes_per_param[precision] / (1024 * 1024)
    return size_mb


def analyze_attention_patterns(attention_weights: torch.Tensor, 
                             save_path: Optional[str] = None) -> Dict[str, Any]:
    """Analyze attention patterns and optionally save visualization."""
    if not PLOTTING_AVAILABLE:
        logger.warning("Matplotlib not available for attention visualization")
        return {}
        
    # attention_weights shape: [batch, heads, seq_len, seq_len]
    batch_size, num_heads, seq_len, _ = attention_weights.shape
    
    # Average across batch and heads for analysis
    avg_attention = attention_weights.mean(dim=(0, 1)).cpu().numpy()
    
    stats = {
        "entropy": float(-np.sum(avg_attention * np.log(avg_attention + 1e-8))),
        "sparsity": float(np.sum(avg_attention < 0.01) / avg_attention.size),
        "diagonal_attention": float(np.mean(np.diag(avg_attention))),
        "max_attention": float(np.max(avg_attention)),
        "attention_spread": float(np.std(avg_attention))
    }
    
    if save_path:
        plt.figure(figsize=(10, 8))
        sns.heatmap(avg_attention, cmap='Blues', cbar=True)
        plt.title('Average Attention Pattern')
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Attention pattern saved to {save_path}")
        
    return stats


def compare_models(model1: nn.Module, model2: nn.Module) -> Dict[str, Any]:
    """Compare two models and return differences."""
    analyzer1 = ModelAnalyzer(model1)
    analyzer2 = ModelAnalyzer(model2)
    
    stats1 = analyzer1.analyze_parameters()
    stats2 = analyzer2.analyze_parameters()
    
    comparison = {
        "parameter_diff": stats2["total_params"] - stats1["total_params"],
        "trainable_diff": stats2["trainable_params"] - stats1["trainable_params"],
        "memory_diff_mb": stats2["memory_usage_mb"] - stats1["memory_usage_mb"],
        "size_ratio": stats2["total_params"] / max(stats1["total_params"], 1),
        "layer_differences": {}
    }
    
    # Compare layer breakdown
    all_layers = set(stats1["layer_breakdown"].keys()) | set(stats2["layer_breakdown"].keys())
    
    for layer in all_layers:
        layer1_params = stats1["layer_breakdown"].get(layer, {"params": 0})["params"]
        layer2_params = stats2["layer_breakdown"].get(layer, {"params": 0})["params"]
        
        comparison["layer_differences"][layer] = {
            "param_diff": layer2_params - layer1_params,
            "ratio": layer2_params / max(layer1_params, 1)
        }
        
    return comparison

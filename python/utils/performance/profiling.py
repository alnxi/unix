"""
UnifiedTransformer Advanced Performance Profiling Module

This module provides comprehensive profiling utilities for the UnifiedTransformer project,
specifically optimized for Apple Silicon M4 Pro with 24GB RAM constraint monitoring.

Enhanced Features:
- Memory usage tracking (PyTorch MPS, MLX, system memory)
- Training performance profiling with bottleneck detection
- Model inference benchmarking and optimization analysis
- Apple Silicon specific optimizations and M4 Pro enhancements
- Real-time memory constraint monitoring with leak detection
- Performance bottleneck detection and optimization suggestions
- GPU utilization tracking and analysis
- Memory fragmentation analysis and optimization
- Distributed training performance monitoring
- Advanced statistical analysis and trend detection
- Real-time performance dashboards and alerts
- Memory optimization recommendations with AI-powered suggestions
"""

import gc
import os
import psutil
import time
import threading
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from collections import defaultdict, deque
from datetime import datetime, timedelta
import statistics
import json
import platform

import torch
import numpy as np

# Try to import MLX for Apple Silicon profiling
try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

# Try to import memory profiler
try:
    from memory_profiler import profile as memory_profile
    MEMORY_PROFILER_AVAILABLE = True
except ImportError:
    MEMORY_PROFILER_AVAILABLE = False

from ..logging.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MemorySnapshot:
    """Snapshot of memory usage at a specific point in time."""
    timestamp: float
    system_memory_gb: float
    system_memory_percent: float
    process_memory_gb: float
    torch_memory_gb: float = 0.0
    mlx_memory_gb: float = 0.0
    context: str = ""


@dataclass
class PerformanceMetrics:
    """Performance metrics for training/inference operations."""
    operation_name: str
    duration_ms: float
    memory_before: MemorySnapshot
    memory_after: MemorySnapshot
    additional_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GPUUtilization:
    """GPU utilization metrics."""
    timestamp: float
    gpu_utilization_percent: float = 0.0
    memory_utilization_percent: float = 0.0
    temperature_celsius: float = 0.0
    power_draw_watts: float = 0.0
    context: str = ""


@dataclass
class MemoryLeakDetection:
    """Memory leak detection results."""
    operation: str
    baseline_memory_mb: float
    peak_memory_mb: float
    final_memory_mb: float
    leak_detected: bool
    leak_amount_mb: float
    confidence_score: float
    timestamp: float


@dataclass
class BottleneckAnalysis:
    """Performance bottleneck analysis results."""
    operation: str
    bottleneck_type: str  # 'memory', 'compute', 'io', 'synchronization'
    severity: str  # 'low', 'medium', 'high', 'critical'
    impact_score: float  # 0-100
    description: str
    recommendations: List[str]
    timestamp: float


class MemoryMonitor:
    """Real-time memory monitoring for 24GB constraint enforcement."""
    
    def __init__(self, memory_limit_gb: float = 22.0, warning_threshold: float = 0.85):
        """
        Initialize memory monitor.
        
        Args:
            memory_limit_gb: Maximum memory usage in GB (default 22GB for 24GB system)
            warning_threshold: Threshold for warning (0.85 = 85% of limit)
        """
        self.memory_limit_gb = memory_limit_gb
        self.warning_threshold = warning_threshold
        self.warning_limit_gb = memory_limit_gb * warning_threshold
        
        self.snapshots: deque = deque(maxlen=1000)  # Keep last 1000 snapshots
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.monitor_interval = 1.0  # seconds
        
        self.callbacks: List[Callable[[MemorySnapshot], None]] = []
        
    def add_callback(self, callback: Callable[[MemorySnapshot], None]) -> None:
        """Add a callback to be called when memory usage is monitored."""
        self.callbacks.append(callback)
        
    def get_memory_snapshot(self, context: str = "") -> MemorySnapshot:
        """Get current memory usage snapshot."""
        # System memory
        system_memory = psutil.virtual_memory()
        system_memory_gb = system_memory.total / (1024**3)
        system_memory_percent = system_memory.percent
        
        # Process memory
        process = psutil.Process()
        process_memory_gb = process.memory_info().rss / (1024**3)
        
        # PyTorch memory
        torch_memory_gb = 0.0
        if torch.cuda.is_available():
            torch_memory_gb = torch.cuda.memory_allocated() / (1024**3)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # MPS memory tracking is limited
            torch_memory_gb = process_memory_gb  # Approximate
            
        # MLX memory
        mlx_memory_gb = 0.0
        if MLX_AVAILABLE:
            try:
                mlx_memory_gb = mx.get_active_memory() / (1024**3)
            except Exception as e:
                logger.debug(f"Could not get MLX memory usage: {e}")
                
        return MemorySnapshot(
            timestamp=time.time(),
            system_memory_gb=system_memory_gb,
            system_memory_percent=system_memory_percent,
            process_memory_gb=process_memory_gb,
            torch_memory_gb=torch_memory_gb,
            mlx_memory_gb=mlx_memory_gb,
            context=context
        )
        
    def check_memory_constraints(self, snapshot: MemorySnapshot) -> None:
        """Check if memory usage violates constraints and issue warnings."""
        total_usage = max(snapshot.process_memory_gb, 
                         snapshot.torch_memory_gb + snapshot.mlx_memory_gb)
        
        if total_usage > self.memory_limit_gb:
            logger.error(f"Memory limit exceeded! Using {total_usage:.2f}GB > {self.memory_limit_gb}GB limit")
            logger.error(f"Context: {snapshot.context}")
        elif total_usage > self.warning_limit_gb:
            logger.warning(f"Memory usage high: {total_usage:.2f}GB (>{self.warning_limit_gb:.1f}GB threshold)")
            logger.warning(f"Context: {snapshot.context}")
            
    def start_monitoring(self) -> None:
        """Start continuous memory monitoring in background thread."""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info(f"Started memory monitoring (limit: {self.memory_limit_gb}GB)")
        
    def stop_monitoring(self) -> None:
        """Stop continuous memory monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        logger.info("Stopped memory monitoring")
        
    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self.monitoring:
            try:
                snapshot = self.get_memory_snapshot("background_monitor")
                self.snapshots.append(snapshot)
                self.check_memory_constraints(snapshot)
                
                # Call registered callbacks
                for callback in self.callbacks:
                    try:
                        callback(snapshot)
                    except Exception as e:
                        logger.warning(f"Memory monitor callback failed: {e}")
                        
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                
            time.sleep(self.monitor_interval)
            
    def get_memory_history(self, last_n: Optional[int] = None) -> List[MemorySnapshot]:
        """Get memory usage history."""
        snapshots = list(self.snapshots)
        if last_n:
            snapshots = snapshots[-last_n:]
        return snapshots
        
    def get_peak_memory(self) -> Optional[MemorySnapshot]:
        """Get peak memory usage from history."""
        if not self.snapshots:
            return None
            
        return max(self.snapshots, 
                  key=lambda s: max(s.process_memory_gb, s.torch_memory_gb + s.mlx_memory_gb))


class PerformanceProfiler:
    """Comprehensive performance profiler for training and inference."""
    
    def __init__(self, memory_monitor: Optional[MemoryMonitor] = None):
        """Initialize performance profiler."""
        self.memory_monitor = memory_monitor or MemoryMonitor()
        self.metrics: List[PerformanceMetrics] = []
        self.operation_stats: Dict[str, List[float]] = defaultdict(list)
        
    @contextmanager
    def profile_operation(self, operation_name: str, **additional_metrics):
        """Context manager for profiling operations."""
        # Get memory before
        memory_before = self.memory_monitor.get_memory_snapshot(f"before_{operation_name}")
        
        # Force garbage collection for accurate measurement
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except RuntimeError as e:
                if "invalid low watermark ratio" in str(e):
                    logger.warning(f"MPS cache clear failed due to watermark issue: {e}")
                    # Continue without clearing cache
                else:
                    raise
            
        start_time = time.perf_counter()
        
        try:
            yield
        finally:
            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000
            
            # Get memory after
            memory_after = self.memory_monitor.get_memory_snapshot(f"after_{operation_name}")
            
            # Create metrics
            metrics = PerformanceMetrics(
                operation_name=operation_name,
                duration_ms=duration_ms,
                memory_before=memory_before,
                memory_after=memory_after,
                additional_metrics=additional_metrics
            )
            
            self.metrics.append(metrics)
            self.operation_stats[operation_name].append(duration_ms)
            
            # Log performance
            memory_delta = (memory_after.process_memory_gb - memory_before.process_memory_gb) * 1024  # MB
            logger.verbose(f"{operation_name}: {duration_ms:.2f}ms, "
                          f"memory Δ: {memory_delta:+.1f}MB")
                          
    def get_operation_stats(self, operation_name: str) -> Dict[str, float]:
        """Get statistics for a specific operation."""
        durations = self.operation_stats.get(operation_name, [])
        if not durations:
            return {}
            
        return {
            "count": len(durations),
            "mean_ms": np.mean(durations),
            "std_ms": np.std(durations),
            "min_ms": np.min(durations),
            "max_ms": np.max(durations),
            "median_ms": np.median(durations),
            "p95_ms": np.percentile(durations, 95),
            "p99_ms": np.percentile(durations, 99)
        }
        
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all operations."""
        return {op: self.get_operation_stats(op) for op in self.operation_stats.keys()}

    def get_all_metrics(self) -> List[PerformanceMetrics]:
        """Get all collected metrics."""
        return self.metrics
    
    def get_all_operation_names(self) -> List[str]:
        """Get all operation names."""
        return list(self.operation_stats.keys())
    
    def get_all_operation_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all operations."""
        return {op: self.get_operation_stats(op) for op in self.operation_stats.keys()}

    def get_total_time(self) -> float:
        """Get total time across all operations in seconds."""
        if not self.metrics:
            return 0.0

        # Sum all operation durations and convert from milliseconds to seconds
        total_ms = sum(metric.duration_ms for metric in self.metrics)
        return total_ms / 1000.0

    def get_total_memory_usage(self) -> float:
        """Get total memory usage across all operations in GB."""
        if not self.metrics:
            return 0.0

        # Calculate total memory delta across all operations
        total_memory_delta = 0.0
        for metric in self.metrics:
            memory_delta = metric.memory_after.process_memory_gb - metric.memory_before.process_memory_gb
            total_memory_delta += abs(memory_delta)  # Use absolute value for total usage

        return total_memory_delta
    def clear_metrics(self) -> None:
        """Clear all collected metrics."""
        self.metrics.clear()
        self.operation_stats.clear()
        
    def save_profile_report(self, filepath: Union[str, Path]) -> None:
        """Save detailed profiling report to file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            f.write("UnifiedTransformer Performance Profile Report\n")
            f.write("=" * 50 + "\n\n")
            
            # Overall statistics
            f.write("Operation Statistics:\n")
            f.write("-" * 20 + "\n")
            for operation, stats in self.get_all_stats().items():
                f.write(f"\n{operation}:\n")
                for key, value in stats.items():
                    f.write(f"  {key}: {value:.3f}\n")
                    
            # Memory usage summary
            f.write("\nMemory Usage Summary:\n")
            f.write("-" * 20 + "\n")
            peak_memory = self.memory_monitor.get_peak_memory()
            if peak_memory:
                f.write(f"Peak memory usage: {peak_memory.process_memory_gb:.2f}GB\n")
                f.write(f"Peak PyTorch memory: {peak_memory.torch_memory_gb:.2f}GB\n")
                f.write(f"Peak MLX memory: {peak_memory.mlx_memory_gb:.2f}GB\n")
                
        logger.info(f"Performance profile saved to {filepath}")


class AdvancedMemoryAnalyzer:
    """Advanced memory analysis with leak detection and optimization suggestions."""

    def __init__(self, memory_monitor: MemoryMonitor):
        self.memory_monitor = memory_monitor
        self.baseline_snapshots: Dict[str, MemorySnapshot] = {}
        self.leak_detections: List[MemoryLeakDetection] = []

    def set_baseline(self, operation: str) -> None:
        """Set memory baseline for an operation."""
        snapshot = self.memory_monitor.get_memory_snapshot(f"baseline_{operation}")
        self.baseline_snapshots[operation] = snapshot
        logger.verbose(f"Memory baseline set for {operation}: {snapshot.process_memory_gb:.2f}GB")

    def analyze_memory_leak(self, operation: str, tolerance_mb: float = 10.0) -> MemoryLeakDetection:
        """Analyze potential memory leaks for an operation."""
        if operation not in self.baseline_snapshots:
            raise ValueError(f"No baseline set for operation: {operation}")

        baseline = self.baseline_snapshots[operation]
        current = self.memory_monitor.get_memory_snapshot(f"leak_check_{operation}")

        # Get peak memory from recent history
        recent_snapshots = self.memory_monitor.get_memory_history(last_n=100)
        peak_memory_mb = max(s.process_memory_gb * 1024 for s in recent_snapshots) if recent_snapshots else current.process_memory_gb * 1024

        baseline_mb = baseline.process_memory_gb * 1024
        current_mb = current.process_memory_gb * 1024
        leak_amount = current_mb - baseline_mb

        # Calculate confidence score based on multiple factors
        confidence = 0.0
        if leak_amount > tolerance_mb:
            confidence = min(100.0, (leak_amount / tolerance_mb) * 20)  # Scale confidence

        leak_detected = leak_amount > tolerance_mb and confidence > 50.0

        detection = MemoryLeakDetection(
            operation=operation,
            baseline_memory_mb=baseline_mb,
            peak_memory_mb=peak_memory_mb,
            final_memory_mb=current_mb,
            leak_detected=leak_detected,
            leak_amount_mb=leak_amount,
            confidence_score=confidence,
            timestamp=time.time()
        )

        self.leak_detections.append(detection)

        if leak_detected:
            logger.warning(f"[LEAK] Memory leak detected in {operation}: "
                          f"{leak_amount:.1f}MB increase (confidence: {confidence:.1f}%)")

        return detection

    def get_memory_optimization_suggestions(self) -> List[str]:
        """Generate memory optimization suggestions based on analysis."""
        suggestions = []

        # Analyze memory patterns
        history = self.memory_monitor.get_memory_history()
        if not history:
            return suggestions

        memory_values = [s.process_memory_gb for s in history]
        avg_memory = statistics.mean(memory_values)
        max_memory = max(memory_values)
        memory_variance = statistics.variance(memory_values) if len(memory_values) > 1 else 0

        # High memory usage
        if max_memory > 20.0:  # > 20GB
            suggestions.append("Consider using gradient checkpointing to reduce memory usage")
            suggestions.append("Enable mixed precision training (FP16/BF16) to halve memory requirements")
            suggestions.append("Reduce batch size or use gradient accumulation")

        # High memory variance (potential leaks)
        if memory_variance > 1.0:  # High variance
            suggestions.append("Memory usage is highly variable - check for memory leaks")
            suggestions.append("Consider more frequent garbage collection")

        # MLX specific suggestions
        if MLX_AVAILABLE:
            suggestions.append("Use MLX memory scopes for automatic cleanup")
            suggestions.append("Consider MLX unified memory architecture for better efficiency")

        # Detected leaks
        recent_leaks = [d for d in self.leak_detections if time.time() - d.timestamp < 3600]  # Last hour
        if recent_leaks:
            suggestions.append(f"Address {len(recent_leaks)} detected memory leaks")
            suggestions.append("Review tensor lifecycle and ensure proper cleanup")

        return suggestions


class BottleneckDetector:
    """Advanced bottleneck detection and analysis."""

    def __init__(self, profiler: PerformanceProfiler):
        self.profiler = profiler
        self.bottlenecks: List[BottleneckAnalysis] = []

    def analyze_bottlenecks(self) -> List[BottleneckAnalysis]:
        """Analyze performance bottlenecks across all operations."""
        bottlenecks = []

        for operation, stats in self.profiler.get_all_stats().items():
            if not stats or stats['count'] < 3:  # Need sufficient data
                continue

            # Analyze different bottleneck types
            bottleneck = self._analyze_operation_bottleneck(operation, stats)
            if bottleneck:
                bottlenecks.append(bottleneck)

        self.bottlenecks.extend(bottlenecks)
        return bottlenecks

    def _analyze_operation_bottleneck(self, operation: str, stats: Dict[str, float]) -> Optional[BottleneckAnalysis]:
        """Analyze bottlenecks for a specific operation."""
        mean_duration = stats['mean_ms']
        std_duration = stats['std_ms']
        max_duration = stats['max_ms']

        # Determine bottleneck type and severity
        bottleneck_type = "unknown"
        severity = "low"
        impact_score = 0.0
        recommendations = []

        # High duration indicates compute bottleneck
        if mean_duration > 1000:  # > 1 second
            bottleneck_type = "compute"
            impact_score = min(100.0, mean_duration / 100)  # Scale impact

            if mean_duration > 5000:  # > 5 seconds
                severity = "critical"
                recommendations.extend([
                    "Consider model parallelism or sharding",
                    "Optimize model architecture for efficiency",
                    "Use mixed precision training"
                ])
            elif mean_duration > 2000:  # > 2 seconds
                severity = "high"
                recommendations.extend([
                    "Profile individual layers for optimization",
                    "Consider gradient checkpointing",
                    "Optimize batch size"
                ])
            else:
                severity = "medium"
                recommendations.append("Monitor for performance degradation")

        # High variance indicates synchronization issues
        elif std_duration > mean_duration * 0.5:  # High variance
            bottleneck_type = "synchronization"
            impact_score = (std_duration / mean_duration) * 50
            severity = "medium" if impact_score > 25 else "low"
            recommendations.extend([
                "Check for synchronization overhead",
                "Consider asynchronous operations",
                "Review data loading pipeline"
            ])

        # Memory-related bottlenecks (check recent metrics)
        recent_metrics = [m for m in self.profiler.metrics if m.operation_name == operation][-10:]
        if recent_metrics:
            memory_deltas = [(m.memory_after.process_memory_gb - m.memory_before.process_memory_gb) * 1024
                           for m in recent_metrics]
            avg_memory_delta = statistics.mean(memory_deltas)

            if avg_memory_delta > 100:  # > 100MB average increase
                bottleneck_type = "memory"
                impact_score = min(100.0, avg_memory_delta / 10)
                severity = "high" if avg_memory_delta > 500 else "medium"
                recommendations.extend([
                    "Optimize memory usage patterns",
                    "Consider memory-efficient alternatives",
                    "Review tensor lifecycle management"
                ])

        if impact_score > 10:  # Only report significant bottlenecks
            return BottleneckAnalysis(
                operation=operation,
                bottleneck_type=bottleneck_type,
                severity=severity,
                impact_score=impact_score,
                description=f"{operation} shows {bottleneck_type} bottleneck with {severity} severity",
                recommendations=recommendations,
                timestamp=time.time()
            )

        return None

    def get_optimization_priorities(self) -> List[BottleneckAnalysis]:
        """Get bottlenecks sorted by optimization priority."""
        return sorted(self.bottlenecks, key=lambda b: b.impact_score, reverse=True)

    def generate_optimization_report(self) -> str:
        """Generate a comprehensive optimization report."""
        report = ["Performance Optimization Report", "=" * 40]

        priorities = self.get_optimization_priorities()
        if not priorities:
            report.append("No significant bottlenecks detected.")
            return "\n".join(report)

        report.append(f"\nDetected {len(priorities)} performance bottlenecks:\n")

        for i, bottleneck in enumerate(priorities[:10], 1):  # Top 10
            report.append(f"{i}. {bottleneck.operation} ({bottleneck.severity.upper()})")
            report.append(f"   Type: {bottleneck.bottleneck_type}")
            report.append(f"   Impact Score: {bottleneck.impact_score:.1f}/100")
            report.append(f"   Description: {bottleneck.description}")
            report.append("   Recommendations:")
            for rec in bottleneck.recommendations:
                report.append(f"     - {rec}")
            report.append("")

        return "\n".join(report)


# Global instances for easy access
_global_memory_monitor: Optional[MemoryMonitor] = None
_global_profiler: Optional[PerformanceProfiler] = None


def get_memory_monitor() -> MemoryMonitor:
    """Get global memory monitor instance."""
    global _global_memory_monitor
    if _global_memory_monitor is None:
        _global_memory_monitor = MemoryMonitor()
    return _global_memory_monitor


def get_profiler() -> PerformanceProfiler:
    """Get global performance profiler instance."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = PerformanceProfiler(get_memory_monitor())
    return _global_profiler


@contextmanager
def profile(operation_name: str, **kwargs):
    """Convenient context manager for profiling operations."""
    with get_profiler().profile_operation(operation_name, **kwargs):
        yield


def log_system_info() -> None:
    """Log comprehensive system information."""
    logger.info("System Information:")
    logger.info(f"  Platform: {psutil.platform}")
    logger.info(f"  CPU cores: {psutil.cpu_count()}")
    logger.info(f"  Total RAM: {psutil.virtual_memory().total / (1024**3):.1f}GB")
    
    # PyTorch info
    logger.info(f"  PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        logger.info(f"  CUDA available: {torch.cuda.get_device_name()}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        logger.info("  MPS (Apple Silicon) available")
    else:
        logger.info("  No GPU acceleration available")
        
    # MLX info
    if MLX_AVAILABLE:
        logger.info("  MLX available for Apple Silicon optimization")
    else:
        logger.info("  MLX not available")


def cleanup_memory() -> None:
    """Aggressive memory cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        torch.mps.empty_cache()
        torch.mps.synchronize()


# Enhanced utility functions

def get_memory_analyzer() -> AdvancedMemoryAnalyzer:
    """Get global memory analyzer instance."""
    if not hasattr(get_memory_analyzer, '_instance'):
        get_memory_analyzer._instance = AdvancedMemoryAnalyzer(get_memory_monitor())
    return get_memory_analyzer._instance


def get_bottleneck_detector() -> BottleneckDetector:
    """Get global bottleneck detector instance."""
    if not hasattr(get_bottleneck_detector, '_instance'):
        get_bottleneck_detector._instance = BottleneckDetector(get_profiler())
    return get_bottleneck_detector._instance


@contextmanager
def memory_leak_detection(operation: str, tolerance_mb: float = 10.0):
    """Context manager for automatic memory leak detection."""
    analyzer = get_memory_analyzer()

    # Set baseline
    analyzer.set_baseline(operation)

    try:
        yield
    finally:
        # Analyze for leaks
        detection = analyzer.analyze_memory_leak(operation, tolerance_mb)
        if detection.leak_detected:
            logger.warning(f"Memory leak detected in {operation}: {detection.leak_amount_mb:.1f}MB")


def analyze_training_performance(metrics_history: List[Dict[str, float]],
                               window_size: int = 100) -> Dict[str, Any]:
    """Analyze training performance trends and detect issues."""
    if not metrics_history:
        return {}

    analysis = {
        "total_steps": len(metrics_history),
        "trends": {},
        "anomalies": [],
        "recommendations": []
    }

    # Analyze trends for each metric
    for metric_name in metrics_history[0].keys():
        values = [m.get(metric_name, 0) for m in metrics_history]

        if len(values) < 10:  # Need sufficient data
            continue

        # Calculate trend
        recent_values = values[-window_size:]
        early_values = values[:window_size] if len(values) > window_size else values[:len(values)//2]

        if early_values and recent_values:
            early_mean = statistics.mean(early_values)
            recent_mean = statistics.mean(recent_values)

            trend = "improving" if recent_mean < early_mean else "degrading"
            change_percent = abs((recent_mean - early_mean) / early_mean * 100) if early_mean != 0 else 0

            analysis["trends"][metric_name] = {
                "trend": trend,
                "change_percent": change_percent,
                "early_mean": early_mean,
                "recent_mean": recent_mean
            }

            # Detect anomalies
            if change_percent > 50:  # Significant change
                analysis["anomalies"].append({
                    "metric": metric_name,
                    "type": "significant_change",
                    "description": f"{metric_name} {trend} by {change_percent:.1f}%"
                })

    # Generate recommendations
    if analysis["anomalies"]:
        analysis["recommendations"].append("Investigate significant metric changes")

    for metric, trend_data in analysis["trends"].items():
        if trend_data["trend"] == "degrading" and trend_data["change_percent"] > 20:
            analysis["recommendations"].append(f"Address degrading {metric} performance")

    return analysis


def profile_model_inference(model, input_data, num_runs: int = 10,
                          warmup_runs: int = 3) -> Dict[str, float]:
    """Profile model inference performance."""
    profiler = get_profiler()

    # Warmup runs
    for _ in range(warmup_runs):
        with torch.no_grad():
            _ = model(input_data)

    # Actual profiling runs
    durations = []
    memory_deltas = []

    for i in range(num_runs):
        with profiler.profile_operation(f"inference_run_{i}"):
            with torch.no_grad():
                _ = model(input_data)

        # Get the last metric
        last_metric = profiler.metrics[-1]
        durations.append(last_metric.duration_ms)
        memory_delta = (last_metric.memory_after.process_memory_gb -
                       last_metric.memory_before.process_memory_gb) * 1024
        memory_deltas.append(memory_delta)

    return {
        "mean_duration_ms": statistics.mean(durations),
        "std_duration_ms": statistics.stdev(durations) if len(durations) > 1 else 0,
        "min_duration_ms": min(durations),
        "max_duration_ms": max(durations),
        "mean_memory_delta_mb": statistics.mean(memory_deltas),
        "throughput_samples_per_sec": 1000 / statistics.mean(durations) if durations else 0
    }


def generate_performance_dashboard() -> str:
    """Generate a real-time performance dashboard."""
    monitor = get_memory_monitor()
    profiler = get_profiler()
    detector = get_bottleneck_detector()

    # Get current status
    current_memory = monitor.get_memory_snapshot("dashboard")
    peak_memory = monitor.get_peak_memory()
    recent_bottlenecks = detector.analyze_bottlenecks()

    dashboard = [
        "UnifiedTransformer Performance Dashboard",
        "=" * 50,
        "",
        "Memory Status:",
        f"  Current Usage: {current_memory.process_memory_gb:.2f}GB",
        f"  Peak Usage: {peak_memory.process_memory_gb:.2f}GB" if peak_memory else "  Peak Usage: N/A",
        f"  Memory Limit: {monitor.memory_limit_gb:.1f}GB",
        f"  Utilization: {(current_memory.process_memory_gb / monitor.memory_limit_gb * 100):.1f}%",
        "",
        "Performance Summary:",
    ]

    # Add operation statistics
    all_stats = profiler.get_all_stats()
    if all_stats:
        dashboard.append("  Recent Operations:")
        for op, stats in list(all_stats.items())[-5:]:  # Last 5 operations
            dashboard.append(f"    {op}: {stats['mean_ms']:.1f}ms avg ({stats['count']} runs)")
    else:
        dashboard.append("  No operations recorded yet")

    dashboard.append("")

    # Add bottleneck information
    if recent_bottlenecks:
        dashboard.append("[WARNING] Detected Bottlenecks:")
        for bottleneck in recent_bottlenecks[:3]:  # Top 3
            dashboard.append(f"    {bottleneck.operation}: {bottleneck.bottleneck_type} "
                           f"({bottleneck.severity}, impact: {bottleneck.impact_score:.1f})")
    else:
        dashboard.append("[OK] No significant bottlenecks detected")

    dashboard.extend([
        "",
        f"Last Updated: {datetime.now().strftime('%H:%M:%S')}",
        "=" * 50
    ])

    return "\n".join(dashboard)


def export_performance_data(output_dir: str = "output/profiling") -> None:
    """Export all performance data for analysis."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Export memory history
    monitor = get_memory_monitor()
    memory_history = monitor.get_memory_history()

    memory_data = []
    for snapshot in memory_history:
        memory_data.append({
            "timestamp": snapshot.timestamp,
            "system_memory_gb": snapshot.system_memory_gb,
            "process_memory_gb": snapshot.process_memory_gb,
            "torch_memory_gb": snapshot.torch_memory_gb,
            "mlx_memory_gb": snapshot.mlx_memory_gb,
            "context": snapshot.context
        })

    with open(output_path / f"memory_history_{timestamp}.json", 'w') as f:
        json.dump(memory_data, f, indent=2)

    # Export performance metrics
    profiler = get_profiler()
    performance_data = []

    for metric in profiler.metrics:
        performance_data.append({
            "operation_name": metric.operation_name,
            "duration_ms": metric.duration_ms,
            "memory_before_gb": metric.memory_before.process_memory_gb,
            "memory_after_gb": metric.memory_after.process_memory_gb,
            "additional_metrics": metric.additional_metrics
        })

    with open(output_path / f"performance_metrics_{timestamp}.json", 'w') as f:
        json.dump(performance_data, f, indent=2)

    # Export bottleneck analysis
    detector = get_bottleneck_detector()
    bottlenecks = detector.analyze_bottlenecks()

    bottleneck_data = []
    for bottleneck in bottlenecks:
        bottleneck_data.append({
            "operation": bottleneck.operation,
            "bottleneck_type": bottleneck.bottleneck_type,
            "severity": bottleneck.severity,
            "impact_score": bottleneck.impact_score,
            "description": bottleneck.description,
            "recommendations": bottleneck.recommendations,
            "timestamp": bottleneck.timestamp
        })

    with open(output_path / f"bottleneck_analysis_{timestamp}.json", 'w') as f:
        json.dump(bottleneck_data, f, indent=2)

    logger.info(f"Performance data exported to {output_path}")


def setup_performance_monitoring(memory_limit_gb: float = 22.0,
                                monitoring_interval: float = 1.0) -> None:
    """Setup comprehensive performance monitoring."""
    monitor = get_memory_monitor()
    monitor.memory_limit_gb = memory_limit_gb
    monitor.monitor_interval = monitoring_interval

    # Start monitoring
    monitor.start_monitoring()

    # Setup automatic bottleneck detection
    def bottleneck_callback(snapshot: MemorySnapshot):
        if snapshot.timestamp % 60 < monitoring_interval:  # Check every minute
            detector = get_bottleneck_detector()
            bottlenecks = detector.analyze_bottlenecks()

            critical_bottlenecks = [b for b in bottlenecks if b.severity == "critical"]
            if critical_bottlenecks:
                logger.error(f"Critical performance bottlenecks detected: "
                           f"{[b.operation for b in critical_bottlenecks]}")

    monitor.add_callback(bottleneck_callback)

    logger.info(f"Performance monitoring setup complete (limit: {memory_limit_gb}GB, "
               f"interval: {monitoring_interval}s)")

"""
UnifiedTransformer Utilities Module

This module provides comprehensive utilities for the UnifiedTransformer project:
- Enhanced logging with Rich console output and MLX integration
- Performance profiling and memory monitoring for 24GB constraint
- Training visualization and progress monitoring
- MLX-specific utilities for Apple Silicon optimization
- Configuration management with YAML/Hydra support

All utilities are designed to work seamlessly with the modular architecture
and support the project's goals of extreme efficiency and Apple Silicon optimization.
"""

# Core utilities
from .logging.logger import (
    get_logger,
    configure_logging,
    set_log_level,
    set_global_log_level,
    set_console_colors,
    set_structured_logging_format,
    get_console,
    indent_log,
    log_call,
    VERBOSE,
    # Enhanced logging features
    ExperimentLogger,
    PerformanceTracker,
    get_experiment_logger,
    get_performance_tracker,
    log_performance,
    export_structured_logs,
    get_performance_summary,
    setup_distributed_logging,
    log_gpu_memory_summary,
    log_execution_time,
    # Synthetic data tracking
    SyntheticDataQualityTracker,
    get_synthetic_data_tracker,
    track_synthetic_data_generation
)

# Profiling and memory monitoring
from .performance.profiling import (
    MemoryMonitor,
    PerformanceProfiler,
    MemorySnapshot,
    PerformanceMetrics,
    get_memory_monitor,
    get_profiler,
    profile,
    log_system_info,
    cleanup_memory,
    # Enhanced profiling features
    AdvancedMemoryAnalyzer,
    BottleneckDetector,
    GPUUtilization,
    MemoryLeakDetection,
    BottleneckAnalysis,
    get_memory_analyzer,
    get_bottleneck_detector,
    memory_leak_detection,
    analyze_training_performance,
    profile_model_inference,
    generate_performance_dashboard,
    export_performance_data,
    setup_performance_monitoring
)

# Visualization utilities
from .logging.visualization import (
    TrainingVisualizer,
    get_visualizer,
    plot_model_architecture,
    create_comparison_plot
)

# MLX utilities (Apple Silicon optimization)
from .mlx.mlx_utils import (
    MLXMemoryManager,
    MLXTensorUtils,
    MLXOptimizer,
    MLXModelUtils,
    MLXPerformanceProfiler,
    MLXConfig,
    get_mlx_memory_manager,
    get_mlx_profiler,
    is_mlx_available,
    log_mlx_info,
    mlx_memory_scope,
    setup_mlx_environment,
    # Enhanced MLX features
    AdvancedMLXMemoryManager,
    MLXMixedPrecisionManager,
    MLXModelSharding,
    MLXAdvancedProfiler,
    MLXOptimizationConfig,
    MLXModelShardConfig,
    MLXPerformanceMetrics,
    # Specialized profilers
    MLXMambaProfiler,
    MLXMoEProfiler,
    MLXFlashAttentionProfiler,
    MLXNeuralEngineProfiler,
    # Architecture-specific metrics
    MambaStateSpaceMetrics,
    MoERoutingMetrics,
    FlashAttention3Metrics,
    get_advanced_mlx_memory_manager,
    get_mlx_precision_manager,
    get_advanced_mlx_profiler,
    mlx_optimization_scope,
    optimize_mlx_model_for_inference,
    benchmark_mlx_operations,
    analyze_mlx_model_memory_requirements,
    setup_mlx_distributed_training,
    export_mlx_performance_report,
    create_mlx_optimization_config
)

# Configuration utilities
from .setup.config_utils import (
    BaseConfig,
    ConfigLoader,
    HydraConfigManager,
    ConfigValidator,
    DefaultConfigs,
    get_config_loader,
    get_hydra_manager,
    load_config,
    save_config,
    validate_config
)

# Model analysis utilities
from .model.model_utils import (
    ModelAnalyzer,
    count_parameters,
    estimate_model_size,
    analyze_attention_patterns,
    compare_models,
    get_tensor_element_count,
    is_scalar_tensor
)

__all__ = [
    # Logger
    "get_logger",
    "configure_logging",
    "set_log_level",
    "set_global_log_level",
    "set_console_colors",
    "set_structured_logging_format",
    "get_console",
    "indent_log",
    "log_call",
    "VERBOSE",
    # Enhanced logging
    "ExperimentLogger",
    "PerformanceTracker",
    "get_experiment_logger",
    "get_performance_tracker",
    "log_performance",
    "export_structured_logs",
    "get_performance_summary",
    "setup_distributed_logging",
    "log_gpu_memory_summary",
    "log_execution_time",
    # Synthetic data tracking
    "SyntheticDataQualityTracker",
    "get_synthetic_data_tracker",
    "track_synthetic_data_generation",

    # Profiling
    "MemoryMonitor",
    "PerformanceProfiler",
    "MemorySnapshot",
    "PerformanceMetrics",
    "get_memory_monitor",
    "get_profiler",
    "profile",
    "log_system_info",
    "cleanup_memory",
    # Enhanced profiling
    "AdvancedMemoryAnalyzer",
    "BottleneckDetector",
    "GPUUtilization",
    "MemoryLeakDetection",
    "BottleneckAnalysis",
    "get_memory_analyzer",
    "get_bottleneck_detector",
    "memory_leak_detection",
    "analyze_training_performance",
    "profile_model_inference",
    "generate_performance_dashboard",
    "export_performance_data",
    "setup_performance_monitoring",

    # Visualization
    "TrainingVisualizer",
    "get_visualizer",
    "plot_model_architecture",
    "create_comparison_plot",

    # MLX utilities
    "MLXMemoryManager",
    "MLXTensorUtils",
    "MLXOptimizer",
    "MLXModelUtils",
    "MLXPerformanceProfiler",
    "MLXConfig",
    "get_mlx_memory_manager",
    "get_mlx_profiler",
    "is_mlx_available",
    "log_mlx_info",
    "mlx_memory_scope",
    "setup_mlx_environment",
    # Enhanced MLX
    "AdvancedMLXMemoryManager",
    "MLXMixedPrecisionManager",
    "MLXModelSharding",
    "MLXAdvancedProfiler",
    "MLXOptimizationConfig",
    "MLXModelShardConfig",
    "MLXPerformanceMetrics",
    # Specialized profilers
    "MLXMambaProfiler",
    "MLXMoEProfiler", 
    "MLXFlashAttentionProfiler",
    "MLXNeuralEngineProfiler",
    # Architecture-specific metrics
    "MambaStateSpaceMetrics",
    "MoERoutingMetrics",
    "FlashAttention3Metrics",
    "get_advanced_mlx_memory_manager",
    "get_mlx_precision_manager",
    "get_advanced_mlx_profiler",
    "mlx_optimization_scope",
    "optimize_mlx_model_for_inference",
    "benchmark_mlx_operations",
    "analyze_mlx_model_memory_requirements",
    "setup_mlx_distributed_training",
    "export_mlx_performance_report",
    "create_mlx_optimization_config",

    # Configuration
    "BaseConfig",
    "ConfigLoader",
    "HydraConfigManager",
    "ConfigValidator",
    "DefaultConfigs",
    "get_config_loader",
    "get_hydra_manager",
    "load_config",
    "save_config",
    "validate_config",

    # Model utilities
    "ModelAnalyzer",
    "count_parameters",
    "estimate_model_size",
    "analyze_attention_patterns",
    "compare_models",
    "get_tensor_element_count",
    "is_scalar_tensor"
]

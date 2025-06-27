# Utility Modules

This directory contains comprehensive utility modules designed to support the development of advanced multimodal AI systems optimized for Apple Silicon.

## Overview

The utilities are designed with the following principles:
- **Apple Silicon Optimization**: MLX integration and Metal Performance Shaders support
- **Memory Constraint Awareness**: 24GB RAM monitoring and optimization
- **Production Ready**: Robust error handling and comprehensive logging
- **Modular Design**: Each utility can be used independently
- **Rich Console Output**: Beautiful terminal interfaces with Rich library

## Modules

### Logger (`logger.py`)
Enhanced logging system with Rich console output and MLX integration.

**Features:**
- Multiple log levels (DEBUG, VERBOSE, INFO, WARNING, ERROR, CRITICAL)
- Rich color-coded console output with syntax highlighting
- File logging with rotation
- Memory usage tracking
- Model information logging
- Context manager for indented logging

**Usage:**
```python
from src.utils import get_logger, indent_log

logger = get_logger(__name__)
logger.info("Starting training")

with indent_log(4):
    logger.verbose("Detailed information")
    logger.log_memory_usage("training_step")
```

### Profiling (`profiling.py`)
Performance profiling and memory monitoring for Apple Silicon.

**Features:**
- Real-time memory monitoring with 24GB constraint enforcement
- Training/inference performance profiling
- Apple Silicon specific optimizations
- Memory snapshot tracking
- Performance bottleneck detection

**Usage:**
```python
from src.utils import get_memory_monitor, profile

# Memory monitoring
monitor = get_memory_monitor()
monitor.start_monitoring()

# Performance profiling
with profile("forward_pass"):
    output = model(input_data)
```

### Visualization (`visualization.py`)
Training visualization and progress monitoring.

**Features:**
- Real-time training progress visualization
- Memory usage plots
- Rich console dashboards
- Integration with wandb/tensorboard
- Training curve plotting
- Performance metrics dashboards

**Usage:**
```python
from src.utils import get_visualizer

visualizer = get_visualizer("my_experiment")
visualizer.log_metric("loss", 0.5)
visualizer.plot_training_curves(["loss", "accuracy"])
```

### MLX Utils (`mlx_utils.py`)
Apple Silicon optimization utilities with MLX integration.

**Features:**
- MLX memory management
- PyTorch-MLX tensor conversion
- Apple Silicon performance monitoring
- Metal Performance Shaders integration
- Unified memory architecture utilities

**Usage:**
```python
from src.utils import setup_mlx_environment, mlx_memory_scope

# Setup MLX environment
config = setup_mlx_environment()

# Memory management
with mlx_memory_scope():
    # MLX operations here
    pass
```

### Configuration (`config_utils.py`)
Configuration management with YAML/Hydra support.

**Features:**
- YAML configuration loading with inheritance
- Environment variable substitution
- Hydra integration for experiment management
- Configuration validation
- Default configuration templates

**Usage:**
```python
from src.utils import load_config, get_hydra_manager

# Load configuration
config = load_config("model/base.yaml")

# Hydra integration
hydra_manager = get_hydra_manager()
cfg = hydra_manager.compose_config("experiment=baseline")
```

### Model Utils (`model_utils.py`)
Model analysis and debugging utilities.

**Features:**
- Parameter counting and analysis
- Model architecture inspection
- Gradient flow analysis
- Memory usage profiling per layer
- Attention pattern visualization
- Model health checking

**Usage:**
```python
from src.utils import ModelAnalyzer, count_parameters

# Parameter analysis
total_params, trainable_params = count_parameters(model)

# Comprehensive analysis
analyzer = ModelAnalyzer(model)
report = analyzer.generate_model_report()
health = analyzer.check_model_health()
```

## Installation

The utilities require the following dependencies:

### Core Dependencies
```bash
pip install torch torchvision torchaudio
pip install rich matplotlib seaborn
pip install psutil numpy scipy
pip install pyyaml omegaconf
```

### Optional Dependencies
```bash
# For MLX (Apple Silicon)
pip install mlx mlx-lm

# For Hydra configuration management
pip install hydra-core

# For advanced monitoring
pip install wandb tensorboard
pip install memory-profiler line-profiler

# For development
pip install pytest pytest-cov
```

## Quick Start

```python
# Import all utilities
from src.utils import (
    get_logger, get_memory_monitor, get_visualizer,
    setup_mlx_environment, load_config, ModelAnalyzer
)

# Setup logging
logger = get_logger(__name__)
logger.info("Starting application")

# Setup memory monitoring
monitor = get_memory_monitor()
monitor.start_monitoring()

# Setup MLX (if available)
if is_mlx_available():
    mlx_config = setup_mlx_environment()
    logger.info(f"MLX configured: {mlx_config}")

# Load configuration
config = load_config("configs/model/base.yaml")

# Setup visualization
visualizer = get_visualizer("my_experiment")

# Your training/inference code here...
```

## Testing

Run the test suite using `pytest` from the repository root:

```bash
pytest
```

## Architecture Integration

The utilities are designed to integrate seamlessly with the model architecture:

- **Core Module**: Configuration and base classes use config utilities
- **Training Module**: Uses profiling, visualization, and logging
- **Architectures**: Use model utilities for analysis
- **Compression**: Uses profiling for optimization tracking
- **MLX Integration**: All modules support Apple Silicon optimization

## Memory Management

Special attention is paid to the 24GB RAM constraint:

- **Automatic Monitoring**: Real-time memory usage tracking
- **Warning System**: Alerts when approaching memory limits
- **Cleanup Utilities**: Aggressive memory cleanup functions
- **MLX Optimization**: Apple Silicon specific memory management

## Performance Optimization

The utilities are optimized for training efficiency:

- **Minimal Overhead**: Profiling with negligible performance impact
- **Async Logging**: Non-blocking log operations
- **Efficient Visualization**: Optimized plotting and dashboard updates
- **Apple Silicon**: Native MLX integration for maximum performance

## Contributing

When adding new utilities:

1. Follow the existing code structure and documentation style
2. Add comprehensive error handling and logging
3. Include type hints and docstrings
4. Add tests to `test_utils.py`
5. Update this README with new functionality
6. Ensure Apple Silicon compatibility where applicable

## License

Part of the project. See the main LICENSE for details.

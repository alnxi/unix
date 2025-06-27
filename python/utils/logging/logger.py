"""
Advanced Logging Module

This module provides a comprehensive logging solution for machine learning projects.
It includes:
- Multiple log levels (DEBUG, INFO, VERBOSE, WARNING, ERROR, CRITICAL)
- Console and file logging with configurable formats
- Rich color-coded console output with syntax highlighting
- Log file rotation and structured JSON logging
- Context manager for indented logging
- Integration with the project's configuration system
- MLX-specific logging for Apple Silicon optimization
- Memory usage tracking for 24GB constraint monitoring
- Distributed training support with process coordination
- Performance metrics integration and timing
- Advanced tensor logging with statistics
- Experiment tracking integration (W&B, TensorBoard)
- Structured logging for better analysis and parsing
"""

import datetime
import functools
import inspect
import json
import logging
import multiprocessing
import os
import sys
import threading
import time
import warnings
from contextlib import contextmanager
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, Union, cast
from collections import defaultdict, deque

import torch
import numpy as np
import psutil
import platform

# Import Rich for beautiful console output
try:
    import rich
    from rich.console import Console
    from rich.highlighter import ReprHighlighter
    from rich.logging import RichHandler
    from rich.theme import Theme
    from rich.traceback import install as install_rich_traceback

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    # Define dummy classes for type annotations when Rich is not available
    class Console:
        pass
    class RichHandler:
        pass

# MLX is an optional dependency
try:
    import mlx.core as mx
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

# Custom log level between DEBUG and INFO
VERBOSE = 15
logging.addLevelName(VERBOSE, "VERBOSE")

# Thread-local storage for indentation
_log_state = threading.local()

# Default theme for Rich
DEFAULT_THEME = {
    "logging.level.debug": "dim cyan",
    "logging.level.verbose": "bright_blue",
    "logging.level.info": "green",
    "logging.level.warning": "yellow",
    "logging.level.error": "bold red",
    "logging.level.critical": "bold white on red",
    "repr.number": "cyan",
    "repr.str": "green",
    "repr.bool": "bold magenta",
    "repr.none": "dim",
    "repr.url": "underline bright_blue",
    "repr.path": "underline bright_magenta",
    "repr.filename": "bright_magenta",
    "repr.tag_name": "bright_blue",
    "repr.attrib_name": "bright_yellow",
    "log.time": "bright_black",
    "log.message": "",
}

# Global state (managed by LoggingManager)
_RICH_BANNER_PRINTED_MAIN_PROCESS_ONLY = False
_ACTIVE_PROGRESS = None

# Performance tracking
_PERFORMANCE_METRICS = defaultdict(list)
_TIMING_STACK = []

# Experiment tracking
_EXPERIMENT_LOGGERS = {}

# Structured logging
_STRUCTURED_LOGS = deque(maxlen=10000)


class ApplicationLogger(logging.Logger):
    """Custom logger class with additional log levels and features."""

    def verbose(self, msg: str, *args, **kwargs) -> None:
        """Log at VERBOSE level (between DEBUG and INFO)."""
        if self.isEnabledFor(VERBOSE):
            self.log(VERBOSE, msg, *args, **kwargs)

    def success(self, msg: str, *args, **kwargs) -> None:
        """Log a success message at INFO level with special formatting."""
        if self.isEnabledFor(logging.INFO):
            # Add a success prefix to the message
            msg = f"✓ {msg}"
            self.info(msg, *args, **kwargs)

    def log_tensor_shape(self, name: str, tensor_or_shape, *args, **kwargs) -> None:
        """Log tensor shape at VERBOSE level."""
        if self.isEnabledFor(VERBOSE):
            shape = (
                tensor_or_shape.shape
                if hasattr(tensor_or_shape, "shape")
                else tensor_or_shape
            )
            self.verbose(f"{name} shape: {shape}", *args, **kwargs)

    def log_memory_usage(self, context: str = "", *args, **kwargs) -> None:
        """Log current memory usage (PyTorch and MLX if available)."""
        if self.isEnabledFor(VERBOSE):
            memory_info = []

            # PyTorch memory
            if torch.cuda.is_available():
                memory_info.append(f"CUDA: {torch.cuda.memory_allocated() / 1024**3:.2f}GB")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                # MPS memory tracking is limited, but we can try
                memory_info.append("MPS: Available")

            # MLX memory if available
            if MLX_AVAILABLE:
                try:
                    mlx_memory = mx.get_active_memory() / 1024**3
                    memory_info.append(f"MLX: {mlx_memory:.2f}GB")
                except Exception as e:
                    self.debug(f"Could not get MLX memory usage: {e}")
                    memory_info.append("MLX: Available")

            memory_str = " | ".join(memory_info) if memory_info else "No GPU memory info"
            self.verbose(f"Memory usage {context}: {memory_str}", *args, **kwargs)

    def log_model_info(self, model, name: str = "Model", *args, **kwargs) -> None:
        """Log model information including parameter count and memory usage."""
        if self.isEnabledFor(logging.INFO):
            try:
                if hasattr(model, 'parameters'):
                    total_params = sum(p.numel() for p in model.parameters())
                    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

                    self.info(f"{name} - Total params: {total_params:,} | "
                             f"Trainable: {trainable_params:,} | "
                             f"Size: ~{total_params * 4 / 1024**2:.1f}MB", *args, **kwargs)
                else:
                    self.info(f"{name} - Parameter info not available", *args, **kwargs)
            except Exception as e:
                self.warning(f"Failed to log model info for {name}: {e}", *args, **kwargs)

    def log_tensor_stats(self, name: str, tensor, *args, **kwargs) -> None:
        """Log detailed tensor statistics."""
        if self.isEnabledFor(VERBOSE):
            try:
                if hasattr(tensor, 'shape'):
                    stats = {
                        'shape': tuple(tensor.shape),
                        'dtype': str(tensor.dtype),
                        'device': str(tensor.device) if hasattr(tensor, 'device') else 'unknown',
                        'numel': tensor.numel() if hasattr(tensor, 'numel') else 'unknown',
                        'requires_grad': getattr(tensor, 'requires_grad', False)
                    }

                    # Add numerical statistics if tensor is numeric
                    if hasattr(tensor, 'mean') and tensor.numel() > 0:
                        try:
                            stats.update({
                                'mean': float(tensor.mean()),
                                'std': float(tensor.std()),
                                'min': float(tensor.min()),
                                'max': float(tensor.max()),
                                'has_nan': bool(torch.isnan(tensor).any()) if hasattr(torch, 'isnan') else False,
                                'has_inf': bool(torch.isinf(tensor).any()) if hasattr(torch, 'isinf') else False
                            })
                        except Exception:
                            # Not all tensors will support these operations (e.g., non-numeric)
                            pass

                    self.verbose(f"{name} stats: {stats}", *args, **kwargs)
                else:
                    self.verbose(f"{name}: {type(tensor)}", *args, **kwargs)
            except Exception as e:
                self.warning(f"Failed to log tensor stats for {name}: {e}", *args, **kwargs)

    def log_performance(self, operation: str, duration_ms: float, memory_delta_mb: float = 0,
                       **metrics) -> None:
        """Log performance metrics for an operation."""
        if self.isEnabledFor(VERBOSE):
            perf_data = {
                'operation': operation,
                'duration_ms': duration_ms,
                'memory_delta_mb': memory_delta_mb,
                'timestamp': time.time(),
                **metrics
            }

            # Store in global metrics
            _PERFORMANCE_METRICS[operation].append(perf_data)

            # Log the performance
            msg = f"⚡ {operation}: {duration_ms:.2f}ms"
            if memory_delta_mb != 0:
                msg += f", memory Δ: {memory_delta_mb:+.1f}MB"
            if metrics:
                msg += f", {metrics}"

            self.verbose(msg)

    def log_system_info(self, context: str = "") -> None:
        """Log comprehensive system information."""
        if self.isEnabledFor(logging.INFO):
            try:
                # System info
                system_info = {
                    'platform': platform.platform(),
                    'python_version': platform.python_version(),
                    'cpu_count': psutil.cpu_count(),
                    'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                    'memory_available_gb': psutil.virtual_memory().available / (1024**3),
                    'memory_percent': psutil.virtual_memory().percent
                }

                # PyTorch info
                if torch.cuda.is_available():
                    system_info['cuda_available'] = True
                    system_info['cuda_device_count'] = torch.cuda.device_count()
                    system_info['cuda_current_device'] = torch.cuda.current_device()
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    system_info['mps_available'] = True

                # MLX info
                if MLX_AVAILABLE:
                    system_info['mlx_available'] = True
                    try:
                        system_info['mlx_memory_gb'] = mx.get_active_memory() / (1024**3)
                    except Exception as e:
                        self.debug(f"Could not get MLX memory info for system info: {e}")

                self.info(f"System info {context}: {system_info}")

            except Exception as e:
                self.warning(f"Failed to log system info: {e}")

    def log_structured(self, event: str, data: Dict[str, Any], level: int = logging.INFO) -> None:
        """Log structured data for analysis."""
        if self.isEnabledFor(level):
            structured_entry = {
                'timestamp': datetime.datetime.now().isoformat(),
                'event': event,
                'data': data,
                'process_id': os.getpid(),
                'thread_id': threading.get_ident()
            }

            # Store in global structured logs
            _STRUCTURED_LOGS.append(structured_entry)

            # Get configuration for structured logging
            config = _get_config()
            structured_format = config.get("structured_format", "readable")
            show_raw = config.get("show_structured_raw", False)

            # Log with better formatting for console readability
            if structured_format == "readable":
                self._log_structured_formatted(structured_entry, level)

            # Also log raw JSON for file parsing (if enabled or for compact format)
            if show_raw or structured_format == "compact":
                self.log(level, f"STRUCTURED: {json.dumps(structured_entry)}")

    def _log_structured_formatted(self, entry: Dict[str, Any], level: int) -> None:
        """Log structured data with improved formatting."""
        event = entry['event']
        data = entry['data']
        timestamp = entry['timestamp']

        # Format timestamp for readability
        try:
            dt = datetime.datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            formatted_time = dt.strftime('%H:%M:%S')
        except:
            formatted_time = timestamp.split('T')[1][:8] if 'T' in timestamp else timestamp

        # Create a readable header
        header = f"📊 {event.upper().replace('_', ' ')} [{formatted_time}]"

        # Format the data based on event type
        if event == "experiment_start":
            self._format_experiment_start(header, data, level)
        elif event == "experiment_end":
            self._format_experiment_end(header, data, level)
        elif event == "metric":
            self._format_metric(header, data, level)
        elif event == "hyperparameters":
            self._format_hyperparameters(header, data, level)
        else:
            # Generic formatting
            self._format_generic_structured(header, data, level)

    def _format_experiment_start(self, header: str, data: Dict[str, Any], level: int) -> None:
        """Format experiment start event."""
        self.log(level, header)

        # Experiment info
        exp_name = data.get('experiment_name', 'Unknown')
        self.log(level, f"  🔬 Experiment: {exp_name}")

        # System info
        if 'system_info' in data:
            sys_info = data['system_info']
            self.log(level, f"  💻 Platform: {sys_info.get('platform', 'Unknown')}")
            self.log(level, f"  🐍 Python: {sys_info.get('python_version', 'Unknown')}")
            self.log(level, f"  🔥 PyTorch: {sys_info.get('torch_version', 'Unknown')}")

            # Hardware info
            if sys_info.get('mps_available'):
                self.log(level, f"  🍎 Apple Silicon: MPS Available")
            if sys_info.get('mlx_available'):
                self.log(level, f"  ⚡ MLX: Available")
            if sys_info.get('cuda_available'):
                self.log(level, f"  🚀 CUDA: Available")

            self.log(level, f"  🧠 CPU Cores: {sys_info.get('cpu_count', 'Unknown')}")
            self.log(level, f"  💾 Memory: {sys_info.get('memory_total_gb', 'Unknown')}GB")

    def _format_experiment_end(self, header: str, data: Dict[str, Any], level: int) -> None:
        """Format experiment end event."""
        self.log(level, header)

        exp_name = data.get('experiment_name', 'Unknown')
        duration = data.get('duration_seconds', 0)

        # Format duration
        if duration > 3600:
            duration_str = f"{duration/3600:.1f}h"
        elif duration > 60:
            duration_str = f"{duration/60:.1f}m"
        else:
            duration_str = f"{duration:.1f}s"

        self.log(level, f"  ✅ Completed: {exp_name}")
        self.log(level, f"  ⏱️  Duration: {duration_str}")

        # Metric summaries
        if 'metric_summaries' in data and data['metric_summaries']:
            self.log(level, f"  📈 Metrics: {len(data['metric_summaries'])} tracked")

    def _format_metric(self, header: str, data: Dict[str, Any], level: int) -> None:
        """Format metric event."""
        name = data.get('name', 'Unknown')
        value = data.get('value', 0)
        step = data.get('step')
        epoch = data.get('epoch')

        metric_str = f"📊 {name}: {value:.6f}"
        if step is not None:
            metric_str += f" (step {step})"
        if epoch is not None:
            metric_str += f" (epoch {epoch})"

        self.log(level, metric_str)

    def _format_hyperparameters(self, header: str, data: Dict[str, Any], level: int) -> None:
        """Format hyperparameters event."""
        self.log(level, header)

        # Group parameters by category
        categories = {
            'model': ['vocab_size', 'd_model', 'num_layers', 'max_seq_len'],
            'training': ['learning_rate', 'batch_size', 'epochs', 'weight_decay'],
            'execution': ['execution_mode', 'compression_enabled']
        }

        for category, keys in categories.items():
            category_params = {k: v for k, v in data.items() if k in keys}
            if category_params:
                self.log(level, f"  🔧 {category.title()}: {category_params}")

        # Log remaining parameters
        used_keys = set().union(*categories.values())
        remaining = {k: v for k, v in data.items() if k not in used_keys}
        if remaining:
            self.log(level, f"  ⚙️  Other: {remaining}")

    def _format_generic_structured(self, header: str, data: Dict[str, Any], level: int) -> None:
        """Format generic structured event."""
        self.log(level, header)

        # Format data in a readable way
        if isinstance(data, dict) and len(data) <= 5:
            for key, value in data.items():
                if isinstance(value, (int, float, str, bool)):
                    self.log(level, f"  • {key}: {value}")
                else:
                    self.log(level, f"  • {key}: {type(value).__name__}")
        else:
            # For complex data, just show summary
            self.log(level, f"  📋 Data: {len(data) if isinstance(data, dict) else type(data).__name__}")

    @contextmanager
    def timing_context(self, operation: str, log_level: int = VERBOSE):
        """Context manager for timing operations with automatic logging."""
        start_time = time.perf_counter()
        start_memory = self._get_memory_usage()

        _TIMING_STACK.append({
            'operation': operation,
            'start_time': start_time,
            'start_memory': start_memory
        })

        try:
            yield
        finally:
            end_time = time.perf_counter()
            end_memory = self._get_memory_usage()

            timing_info = _TIMING_STACK.pop()
            duration_ms = (end_time - start_time) * 1000
            memory_delta_mb = (end_memory - start_memory) if end_memory and start_memory else 0

            if self.isEnabledFor(log_level):
                self.log_performance(operation, duration_ms, memory_delta_mb)

    def _get_memory_usage(self) -> Optional[float]:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / (1024 * 1024)
        except Exception as e:
            self.debug(f"Could not get memory usage: {e}")
            return None


class StructuredJSONHandler(logging.Handler):
    """Handler for structured JSON logging."""

    def __init__(self, filename: str):
        super().__init__()
        self.filename = filename
        os.makedirs(os.path.dirname(filename), exist_ok=True)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            # Check if this is a structured log entry
            if hasattr(record, 'msg') and isinstance(record.msg, str) and record.msg.startswith('STRUCTURED:'):
                json_data = record.msg[11:]  # Remove 'STRUCTURED:' prefix

                with open(self.filename, 'a', encoding='utf-8') as f:
                    f.write(json_data + '\n')
        except Exception:
            self.handleError(record)


from ..performance.experiment_logger import ExperimentLogger


from ..performance.performance_tracker import PerformanceTracker


class IndentedRichHandler(RichHandler):
    """Custom Rich handler that supports indentation and progress coordination."""

    def emit(self, record: logging.LogRecord) -> None:
        # Apply indentation to the message
        indent = getattr(_log_state, "indentation", 0)
        if indent > 0:
            indent_str = " " * indent
            if isinstance(record.msg, str):
                record.msg = "\n".join(
                    indent_str + line for line in record.msg.splitlines()
                )

        # Add level name to the record for custom formatting
        if record.levelno == VERBOSE:
            record.levelname = "VERBOSE"

        # Call the parent's emit method
        super().emit(record)
        
        # Key: Refresh active progress bar after logging to prevent overwriting
        try:
            refresh_active_progress()
        except Exception:
            pass


class IndentedFormatter(logging.Formatter):
    """Formatter that applies indentation to log messages."""

    def format(self, record: logging.LogRecord) -> str:
        # Get the original formatted message
        formatted_msg = super().format(record)

        # Apply indentation
        indent = getattr(_log_state, "indentation", 0)
        if indent > 0:
            indent_str = " " * indent
            formatted_msg = "\n".join(
                indent_str + line for line in formatted_msg.splitlines()
            )

        return formatted_msg


def _showwarning(message, category, filename, lineno, file=None, line=None) -> None:
    """Custom warnings handler that logs warnings via our logger."""
    logger = logging.getLogger("py.warnings")
    logger.warning(f"{category.__name__}: {message} ({filename}:{lineno})")


def _get_default_config() -> Dict[str, Any]:
    """Get default logging configuration."""
    return {
        "level": "INFO",
        "console_level": "INFO",
        "file_level": "DEBUG",
        "format": "%(asctime)s [%(levelname)8s] %(name)s: %(message)s",
        "date_format": "%Y-%m-%d %H:%M:%S",
        "console_format": "%(asctime)s [%(levelname)8s] %(name)s: %(message)s",
        "file_format": "%(asctime)s [%(levelname)8s] %(name)s (%(filename)s:%(lineno)d): %(message)s",
        "use_colors": True,
        "log_dir": "output/logs",
        "log_file": "application.log",
        "max_file_size_mb": 10,
        "backup_count": 5,
        "capture_warnings": True,
        "propagate": False,
        "rich_theme": DEFAULT_THEME,
        "progress_refresh_per_sec": 20,
        "shape_log_level": "VERBOSE",
        "structured_format": "readable",  # "readable" or "compact"
        "show_structured_raw": False,  # Whether to also show raw JSON
    }


def _get_config() -> Dict[str, Any]:
    """Get logging configuration."""
    # In the future, this could be extended to load from a file
    return _get_default_config()


def _get_log_level(level: Union[str, int]) -> int:
    """Convert a log level name to its numeric value."""
    if isinstance(level, int):
        return level

    level_map = {
        "DEBUG": logging.DEBUG,
        "VERBOSE": VERBOSE,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    return level_map.get(level.upper(), logging.INFO)


class LoggingManager:
    """Manages the global state and configuration of the logging system."""

    def __init__(self):
        self.initialised = False
        self.root_console: Optional[Console] = None
        self.rich_banner_printed = False
        self.active_progress: Optional[Any] = None
        self.config: Dict[str, Any] = {}
        self.lock = threading.Lock() # Ensure thread-safe initialization

    def get_console(self) -> Optional[Console]:
        """Get the Rich console instance, initializing if necessary."""
        if not self.root_console:
            self.ensure_initialised()
        return self.root_console

    def ensure_initialised(self):
        """Initialize the logging system if it hasn't been already."""
        with self.lock:
            if not self.initialised:
                self._initialise()
                self.initialised = True

    def _initialise(self):
        """Perform the actual initialization of the root logger."""
        self.config = _get_default_config()

        # Register our custom logger class
        logging.setLoggerClass(ApplicationLogger)

        # Configure root logger
        level = _get_log_level(self.config.get("level", "INFO"))
        logging.basicConfig(handlers=[], level=level, force=True)  # Wipe defaults

        root_logger = logging.getLogger()

        self._setup_console_handler(root_logger)
        self._setup_file_handler(root_logger)

        # Capture warnings
        capture = self.config.get("capture_warnings", True)
        logging.captureWarnings(capture)
        if capture:
            warnings.showwarning = _showwarning

        root_logger.propagate = self.config.get("propagate", False)

    def _setup_console_handler(self, root_logger: logging.Logger):
        """Set up the console handler (Rich or standard)."""
        if RICH_AVAILABLE and self.config.get("use_colors", True):
            install_rich_traceback(show_locals=True)
            theme = Theme(self.config.get("rich_theme", DEFAULT_THEME))
            self.root_console = Console(theme=theme, highlight=True, force_terminal=True)

            if not self.rich_banner_printed and multiprocessing.current_process().name == "MainProcess":
                self.root_console.print("[bold green]Logger initialized with Rich console output[/]")
                self.rich_banner_printed = True

            console_level = _get_log_level(self.config.get("console_level", "INFO"))
            rich_hdl = IndentedRichHandler(
                console=self.root_console,
                show_time=True,
                show_path=False,
                markup=True,
                rich_tracebacks=True,
                log_time_format=self.config.get("date_format", "%Y-%m-%d %H:%M:%S"),
            )
            rich_hdl.setLevel(console_level)
            root_logger.addHandler(rich_hdl)
        else:
            if not self.rich_banner_printed and multiprocessing.current_process().name == "MainProcess":
                print("Logger initialized with standard console output")
                self.rich_banner_printed = True

            console_handler = logging.StreamHandler(sys.stdout)
            console_level = _get_log_level(self.config.get("console_level", "INFO"))
            console_handler.setLevel(console_level)

            formatter = IndentedFormatter(
                self.config.get("console_format", "%(asctime)s [%(levelname)8s] %(name)s: %(message)s"),
                self.config.get("date_format", "%Y-%m-%d %H:%M:%S"),
            )
            console_handler.setFormatter(formatter)
            root_logger.addHandler(console_handler)

    def _setup_file_handler(self, root_logger: logging.Logger):
        """Set up the rotating file handler."""
        try:
            log_dir = Path(self.config.get("log_dir", "output/logs"))
            log_dir.mkdir(parents=True, exist_ok=True)

            log_file = self.config.get("log_file", "application.log")
            log_path = log_dir / log_file

            file_level = _get_log_level(self.config.get("file_level", "DEBUG"))
            max_size = self.config.get("max_file_size_mb", 10) * 1024 * 1024
            backup_count = self.config.get("backup_count", 5)

            file_handler = RotatingFileHandler(
                log_path, maxBytes=max_size, backupCount=backup_count, encoding="utf-8"
            )
            file_formatter = logging.Formatter(
                self.config.get("file_format", "%(asctime)s [%(levelname)8s] %(name)s (%(filename)s:%(lineno)d): %(message)s"),
                self.config.get("date_format", "%Y-%m-%d %H:%M:%S"),
            )
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(file_level)
            root_logger.addHandler(file_handler)

        except Exception as e:
            root_logger.warning(f"Failed to set up file logging: {e}")

    def shutdown(self):
        """Gracefully shut down the logging system."""
        logging.shutdown()


# Instantiate the manager
_log_manager = LoggingManager()


def get_logger(name: Optional[str] = None) -> "ApplicationLogger":
    """Get a logger, initializing the system if needed."""
    if name is None:
        # Dynamic name resolution from caller frame
        frame = inspect.currentframe().f_back
        name = inspect.getmodule(frame).__name__ if frame else "application"

    _log_manager.ensure_initialised()
    return cast(ApplicationLogger, logging.getLogger(name))


@contextmanager
def indent_log(spaces: int = 2):
    """
    Context manager for indented logging.

    Args:
        spaces: Number of spaces to indent.
    """
    # Initialize indentation if not already set
    if not hasattr(_log_state, "indentation"):
        _log_state.indentation = 0

    # Increase indentation
    _log_state.indentation += spaces

    try:
        yield
    finally:
        # Decrease indentation
        _log_state.indentation -= spaces


def log_call(logger=None, level: str = "DEBUG"):
    """
    Decorator to log function calls with arguments, return values, and performance metrics.

    Args:
        logger: The logger to use. If None, gets a logger based on the module.
        level: The log level to use.
    """

    def decorator(func):
        level_num = _get_log_level(level)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            log = logger or get_logger(func.__module__)

            if not log.isEnabledFor(level_num):
                return func(*args, **kwargs)

            # Simplified signature for brevity
            arg_reprs = [repr(a) for a in args]
            kwarg_reprs = [f"{k}={v!r}" for k, v in kwargs.items()]
            signature = ", ".join(arg_reprs + kwarg_reprs)
            log.log(level_num, f"Calling {func.__name__}({signature[:100]}...)")

            # Use the performance context manager for timing and memory
            with log_performance(func.__name__, logger=log, log_level=level_num):
                try:
                    result = func(*args, **kwargs)
                    log.log(level_num, f"{func.__name__} returned: {repr(result)[:100]}...")
                    return result
                except Exception as e:
                    log.error(f"{func.__name__} raised: {repr(e)}", exc_info=True)
                    raise

        return wrapper

    return decorator


def set_log_level(name: str, level: Union[str, int]) -> None:
    """
    Set the log level for a specific logger.

    Args:
        name: The logger name.
        level: The log level (name or number).
    """
    logger = get_logger(name)
    level_num = _get_log_level(level)
    logger.setLevel(level_num)


def set_global_log_level(level: Union[str, int]) -> None:
    """
    Set the log level for all loggers.

    Args:
        level: The log level (name or number).
    """
    level_num = _get_log_level(level)
    logging.getLogger().setLevel(level_num)


def configure_logging(level: Union[str, int] = "INFO", **overrides) -> None:
    """Configure the root logger with optional overrides.

    This can be called at program start to ensure logging is initialized with
    a particular level and any custom configuration values.

    Args:
        level: The root log level to use.
        **overrides: Additional configuration keys to override the defaults.
    """
    log_manager = LoggingManager()
    config = log_manager.config
    config.update(overrides)
    config["level"] = level

    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    log_manager.initialised = False
    log_manager.ensure_initialised()

    root_logger.setLevel(_get_log_level(level))


def set_structured_logging_format(format_type: str = "readable", show_raw: bool = False) -> None:
    """
    Configure structured logging format.

    Args:
        format_type: "readable" for formatted output, "compact" for JSON only
        show_raw: Whether to also show raw JSON in readable mode
    """
    config = _get_config()
    config["structured_format"] = format_type
    config["show_structured_raw"] = show_raw


def set_console_colors(enabled: bool) -> None:
    """
    Enable or disable colored console output.

    Args:
        enabled: Whether to enable colors.
    """
    # Update configuration
    config = _get_config()
    config["use_colors"] = enabled

    # Get the root logger
    root_logger = logging.getLogger()

    # Store the current level
    current_level = root_logger.level
    current_level_num = _get_log_level(current_level)

    # Remove all handlers
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    # Reset the initialized flag and the root console to force recreation
    global _LOGGER_INITIALISED, _ROOT_CONSOLE
    _LOGGER_INITIALISED = False
    _ROOT_CONSOLE = None # Force recreation of the Console object

    # Propagate the current root logger's level to the console_level
    # for the new handler configuration.
    config["console_level"] = current_level_num

    # Reinitialize the logger
    _ensure_root(config)

    # Restore the level
    root_logger.setLevel(current_level)

    # Log a message about the color change
    logger = get_logger("logger") # Use a specific logger name to avoid self-logging issues if name is None
    if enabled:
        logger.info("Console colors enabled")
    else:
        logger.info("Console colors disabled")


def get_console() -> Optional[Console]:
    """Get the Rich console instance used for logging."""
    return _log_manager.get_console()



def set_active_progress(progress: Optional[Any]) -> None:
    """Register the currently active Rich progress instance."""
    _log_manager.active_progress = progress


def refresh_active_progress() -> None:
    """Refresh the active progress bar if one is registered."""
    if _log_manager.active_progress is not None:
        try:
            _log_manager.active_progress.refresh()
        except Exception:
            pass # Suppress errors if progress bar is already stopped


# Enhanced utility functions

def get_experiment_logger(experiment_name: str) -> ExperimentLogger:
    """Get or create an experiment logger."""
    if experiment_name not in _EXPERIMENT_LOGGERS:
        _EXPERIMENT_LOGGERS[experiment_name] = ExperimentLogger(experiment_name)
    return _EXPERIMENT_LOGGERS[experiment_name]


def get_performance_tracker() -> PerformanceTracker:
    """Get the global performance tracker."""
    if not hasattr(get_performance_tracker, '_instance'):
        get_performance_tracker._instance = PerformanceTracker()
    return get_performance_tracker._instance


@contextmanager
def log_performance(operation: str, logger=None, log_level: int = VERBOSE):
    """Context manager for automatic performance logging."""
    logger = logger or get_logger()
    tracker = get_performance_tracker()

    start_time = time.perf_counter()
    start_memory = None

    try:
        process = psutil.Process()
        start_memory = process.memory_info().rss / (1024 * 1024)  # MB
    except Exception as e:
        logger.debug(f"Could not get initial memory for performance tracking: {e}")
        start_memory = None

    try:
        yield
    finally:
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000

        memory_delta_mb = 0
        if start_memory:
            try:
                process = psutil.Process()
                end_memory = process.memory_info().rss / (1024 * 1024)  # MB
                memory_delta_mb = end_memory - start_memory
            except Exception as e:
                logger.debug(f"Could not get final memory for performance tracking: {e}")
                memory_delta_mb = 0

        # Record in tracker
        tracker.record_operation(operation, duration_ms, memory_delta_mb)

        # Log if enabled
        if logger.isEnabledFor(log_level):
            logger.log_performance(operation, duration_ms, memory_delta_mb)


def export_structured_logs(output_file: str, format: str = "json") -> None:
    """Export structured logs to file."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format.lower() == "json":
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(list(_STRUCTURED_LOGS), f, indent=2, default=str)
    elif format.lower() == "csv":
        import csv
        if _STRUCTURED_LOGS:
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                # Get all possible field names
                fieldnames = set()
                for log in _STRUCTURED_LOGS:
                    fieldnames.update(log.keys())
                    if 'data' in log and isinstance(log['data'], dict):
                        fieldnames.update(f"data.{k}" for k in log['data'].keys())

                writer = csv.DictWriter(f, fieldnames=sorted(fieldnames))
                writer.writeheader()

                for log in _STRUCTURED_LOGS:
                    row = log.copy()
                    if 'data' in row and isinstance(row['data'], dict):
                        for k, v in row['data'].items():
                            row[f"data.{k}"] = v
                        del row['data']
                    writer.writerow(row)
    else:
        raise ValueError(f"Unsupported format: {format}")


def get_performance_summary() -> Dict[str, Any]:
    """Get a summary of performance metrics."""
    tracker = get_performance_tracker()

    summary = {
        "total_operations": sum(len(ops) for ops in tracker.operations.values()),
        "operation_types": len(tracker.operations),
        "bottlenecks": tracker.detect_bottlenecks(),
        "operation_stats": {}
    }

    for operation in tracker.operations:
        summary["operation_stats"][operation] = tracker.get_operation_stats(operation)

    return summary


def setup_distributed_logging(rank: int, world_size: int, log_dir: str = "output/logs") -> None:
    """Setup logging for distributed training."""
    # Create rank-specific log directory
    rank_log_dir = Path(log_dir) / f"rank_{rank}"
    rank_log_dir.mkdir(parents=True, exist_ok=True)

    # Configure logging with rank-specific settings
    config = _get_default_config()
    config["log_file"] = f"application_rank_{rank}.log"
    config["log_dir"] = str(rank_log_dir)

    # Only rank 0 should log to console in distributed setting
    if rank != 0:
        config["console_level"] = "ERROR"  # Minimize console output from workers

    configure_logging(**config)

    logger = get_logger("distributed")
    logger.info(f"Distributed logging setup for rank {rank}/{world_size}")


def log_gpu_memory_summary() -> None:
    """Log comprehensive GPU memory summary."""
    logger = get_logger("memory")

    try:
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                device = f"cuda:{i}"
                allocated = torch.cuda.memory_allocated(device) / (1024**3)
                reserved = torch.cuda.memory_reserved(device) / (1024**3)
                max_allocated = torch.cuda.max_memory_allocated(device) / (1024**3)
                max_reserved = torch.cuda.max_memory_reserved(device) / (1024**3)

                logger.info(f"GPU {i} Memory - Allocated: {allocated:.2f}GB, "
                           f"Reserved: {reserved:.2f}GB, "
                           f"Max Allocated: {max_allocated:.2f}GB, "
                           f"Max Reserved: {max_reserved:.2f}GB")

        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("MPS Memory - Limited tracking available")

        if MLX_AVAILABLE:
            try:
                active = mx.get_active_memory() / (1024**3)
                peak = mx.get_peak_memory() / (1024**3)
                cache = mx.get_cache_memory() / (1024**3)

                logger.info(f"MLX Memory - Active: {active:.2f}GB, "
                           f"Peak: {peak:.2f}GB, Cache: {cache:.2f}GB")
            except Exception as e:
                logger.warning(f"Failed to get MLX memory info: {e}")

    except Exception as e:
        logger.error(f"Failed to log GPU memory summary: {e}")




from ..data.quality import SyntheticDataQualityTracker


def shutdown_logging() -> None:
    """Gracefully shut down the logging system."""
    _log_manager.shutdown()
def get_synthetic_data_tracker() -> SyntheticDataQualityTracker:
    """Get global synthetic data quality tracker."""
    if not hasattr(get_synthetic_data_tracker, '_instance'):
        get_synthetic_data_tracker._instance = SyntheticDataQualityTracker()
    return get_synthetic_data_tracker._instance


@contextmanager
def track_synthetic_data_generation(data_batch: Dict[str, Any], 
                                   difficulty_level: str = "medium"):
    """Context manager for tracking synthetic data generation quality."""
    tracker = get_synthetic_data_tracker()
    start_time = time.time()
    
    try:
        yield
    finally:
        # This would typically be called with actual quality scores
        # For now, we create placeholder scores
        quality_scores = {
            "coherence": 0.8,  # Would be calculated by quality assessment
            "correctness": 0.75,
            "diversity": 0.85,
            "complexity": 0.7
        }
        
        tracker.log_generation_quality(data_batch, quality_scores, difficulty_level)

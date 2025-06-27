"""
Visualization Utilities

This module provides comprehensive visualization tools for machine learning projects,
including training progress monitoring, model analysis, and performance visualization.

Features:
- Real-time training progress visualization
- Memory usage plots
- Model architecture visualization
- Performance metrics dashboards
- Loss and metric tracking
- Rich console progress bars
- Integration with wandb/tensorboard
"""

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from collections import defaultdict, deque

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.animation import FuncAnimation
import seaborn as sns
from datetime import datetime

# Rich imports for console visualization
try:
    from rich.console import Console
    from rich.progress import (
        Progress,
        SpinnerColumn,
        TextColumn,
        BarColumn,
        TaskProgressColumn,
        TimeRemainingColumn,
        TimeElapsedColumn,
    )
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.live import Live
    from rich.text import Text
    from rich.tree import Tree
    RICH_AVAILABLE = True
except ImportError:
    from typing import Any

    Console = Any  # type: ignore
    Progress = Any  # type: ignore
    SpinnerColumn = Any  # type: ignore
    TextColumn = Any  # type: ignore
    BarColumn = Any  # type: ignore
    TaskProgressColumn = Any  # type: ignore
    TimeRemainingColumn = Any  # type: ignore
    TimeElapsedColumn = Any  # type: ignore
    Table = Any  # type: ignore
    Panel = Any  # type: ignore
    Layout = Any  # type: ignore
    Live = Any  # type: ignore
    Text = Any  # type: ignore
    Tree = Any # type: ignore
    RICH_AVAILABLE = False

# Optional integrations
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter
    import torch
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    class torch:
        nn = type("nn", (), {"Module": object})

try:
    import mlx.core as mx
    import mlx.nn as mlx_nn
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False
    class mlx_nn:
        Module = object

from .logger import get_logger, get_console

logger = get_logger(__name__)

# Set matplotlib style
plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
sns.set_palette("husl")


class TrainingVisualizer:
    """Comprehensive training visualization and monitoring."""
    
    def __init__(self,
                 experiment_name: str = "default_experiment",
                 log_dir: str = "logs/training",
                 use_wandb: bool = False,
                 use_tensorboard: bool = True):
        """
        Initialize training visualizer.
        
        Args:
            experiment_name: Name of the experiment
            log_dir: Directory for saving logs and plots
            use_wandb: Whether to use Weights & Biases
            use_tensorboard: Whether to use TensorBoard
        """
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics storage
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self.timestamps: List[float] = []
        self.step_count = 0
        
        # Memory monitoring (lazy import to avoid circular dependencies)
        self.memory_monitor = None
        self.memory_history: deque = deque(maxlen=1000)
        
        # Rich console setup
        self.console = Console() if RICH_AVAILABLE else None
        self.progress: Optional[Progress] = None
        self.current_tasks: Dict[str, int] = {}
        
        # External logging setup
        self.tensorboard_writer: Optional[SummaryWriter] = None
        self.wandb_run = None
        
        if use_tensorboard and TENSORBOARD_AVAILABLE:
            self.setup_tensorboard()
            
        if use_wandb and WANDB_AVAILABLE:
            self.setup_wandb()
            
    def _get_memory_monitor(self):
        """Lazy import of memory monitor to avoid circular dependencies."""
        if self.memory_monitor is None:
            try:
                from .profiling import get_memory_monitor, MemorySnapshot
                self.memory_monitor = get_memory_monitor()
                # Store MemorySnapshot class for type hints
                self._MemorySnapshot = MemorySnapshot
            except ImportError:
                logger.warning("Could not import memory monitoring - profiling disabled")
                self.memory_monitor = None
        return self.memory_monitor
            
    def setup_tensorboard(self) -> None:
        """Setup TensorBoard logging."""
        try:
            tb_log_dir = self.log_dir / "tensorboard"
            tb_log_dir.mkdir(exist_ok=True)
            self.tensorboard_writer = SummaryWriter(tb_log_dir)
            logger.info(f"TensorBoard logging enabled: {tb_log_dir}")
        except Exception as e:
            logger.warning(f"Failed to setup TensorBoard: {e}")
            
    def setup_wandb(self) -> None:
        """Setup Weights & Biases logging."""
        try:
            self.wandb_run = wandb.init(
                project="unified-transformer",
                name=self.experiment_name,
                dir=str(self.log_dir)
            )
            logger.info("Weights & Biases logging enabled")
        except Exception as e:
            logger.warning(f"Failed to setup wandb: {e}")
            
    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        """Log a metric value."""
        if step is None:
            step = self.step_count
            self.step_count += 1
            
        self.metrics[name].append(value)
        if len(self.timestamps) <= step:
            self.timestamps.append(time.time())
            
        # Log to external services
        if self.tensorboard_writer:
            self.tensorboard_writer.add_scalar(name, value, step)
            
        if self.wandb_run:
            self.wandb_run.log({name: value}, step=step)
            
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log multiple metrics at once."""
        for name, value in metrics.items():
            self.log_metric(name, value, step)
            
    def create_training_progress(self, total_steps: int, description: str = "Training") -> int:
        """Create a training progress bar."""
        if not RICH_AVAILABLE:
            return 0
            
        if self.progress is None:
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=self.console
            )
            self.progress.start()
            
        task_id = self.progress.add_task(description, total=total_steps)
        self.current_tasks[description] = task_id
        return task_id
        
    def update_progress(self, task_id: int, advance: int = 1, **kwargs) -> None:
        """Update progress bar."""
        if self.progress:
            self.progress.update(task_id, advance=advance, **kwargs)
            
    def finish_progress(self) -> None:
        """Finish and cleanup progress bars."""
        if self.progress:
            self.progress.stop()
            self.progress = None
            self.current_tasks.clear()
            
    def plot_training_curves(self, 
                           metrics: Optional[List[str]] = None,
                           save_path: Optional[str] = None,
                           show: bool = False,
                           window_size: int = 20) -> None:
        """Plot training curves with optional smoothing."""
        if not self.metrics:
            logger.warning("No metrics to plot")
            return

        metrics_to_plot = metrics or list(self.metrics.keys())
        available_metrics = [m for m in metrics_to_plot if m in self.metrics and self.metrics[m]]

        if not available_metrics:
            logger.warning("No available metrics to plot")
            return

        n_metrics = len(available_metrics)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 5 * n_metrics), squeeze=False)
        axes = axes.flatten() # Ensure axes is always a 1D array

        for i, metric_name in enumerate(available_metrics):
            values = self.metrics[metric_name]
            steps = list(range(len(values)))

            # Raw data
            axes[i].plot(steps, values, label=f'Raw {metric_name}', alpha=0.4, color='gray')

            # Smoothed data
            if len(values) >= window_size:
                smoothed_values = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
                axes[i].plot(steps[window_size-1:], smoothed_values, label=f'Smoothed {metric_name}', linewidth=2)
            else:
                axes[i].plot(steps, values, label=f'Raw {metric_name}', linewidth=2) # Plot raw if too short

            axes[i].set_title(f"{metric_name.replace('_', ' ').title()}")
            axes[i].set_ylabel("Value")
            axes[i].set_xlabel("Step")
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training curves saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()
            
    def plot_memory_usage(self, 
                         save_path: Optional[str] = None,
                         show: bool = False) -> None:
        """Plot memory usage over time."""
        if not self.memory_history:
            logger.warning("No memory history to plot")
            return

        timestamps = [s.timestamp for s in self.memory_history]
        process_memory = [s.process_memory_gb for s in self.memory_history]
        torch_memory = [s.torch_memory_gb for s in self.memory_history]
        mlx_memory = [s.mlx_memory_gb for s in self.memory_history]
        times = [datetime.fromtimestamp(t) for t in timestamps]

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(times, process_memory, label='Process Memory', linewidth=2)
        ax.plot(times, torch_memory, label='PyTorch Memory', linewidth=2)
        if any(m > 0 for m in mlx_memory):
            ax.plot(times, mlx_memory, label='MLX Memory', linewidth=2)

        memory_monitor = self._get_memory_monitor()
        if memory_monitor:
            memory_limit = getattr(memory_monitor, 'memory_limit_gb', 24.0)
            warning_limit = getattr(memory_monitor, 'warning_limit_gb', 20.0)
            ax.axhline(y=memory_limit, color='red', linestyle='--', label=f'Memory Limit ({memory_limit}GB)')
            ax.axhline(y=warning_limit, color='orange', linestyle='--', alpha=0.7, label='Warning Threshold')

        ax.set_title('Memory Usage Over Time')
        ax.set_ylabel('Memory (GB)')
        ax.set_xlabel('Time')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Memory usage plot saved to {save_path}")

        if show:
            plt.show()
        else:
            plt.close()
            
    def create_dashboard(self) -> Optional[Layout]:
        """Create a Rich dashboard for real-time monitoring."""
        if not RICH_AVAILABLE:
            return None

        layout = Layout()
        layout.split(
            Layout(name="header", size=3),
            Layout(ratio=1, name="main"),
            Layout(size=5, name="footer"),
        )
        layout["main"].split_row(Layout(name="metrics"), Layout(name="memory"))
        return layout
        
    def update_dashboard(self, layout: Layout) -> None:
        """Update dashboard with current metrics."""
        if not RICH_AVAILABLE or not layout:
            return
            
        # Header
        header_text = Text(f"Training Dashboard - {self.experiment_name}",
                          style="bold blue")
        layout["header"].update(Panel(header_text, title="Status"))
        
        # Metrics table
        metrics_table = Table(title="Latest Metrics")
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Value", style="green")
        metrics_table.add_column("Trend", style="yellow")
        
        for name, values in self.metrics.items():
            if values:
                latest = values[-1]
                trend = "UP" if len(values) > 1 and values[-1] > values[-2] else "DOWN"
                metrics_table.add_row(name, f"{latest:.4f}", trend)
                
        layout["metrics"].update(metrics_table)
        
        # Memory info
        if self.memory_history:
            latest_memory = self.memory_history[-1]
            memory_table = Table(title="Memory Usage")
            memory_table.add_column("Type", style="cyan")
            memory_table.add_column("Usage (GB)", style="green")
            memory_table.add_column("Status", style="yellow")
            
            process_mem = latest_memory.process_memory_gb
            memory_monitor = self._get_memory_monitor()
            warning_limit = memory_monitor.warning_limit_gb if memory_monitor else 20.0
            status = "[HIGH]" if process_mem > warning_limit else "[OK]"
            memory_table.add_row("Process", f"{process_mem:.2f}", status)
            memory_table.add_row("PyTorch", f"{latest_memory.torch_memory_gb:.2f}", "")
            memory_table.add_row("MLX", f"{latest_memory.mlx_memory_gb:.2f}", "")
            
            layout["memory"].update(memory_table)
            
        # Footer
        footer_text = f"Step: {self.step_count} | Time: {datetime.now().strftime('%H:%M:%S')}"
        layout["footer"].update(Panel(footer_text, title="Info"))
        
    def save_summary_report(self, filepath: Optional[str] = None) -> None:
        """Save a comprehensive summary report."""
        if filepath is None:
            filepath = self.log_dir / f"{self.experiment_name}_summary.txt"
            
        with open(filepath, 'w') as f:
            f.write(f"Training Summary: {self.experiment_name}\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Total steps: {self.step_count}\n")
            f.write(f"Training duration: {len(self.timestamps)} time points\n\n")
            
            f.write("Final Metrics:\n")
            f.write("-" * 20 + "\n")
            for name, values in self.metrics.items():
                if values:
                    f.write(f"{name}: {values[-1]:.6f}\n")
                    
            f.write("\nMetric Statistics:\n")
            f.write("-" * 20 + "\n")
            for name, values in self.metrics.items():
                if values:
                    f.write(f"\n{name}:\n")
                    f.write(f"  Min: {min(values):.6f}\n")
                    f.write(f"  Max: {max(values):.6f}\n")
                    f.write(f"  Mean: {np.mean(values):.6f}\n")
                    f.write(f"  Std: {np.std(values):.6f}\n")
                    
            # Memory summary
            if self.memory_history:
                peak_memory = max(self.memory_history, key=lambda s: s.process_memory_gb)
                f.write(f"\nPeak Memory Usage: {peak_memory.process_memory_gb:.2f}GB\n")
                memory_monitor = self._get_memory_monitor()
                if memory_monitor:
                    f.write(f"Memory limit: {memory_monitor.memory_limit_gb}GB\n")
                
        logger.info(f"Summary report saved to {filepath}")
        
    def cleanup(self) -> None:
        """Cleanup resources."""
        self.finish_progress()
        
        if self.tensorboard_writer:
            self.tensorboard_writer.close()
            
        if self.wandb_run:
            self.wandb_run.finish()


# Global visualizer instance
_global_visualizer: Optional[TrainingVisualizer] = None


def get_visualizer(experiment_name: str = "default_experiment") -> "TrainingVisualizer":
    """Get global visualizer instance."""
    global _global_visualizer
    if _global_visualizer is None:
        _global_visualizer = TrainingVisualizer(experiment_name)
    return _global_visualizer


def plot_model_architecture(model: torch.nn.Module, save_path: Optional[str] = None) -> None:
    """
    Prints a tree visualization of the model architecture to the console
    and optionally saves it to a file.
    """
    logger.info("Visualizing model architecture...")
    console = get_console()
    if not console or not RICH_AVAILABLE:
        logger.warning("Rich console not available for visualization. Printing model string.")
        logger.info(str(model))
        if save_path:
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(str(model))
            logger.info(f"Model architecture saved to {save_path}")
        return

    is_mlx_model = MLX_AVAILABLE and isinstance(model, mlx_nn.Module)
    is_torch_model = TENSORBOARD_AVAILABLE and isinstance(model, torch.nn.Module)

    def count_params(module):
        if is_mlx_model:
            try:
                # MLX parameters are in a dict
                return sum(p.size for p in module.parameters().values())
            except Exception:
                return 0
        elif is_torch_model:
            try:
                return sum(p.numel() for p in module.parameters() if p.requires_grad)
            except Exception:
                return 0
        return 0
    
    def get_children(module):
        if is_torch_model and hasattr(module, 'named_children'):
            return module.named_children()
        elif is_mlx_model and hasattr(module, 'named_modules'):
            # This is a trick to get direct children from MLX's named_modules
            children = []
            for name, sub_module in module.named_modules():
                 if '.' not in name and name:
                     children.append((name, sub_module))
            return children
        return []

    tree = Tree(
        f"[bold magenta]{type(model).__name__}[/bold magenta] ([dim]{count_params(model):,} trainable params[/dim])",
        guide_style="bold bright_blue"
    )

    def add_nodes(tree_node: Tree, module):
        children = get_children(module)
        for name, child in children:
            extra_info = ""
            if hasattr(child, 'in_features') and hasattr(child, 'out_features'):
                extra_info += f" (in: {getattr(child, 'in_features', '?')}, out: {getattr(child, 'out_features', '?')})"
            if hasattr(child, 'd_model'):
                extra_info += f" (d_model: {getattr(child, 'd_model', '?')})"
            if hasattr(child, 'n_heads'):
                 extra_info += f" (heads: {getattr(child, 'n_heads', '?')})"
            elif hasattr(child, 'num_attention_heads'):
                 extra_info += f" (heads: {getattr(child, 'num_attention_heads', '?')})"
            
            num_params = count_params(child)
            param_str = f" ([dim]{num_params:,} params[/dim])" if num_params > 0 else ""

            child_text = Text.from_markup(
                f"[green]{name}[/green]: [cyan]{type(child).__name__}[/cyan]{extra_info}{param_str}"
            )
            
            child_node = tree_node.add(child_text)
            add_nodes(child_node, child)

    add_nodes(tree, model)
    
    console.print(tree)
    
    if save_path:
        from rich.console import Console as FileConsole
        try:
            with open(save_path, "w", encoding="utf-8") as f:
                file_console = FileConsole(file=f, force_terminal=True, color_system="truecolor", record=True, width=120)
                file_console.print(tree)
            logger.info(f"Model architecture visualization saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save architecture visualization to {save_path}: {e}")


def create_comparison_plot(experiments: Dict[str, Dict[str, List[float]]], 
                          metric_name: str,
                          save_path: Optional[str] = None,
                          show: bool = False,
                          smoothing_window: int = 10) -> None:
    """Create comparison plot across multiple experiments with smoothing."""
    plt.figure(figsize=(12, 6))

    for exp_name, metrics in experiments.items():
        if metric_name not in metrics or not metrics[metric_name]:
            logger.warning(f"Metric '{metric_name}' not found for experiment '{exp_name}'. Skipping.")
            continue

        values = metrics[metric_name]
        if len(values) >= smoothing_window:
            smoothed_values = np.convolve(values, np.ones(smoothing_window)/smoothing_window, mode='valid')
            plt.plot(np.arange(smoothing_window-1, len(values)), smoothed_values, label=f'{exp_name} (smoothed)', linewidth=2)
        else:
            plt.plot(values, label=exp_name, linewidth=2)

    plt.title(f"{metric_name.replace('_', ' ').title()} Comparison")
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Comparison plot saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

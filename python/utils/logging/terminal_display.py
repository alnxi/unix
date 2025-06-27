"""
Enhanced Terminal Display Utilities.

This module provides clean, anti-flickering progress bars and enhanced terminal output.
Features include:

- Anti-flickering progress bars with controlled refresh rates
- Coordinated logging and progress display
- Context managers for clean progress tracking
- Enhanced formatting and styling
- Memory-efficient progress updates
"""

import time
from contextlib import contextmanager
from typing import Any, Dict, Optional, Callable, List, Union

try:
    from rich.console import Console
    from rich.progress import (
        Progress, 
        BarColumn, 
        TextColumn, 
        TimeRemainingColumn, 
        TimeElapsedColumn,
        SpinnerColumn,
        MofNCompleteColumn,
        ProgressColumn
    )
    from rich.live import Live
    from rich.theme import Theme
    from rich.table import Table
    from rich.panel import Panel
    from rich.columns import Columns
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    Progress = None
    Live = None

from .logger import get_logger, set_active_progress, get_console, DEFAULT_THEME

logger = get_logger(__name__)

# Global state for progress coordination
_ACTIVE_LIVE_CONTEXT = None
_PROGRESS_INSTANCES = []


class EnhancedProgressColumn(ProgressColumn):
    """Custom progress column with enhanced formatting."""
    
    def __init__(self, field_name: str, format_func: Optional[Callable] = None):
        super().__init__()
        self.field_name = field_name
        self.format_func = format_func or (lambda x: f"{x}")
        
    def render(self, task) -> Text:
        value = task.fields.get(self.field_name, "N/A")
        formatted = self.format_func(value)
        return Text(formatted, style="bold cyan")


def create_training_progress() -> Progress:
    """Create optimized progress bar for training with anti-flicker settings."""
    if not RICH_AVAILABLE:
        return None
        
    progress_columns = [
        SpinnerColumn(spinner_name="dots", style="cyan"),
        TextColumn("[bold blue]{task.description}", justify="right"),
        BarColumn(bar_width=None, complete_style="green", finished_style="green"),
        TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
        TextColumn("•", style="dim"),
        MofNCompleteColumn(),
        TextColumn("•", style="dim"),
        TimeRemainingColumn(),
        TextColumn("•", style="dim"),
        TimeElapsedColumn(),
        TextColumn("•", style="dim"),
        EnhancedProgressColumn("loss", lambda x: f"Loss: {x:.4f}" if isinstance(x, (int, float)) else f"Loss: {x}"),
        TextColumn("•", style="dim"),
        EnhancedProgressColumn("lr", lambda x: f"LR: {x:.2e}" if isinstance(x, (int, float)) else f"LR: {x}"),
        TextColumn("•", style="dim"),
        EnhancedProgressColumn("memory", lambda x: f"Mem: {x:.1f}GB" if isinstance(x, (int, float)) else f"Mem: {x}"),
        TextColumn("•", style="dim"),
        EnhancedProgressColumn("step_time", lambda x: f"Step: {x:.2f}s" if isinstance(x, (int, float)) else f"Step: {x}"),
    ]
    
    console = get_console()
    if console is None:
        console = Console(theme=Theme(DEFAULT_THEME))
    
    return Progress(
        *progress_columns,
        console=console,
        refresh_per_second=2,  # Configured refresh rate to prevent flickering
        transient=False,  # Keep progress visible after completion
        expand=True
    )


def create_data_loading_progress() -> Progress:
    """Create progress bar optimized for data loading operations."""
    if not RICH_AVAILABLE:
        return None
        
    progress_columns = [
        SpinnerColumn(spinner_name="dots", style="green"),
        TextColumn("[bold green]{task.description}"),
        BarColumn(bar_width=40, complete_style="green"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        EnhancedProgressColumn("speed", lambda x: f"{x} items/s" if x != "N/A" else ""),
    ]
    
    console = get_console()
    if console is None:
        console = Console(theme=Theme(DEFAULT_THEME))
    
    return Progress(
        *progress_columns,
        console=console,
        refresh_per_second=20,  # Higher refresh for data loading
        transient=True,  # Disappear after completion
        expand=True
    )


@contextmanager
def progress_context(
    description: str, 
    total: int, 
    progress_type: str = "training",
    refresh_per_second: int = 10,
    show_in_live: bool = True,
    **progress_kwargs
):
    """
    Enhanced context manager for progress tracking with anti-flickering.
    
    Args:
        description: Task description
        total: Total number of items
        progress_type: Type of progress bar ("training", "data_loading", "custom")
        refresh_per_second: Refresh rate (lower = less flickering)
        show_in_live: Whether to show in Live context
        **progress_kwargs: Additional progress configuration
    """
    if not RICH_AVAILABLE:
        # Fallback for when Rich is not available
        class MockProgress:
            def advance(self, advance=1):
                pass
            def update(self, **kwargs):
                pass
        yield MockProgress()
        return
    
    # Create appropriate progress bar
    if progress_type == "training":
        progress = create_training_progress()
    elif progress_type == "data_loading":
        progress = create_data_loading_progress()
    else:
        # Custom progress bar
        console = get_console() or Console(theme=Theme(DEFAULT_THEME))
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
            refresh_per_second=refresh_per_second,
            **progress_kwargs
        )
    
    # Add task
    task_id = progress.add_task(description, total=total)
    
    # Register active progress for logger coordination
    set_active_progress(progress)
    _PROGRESS_INSTANCES.append(progress)
    
    class ProgressManager:
        def __init__(self, progress_instance, task_id):
            self.progress = progress_instance
            self.task_id = task_id
            
        def advance(self, advance: int = 1):
            """Advance progress with coordinated refresh."""
            self.progress.advance(self.task_id, advance)
            
        def update(self, **kwargs):
            """Update progress fields with coordinated refresh."""
            self.progress.update(self.task_id, **kwargs)
            
        def set_description(self, description: str):
            """Update task description."""
            self.progress.update(self.task_id, description=description)
            
        def set_fields(self, **fields):
            """Update custom fields."""
            self.progress.update(self.task_id, **fields)
    
    manager = ProgressManager(progress, task_id)
    
    try:
        if show_in_live:
            # Use Live context for coordinated display
            global _ACTIVE_LIVE_CONTEXT
            if _ACTIVE_LIVE_CONTEXT is None:
                with Live(
                    progress,
                    console=progress.console,
                    refresh_per_second=refresh_per_second,
                    vertical_overflow="visible",
                ) as live:
                    _ACTIVE_LIVE_CONTEXT = live
                    yield manager
                    _ACTIVE_LIVE_CONTEXT = None
            else:
                # Already in a Live context, just yield the manager
                yield manager
        else:
            # Manual progress display
            with progress:
                yield manager
    finally:
        # Cleanup
        if progress in _PROGRESS_INSTANCES:
            _PROGRESS_INSTANCES.remove(progress)
        if not _PROGRESS_INSTANCES:
            set_active_progress(None)
        elif _PROGRESS_INSTANCES:
            # Set the most recent progress as active
            set_active_progress(_PROGRESS_INSTANCES[-1])


@contextmanager
def multi_progress_context(progress_configs: List[Dict[str, Any]], refresh_per_second: int = 10):
    """
    Context manager for multiple coordinated progress bars.
    
    Args:
        progress_configs: List of progress configurations with keys:
            - description: str
            - total: int
            - progress_type: str (optional)
        refresh_per_second: Refresh rate for all progress bars
    """
    if not RICH_AVAILABLE:
        yield [None] * len(progress_configs)
        return
    
    # Create combined progress display
    console = get_console() or Console(theme=Theme(DEFAULT_THEME))
    
    # Create a new progress instance with the desired console and refresh rate
    progress_columns = [
        SpinnerColumn(spinner_name="dots", style="cyan"),
        TextColumn("[bold blue]{task.description}", justify="right"),
        BarColumn(bar_width=None, complete_style="green", finished_style="green"),
        TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    ]
    
    progress = Progress(
        *progress_columns,
        console=console,
        refresh_per_second=refresh_per_second,
        transient=False,
        expand=True
    )
    
    # Add all tasks
    tasks = []
    for config in progress_configs:
        task_id = progress.add_task(
            config["description"], 
            total=config["total"]
        )
        tasks.append(task_id)
    
    # Register active progress
    set_active_progress(progress)
    _PROGRESS_INSTANCES.append(progress)
    
    class MultiProgressManager:
        def __init__(self, progress_instance, task_ids):
            self.progress = progress_instance
            self.task_ids = task_ids
            
        def __getitem__(self, index):
            """Get progress manager for specific task."""
            class TaskProgressManager:
                def __init__(self, progress, task_id):
                    self.progress = progress
                    self.task_id = task_id
                    
                def advance(self, advance: int = 1):
                    self.progress.advance(self.task_id, advance)
                    
                def update(self, **kwargs):
                    self.progress.update(self.task_id, **kwargs)
                    
                def set_description(self, description: str):
                    self.progress.update(self.task_id, description=description)
                    
                def set_fields(self, **fields):
                    self.progress.update(self.task_id, **fields)
                    
            return TaskProgressManager(self.progress, self.task_ids[index])
    
    manager = MultiProgressManager(progress, tasks)
    
    try:
        with Live(
            progress,
            console=console,
            refresh_per_second=refresh_per_second,
            vertical_overflow="visible",
        ) as live:
            global _ACTIVE_LIVE_CONTEXT
            _ACTIVE_LIVE_CONTEXT = live
            yield manager
            _ACTIVE_LIVE_CONTEXT = None
    finally:
        # Cleanup
        if progress in _PROGRESS_INSTANCES:
            _PROGRESS_INSTANCES.remove(progress)
        if not _PROGRESS_INSTANCES:
            set_active_progress(None)
        elif _PROGRESS_INSTANCES:
            set_active_progress(_PROGRESS_INSTANCES[-1])


def create_status_panel(title: str, content: Dict[str, Any], style: str = "blue") -> Panel:
    """Create a status panel for displaying training information."""
    if not RICH_AVAILABLE:
        return None
        
    table = Table.grid(padding=1)
    table.add_column(style="cyan", no_wrap=True)
    table.add_column()
    
    for key, value in content.items():
        if isinstance(value, float):
            if abs(value) < 0.001:
                formatted_value = f"{value:.2e}"
            else:
                formatted_value = f"{value:.4f}"
        elif isinstance(value, int):
            formatted_value = f"{value:,}"
        else:
            formatted_value = str(value)
            
        table.add_row(f"{key}:", formatted_value)
    
    return Panel(table, title=title, style=style, expand=False)


def display_training_summary(
    epoch: int,
    metrics: Dict[str, float],
    model_info: Optional[Dict[str, Any]] = None,
    system_info: Optional[Dict[str, Any]] = None
):
    """Display comprehensive training summary with enhanced formatting."""
    if not RICH_AVAILABLE:
        logger.info(f"Epoch {epoch} - Metrics: {metrics}")
        return
        
    console = get_console()
    if console is None:
        console = Console()
    
    panels = []
    
    # Training metrics panel
    metrics_panel = create_status_panel("Training Metrics", metrics, "green")
    if metrics_panel:
        panels.append(metrics_panel)
    
    # Model info panel
    if model_info:
        model_panel = create_status_panel("Model Info", model_info, "blue")
        if model_panel:
            panels.append(model_panel)
    
    # System info panel
    if system_info:
        system_panel = create_status_panel("System Info", system_info, "yellow")
        if system_panel:
            panels.append(system_panel)
    
    if panels:
        # Display panels in columns if we have multiple
        if len(panels) > 1:
            console.print(Columns(panels, equal=True, expand=True))
        else:
            console.print(panels[0])
    
    console.print()  # Add spacing


@contextmanager
def enhanced_logging_context(operation_name: str, show_progress: bool = True):
    """
    Enhanced context manager that combines logging with progress tracking.
    
    Args:
        operation_name: Name of the operation for logging
        show_progress: Whether to show progress indicators
    """
    logger.info(f"🚀 Starting {operation_name}")
    start_time = time.time()
    
    try:
        if show_progress and RICH_AVAILABLE:
            # Create a simple spinner for indefinite operations
            console = get_console()
            if console:
                with console.status(f"[bold green]Running {operation_name}...") as status:
                    yield status
            else:
                yield None
        else:
            yield None
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"❌ {operation_name} failed after {duration:.2f}s: {e}")
        raise
    else:
        duration = time.time() - start_time
        logger.success(f"✅ {operation_name} completed in {duration:.2f}s")


def setup_enhanced_terminal():
    """Setup enhanced terminal display capabilities."""
    if RICH_AVAILABLE:
        console = get_console()
        if console:
            console.print("[bold green]Enhanced terminal display initialized[/bold green]")
            logger.info("Rich terminal display features available")
        else:
            logger.warning("Rich console not available")
    else:
        logger.warning("Rich library not available - using fallback terminal display")


# Utility functions for specific use cases

def create_epoch_progress(num_epochs: int, current_epoch: int = 0):
    """Create progress bar specifically for epoch tracking."""
    return progress_context(
        description=f"Training Progress",
        total=num_epochs,
        progress_type="training",
        refresh_per_second=5  # Lower refresh for epoch-level progress
    )


def create_batch_progress(num_batches: int, epoch: int):
    """Create progress bar specifically for batch processing within an epoch."""
    return progress_context(
        description=f"Epoch {epoch} Batches",
        total=num_batches,
        progress_type="training",
        refresh_per_second=10
    )


def create_validation_progress(num_batches: int):
    """Create progress bar for validation phase."""
    return progress_context(
        description="Validation",
        total=num_batches,
        progress_type="data_loading",
        refresh_per_second=15
    )
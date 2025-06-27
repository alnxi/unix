from collections import defaultdict
from typing import Any, Dict, Optional
import time
import platform
from pathlib import Path
import psutil
import torch

from ..logging.logger import get_logger, StructuredJSONHandler, MLX_AVAILABLE

class ExperimentLogger:
    """Enhanced experiment tracking and logging."""

    def __init__(self, experiment_name: str, log_dir: str = "output/logs/experiments"):
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.logger = get_logger(f"experiment.{experiment_name}")
        self.metrics = defaultdict(list)
        self.start_time = time.time()

        # Setup structured logging for this experiment
        json_handler = StructuredJSONHandler(
            str(self.log_dir / f"{experiment_name}_structured.jsonl")
        )
        self.logger.addHandler(json_handler)

        # Log experiment start
        self.log_event("experiment_start", {
            "experiment_name": experiment_name,
            "start_time": time.time(),
            "system_info": self._get_system_info()
        })

    def log_metric(self, name: str, value: float, step: Optional[int] = None,
                   epoch: Optional[int] = None) -> None:
        """Log a metric value."""
        metric_data = {
            "name": name,
            "value": value,
            "timestamp": time.time(),
            "step": step,
            "epoch": epoch
        }

        self.metrics[name].append(metric_data)
        self.log_event("metric", metric_data)

        # Also log to console
        msg = f"📊 {name}: {value:.6f}"
        if step is not None:
            msg += f" (step {step})"
        if epoch is not None:
            msg += f" (epoch {epoch})"

        self.logger.info(msg)

    def log_hyperparameters(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters."""
        self.log_event("hyperparameters", params)
        # The structured logging will handle the formatting

    def log_model_checkpoint(self, checkpoint_path: str, metrics: Dict[str, float]) -> None:
        """Log model checkpoint information."""
        checkpoint_data = {
            "checkpoint_path": checkpoint_path,
            "metrics": metrics,
            "timestamp": time.time()
        }

        self.log_event("checkpoint", checkpoint_data)
        self.logger.info(f"💾 Checkpoint saved: {checkpoint_path} | Metrics: {metrics}")

    def log_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Log a structured event."""
        self.logger.log_structured(event_type, data)

    def get_metric_summary(self, metric_name: str) -> Dict[str, float]:
        """Get summary statistics for a metric."""
        if metric_name not in self.metrics:
            return {}

        values = [m["value"] for m in self.metrics[metric_name]]
        if not values:
            return {}

        import statistics
        return {
            "count": len(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0,
            "min": min(values),
            "max": max(values),
            "latest": values[-1]
        }

    def finalize(self) -> None:
        """Finalize the experiment and log summary."""
        duration = time.time() - self.start_time

        summary = {
            "experiment_name": self.experiment_name,
            "duration_seconds": duration,
            "end_time": time.time(),
            "metric_summaries": {name: self.get_metric_summary(name) for name in self.metrics}
        }

        self.log_event("experiment_end", summary)
        self.logger.success(f"Experiment '{self.experiment_name}' completed in {duration:.2f}s")

    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for experiment logging."""
        try:
            return {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "mps_available": hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(),
                "mlx_available": MLX_AVAILABLE,
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": psutil.virtual_memory().total / (1024**3)
            }
        except Exception:
            return {}

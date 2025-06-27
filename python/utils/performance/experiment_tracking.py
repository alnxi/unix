"""
Experiment Tracking Utilities for UnifiedTransformer

Provides utilities for consistent experiment naming and tracking.
"""

import json
import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

class ExperimentNamer:
    """Utility for generating consistent experiment names."""
    
    def __init__(self, config_path: str = "configs/tracking/naming_conventions.json"):
        self.config_path = Path(config_path)
        self.conventions = self._load_conventions()
    
    def _load_conventions(self) -> Dict[str, Any]:
        """Load naming conventions from config."""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return json.load(f)
        else:
            return {}
    
    def generate_name(
        self,
        architecture: str,
        size: str,
        run_id: Optional[int] = None,
        project: str = "ut"
    ) -> str:
        """Generate experiment name following conventions."""
        date_str = datetime.datetime.now().strftime("%Y%m%d")
        
        if run_id is None:
            run_id = self._get_next_run_id(architecture, size, date_str)
        
        run_id_str = f"{run_id:03d}"
        
        return f"{project}_{architecture}_{size}_{date_str}_{run_id_str}"
    
    def _get_next_run_id(self, architecture: str, size: str, date_str: str) -> int:
        """Get next available run ID for the day."""
        # This would typically check existing experiments
        # For now, return 1
        return 1
    
    def parse_name(self, name: str) -> Dict[str, str]:
        """Parse experiment name into components."""
        parts = name.split('_')
        if len(parts) >= 5:
            return {
                "project": parts[0],
                "architecture": parts[1],
                "size": parts[2],
                "date": parts[3],
                "run_id": parts[4]
            }
        return {}

class ExperimentTracker:
    """Unified experiment tracking interface."""
    
    def __init__(self, use_mlflow: bool = True, use_wandb: bool = True, use_tensorboard: bool = True):
        self.use_mlflow = use_mlflow
        self.use_wandb = use_wandb
        self.use_tensorboard = use_tensorboard
        
        self.mlflow_run = None
        self.wandb_run = None
        self.tb_writer = None
        
        self._setup_trackers()
    
    def _setup_trackers(self):
        """Initialize tracking backends."""
        if self.use_mlflow:
            try:
                import mlflow
                # Load MLflow config
                config_path = Path("configs/tracking/mlflow_config.json")
                if config_path.exists():
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    mlflow.set_tracking_uri(config["tracking_uri"])
                    mlflow.set_experiment(config["experiment_name"])
            except Exception as e:
                print(f"Warning: MLflow setup failed: {e}")
                self.use_mlflow = False
        
        if self.use_wandb:
            try:
                import wandb
                # Load W&B config
                config_path = Path("configs/tracking/wandb_config.json")
                if config_path.exists():
                    with open(config_path, 'r') as f:
                        self.wandb_config = json.load(f)
                else:
                    self.wandb_config = {"project": "unified-transformer"}
            except Exception as e:
                print(f"Warning: W&B setup failed: {e}")
                self.use_wandb = False
        
        if self.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                # Load TensorBoard config
                config_path = Path("configs/tracking/tensorboard_config.json")
                if config_path.exists():
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    self.tb_log_dir = config["log_dir"]
                else:
                    self.tb_log_dir = "experiments/tensorboard"
            except Exception as e:
                print(f"Warning: TensorBoard setup failed: {e}")
                self.use_tensorboard = False
    
    def start_run(self, experiment_name: str, config: Dict[str, Any], tags: List[str] = None):
        """Start tracking run across all backends."""
        if self.use_mlflow:
            try:
                import mlflow
                self.mlflow_run = mlflow.start_run(run_name=experiment_name)
                mlflow.log_params(config)
                if tags:
                    mlflow.set_tags({f"tag_{i}": tag for i, tag in enumerate(tags)})
            except Exception as e:
                print(f"Warning: MLflow run start failed: {e}")
        
        if self.use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    name=experiment_name,
                    config=config,
                    tags=tags,
                    **self.wandb_config
                )
            except Exception as e:
                print(f"Warning: W&B run start failed: {e}")
        
        if self.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                run_dir = Path(self.tb_log_dir) / experiment_name
                self.tb_writer = SummaryWriter(run_dir)
            except Exception as e:
                print(f"Warning: TensorBoard run start failed: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics across all backends."""
        if self.use_mlflow and self.mlflow_run:
            try:
                import mlflow
                for key, value in metrics.items():
                    mlflow.log_metric(key, value, step)
            except Exception as e:
                print(f"Warning: MLflow metric logging failed: {e}")
        
        if self.use_wandb and self.wandb_run:
            try:
                self.wandb_run.log(metrics, step=step)
            except Exception as e:
                print(f"Warning: W&B metric logging failed: {e}")
        
        if self.use_tensorboard and self.tb_writer:
            try:
                for key, value in metrics.items():
                    self.tb_writer.add_scalar(key, value, step)
            except Exception as e:
                print(f"Warning: TensorBoard metric logging failed: {e}")
    
    def end_run(self):
        """End tracking run across all backends."""
        if self.use_mlflow and self.mlflow_run:
            try:
                import mlflow
                mlflow.end_run()
            except Exception as e:
                print(f"Warning: MLflow run end failed: {e}")
        
        if self.use_wandb and self.wandb_run:
            try:
                self.wandb_run.finish()
            except Exception as e:
                print(f"Warning: W&B run end failed: {e}")
        
        if self.use_tensorboard and self.tb_writer:
            try:
                self.tb_writer.close()
            except Exception as e:
                print(f"Warning: TensorBoard run end failed: {e}")

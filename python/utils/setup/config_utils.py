"""
Configuration Utilities

This module provides utilities for configuration management, including
YAML/Hydra integration, environment variable handling, and configuration validation.

Features:
- YAML configuration loading and validation
- Hydra integration for experiment management
- Environment variable substitution
- Configuration merging and inheritance
- Type checking and validation
- Default configuration management
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Type
from dataclasses import dataclass, field, fields
from copy import deepcopy

import yaml
from omegaconf import OmegaConf, DictConfig

# Optional Hydra integration
try:
    import hydra
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    HYDRA_AVAILABLE = True
except ImportError:
    HYDRA_AVAILABLE = False

from ..logging.logger import get_logger

logger = get_logger(__name__)


@dataclass
class BaseConfig:
    """Base configuration class with common utilities."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        result = {}
        for field_info in fields(self):
            value = getattr(self, field_info.name)
            if hasattr(value, 'to_dict'):
                result[field_info.name] = value.to_dict()
            elif isinstance(value, (list, tuple)):
                result[field_info.name] = [
                    item.to_dict() if hasattr(item, 'to_dict') else item 
                    for item in value
                ]
            else:
                result[field_info.name] = value
        return result
        
    def update_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Update configuration from dictionary."""
        for key, value in config_dict.items():
            if hasattr(self, key):
                current_value = getattr(self, key)
                if hasattr(current_value, 'update_from_dict') and isinstance(value, dict):
                    current_value.update_from_dict(value)
                else:
                    setattr(self, key, value)
            else:
                logger.warning(f"Unknown configuration key: {key}")
                
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        # Override in subclasses for specific validation
        return errors


class ConfigLoader:
    """Configuration loader with YAML and environment variable support."""
    
    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        """
        Initialize configuration loader.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir) if config_dir else Path("configs")
        self.env_var_pattern = re.compile(r'\$\{([^}]+)\}')
        
    def load_yaml(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """Load YAML configuration file with environment variable substitution."""
        config_path = Path(config_path)
        
        if not config_path.is_absolute():
            config_path = self.config_dir / config_path
            
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            content = f.read()
            
        # Substitute environment variables
        content = self._substitute_env_vars(content)
        
        try:
            config = yaml.safe_load(content)
            logger.verbose(f"Loaded configuration from {config_path}")
            return config
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {config_path}: {e}")
            
    def _substitute_env_vars(self, content: str) -> str:
        """Substitute environment variables in configuration content."""
        def replace_env_var(match):
            var_expr = match.group(1)
            
            # Handle default values: ${VAR:default}
            if ':' in var_expr:
                var_name, default_value = var_expr.split(':', 1)
                return os.getenv(var_name.strip(), default_value.strip())
            else:
                var_value = os.getenv(var_expr.strip())
                if var_value is None:
                    logger.warning(f"Environment variable {var_expr} not found")
                    return match.group(0)  # Return original if not found
                return var_value
                
        return self.env_var_pattern.sub(replace_env_var, content)
        
    def merge_configs(self, *configs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge multiple configuration dictionaries."""
        if not configs:
            return {}
            
        result = deepcopy(configs[0])
        
        for config in configs[1:]:
            result = self._deep_merge(result, config)
            
        return result
        
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = deepcopy(value)
                
        return result
        
    def load_with_inheritance(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """Load configuration with inheritance support."""
        config = self.load_yaml(config_path)
        
        # Handle inheritance
        if '_base_' in config:
            base_configs = config.pop('_base_')
            if isinstance(base_configs, str):
                base_configs = [base_configs]
                
            # Load base configurations
            base_config = {}
            for base_path in base_configs:
                base = self.load_with_inheritance(base_path)
                base_config = self.merge_configs(base_config, base)
                
            # Merge with current config
            config = self.merge_configs(base_config, config)
            
        return config


class HydraConfigManager:
    """Hydra configuration manager for experiment management."""
    
    def __init__(self, config_dir: Union[str, Path] = "configs"):
        """Initialize Hydra configuration manager."""
        if not HYDRA_AVAILABLE:
            raise ImportError("Hydra is not available. Install with: pip install hydra-core")
            
        self.config_dir = Path(config_dir).absolute()
        self.initialized = False
        
    def initialize(self, version_base: Optional[str] = None) -> None:
        """Initialize Hydra configuration system."""
        if self.initialized:
            return
            
        try:
            # Clear any existing Hydra instance
            if GlobalHydra().is_initialized():
                GlobalHydra.instance().clear()
                
            # Initialize with config directory
            initialize_config_dir(
                config_dir=str(self.config_dir),
                version_base=version_base
            )
            self.initialized = True
            logger.info(f"Hydra initialized with config directory: {self.config_dir}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Hydra: {e}")
            raise
            
    def compose_config(self, 
                      config_name: str,
                      overrides: Optional[List[str]] = None,
                      return_hydra_config: bool = False) -> Union[DictConfig, Dict[str, Any]]:
        """Compose configuration using Hydra."""
        if not self.initialized:
            self.initialize()
            
        try:
            cfg = compose(config_name=config_name, overrides=overrides or [])
            
            if return_hydra_config:
                return cfg
            else:
                return OmegaConf.to_container(cfg, resolve=True)
                
        except Exception as e:
            logger.error(f"Failed to compose config '{config_name}': {e}")
            raise
            
    def cleanup(self) -> None:
        """Cleanup Hydra configuration."""
        if self.initialized and GlobalHydra().is_initialized():
            GlobalHydra.instance().clear()
            self.initialized = False


class ConfigValidator:
    """Configuration validation utilities."""
    
    @staticmethod
    def validate_type(value: Any, expected_type: Type, field_name: str) -> List[str]:
        """Validate value type."""
        errors = []
        
        if not isinstance(value, expected_type):
            errors.append(f"{field_name}: expected {expected_type.__name__}, got {type(value).__name__}")
            
        return errors
        
    @staticmethod
    def validate_range(value: Union[int, float], 
                      min_val: Optional[Union[int, float]] = None,
                      max_val: Optional[Union[int, float]] = None,
                      field_name: str = "value") -> List[str]:
        """Validate numeric range."""
        errors = []
        
        if min_val is not None and value < min_val:
            errors.append(f"{field_name}: {value} is less than minimum {min_val}")
            
        if max_val is not None and value > max_val:
            errors.append(f"{field_name}: {value} is greater than maximum {max_val}")
            
        return errors
        
    @staticmethod
    def validate_choices(value: Any, choices: List[Any], field_name: str) -> List[str]:
        """Validate value is in allowed choices."""
        errors = []
        
        if value not in choices:
            errors.append(f"{field_name}: {value} not in allowed choices {choices}")
            
        return errors
        
    @staticmethod
    def validate_path(path: Union[str, Path], 
                     must_exist: bool = False,
                     must_be_file: bool = False,
                     must_be_dir: bool = False,
                     field_name: str = "path") -> List[str]:
        """Validate file/directory path."""
        errors = []
        path = Path(path)
        
        if must_exist and not path.exists():
            errors.append(f"{field_name}: path does not exist: {path}")
            return errors
            
        if must_be_file and path.exists() and not path.is_file():
            errors.append(f"{field_name}: path is not a file: {path}")
            
        if must_be_dir and path.exists() and not path.is_dir():
            errors.append(f"{field_name}: path is not a directory: {path}")
            
        return errors


class DefaultConfigs:
    """Default configuration templates."""
    
    @staticmethod
    def get_training_config() -> Dict[str, Any]:
        """Get default training configuration."""
        return {
            "model": {
                "architecture": "hybrid",
                "hidden_size": 768,
                "num_layers": 12,
                "num_attention_heads": 12,
                "intermediate_size": 3072,
                "max_position_embeddings": 2048,
                "vocab_size": 50000
            },
            "training": {
                "batch_size": 16,
                "learning_rate": 5e-4,
                "num_epochs": 10,
                "warmup_steps": 1000,
                "weight_decay": 0.01,
                "gradient_accumulation_steps": 4,
                "max_grad_norm": 1.0,
                "save_steps": 1000,
                "eval_steps": 500,
                "logging_steps": 100
            },
            "data": {
                "dataset_name": "default_dataset",
                "max_length": 1024,
                "preprocessing_num_workers": 4
            },
            "optimization": {
                "optimizer": "adamw",
                "scheduler": "cosine",
                "fp16": True,
                "gradient_checkpointing": True
            },
            "logging": {
                "log_level": "INFO",
                "use_wandb": False,
                "use_tensorboard": True,
                "experiment_name": "default_experiment"
            }
        }
        
    @staticmethod
    def get_model_config() -> Dict[str, Any]:
        """Get default model configuration."""
        return {
            "architecture": "hybrid",
            "hidden_size": 768,
            "num_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "max_position_embeddings": 2048,
            "vocab_size": 50000,
            "hidden_dropout_prob": 0.1,
            "attention_probs_dropout_prob": 0.1,
            "layer_norm_eps": 1e-12,
            "initializer_range": 0.02,
            "use_cache": True,
            "pad_token_id": 0,
            "bos_token_id": 1,
            "eos_token_id": 2
        }


# Global configuration loader
_global_config_loader: Optional[ConfigLoader] = None
_global_hydra_manager: Optional[HydraConfigManager] = None


def get_config_loader(config_dir: Optional[Union[str, Path]] = None) -> ConfigLoader:
    """Get global configuration loader."""
    global _global_config_loader
    if _global_config_loader is None:
        _global_config_loader = ConfigLoader(config_dir)
    return _global_config_loader


def get_hydra_manager(config_dir: Union[str, Path] = "configs") -> Optional[HydraConfigManager]:
    """Get global Hydra configuration manager."""
    global _global_hydra_manager
    if HYDRA_AVAILABLE and _global_hydra_manager is None:
        _global_hydra_manager = HydraConfigManager(config_dir)
    return _global_hydra_manager


def load_config(config_path: Union[str, Path], 
               config_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Convenient function to load configuration."""
    loader = get_config_loader(config_dir)
    return loader.load_with_inheritance(config_path)


def save_config(config: Dict[str, Any], save_path: Union[str, Path]) -> None:
    """Save configuration to YAML file."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
        
    logger.info(f"Configuration saved to {save_path}")


def validate_config(config: Dict[str, Any], config_class: Type[BaseConfig]) -> List[str]:
    """Validate configuration against a configuration class."""
    try:
        # Create instance and validate
        config_instance = config_class(**config)
        return config_instance.validate()
    except Exception as e:
        return [f"Configuration validation error: {e}"]


def apply_cli_overrides(config: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """
    Apply CLI overrides to configuration using dot notation.

    Args:
        config: Base configuration dictionary
        overrides: List of override strings in format "key.subkey=value"

    Returns:
        Updated configuration dictionary

    Example:
        overrides = ["trainer.batch_size=32", "model.num_layers=12"]
    """
    config = deepcopy(config)

    for override in overrides:
        if '=' not in override:
            logger.warning(f"Invalid override format (missing '='): {override}")
            continue

        key_path, value_str = override.split('=', 1)
        keys = key_path.split('.')

        # Navigate to the parent dictionary
        current = config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            elif not isinstance(current[key], dict):
                logger.warning(f"Cannot override {key_path}: {key} is not a dictionary")
                break
            current = current[key]
        else:
            # Set the final value with type conversion
            final_key = keys[-1]
            try:
                # Try to parse as Python literal (handles int, float, bool, None, lists, etc.)
                import ast
                try:
                    parsed_value = ast.literal_eval(value_str)
                except (ValueError, SyntaxError):
                    # If literal_eval fails, treat as string
                    parsed_value = value_str

                current[final_key] = parsed_value
                logger.verbose(f"Applied override: {key_path} = {parsed_value}")

            except Exception as e:
                logger.warning(f"Failed to apply override {override}: {e}")

    return config


def auto_backend_selection() -> str:
    """
    Automatically select the best backend based on available hardware.

    Returns:
        Backend name: "mlx", "torch", or "cpu"
    """
    import platform, os

    # Honour explicit user preference first – this is important for power-users.
    # Set UT_USE_MLX=1 (or true/yes) to force MLX even for training. Any other
    # value falls through to the automatic heuristics below.
    _env_force_mlx = os.getenv("UT_USE_MLX", "0").lower() in {"1", "true", "yes"}
    
    # Set UT_USE_TORCH=1 (or true/yes) to force PyTorch backend instead of MLX
    _env_force_torch = os.getenv("UT_USE_TORCH", "0").lower() in {"1", "true", "yes"}

    # Heuristic:  MLX training support in the current codebase is still
    # experimental and missing gradient flow for several mixed-backend paths.
    # To avoid silent training failures (e.g. loss stuck ~30), we default to the
    # mature PyTorch backend for TRAINING jobs.  Users can still force MLX via
    # the environment variable above once the MLX path is validated.

    # Check for explicit environment variable overrides
    if _env_force_torch:
        try:
            import torch  # noqa: F401 – just import-check
            logger.info("⚙️  UT_USE_TORCH=1 – forcing PyTorch backend")
            return "torch"
        except ImportError:
            logger.warning("UT_USE_TORCH requested but PyTorch not available – falling back to auto-selection")
    
    # If the user explicitly forces MLX we try to honour that request.
    if _env_force_mlx:
        try:
            import mlx.core as mx  # noqa: F401 – just import-check
            
            # Validate hardware requirements for MLX
            from .hardware_detection import validate_mlx_requirements
            validate_mlx_requirements(strict=False)  # Warn but don't abort for forced MLX
            
            logger.info("⚙️  UT_USE_MLX=1 – forcing MLX backend")
            return "mlx"
        except ImportError:
            logger.warning("UT_USE_MLX requested but MLX not available – falling back to PyTorch")

    # Otherwise – prefer a stable PyTorch backend if available (CUDA/MPS/CPU).
    try:
        import torch  # noqa: F401
        if torch.cuda.is_available():
            logger.info("CUDA available – selecting PyTorch backend")
            return "torch"
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("MPS available – selecting PyTorch backend")
            return "torch"
    except ImportError:
        logger.warning("PyTorch not available – checking for MLX as fallback")

    # If PyTorch is not available (rare in training env) or user forced MLX & it failed,
    # attempt MLX next (for inference-only environments this can still be useful).
    try:
        import mlx.core as mx  # noqa: F401
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            # Validate hardware requirements for MLX (strict mode for auto-selection)
            from .hardware_detection import validate_mlx_requirements
            if validate_mlx_requirements(strict=True):
                logger.info("PyTorch unavailable; MLX detected – selecting MLX backend")
                return "mlx"
            else:
                logger.warning("MLX available but hardware requirements not met – falling back to CPU")
    except ImportError:
        pass

    # Fallback to CPU (very slow, but at least functional).
    logger.info("No accelerated backend available – defaulting to CPU backend")
    return "cpu"


def save_effective_config(config: Dict[str, Any], output_dir: Union[str, Path]) -> None:
    """
    Save the effective configuration to output directory for reproducibility.

    Args:
        config: Configuration dictionary to save
        output_dir: Output directory path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "effective_config.yaml"
    save_config(config, config_path)
    logger.info(f"Effective configuration saved to {config_path}")


def create_training_config_from_dict(config_dict: Dict[str, Any]) -> 'TrainingExecutionConfig':
    """
    Create TrainingExecutionConfig from configuration dictionary.

    Args:
        config_dict: Configuration dictionary

    Returns:
        TrainingExecutionConfig instance
    """
    # Import here to avoid circular imports
    from ..training.training_executor import TrainingExecutionConfig
    from ..training.trainers.mlx_trainer import MLXTrainerConfig

    # Extract sections
    execution_config = config_dict.get('execution', {})
    model_config = config_dict.get('model', {})
    trainer_config_dict = config_dict.get('trainer', {})
    data_config = config_dict.get('data', {})
    compression_config = config_dict.get('compression', {})
    monitoring_config = config_dict.get('monitoring', {})
    hardware_config = config_dict.get('hardware', {})
    output_config = config_dict.get('output', {})

    # Create trainer config
    trainer_config = MLXTrainerConfig(
        batch_size=trainer_config_dict.get('batch_size', 16),
        gradient_accumulation_steps=trainer_config_dict.get('gradient_accumulation_steps') or trainer_config_dict.get('gradient_accumulation', 4),
        learning_rate=trainer_config_dict.get('learning_rate', 5e-4),
        weight_decay=trainer_config_dict.get('weight_decay', 0.01),
        max_grad_norm=trainer_config_dict.get('max_grad_norm', 1.0),
        warmup_steps=trainer_config_dict.get('warmup_steps', 1000),
        max_steps=trainer_config_dict.get('max_steps'),
        mixed_precision=trainer_config_dict.get('mixed_precision', True),
        optimizer_type=trainer_config_dict.get('optimizer_type', 'adamw'),
        scheduler_type=trainer_config_dict.get('scheduler_type', 'cosine'),
        logging_steps=monitoring_config.get('logging_steps', 50),
        save_steps=monitoring_config.get('save_steps', 1000),
        eval_steps=monitoring_config.get('eval_steps', 500),
        memory_limit_gb=hardware_config.get('memory_limit_gb', 22),
    )

    # Create execution config
    return TrainingExecutionConfig(
        execution_mode=execution_config.get('mode', 'progressive'),
        validation_epochs=execution_config.get('validation_epochs', 2),
        full_training_epochs=execution_config.get('full_training_epochs', 10),

        model_config=model_config,
        trainer_config=trainer_config,
        data_pipeline_config=data_config,
        hardware_config=hardware_config,

        use_compression=compression_config.get('use_compression', False),
        compression_config=compression_config,

        target_convergence_improvement=0.35,
        target_performance_improvement=0.175,
        memory_limit_gb=hardware_config.get('memory_limit_gb', 22),

        output_dir=output_config.get('log_dir', 'logs/app'),
        checkpoint_dir=output_config.get('checkpoint_dir', 'models/checkpoints/app'),
        log_dir=output_config.get('log_dir', 'logs/app'),
    )

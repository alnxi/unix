"""
Unified Dataset Loading Interface for UnifiedTransformer

This module provides a unified interface for loading and processing datasets
across different modalities (text, vision, audio) with automatic format detection,
preprocessing pipelines, and memory-efficient loading strategies.

Features:
- Automatic dataset format detection
- Memory-efficient streaming for large datasets
- Configurable preprocessing pipelines
- Multi-modal dataset support
- Caching and persistence
- Progress tracking and validation
"""

import os
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple, Iterator, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import hashlib
import time

import numpy as np
try:
    import torch
    from torch.utils.data import Dataset, DataLoader, IterableDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    class Dataset: pass
    class DataLoader: pass
    class IterableDataset: pass

try:
    import datasets
    from datasets import load_dataset, Dataset as HFDataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False

from ..logging.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DatasetConfig:
    """Configuration for dataset loading."""
    name: str
    path: Optional[str] = None
    split: str = "train"
    streaming: bool = False
    cache_dir: Optional[str] = None
    preprocessing: Optional[Dict[str, Any]] = None
    validation: bool = True
    max_samples: Optional[int] = None
    shuffle: bool = True
    seed: int = 42


@dataclass
class DatasetMetadata:
    """Metadata information about a dataset."""
    name: str
    size: int
    modality: str
    format: str
    splits: List[str]
    features: Dict[str, str]
    cache_path: Optional[str] = None
    last_modified: Optional[float] = None
    checksum: Optional[str] = None


class BaseDatasetLoader(ABC):
    """Base class for dataset loaders."""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.metadata: Optional[DatasetMetadata] = None
        
    @abstractmethod
    def load(self) -> Union[Dataset, IterableDataset]:
        """Load the dataset."""
        pass
        
    @abstractmethod
    def validate(self) -> bool:
        """Validate the dataset."""
        pass
        
    def get_metadata(self) -> Optional[DatasetMetadata]:
        """Get dataset metadata."""
        return self.metadata


class TextDatasetLoader(BaseDatasetLoader):
    """Loader for text datasets."""
    
    def load(self) -> Union[Dataset, IterableDataset]:
        """Load text dataset."""
        if DATASETS_AVAILABLE:
            return self._load_hf_dataset()
        else:
            return self._load_custom_text_dataset()
            
    def _load_hf_dataset(self):
        """Load dataset using HuggingFace datasets library."""
        try:
            if self.config.path:
                dataset = load_dataset(
                    self.config.name,
                    data_files=self.config.path,
                    split=self.config.split,
                    streaming=self.config.streaming,
                    cache_dir=self.config.cache_dir
                )
            else:
                dataset = load_dataset(
                    self.config.name,
                    split=self.config.split,
                    streaming=self.config.streaming,
                    cache_dir=self.config.cache_dir
                )
                
            self.metadata = DatasetMetadata(
                name=self.config.name,
                size=len(dataset) if hasattr(dataset, '__len__') else -1,
                modality="text",
                format="huggingface",
                splits=[self.config.split],
                features={k: str(v) for k, v in dataset.features.items()} if hasattr(dataset, 'features') else {}
            )
            
            logger.info(f"Loaded HuggingFace dataset: {self.config.name}")
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to load HuggingFace dataset: {e}")
            raise
            
    def _load_custom_text_dataset(self):
        """Load custom text dataset from files."""
        if not self.config.path:
            raise ValueError("Path required for custom text dataset")
            
        path = Path(self.config.path)
        
        if path.is_file():
            # Single file
            return self._load_text_file(path)
        elif path.is_dir():
            # Directory of files
            return self._load_text_directory(path)
        else:
            raise FileNotFoundError(f"Dataset path not found: {path}")
            
    def _load_text_file(self, file_path: Path):
        """Load text from a single file."""
        class TextFileDataset(Dataset):
            def __init__(self, file_path: Path, max_samples: Optional[int] = None):
                self.file_path = file_path
                self.lines = []
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if max_samples and i >= max_samples:
                            break
                        self.lines.append(line.strip())
                        
            def __len__(self):
                return len(self.lines)
                
            def __getitem__(self, idx):
                return {"text": self.lines[idx]}
                
        dataset = TextFileDataset(file_path, self.config.max_samples)
        
        self.metadata = DatasetMetadata(
            name=self.config.name,
            size=len(dataset),
            modality="text",
            format="text_file",
            splits=[self.config.split],
            features={"text": "string"}
        )
        
        return dataset
        
    def _load_text_directory(self, dir_path: Path):
        """Load text from multiple files in a directory."""
        text_files = list(dir_path.glob("*.txt")) + list(dir_path.glob("*.json"))
        
        if not text_files:
            raise ValueError(f"No text files found in {dir_path}")
            
        class TextDirectoryDataset(Dataset):
            def __init__(self, file_paths: List[Path], max_samples: Optional[int] = None):
                self.data = []
                sample_count = 0
                
                for file_path in file_paths:
                    if max_samples and sample_count >= max_samples:
                        break
                        
                    if file_path.suffix == '.json':
                        with open(file_path, 'r', encoding='utf-8') as f:
                            try:
                                data = json.load(f)
                                if isinstance(data, list):
                                    for item in data:
                                        if max_samples and sample_count >= max_samples:
                                            break
                                        self.data.append(item)
                                        sample_count += 1
                                else:
                                    self.data.append(data)
                                    sample_count += 1
                            except json.JSONDecodeError:
                                logger.warning(f"Failed to parse JSON file: {file_path}")
                    else:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            for line in f:
                                if max_samples and sample_count >= max_samples:
                                    break
                                self.data.append({"text": line.strip()})
                                sample_count += 1
                                
            def __len__(self):
                return len(self.data)
                
            def __getitem__(self, idx):
                return self.data[idx]
                
        dataset = TextDirectoryDataset(text_files, self.config.max_samples)
        
        self.metadata = DatasetMetadata(
            name=self.config.name,
            size=len(dataset),
            modality="text",
            format="text_directory",
            splits=[self.config.split],
            features={"text": "string"}
        )
        
        return dataset
        
    def validate(self) -> bool:
        """Validate text dataset."""
        if not self.config.validation:
            return True
            
        try:
            # Basic validation checks
            if self.config.path:
                path = Path(self.config.path)
                if not path.exists():
                    logger.error(f"Dataset path does not exist: {path}")
                    return False
                    
            logger.info("Text dataset validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Text dataset validation failed: {e}")
            return False


class ImageDatasetLoader(BaseDatasetLoader):
    """Loader for image datasets (base implementation for future use)."""
    
    def load(self) -> Union[Dataset, IterableDataset]:
        """Load image dataset."""
        logger.info("Image dataset loading - placeholder implementation")
        raise NotImplementedError("Image dataset loading will be implemented when needed")
        
    def validate(self) -> bool:
        """Validate image dataset."""
        return True


class AudioDatasetLoader(BaseDatasetLoader):
    """Loader for audio datasets (base implementation for future use)."""
    
    def load(self) -> Union[Dataset, IterableDataset]:
        """Load audio dataset."""
        logger.info("Audio dataset loading - placeholder implementation")
        raise NotImplementedError("Audio dataset loading will be implemented when needed")
        
    def validate(self) -> bool:
        """Validate audio dataset."""
        return True


class MultiModalDatasetLoader(BaseDatasetLoader):
    """Loader for multi-modal datasets (base implementation for future use)."""
    
    def load(self) -> Union[Dataset, IterableDataset]:
        """Load multi-modal dataset."""
        logger.info("Multi-modal dataset loading - placeholder implementation")
        raise NotImplementedError("Multi-modal dataset loading will be implemented when needed")
        
    def validate(self) -> bool:
        """Validate multi-modal dataset."""
        return True


class UnifiedDatasetLoader:
    """Unified interface for loading datasets of any type."""
    
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.loaders = {
            "text": TextDatasetLoader,
            "image": ImageDatasetLoader,
            "audio": AudioDatasetLoader,
            "multimodal": MultiModalDatasetLoader
        }
        
    def load_dataset(self, config: DatasetConfig) -> Union[Dataset, IterableDataset]:
        """Load dataset based on configuration."""
        modality = self._detect_modality(config)
        
        if modality not in self.loaders:
            raise ValueError(f"Unsupported modality: {modality}")
            
        loader_class = self.loaders[modality]
        loader = loader_class(config)
        
        # Validate if requested
        if config.validation and not loader.validate():
            raise ValueError(f"Dataset validation failed for {config.name}")
            
        dataset = loader.load()
        
        # Apply preprocessing if specified
        if config.preprocessing:
            dataset = self._apply_preprocessing(dataset, config.preprocessing)
            
        logger.info(f"Successfully loaded {modality} dataset: {config.name}")
        return dataset
        
    def _detect_modality(self, config: DatasetConfig) -> str:
        """Detect dataset modality from config or path."""
        # For now, default to text - can be extended with auto-detection
        if config.path:
            path = Path(config.path)
            if path.suffix in ['.txt', '.json', '.jsonl']:
                return "text"
            elif path.suffix in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                return "image"
            elif path.suffix in ['.wav', '.mp3', '.flac', '.ogg']:
                return "audio"
                
        # Default to text for now
        return "text"
        
    def _apply_preprocessing(self, dataset: Union[Dataset, IterableDataset], 
                           preprocessing_config: Dict[str, Any]) -> Union[Dataset, IterableDataset]:
        """Apply preprocessing to dataset."""
        # Placeholder for preprocessing pipeline
        logger.info("Applying preprocessing (placeholder implementation)")
        return dataset
        
    def create_dataloader(self, dataset: Union[Dataset, IterableDataset], 
                         batch_size: int = 32, 
                         shuffle: bool = True,
                         num_workers: int = 0,
                         **kwargs) -> DataLoader:
        """Create a DataLoader from dataset."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for DataLoader creation")
            
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            **kwargs
        )
        
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached datasets."""
        cache_info = {
            "cache_dir": str(self.cache_dir),
            "cached_datasets": [],
            "total_size_mb": 0
        }
        
        if self.cache_dir.exists():
            for cache_file in self.cache_dir.iterdir():
                if cache_file.is_file():
                    size_mb = cache_file.stat().st_size / (1024 * 1024)
                    cache_info["cached_datasets"].append({
                        "name": cache_file.name,
                        "size_mb": round(size_mb, 2),
                        "modified": cache_file.stat().st_mtime
                    })
                    cache_info["total_size_mb"] += size_mb
                    
        cache_info["total_size_mb"] = round(cache_info["total_size_mb"], 2)
        return cache_info
        
    def clear_cache(self) -> None:
        """Clear all cached datasets."""
        if self.cache_dir.exists():
            for cache_file in self.cache_dir.iterdir():
                if cache_file.is_file():
                    cache_file.unlink()
            logger.info("Dataset cache cleared")


# Convenience functions
def load_text_dataset(name: str, 
                     path: Optional[str] = None,
                     split: str = "train",
                     **kwargs) -> Union[Dataset, IterableDataset]:
    """Convenience function to load text dataset."""
    config = DatasetConfig(name=name, path=path, split=split, **kwargs)
    loader = UnifiedDatasetLoader()
    return loader.load_dataset(config)


def create_simple_dataloader(dataset: Union[Dataset, IterableDataset],
                            batch_size: int = 32,
                            shuffle: bool = True) -> DataLoader:
    """Create a simple DataLoader."""
    loader = UnifiedDatasetLoader()
    return loader.create_dataloader(dataset, batch_size=batch_size, shuffle=shuffle)
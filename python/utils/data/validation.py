"""
Data Validation Utilities for UnifiedTransformer

This module provides comprehensive data validation and integrity checking
for datasets and data pipelines with support for various data formats
and quality metrics.

Features:
- Dataset integrity validation
- Schema validation and type checking
- Data quality metrics and profiling
- Anomaly detection in datasets
- Duplicate detection and handling
- Statistical validation
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple, Set
from dataclasses import dataclass, field
from collections import Counter, defaultdict
import statistics

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    class Dataset: pass

from ..logging.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationConfig:
    """Configuration for data validation."""
    check_duplicates: bool = True
    check_empty_values: bool = True
    check_schema: bool = True
    check_statistics: bool = True
    check_anomalies: bool = True
    duplicate_threshold: float = 0.95  # Similarity threshold for duplicates
    outlier_threshold: float = 3.0  # Z-score threshold for outliers
    min_samples: int = 10  # Minimum samples required
    max_null_ratio: float = 0.1  # Maximum allowed null ratio


@dataclass
class ValidationResult:
    """Result of data validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)


@dataclass
class DataQualityMetrics:
    """Data quality metrics."""
    total_samples: int
    unique_samples: int
    duplicate_count: int
    empty_count: int
    null_ratio: float
    avg_length: float
    length_std: float
    min_length: int
    max_length: int
    outlier_count: int
    schema_violations: int


class DataValidator:
    """Comprehensive data validator."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        
    def validate_dataset(self, dataset: Union[Dataset, List[Dict[str, Any]], List[str]], 
                        schema: Optional[Dict[str, str]] = None) -> ValidationResult:
        """Validate a complete dataset."""
        result = ValidationResult(is_valid=True)
        
        try:
            # Convert dataset to standard format
            data = self._normalize_dataset(dataset)
            
            if not data:
                result.is_valid = False
                result.errors.append("Dataset is empty")
                return result
                
            # Check minimum samples
            if len(data) < self.config.min_samples:
                result.warnings.append(f"Dataset has only {len(data)} samples (< {self.config.min_samples} recommended)")
                
            # Run validation checks
            if self.config.check_duplicates:
                self._check_duplicates(data, result)
                
            if self.config.check_empty_values:
                self._check_empty_values(data, result)
                
            if self.config.check_schema and schema:
                self._check_schema(data, schema, result)
                
            if self.config.check_statistics:
                self._check_statistics(data, result)
                
            if self.config.check_anomalies:
                self._check_anomalies(data, result)
                
            # Generate quality metrics
            result.metrics = self._compute_quality_metrics(data)
            
            # Generate suggestions
            result.suggestions = self._generate_suggestions(result)
            
            # Final validation status
            if result.errors:
                result.is_valid = False
                
            logger.info(f"Dataset validation completed: {'PASSED' if result.is_valid else 'FAILED'}")
            
        except Exception as e:
            result.is_valid = False
            result.errors.append(f"Validation failed with error: {str(e)}")
            logger.error(f"Dataset validation error: {e}")
            
        return result
        
    def _normalize_dataset(self, dataset: Union[Dataset, List[Dict[str, Any]], List[str]]) -> List[Dict[str, Any]]:
        """Normalize dataset to standard format."""
        if isinstance(dataset, list):
            if not dataset:
                return []
                
            # Check if list of strings
            if isinstance(dataset[0], str):
                return [{"text": item} for item in dataset]
            elif isinstance(dataset[0], dict):
                return dataset
            else:
                return [{"data": str(item)} for item in dataset]
                
        elif hasattr(dataset, '__len__') and hasattr(dataset, '__getitem__'):
            # PyTorch Dataset or similar
            data = []
            for i in range(len(dataset)):
                try:
                    item = dataset[i]
                    if isinstance(item, dict):
                        data.append(item)
                    elif isinstance(item, str):
                        data.append({"text": item})
                    else:
                        data.append({"data": str(item)})
                except Exception as e:
                    logger.warning(f"Failed to access dataset item {i}: {e}")
            return data
            
        else:
            raise ValueError(f"Unsupported dataset type: {type(dataset)}")
            
    def _check_duplicates(self, data: List[Dict[str, Any]], result: ValidationResult) -> None:
        """Check for duplicate entries."""
        seen_hashes = set()
        duplicates = []
        
        for i, item in enumerate(data):
            # Create hash of item content
            item_str = json.dumps(item, sort_keys=True)
            item_hash = hashlib.md5(item_str.encode()).hexdigest()
            
            if item_hash in seen_hashes:
                duplicates.append(i)
            else:
                seen_hashes.add(item_hash)
                
        if duplicates:
            duplicate_ratio = len(duplicates) / len(data)
            if duplicate_ratio > 0.05:  # > 5% duplicates
                result.warnings.append(f"Found {len(duplicates)} duplicate entries ({duplicate_ratio:.1%})")
            else:
                logger.info(f"Found {len(duplicates)} duplicate entries ({duplicate_ratio:.1%})")
                
        result.metrics["duplicate_count"] = len(duplicates)
        result.metrics["duplicate_ratio"] = len(duplicates) / len(data) if data else 0
        
    def _check_empty_values(self, data: List[Dict[str, Any]], result: ValidationResult) -> None:
        """Check for empty or null values."""
        empty_count = 0
        field_null_counts = defaultdict(int)
        
        for item in data:
            is_empty = True
            for key, value in item.items():
                if value is None or value == "" or (isinstance(value, str) and not value.strip()):
                    field_null_counts[key] += 1
                else:
                    is_empty = False
                    
            if is_empty:
                empty_count += 1
                
        # Check overall empty ratio
        empty_ratio = empty_count / len(data) if data else 0
        if empty_ratio > 0.01:  # > 1% empty
            result.warnings.append(f"Found {empty_count} completely empty entries ({empty_ratio:.1%})")
            
        # Check field-specific null ratios
        for field, null_count in field_null_counts.items():
            null_ratio = null_count / len(data)
            if null_ratio > self.config.max_null_ratio:
                result.warnings.append(f"Field '{field}' has high null ratio: {null_ratio:.1%}")
                
        result.metrics["empty_count"] = empty_count
        result.metrics["empty_ratio"] = empty_ratio
        result.metrics["field_null_ratios"] = dict(field_null_counts)
        
    def _check_schema(self, data: List[Dict[str, Any]], schema: Dict[str, str], 
                     result: ValidationResult) -> None:
        """Check schema compliance."""
        schema_violations = 0
        missing_fields = defaultdict(int)
        type_violations = defaultdict(int)
        
        for item in data:
            # Check required fields
            for field, expected_type in schema.items():
                if field not in item:
                    missing_fields[field] += 1
                    schema_violations += 1
                else:
                    # Check type
                    value = item[field]
                    if not self._check_type(value, expected_type):
                        type_violations[field] += 1
                        schema_violations += 1
                        
        if schema_violations > 0:
            violation_ratio = schema_violations / (len(data) * len(schema))
            if violation_ratio > 0.05:  # > 5% violations
                result.errors.append(f"Schema violations: {schema_violations} ({violation_ratio:.1%})")
            else:
                result.warnings.append(f"Schema violations: {schema_violations} ({violation_ratio:.1%})")
                
        result.metrics["schema_violations"] = schema_violations
        result.metrics["missing_fields"] = dict(missing_fields)
        result.metrics["type_violations"] = dict(type_violations)
        
    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected type."""
        type_mapping = {
            "str": str,
            "int": int,
            "float": float,
            "bool": bool,
            "list": list,
            "dict": dict
        }
        
        if expected_type not in type_mapping:
            return True  # Unknown type, assume valid
            
        return isinstance(value, type_mapping[expected_type])
        
    def _check_statistics(self, data: List[Dict[str, Any]], result: ValidationResult) -> None:
        """Check statistical properties."""
        text_fields = self._identify_text_fields(data)
        
        for field in text_fields:
            lengths = []
            for item in data:
                if field in item and item[field]:
                    lengths.append(len(str(item[field])))
                    
            if lengths:
                mean_length = statistics.mean(lengths)
                std_length = statistics.stdev(lengths) if len(lengths) > 1 else 0
                min_length = min(lengths)
                max_length = max(lengths)
                
                # Check for extreme values
                if std_length > mean_length:  # High variance
                    result.warnings.append(f"Field '{field}' has high length variance (std: {std_length:.1f}, mean: {mean_length:.1f})")
                    
                result.metrics[f"{field}_length_stats"] = {
                    "mean": mean_length,
                    "std": std_length,
                    "min": min_length,
                    "max": max_length
                }
                
    def _check_anomalies(self, data: List[Dict[str, Any]], result: ValidationResult) -> None:
        """Check for anomalous data points."""
        text_fields = self._identify_text_fields(data)
        outlier_count = 0
        
        for field in text_fields:
            lengths = []
            for item in data:
                if field in item and item[field]:
                    lengths.append(len(str(item[field])))
                    
            if len(lengths) > 10:  # Need sufficient data for outlier detection
                mean_length = statistics.mean(lengths)
                std_length = statistics.stdev(lengths)
                
                if std_length > 0:
                    for length in lengths:
                        z_score = abs(length - mean_length) / std_length
                        if z_score > self.config.outlier_threshold:
                            outlier_count += 1
                            
        if outlier_count > 0:
            outlier_ratio = outlier_count / len(data)
            if outlier_ratio > 0.05:  # > 5% outliers
                result.warnings.append(f"Found {outlier_count} statistical outliers ({outlier_ratio:.1%})")
                
        result.metrics["outlier_count"] = outlier_count
        
    def _identify_text_fields(self, data: List[Dict[str, Any]]) -> List[str]:
        """Identify text fields in the dataset."""
        if not data:
            return []
            
        text_fields = []
        sample_item = data[0]
        
        for key, value in sample_item.items():
            if isinstance(value, str):
                text_fields.append(key)
                
        return text_fields
        
    def _compute_quality_metrics(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute comprehensive data quality metrics."""
        if not data:
            return {}
            
        # Basic counts
        total_samples = len(data)
        unique_samples = len(set(json.dumps(item, sort_keys=True) for item in data))
        
        # Text statistics
        text_fields = self._identify_text_fields(data)
        lengths = []
        
        for item in data:
            for field in text_fields:
                if field in item and item[field]:
                    lengths.append(len(str(item[field])))
                    
        metrics = {
            "total_samples": total_samples,
            "unique_samples": unique_samples,
            "uniqueness_ratio": unique_samples / total_samples if total_samples > 0 else 0,
            "text_fields": text_fields,
        }
        
        if lengths:
            metrics.update({
                "avg_length": statistics.mean(lengths),
                "length_std": statistics.stdev(lengths) if len(lengths) > 1 else 0,
                "min_length": min(lengths),
                "max_length": max(lengths),
                "median_length": statistics.median(lengths)
            })
            
        return metrics
        
    def _generate_suggestions(self, result: ValidationResult) -> List[str]:
        """Generate improvement suggestions based on validation results."""
        suggestions = []
        
        # Duplicate handling
        if "duplicate_ratio" in result.metrics and result.metrics["duplicate_ratio"] > 0.05:
            suggestions.append("Consider removing or handling duplicate entries")
            
        # Empty values
        if "empty_ratio" in result.metrics and result.metrics["empty_ratio"] > 0.01:
            suggestions.append("Filter out empty entries to improve data quality")
            
        # Length variance
        if "length_std" in result.metrics and "avg_length" in result.metrics:
            if result.metrics["length_std"] > result.metrics["avg_length"]:
                suggestions.append("High length variance detected - consider text normalization")
                
        # Outliers
        if "outlier_count" in result.metrics and result.metrics["outlier_count"] > 0:
            suggestions.append("Review and potentially filter statistical outliers")
            
        # Schema violations
        if "schema_violations" in result.metrics and result.metrics["schema_violations"] > 0:
            suggestions.append("Fix schema violations or update schema definition")
            
        return suggestions


class IntegrityChecker:
    """Data integrity and consistency checker."""
    
    def __init__(self):
        pass
        
    def check_file_integrity(self, file_path: Union[str, Path]) -> ValidationResult:
        """Check file integrity using checksums."""
        result = ValidationResult(is_valid=True)
        file_path = Path(file_path)
        
        if not file_path.exists():
            result.is_valid = False
            result.errors.append(f"File does not exist: {file_path}")
            return result
            
        try:
            # Calculate file checksum
            checksum = self._calculate_checksum(file_path)
            file_size = file_path.stat().st_size
            
            result.metrics.update({
                "file_path": str(file_path),
                "file_size_bytes": file_size,
                "checksum": checksum,
                "is_readable": True
            })
            
            # Basic file validation
            if file_size == 0:
                result.warnings.append("File is empty")
                
            logger.info(f"File integrity check passed: {file_path}")
            
        except Exception as e:
            result.is_valid = False
            result.errors.append(f"File integrity check failed: {str(e)}")
            
        return result
        
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate MD5 checksum of file."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
        
    def compare_datasets(self, dataset1: Union[Dataset, List], 
                        dataset2: Union[Dataset, List]) -> ValidationResult:
        """Compare two datasets for consistency."""
        result = ValidationResult(is_valid=True)
        
        try:
            validator = DataValidator(ValidationConfig())
            data1 = validator._normalize_dataset(dataset1)
            data2 = validator._normalize_dataset(dataset2)
            
            # Size comparison
            size_diff = abs(len(data1) - len(data2))
            if size_diff > 0:
                result.warnings.append(f"Dataset size difference: {size_diff} samples")
                
            # Content comparison (sample-based for large datasets)
            sample_size = min(100, len(data1), len(data2))
            if sample_size > 0:
                sample1 = data1[:sample_size]
                sample2 = data2[:sample_size]
                
                matches = sum(1 for i, j in zip(sample1, sample2) if i == j)
                similarity = matches / sample_size
                
                result.metrics.update({
                    "size1": len(data1),
                    "size2": len(data2),
                    "size_difference": size_diff,
                    "sample_similarity": similarity
                })
                
                if similarity < 0.9:
                    result.warnings.append(f"Low content similarity: {similarity:.1%}")
                    
        except Exception as e:
            result.is_valid = False
            result.errors.append(f"Dataset comparison failed: {str(e)}")
            
        return result


# Convenience functions
def validate_text_dataset(dataset: Union[Dataset, List[str]], 
                         min_length: int = 1,
                         max_length: int = 1000000,
                         check_duplicates: bool = True) -> ValidationResult:
    """Quick validation for text datasets."""
    config = ValidationConfig(
        check_duplicates=check_duplicates,
        check_empty_values=True,
        check_statistics=True,
        min_samples=1
    )
    
    validator = DataValidator(config)
    return validator.validate_dataset(dataset)


def check_data_quality(data: List[Dict[str, Any]]) -> DataQualityMetrics:
    """Get data quality metrics."""
    config = ValidationConfig()
    validator = DataValidator(config)
    result = validator.validate_dataset(data)
    
    metrics = result.metrics
    return DataQualityMetrics(
        total_samples=metrics.get("total_samples", 0),
        unique_samples=metrics.get("unique_samples", 0),
        duplicate_count=metrics.get("duplicate_count", 0),
        empty_count=metrics.get("empty_count", 0),
        null_ratio=metrics.get("empty_ratio", 0.0),
        avg_length=metrics.get("avg_length", 0.0),
        length_std=metrics.get("length_std", 0.0),
        min_length=metrics.get("min_length", 0),
        max_length=metrics.get("max_length", 0),
        outlier_count=metrics.get("outlier_count", 0),
        schema_violations=metrics.get("schema_violations", 0)
    )
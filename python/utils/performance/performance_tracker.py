from collections import defaultdict
from typing import Any, Dict, List
import statistics
import time

from ..logging.logger import get_logger


class PerformanceTracker:
    """Advanced performance tracking and analysis."""

    def __init__(self):
        self.operations = defaultdict(list)
        self.logger = get_logger("performance")

    def record_operation(self, operation: str, duration_ms: float,
                        memory_delta_mb: float = 0, memory_absolute_mb: float = 0, **metadata) -> None:
        """Record a performance measurement with improved memory monitoring."""
        record = {
            "operation": operation,
            "duration_ms": duration_ms,
            "memory_delta_mb": memory_delta_mb,
            "memory_absolute_mb": memory_absolute_mb,
            "timestamp": time.time(),
            **metadata
        }

        self.operations[operation].append(record)

        # Log if significant performance issue (improved thresholds)
        if duration_ms > 1000:  # > 1 second
            self.logger.warning(f"⚠️  Slow operation: {operation} took {duration_ms:.2f}ms")

        # Use absolute memory for warnings, ignore negative deltas (GC events)
        if memory_absolute_mb > 0:
            # Only warn for absolute memory usage above reasonable thresholds
            if memory_absolute_mb > 5000:  # > 5GB absolute usage
                self.logger.warning(f"⚠️  High memory usage: {operation} using {memory_absolute_mb:.1f}MB absolute")
            elif memory_delta_mb > 500:  # > 500MB positive delta
                self.logger.debug(f"Memory increase: {operation} +{memory_delta_mb:.1f}MB")
        elif memory_delta_mb < -100:  # Large negative delta indicates GC
            self.logger.debug(f"Memory freed: {operation} -{abs(memory_delta_mb):.1f}MB (GC event)")

    def get_operation_stats(self, operation: str) -> Dict[str, float]:
        """Get statistics for an operation."""
        if operation not in self.operations:
            return {}

        durations = [op["duration_ms"] for op in self.operations[operation]]
        memory_deltas = [op["memory_delta_mb"] for op in self.operations[operation]]

        stats = {
            "count": len(durations),
            "duration_mean_ms": statistics.mean(durations),
            "duration_median_ms": statistics.median(durations),
            "duration_std_ms": statistics.stdev(durations) if len(durations) > 1 else 0,
            "duration_min_ms": min(durations),
            "duration_max_ms": max(durations),
            "memory_mean_mb": statistics.mean(memory_deltas),
            "memory_total_mb": sum(memory_deltas)
        }

        return stats

    def detect_bottlenecks(self, threshold_ms: float = 100) -> List[str]:
        """Detect performance bottlenecks."""
        bottlenecks = []

        for operation, records in self.operations.items():
            if not records:
                continue

            avg_duration = sum(r["duration_ms"] for r in records) / len(records)
            if avg_duration > threshold_ms:
                bottlenecks.append(operation)

        return bottlenecks

    def generate_report(self) -> str:
        """Generate a performance report."""
        report = ["Performance Report", "=" * 50]

        for operation in sorted(self.operations.keys()):
            stats = self.get_operation_stats(operation)
            if stats:
                report.append(f"\n{operation}:")
                report.append(f"  Count: {stats['count']}")
                report.append(f"  Duration: {stats['duration_mean_ms']:.2f}ms ± {stats['duration_std_ms']:.2f}ms")
                report.append(f"  Range: {stats['duration_min_ms']:.2f}ms - {stats['duration_max_ms']:.2f}ms")
                report.append(f"  Memory: {stats['memory_mean_mb']:.1f}MB avg, {stats['memory_total_mb']:.1f}MB total")

        bottlenecks = self.detect_bottlenecks()
        if bottlenecks:
            report.append(f"\nBottlenecks (>100ms avg): {', '.join(bottlenecks)}")

        return "\n".join(report)

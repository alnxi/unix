from collections import defaultdict
from typing import Any, Dict, List
import numpy as np
import time
from dataclasses import dataclass, field

from ..logging.logger import get_logger


@dataclass
class QualityMetric:
    """Dataclass for a single quality metric entry."""
    timestamp: float = field(default_factory=time.time)
    batch_size: int
    difficulty_level: str
    quality_scores: Dict[str, float]
    generation_method: str
    overall_quality: float
    diversity_score: float = 0.0
    coherence_score: float = 0.0
    correctness_score: float = 0.0
    complexity_score: float = 0.0


@dataclass
class CurriculumStage:
    """Dataclass for a curriculum learning stage."""
    timestamp: float = field(default_factory=time.time)
    stage_name: str
    difficulty_progression: Dict[str, float]
    data_statistics: Dict[str, Any]
    samples_generated: int
    avg_quality: float
    quality_improvement: float


class SyntheticDataQualityTracker:
    """Track quality metrics for synthetic data generation."""

    def __init__(self,
                 low_quality_threshold: float = 0.7,
                 recent_metrics_count: int = 100,
                 trend_decline_threshold: float = -0.05,
                 low_quality_batch_threshold: float = 0.6,
                 low_quality_proportion_threshold: float = 0.2,
                 stagnation_variance_threshold: float = 0.001,
                 regression_threshold: float = -0.05,
                 low_efficiency_threshold: float = 0.02,
                 trend_min_values: int = 3):
        self.quality_metrics: List[QualityMetric] = []
        self.curriculum_stages: List[CurriculumStage] = []
        self.logger = get_logger("synthetic_data")

        # Configuration constants
        self.low_quality_threshold = low_quality_threshold
        self.recent_metrics_count = recent_metrics_count
        self.trend_decline_threshold = trend_decline_threshold
        self.low_quality_batch_threshold = low_quality_batch_threshold
        self.low_quality_proportion_threshold = low_quality_proportion_threshold
        self.stagnation_variance_threshold = stagnation_variance_threshold
        self.regression_threshold = regression_threshold
        self.low_efficiency_threshold = low_efficiency_threshold
        self.trend_min_values = trend_min_values

    def reset(self):
        """Reset the tracker's state."""
        self.quality_metrics.clear()
        self.curriculum_stages.clear()
        self.logger.info("SyntheticDataQualityTracker has been reset.")

    def log_generation_quality(self,
                             data_batch: Dict[str, Any],
                             quality_scores: Dict[str, float],
                             difficulty_level: str = "medium") -> None:
        """Log quality metrics for a batch of synthetic data."""
        overall_quality = np.mean(list(quality_scores.values()))
        
        quality_entry = QualityMetric(
            batch_size=data_batch.get("batch_size", 0),
            difficulty_level=difficulty_level,
            quality_scores=quality_scores,
            generation_method=data_batch.get("method", "unknown"),
            diversity_score=quality_scores.get("diversity", 0.0),
            coherence_score=quality_scores.get("coherence", 0.0),
            correctness_score=quality_scores.get("correctness", 0.0),
            complexity_score=quality_scores.get("complexity", 0.0),
            overall_quality=overall_quality
        )
        self.quality_metrics.append(quality_entry)

        # Log quality summary
        summary = (
            f"📊 Synthetic Data Quality - {difficulty_level}: "
            f"Overall: {quality_entry.overall_quality:.3f} | "
            f"Coherence: {quality_scores.get('coherence', 0):.3f} | "
            f"Correctness: {quality_scores.get('correctness', 0):.3f}"
        )
        self.logger.info(summary)

        # Warn if quality is low
        if quality_entry.overall_quality < self.low_quality_threshold:
            self.logger.warning(
                f"⚠️ Low quality synthetic data detected: "
                f"{quality_entry.overall_quality:.3f} < {self.low_quality_threshold} threshold"
            )

    def log_curriculum_stage(self,
                           stage_name: str,
                           difficulty_progression: Dict[str, float],
                           data_statistics: Dict[str, Any]) -> None:
        """Log curriculum learning stage progression."""
        quality_improvement = self._calculate_quality_improvement()
        
        stage_entry = CurriculumStage(
            stage_name=stage_name,
            difficulty_progression=difficulty_progression,
            data_statistics=data_statistics,
            samples_generated=data_statistics.get("total_samples", 0),
            avg_quality=data_statistics.get("avg_quality", 0.0),
            quality_improvement=quality_improvement
        )
        self.curriculum_stages.append(stage_entry)

        summary = (
            f"🎓 Curriculum Stage: {stage_name} | "
            f"Samples: {stage_entry.samples_generated:,} | "
            f"Avg Quality: {stage_entry.avg_quality:.3f} | "
            f"Improvement: {stage_entry.quality_improvement:+.3f}"
        )
        self.logger.info(summary)

    def _calculate_quality_improvement(self) -> float:
        """Calculate quality improvement over the last few stages."""
        if len(self.curriculum_stages) < 2:
            return 0.0
        current_avg = self._get_avg_quality_from_stage(-1)
        previous_avg = self._get_avg_quality_from_stage(-2)
        return current_avg - previous_avg

    def analyze_data_quality_trends(self) -> Dict[str, Any]:
        """Analyze trends in synthetic data quality over time."""
        if not self.quality_metrics:
            return {}

        recent_metrics = self.quality_metrics[-self.recent_metrics_count:]

        by_difficulty = defaultdict(list)
        for metric in recent_metrics:
            by_difficulty[metric.difficulty_level].append(metric.overall_quality)

        analysis = {
            "total_batches_analyzed": len(recent_metrics),
            "overall_quality_trend": self._calculate_trend([m.overall_quality for m in recent_metrics]),
            "quality_by_difficulty": {},
            "quality_distribution": self._calculate_quality_distribution(recent_metrics),
            "recommendations": []
        }

        for difficulty, qualities in by_difficulty.items():
            if qualities:
                analysis["quality_by_difficulty"][difficulty] = {
                    "avg_quality": np.mean(qualities),
                    "quality_std": np.std(qualities),
                    "batch_count": len(qualities),
                    "trend": self._calculate_trend(qualities)
                }

        # Generate recommendations
        if analysis["overall_quality_trend"] < self.trend_decline_threshold:
            analysis["recommendations"].append("Data quality is declining - review generation parameters")

        low_quality_batches = [m for m in recent_metrics if m.overall_quality < self.low_quality_batch_threshold]
        if len(low_quality_batches) > len(recent_metrics) * self.low_quality_proportion_threshold:
            analysis["recommendations"].append(
                f"High proportion of low-quality batches ({len(low_quality_batches)}/{len(recent_metrics)})"
            )

        return analysis

    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend direction (-1 to 1, negative = declining)."""
        if len(values) < self.trend_min_values:
            return 0.0

        x = np.arange(len(values))
        y = np.array(values)

        try:
            slope = np.polyfit(x, y, 1)[0]
            return np.clip(slope * 10, -1.0, 1.0)
        except np.linalg.LinAlgError:
            self.logger.warning("Could not compute trend due to singular matrix.")
            return 0.0

    def _calculate_quality_distribution(self, metrics: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate distribution of quality scores."""
        if not metrics:
            return {}

        qualities = [m.overall_quality for m in metrics]
        count = len(qualities)

        return {
            "excellent": len([q for q in qualities if q >= 0.9]) / count,
            "good": len([q for q in qualities if 0.7 <= q < 0.9]) / count,
            "fair": len([q for q in qualities if 0.5 <= q < 0.7]) / count,
            "poor": len([q for q in qualities if q < 0.5]) / count
        }

    def get_curriculum_progress_report(self) -> Dict[str, Any]:
        """Generate comprehensive curriculum learning progress report."""
        if not self.curriculum_stages:
            return {"status": "No curriculum stages recorded"}

        latest_stage = self.curriculum_stages[-1]
        quality_progression = [stage.avg_quality for stage in self.curriculum_stages]

        return {
            "current_stage": latest_stage.stage_name,
            "total_stages_completed": len(self.curriculum_stages),
            "overall_quality_improvement": quality_progression[-1] - quality_progression[0] if len(quality_progression) >= 2 else 0.0,
            "quality_progression": quality_progression,
            "total_samples_generated": sum(s.samples_generated for s in self.curriculum_stages),
            "current_quality": latest_stage.avg_quality,
            "curriculum_efficiency": self._calculate_curriculum_efficiency(),
            "recommendations": self._generate_curriculum_recommendations()
        }

    def _get_avg_quality_from_stage(self, index: int) -> float:
        """Helper to get average quality from a curriculum stage."""
        return self.curriculum_stages[index].avg_quality

    def _calculate_curriculum_efficiency(self) -> float:
        """Calculate how efficiently the curriculum is improving quality."""
        if len(self.curriculum_stages) < 2:
            return 1.0

        improvements = [
            max(0, self._get_avg_quality_from_stage(i) - self._get_avg_quality_from_stage(i - 1))
            for i in range(1, len(self.curriculum_stages))
        ]
        return np.mean(improvements) if improvements else 0.0

    def _generate_curriculum_recommendations(self) -> List[str]:
        """Generate recommendations for curriculum learning optimization."""
        if not self.curriculum_stages:
            return ["Begin curriculum learning by defining difficulty stages"]

        recommendations = []
        num_stages = len(self.curriculum_stages)

        # Check for stagnation
        if num_stages >= 3:
            recent_qualities = [self._get_avg_quality_from_stage(i) for i in range(-3, 0)]
            if np.var(recent_qualities) < self.stagnation_variance_threshold:
                recommendations.append("Quality improvement has stagnated - consider adjusting difficulty progression")

        # Check for quality regression
        if num_stages >= 2:
            if self._get_avg_quality_from_stage(-1) < self._get_avg_quality_from_stage(-2) + self.regression_threshold:
                recommendations.append("Quality regression detected - review recent changes to generation parameters")

        # Check curriculum efficiency
        if self._calculate_curriculum_efficiency() < self.low_efficiency_threshold:
            recommendations.append("Low curriculum efficiency - consider larger difficulty steps or different progression strategy")

        return recommendations

"""Automated optimization recommendations system.

This module implements an intelligent recommendation engine that:
- Analyzes performance metrics and bottlenecks
- Provides actionable optimization recommendations
- Learns from historical optimization outcomes
- Prioritizes recommendations by impact and ease of implementation
"""

import logging
import time
import json
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque
import numpy as np
import math

from ..monitoring.performance_monitor import PerformanceMetric, BottleneckDetection, BottleneckType, AlertSeverity

logger = logging.getLogger(__name__)


class RecommendationType(Enum):
    """Types of optimization recommendations."""
    CONFIGURATION = "configuration"
    HARDWARE = "hardware" 
    MODEL_ARCHITECTURE = "model_architecture"
    TRAINING_STRATEGY = "training_strategy"
    DATA_PIPELINE = "data_pipeline"
    DISTRIBUTED_TRAINING = "distributed_training"
    MEMORY_OPTIMIZATION = "memory_optimization"
    PERFORMANCE_TUNING = "performance_tuning"


class RecommendationPriority(Enum):
    """Priority levels for recommendations."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class ImplementationDifficulty(Enum):
    """Implementation difficulty levels."""
    EASY = "easy"           # Simple config change
    MODERATE = "moderate"   # Code changes required
    HARD = "hard"          # Significant refactoring
    COMPLEX = "complex"     # System architecture changes


@dataclass
class OptimizationRecommendation:
    """Individual optimization recommendation."""
    recommendation_id: str
    title: str
    description: str
    recommendation_type: RecommendationType
    priority: RecommendationPriority
    difficulty: ImplementationDifficulty
    
    # Expected impact
    expected_improvement: Dict[str, float]  # metric -> improvement percentage
    confidence: float  # 0.0 to 1.0
    
    # Implementation details
    implementation_steps: List[str]
    configuration_changes: Dict[str, Any]
    code_changes: List[str]
    
    # Context
    triggered_by_bottleneck: Optional[str] = None
    applicable_metrics: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    estimated_implementation_time: int = 30  # minutes
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'recommendation_id': self.recommendation_id,
            'title': self.title,
            'description': self.description,
            'recommendation_type': self.recommendation_type.value,
            'priority': self.priority.value,
            'difficulty': self.difficulty.value,
            'expected_improvement': self.expected_improvement,
            'confidence': self.confidence,
            'implementation_steps': self.implementation_steps,
            'configuration_changes': self.configuration_changes,
            'code_changes': self.code_changes,
            'triggered_by_bottleneck': self.triggered_by_bottleneck,
            'applicable_metrics': self.applicable_metrics,
            'prerequisites': self.prerequisites,
            'created_at': self.created_at.isoformat(),
            'estimated_implementation_time': self.estimated_implementation_time
        }


@dataclass
class RecommendationFeedback:
    """Feedback on implemented recommendations."""
    recommendation_id: str
    implemented: bool
    implementation_time_minutes: int
    actual_improvement: Dict[str, float]
    user_satisfaction: float  # 1-5 scale
    notes: str
    feedback_date: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class RecommendationRules:
    """Rule-based recommendation generation."""
    
    @staticmethod
    def get_cpu_bottleneck_recommendations() -> List[OptimizationRecommendation]:
        """Get recommendations for CPU bottlenecks."""
        return [
            OptimizationRecommendation(
                recommendation_id="cpu_reduce_batch_size",
                title="Reduce Batch Size",
                description="Decrease batch size to reduce CPU computational load per iteration",
                recommendation_type=RecommendationType.CONFIGURATION,
                priority=RecommendationPriority.HIGH,
                difficulty=ImplementationDifficulty.EASY,
                expected_improvement={"cpu_utilization": -15.0, "training_time": 5.0},
                confidence=0.8,
                implementation_steps=[
                    "Identify current batch size in configuration",
                    "Reduce batch size by 25-50%",
                    "Adjust learning rate proportionally if needed",
                    "Monitor training stability"
                ],
                configuration_changes={"batch_size": "reduce_by_factor", "learning_rate": "scale_proportionally"},
                code_changes=[],
                applicable_metrics=["cpu_utilization", "avg_batch_processing_time"],
                estimated_implementation_time=5
            ),
            OptimizationRecommendation(
                recommendation_id="cpu_mixed_precision",
                title="Enable Mixed Precision Training",
                description="Use FP16 instead of FP32 to reduce computational load and memory usage",
                recommendation_type=RecommendationType.TRAINING_STRATEGY,
                priority=RecommendationPriority.HIGH,
                difficulty=ImplementationDifficulty.MODERATE,
                expected_improvement={"cpu_utilization": -20.0, "memory_usage": -30.0, "training_speed": 40.0},
                confidence=0.85,
                implementation_steps=[
                    "Enable automatic mixed precision (AMP)",
                    "Add GradScaler for gradient scaling",
                    "Wrap forward pass with autocast",
                    "Test for numerical stability"
                ],
                configuration_changes={"mixed_precision": True},
                code_changes=[
                    "Import torch.cuda.amp",
                    "Add scaler = GradScaler()",
                    "Wrap forward pass with autocast()",
                    "Scale gradients before backward()"
                ],
                applicable_metrics=["cpu_utilization", "memory_utilization", "training_speed"],
                estimated_implementation_time=30
            ),
            OptimizationRecommendation(
                recommendation_id="cpu_data_loading_optimization",
                title="Optimize Data Loading",
                description="Increase number of data loader workers and enable prefetching",
                recommendation_type=RecommendationType.DATA_PIPELINE,
                priority=RecommendationPriority.MEDIUM,
                difficulty=ImplementationDifficulty.EASY,
                expected_improvement={"cpu_utilization": -10.0, "data_loading_time": -25.0},
                confidence=0.75,
                implementation_steps=[
                    "Increase num_workers in DataLoader",
                    "Enable pin_memory for GPU training",
                    "Set prefetch_factor for data prefetching",
                    "Optimize data preprocessing pipeline"
                ],
                configuration_changes={
                    "num_workers": "increase_to_cpu_count",
                    "pin_memory": True,
                    "prefetch_factor": 2
                },
                code_changes=[],
                applicable_metrics=["cpu_utilization", "data_loading_time"],
                estimated_implementation_time=15
            )
        ]
    
    @staticmethod
    def get_memory_bottleneck_recommendations() -> List[OptimizationRecommendation]:
        """Get recommendations for memory bottlenecks."""
        return [
            OptimizationRecommendation(
                recommendation_id="memory_gradient_checkpointing",
                title="Enable Gradient Checkpointing",
                description="Trade computation for memory by recomputing activations during backward pass",
                recommendation_type=RecommendationType.MEMORY_OPTIMIZATION,
                priority=RecommendationPriority.HIGH,
                difficulty=ImplementationDifficulty.MODERATE,
                expected_improvement={"memory_utilization": -40.0, "training_time": 15.0},
                confidence=0.9,
                implementation_steps=[
                    "Enable gradient checkpointing in model",
                    "Identify memory-intensive layers",
                    "Apply checkpointing strategically",
                    "Monitor training performance"
                ],
                configuration_changes={"gradient_checkpointing": True},
                code_changes=[
                    "Import torch.utils.checkpoint",
                    "Wrap forward pass with checkpoint()",
                    "Enable checkpointing in transformer layers"
                ],
                applicable_metrics=["memory_utilization", "gpu_memory_usage"],
                estimated_implementation_time=45
            ),
            OptimizationRecommendation(
                recommendation_id="memory_reduce_batch_size",
                title="Reduce Batch Size with Gradient Accumulation",
                description="Lower batch size and use gradient accumulation to maintain effective batch size",
                recommendation_type=RecommendationType.TRAINING_STRATEGY,
                priority=RecommendationPriority.HIGH,
                difficulty=ImplementationDifficulty.EASY,
                expected_improvement={"memory_utilization": -30.0},
                confidence=0.85,
                implementation_steps=[
                    "Reduce physical batch size",
                    "Calculate accumulation steps to maintain effective batch size",
                    "Modify training loop for accumulation",
                    "Update learning rate scaling if needed"
                ],
                configuration_changes={
                    "batch_size": "reduce_by_factor",
                    "gradient_accumulation_steps": "calculate_to_maintain_effective_size"
                },
                code_changes=[
                    "Add gradient accumulation loop",
                    "Scale loss by accumulation steps"
                ],
                applicable_metrics=["memory_utilization"],
                estimated_implementation_time=20
            ),
            OptimizationRecommendation(
                recommendation_id="memory_cpu_offloading",
                title="Enable CPU Offloading",
                description="Offload model parameters and optimizer states to CPU when not in use",
                recommendation_type=RecommendationType.MEMORY_OPTIMIZATION,
                priority=RecommendationPriority.MEDIUM,
                difficulty=ImplementationDifficulty.HARD,
                expected_improvement={"memory_utilization": -50.0, "training_time": 25.0},
                confidence=0.7,
                implementation_steps=[
                    "Implement parameter offloading",
                    "Move optimizer states to CPU",
                    "Add asynchronous data movement",
                    "Optimize transfer patterns"
                ],
                configuration_changes={"cpu_offloading": True},
                code_changes=[
                    "Implement offloading hooks",
                    "Add async parameter movement",
                    "Modify optimizer for CPU states"
                ],
                applicable_metrics=["memory_utilization", "gpu_memory_usage"],
                estimated_implementation_time=120
            )
        ]
    
    @staticmethod
    def get_gpu_bottleneck_recommendations() -> List[OptimizationRecommendation]:
        """Get recommendations for GPU bottlenecks."""
        return [
            OptimizationRecommendation(
                recommendation_id="gpu_model_parallelism",
                title="Implement Model Parallelism",
                description="Split model across multiple GPUs to increase computational capacity",
                recommendation_type=RecommendationType.DISTRIBUTED_TRAINING,
                priority=RecommendationPriority.HIGH,
                difficulty=ImplementationDifficulty.HARD,
                expected_improvement={"gpu_utilization": -30.0, "training_speed": 60.0},
                confidence=0.8,
                implementation_steps=[
                    "Analyze model architecture for parallelism",
                    "Implement model splitting strategy",
                    "Add inter-GPU communication",
                    "Optimize pipeline execution"
                ],
                configuration_changes={"model_parallel": True, "pipeline_stages": "auto"},
                code_changes=[
                    "Implement model splitting",
                    "Add pipeline parallel training",
                    "Optimize gradient synchronization"
                ],
                applicable_metrics=["gpu_utilization", "training_speed"],
                prerequisites=["multiple_gpus"],
                estimated_implementation_time=180
            ),
            OptimizationRecommendation(
                recommendation_id="gpu_kernel_optimization",
                title="Optimize GPU Kernels",
                description="Use optimized kernels and enable kernel fusion for better GPU utilization",
                recommendation_type=RecommendationType.PERFORMANCE_TUNING,
                priority=RecommendationPriority.MEDIUM,
                difficulty=ImplementationDifficulty.MODERATE,
                expected_improvement={"gpu_utilization": 15.0, "training_speed": 20.0},
                confidence=0.7,
                implementation_steps=[
                    "Enable kernel fusion optimizations",
                    "Use optimized attention implementations",
                    "Optimize memory access patterns",
                    "Profile kernel performance"
                ],
                configuration_changes={"kernel_fusion": True, "optimized_attention": True},
                code_changes=[
                    "Replace standard operations with fused versions",
                    "Use flash attention or similar optimizations"
                ],
                applicable_metrics=["gpu_utilization", "training_speed"],
                estimated_implementation_time=60
            )
        ]
    
    @staticmethod
    def get_data_loading_recommendations() -> List[OptimizationRecommendation]:
        """Get recommendations for data loading bottlenecks."""
        return [
            OptimizationRecommendation(
                recommendation_id="data_caching",
                title="Implement Data Caching",
                description="Cache preprocessed data to reduce I/O and preprocessing overhead",
                recommendation_type=RecommendationType.DATA_PIPELINE,
                priority=RecommendationPriority.HIGH,
                difficulty=ImplementationDifficulty.MODERATE,
                expected_improvement={"data_loading_time": -50.0, "cpu_utilization": -15.0},
                confidence=0.85,
                implementation_steps=[
                    "Implement intelligent caching layer",
                    "Cache preprocessed samples",
                    "Add cache invalidation logic",
                    "Monitor cache hit rates"
                ],
                configuration_changes={"enable_data_cache": True, "cache_size_gb": 4},
                code_changes=[
                    "Add caching to data loader",
                    "Implement LRU cache eviction",
                    "Add cache warming strategies"
                ],
                applicable_metrics=["data_loading_time", "disk_io"],
                estimated_implementation_time=90
            ),
            OptimizationRecommendation(
                recommendation_id="data_prefetching",
                title="Advanced Data Prefetching",
                description="Implement multi-level prefetching to hide data loading latency",
                recommendation_type=RecommendationType.DATA_PIPELINE,
                priority=RecommendationPriority.MEDIUM,
                difficulty=ImplementationDifficulty.MODERATE,
                expected_improvement={"data_loading_time": -30.0, "gpu_idle_time": -25.0},
                confidence=0.75,
                implementation_steps=[
                    "Implement async data prefetching",
                    "Add GPU memory prefetching",
                    "Optimize prefetch queue size",
                    "Balance memory usage and performance"
                ],
                configuration_changes={"prefetch_factor": 4, "prefetch_to_gpu": True},
                code_changes=[
                    "Implement async prefetch pipeline",
                    "Add GPU memory management for prefetch"
                ],
                applicable_metrics=["data_loading_time", "gpu_utilization"],
                estimated_implementation_time=75
            )
        ]


class LearningRecommendationEngine:
    """Machine learning-based recommendation engine that learns from feedback."""
    
    def __init__(self):
        """Initialize learning engine."""
        self.recommendation_history: List[OptimizationRecommendation] = []
        self.feedback_history: List[RecommendationFeedback] = []
        self.effectiveness_scores: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # Simple learning parameters
        self.learning_rate = 0.1
        self.confidence_decay = 0.95
        
    def update_recommendation_effectiveness(
        self, 
        feedback: RecommendationFeedback
    ) -> None:
        """Update recommendation effectiveness based on feedback."""
        self.feedback_history.append(feedback)
        
        if not feedback.implemented:
            return
        
        # Find the original recommendation
        recommendation = None
        for rec in self.recommendation_history:
            if rec.recommendation_id == feedback.recommendation_id:
                recommendation = rec
                break
        
        if not recommendation:
            return
        
        # Calculate effectiveness score
        effectiveness = 0.0
        
        # Compare expected vs actual improvements
        for metric, expected_improvement in recommendation.expected_improvement.items():
            if metric in feedback.actual_improvement:
                actual_improvement = feedback.actual_improvement[metric]
                
                # Score based on how close actual was to expected
                if expected_improvement != 0:
                    ratio = actual_improvement / expected_improvement
                    effectiveness += min(1.0, ratio) * 0.5  # Cap at 1.0, weight by 0.5
        
        # Factor in user satisfaction
        effectiveness += (feedback.user_satisfaction / 5.0) * 0.3
        
        # Factor in implementation time accuracy
        expected_time = recommendation.estimated_implementation_time
        actual_time = feedback.implementation_time_minutes
        
        if expected_time > 0:
            time_accuracy = min(1.0, expected_time / max(actual_time, 1))
            effectiveness += time_accuracy * 0.2
        
        # Store effectiveness score
        rec_type = recommendation.recommendation_type.value
        bottleneck_type = recommendation.triggered_by_bottleneck or "general"
        
        if bottleneck_type not in self.effectiveness_scores[rec_type]:
            self.effectiveness_scores[rec_type][bottleneck_type] = effectiveness
        else:
            # Update with learning rate
            current_score = self.effectiveness_scores[rec_type][bottleneck_type]
            self.effectiveness_scores[rec_type][bottleneck_type] = (
                current_score * (1 - self.learning_rate) + effectiveness * self.learning_rate
            )
        
        logger.info(f"Updated effectiveness for {rec_type}/{bottleneck_type}: {effectiveness:.3f}")
    
    def adjust_recommendation_confidence(
        self, 
        recommendation: OptimizationRecommendation
    ) -> float:
        """Adjust recommendation confidence based on historical effectiveness."""
        rec_type = recommendation.recommendation_type.value
        bottleneck_type = recommendation.triggered_by_bottleneck or "general"
        
        if (rec_type in self.effectiveness_scores and 
            bottleneck_type in self.effectiveness_scores[rec_type]):
            
            historical_effectiveness = self.effectiveness_scores[rec_type][bottleneck_type]
            
            # Adjust confidence based on historical performance
            adjusted_confidence = (
                recommendation.confidence * 0.7 + historical_effectiveness * 0.3
            )
            
            return min(1.0, max(0.1, adjusted_confidence))
        
        return recommendation.confidence
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get learning statistics."""
        total_feedback = len(self.feedback_history)
        implemented_feedback = len([f for f in self.feedback_history if f.implemented])
        
        avg_satisfaction = 0.0
        if implemented_feedback > 0:
            avg_satisfaction = np.mean([
                f.user_satisfaction for f in self.feedback_history if f.implemented
            ])
        
        return {
            'total_recommendations': len(self.recommendation_history),
            'total_feedback': total_feedback,
            'implemented_recommendations': implemented_feedback,
            'implementation_rate': implemented_feedback / total_feedback if total_feedback > 0 else 0,
            'average_satisfaction': avg_satisfaction,
            'effectiveness_scores': dict(self.effectiveness_scores)
        }


class AutomatedRecommendationEngine:
    """Main recommendation engine that combines rules and learning."""
    
    def __init__(self, enable_learning: bool = True):
        """Initialize recommendation engine.
        
        Args:
            enable_learning: Whether to enable learning from feedback
        """
        self.enable_learning = enable_learning
        self.learning_engine = LearningRecommendationEngine() if enable_learning else None
        
        # Generated recommendations
        self.active_recommendations: Dict[str, OptimizationRecommendation] = {}
        self.recommendation_history: List[OptimizationRecommendation] = []
        
        # Analytics
        self.generation_count = 0
        self.bottleneck_triggers: Dict[str, int] = defaultdict(int)
        
        logger.info(f"Automated recommendation engine initialized (learning={'enabled' if enable_learning else 'disabled'})")
    
    def generate_recommendations(
        self,
        metrics: List[PerformanceMetric],
        bottlenecks: List[BottleneckDetection],
        system_context: Dict[str, Any] = None
    ) -> List[OptimizationRecommendation]:
        """Generate optimization recommendations based on current state.
        
        Args:
            metrics: Current performance metrics
            bottlenecks: Detected bottlenecks
            system_context: Additional system context
            
        Returns:
            List of prioritized recommendations
        """
        recommendations = []
        system_context = system_context or {}
        
        # Generate rule-based recommendations for each bottleneck
        for bottleneck in bottlenecks:
            bottleneck_recommendations = self._generate_bottleneck_recommendations(bottleneck)
            recommendations.extend(bottleneck_recommendations)
            self.bottleneck_triggers[bottleneck.bottleneck_type.value] += 1
        
        # Generate proactive recommendations based on metrics
        proactive_recommendations = self._generate_proactive_recommendations(metrics, system_context)
        recommendations.extend(proactive_recommendations)
        
        # Remove duplicates and prioritize
        recommendations = self._deduplicate_and_prioritize(recommendations)
        
        # Apply learning adjustments
        if self.learning_engine:
            for recommendation in recommendations:
                recommendation.confidence = self.learning_engine.adjust_recommendation_confidence(
                    recommendation
                )
        
        # Store recommendations
        for recommendation in recommendations:
            self.active_recommendations[recommendation.recommendation_id] = recommendation
            if self.learning_engine:
                self.learning_engine.recommendation_history.append(recommendation)
        
        self.recommendation_history.extend(recommendations)
        self.generation_count += 1
        
        logger.info(f"Generated {len(recommendations)} optimization recommendations")
        
        return recommendations
    
    def _generate_bottleneck_recommendations(
        self, 
        bottleneck: BottleneckDetection
    ) -> List[OptimizationRecommendation]:
        """Generate recommendations for a specific bottleneck."""
        recommendations = []
        
        # Get rule-based recommendations
        if bottleneck.bottleneck_type == BottleneckType.CPU_BOUND:
            recommendations = RecommendationRules.get_cpu_bottleneck_recommendations()
        elif bottleneck.bottleneck_type == BottleneckType.MEMORY_BOUND:
            recommendations = RecommendationRules.get_memory_bottleneck_recommendations()
        elif bottleneck.bottleneck_type == BottleneckType.GPU_BOUND:
            recommendations = RecommendationRules.get_gpu_bottleneck_recommendations()
        elif bottleneck.bottleneck_type == BottleneckType.DATA_LOADING:
            recommendations = RecommendationRules.get_data_loading_recommendations()
        
        # Adjust priorities based on bottleneck severity
        severity_multiplier = {
            AlertSeverity.INFO: 1.0,
            AlertSeverity.WARNING: 1.2,
            AlertSeverity.ERROR: 1.5,
            AlertSeverity.CRITICAL: 2.0
        }
        
        multiplier = severity_multiplier.get(bottleneck.severity, 1.0)
        
        for recommendation in recommendations:
            # Link to bottleneck
            recommendation.triggered_by_bottleneck = bottleneck.bottleneck_id
            
            # Adjust priority based on severity
            if multiplier > 1.0 and recommendation.priority != RecommendationPriority.CRITICAL:
                if recommendation.priority == RecommendationPriority.LOW:
                    recommendation.priority = RecommendationPriority.MEDIUM
                elif recommendation.priority == RecommendationPriority.MEDIUM:
                    recommendation.priority = RecommendationPriority.HIGH
                elif recommendation.priority == RecommendationPriority.HIGH:
                    recommendation.priority = RecommendationPriority.CRITICAL
            
            # Adjust confidence based on bottleneck confidence
            recommendation.confidence *= bottleneck.confidence
        
        return recommendations
    
    def _generate_proactive_recommendations(
        self,
        metrics: List[PerformanceMetric],
        system_context: Dict[str, Any]
    ) -> List[OptimizationRecommendation]:
        """Generate proactive recommendations based on metrics trends."""
        recommendations = []
        
        # Analyze metrics for potential optimizations
        metrics_by_name = {metric.metric_name: metric.value for metric in metrics}
        
        # Memory efficiency recommendation
        memory_utilization = metrics_by_name.get('memory_utilization', 0)
        if 50 <= memory_utilization <= 70:  # Moderate usage, room for optimization
            recommendations.append(
                OptimizationRecommendation(
                    recommendation_id="proactive_memory_opt",
                    title="Proactive Memory Optimization",
                    description="Memory usage is moderate but could be optimized preemptively",
                    recommendation_type=RecommendationType.MEMORY_OPTIMIZATION,
                    priority=RecommendationPriority.LOW,
                    difficulty=ImplementationDifficulty.EASY,
                    expected_improvement={"memory_utilization": -15.0},
                    confidence=0.6,
                    implementation_steps=[
                        "Enable memory profiling",
                        "Identify memory-intensive operations",
                        "Implement memory pooling",
                        "Add garbage collection optimization"
                    ],
                    configuration_changes={"memory_optimization": True},
                    code_changes=[],
                    applicable_metrics=["memory_utilization"],
                    estimated_implementation_time=20
                )
            )
        
        # Training speed optimization
        batch_time = metrics_by_name.get('avg_batch_processing_time', 0)
        if batch_time > 500:  # Slow batches (>500ms)
            recommendations.append(
                OptimizationRecommendation(
                    recommendation_id="proactive_speed_opt",
                    title="Training Speed Optimization",
                    description="Batch processing time suggests room for speed improvements",
                    recommendation_type=RecommendationType.PERFORMANCE_TUNING,
                    priority=RecommendationPriority.MEDIUM,
                    difficulty=ImplementationDifficulty.MODERATE,
                    expected_improvement={"avg_batch_processing_time": -25.0},
                    confidence=0.7,
                    implementation_steps=[
                        "Profile training pipeline",
                        "Optimize critical path operations",
                        "Enable compilation optimizations",
                        "Consider mixed precision training"
                    ],
                    configuration_changes={"enable_optimizations": True},
                    code_changes=["Add profiling hooks", "Optimize bottleneck operations"],
                    applicable_metrics=["avg_batch_processing_time"],
                    estimated_implementation_time=45
                )
            )
        
        # GPU utilization optimization
        gpu_utilization = metrics_by_name.get('gpu_0_utilization', 0)
        if 0 < gpu_utilization < 60:  # Low GPU usage
            recommendations.append(
                OptimizationRecommendation(
                    recommendation_id="proactive_gpu_opt",
                    title="GPU Utilization Improvement",
                    description="GPU utilization is low, suggesting optimization opportunities",
                    recommendation_type=RecommendationType.PERFORMANCE_TUNING,
                    priority=RecommendationPriority.MEDIUM,
                    difficulty=ImplementationDifficulty.MODERATE,
                    expected_improvement={"gpu_0_utilization": 20.0, "training_speed": 15.0},
                    confidence=0.65,
                    implementation_steps=[
                        "Analyze GPU kernel efficiency",
                        "Increase batch size if memory allows",
                        "Optimize data transfer patterns",
                        "Consider model complexity adjustments"
                    ],
                    configuration_changes={"optimize_gpu_usage": True},
                    code_changes=["Add GPU profiling", "Optimize data transfers"],
                    applicable_metrics=["gpu_0_utilization", "training_speed"],
                    estimated_implementation_time=60
                )
            )
        
        return recommendations
    
    def _deduplicate_and_prioritize(
        self, 
        recommendations: List[OptimizationRecommendation]
    ) -> List[OptimizationRecommendation]:
        """Remove duplicates and prioritize recommendations."""
        # Remove duplicates by ID
        seen_ids = set()
        unique_recommendations = []
        
        for rec in recommendations:
            if rec.recommendation_id not in seen_ids:
                seen_ids.add(rec.recommendation_id)
                unique_recommendations.append(rec)
        
        # Sort by priority, confidence, and expected impact
        def priority_score(rec):
            priority_weight = rec.priority.value * 100
            confidence_weight = rec.confidence * 50
            impact_weight = sum(abs(v) for v in rec.expected_improvement.values())
            difficulty_penalty = {
                ImplementationDifficulty.EASY: 0,
                ImplementationDifficulty.MODERATE: 5,
                ImplementationDifficulty.HARD: 15,
                ImplementationDifficulty.COMPLEX: 30
            }[rec.difficulty]
            
            return priority_weight + confidence_weight + impact_weight - difficulty_penalty
        
        unique_recommendations.sort(key=priority_score, reverse=True)
        
        # Limit to top recommendations
        return unique_recommendations[:10]
    
    def submit_feedback(
        self, 
        recommendation_id: str, 
        feedback: RecommendationFeedback
    ) -> None:
        """Submit feedback for a recommendation.
        
        Args:
            recommendation_id: ID of the recommendation
            feedback: Feedback object
        """
        if self.learning_engine:
            self.learning_engine.update_recommendation_effectiveness(feedback)
            logger.info(f"Received feedback for recommendation {recommendation_id}")
        
        # Remove from active recommendations if implemented
        if feedback.implemented and recommendation_id in self.active_recommendations:
            del self.active_recommendations[recommendation_id]
    
    def get_active_recommendations(
        self, 
        limit: int = 5,
        min_confidence: float = 0.0
    ) -> List[OptimizationRecommendation]:
        """Get active recommendations.
        
        Args:
            limit: Maximum number of recommendations to return
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of active recommendations
        """
        recommendations = [
            rec for rec in self.active_recommendations.values()
            if rec.confidence >= min_confidence
        ]
        
        # Sort by priority and confidence
        recommendations.sort(
            key=lambda r: (r.priority.value, r.confidence), 
            reverse=True
        )
        
        return recommendations[:limit]
    
    def get_recommendation_by_id(self, recommendation_id: str) -> Optional[OptimizationRecommendation]:
        """Get recommendation by ID."""
        return self.active_recommendations.get(recommendation_id)
    
    def dismiss_recommendation(self, recommendation_id: str, reason: str = "") -> bool:
        """Dismiss a recommendation.
        
        Args:
            recommendation_id: ID of recommendation to dismiss
            reason: Reason for dismissal
            
        Returns:
            True if successfully dismissed
        """
        if recommendation_id in self.active_recommendations:
            del self.active_recommendations[recommendation_id]
            logger.info(f"Dismissed recommendation {recommendation_id}: {reason}")
            return True
        
        return False
    
    def get_recommendation_stats(self) -> Dict[str, Any]:
        """Get recommendation engine statistics."""
        stats = {
            'total_recommendations_generated': len(self.recommendation_history),
            'active_recommendations': len(self.active_recommendations),
            'generation_cycles': self.generation_count,
            'bottleneck_triggers': dict(self.bottleneck_triggers),
            'recommendation_types': defaultdict(int)
        }
        
        # Analyze recommendation types
        for rec in self.recommendation_history:
            stats['recommendation_types'][rec.recommendation_type.value] += 1
        
        stats['recommendation_types'] = dict(stats['recommendation_types'])
        
        # Add learning stats if available
        if self.learning_engine:
            stats['learning_stats'] = self.learning_engine.get_learning_stats()
        
        return stats
    
    def export_recommendations_report(self, output_path: str) -> None:
        """Export comprehensive recommendations report."""
        report = {
            'engine_stats': self.get_recommendation_stats(),
            'active_recommendations': [
                rec.to_dict() for rec in self.active_recommendations.values()
            ],
            'recent_recommendations': [
                rec.to_dict() for rec in self.recommendation_history[-20:]
            ],
            'export_timestamp': datetime.now().isoformat()
        }
        
        if self.learning_engine:
            report['feedback_history'] = [
                feedback.to_dict() for feedback in self.learning_engine.feedback_history[-50:]
            ]
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Recommendations report exported to {output_path}")


# Global recommendation engine instance
_global_recommendation_engine: Optional[AutomatedRecommendationEngine] = None


def get_recommendation_engine(enable_learning: bool = True) -> AutomatedRecommendationEngine:
    """Get global recommendation engine instance."""
    global _global_recommendation_engine
    if _global_recommendation_engine is None:
        _global_recommendation_engine = AutomatedRecommendationEngine(enable_learning)
    return _global_recommendation_engine


def generate_optimization_recommendations(
    metrics: List[PerformanceMetric],
    bottlenecks: List[BottleneckDetection],
    **kwargs
) -> List[OptimizationRecommendation]:
    """Convenience function to generate recommendations."""
    engine = get_recommendation_engine()
    return engine.generate_recommendations(metrics, bottlenecks, **kwargs)


def submit_recommendation_feedback(
    recommendation_id: str,
    implemented: bool,
    implementation_time_minutes: int,
    actual_improvement: Dict[str, float],
    user_satisfaction: float,
    notes: str = ""
) -> None:
    """Convenience function to submit feedback."""
    engine = get_recommendation_engine()
    
    feedback = RecommendationFeedback(
        recommendation_id=recommendation_id,
        implemented=implemented,
        implementation_time_minutes=implementation_time_minutes,
        actual_improvement=actual_improvement,
        user_satisfaction=user_satisfaction,
        notes=notes
    )
    
    engine.submit_feedback(recommendation_id, feedback)
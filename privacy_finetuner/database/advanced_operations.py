"""Advanced database operations for privacy-preserving machine learning."""

import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_, case, select, update, delete
from sqlalchemy.sql import text
import numpy as np
import pandas as pd
from uuid import UUID

from .models import TrainingJob, PrivacyBudgetEntry, Dataset, Model, User, AuditLog
from .repositories import (
    TrainingJobRepository, PrivacyBudgetRepository, 
    DatasetRepository, ModelRepository, AuditLogRepository
)
from .query_optimizer import QueryOptimizer, PrivacyQueryMixin

logger = logging.getLogger(__name__)


@dataclass
class PrivacyBudgetAnalysis:
    """Analysis of privacy budget consumption patterns."""
    user_id: str
    total_epsilon_spent: float
    budget_utilization_percent: float
    average_epsilon_per_job: float
    jobs_in_period: int
    projected_depletion_date: Optional[datetime]
    risk_level: str
    recommendations: List[str]


@dataclass
class ModelPerformanceMetrics:
    """Comprehensive model performance analysis."""
    model_id: str
    privacy_utility_ratio: float
    accuracy_degradation_percent: float
    training_efficiency_score: float
    compliance_score: float
    deployment_readiness: bool
    benchmark_comparison: Dict[str, float]


class AdvancedPrivacyOperations:
    """Advanced operations for privacy budget management and analysis."""
    
    def __init__(self, session: Session, query_optimizer: QueryOptimizer):
        """Initialize advanced privacy operations.
        
        Args:
            session: Database session
            query_optimizer: Query optimization engine
        """
        self.session = session
        self.optimizer = query_optimizer
        self.privacy_repo = PrivacyBudgetRepository(session)
        self.training_repo = TrainingJobRepository(session)
        self.model_repo = ModelRepository(session)
        self.audit_repo = AuditLogRepository(session)
    
    def analyze_privacy_budget_patterns(
        self, 
        user_id: UUID,
        analysis_period_days: int = 30
    ) -> PrivacyBudgetAnalysis:
        """Comprehensive privacy budget pattern analysis.
        
        Args:
            user_id: User ID to analyze
            analysis_period_days: Period for analysis in days
            
        Returns:
            Detailed privacy budget analysis
        """
        cutoff_date = datetime.utcnow() - timedelta(days=analysis_period_days)
        
        # Get budget entries for the period
        entries = self.session.query(PrivacyBudgetEntry).filter(
            and_(
                PrivacyBudgetEntry.user_id == user_id,
                PrivacyBudgetEntry.created_at >= cutoff_date
            )
        ).all()
        
        if not entries:
            return PrivacyBudgetAnalysis(
                user_id=str(user_id),
                total_epsilon_spent=0.0,
                budget_utilization_percent=0.0,
                average_epsilon_per_job=0.0,
                jobs_in_period=0,
                projected_depletion_date=None,
                risk_level="low",
                recommendations=["No privacy budget usage in analysis period"]
            )
        
        # Calculate basic metrics
        total_epsilon = sum(entry.epsilon_spent for entry in entries)
        unique_jobs = len(set(entry.training_job_id for entry in entries if entry.training_job_id))
        avg_epsilon_per_job = total_epsilon / unique_jobs if unique_jobs > 0 else 0
        
        # Estimate daily consumption rate
        daily_rate = total_epsilon / analysis_period_days
        
        # Assume a typical privacy budget of 10.0 epsilon for projection
        # In practice, this would come from user settings or organization policy
        assumed_total_budget = 10.0
        remaining_budget = max(0, assumed_total_budget - total_epsilon)
        
        # Project depletion date
        projected_depletion = None
        if daily_rate > 0 and remaining_budget > 0:
            days_remaining = remaining_budget / daily_rate
            projected_depletion = datetime.utcnow() + timedelta(days=days_remaining)
        
        # Determine risk level
        utilization_percent = (total_epsilon / assumed_total_budget) * 100
        if utilization_percent >= 90:
            risk_level = "critical"
        elif utilization_percent >= 70:
            risk_level = "high"
        elif utilization_percent >= 40:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        # Generate recommendations
        recommendations = self._generate_budget_recommendations(
            utilization_percent, daily_rate, unique_jobs, avg_epsilon_per_job
        )
        
        return PrivacyBudgetAnalysis(
            user_id=str(user_id),
            total_epsilon_spent=total_epsilon,
            budget_utilization_percent=utilization_percent,
            average_epsilon_per_job=avg_epsilon_per_job,
            jobs_in_period=unique_jobs,
            projected_depletion_date=projected_depletion,
            risk_level=risk_level,
            recommendations=recommendations
        )
    
    def optimize_privacy_allocation(
        self,
        user_id: UUID,
        planned_jobs: List[Dict[str, Any]],
        total_budget: float = 10.0
    ) -> Dict[str, Any]:
        """Optimize privacy budget allocation across planned training jobs.
        
        Args:
            user_id: User ID
            planned_jobs: List of planned training job configurations
            total_budget: Total available privacy budget
            
        Returns:
            Optimized allocation recommendations
        """
        # Get current budget usage
        current_usage = self.privacy_repo.get_user_budget_summary(user_id)
        remaining_budget = total_budget - current_usage['total_epsilon_spent']
        
        if remaining_budget <= 0:
            return {
                "status": "budget_exhausted",
                "message": "No remaining privacy budget available",
                "current_usage": current_usage
            }
        
        # Estimate epsilon cost for each planned job
        job_estimates = []
        for job in planned_jobs:
            estimated_cost = self._estimate_job_privacy_cost(job)
            job_estimates.append({
                "job_config": job,
                "estimated_epsilon": estimated_cost,
                "priority": job.get("priority", "medium")
            })
        
        # Sort by priority and efficiency
        priority_weights = {"high": 3, "medium": 2, "low": 1}
        job_estimates.sort(
            key=lambda x: (
                priority_weights.get(x["priority"], 2),
                -x["estimated_epsilon"]  # Prefer lower epsilon cost
            ),
            reverse=True
        )
        
        # Allocate budget optimally
        allocated_jobs = []
        remaining = remaining_budget
        
        for job_est in job_estimates:
            if job_est["estimated_epsilon"] <= remaining:
                allocated_jobs.append({
                    **job_est,
                    "allocated": True,
                    "recommended_epsilon": job_est["estimated_epsilon"]
                })
                remaining -= job_est["estimated_epsilon"]
            else:
                # Try to fit with reduced epsilon
                min_epsilon = job_est["estimated_epsilon"] * 0.5  # Minimum 50% of estimated
                if min_epsilon <= remaining:
                    allocated_jobs.append({
                        **job_est,
                        "allocated": True,
                        "recommended_epsilon": remaining,
                        "note": "Reduced epsilon allocation"
                    })
                    remaining = 0
                else:
                    allocated_jobs.append({
                        **job_est,
                        "allocated": False,
                        "reason": "Insufficient budget"
                    })
        
        return {
            "status": "optimization_complete",
            "total_budget": total_budget,
            "remaining_budget": remaining_budget,
            "budget_after_allocation": remaining,
            "allocated_jobs": allocated_jobs,
            "allocation_efficiency": (remaining_budget - remaining) / remaining_budget * 100
        }
    
    def analyze_model_privacy_utility_tradeoff(self, model_id: UUID) -> ModelPerformanceMetrics:
        """Analyze privacy-utility tradeoff for a trained model.
        
        Args:
            model_id: Model ID to analyze
            
        Returns:
            Comprehensive model performance analysis
        """
        model = self.model_repo.get_by_id(model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found")
        
        # Get training job details
        training_job = model.training_job
        if not training_job:
            raise ValueError(f"No training job found for model {model_id}")
        
        # Calculate privacy-utility ratio
        # Higher is better (more utility per epsilon spent)
        accuracy = model.eval_accuracy or 0.0
        epsilon_spent = model.epsilon_spent or 1.0  # Avoid division by zero
        privacy_utility_ratio = accuracy / epsilon_spent
        
        # Estimate accuracy degradation compared to non-private baseline
        # This would ideally come from benchmark data
        baseline_accuracy = self._estimate_baseline_accuracy(model.base_model, training_job.dataset)
        accuracy_degradation = ((baseline_accuracy - accuracy) / baseline_accuracy) * 100 if baseline_accuracy > 0 else 0
        
        # Calculate training efficiency score
        training_time = training_job.training_time_seconds or 0
        gpu_hours = training_job.gpu_hours_used or 0
        efficiency_score = accuracy / (training_time + gpu_hours + 1)  # Avoid division by zero
        
        # Compliance score based on privacy parameters
        compliance_score = self._calculate_compliance_score(model)
        
        # Deployment readiness check
        deployment_ready = (
            model.eval_accuracy and model.eval_accuracy > 0.7 and
            model.epsilon_spent <= 3.0 and  # Reasonable privacy budget
            model.model_path and
            compliance_score >= 0.8
        )
        
        # Benchmark comparison (mock data - would be real benchmarks in production)
        benchmark_comparison = {
            "privacy_utility_ratio_percentile": min(95, max(5, privacy_utility_ratio * 20)),
            "accuracy_percentile": min(95, max(5, accuracy * 100)),
            "efficiency_percentile": min(95, max(5, efficiency_score * 1000))
        }
        
        return ModelPerformanceMetrics(
            model_id=str(model_id),
            privacy_utility_ratio=privacy_utility_ratio,
            accuracy_degradation_percent=accuracy_degradation,
            training_efficiency_score=efficiency_score,
            compliance_score=compliance_score,
            deployment_readiness=deployment_ready,
            benchmark_comparison=benchmark_comparison
        )
    
    def detect_privacy_anomalies(
        self, 
        lookback_hours: int = 24
    ) -> List[Dict[str, Any]]:
        """Detect anomalous privacy budget consumption patterns.
        
        Args:
            lookback_hours: Hours to look back for anomaly detection
            
        Returns:
            List of detected anomalies
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=lookback_hours)
        
        # Get recent privacy budget entries
        recent_entries = self.session.query(PrivacyBudgetEntry).filter(
            PrivacyBudgetEntry.created_at >= cutoff_time
        ).all()
        
        if not recent_entries:
            return []
        
        anomalies = []
        
        # Group by user for per-user analysis
        user_entries = {}
        for entry in recent_entries:
            user_id = str(entry.user_id)
            if user_id not in user_entries:
                user_entries[user_id] = []
            user_entries[user_id].append(entry)
        
        for user_id, entries in user_entries.items():
            epsilon_values = [entry.epsilon_spent for entry in entries]
            
            if len(epsilon_values) < 3:  # Need minimum data for anomaly detection
                continue
            
            # Statistical anomaly detection
            mean_epsilon = np.mean(epsilon_values)
            std_epsilon = np.std(epsilon_values)
            threshold = mean_epsilon + 2 * std_epsilon  # 2-sigma threshold
            
            # Check for unusual consumption patterns
            for entry in entries:
                if entry.epsilon_spent > threshold:
                    anomalies.append({
                        "type": "high_epsilon_consumption",
                        "user_id": user_id,
                        "training_job_id": str(entry.training_job_id) if entry.training_job_id else None,
                        "epsilon_spent": entry.epsilon_spent,
                        "threshold": threshold,
                        "timestamp": entry.created_at,
                        "severity": "high" if entry.epsilon_spent > threshold * 1.5 else "medium"
                    })
            
            # Check for rapid consecutive consumption
            sorted_entries = sorted(entries, key=lambda x: x.created_at)
            for i in range(1, len(sorted_entries)):
                time_diff = (sorted_entries[i].created_at - sorted_entries[i-1].created_at).total_seconds()
                if time_diff < 300:  # Less than 5 minutes apart
                    combined_epsilon = sorted_entries[i].epsilon_spent + sorted_entries[i-1].epsilon_spent
                    if combined_epsilon > mean_epsilon * 1.5:
                        anomalies.append({
                            "type": "rapid_consumption",
                            "user_id": user_id,
                            "time_window_seconds": time_diff,
                            "combined_epsilon": combined_epsilon,
                            "timestamp": sorted_entries[i].created_at,
                            "severity": "medium"
                        })
        
        return anomalies
    
    def generate_privacy_compliance_report(
        self,
        user_id: Optional[UUID] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive privacy compliance report.
        
        Args:
            user_id: Optional user ID filter
            start_date: Report start date
            end_date: Report end date
            
        Returns:
            Compliance report with audit trail
        """
        # Set default date range
        if not end_date:
            end_date = datetime.utcnow()
        if not start_date:
            start_date = end_date - timedelta(days=30)
        
        # Build base query for training jobs
        query = self.session.query(TrainingJob).filter(
            and_(
                TrainingJob.created_at >= start_date,
                TrainingJob.created_at <= end_date
            )
        )
        
        if user_id:
            query = query.filter(TrainingJob.user_id == user_id)
        
        jobs = query.all()
        
        # Aggregate compliance metrics
        total_jobs = len(jobs)
        compliant_jobs = sum(1 for job in jobs if job.epsilon_spent <= job.target_epsilon)
        
        privacy_violations = []
        for job in jobs:
            if job.epsilon_spent > job.target_epsilon:
                privacy_violations.append({
                    "job_id": str(job.id),
                    "job_name": job.job_name,
                    "target_epsilon": job.target_epsilon,
                    "actual_epsilon": job.epsilon_spent,
                    "violation_amount": job.epsilon_spent - job.target_epsilon,
                    "timestamp": job.completed_at or job.created_at
                })
        
        # Privacy budget distribution
        epsilon_distribution = {
            "0.0-1.0": sum(1 for job in jobs if 0 <= job.epsilon_spent <= 1.0),
            "1.0-3.0": sum(1 for job in jobs if 1.0 < job.epsilon_spent <= 3.0),
            "3.0-5.0": sum(1 for job in jobs if 3.0 < job.epsilon_spent <= 5.0),
            "5.0+": sum(1 for job in jobs if job.epsilon_spent > 5.0)
        }
        
        # Compliance score calculation
        compliance_score = (compliant_jobs / total_jobs * 100) if total_jobs > 0 else 100
        
        # Get audit trail
        audit_entries = self.audit_repo.get_privacy_events(
            days=(end_date - start_date).days
        )
        
        return {
            "report_period": {
                "start_date": start_date,
                "end_date": end_date,
                "duration_days": (end_date - start_date).days
            },
            "compliance_summary": {
                "total_training_jobs": total_jobs,
                "compliant_jobs": compliant_jobs,
                "compliance_rate_percent": compliance_score,
                "privacy_violations": len(privacy_violations)
            },
            "privacy_budget_analysis": {
                "epsilon_distribution": epsilon_distribution,
                "total_epsilon_consumed": sum(job.epsilon_spent for job in jobs),
                "average_epsilon_per_job": sum(job.epsilon_spent for job in jobs) / total_jobs if total_jobs > 0 else 0
            },
            "violations": privacy_violations,
            "audit_trail_entries": len(audit_entries),
            "recommendations": self._generate_compliance_recommendations(compliance_score, privacy_violations)
        }
    
    def _estimate_job_privacy_cost(self, job_config: Dict[str, Any]) -> float:
        """Estimate privacy cost for a training job configuration."""
        # Simplified estimation based on typical parameters
        epochs = job_config.get("epochs", 3)
        batch_size = job_config.get("batch_size", 8)
        learning_rate = job_config.get("learning_rate", 5e-5)
        dataset_size = job_config.get("dataset_size", 1000)
        
        # Basic estimation formula (this would be more sophisticated in practice)
        base_cost = 0.5
        epoch_factor = epochs * 0.2
        batch_factor = (32 / batch_size) * 0.1  # Smaller batches = higher cost
        lr_factor = learning_rate * 10000 * 0.05
        size_factor = (dataset_size / 1000) * 0.1
        
        estimated_cost = base_cost + epoch_factor + batch_factor + lr_factor + size_factor
        return max(0.1, min(10.0, estimated_cost))  # Clamp between 0.1 and 10.0
    
    def _estimate_baseline_accuracy(self, base_model: str, dataset: Dataset) -> float:
        """Estimate baseline accuracy for non-private training."""
        # This would typically come from benchmark databases
        # Using mock values for demonstration
        baseline_accuracies = {
            "meta-llama/Llama-2-7b-hf": 0.85,
            "meta-llama/Llama-2-13b-hf": 0.88,
            "mistralai/Mistral-7B-v0.1": 0.86,
        }
        return baseline_accuracies.get(base_model, 0.80)  # Default to 80%
    
    def _calculate_compliance_score(self, model: Model) -> float:
        """Calculate compliance score based on model properties."""
        score = 1.0
        
        # Penalize high epsilon usage
        if model.epsilon_spent > 5.0:
            score *= 0.7
        elif model.epsilon_spent > 3.0:
            score *= 0.85
        
        # Reward good delta values
        if model.delta_value <= 1e-5:
            score *= 1.1
        
        # Penalize missing privacy parameters
        if not model.noise_multiplier or not model.max_grad_norm:
            score *= 0.8
        
        return min(1.0, max(0.0, score))
    
    def _generate_budget_recommendations(
        self,
        utilization_percent: float,
        daily_rate: float,
        jobs_count: int,
        avg_epsilon: float
    ) -> List[str]:
        """Generate privacy budget recommendations."""
        recommendations = []
        
        if utilization_percent >= 90:
            recommendations.extend([
                "CRITICAL: Privacy budget nearly exhausted",
                "Consider increasing noise multiplier to reduce epsilon consumption",
                "Evaluate if all training jobs are necessary"
            ])
        elif utilization_percent >= 70:
            recommendations.extend([
                "HIGH: Privacy budget usage is high",
                "Monitor consumption closely",
                "Consider batch training to optimize budget usage"
            ])
        
        if avg_epsilon > 2.0:
            recommendations.append(
                f"Average epsilon per job ({avg_epsilon:.2f}) is high - consider parameter tuning"
            )
        
        if daily_rate > 0.5:
            recommendations.append(
                f"Daily consumption rate ({daily_rate:.2f}) is high - consider spacing out training jobs"
            )
        
        if jobs_count > 10:
            recommendations.append(
                "High number of training jobs - consider consolidating datasets or using transfer learning"
            )
        
        return recommendations
    
    def _generate_compliance_recommendations(
        self,
        compliance_score: float,
        violations: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate compliance recommendations."""
        recommendations = []
        
        if compliance_score < 80:
            recommendations.append("CRITICAL: Compliance rate below acceptable threshold (80%)")
        
        if violations:
            recommendations.extend([
                f"Address {len(violations)} privacy budget violations",
                "Review training job configurations to stay within epsilon targets",
                "Implement automated budget checking before job execution"
            ])
        
        if compliance_score < 95:
            recommendations.extend([
                "Implement stricter privacy budget controls",
                "Provide additional training on privacy-preserving ML practices",
                "Consider automated compliance monitoring"
            ])
        
        return recommendations
"""Comprehensive benchmarking suite for privacy-preserving ML research.

This module provides standardized benchmarks for evaluating privacy algorithms,
measuring privacy-utility tradeoffs, and comparing different approaches.
"""

import time
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import statistics
import hashlib

# Handle numpy import gracefully
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    # Create basic numpy-like functionality for benchmarking
    class NumpyStub:
        @staticmethod
        def sqrt(x):
            import math
            return math.sqrt(x)
        
        @staticmethod
        def mean(data):
            return sum(data) / len(data) if data else 0
    
    np = NumpyStub()

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    algorithm_name: str
    dataset_name: str
    privacy_budget: float
    accuracy: float
    privacy_leakage: float
    training_time: float
    memory_usage: float
    convergence_steps: int
    hyperparameters: Dict[str, Any]
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass 
class BenchmarkConfig:
    """Configuration for benchmark experiments."""
    datasets: List[str]
    privacy_budgets: List[float]
    algorithms: List[str]
    num_runs: int = 3
    max_epochs: int = 10
    batch_sizes: List[int] = None
    learning_rates: List[float] = None
    
    def __post_init__(self):
        if self.batch_sizes is None:
            self.batch_sizes = [8, 16, 32]
        if self.learning_rates is None:
            self.learning_rates = [1e-5, 5e-5, 1e-4]


class PrivacyBenchmarkSuite:
    """Comprehensive benchmarking suite for privacy-preserving algorithms."""
    
    def __init__(
        self, 
        output_dir: str = "benchmark_results",
        baseline_algorithms: Optional[List[str]] = None
    ):
        """Initialize benchmark suite.
        
        Args:
            output_dir: Directory to save results
            baseline_algorithms: List of baseline algorithms for comparison
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.baseline_algorithms = baseline_algorithms or [
            "standard_sgd", "dp_sgd", "fedavg", "local_dp"
        ]
        self.results: List[BenchmarkResult] = []
        
        logger.info(f"Initialized PrivacyBenchmarkSuite with output dir: {output_dir}")
    
    def run_comprehensive_benchmark(
        self, 
        config: BenchmarkConfig,
        custom_algorithms: Optional[Dict[str, Callable]] = None
    ) -> Dict[str, List[BenchmarkResult]]:
        """Run comprehensive benchmark across all configurations.
        
        Args:
            config: Benchmark configuration
            custom_algorithms: Dictionary of custom algorithm implementations
            
        Returns:
            Dictionary mapping algorithm names to their results
        """
        logger.info("Starting comprehensive privacy benchmark")
        logger.info(f"Config: {len(config.datasets)} datasets, {len(config.algorithms)} algorithms")
        
        all_results = {}
        custom_algorithms = custom_algorithms or {}
        
        for algorithm in config.algorithms:
            algorithm_results = []
            logger.info(f"Benchmarking algorithm: {algorithm}")
            
            for dataset in config.datasets:
                for privacy_budget in config.privacy_budgets:
                    for batch_size in config.batch_sizes:
                        for lr in config.learning_rates:
                            # Run multiple trials for statistical significance
                            trial_results = []
                            
                            for run in range(config.num_runs):
                                result = self._run_single_benchmark(
                                    algorithm=algorithm,
                                    dataset=dataset,
                                    privacy_budget=privacy_budget,
                                    batch_size=batch_size,
                                    learning_rate=lr,
                                    max_epochs=config.max_epochs,
                                    custom_implementation=custom_algorithms.get(algorithm),
                                    run_id=run
                                )
                                trial_results.append(result)
                            
                            # Aggregate trial results
                            aggregated = self._aggregate_trial_results(
                                trial_results, algorithm, dataset, privacy_budget
                            )
                            algorithm_results.append(aggregated)
            
            all_results[algorithm] = algorithm_results
        
        self.results.extend([r for results in all_results.values() for r in results])
        self._save_results()
        
        logger.info("Comprehensive benchmark completed")
        return all_results
    
    def _run_single_benchmark(
        self,
        algorithm: str,
        dataset: str, 
        privacy_budget: float,
        batch_size: int,
        learning_rate: float,
        max_epochs: int,
        custom_implementation: Optional[Callable] = None,
        run_id: int = 0
    ) -> BenchmarkResult:
        """Run a single benchmark trial."""
        import time
        import psutil
        import os
        
        start_time = time.time()
        process = psutil.Process(os.getpid())
        start_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        logger.debug(f"Running {algorithm} on {dataset} (ε={privacy_budget}, run={run_id})")
        
        # Simulate algorithm execution (replace with actual implementations)
        if custom_implementation:
            result_metrics = custom_implementation(
                dataset=dataset,
                privacy_budget=privacy_budget,
                batch_size=batch_size,
                learning_rate=learning_rate,
                max_epochs=max_epochs
            )
        else:
            result_metrics = self._simulate_algorithm_execution(
                algorithm, dataset, privacy_budget, batch_size, learning_rate, max_epochs
            )
        
        end_time = time.time()
        end_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        return BenchmarkResult(
            algorithm_name=algorithm,
            dataset_name=dataset,
            privacy_budget=privacy_budget,
            accuracy=result_metrics["accuracy"],
            privacy_leakage=result_metrics["privacy_leakage"],
            training_time=end_time - start_time,
            memory_usage=end_memory - start_memory,
            convergence_steps=result_metrics["convergence_steps"],
            hyperparameters={
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "max_epochs": max_epochs
            },
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
    
    def _simulate_algorithm_execution(
        self,
        algorithm: str,
        dataset: str,
        privacy_budget: float,
        batch_size: int,
        learning_rate: float,
        max_epochs: int
    ) -> Dict[str, Any]:
        """Simulate algorithm execution for benchmarking."""
        import random
        import time
        
        # Simulate computation time
        time.sleep(0.1 + random.random() * 0.2)
        
        # Algorithm-specific performance characteristics
        if algorithm == "standard_sgd":
            base_accuracy = 0.95
            privacy_leakage = 0.8  # High privacy leakage
        elif algorithm == "dp_sgd":
            base_accuracy = 0.90 * (1 - 1/privacy_budget)  # Lower accuracy with stronger privacy
            privacy_leakage = 1.0 / privacy_budget  # Lower leakage with stronger privacy
        elif algorithm == "fedavg":
            base_accuracy = 0.88
            privacy_leakage = 0.3
        elif algorithm == "local_dp":
            base_accuracy = 0.82 * (privacy_budget / 10)  # Very dependent on budget
            privacy_leakage = 0.1
        else:
            base_accuracy = 0.85
            privacy_leakage = 0.5
        
        # Add noise for realistic variation
        accuracy = max(0.0, min(1.0, base_accuracy + random.gauss(0, 0.05)))
        privacy_leakage = max(0.0, min(1.0, privacy_leakage + random.gauss(0, 0.1)))
        convergence_steps = random.randint(max_epochs // 2, max_epochs)
        
        return {
            "accuracy": accuracy,
            "privacy_leakage": privacy_leakage,
            "convergence_steps": convergence_steps
        }
    
    def _aggregate_trial_results(
        self,
        trial_results: List[BenchmarkResult],
        algorithm: str,
        dataset: str,
        privacy_budget: float
    ) -> BenchmarkResult:
        """Aggregate results from multiple trial runs."""
        if not trial_results:
            raise ValueError("No trial results to aggregate")
        
        # Calculate statistics across trials
        accuracies = [r.accuracy for r in trial_results]
        privacy_leakages = [r.privacy_leakage for r in trial_results]
        training_times = [r.training_time for r in trial_results]
        memory_usages = [r.memory_usage for r in trial_results]
        convergence_steps = [r.convergence_steps for r in trial_results]
        
        return BenchmarkResult(
            algorithm_name=algorithm,
            dataset_name=dataset,
            privacy_budget=privacy_budget,
            accuracy=statistics.mean(accuracies),
            privacy_leakage=statistics.mean(privacy_leakages),
            training_time=statistics.mean(training_times),
            memory_usage=statistics.mean(memory_usages),
            convergence_steps=int(statistics.mean(convergence_steps)),
            hyperparameters={
                "accuracy_std": statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0,
                "trials": len(trial_results),
                "confidence_95": 1.96 * (statistics.stdev(accuracies) / np.sqrt(len(accuracies))) if len(accuracies) > 1 else 0.0
            },
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
    
    def generate_comparative_report(self, output_path: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive comparative analysis report."""
        if not self.results:
            logger.warning("No benchmark results available for report generation")
            return {}
        
        logger.info("Generating comparative analysis report")
        
        # Group results by algorithm
        algorithm_results = {}
        for result in self.results:
            if result.algorithm_name not in algorithm_results:
                algorithm_results[result.algorithm_name] = []
            algorithm_results[result.algorithm_name].append(result)
        
        # Calculate comparative metrics
        report = {
            "summary": self._generate_summary_statistics(algorithm_results),
            "privacy_utility_analysis": self._analyze_privacy_utility_tradeoff(algorithm_results),
            "performance_comparison": self._compare_algorithm_performance(algorithm_results),
            "statistical_significance": self._test_statistical_significance(algorithm_results),
            "recommendations": self._generate_recommendations(algorithm_results)
        }
        
        # Save report
        output_path = output_path or self.output_dir / "comparative_report.json"
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Comparative report saved to: {output_path}")
        return report
    
    def _generate_summary_statistics(self, algorithm_results: Dict[str, List[BenchmarkResult]]) -> Dict[str, Any]:
        """Generate summary statistics for all algorithms."""
        summary = {}
        
        for algorithm, results in algorithm_results.items():
            accuracies = [r.accuracy for r in results]
            privacy_leakages = [r.privacy_leakage for r in results]
            training_times = [r.training_time for r in results]
            
            summary[algorithm] = {
                "total_runs": len(results),
                "accuracy": {
                    "mean": statistics.mean(accuracies),
                    "std": statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0,
                    "min": min(accuracies),
                    "max": max(accuracies)
                },
                "privacy_leakage": {
                    "mean": statistics.mean(privacy_leakages),
                    "std": statistics.stdev(privacy_leakages) if len(privacy_leakages) > 1 else 0.0
                },
                "training_time": {
                    "mean": statistics.mean(training_times),
                    "std": statistics.stdev(training_times) if len(training_times) > 1 else 0.0
                }
            }
        
        return summary
    
    def _analyze_privacy_utility_tradeoff(self, algorithm_results: Dict[str, List[BenchmarkResult]]) -> Dict[str, Any]:
        """Analyze privacy-utility tradeoffs across algorithms."""
        tradeoff_analysis = {}
        
        for algorithm, results in algorithm_results.items():
            # Group by privacy budget for tradeoff analysis
            budget_groups = {}
            for result in results:
                budget = result.privacy_budget
                if budget not in budget_groups:
                    budget_groups[budget] = []
                budget_groups[budget].append(result)
            
            # Calculate tradeoff metrics
            tradeoff_points = []
            for budget, budget_results in budget_groups.items():
                avg_accuracy = statistics.mean([r.accuracy for r in budget_results])
                avg_privacy_leakage = statistics.mean([r.privacy_leakage for r in budget_results])
                
                tradeoff_points.append({
                    "privacy_budget": budget,
                    "accuracy": avg_accuracy,
                    "privacy_leakage": avg_privacy_leakage,
                    "utility_privacy_ratio": avg_accuracy / max(avg_privacy_leakage, 0.001)
                })
            
            tradeoff_analysis[algorithm] = {
                "tradeoff_points": sorted(tradeoff_points, key=lambda x: x["privacy_budget"]),
                "pareto_optimal": self._find_pareto_optimal_points(tradeoff_points)
            }
        
        return tradeoff_analysis
    
    def _find_pareto_optimal_points(self, points: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """Find Pareto optimal points in privacy-utility tradeoff."""
        pareto_points = []
        
        for i, point1 in enumerate(points):
            is_dominated = False
            for j, point2 in enumerate(points):
                if i != j:
                    # Point1 is dominated if point2 has better accuracy AND better privacy
                    if (point2["accuracy"] >= point1["accuracy"] and 
                        point2["privacy_leakage"] <= point1["privacy_leakage"] and
                        (point2["accuracy"] > point1["accuracy"] or point2["privacy_leakage"] < point1["privacy_leakage"])):
                        is_dominated = True
                        break
            
            if not is_dominated:
                pareto_points.append(point1)
        
        return pareto_points
    
    def _compare_algorithm_performance(self, algorithm_results: Dict[str, List[BenchmarkResult]]) -> Dict[str, Any]:
        """Compare overall algorithm performance."""
        performance_comparison = {}
        
        algorithms = list(algorithm_results.keys())
        if len(algorithms) < 2:
            return {"error": "Need at least 2 algorithms for comparison"}
        
        # Rank algorithms by different metrics
        accuracy_rankings = []
        privacy_rankings = []
        efficiency_rankings = []
        
        for algorithm, results in algorithm_results.items():
            avg_accuracy = statistics.mean([r.accuracy for r in results])
            avg_privacy_leakage = statistics.mean([r.privacy_leakage for r in results])
            avg_training_time = statistics.mean([r.training_time for r in results])
            
            accuracy_rankings.append((algorithm, avg_accuracy))
            privacy_rankings.append((algorithm, 1.0 - avg_privacy_leakage))  # Higher is better
            efficiency_rankings.append((algorithm, 1.0 / max(avg_training_time, 0.001)))  # Higher is better
        
        performance_comparison = {
            "accuracy_ranking": sorted(accuracy_rankings, key=lambda x: x[1], reverse=True),
            "privacy_ranking": sorted(privacy_rankings, key=lambda x: x[1], reverse=True),
            "efficiency_ranking": sorted(efficiency_rankings, key=lambda x: x[1], reverse=True),
            "overall_score": self._calculate_overall_scores(algorithm_results)
        }
        
        return performance_comparison
    
    def _calculate_overall_scores(self, algorithm_results: Dict[str, List[BenchmarkResult]]) -> Dict[str, float]:
        """Calculate weighted overall scores for algorithms."""
        scores = {}
        
        # Weights for different criteria
        weights = {"accuracy": 0.4, "privacy": 0.4, "efficiency": 0.2}
        
        for algorithm, results in algorithm_results.items():
            avg_accuracy = statistics.mean([r.accuracy for r in results])
            avg_privacy_leakage = statistics.mean([r.privacy_leakage for r in results])
            avg_training_time = statistics.mean([r.training_time for r in results])
            
            # Normalize metrics (0-1 scale)
            accuracy_score = avg_accuracy
            privacy_score = 1.0 - avg_privacy_leakage
            efficiency_score = min(1.0, 10.0 / max(avg_training_time, 0.1))
            
            overall_score = (
                weights["accuracy"] * accuracy_score +
                weights["privacy"] * privacy_score +
                weights["efficiency"] * efficiency_score
            )
            
            scores[algorithm] = overall_score
        
        return scores
    
    def _test_statistical_significance(self, algorithm_results: Dict[str, List[BenchmarkResult]]) -> Dict[str, Any]:
        """Test statistical significance of differences between algorithms."""
        significance_tests = {}
        
        algorithms = list(algorithm_results.keys())
        
        for i, alg1 in enumerate(algorithms):
            for j, alg2 in enumerate(algorithms[i+1:], i+1):
                acc1 = [r.accuracy for r in algorithm_results[alg1]]
                acc2 = [r.accuracy for r in algorithm_results[alg2]]
                
                if len(acc1) > 1 and len(acc2) > 1:
                    # Perform t-test (simplified version)
                    mean1, mean2 = statistics.mean(acc1), statistics.mean(acc2)
                    std1, std2 = statistics.stdev(acc1), statistics.stdev(acc2)
                    n1, n2 = len(acc1), len(acc2)
                    
                    # Simplified t-statistic
                    pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2))
                    t_stat = (mean1 - mean2) / (pooled_std * np.sqrt(1/n1 + 1/n2))
                    
                    significance_tests[f"{alg1}_vs_{alg2}"] = {
                        "mean_difference": mean1 - mean2,
                        "t_statistic": t_stat,
                        "significant": abs(t_stat) > 2.0,  # Simplified significance test
                        "effect_size": abs(mean1 - mean2) / pooled_std if pooled_std > 0 else 0
                    }
        
        return significance_tests
    
    def _generate_recommendations(self, algorithm_results: Dict[str, List[BenchmarkResult]]) -> List[str]:
        """Generate recommendations based on benchmark results."""
        recommendations = []
        
        # Find best algorithm for accuracy
        best_accuracy = max(algorithm_results.keys(), 
                           key=lambda alg: statistics.mean([r.accuracy for r in algorithm_results[alg]]))
        
        # Find best algorithm for privacy
        best_privacy = min(algorithm_results.keys(),
                          key=lambda alg: statistics.mean([r.privacy_leakage for r in algorithm_results[alg]]))
        
        # Find most efficient algorithm
        best_efficiency = min(algorithm_results.keys(),
                             key=lambda alg: statistics.mean([r.training_time for r in algorithm_results[alg]]))
        
        recommendations.extend([
            f"For highest accuracy: Use {best_accuracy}",
            f"For strongest privacy protection: Use {best_privacy}",
            f"For fastest training: Use {best_efficiency}",
            "Consider privacy-utility tradeoffs based on your specific requirements",
            "Run additional experiments with domain-specific datasets for better insights"
        ])
        
        return recommendations
    
    def _save_results(self) -> None:
        """Save benchmark results to file."""
        results_path = self.output_dir / "benchmark_results.json"
        
        serializable_results = [result.to_dict() for result in self.results]
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        logger.info(f"Saved {len(self.results)} benchmark results to {results_path}")
    
    def load_results(self, results_path: str) -> None:
        """Load benchmark results from file."""
        with open(results_path, 'r') as f:
            results_data = json.load(f)
        
        self.results = []
        for result_dict in results_data:
            self.results.append(BenchmarkResult(**result_dict))
        
        logger.info(f"Loaded {len(self.results)} benchmark results from {results_path}")
    
    def export_to_csv(self, output_path: Optional[str] = None) -> None:
        """Export results to CSV format for external analysis."""
        import csv
        
        output_path = output_path or self.output_dir / "benchmark_results.csv"
        
        if not self.results:
            logger.warning("No results to export")
            return
        
        fieldnames = list(self.results[0].to_dict().keys())
        
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for result in self.results:
                writer.writerow(result.to_dict())
        
        logger.info(f"Exported {len(self.results)} results to CSV: {output_path}")
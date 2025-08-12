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
    
    def conduct_statistical_significance_testing(
        self,
        algorithm_pairs: Optional[List[Tuple[str, str]]] = None,
        significance_level: float = 0.05,
        test_type: str = "paired_ttest"
    ) -> Dict[str, Any]:
        """Conduct statistical significance testing between algorithm pairs.
        
        Args:
            algorithm_pairs: Pairs of algorithms to compare (None = all pairs)
            significance_level: Statistical significance level (alpha)
            test_type: Type of statistical test ('paired_ttest', 'wilcoxon', 'mann_whitney')
            
        Returns:
            Statistical test results with p-values and effect sizes
        """
        if not self.results:
            return {"error": "No benchmark results available for statistical testing"}
        
        logger.info("Conducting statistical significance testing")
        
        # Group results by algorithm
        algorithm_results = {}
        for result in self.results:
            alg_name = result.algorithm_name
            if alg_name not in algorithm_results:
                algorithm_results[alg_name] = []
            algorithm_results[alg_name].append(result)
        
        # Generate algorithm pairs if not provided
        if algorithm_pairs is None:
            algorithms = list(algorithm_results.keys())
            algorithm_pairs = []
            for i, alg1 in enumerate(algorithms):
                for j, alg2 in enumerate(algorithms[i+1:], i+1):
                    algorithm_pairs.append((alg1, alg2))
        
        significance_results = {}
        
        for alg1, alg2 in algorithm_pairs:
            if alg1 not in algorithm_results or alg2 not in algorithm_results:
                continue
            
            results1 = algorithm_results[alg1]
            results2 = algorithm_results[alg2]
            
            # Extract metrics for comparison
            accuracy1 = [r.accuracy for r in results1]
            accuracy2 = [r.accuracy for r in results2]
            privacy_leakage1 = [r.privacy_leakage for r in results1]
            privacy_leakage2 = [r.privacy_leakage for r in results2]
            
            pair_key = f"{alg1}_vs_{alg2}"
            
            # Accuracy comparison
            accuracy_test = self._perform_statistical_test(
                accuracy1, accuracy2, test_type, significance_level
            )
            
            # Privacy leakage comparison
            privacy_test = self._perform_statistical_test(
                privacy_leakage1, privacy_leakage2, test_type, significance_level
            )
            
            significance_results[pair_key] = {
                "algorithm1": alg1,
                "algorithm2": alg2,
                "accuracy_test": accuracy_test,
                "privacy_leakage_test": privacy_test,
                "sample_sizes": (len(accuracy1), len(accuracy2)),
                "test_type": test_type,
                "significance_level": significance_level
            }
        
        # Overall summary
        significant_differences = sum(
            1 for result in significance_results.values()
            if (result["accuracy_test"]["significant"] or result["privacy_leakage_test"]["significant"])
        )
        
        summary = {
            "total_comparisons": len(significance_results),
            "significant_differences": significant_differences,
            "significance_rate": significant_differences / max(len(significance_results), 1),
            "test_results": significance_results,
            "recommendations": self._generate_statistical_recommendations(significance_results)
        }
        
        return summary
    
    def _perform_statistical_test(
        self,
        sample1: List[float],
        sample2: List[float], 
        test_type: str,
        alpha: float
    ) -> Dict[str, Any]:
        """Perform statistical test between two samples."""
        if len(sample1) == 0 or len(sample2) == 0:
            return {"error": "Empty sample(s)", "significant": False}
        
        if len(sample1) == 1 and len(sample2) == 1:
            # Single point comparison
            diff = abs(sample1[0] - sample2[0])
            return {
                "test_statistic": diff,
                "p_value": 1.0 if diff == 0 else 0.5,
                "significant": False,
                "effect_size": 0.0,
                "method": "single_point"
            }
        
        if test_type == "paired_ttest":
            return self._paired_ttest(sample1, sample2, alpha)
        elif test_type == "wilcoxon":
            return self._wilcoxon_test(sample1, sample2, alpha)
        elif test_type == "mann_whitney":
            return self._mann_whitney_test(sample1, sample2, alpha)
        else:
            # Default to independent t-test
            return self._independent_ttest(sample1, sample2, alpha)
    
    def _paired_ttest(self, sample1: List[float], sample2: List[float], alpha: float) -> Dict[str, Any]:
        """Paired t-test implementation."""
        # Ensure equal length samples for pairing
        min_len = min(len(sample1), len(sample2))
        s1 = sample1[:min_len]
        s2 = sample2[:min_len]
        
        if min_len < 2:
            return {"error": "Insufficient paired samples", "significant": False}
        
        # Calculate differences
        differences = [a - b for a, b in zip(s1, s2)]
        
        # Calculate test statistic
        mean_diff = statistics.mean(differences)
        
        if len(differences) == 1:
            return {
                "test_statistic": 0.0,
                "p_value": 1.0,
                "significant": False,
                "effect_size": 0.0,
                "mean_difference": mean_diff,
                "method": "paired_ttest"
            }
        
        std_diff = statistics.stdev(differences)
        
        if std_diff == 0:
            # No variance in differences
            t_stat = float('inf') if mean_diff != 0 else 0.0
            p_value = 0.0 if mean_diff != 0 else 1.0
        else:
            t_stat = mean_diff / (std_diff / math.sqrt(len(differences)))
            # Simplified p-value calculation (normally would use t-distribution)
            p_value = 2 * (1 - min(0.999, abs(t_stat) / 3.0))  # Rough approximation
        
        # Effect size (Cohen's d for paired samples)
        effect_size = abs(mean_diff) / max(std_diff, 1e-10)
        
        return {
            "test_statistic": t_stat,
            "p_value": max(0.0, min(1.0, p_value)),
            "significant": p_value < alpha,
            "effect_size": effect_size,
            "mean_difference": mean_diff,
            "degrees_of_freedom": len(differences) - 1,
            "method": "paired_ttest"
        }
    
    def _independent_ttest(self, sample1: List[float], sample2: List[float], alpha: float) -> Dict[str, Any]:
        """Independent samples t-test implementation."""
        if len(sample1) < 2 or len(sample2) < 2:
            return {"error": "Insufficient samples for t-test", "significant": False}
        
        mean1, mean2 = statistics.mean(sample1), statistics.mean(sample2)
        var1, var2 = statistics.variance(sample1), statistics.variance(sample2)
        n1, n2 = len(sample1), len(sample2)
        
        # Pooled standard error
        pooled_se = math.sqrt(var1/n1 + var2/n2)
        
        if pooled_se == 0:
            t_stat = float('inf') if mean1 != mean2 else 0.0
            p_value = 0.0 if mean1 != mean2 else 1.0
        else:
            t_stat = (mean1 - mean2) / pooled_se
            # Simplified p-value approximation
            p_value = 2 * (1 - min(0.999, abs(t_stat) / 3.0))
        
        # Effect size (Cohen's d)
        pooled_std = math.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
        effect_size = abs(mean1 - mean2) / max(pooled_std, 1e-10)
        
        return {
            "test_statistic": t_stat,
            "p_value": max(0.0, min(1.0, p_value)),
            "significant": p_value < alpha,
            "effect_size": effect_size,
            "mean_difference": mean1 - mean2,
            "degrees_of_freedom": n1 + n2 - 2,
            "method": "independent_ttest"
        }
    
    def _wilcoxon_test(self, sample1: List[float], sample2: List[float], alpha: float) -> Dict[str, Any]:
        """Wilcoxon signed-rank test implementation (simplified)."""
        min_len = min(len(sample1), len(sample2))
        s1 = sample1[:min_len]
        s2 = sample2[:min_len]
        
        if min_len < 3:
            return {"error": "Insufficient samples for Wilcoxon test", "significant": False}
        
        # Calculate differences and their ranks
        differences = [(a - b) for a, b in zip(s1, s2)]
        non_zero_diffs = [d for d in differences if abs(d) > 1e-10]
        
        if not non_zero_diffs:
            return {
                "test_statistic": 0.0,
                "p_value": 1.0,
                "significant": False,
                "effect_size": 0.0,
                "method": "wilcoxon"
            }
        
        # Rank absolute differences
        abs_diffs = [abs(d) for d in non_zero_diffs]
        ranks = self._calculate_ranks(abs_diffs)
        
        # Sum of positive ranks
        W_plus = sum(rank for diff, rank in zip(non_zero_diffs, ranks) if diff > 0)
        
        # Expected value and variance under null hypothesis
        n = len(non_zero_diffs)
        expected_W = n * (n + 1) / 4
        var_W = n * (n + 1) * (2 * n + 1) / 24
        
        # Z-statistic (normal approximation)
        if var_W == 0:
            z_stat = 0.0
        else:
            z_stat = (W_plus - expected_W) / math.sqrt(var_W)
        
        # P-value approximation
        p_value = 2 * (1 - min(0.999, abs(z_stat) / 2.0))
        
        return {
            "test_statistic": W_plus,
            "z_statistic": z_stat,
            "p_value": max(0.0, min(1.0, p_value)),
            "significant": p_value < alpha,
            "effect_size": abs(z_stat) / math.sqrt(n),  # r = Z / sqrt(N)
            "method": "wilcoxon"
        }
    
    def _mann_whitney_test(self, sample1: List[float], sample2: List[float], alpha: float) -> Dict[str, Any]:
        """Mann-Whitney U test implementation (simplified)."""
        n1, n2 = len(sample1), len(sample2)
        
        if n1 < 3 or n2 < 3:
            return {"error": "Insufficient samples for Mann-Whitney test", "significant": False}
        
        # Combine and rank all observations
        combined = [(val, 1) for val in sample1] + [(val, 2) for val in sample2]
        combined.sort(key=lambda x: x[0])
        
        values = [x[0] for x in combined]
        ranks = self._calculate_ranks(values)
        
        # Sum of ranks for sample 1
        R1 = sum(rank for (val, group), rank in zip(combined, ranks) if group == 1)
        
        # Mann-Whitney U statistic
        U1 = R1 - n1 * (n1 + 1) / 2
        U2 = n1 * n2 - U1
        U = min(U1, U2)
        
        # Expected value and variance
        expected_U = n1 * n2 / 2
        var_U = n1 * n2 * (n1 + n2 + 1) / 12
        
        # Z-statistic
        if var_U == 0:
            z_stat = 0.0
        else:
            z_stat = (U - expected_U) / math.sqrt(var_U)
        
        # P-value approximation
        p_value = 2 * (1 - min(0.999, abs(z_stat) / 2.0))
        
        return {
            "test_statistic": U,
            "z_statistic": z_stat,
            "p_value": max(0.0, min(1.0, p_value)),
            "significant": p_value < alpha,
            "effect_size": abs(z_stat) / math.sqrt(n1 + n2),
            "method": "mann_whitney"
        }
    
    def _calculate_ranks(self, values: List[float]) -> List[float]:
        """Calculate ranks for a list of values (handling ties with average ranks)."""
        # Sort indices by values
        sorted_indices = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        
        i = 0
        while i < len(sorted_indices):
            j = i
            # Find the end of tied values
            while j < len(sorted_indices) and values[sorted_indices[j]] == values[sorted_indices[i]]:
                j += 1
            
            # Average rank for tied values
            avg_rank = (i + j - 1) / 2 + 1  # +1 because ranks start at 1
            
            for k in range(i, j):
                ranks[sorted_indices[k]] = avg_rank
            
            i = j
        
        return ranks
    
    def _generate_statistical_recommendations(
        self,
        significance_results: Dict[str, Dict[str, Any]]
    ) -> List[str]:
        """Generate recommendations based on statistical test results."""
        recommendations = []
        
        if not significance_results:
            recommendations.append("No statistical comparisons performed")
            return recommendations
        
        significant_count = sum(
            1 for result in significance_results.values()
            if (result.get("accuracy_test", {}).get("significant", False) or
                result.get("privacy_leakage_test", {}).get("significant", False))
        )
        
        total_comparisons = len(significance_results)
        significance_rate = significant_count / total_comparisons if total_comparisons > 0 else 0
        
        if significance_rate > 0.8:
            recommendations.append("High rate of significant differences detected - algorithms show clear performance distinctions")
        elif significance_rate > 0.5:
            recommendations.append("Moderate rate of significant differences - some algorithms outperform others")
        elif significance_rate > 0.2:
            recommendations.append("Low rate of significant differences - most algorithms perform similarly")
        else:
            recommendations.append("Very low rate of significant differences - consider larger sample sizes or different algorithms")
        
        # Effect size recommendations
        large_effects = []
        for comparison, results in significance_results.items():
            acc_effect = results.get("accuracy_test", {}).get("effect_size", 0)
            priv_effect = results.get("privacy_leakage_test", {}).get("effect_size", 0)
            
            if acc_effect > 0.8 or priv_effect > 0.8:
                large_effects.append(comparison)
        
        if large_effects:
            recommendations.append(f"Large effect sizes detected in: {', '.join(large_effects[:3])}")
        
        recommendations.extend([
            "Consider multiple comparison corrections (Bonferroni, FDR) for multiple testing",
            "Validate results with independent test sets",
            "Report confidence intervals alongside p-values",
            "Consider practical significance in addition to statistical significance"
        ])
        
        return recommendations
    
    def generate_reproducibility_report(
        self,
        experiment_config: Dict[str, Any],
        system_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive reproducibility report.
        
        Args:
            experiment_config: Configuration used for experiments
            system_info: System information for reproducibility
            
        Returns:
            Detailed reproducibility report
        """
        logger.info("Generating reproducibility report")
        
        # System information
        import platform
        import sys
        system_info = system_info or {
            "platform": platform.platform(),
            "python_version": sys.version,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "numpy_available": NUMPY_AVAILABLE
        }
        
        # Experiment metadata
        if not self.results:
            return {"error": "No results available for reproducibility report"}
        
        # Extract experiment parameters
        experiment_metadata = {
            "total_experiments": len(self.results),
            "unique_algorithms": len(set(r.algorithm_name for r in self.results)),
            "unique_datasets": len(set(r.dataset_name for r in self.results)),
            "privacy_budgets_tested": sorted(list(set(r.privacy_budget for r in self.results))),
            "experiment_duration": self._calculate_experiment_duration()
        }
        
        # Reproducibility checklist
        reproducibility_checklist = {
            "experiment_config_documented": bool(experiment_config),
            "random_seeds_set": "random_seed" in experiment_config,
            "system_info_recorded": bool(system_info),
            "statistical_testing_performed": hasattr(self, '_statistical_results'),
            "code_version_tracked": "code_version" in experiment_config,
            "data_version_tracked": "data_version" in experiment_config,
            "hyperparameters_documented": "hyperparameters" in experiment_config
        }
        
        reproducibility_score = sum(reproducibility_checklist.values()) / len(reproducibility_checklist)
        
        # Generate reproducibility instructions
        instructions = self._generate_reproducibility_instructions(
            experiment_config, system_info, reproducibility_checklist
        )
        
        # Data integrity checks
        integrity_checks = self._perform_data_integrity_checks()
        
        # Variance analysis
        variance_analysis = self._analyze_result_variance()
        
        reproducibility_report = {
            "system_information": system_info,
            "experiment_metadata": experiment_metadata,
            "reproducibility_checklist": reproducibility_checklist,
            "reproducibility_score": reproducibility_score,
            "reproducibility_grade": self._assign_reproducibility_grade(reproducibility_score),
            "reproducibility_instructions": instructions,
            "data_integrity_checks": integrity_checks,
            "variance_analysis": variance_analysis,
            "recommendations": self._generate_reproducibility_recommendations(
                reproducibility_checklist, variance_analysis
            )
        }
        
        return reproducibility_report
    
    def _calculate_experiment_duration(self) -> Dict[str, Any]:
        """Calculate total experiment duration."""
        if not self.results:
            return {"total_time": 0, "average_time": 0}
        
        timestamps = []
        for result in self.results:
            if hasattr(result, 'timestamp') and result.timestamp:
                try:
                    # Try to parse timestamp
                    import datetime
                    ts = datetime.datetime.strptime(result.timestamp, "%Y-%m-%d %H:%M:%S")
                    timestamps.append(ts)
                except (ValueError, AttributeError):
                    continue
        
        if len(timestamps) >= 2:
            duration = max(timestamps) - min(timestamps)
            return {
                "total_time": duration.total_seconds(),
                "start_time": min(timestamps).isoformat(),
                "end_time": max(timestamps).isoformat(),
                "experiments_with_timestamps": len(timestamps)
            }
        
        # Fallback: estimate from training times
        training_times = [r.training_time for r in self.results]
        total_training_time = sum(training_times)
        
        return {
            "estimated_total_time": total_training_time,
            "average_training_time": statistics.mean(training_times),
            "experiments_counted": len(training_times)
        }
    
    def _generate_reproducibility_instructions(
        self,
        config: Dict[str, Any],
        system_info: Dict[str, Any],
        checklist: Dict[str, bool]
    ) -> List[str]:
        """Generate step-by-step reproducibility instructions."""
        instructions = []
        
        instructions.append("# Reproducibility Instructions")
        instructions.append("")
        
        # System setup
        instructions.append("## System Setup")
        if system_info.get("python_version"):
            instructions.append(f"- Python version: {system_info['python_version']}")
        if system_info.get("platform"):
            instructions.append(f"- Platform: {system_info['platform']}")
        
        instructions.append("")
        
        # Environment setup
        instructions.append("## Environment Setup")
        instructions.append("1. Install required dependencies:")
        instructions.append("   ```bash")
        instructions.append("   pip install numpy scipy scikit-learn")
        instructions.append("   # Add other dependencies as needed")
        instructions.append("   ```")
        
        # Configuration
        if config:
            instructions.append("## Experiment Configuration")
            instructions.append("Use the following configuration:")
            instructions.append("```python")
            instructions.append("config = {")
            for key, value in config.items():
                instructions.append(f"    '{key}': {repr(value)},")
            instructions.append("}")
            instructions.append("```")
        
        # Execution steps
        instructions.append("## Execution Steps")
        instructions.append("1. Load the benchmark suite:")
        instructions.append("   ```python")
        instructions.append("   from privacy_finetuner.research.benchmark_suite import PrivacyBenchmarkSuite")
        instructions.append("   suite = PrivacyBenchmarkSuite()")
        instructions.append("   ```")
        
        instructions.append("2. Run benchmark with configuration:")
        instructions.append("   ```python")
        instructions.append("   results = suite.run_comprehensive_benchmark(config)")
        instructions.append("   ```")
        
        # Verification
        instructions.append("## Result Verification")
        instructions.append(f"Expected number of results: {len(self.results)}")
        if self.results:
            example_result = self.results[0]
            instructions.append(f"Example result structure: {list(example_result.to_dict().keys())}")
        
        return instructions
    
    def _perform_data_integrity_checks(self) -> Dict[str, Any]:
        """Perform data integrity checks on results."""
        if not self.results:
            return {"status": "no_data"}
        
        integrity_checks = {
            "total_results": len(self.results),
            "missing_values": 0,
            "outlier_count": 0,
            "duplicate_count": 0,
            "value_ranges": {},
            "data_consistency": True
        }
        
        # Check for missing values
        for result in self.results:
            result_dict = result.to_dict()
            for key, value in result_dict.items():
                if value is None or (isinstance(value, float) and math.isnan(value)):
                    integrity_checks["missing_values"] += 1
        
        # Check value ranges
        accuracy_values = [r.accuracy for r in self.results]
        privacy_values = [r.privacy_leakage for r in self.results]
        time_values = [r.training_time for r in self.results]
        
        integrity_checks["value_ranges"] = {
            "accuracy": {"min": min(accuracy_values), "max": max(accuracy_values)},
            "privacy_leakage": {"min": min(privacy_values), "max": max(privacy_values)},
            "training_time": {"min": min(time_values), "max": max(time_values)}
        }
        
        # Check for outliers (simple IQR method)
        for values, name in [(accuracy_values, "accuracy"), (privacy_values, "privacy_leakage")]:
            if len(values) >= 4:
                values_sorted = sorted(values)
                q1 = values_sorted[len(values_sorted)//4]
                q3 = values_sorted[3*len(values_sorted)//4]
                iqr = q3 - q1
                
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outliers = [v for v in values if v < lower_bound or v > upper_bound]
                integrity_checks["outlier_count"] += len(outliers)
        
        # Check for duplicates
        result_hashes = []
        for result in self.results:
            # Create hash of key result attributes
            result_str = f"{result.algorithm_name}_{result.dataset_name}_{result.privacy_budget}_{result.accuracy:.4f}"
            result_hash = hashlib.md5(result_str.encode()).hexdigest()
            result_hashes.append(result_hash)
        
        integrity_checks["duplicate_count"] = len(result_hashes) - len(set(result_hashes))
        
        # Overall consistency check
        integrity_checks["data_consistency"] = (
            integrity_checks["missing_values"] == 0 and
            integrity_checks["duplicate_count"] == 0 and
            all(0 <= acc <= 1 for acc in accuracy_values) and
            all(0 <= priv <= 1 for priv in privacy_values)
        )
        
        return integrity_checks
    
    def _analyze_result_variance(self) -> Dict[str, Any]:
        """Analyze variance in experimental results."""
        if not self.results:
            return {"status": "no_data"}
        
        # Group results by algorithm and configuration
        grouped_results = {}
        
        for result in self.results:
            key = f"{result.algorithm_name}_{result.dataset_name}_{result.privacy_budget}"
            if key not in grouped_results:
                grouped_results[key] = []
            grouped_results[key].append(result)
        
        variance_analysis = {
            "configuration_groups": len(grouped_results),
            "groups_with_multiple_runs": 0,
            "average_runs_per_group": 0,
            "high_variance_groups": [],
            "low_variance_groups": [],
            "overall_variance_score": 0.0
        }
        
        multiple_run_groups = []
        variance_scores = []
        
        for config_key, config_results in grouped_results.items():
            if len(config_results) > 1:
                variance_analysis["groups_with_multiple_runs"] += 1
                multiple_run_groups.append(config_key)
                
                # Calculate variance for this group
                accuracies = [r.accuracy for r in config_results]
                privacy_values = [r.privacy_leakage for r in config_results]
                
                acc_variance = statistics.variance(accuracies) if len(accuracies) > 1 else 0
                priv_variance = statistics.variance(privacy_values) if len(privacy_values) > 1 else 0
                
                # Normalized variance score
                combined_variance = acc_variance + priv_variance
                variance_scores.append(combined_variance)
                
                if combined_variance > 0.01:  # High variance threshold
                    variance_analysis["high_variance_groups"].append({
                        "configuration": config_key,
                        "runs": len(config_results),
                        "accuracy_variance": acc_variance,
                        "privacy_variance": priv_variance
                    })
                elif combined_variance < 0.001:  # Low variance threshold
                    variance_analysis["low_variance_groups"].append(config_key)
        
        if multiple_run_groups:
            variance_analysis["average_runs_per_group"] = sum(
                len(grouped_results[key]) for key in multiple_run_groups
            ) / len(multiple_run_groups)
            
            if variance_scores:
                variance_analysis["overall_variance_score"] = statistics.mean(variance_scores)
        
        return variance_analysis
    
    def _assign_reproducibility_grade(self, score: float) -> str:
        """Assign letter grade based on reproducibility score."""
        if score >= 0.9:
            return "A+"
        elif score >= 0.8:
            return "A"
        elif score >= 0.7:
            return "B+"
        elif score >= 0.6:
            return "B"
        elif score >= 0.5:
            return "C+"
        elif score >= 0.4:
            return "C"
        else:
            return "F"
    
    def _generate_reproducibility_recommendations(
        self,
        checklist: Dict[str, bool],
        variance_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations for improving reproducibility."""
        recommendations = []
        
        # Missing checklist items
        missing_items = [item for item, present in checklist.items() if not present]
        
        for item in missing_items:
            if item == "random_seeds_set":
                recommendations.append("Set and document random seeds for all stochastic components")
            elif item == "code_version_tracked":
                recommendations.append("Track and document code version/commit hash")
            elif item == "data_version_tracked":
                recommendations.append("Track and document dataset versions and preprocessing steps")
            elif item == "hyperparameters_documented":
                recommendations.append("Document all hyperparameters and their selection rationale")
        
        # Variance-based recommendations
        if variance_analysis.get("groups_with_multiple_runs", 0) < 3:
            recommendations.append("Increase number of repeated experiments for better statistical power")
        
        high_variance_count = len(variance_analysis.get("high_variance_groups", []))
        if high_variance_count > 0:
            recommendations.append(f"Investigate high variance in {high_variance_count} configuration groups")
        
        # General best practices
        recommendations.extend([
            "Use containerization (Docker) to ensure consistent environment",
            "Document computational resources used (CPU, memory, GPU)",
            "Archive exact versions of all dependencies",
            "Provide automated scripts for result reproduction",
            "Include data preprocessing and feature engineering code",
            "Document any manual intervention or decision points"
        ])
        
        return recommendations
    
    def export_publication_ready_results(
        self,
        output_format: str = "latex",
        include_statistical_tests: bool = True,
        output_path: Optional[str] = None
    ) -> str:
        """Export results in publication-ready format.
        
        Args:
            output_format: Output format ('latex', 'markdown', 'csv')
            include_statistical_tests: Whether to include statistical test results
            output_path: Output file path (None for string return)
            
        Returns:
            Formatted results string
        """
        if not self.results:
            return "No results available for export"
        
        logger.info(f"Exporting publication-ready results in {output_format} format")
        
        if output_format == "latex":
            formatted_results = self._export_latex_table()
        elif output_format == "markdown":
            formatted_results = self._export_markdown_table()
        else:
            formatted_results = self._export_csv_results()
        
        # Add statistical tests if requested
        if include_statistical_tests and hasattr(self, '_statistical_results'):
            if output_format == "latex":
                formatted_results += "\n\n" + self._export_statistical_tests_latex()
            elif output_format == "markdown":
                formatted_results += "\n\n" + self._export_statistical_tests_markdown()
        
        # Save to file if path provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write(formatted_results)
            logger.info(f"Exported results to {output_path}")
        
        return formatted_results
    
    def _export_latex_table(self) -> str:
        """Export results as LaTeX table."""
        # Group results by algorithm
        algorithm_results = {}
        for result in self.results:
            if result.algorithm_name not in algorithm_results:
                algorithm_results[result.algorithm_name] = []
            algorithm_results[result.algorithm_name].append(result)
        
        latex_table = []
        latex_table.append("\\begin{table}[htbp]")
        latex_table.append("\\centering")
        latex_table.append("\\caption{Privacy-Preserving Algorithm Benchmark Results}")
        latex_table.append("\\label{tab:benchmark_results}")
        latex_table.append("\\begin{tabular}{l|c|c|c|c}")
        latex_table.append("\\hline")
        latex_table.append("Algorithm & Accuracy & Privacy Leakage & Training Time (s) & Memory Usage (MB) \\\\")
        latex_table.append("\\hline")
        
        for algorithm, results in algorithm_results.items():
            avg_accuracy = statistics.mean([r.accuracy for r in results])
            avg_privacy = statistics.mean([r.privacy_leakage for r in results])
            avg_time = statistics.mean([r.training_time for r in results])
            avg_memory = statistics.mean([r.memory_usage for r in results])
            
            # Calculate standard deviations
            std_accuracy = statistics.stdev([r.accuracy for r in results]) if len(results) > 1 else 0
            std_privacy = statistics.stdev([r.privacy_leakage for r in results]) if len(results) > 1 else 0
            
            # Format with confidence intervals
            accuracy_str = f"{avg_accuracy:.3f} ± {std_accuracy:.3f}"
            privacy_str = f"{avg_privacy:.3f} ± {std_privacy:.3f}"
            
            latex_table.append(
                f"{algorithm} & {accuracy_str} & {privacy_str} & "
                f"{avg_time:.2f} & {avg_memory:.1f} \\\\"
            )
        
        latex_table.append("\\hline")
        latex_table.append("\\end{tabular}")
        latex_table.append("\\end{table}")
        
        return "\n".join(latex_table)
    
    def _export_markdown_table(self) -> str:
        """Export results as Markdown table."""
        # Group results by algorithm
        algorithm_results = {}
        for result in self.results:
            if result.algorithm_name not in algorithm_results:
                algorithm_results[result.algorithm_name] = []
            algorithm_results[result.algorithm_name].append(result)
        
        markdown_table = []
        markdown_table.append("# Privacy-Preserving Algorithm Benchmark Results")
        markdown_table.append("")
        markdown_table.append("| Algorithm | Accuracy | Privacy Leakage | Training Time (s) | Memory Usage (MB) |")
        markdown_table.append("|-----------|----------|-----------------|------------------|------------------|")
        
        for algorithm, results in algorithm_results.items():
            avg_accuracy = statistics.mean([r.accuracy for r in results])
            avg_privacy = statistics.mean([r.privacy_leakage for r in results])
            avg_time = statistics.mean([r.training_time for r in results])
            avg_memory = statistics.mean([r.memory_usage for r in results])
            
            # Calculate standard deviations
            std_accuracy = statistics.stdev([r.accuracy for r in results]) if len(results) > 1 else 0
            std_privacy = statistics.stdev([r.privacy_leakage for r in results]) if len(results) > 1 else 0
            
            # Format with confidence intervals
            accuracy_str = f"{avg_accuracy:.3f} ± {std_accuracy:.3f}"
            privacy_str = f"{avg_privacy:.3f} ± {std_privacy:.3f}"
            
            markdown_table.append(
                f"| {algorithm} | {accuracy_str} | {privacy_str} | "
                f"{avg_time:.2f} | {avg_memory:.1f} |"
            )
        
        return "\n".join(markdown_table)
    
    def _export_csv_results(self) -> str:
        """Export detailed results as CSV."""
        import io
        output = io.StringIO()
        
        if not self.results:
            return "No results to export"
        
        fieldnames = list(self.results[0].to_dict().keys())
        
        # Write header
        output.write(",".join(fieldnames) + "\n")
        
        # Write results
        for result in self.results:
            result_dict = result.to_dict()
            row = [str(result_dict.get(field, "")) for field in fieldnames]
            output.write(",".join(row) + "\n")
        
        return output.getvalue()
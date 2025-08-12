"""Advanced Federated Learning with Privacy Preservation.

This module implements cutting-edge federated learning approaches including
Byzantine-robust aggregation with differential privacy, personalized FL with
local privacy, cross-silo/device hybrid architectures, and privacy-preserving
transfer learning across federated networks.

Research Reference: Novel federated learning algorithms with enhanced privacy
guarantees and robustness against adversarial participants.
"""

import logging
import time
import math
import random
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Callable, Union, Set
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import json

# Handle numpy import gracefully
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    # Import stub from novel_algorithms
    from .novel_algorithms import NumpyStub, RandomStub
    np = NumpyStub()
    np.random = RandomStub()

logger = logging.getLogger(__name__)


class FederatedLearningMode(Enum):
    """Federated learning operation modes."""
    CROSS_SILO = "cross_silo"
    CROSS_DEVICE = "cross_device"
    HYBRID = "hybrid"
    HIERARCHICAL = "hierarchical"


class AggregationMethod(Enum):
    """Aggregation methods for federated learning."""
    FEDAVG = "federated_averaging"
    BYZANTINE_ROBUST = "byzantine_robust"
    PERSONALIZED = "personalized"
    ADAPTIVE_WEIGHTED = "adaptive_weighted"
    DIFFERENTIAL_PRIVATE = "differential_private"


@dataclass
class ClientProfile:
    """Profile information for federated learning client."""
    client_id: str
    device_type: str  # mobile, server, edge, etc.
    computational_capacity: float  # 0-1 scale
    bandwidth: float  # MB/s
    data_size: int
    privacy_sensitivity: float  # 0-1 scale
    reliability_score: float  # Historical reliability
    geographic_region: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "client_id": self.client_id,
            "device_type": self.device_type,
            "computational_capacity": self.computational_capacity,
            "bandwidth": self.bandwidth,
            "data_size": self.data_size,
            "privacy_sensitivity": self.privacy_sensitivity,
            "reliability_score": self.reliability_score,
            "geographic_region": self.geographic_region
        }


@dataclass
class FederatedRound:
    """Information about a federated learning round."""
    round_number: int
    participating_clients: List[str]
    aggregation_method: AggregationMethod
    privacy_budget_used: float
    convergence_metric: float
    byzantine_detection_results: Dict[str, bool]
    personalization_updates: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_number": self.round_number,
            "participating_clients": self.participating_clients,
            "aggregation_method": self.aggregation_method.value,
            "privacy_budget_used": self.privacy_budget_used,
            "convergence_metric": self.convergence_metric,
            "byzantine_detection_results": self.byzantine_detection_results,
            "personalization_updates": self.personalization_updates
        }


class ByzantineRobustAggregator:
    """Byzantine-robust aggregation with differential privacy guarantees.
    
    Implements robust aggregation methods that can handle malicious clients
    while maintaining differential privacy guarantees.
    """
    
    def __init__(
        self,
        aggregation_rule: str = "trimmed_mean",
        byzantine_tolerance: float = 0.3,
        privacy_epsilon: float = 1.0,
        privacy_delta: float = 1e-5,
        clipping_threshold: float = 1.0
    ):
        """Initialize Byzantine-robust aggregator.
        
        Args:
            aggregation_rule: Aggregation method ('trimmed_mean', 'median', 'krum')
            byzantine_tolerance: Maximum fraction of Byzantine clients to tolerate
            privacy_epsilon: Differential privacy epsilon parameter
            privacy_delta: Differential privacy delta parameter
            clipping_threshold: Gradient clipping threshold
        """
        self.aggregation_rule = aggregation_rule
        self.byzantine_tolerance = byzantine_tolerance
        self.privacy_epsilon = privacy_epsilon
        self.privacy_delta = privacy_delta
        self.clipping_threshold = clipping_threshold
        
        # Byzantine detection state
        self.client_reputation_scores = {}
        self.suspicious_behavior_history = {}
        self.aggregation_history = []
        
        logger.info(f"Initialized Byzantine-robust aggregator with {aggregation_rule} rule")
    
    def robust_aggregate(
        self,
        client_updates: Dict[str, np.ndarray],
        client_profiles: Dict[str, ClientProfile],
        round_number: int
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Perform Byzantine-robust aggregation with privacy preservation.
        
        Args:
            client_updates: Updates from each client
            client_profiles: Profiles of participating clients
            round_number: Current federated learning round
            
        Returns:
            Tuple of (aggregated_update, aggregation_metadata)
        """
        if not client_updates:
            raise ValueError("No client updates provided for aggregation")
        
        logger.debug(f"Robust aggregation for {len(client_updates)} clients")
        
        # Step 1: Preprocess updates (clipping, normalization)
        processed_updates = self._preprocess_updates(client_updates)
        
        # Step 2: Byzantine detection and filtering
        byzantine_results = self._detect_byzantine_clients(
            processed_updates, client_profiles, round_number
        )
        
        # Step 3: Select honest clients for aggregation
        honest_updates = self._filter_honest_updates(
            processed_updates, byzantine_results
        )
        
        if not honest_updates:
            logger.warning("No honest clients detected, falling back to all clients")
            honest_updates = processed_updates
        
        # Step 4: Robust aggregation
        aggregated_update = self._apply_robust_aggregation(honest_updates)
        
        # Step 5: Add differential privacy noise
        private_update = self._add_privacy_noise(aggregated_update)
        
        # Step 6: Update client reputation scores
        self._update_client_reputations(byzantine_results, client_profiles)
        
        # Aggregation metadata
        aggregation_metadata = {
            "round_number": round_number,
            "total_clients": len(client_updates),
            "honest_clients": len(honest_updates),
            "byzantine_detected": len(client_updates) - len(honest_updates),
            "byzantine_results": byzantine_results,
            "aggregation_rule": self.aggregation_rule,
            "privacy_noise_added": True,
            "clipping_applied": True,
            "reputation_updates": {
                client_id: self.client_reputation_scores.get(client_id, 0.5)
                for client_id in client_updates.keys()
            }
        }
        
        self.aggregation_history.append(aggregation_metadata)
        
        return private_update, aggregation_metadata
    
    def _preprocess_updates(self, client_updates: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Preprocess client updates with clipping and normalization."""
        processed_updates = {}
        
        for client_id, update in client_updates.items():
            # Gradient clipping for robustness and privacy
            update_norm = np.linalg.norm(update)
            
            if update_norm > self.clipping_threshold:
                clipped_update = update * (self.clipping_threshold / update_norm)
            else:
                clipped_update = update.copy()
            
            processed_updates[client_id] = clipped_update
        
        return processed_updates
    
    def _detect_byzantine_clients(
        self,
        updates: Dict[str, np.ndarray],
        profiles: Dict[str, ClientProfile],
        round_number: int
    ) -> Dict[str, bool]:
        """Detect Byzantine (malicious) clients using statistical methods."""
        byzantine_results = {}
        client_ids = list(updates.keys())
        
        if len(client_ids) < 3:
            # Not enough clients for robust detection
            return {client_id: False for client_id in client_ids}
        
        # Convert updates to matrix for analysis
        update_matrix = np.array([updates[client_id].flatten() for client_id in client_ids])
        
        # Method 1: Distance-based detection
        distance_scores = self._compute_distance_scores(update_matrix)
        
        # Method 2: Statistical outlier detection
        statistical_scores = self._compute_statistical_outlier_scores(update_matrix)
        
        # Method 3: Historical behavior analysis
        historical_scores = self._compute_historical_scores(client_ids, round_number)
        
        # Combine detection methods
        for i, client_id in enumerate(client_ids):
            combined_score = (
                0.4 * distance_scores[i] +
                0.4 * statistical_scores[i] +
                0.2 * historical_scores[i]
            )
            
            # Byzantine threshold (higher score = more suspicious)
            byzantine_threshold = 0.7
            is_byzantine = combined_score > byzantine_threshold
            
            byzantine_results[client_id] = is_byzantine
            
            # Update suspicious behavior history
            if client_id not in self.suspicious_behavior_history:
                self.suspicious_behavior_history[client_id] = []
            
            self.suspicious_behavior_history[client_id].append({
                "round": round_number,
                "suspicion_score": combined_score,
                "detected_as_byzantine": is_byzantine
            })
        
        # Ensure we don't reject too many clients
        num_byzantine = sum(byzantine_results.values())
        max_byzantine = int(len(client_ids) * self.byzantine_tolerance)
        
        if num_byzantine > max_byzantine:
            # Keep only the most suspicious clients as Byzantine
            suspicious_scores = [
                (client_id, statistical_scores[i] + distance_scores[i])
                for i, client_id in enumerate(client_ids)
                if byzantine_results[client_id]
            ]
            
            suspicious_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Reset all to honest first
            byzantine_results = {client_id: False for client_id in client_ids}
            
            # Mark only the most suspicious ones as Byzantine
            for client_id, _ in suspicious_scores[:max_byzantine]:
                byzantine_results[client_id] = True
        
        return byzantine_results
    
    def _compute_distance_scores(self, update_matrix: np.ndarray) -> List[float]:
        """Compute distance-based suspicion scores for clients."""
        scores = []
        num_clients = update_matrix.shape[0]
        
        for i in range(num_clients):
            # Compute distance to other clients
            distances = []
            for j in range(num_clients):
                if i != j:
                    distance = np.linalg.norm(update_matrix[i] - update_matrix[j])
                    distances.append(distance)
            
            # Suspicion score based on median distance to others
            if distances:
                median_distance = np.median(distances)
                max_distance = np.max(distances)
                
                # Normalize score (higher = more suspicious)
                suspicion_score = median_distance / max(max_distance, 1e-6)
            else:
                suspicion_score = 0.0
            
            scores.append(suspicion_score)
        
        return scores
    
    def _compute_statistical_outlier_scores(self, update_matrix: np.ndarray) -> List[float]:
        """Compute statistical outlier scores for clients."""
        scores = []
        num_clients, num_features = update_matrix.shape
        
        # Compute global statistics
        global_mean = np.mean(update_matrix, axis=0)
        global_std = np.std(update_matrix, axis=0) + 1e-6  # Avoid division by zero
        
        for i in range(num_clients):
            client_update = update_matrix[i]
            
            # Z-score based outlier detection
            z_scores = np.abs((client_update - global_mean) / global_std)
            
            # Suspicion score based on max z-score
            max_z_score = np.max(z_scores)
            suspicion_score = min(1.0, max_z_score / 3.0)  # Normalize to [0, 1]
            
            scores.append(suspicion_score)
        
        return scores
    
    def _compute_historical_scores(self, client_ids: List[str], round_number: int) -> List[float]:
        """Compute historical suspicion scores based on past behavior."""
        scores = []
        
        for client_id in client_ids:
            if client_id not in self.suspicious_behavior_history:
                # New client, neutral score
                scores.append(0.5)
                continue
            
            history = self.suspicious_behavior_history[client_id]
            recent_history = [
                entry for entry in history
                if round_number - entry["round"] <= 10  # Last 10 rounds
            ]
            
            if not recent_history:
                scores.append(0.5)
                continue
            
            # Average suspicion score from recent history
            avg_suspicion = np.mean([entry["suspicion_score"] for entry in recent_history])
            
            # Rate of Byzantine detection in recent history
            byzantine_rate = np.mean([entry["detected_as_byzantine"] for entry in recent_history])
            
            # Combined historical score
            historical_score = 0.6 * avg_suspicion + 0.4 * byzantine_rate
            scores.append(historical_score)
        
        return scores
    
    def _filter_honest_updates(
        self,
        updates: Dict[str, np.ndarray],
        byzantine_results: Dict[str, bool]
    ) -> Dict[str, np.ndarray]:
        """Filter out Byzantine clients from updates."""
        honest_updates = {}
        
        for client_id, update in updates.items():
            if not byzantine_results.get(client_id, False):
                honest_updates[client_id] = update
        
        return honest_updates
    
    def _apply_robust_aggregation(self, honest_updates: Dict[str, np.ndarray]) -> np.ndarray:
        """Apply robust aggregation rule to honest client updates."""
        if not honest_updates:
            raise ValueError("No honest updates available for aggregation")
        
        client_ids = list(honest_updates.keys())
        update_matrix = np.array([honest_updates[client_id].flatten() for client_id in client_ids])
        
        if self.aggregation_rule == "trimmed_mean":
            return self._trimmed_mean_aggregation(update_matrix)
        elif self.aggregation_rule == "median":
            return self._median_aggregation(update_matrix)
        elif self.aggregation_rule == "krum":
            return self._krum_aggregation(update_matrix, honest_updates)
        else:
            # Default to simple averaging
            return np.mean(update_matrix, axis=0)
    
    def _trimmed_mean_aggregation(self, update_matrix: np.ndarray) -> np.ndarray:
        """Trimmed mean aggregation for robustness."""
        # Remove top and bottom 10% of values for each parameter
        trim_fraction = 0.1
        num_clients = update_matrix.shape[0]
        trim_count = max(1, int(num_clients * trim_fraction))
        
        # Sort and trim for each parameter
        aggregated_update = []
        
        for param_idx in range(update_matrix.shape[1]):
            param_values = update_matrix[:, param_idx]
            sorted_values = np.sort(param_values)
            
            # Remove extremes
            if len(sorted_values) > 2 * trim_count:
                trimmed_values = sorted_values[trim_count:-trim_count]
            else:
                trimmed_values = sorted_values
            
            aggregated_update.append(np.mean(trimmed_values))
        
        return np.array(aggregated_update)
    
    def _median_aggregation(self, update_matrix: np.ndarray) -> np.ndarray:
        """Median-based aggregation for robustness."""
        return np.median(update_matrix, axis=0)
    
    def _krum_aggregation(
        self,
        update_matrix: np.ndarray,
        honest_updates: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Krum aggregation method for Byzantine robustness."""
        client_ids = list(honest_updates.keys())
        num_clients = len(client_ids)
        
        if num_clients == 1:
            return update_matrix[0]
        
        # Compute pairwise distances
        distances = np.zeros((num_clients, num_clients))
        
        for i in range(num_clients):
            for j in range(i + 1, num_clients):
                dist = np.linalg.norm(update_matrix[i] - update_matrix[j])
                distances[i, j] = dist
                distances[j, i] = dist
        
        # For each client, compute sum of distances to closest clients
        m = max(1, num_clients - int(num_clients * self.byzantine_tolerance) - 2)
        krum_scores = []
        
        for i in range(num_clients):
            # Get distances to all other clients
            client_distances = [(distances[i, j], j) for j in range(num_clients) if j != i]
            client_distances.sort()
            
            # Sum of distances to m closest clients
            closest_distances = [dist for dist, _ in client_distances[:m]]
            krum_score = sum(closest_distances)
            krum_scores.append(krum_score)
        
        # Select client with minimum Krum score
        selected_client_idx = np.argmin(krum_scores)
        
        return update_matrix[selected_client_idx]
    
    def _add_privacy_noise(self, aggregated_update: np.ndarray) -> np.ndarray:
        """Add differential privacy noise to aggregated update."""
        # Compute noise scale based on sensitivity and privacy parameters
        sensitivity = self.clipping_threshold  # L2 sensitivity
        noise_scale = sensitivity * math.sqrt(2 * math.log(1.25 / self.privacy_delta)) / self.privacy_epsilon
        
        # Add Gaussian noise
        noise = np.random.normal(0, noise_scale, aggregated_update.shape)
        private_update = aggregated_update + noise
        
        return private_update
    
    def _update_client_reputations(
        self,
        byzantine_results: Dict[str, bool],
        client_profiles: Dict[str, ClientProfile]
    ) -> None:
        """Update client reputation scores based on Byzantine detection results."""
        for client_id, is_byzantine in byzantine_results.items():
            if client_id not in self.client_reputation_scores:
                self.client_reputation_scores[client_id] = 0.5  # Neutral starting reputation
            
            current_reputation = self.client_reputation_scores[client_id]
            
            if is_byzantine:
                # Decrease reputation for Byzantine behavior
                new_reputation = current_reputation * 0.8
            else:
                # Increase reputation for honest behavior
                new_reputation = current_reputation * 0.95 + 0.05
            
            # Clamp reputation to [0, 1]
            self.client_reputation_scores[client_id] = max(0.0, min(1.0, new_reputation))


class PersonalizedFederatedLearning:
    """Personalized federated learning with local privacy preservation.
    
    Implements personalization strategies that adapt the global model to
    individual clients while maintaining strong privacy guarantees.
    """
    
    def __init__(
        self,
        personalization_method: str = "fedper",
        local_privacy_epsilon: float = 5.0,
        personalization_layers: Optional[List[str]] = None,
        similarity_threshold: float = 0.8
    ):
        """Initialize personalized federated learning system.
        
        Args:
            personalization_method: Method for personalization ('fedper', 'fedbn', 'clustered')
            local_privacy_epsilon: Local differential privacy parameter
            personalization_layers: Layers to personalize (None = all layers)
            similarity_threshold: Threshold for client clustering
        """
        self.personalization_method = personalization_method
        self.local_privacy_epsilon = local_privacy_epsilon
        self.personalization_layers = personalization_layers
        self.similarity_threshold = similarity_threshold
        
        # Personalization state
        self.client_personal_models = {}
        self.client_clusters = {}
        self.cluster_models = {}
        self.personalization_history = []
        
        logger.info(f"Initialized Personalized FL with {personalization_method} method")
    
    def personalized_aggregation(
        self,
        global_model: Dict[str, np.ndarray],
        client_updates: Dict[str, np.ndarray],
        client_profiles: Dict[str, ClientProfile],
        client_data_stats: Dict[str, Dict[str, Any]],
        round_number: int
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Perform personalized aggregation for federated learning.
        
        Args:
            global_model: Current global model parameters
            client_updates: Updates from participating clients
            client_profiles: Profiles of participating clients
            client_data_stats: Statistics about client data distributions
            round_number: Current federated learning round
            
        Returns:
            Tuple of (personalized_models, personalization_metadata)
        """
        logger.debug(f"Personalized aggregation for {len(client_updates)} clients")
        
        if self.personalization_method == "fedper":
            return self._fedper_aggregation(
                global_model, client_updates, client_profiles, round_number
            )
        elif self.personalization_method == "fedbn":
            return self._fedbn_aggregation(
                global_model, client_updates, client_profiles, round_number
            )
        elif self.personalization_method == "clustered":
            return self._clustered_aggregation(
                global_model, client_updates, client_profiles, client_data_stats, round_number
            )
        else:
            raise ValueError(f"Unknown personalization method: {self.personalization_method}")
    
    def _fedper_aggregation(
        self,
        global_model: Dict[str, np.ndarray],
        client_updates: Dict[str, np.ndarray],
        client_profiles: Dict[str, ClientProfile],
        round_number: int
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """FedPer: Personalized layers + shared base layers."""
        personalized_models = {}
        
        # Update global shared layers
        shared_layers = self._get_shared_layers(global_model)
        updated_global_model = self._aggregate_shared_layers(
            global_model, client_updates, shared_layers
        )
        
        # Create personalized models for each client
        for client_id, client_update in client_updates.items():
            # Start with updated global model
            personalized_model = updated_global_model.copy()
            
            # Apply local personalization with privacy
            personalized_layers = self._apply_local_personalization(
                client_id, client_update, personalized_model
            )
            
            # Merge personalized layers
            for layer_name, layer_params in personalized_layers.items():
                personalized_model[layer_name] = layer_params
            
            personalized_models[client_id] = personalized_model
            self.client_personal_models[client_id] = personalized_model
        
        personalization_metadata = {
            "method": "fedper",
            "round_number": round_number,
            "clients_personalized": list(personalized_models.keys()),
            "shared_layers": list(shared_layers),
            "personalized_layers": self.personalization_layers or ["head_layers"],
            "global_model_updated": True
        }
        
        return personalized_models, personalization_metadata
    
    def _fedbn_aggregation(
        self,
        global_model: Dict[str, np.ndarray],
        client_updates: Dict[str, np.ndarray],
        client_profiles: Dict[str, ClientProfile],
        round_number: int
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """FedBN: Personalized batch normalization layers."""
        personalized_models = {}
        
        # Aggregate non-BN layers globally
        non_bn_layers = {k: v for k, v in global_model.items() if 'bn' not in k.lower()}
        updated_global_model = self._aggregate_non_bn_layers(
            global_model, client_updates, non_bn_layers
        )
        
        # Keep BN layers personalized per client
        for client_id, client_update in client_updates.items():
            personalized_model = updated_global_model.copy()
            
            # Apply client-specific BN parameters with local privacy
            bn_layers = self._extract_bn_layers(client_update)
            private_bn_layers = self._apply_local_privacy_to_bn(client_id, bn_layers)
            
            # Merge BN layers
            for layer_name, layer_params in private_bn_layers.items():
                personalized_model[layer_name] = layer_params
            
            personalized_models[client_id] = personalized_model
            self.client_personal_models[client_id] = personalized_model
        
        personalization_metadata = {
            "method": "fedbn",
            "round_number": round_number,
            "clients_personalized": list(personalized_models.keys()),
            "personalized_bn_layers": len([k for k in global_model.keys() if 'bn' in k.lower()]),
            "global_layers_updated": len(non_bn_layers)
        }
        
        return personalized_models, personalization_metadata
    
    def _clustered_aggregation(
        self,
        global_model: Dict[str, np.ndarray],
        client_updates: Dict[str, np.ndarray],
        client_profiles: Dict[str, ClientProfile],
        client_data_stats: Dict[str, Dict[str, Any]],
        round_number: int
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Clustered personalization based on client similarity."""
        # Update client clusters based on similarity
        self._update_client_clusters(client_updates, client_data_stats, client_profiles)
        
        # Aggregate updates within clusters
        cluster_aggregated_models = self._aggregate_within_clusters(
            global_model, client_updates
        )
        
        # Create personalized models based on cluster assignments
        personalized_models = {}
        
        for client_id, client_update in client_updates.items():
            cluster_id = self.client_clusters.get(client_id, "default")
            
            if cluster_id in cluster_aggregated_models:
                base_model = cluster_aggregated_models[cluster_id]
            else:
                base_model = global_model
            
            # Apply local personalization on top of cluster model
            personalized_model = base_model.copy()
            personalized_layers = self._apply_local_personalization(
                client_id, client_update, personalized_model
            )
            
            # Merge personalized layers
            for layer_name, layer_params in personalized_layers.items():
                personalized_model[layer_name] = layer_params
            
            personalized_models[client_id] = personalized_model
            self.client_personal_models[client_id] = personalized_model
        
        personalization_metadata = {
            "method": "clustered",
            "round_number": round_number,
            "clients_personalized": list(personalized_models.keys()),
            "num_clusters": len(set(self.client_clusters.values())),
            "cluster_assignments": self.client_clusters.copy(),
            "cluster_sizes": self._get_cluster_sizes()
        }
        
        return personalized_models, personalization_metadata
    
    def _get_shared_layers(self, global_model: Dict[str, np.ndarray]) -> List[str]:
        """Get list of shared (non-personalized) layers."""
        if self.personalization_layers is None:
            # Default: personalize final layers, share base layers
            all_layers = list(global_model.keys())
            num_layers = len(all_layers)
            
            # Share first 80% of layers
            num_shared = max(1, int(num_layers * 0.8))
            return all_layers[:num_shared]
        else:
            # Share all layers except specified personalization layers
            return [
                layer_name for layer_name in global_model.keys()
                if layer_name not in self.personalization_layers
            ]
    
    def _aggregate_shared_layers(
        self,
        global_model: Dict[str, np.ndarray],
        client_updates: Dict[str, np.ndarray],
        shared_layers: List[str]
    ) -> Dict[str, np.ndarray]:
        """Aggregate shared layers using FedAvg."""
        updated_model = global_model.copy()
        
        if not client_updates:
            return updated_model
        
        # Simple averaging for shared layers
        num_clients = len(client_updates)
        
        for layer_name in shared_layers:
            if layer_name in global_model:
                layer_sum = np.zeros_like(global_model[layer_name])
                
                for client_update in client_updates.values():
                    if layer_name in client_update:
                        # Assume client_update contains layer differences
                        layer_sum += client_update[layer_name]
                
                # Update with averaged changes
                updated_model[layer_name] = global_model[layer_name] + layer_sum / num_clients
        
        return updated_model
    
    def _apply_local_personalization(
        self,
        client_id: str,
        client_update: np.ndarray,
        base_model: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Apply local personalization with privacy preservation."""
        personalized_layers = {}
        
        # Get personalization layers
        if self.personalization_layers is None:
            # Default: personalize final layers
            all_layer_names = list(base_model.keys())
            personalization_layer_names = all_layer_names[-2:]  # Last 2 layers
        else:
            personalization_layer_names = self.personalization_layers
        
        # Apply local differential privacy to personalized layers
        for layer_name in personalization_layer_names:
            if layer_name in base_model:
                # Get base layer parameters
                base_params = base_model[layer_name]
                
                # Apply client update (assuming it's a gradient)
                updated_params = base_params + client_update[:len(base_params.flatten())].reshape(base_params.shape)
                
                # Add local differential privacy noise
                privacy_noise = self._generate_local_privacy_noise(updated_params.shape)
                private_params = updated_params + privacy_noise
                
                personalized_layers[layer_name] = private_params
        
        return personalized_layers
    
    def _generate_local_privacy_noise(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Generate noise for local differential privacy."""
        # Local DP noise scale
        sensitivity = 1.0  # Assume bounded parameters
        noise_scale = sensitivity / self.local_privacy_epsilon
        
        # Gaussian noise for local DP
        noise = np.random.normal(0, noise_scale, shape)
        
        return noise
    
    def _extract_bn_layers(self, client_update: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract batch normalization layers from client update."""
        # Simplified: assume BN parameters are at specific positions
        # In practice, this would be more sophisticated
        bn_layers = {}
        
        # Example: assume first part of update contains BN parameters
        bn_size = min(100, len(client_update) // 4)  # Simplified assumption
        
        if len(client_update) >= bn_size:
            bn_layers["bn_mean"] = client_update[:bn_size//2]
            bn_layers["bn_var"] = client_update[bn_size//2:bn_size]
        
        return bn_layers
    
    def _apply_local_privacy_to_bn(
        self,
        client_id: str,
        bn_layers: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Apply local privacy to batch normalization layers."""
        private_bn_layers = {}
        
        for layer_name, layer_params in bn_layers.items():
            # Add local DP noise to BN parameters
            privacy_noise = self._generate_local_privacy_noise(layer_params.shape)
            private_params = layer_params + privacy_noise
            
            private_bn_layers[layer_name] = private_params
        
        return private_bn_layers
    
    def _update_client_clusters(
        self,
        client_updates: Dict[str, np.ndarray],
        client_data_stats: Dict[str, Dict[str, Any]],
        client_profiles: Dict[str, ClientProfile]
    ) -> None:
        """Update client clusters based on similarity."""
        client_ids = list(client_updates.keys())
        
        if len(client_ids) < 2:
            # Not enough clients for clustering
            for client_id in client_ids:
                self.client_clusters[client_id] = "cluster_0"
            return
        
        # Compute similarity matrix
        similarity_matrix = self._compute_client_similarity_matrix(
            client_updates, client_data_stats, client_profiles
        )
        
        # Simple clustering based on similarity threshold
        clusters = {}
        cluster_id = 0
        
        for i, client_id in enumerate(client_ids):
            if client_id in clusters:
                continue
            
            # Start new cluster
            current_cluster = f"cluster_{cluster_id}"
            clusters[client_id] = current_cluster
            
            # Find similar clients
            for j, other_client_id in enumerate(client_ids):
                if i != j and other_client_id not in clusters:
                    if similarity_matrix[i][j] >= self.similarity_threshold:
                        clusters[other_client_id] = current_cluster
            
            cluster_id += 1
        
        self.client_clusters.update(clusters)
    
    def _compute_client_similarity_matrix(
        self,
        client_updates: Dict[str, np.ndarray],
        client_data_stats: Dict[str, Dict[str, Any]],
        client_profiles: Dict[str, ClientProfile]
    ) -> List[List[float]]:
        """Compute similarity matrix between clients."""
        client_ids = list(client_updates.keys())
        num_clients = len(client_ids)
        
        similarity_matrix = [[0.0] * num_clients for _ in range(num_clients)]
        
        for i in range(num_clients):
            for j in range(i, num_clients):
                if i == j:
                    similarity_matrix[i][j] = 1.0
                else:
                    # Compute similarity based on multiple factors
                    update_similarity = self._compute_update_similarity(
                        client_updates[client_ids[i]],
                        client_updates[client_ids[j]]
                    )
                    
                    data_similarity = self._compute_data_similarity(
                        client_data_stats.get(client_ids[i], {}),
                        client_data_stats.get(client_ids[j], {})
                    )
                    
                    profile_similarity = self._compute_profile_similarity(
                        client_profiles.get(client_ids[i]),
                        client_profiles.get(client_ids[j])
                    )
                    
                    # Combined similarity
                    combined_similarity = (
                        0.5 * update_similarity +
                        0.3 * data_similarity +
                        0.2 * profile_similarity
                    )
                    
                    similarity_matrix[i][j] = combined_similarity
                    similarity_matrix[j][i] = combined_similarity
        
        return similarity_matrix
    
    def _compute_update_similarity(self, update1: np.ndarray, update2: np.ndarray) -> float:
        """Compute cosine similarity between client updates."""
        if len(update1) == 0 or len(update2) == 0:
            return 0.0
        
        # Ensure same length
        min_len = min(len(update1), len(update2))
        u1 = update1[:min_len]
        u2 = update2[:min_len]
        
        # Cosine similarity
        dot_product = np.dot(u1, u2)
        norm1 = np.linalg.norm(u1)
        norm2 = np.linalg.norm(u2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return max(0.0, similarity)  # Clamp to positive
    
    def _compute_data_similarity(self, stats1: Dict[str, Any], stats2: Dict[str, Any]) -> float:
        """Compute similarity based on data distribution statistics."""
        if not stats1 or not stats2:
            return 0.5  # Neutral similarity if no stats
        
        # Compare common statistics
        common_keys = set(stats1.keys()).intersection(set(stats2.keys()))
        
        if not common_keys:
            return 0.5
        
        similarities = []
        
        for key in common_keys:
            val1 = stats1[key]
            val2 = stats2[key]
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Numerical similarity
                max_val = max(abs(val1), abs(val2), 1e-6)
                sim = 1.0 - abs(val1 - val2) / max_val
                similarities.append(max(0.0, sim))
        
        return np.mean(similarities) if similarities else 0.5
    
    def _compute_profile_similarity(
        self,
        profile1: Optional[ClientProfile],
        profile2: Optional[ClientProfile]
    ) -> float:
        """Compute similarity based on client profiles."""
        if not profile1 or not profile2:
            return 0.5
        
        similarities = []
        
        # Device type similarity
        device_sim = 1.0 if profile1.device_type == profile2.device_type else 0.5
        similarities.append(device_sim)
        
        # Geographic similarity
        geo_sim = 1.0 if profile1.geographic_region == profile2.geographic_region else 0.3
        similarities.append(geo_sim)
        
        # Computational capacity similarity
        cap_diff = abs(profile1.computational_capacity - profile2.computational_capacity)
        cap_sim = 1.0 - cap_diff
        similarities.append(cap_sim)
        
        # Data size similarity
        size1, size2 = profile1.data_size, profile2.data_size
        if size1 > 0 and size2 > 0:
            size_ratio = min(size1, size2) / max(size1, size2)
            similarities.append(size_ratio)
        
        return np.mean(similarities)
    
    def _aggregate_within_clusters(
        self,
        global_model: Dict[str, np.ndarray],
        client_updates: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Aggregate client updates within clusters."""
        cluster_models = {}
        
        # Group clients by cluster
        cluster_clients = {}
        for client_id, cluster_id in self.client_clusters.items():
            if cluster_id not in cluster_clients:
                cluster_clients[cluster_id] = []
            cluster_clients[cluster_id].append(client_id)
        
        # Aggregate within each cluster
        for cluster_id, client_ids in cluster_clients.items():
            cluster_updates = {
                client_id: client_updates[client_id]
                for client_id in client_ids
                if client_id in client_updates
            }
            
            if cluster_updates:
                # Simple averaging within cluster
                cluster_model = global_model.copy()
                
                # Aggregate updates for each layer
                for layer_name in cluster_model.keys():
                    layer_sum = np.zeros_like(cluster_model[layer_name])
                    
                    for client_update in cluster_updates.values():
                        # Assume client_update contains flattened parameters
                        layer_size = cluster_model[layer_name].size
                        if len(client_update) >= layer_size:
                            layer_update = client_update[:layer_size].reshape(cluster_model[layer_name].shape)
                            layer_sum += layer_update
                    
                    # Update layer with averaged changes
                    if len(cluster_updates) > 0:
                        cluster_model[layer_name] = cluster_model[layer_name] + layer_sum / len(cluster_updates)
                
                cluster_models[cluster_id] = cluster_model
        
        self.cluster_models.update(cluster_models)
        return cluster_models
    
    def _get_cluster_sizes(self) -> Dict[str, int]:
        """Get sizes of each cluster."""
        cluster_sizes = {}
        
        for cluster_id in self.client_clusters.values():
            cluster_sizes[cluster_id] = cluster_sizes.get(cluster_id, 0) + 1
        
        return cluster_sizes


class HybridFederatedArchitecture:
    """Hybrid federated learning architecture supporting cross-silo and cross-device.
    
    Implements hierarchical federated learning with different aggregation strategies
    for different types of participants (servers, edge devices, mobile devices).
    """
    
    def __init__(
        self,
        architecture_config: Dict[str, Any] = None,
        privacy_budgets: Dict[str, float] = None
    ):
        """Initialize hybrid federated architecture.
        
        Args:
            architecture_config: Configuration for hybrid architecture
            privacy_budgets: Privacy budgets for different levels
        """
        self.config = architecture_config or self._default_architecture_config()
        self.privacy_budgets = privacy_budgets or {
            "device_level": 10.0,
            "edge_level": 5.0,
            "silo_level": 1.0,
            "global_level": 0.5
        }
        
        # Architecture state
        self.device_clusters = {}  # Device to edge mapping
        self.edge_silos = {}       # Edge to silo mapping
        self.silo_global = {}      # Silo to global mapping
        
        # Aggregators for different levels
        self.device_aggregator = ByzantineRobustAggregator(
            privacy_epsilon=self.privacy_budgets["device_level"]
        )
        self.edge_aggregator = ByzantineRobustAggregator(
            privacy_epsilon=self.privacy_budgets["edge_level"]
        )
        self.silo_aggregator = ByzantineRobustAggregator(
            privacy_epsilon=self.privacy_budgets["silo_level"]
        )
        
        # Hierarchical state
        self.level_models = {
            "device": {},
            "edge": {},
            "silo": {},
            "global": {}
        }
        
        logger.info("Initialized Hybrid Federated Architecture")
    
    def _default_architecture_config(self) -> Dict[str, Any]:
        """Default configuration for hybrid architecture."""
        return {
            "max_devices_per_edge": 10,
            "max_edges_per_silo": 5,
            "max_silos_per_global": 3,
            "aggregation_frequency": {
                "device_to_edge": 5,    # Every 5 rounds
                "edge_to_silo": 10,     # Every 10 rounds  
                "silo_to_global": 20    # Every 20 rounds
            },
            "privacy_amplification": True,
            "adaptive_hierarchy": True
        }
    
    def hierarchical_training_round(
        self,
        participant_updates: Dict[str, Dict[str, Any]],  # participant_id -> {update, profile, type}
        global_model: Dict[str, np.ndarray],
        round_number: int
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Execute one round of hierarchical federated training.
        
        Args:
            participant_updates: Updates from all participants with metadata
            global_model: Current global model
            round_number: Current training round
            
        Returns:
            Tuple of (updated_models_by_level, round_metadata)
        """
        logger.debug(f"Hierarchical training round {round_number}")
        
        # Step 1: Organize participants by type and hierarchy
        hierarchy_map = self._organize_participants_by_hierarchy(participant_updates)
        
        # Step 2: Device-level aggregation
        edge_updates = {}
        if round_number % self.config["aggregation_frequency"]["device_to_edge"] == 0:
            edge_updates = self._aggregate_device_to_edge(
                hierarchy_map["devices"], round_number
            )
        
        # Step 3: Edge-level aggregation  
        silo_updates = {}
        if round_number % self.config["aggregation_frequency"]["edge_to_silo"] == 0:
            silo_updates = self._aggregate_edge_to_silo(
                edge_updates, hierarchy_map["edges"], round_number
            )
        
        # Step 4: Silo-level aggregation
        global_updates = {}
        if round_number % self.config["aggregation_frequency"]["silo_to_global"] == 0:
            global_updates = self._aggregate_silo_to_global(
                silo_updates, hierarchy_map["silos"], global_model, round_number
            )
        
        # Step 5: Update models at each level
        updated_models = self._update_hierarchical_models(
            global_model, global_updates, silo_updates, edge_updates
        )
        
        # Step 6: Adapt hierarchy if needed
        if self.config["adaptive_hierarchy"]:
            self._adapt_hierarchy_structure(hierarchy_map, round_number)
        
        round_metadata = {
            "round_number": round_number,
            "participants_by_level": {
                level: len(participants) 
                for level, participants in hierarchy_map.items()
            },
            "aggregations_performed": {
                "device_to_edge": len(edge_updates) > 0,
                "edge_to_silo": len(silo_updates) > 0,
                "silo_to_global": len(global_updates) > 0
            },
            "privacy_budgets_used": self._compute_privacy_budget_usage(),
            "hierarchy_structure": self._get_hierarchy_structure_summary()
        }
        
        return updated_models, round_metadata
    
    def _organize_participants_by_hierarchy(
        self,
        participant_updates: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Organize participants into hierarchical structure."""
        hierarchy_map = {
            "devices": {},
            "edges": {},
            "silos": {}
        }
        
        for participant_id, participant_data in participant_updates.items():
            participant_type = participant_data.get("profile", {}).get("device_type", "mobile")
            
            if participant_type in ["mobile", "iot", "smartphone"]:
                # Device-level participant
                edge_id = self._assign_device_to_edge(participant_id, participant_data)
                
                if edge_id not in hierarchy_map["devices"]:
                    hierarchy_map["devices"][edge_id] = {}
                
                hierarchy_map["devices"][edge_id][participant_id] = participant_data
            
            elif participant_type in ["edge", "gateway", "router"]:
                # Edge-level participant
                silo_id = self._assign_edge_to_silo(participant_id, participant_data)
                
                if silo_id not in hierarchy_map["edges"]:
                    hierarchy_map["edges"][silo_id] = {}
                
                hierarchy_map["edges"][silo_id][participant_id] = participant_data
            
            elif participant_type in ["server", "datacenter", "cloud"]:
                # Silo-level participant
                hierarchy_map["silos"][participant_id] = participant_data
        
        return hierarchy_map
    
    def _assign_device_to_edge(
        self,
        device_id: str,
        device_data: Dict[str, Any]
    ) -> str:
        """Assign device to edge cluster based on proximity and capacity."""
        profile = device_data.get("profile", {})
        
        # Use geographic region and computational capacity for assignment
        region = profile.get("geographic_region", "unknown")
        capacity = profile.get("computational_capacity", 0.5)
        
        # Find or create appropriate edge cluster
        candidate_edges = []
        
        for edge_id, devices in self.device_clusters.items():
            if len(devices) < self.config["max_devices_per_edge"]:
                # Check if edge serves this region
                edge_region = edge_id.split("_")[1] if "_" in edge_id else "unknown"
                if edge_region == region:
                    candidate_edges.append(edge_id)
        
        if candidate_edges:
            # Assign to edge with most similar capacity devices
            best_edge = min(candidate_edges, key=lambda e: len(self.device_clusters[e]))
            edge_id = best_edge
        else:
            # Create new edge cluster
            edge_id = f"edge_{region}_{len(self.device_clusters)}"
            self.device_clusters[edge_id] = []
        
        if device_id not in self.device_clusters[edge_id]:
            self.device_clusters[edge_id].append(device_id)
        
        return edge_id
    
    def _assign_edge_to_silo(
        self,
        edge_id: str,
        edge_data: Dict[str, Any]
    ) -> str:
        """Assign edge to silo based on capacity and region."""
        profile = edge_data.get("profile", {})
        region = profile.get("geographic_region", "unknown")
        
        # Find or create appropriate silo
        candidate_silos = []
        
        for silo_id, edges in self.edge_silos.items():
            if len(edges) < self.config["max_edges_per_silo"]:
                silo_region = silo_id.split("_")[1] if "_" in silo_id else "unknown"
                if silo_region == region:
                    candidate_silos.append(silo_id)
        
        if candidate_silos:
            silo_id = min(candidate_silos, key=lambda s: len(self.edge_silos[s]))
        else:
            silo_id = f"silo_{region}_{len(self.edge_silos)}"
            self.edge_silos[silo_id] = []
        
        if edge_id not in self.edge_silos[silo_id]:
            self.edge_silos[silo_id].append(edge_id)
        
        return silo_id
    
    def _aggregate_device_to_edge(
        self,
        device_groups: Dict[str, Dict[str, Dict[str, Any]]],
        round_number: int
    ) -> Dict[str, np.ndarray]:
        """Aggregate device updates to edge level."""
        edge_updates = {}
        
        for edge_id, devices in device_groups.items():
            if not devices:
                continue
            
            # Extract device updates and profiles
            device_updates = {}
            device_profiles = {}
            
            for device_id, device_data in devices.items():
                if "update" in device_data:
                    device_updates[device_id] = device_data["update"]
                    device_profiles[device_id] = device_data.get("profile")
            
            if device_updates:
                # Use Byzantine-robust aggregation at device level
                aggregated_update, _ = self.device_aggregator.robust_aggregate(
                    device_updates, device_profiles, round_number
                )
                
                edge_updates[edge_id] = aggregated_update
        
        return edge_updates
    
    def _aggregate_edge_to_silo(
        self,
        edge_updates: Dict[str, np.ndarray],
        edge_groups: Dict[str, Dict[str, Dict[str, Any]]],
        round_number: int
    ) -> Dict[str, np.ndarray]:
        """Aggregate edge updates to silo level."""
        silo_updates = {}
        
        # Include both direct edge participants and aggregated edge updates
        all_edge_data = {}
        
        # Add aggregated device-to-edge updates
        for edge_id, update in edge_updates.items():
            all_edge_data[edge_id] = {"update": update, "profile": None}
        
        # Add direct edge participants
        for silo_id, edges in edge_groups.items():
            for edge_id, edge_data in edges.items():
                all_edge_data[edge_id] = edge_data
        
        # Group by silo assignment
        silo_edge_groups = {}
        for edge_id, edge_data in all_edge_data.items():
            # Find which silo this edge belongs to
            assigned_silo = None
            for silo_id, silo_edges in self.edge_silos.items():
                if edge_id in silo_edges:
                    assigned_silo = silo_id
                    break
            
            if assigned_silo is None:
                # Assign to default silo
                assigned_silo = "silo_default_0"
                if assigned_silo not in self.edge_silos:
                    self.edge_silos[assigned_silo] = []
                self.edge_silos[assigned_silo].append(edge_id)
            
            if assigned_silo not in silo_edge_groups:
                silo_edge_groups[assigned_silo] = {}
            
            silo_edge_groups[assigned_silo][edge_id] = edge_data
        
        # Aggregate within each silo
        for silo_id, edges in silo_edge_groups.items():
            edge_updates_for_silo = {}
            edge_profiles = {}
            
            for edge_id, edge_data in edges.items():
                if "update" in edge_data:
                    edge_updates_for_silo[edge_id] = edge_data["update"]
                    edge_profiles[edge_id] = edge_data.get("profile")
            
            if edge_updates_for_silo:
                aggregated_update, _ = self.edge_aggregator.robust_aggregate(
                    edge_updates_for_silo, edge_profiles, round_number
                )
                
                silo_updates[silo_id] = aggregated_update
        
        return silo_updates
    
    def _aggregate_silo_to_global(
        self,
        silo_updates: Dict[str, np.ndarray],
        silo_participants: Dict[str, Dict[str, Any]],
        global_model: Dict[str, np.ndarray],
        round_number: int
    ) -> Dict[str, np.ndarray]:
        """Aggregate silo updates to global level."""
        # Combine silo aggregation results with direct silo participants
        all_silo_data = {}
        
        # Add aggregated silo updates
        for silo_id, update in silo_updates.items():
            all_silo_data[silo_id] = {"update": update, "profile": None}
        
        # Add direct silo participants
        for silo_id, silo_data in silo_participants.items():
            if "update" in silo_data:
                all_silo_data[silo_id] = silo_data
        
        if not all_silo_data:
            return {"global": global_model["layer_0"]}  # Return unchanged if no updates
        
        # Extract updates and profiles
        silo_updates_dict = {}
        silo_profiles = {}
        
        for silo_id, silo_data in all_silo_data.items():
            if "update" in silo_data:
                silo_updates_dict[silo_id] = silo_data["update"]
                silo_profiles[silo_id] = silo_data.get("profile")
        
        # Global aggregation with highest privacy protection
        if silo_updates_dict:
            global_update, _ = self.silo_aggregator.robust_aggregate(
                silo_updates_dict, silo_profiles, round_number
            )
            
            return {"global": global_update}
        
        return {}
    
    def _update_hierarchical_models(
        self,
        global_model: Dict[str, np.ndarray],
        global_updates: Dict[str, np.ndarray],
        silo_updates: Dict[str, np.ndarray],
        edge_updates: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Update models at each hierarchical level."""
        updated_models = {
            "global": global_model.copy(),
            "silo": {},
            "edge": {},
            "device": {}
        }
        
        # Update global model
        if global_updates:
            for layer_name, layer_params in global_model.items():
                if "global" in global_updates:
                    # Apply global update to this layer
                    layer_size = layer_params.size
                    global_update = global_updates["global"]
                    
                    if len(global_update) >= layer_size:
                        layer_update = global_update[:layer_size].reshape(layer_params.shape)
                        updated_models["global"][layer_name] = layer_params + layer_update * 0.1
        
        # Update silo models based on global model + silo-specific updates
        for silo_id, silo_update in silo_updates.items():
            silo_model = updated_models["global"].copy()
            
            # Apply silo-specific updates
            for layer_name, layer_params in silo_model.items():
                layer_size = layer_params.size
                
                if len(silo_update) >= layer_size:
                    layer_update = silo_update[:layer_size].reshape(layer_params.shape)
                    silo_model[layer_name] = layer_params + layer_update * 0.05
            
            updated_models["silo"][silo_id] = silo_model
        
        # Update edge models
        for edge_id, edge_update in edge_updates.items():
            # Find which silo this edge belongs to
            parent_silo = None
            for silo_id, edges in self.edge_silos.items():
                if edge_id in edges:
                    parent_silo = silo_id
                    break
            
            if parent_silo and parent_silo in updated_models["silo"]:
                base_model = updated_models["silo"][parent_silo]
            else:
                base_model = updated_models["global"]
            
            edge_model = base_model.copy()
            
            # Apply edge-specific updates
            for layer_name, layer_params in edge_model.items():
                layer_size = layer_params.size
                
                if len(edge_update) >= layer_size:
                    layer_update = edge_update[:layer_size].reshape(layer_params.shape)
                    edge_model[layer_name] = layer_params + layer_update * 0.02
            
            updated_models["edge"][edge_id] = edge_model
        
        return updated_models
    
    def _adapt_hierarchy_structure(
        self,
        hierarchy_map: Dict[str, Dict[str, Dict[str, Any]]],
        round_number: int
    ) -> None:
        """Adapt hierarchy structure based on performance and load."""
        # Adapt device-to-edge assignments
        self._rebalance_device_clusters(hierarchy_map["devices"])
        
        # Adapt edge-to-silo assignments  
        self._rebalance_edge_silos(hierarchy_map["edges"])
        
        # Log adaptation
        if round_number % 50 == 0:  # Log every 50 rounds
            logger.info(f"Hierarchy adapted at round {round_number}")
            logger.info(f"Device clusters: {len(self.device_clusters)}")
            logger.info(f"Edge silos: {len(self.edge_silos)}")
    
    def _rebalance_device_clusters(
        self,
        device_groups: Dict[str, Dict[str, Dict[str, Any]]]
    ) -> None:
        """Rebalance device-to-edge assignments."""
        # Find overloaded edges
        overloaded_edges = [
            edge_id for edge_id, devices in self.device_clusters.items()
            if len(devices) > self.config["max_devices_per_edge"]
        ]
        
        # Find underloaded edges
        underloaded_edges = [
            edge_id for edge_id, devices in self.device_clusters.items()
            if len(devices) < self.config["max_devices_per_edge"] // 2
        ]
        
        # Migrate devices from overloaded to underloaded edges
        for overloaded_edge in overloaded_edges:
            if not underloaded_edges:
                break
            
            devices_to_move = self.device_clusters[overloaded_edge][self.config["max_devices_per_edge"]:]
            target_edge = underloaded_edges[0]
            
            # Move excess devices
            for device_id in devices_to_move:
                self.device_clusters[overloaded_edge].remove(device_id)
                self.device_clusters[target_edge].append(device_id)
                
                # Update underloaded edges list
                if len(self.device_clusters[target_edge]) >= self.config["max_devices_per_edge"] // 2:
                    underloaded_edges.remove(target_edge)
                    if not underloaded_edges:
                        break
    
    def _rebalance_edge_silos(
        self,
        edge_groups: Dict[str, Dict[str, Dict[str, Any]]]
    ) -> None:
        """Rebalance edge-to-silo assignments."""
        # Similar rebalancing logic for edge-to-silo assignments
        overloaded_silos = [
            silo_id for silo_id, edges in self.edge_silos.items()
            if len(edges) > self.config["max_edges_per_silo"]
        ]
        
        underloaded_silos = [
            silo_id for silo_id, edges in self.edge_silos.items()
            if len(edges) < self.config["max_edges_per_silo"] // 2
        ]
        
        # Migrate edges from overloaded to underloaded silos
        for overloaded_silo in overloaded_silos:
            if not underloaded_silos:
                break
            
            edges_to_move = self.edge_silos[overloaded_silo][self.config["max_edges_per_silo"]:]
            target_silo = underloaded_silos[0]
            
            for edge_id in edges_to_move:
                self.edge_silos[overloaded_silo].remove(edge_id)
                self.edge_silos[target_silo].append(edge_id)
                
                if len(self.edge_silos[target_silo]) >= self.config["max_edges_per_silo"] // 2:
                    underloaded_silos.remove(target_silo)
                    if not underloaded_silos:
                        break
    
    def _compute_privacy_budget_usage(self) -> Dict[str, float]:
        """Compute privacy budget usage at each level."""
        return {
            "device_level": self.device_aggregator.privacy_spent / self.privacy_budgets["device_level"],
            "edge_level": self.edge_aggregator.privacy_spent / self.privacy_budgets["edge_level"],
            "silo_level": self.silo_aggregator.privacy_spent / self.privacy_budgets["silo_level"],
            "total_composition": (
                self.device_aggregator.privacy_spent +
                self.edge_aggregator.privacy_spent +
                self.silo_aggregator.privacy_spent
            )
        }
    
    def _get_hierarchy_structure_summary(self) -> Dict[str, Any]:
        """Get summary of current hierarchy structure."""
        return {
            "num_device_clusters": len(self.device_clusters),
            "num_edge_silos": len(self.edge_silos),
            "avg_devices_per_edge": np.mean([len(devices) for devices in self.device_clusters.values()]) if self.device_clusters else 0,
            "avg_edges_per_silo": np.mean([len(edges) for edges in self.edge_silos.values()]) if self.edge_silos else 0,
            "total_participants": sum(len(devices) for devices in self.device_clusters.values())
        }


class PrivacyPreservingTransferLearning:
    """Privacy-preserving transfer learning across federated networks.
    
    Enables knowledge transfer between different federated learning domains
    while maintaining privacy guarantees.
    """
    
    def __init__(
        self,
        transfer_method: str = "feature_alignment",
        privacy_epsilon: float = 1.0,
        domain_adaptation_strength: float = 0.5
    ):
        """Initialize privacy-preserving transfer learning.
        
        Args:
            transfer_method: Method for transfer learning
            privacy_epsilon: Privacy parameter for transfer
            domain_adaptation_strength: Strength of domain adaptation
        """
        self.transfer_method = transfer_method
        self.privacy_epsilon = privacy_epsilon
        self.domain_adaptation_strength = domain_adaptation_strength
        
        # Transfer learning state
        self.source_domains = {}
        self.target_domains = {}
        self.transfer_history = []
        
        logger.info(f"Initialized Privacy-Preserving Transfer Learning with {transfer_method}")
    
    def federated_transfer(
        self,
        source_model: Dict[str, np.ndarray],
        target_model: Dict[str, np.ndarray],
        source_domain_info: Dict[str, Any],
        target_domain_info: Dict[str, Any],
        privacy_budget: float
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Perform privacy-preserving transfer learning between federated domains.
        
        Args:
            source_model: Pre-trained source model
            target_model: Target model to adapt
            source_domain_info: Information about source domain
            target_domain_info: Information about target domain
            privacy_budget: Privacy budget for transfer
            
        Returns:
            Tuple of (transferred_model, transfer_metadata)
        """
        logger.debug("Starting federated transfer learning")
        
        if self.transfer_method == "feature_alignment":
            return self._feature_alignment_transfer(
                source_model, target_model, source_domain_info, 
                target_domain_info, privacy_budget
            )
        elif self.transfer_method == "gradual_unfreezing":
            return self._gradual_unfreezing_transfer(
                source_model, target_model, source_domain_info,
                target_domain_info, privacy_budget
            )
        elif self.transfer_method == "domain_adversarial":
            return self._domain_adversarial_transfer(
                source_model, target_model, source_domain_info,
                target_domain_info, privacy_budget
            )
        else:
            raise ValueError(f"Unknown transfer method: {self.transfer_method}")
    
    def _feature_alignment_transfer(
        self,
        source_model: Dict[str, np.ndarray],
        target_model: Dict[str, np.ndarray],
        source_info: Dict[str, Any],
        target_info: Dict[str, Any],
        privacy_budget: float
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Feature alignment-based transfer learning."""
        transferred_model = target_model.copy()
        
        # Identify transferable layers (typically early layers)
        transferable_layers = self._identify_transferable_layers(source_model, target_model)
        
        # Apply privacy-preserving feature alignment
        alignment_info = {}
        
        for layer_name in transferable_layers:
            if layer_name in source_model and layer_name in target_model:
                # Add differential privacy noise to source features
                source_layer = source_model[layer_name]
                private_source_layer = self._add_transfer_privacy_noise(
                    source_layer, privacy_budget / len(transferable_layers)
                )
                
                # Compute alignment transformation
                alignment_transform = self._compute_feature_alignment(
                    private_source_layer, target_model[layer_name]
                )
                
                # Apply aligned features with domain adaptation
                aligned_features = self._apply_domain_adaptation(
                    private_source_layer, alignment_transform
                )
                
                # Weighted combination of source and target features
                alpha = self.domain_adaptation_strength
                transferred_model[layer_name] = (
                    alpha * aligned_features + (1 - alpha) * target_model[layer_name]
                )
                
                alignment_info[layer_name] = {
                    "alignment_quality": self._measure_alignment_quality(
                        aligned_features, target_model[layer_name]
                    ),
                    "domain_distance": self._compute_domain_distance(
                        source_info, target_info
                    )
                }
        
        transfer_metadata = {
            "method": "feature_alignment",
            "transferable_layers": transferable_layers,
            "alignment_info": alignment_info,
            "privacy_budget_used": privacy_budget,
            "domain_adaptation_strength": self.domain_adaptation_strength
        }
        
        return transferred_model, transfer_metadata
    
    def _gradual_unfreezing_transfer(
        self,
        source_model: Dict[str, np.ndarray],
        target_model: Dict[str, np.ndarray],
        source_info: Dict[str, Any],
        target_info: Dict[str, Any],
        privacy_budget: float
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Gradual unfreezing transfer learning with privacy."""
        transferred_model = source_model.copy()
        
        # Add privacy noise to source model
        for layer_name, layer_params in transferred_model.items():
            privacy_noise = self._add_transfer_privacy_noise(
                layer_params, privacy_budget / len(transferred_model)
            )
            transferred_model[layer_name] = layer_params + privacy_noise
        
        # Determine unfreezing schedule based on domain similarity
        domain_similarity = self._compute_domain_similarity(source_info, target_info)
        unfreezing_layers = self._get_unfreezing_schedule(
            list(transferred_model.keys()), domain_similarity
        )
        
        # Fine-tune unfrozen layers with target domain adaptation
        for layer_name in unfreezing_layers:
            if layer_name in target_model:
                # Adaptive learning rate based on domain distance
                domain_distance = self._compute_domain_distance(source_info, target_info)
                adaptation_rate = min(0.5, domain_distance)
                
                # Blend source and target representations
                transferred_model[layer_name] = (
                    (1 - adaptation_rate) * transferred_model[layer_name] +
                    adaptation_rate * target_model[layer_name]
                )
        
        transfer_metadata = {
            "method": "gradual_unfreezing",
            "unfreezing_layers": unfreezing_layers,
            "domain_similarity": domain_similarity,
            "privacy_budget_used": privacy_budget,
            "adaptation_schedule": self._get_adaptation_schedule()
        }
        
        return transferred_model, transfer_metadata
    
    def _domain_adversarial_transfer(
        self,
        source_model: Dict[str, np.ndarray],
        target_model: Dict[str, np.ndarray],
        source_info: Dict[str, Any],
        target_info: Dict[str, Any],
        privacy_budget: float
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Domain adversarial transfer learning with privacy."""
        transferred_model = target_model.copy()
        
        # Create domain discriminator (simplified)
        discriminator_layers = self._create_domain_discriminator()
        
        # Adversarial training with privacy constraints
        adversarial_updates = {}
        
        for layer_name in transferred_model.keys():
            if layer_name in source_model:
                # Compute adversarial gradient
                source_layer = source_model[layer_name]
                target_layer = target_model[layer_name]
                
                # Add privacy noise to gradients
                private_source = self._add_transfer_privacy_noise(
                    source_layer, privacy_budget / (2 * len(transferred_model))
                )
                private_target = self._add_transfer_privacy_noise(
                    target_layer, privacy_budget / (2 * len(transferred_model))
                )
                
                # Adversarial update to make features domain-invariant
                adversarial_gradient = self._compute_adversarial_gradient(
                    private_source, private_target, discriminator_layers
                )
                
                # Apply adversarial update
                learning_rate = 0.01 * self.domain_adaptation_strength
                transferred_model[layer_name] = target_layer - learning_rate * adversarial_gradient
                
                adversarial_updates[layer_name] = {
                    "gradient_norm": np.linalg.norm(adversarial_gradient),
                    "domain_confusion": self._measure_domain_confusion(
                        private_source, private_target
                    )
                }
        
        transfer_metadata = {
            "method": "domain_adversarial",
            "adversarial_updates": adversarial_updates,
            "privacy_budget_used": privacy_budget,
            "discriminator_accuracy": self._evaluate_discriminator(discriminator_layers)
        }
        
        return transferred_model, transfer_metadata
    
    def _identify_transferable_layers(
        self,
        source_model: Dict[str, np.ndarray],
        target_model: Dict[str, np.ndarray]
    ) -> List[str]:
        """Identify layers that can be transferred between models."""
        transferable = []
        
        for layer_name in source_model.keys():
            if layer_name in target_model:
                source_shape = source_model[layer_name].shape
                target_shape = target_model[layer_name].shape
                
                # Check if shapes are compatible
                if source_shape == target_shape:
                    transferable.append(layer_name)
                elif len(source_shape) == len(target_shape):
                    # Partial compatibility - can transfer with adaptation
                    transferable.append(layer_name)
        
        # Typically transfer early layers first
        return sorted(transferable)[:len(transferable)//2]
    
    def _add_transfer_privacy_noise(
        self,
        layer_params: np.ndarray,
        epsilon: float
    ) -> np.ndarray:
        """Add differential privacy noise for transfer learning."""
        if epsilon <= 0:
            return np.zeros_like(layer_params)
        
        # Sensitivity for transfer (L2 norm bounded)
        sensitivity = 1.0
        
        # Gaussian mechanism
        noise_scale = sensitivity * math.sqrt(2 * math.log(1.25 / 1e-5)) / epsilon
        noise = np.random.normal(0, noise_scale, layer_params.shape)
        
        return noise
    
    def _compute_feature_alignment(
        self,
        source_features: np.ndarray,
        target_features: np.ndarray
    ) -> np.ndarray:
        """Compute alignment transformation between feature spaces."""
        # Simplified linear alignment (in practice, could be more complex)
        if source_features.shape != target_features.shape:
            # Handle shape mismatch with padding/truncation
            min_shape = tuple(min(s, t) for s, t in zip(source_features.shape, target_features.shape))
            source_aligned = source_features[:min_shape[0]] if len(source_features.shape) > 0 else source_features
            target_aligned = target_features[:min_shape[0]] if len(target_features.shape) > 0 else target_features
        else:
            source_aligned = source_features
            target_aligned = target_features
        
        # Simple scaling transformation
        source_norm = np.linalg.norm(source_aligned) + 1e-6
        target_norm = np.linalg.norm(target_aligned) + 1e-6
        
        alignment_transform = target_norm / source_norm
        
        return np.full_like(source_features, alignment_transform)
    
    def _apply_domain_adaptation(
        self,
        source_features: np.ndarray,
        alignment_transform: np.ndarray
    ) -> np.ndarray:
        """Apply domain adaptation transformation."""
        return source_features * alignment_transform
    
    def _measure_alignment_quality(
        self,
        aligned_features: np.ndarray,
        target_features: np.ndarray
    ) -> float:
        """Measure quality of feature alignment."""
        if aligned_features.shape != target_features.shape:
            return 0.5  # Partial alignment
        
        # Cosine similarity as alignment quality measure
        dot_product = np.sum(aligned_features * target_features)
        norm_aligned = np.linalg.norm(aligned_features) + 1e-6
        norm_target = np.linalg.norm(target_features) + 1e-6
        
        cosine_similarity = dot_product / (norm_aligned * norm_target)
        
        # Convert to [0, 1] range
        return (cosine_similarity + 1) / 2
    
    def _compute_domain_distance(
        self,
        source_info: Dict[str, Any],
        target_info: Dict[str, Any]
    ) -> float:
        """Compute distance between source and target domains."""
        # Simple domain distance based on available statistics
        distance_factors = []
        
        # Data distribution distance
        if "data_distribution" in source_info and "data_distribution" in target_info:
            source_dist = source_info["data_distribution"]
            target_dist = target_info["data_distribution"]
            
            # Compare means
            source_mean = source_dist.get("mean", 0.0)
            target_mean = target_dist.get("mean", 0.0)
            mean_distance = abs(source_mean - target_mean)
            distance_factors.append(mean_distance)
            
            # Compare variances
            source_var = source_dist.get("variance", 1.0)
            target_var = target_dist.get("variance", 1.0)
            var_distance = abs(source_var - target_var) / max(source_var, target_var, 1e-6)
            distance_factors.append(var_distance)
        
        # Task similarity distance
        source_task = source_info.get("task_type", "unknown")
        target_task = target_info.get("task_type", "unknown")
        
        if source_task == target_task:
            task_distance = 0.0
        elif "classification" in source_task and "classification" in target_task:
            task_distance = 0.3
        else:
            task_distance = 1.0
        
        distance_factors.append(task_distance)
        
        return np.mean(distance_factors) if distance_factors else 0.5
    
    def _compute_domain_similarity(
        self,
        source_info: Dict[str, Any],
        target_info: Dict[str, Any]
    ) -> float:
        """Compute similarity between domains."""
        return 1.0 - self._compute_domain_distance(source_info, target_info)
    
    def _get_unfreezing_schedule(
        self,
        layer_names: List[str],
        domain_similarity: float
    ) -> List[str]:
        """Get schedule for gradual unfreezing."""
        # Unfreeze more layers if domains are similar
        if domain_similarity > 0.8:
            return layer_names[-3:]  # Unfreeze last 3 layers
        elif domain_similarity > 0.5:
            return layer_names[-2:]  # Unfreeze last 2 layers
        else:
            return layer_names[-1:]  # Unfreeze only last layer
    
    def _get_adaptation_schedule(self) -> Dict[str, float]:
        """Get adaptation schedule for transfer learning."""
        return {
            "initial_rate": 0.01,
            "decay_factor": 0.95,
            "min_rate": 0.001,
            "adaptation_steps": 100
        }
    
    def _create_domain_discriminator(self) -> Dict[str, np.ndarray]:
        """Create simple domain discriminator for adversarial training."""
        return {
            "discriminator_weight": np.random.normal(0, 0.1, (64, 32)),
            "discriminator_bias": np.zeros(32),
            "output_weight": np.random.normal(0, 0.1, (32, 1)),
            "output_bias": np.zeros(1)
        }
    
    def _compute_adversarial_gradient(
        self,
        source_features: np.ndarray,
        target_features: np.ndarray,
        discriminator: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """Compute adversarial gradient for domain adaptation."""
        # Simplified adversarial gradient computation
        feature_diff = target_features - source_features
        
        # Simple gradient based on feature difference
        adversarial_gradient = 0.1 * feature_diff / max(np.linalg.norm(feature_diff), 1e-6)
        
        return adversarial_gradient
    
    def _measure_domain_confusion(
        self,
        source_features: np.ndarray,
        target_features: np.ndarray
    ) -> float:
        """Measure domain confusion (higher = better domain adaptation)."""
        # Simple measure based on feature similarity
        if source_features.shape != target_features.shape:
            return 0.5
        
        similarity = np.corrcoef(source_features.flatten(), target_features.flatten())[0, 1]
        
        # Convert correlation to confusion measure
        return abs(similarity) if not np.isnan(similarity) else 0.0
    
    def _evaluate_discriminator(self, discriminator: Dict[str, np.ndarray]) -> float:
        """Evaluate discriminator accuracy (lower = better domain adaptation)."""
        # Simplified evaluation - in practice would use validation data
        return 0.5 + 0.1 * np.random.random()  # Random accuracy around chance level


def create_advanced_federated_system(
    num_clients: int = 10,
    byzantine_tolerance: float = 0.3,
    privacy_budget: float = 1.0
) -> Dict[str, Any]:
    """Create comprehensive advanced federated learning system.
    
    Factory function to initialize all advanced federated learning components.
    
    Args:
        num_clients: Number of federated learning clients
        byzantine_tolerance: Tolerance for Byzantine clients
        privacy_budget: Total privacy budget
        
    Returns:
        Dictionary containing all advanced federated learning components
    """
    logger.info("Creating advanced federated learning system")
    
    return {
        "byzantine_robust_aggregator": ByzantineRobustAggregator(
            byzantine_tolerance=byzantine_tolerance,
            privacy_epsilon=privacy_budget
        ),
        "personalized_fl": PersonalizedFederatedLearning(
            local_privacy_epsilon=privacy_budget * 2
        ),
        "hybrid_architecture": HybridFederatedArchitecture(
            privacy_budgets={
                "device_level": privacy_budget * 5,
                "edge_level": privacy_budget * 2,
                "silo_level": privacy_budget,
                "global_level": privacy_budget * 0.5
            }
        ),
        "transfer_learning": PrivacyPreservingTransferLearning(
            privacy_epsilon=privacy_budget * 0.5
        )
    }
"""Advanced load balancing system for distributed training workloads.

This module implements intelligent load balancing including:
- Dynamic load balancing algorithms (round-robin, least-connections, weighted)
- Health monitoring and automatic failover
- Geographic and topology-aware routing
- Training-aware load distribution considering privacy constraints
"""

import logging
import time
import asyncio
import threading
import hashlib
import random
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import deque, defaultdict
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import aiohttp
import json

logger = logging.getLogger(__name__)


class LoadBalancingAlgorithm(Enum):
    """Load balancing algorithms."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_RESPONSE_TIME = "least_response_time"
    IP_HASH = "ip_hash"
    CONSISTENT_HASH = "consistent_hash"
    PRIVACY_AWARE = "privacy_aware"
    TRAINING_AWARE = "training_aware"
    GEOGRAPHIC = "geographic"
    ADAPTIVE = "adaptive"


class HealthCheckType(Enum):
    """Health check types."""
    HTTP = "http"
    TCP = "tcp"
    GRPC = "grpc"
    CUSTOM = "custom"


class NodeState(Enum):
    """Load balancer node states."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DRAINING = "draining"
    MAINTENANCE = "maintenance"
    STARTING = "starting"


@dataclass
class LoadBalancerConfig:
    """Configuration for load balancer."""
    # Algorithm settings
    algorithm: LoadBalancingAlgorithm = LoadBalancingAlgorithm.ADAPTIVE
    fallback_algorithm: LoadBalancingAlgorithm = LoadBalancingAlgorithm.ROUND_ROBIN
    
    # Health checking
    health_check_enabled: bool = True
    health_check_type: HealthCheckType = HealthCheckType.HTTP
    health_check_interval: int = 10  # seconds
    health_check_timeout: int = 5    # seconds
    health_check_path: str = "/health"
    
    # Failover settings
    max_failures: int = 3
    failure_window: int = 60  # seconds
    recovery_check_interval: int = 30  # seconds
    
    # Connection settings
    max_connections_per_node: int = 100
    connection_timeout: int = 30
    idle_timeout: int = 300
    
    # Privacy and training settings
    privacy_aware_routing: bool = True
    training_step_affinity: bool = True
    min_nodes_for_privacy: int = 3
    
    # Geographic settings
    enable_geographic_routing: bool = False
    prefer_local_region: bool = True
    max_cross_region_ratio: float = 0.3
    
    # Performance settings
    enable_sticky_sessions: bool = False
    session_timeout: int = 1800  # seconds
    async_health_checks: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'algorithm': self.algorithm.value,
            'health_check_enabled': self.health_check_enabled,
            'health_check_type': self.health_check_type.value,
            'max_failures': self.max_failures,
            'privacy_aware_routing': self.privacy_aware_routing,
            'training_step_affinity': self.training_step_affinity,
            'enable_geographic_routing': self.enable_geographic_routing
        }


@dataclass
class LoadBalancerNode:
    """Load balancer backend node."""
    node_id: str
    address: str
    port: int
    weight: float = 1.0
    region: str = "default"
    zone: str = "default"
    
    # State tracking
    state: NodeState = NodeState.HEALTHY
    current_connections: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    
    # Performance metrics
    avg_response_time: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    gpu_usage: float = 0.0
    
    # Health tracking
    consecutive_failures: int = 0
    last_failure_time: Optional[datetime] = None
    last_health_check: Optional[datetime] = None
    health_check_failures: int = 0
    
    # Training-specific metrics
    current_training_step: int = 0
    privacy_budget_remaining: float = 1.0
    training_load: float = 0.0
    
    # Connection tracking
    active_connections: int = 0
    connection_history: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'node_id': self.node_id,
            'address': self.address,
            'port': self.port,
            'weight': self.weight,
            'region': self.region,
            'zone': self.zone,
            'state': self.state.value,
            'current_connections': self.current_connections,
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'avg_response_time': self.avg_response_time,
            'cpu_usage': self.cpu_usage,
            'memory_usage': self.memory_usage,
            'gpu_usage': self.gpu_usage,
            'consecutive_failures': self.consecutive_failures,
            'privacy_budget_remaining': self.privacy_budget_remaining,
            'training_load': self.training_load,
            'active_connections': self.active_connections
        }
    
    def update_response_time(self, response_time: float) -> None:
        """Update average response time."""
        if self.avg_response_time == 0:
            self.avg_response_time = response_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.avg_response_time = alpha * response_time + (1 - alpha) * self.avg_response_time
    
    def record_request(self, success: bool, response_time: float = 0.0) -> None:
        """Record request completion."""
        self.total_requests += 1
        
        if success:
            self.successful_requests += 1
            self.consecutive_failures = 0
            if response_time > 0:
                self.update_response_time(response_time)
        else:
            self.failed_requests += 1
            self.consecutive_failures += 1
            self.last_failure_time = datetime.now()
    
    def get_success_rate(self) -> float:
        """Get success rate."""
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests
    
    def get_load_score(self) -> float:
        """Get current load score (0=no load, 1=maximum load)."""
        # Combine multiple load factors
        connection_load = self.active_connections / 100.0  # Assume max 100 connections
        cpu_load = self.cpu_usage / 100.0
        memory_load = self.memory_usage / 100.0
        gpu_load = self.gpu_usage / 100.0
        training_load_factor = self.training_load
        
        # Weighted combination
        weights = [0.3, 0.25, 0.25, 0.1, 0.1]  # connection, cpu, memory, gpu, training
        loads = [connection_load, cpu_load, memory_load, gpu_load, training_load_factor]
        
        return np.average(loads, weights=weights)


@dataclass
class RoutingRequest:
    """Request for load balancer routing."""
    request_id: str
    client_ip: str
    client_region: str = "unknown"
    request_type: str = "training"
    
    # Training-specific attributes
    training_step: Optional[int] = None
    privacy_level: str = "standard"
    batch_size: int = 32
    
    # Session attributes
    session_id: Optional[str] = None
    requires_affinity: bool = False
    
    # Preferences
    preferred_region: Optional[str] = None
    max_latency_ms: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'request_id': self.request_id,
            'client_ip': self.client_ip,
            'client_region': self.client_region,
            'request_type': self.request_type,
            'training_step': self.training_step,
            'privacy_level': self.privacy_level,
            'batch_size': self.batch_size,
            'session_id': self.session_id,
            'requires_affinity': self.requires_affinity
        }


class HealthChecker:
    """Health monitoring for load balancer nodes."""
    
    def __init__(self, config: LoadBalancerConfig):
        """Initialize health checker.
        
        Args:
            config: Load balancer configuration
        """
        self.config = config
        self.session = None
        self.custom_health_checks: Dict[str, Callable] = {}
        self.health_check_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        
    async def initialize(self) -> None:
        """Initialize async resources."""
        if self.config.health_check_type == HealthCheckType.HTTP:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.health_check_timeout)
            )
    
    async def check_node_health(self, node: LoadBalancerNode) -> bool:
        """Check health of a specific node.
        
        Args:
            node: Node to check
            
        Returns:
            True if node is healthy
        """
        try:
            if self.config.health_check_type == HealthCheckType.HTTP:
                return await self._http_health_check(node)
            elif self.config.health_check_type == HealthCheckType.TCP:
                return await self._tcp_health_check(node)
            elif self.config.health_check_type == HealthCheckType.GRPC:
                return await self._grpc_health_check(node)
            elif self.config.health_check_type == HealthCheckType.CUSTOM:
                return await self._custom_health_check(node)
            
            return False
            
        except Exception as e:
            logger.error(f"Health check failed for node {node.node_id}: {e}")
            return False
    
    async def _http_health_check(self, node: LoadBalancerNode) -> bool:
        """Perform HTTP health check."""
        if not self.session:
            await self.initialize()
        
        url = f"http://{node.address}:{node.port}{self.config.health_check_path}"
        
        try:
            start_time = time.time()
            async with self.session.get(url) as response:
                response_time = time.time() - start_time
                
                if response.status == 200:
                    node.update_response_time(response_time * 1000)  # Convert to ms
                    self.health_check_history[node.node_id].append({
                        'timestamp': datetime.now(),
                        'success': True,
                        'response_time': response_time
                    })
                    return True
                else:
                    logger.warning(f"HTTP health check failed for {node.node_id}: status {response.status}")
                    return False
                    
        except Exception as e:
            logger.debug(f"HTTP health check exception for {node.node_id}: {e}")
            self.health_check_history[node.node_id].append({
                'timestamp': datetime.now(),
                'success': False,
                'error': str(e)
            })
            return False
    
    async def _tcp_health_check(self, node: LoadBalancerNode) -> bool:
        """Perform TCP health check."""
        try:
            # Simple TCP connection test
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(node.address, node.port),
                timeout=self.config.health_check_timeout
            )
            
            writer.close()
            await writer.wait_closed()
            
            return True
            
        except Exception:
            return False
    
    async def _grpc_health_check(self, node: LoadBalancerNode) -> bool:
        """Perform gRPC health check."""
        # This would implement gRPC health checking
        # For now, fall back to TCP
        return await self._tcp_health_check(node)
    
    async def _custom_health_check(self, node: LoadBalancerNode) -> bool:
        """Perform custom health check."""
        if node.node_id in self.custom_health_checks:
            try:
                return await self.custom_health_checks[node.node_id](node)
            except Exception as e:
                logger.error(f"Custom health check failed for {node.node_id}: {e}")
        
        return False
    
    def register_custom_health_check(self, node_id: str, health_check_func: Callable) -> None:
        """Register custom health check function."""
        self.custom_health_checks[node_id] = health_check_func
        logger.info(f"Registered custom health check for node {node_id}")
    
    def get_health_history(self, node_id: str) -> List[Dict[str, Any]]:
        """Get health check history for a node."""
        return list(self.health_check_history[node_id])
    
    async def shutdown(self) -> None:
        """Shutdown health checker."""
        if self.session:
            await self.session.close()


class LoadBalancingAlgorithms:
    """Implementation of various load balancing algorithms."""
    
    @staticmethod
    def round_robin(nodes: List[LoadBalancerNode], state: Dict[str, Any]) -> LoadBalancerNode:
        """Round-robin load balancing."""
        healthy_nodes = [n for n in nodes if n.state == NodeState.HEALTHY]
        if not healthy_nodes:
            raise ValueError("No healthy nodes available")
        
        current_index = state.get('round_robin_index', 0)
        selected_node = healthy_nodes[current_index % len(healthy_nodes)]
        
        state['round_robin_index'] = (current_index + 1) % len(healthy_nodes)
        return selected_node
    
    @staticmethod
    def least_connections(nodes: List[LoadBalancerNode], state: Dict[str, Any]) -> LoadBalancerNode:
        """Least connections load balancing."""
        healthy_nodes = [n for n in nodes if n.state == NodeState.HEALTHY]
        if not healthy_nodes:
            raise ValueError("No healthy nodes available")
        
        return min(healthy_nodes, key=lambda n: n.active_connections)
    
    @staticmethod
    def weighted_round_robin(nodes: List[LoadBalancerNode], state: Dict[str, Any]) -> LoadBalancerNode:
        """Weighted round-robin load balancing."""
        healthy_nodes = [n for n in nodes if n.state == NodeState.HEALTHY]
        if not healthy_nodes:
            raise ValueError("No healthy nodes available")
        
        # Create weighted list
        weighted_nodes = []
        for node in healthy_nodes:
            weight_count = max(1, int(node.weight * 10))  # Scale weight
            weighted_nodes.extend([node] * weight_count)
        
        current_index = state.get('weighted_rr_index', 0)
        selected_node = weighted_nodes[current_index % len(weighted_nodes)]
        
        state['weighted_rr_index'] = (current_index + 1) % len(weighted_nodes)
        return selected_node
    
    @staticmethod
    def least_response_time(nodes: List[LoadBalancerNode], state: Dict[str, Any]) -> LoadBalancerNode:
        """Least response time load balancing."""
        healthy_nodes = [n for n in nodes if n.state == NodeState.HEALTHY]
        if not healthy_nodes:
            raise ValueError("No healthy nodes available")
        
        return min(healthy_nodes, key=lambda n: n.avg_response_time)
    
    @staticmethod
    def ip_hash(nodes: List[LoadBalancerNode], state: Dict[str, Any], client_ip: str) -> LoadBalancerNode:
        """IP hash-based load balancing."""
        healthy_nodes = [n for n in nodes if n.state == NodeState.HEALTHY]
        if not healthy_nodes:
            raise ValueError("No healthy nodes available")
        
        # Hash client IP to select node
        hash_value = int(hashlib.md5(client_ip.encode()).hexdigest(), 16)
        selected_index = hash_value % len(healthy_nodes)
        
        return healthy_nodes[selected_index]
    
    @staticmethod
    def consistent_hash(
        nodes: List[LoadBalancerNode], 
        state: Dict[str, Any], 
        client_ip: str
    ) -> LoadBalancerNode:
        """Consistent hashing load balancing."""
        healthy_nodes = [n for n in nodes if n.state == NodeState.HEALTHY]
        if not healthy_nodes:
            raise ValueError("No healthy nodes available")
        
        # Simple consistent hashing implementation
        hash_ring = state.get('hash_ring', {})
        
        # Build hash ring if not exists or nodes changed
        if not hash_ring or len(hash_ring) != len(healthy_nodes):
            hash_ring = {}
            for node in healthy_nodes:
                # Create multiple virtual nodes for better distribution
                for i in range(3):
                    virtual_node_key = f"{node.node_id}:{i}"
                    hash_value = int(hashlib.md5(virtual_node_key.encode()).hexdigest(), 16)
                    hash_ring[hash_value] = node
            
            state['hash_ring'] = hash_ring
        
        # Find closest node in ring
        client_hash = int(hashlib.md5(client_ip.encode()).hexdigest(), 16)
        sorted_hashes = sorted(hash_ring.keys())
        
        for hash_value in sorted_hashes:
            if hash_value >= client_hash:
                return hash_ring[hash_value]
        
        # Wrap around to first node
        return hash_ring[sorted_hashes[0]]
    
    @staticmethod
    def privacy_aware(
        nodes: List[LoadBalancerNode], 
        state: Dict[str, Any], 
        request: RoutingRequest
    ) -> LoadBalancerNode:
        """Privacy-aware load balancing."""
        healthy_nodes = [n for n in nodes if n.state == NodeState.HEALTHY]
        if not healthy_nodes:
            raise ValueError("No healthy nodes available")
        
        # Filter nodes based on privacy requirements
        if request.privacy_level == "high":
            # Require nodes with sufficient privacy budget
            suitable_nodes = [
                n for n in healthy_nodes 
                if n.privacy_budget_remaining > 0.3
            ]
        else:
            suitable_nodes = healthy_nodes
        
        if not suitable_nodes:
            suitable_nodes = healthy_nodes  # Fallback
        
        # Select based on privacy budget and load
        def privacy_score(node):
            privacy_factor = node.privacy_budget_remaining * 0.7
            load_factor = (1.0 - node.get_load_score()) * 0.3
            return privacy_factor + load_factor
        
        return max(suitable_nodes, key=privacy_score)
    
    @staticmethod
    def training_aware(
        nodes: List[LoadBalancerNode], 
        state: Dict[str, Any], 
        request: RoutingRequest
    ) -> LoadBalancerNode:
        """Training-aware load balancing."""
        healthy_nodes = [n for n in nodes if n.state == NodeState.HEALTHY]
        if not healthy_nodes:
            raise ValueError("No healthy nodes available")
        
        # For training step affinity, prefer nodes on same training step
        if request.training_step is not None:
            same_step_nodes = [
                n for n in healthy_nodes 
                if abs(n.current_training_step - request.training_step) <= 1
            ]
            if same_step_nodes:
                healthy_nodes = same_step_nodes
        
        # Select based on training load and capacity
        def training_score(node):
            load_factor = 1.0 - node.training_load
            gpu_factor = 1.0 - (node.gpu_usage / 100.0) if node.gpu_usage > 0 else 1.0
            return load_factor * 0.6 + gpu_factor * 0.4
        
        return max(healthy_nodes, key=training_score)
    
    @staticmethod
    def geographic(
        nodes: List[LoadBalancerNode], 
        state: Dict[str, Any], 
        request: RoutingRequest
    ) -> LoadBalancerNode:
        """Geographic load balancing."""
        healthy_nodes = [n for n in nodes if n.state == NodeState.HEALTHY]
        if not healthy_nodes:
            raise ValueError("No healthy nodes available")
        
        # Prefer nodes in same region
        if request.client_region != "unknown":
            same_region_nodes = [
                n for n in healthy_nodes 
                if n.region == request.client_region
            ]
            if same_region_nodes:
                # Use least connections within same region
                return min(same_region_nodes, key=lambda n: n.active_connections)
        
        # Fallback to least connections
        return min(healthy_nodes, key=lambda n: n.active_connections)


class IntelligentLoadBalancer:
    """Intelligent load balancer for distributed training workloads."""
    
    def __init__(self, config: Optional[LoadBalancerConfig] = None):
        """Initialize load balancer.
        
        Args:
            config: Load balancer configuration
        """
        self.config = config or LoadBalancerConfig()
        
        # Node management
        self.nodes: Dict[str, LoadBalancerNode] = {}
        self.algorithm_state: Dict[str, Any] = {}
        
        # Health monitoring
        self.health_checker = HealthChecker(self.config)
        self.health_monitor_active = False
        self.health_monitor_task = None
        
        # Session management
        self.session_affinity: Dict[str, str] = {}  # session_id -> node_id
        self.session_timeouts: Dict[str, datetime] = {}
        
        # Statistics
        self.routing_stats = {
            'total_requests': 0,
            'successful_routes': 0,
            'failed_routes': 0,
            'algorithm_switches': 0
        }
        
        # Performance tracking
        self.routing_history: deque = deque(maxlen=1000)
        self.performance_metrics: Dict[str, deque] = {
            'response_times': deque(maxlen=500),
            'throughput': deque(maxlen=500),
            'error_rates': deque(maxlen=500)
        }
        
        # Thread safety
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        logger.info(f"Intelligent load balancer initialized with {self.config.algorithm.value} algorithm")
    
    async def initialize(self) -> None:
        """Initialize async components."""
        await self.health_checker.initialize()
        
        if self.config.health_check_enabled:
            await self.start_health_monitoring()
    
    def add_node(self, node: LoadBalancerNode) -> None:
        """Add a node to the load balancer.
        
        Args:
            node: Node to add
        """
        with self.lock:
            self.nodes[node.node_id] = node
            logger.info(f"Added node {node.node_id} ({node.address}:{node.port})")
    
    def remove_node(self, node_id: str) -> bool:
        """Remove a node from the load balancer.
        
        Args:
            node_id: ID of node to remove
            
        Returns:
            True if node was removed
        """
        with self.lock:
            if node_id in self.nodes:
                del self.nodes[node_id]
                logger.info(f"Removed node {node_id}")
                return True
            return False
    
    def update_node_metrics(self, node_id: str, metrics: Dict[str, Any]) -> None:
        """Update node performance metrics.
        
        Args:
            node_id: Node ID
            metrics: Dictionary of metrics to update
        """
        with self.lock:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                
                for key, value in metrics.items():
                    if hasattr(node, key):
                        setattr(node, key, value)
                
                logger.debug(f"Updated metrics for node {node_id}")
    
    async def route_request(self, request: RoutingRequest) -> Optional[LoadBalancerNode]:
        """Route request to appropriate node.
        
        Args:
            request: Routing request
            
        Returns:
            Selected node or None if no nodes available
        """
        with self.lock:
            self.routing_stats['total_requests'] += 1
            
            try:
                # Check session affinity first
                if (self.config.enable_sticky_sessions and 
                    request.session_id and 
                    request.session_id in self.session_affinity):
                    
                    node_id = self.session_affinity[request.session_id]
                    if (node_id in self.nodes and 
                        self.nodes[node_id].state == NodeState.HEALTHY):
                        
                        selected_node = self.nodes[node_id]
                        self._record_routing(request, selected_node)
                        return selected_node
                    else:
                        # Remove stale session
                        del self.session_affinity[request.session_id]
                
                # Select algorithm
                algorithm = self._select_algorithm(request)
                
                # Route using selected algorithm
                selected_node = await self._route_with_algorithm(algorithm, request)
                
                if selected_node:
                    # Update session affinity if needed
                    if self.config.enable_sticky_sessions and request.session_id:
                        self.session_affinity[request.session_id] = selected_node.node_id
                        self.session_timeouts[request.session_id] = (
                            datetime.now() + timedelta(seconds=self.config.session_timeout)
                        )
                    
                    # Record successful routing
                    self._record_routing(request, selected_node)
                    self.routing_stats['successful_routes'] += 1
                    
                    logger.debug(f"Routed request {request.request_id} to node {selected_node.node_id}")
                    
                    return selected_node
                else:
                    self.routing_stats['failed_routes'] += 1
                    logger.warning(f"Failed to route request {request.request_id}")
                    return None
                    
            except Exception as e:
                logger.error(f"Error routing request {request.request_id}: {e}")
                self.routing_stats['failed_routes'] += 1
                return None
    
    def _select_algorithm(self, request: RoutingRequest) -> LoadBalancingAlgorithm:
        """Select appropriate load balancing algorithm.
        
        Args:
            request: Routing request
            
        Returns:
            Selected algorithm
        """
        if self.config.algorithm == LoadBalancingAlgorithm.ADAPTIVE:
            # Adaptive algorithm selection based on request characteristics
            
            if request.privacy_level == "high":
                return LoadBalancingAlgorithm.PRIVACY_AWARE
            
            if request.request_type == "training" and request.training_step is not None:
                return LoadBalancingAlgorithm.TRAINING_AWARE
            
            if self.config.enable_geographic_routing and request.client_region != "unknown":
                return LoadBalancingAlgorithm.GEOGRAPHIC
            
            # Default to least response time for general requests
            return LoadBalancingAlgorithm.LEAST_RESPONSE_TIME
        
        return self.config.algorithm
    
    async def _route_with_algorithm(
        self, 
        algorithm: LoadBalancingAlgorithm, 
        request: RoutingRequest
    ) -> Optional[LoadBalancerNode]:
        """Route request using specific algorithm.
        
        Args:
            algorithm: Algorithm to use
            request: Routing request
            
        Returns:
            Selected node or None
        """
        nodes = list(self.nodes.values())
        
        if not nodes:
            return None
        
        try:
            if algorithm == LoadBalancingAlgorithm.ROUND_ROBIN:
                return LoadBalancingAlgorithms.round_robin(nodes, self.algorithm_state)
            
            elif algorithm == LoadBalancingAlgorithm.LEAST_CONNECTIONS:
                return LoadBalancingAlgorithms.least_connections(nodes, self.algorithm_state)
            
            elif algorithm == LoadBalancingAlgorithm.WEIGHTED_ROUND_ROBIN:
                return LoadBalancingAlgorithms.weighted_round_robin(nodes, self.algorithm_state)
            
            elif algorithm == LoadBalancingAlgorithm.LEAST_RESPONSE_TIME:
                return LoadBalancingAlgorithms.least_response_time(nodes, self.algorithm_state)
            
            elif algorithm == LoadBalancingAlgorithm.IP_HASH:
                return LoadBalancingAlgorithms.ip_hash(nodes, self.algorithm_state, request.client_ip)
            
            elif algorithm == LoadBalancingAlgorithm.CONSISTENT_HASH:
                return LoadBalancingAlgorithms.consistent_hash(nodes, self.algorithm_state, request.client_ip)
            
            elif algorithm == LoadBalancingAlgorithm.PRIVACY_AWARE:
                return LoadBalancingAlgorithms.privacy_aware(nodes, self.algorithm_state, request)
            
            elif algorithm == LoadBalancingAlgorithm.TRAINING_AWARE:
                return LoadBalancingAlgorithms.training_aware(nodes, self.algorithm_state, request)
            
            elif algorithm == LoadBalancingAlgorithm.GEOGRAPHIC:
                return LoadBalancingAlgorithms.geographic(nodes, self.algorithm_state, request)
            
            else:
                # Fallback to round robin
                return LoadBalancingAlgorithms.round_robin(nodes, self.algorithm_state)
                
        except ValueError as e:
            logger.warning(f"Algorithm {algorithm.value} failed: {e}")
            # Try fallback algorithm
            try:
                return LoadBalancingAlgorithms.round_robin(nodes, self.algorithm_state)
            except:
                return None
    
    def _record_routing(self, request: RoutingRequest, selected_node: LoadBalancerNode) -> None:
        """Record routing decision for analytics."""
        routing_record = {
            'timestamp': datetime.now(),
            'request_id': request.request_id,
            'client_ip': request.client_ip,
            'selected_node': selected_node.node_id,
            'algorithm_used': self.config.algorithm.value,
            'node_load': selected_node.get_load_score(),
            'node_connections': selected_node.active_connections
        }
        
        self.routing_history.append(routing_record)
        
        # Update node connection count
        selected_node.active_connections += 1
    
    def record_request_completion(
        self, 
        request_id: str, 
        node_id: str, 
        success: bool, 
        response_time: float = 0.0
    ) -> None:
        """Record request completion.
        
        Args:
            request_id: Request ID
            node_id: Node that handled the request
            success: Whether request was successful
            response_time: Response time in seconds
        """
        with self.lock:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                node.record_request(success, response_time * 1000)  # Convert to ms
                node.active_connections = max(0, node.active_connections - 1)
                
                # Update performance metrics
                if success and response_time > 0:
                    self.performance_metrics['response_times'].append(response_time * 1000)
                
                logger.debug(f"Recorded completion for request {request_id} on node {node_id}")
    
    async def start_health_monitoring(self) -> None:
        """Start health monitoring for all nodes."""
        if self.health_monitor_active:
            return
        
        self.health_monitor_active = True
        self.health_monitor_task = asyncio.create_task(self._health_monitor_loop())
        
        logger.info("Started health monitoring")
    
    async def stop_health_monitoring(self) -> None:
        """Stop health monitoring."""
        if not self.health_monitor_active:
            return
        
        self.health_monitor_active = False
        if self.health_monitor_task:
            self.health_monitor_task.cancel()
            try:
                await self.health_monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped health monitoring")
    
    async def _health_monitor_loop(self) -> None:
        """Main health monitoring loop."""
        while self.health_monitor_active:
            try:
                # Check health of all nodes
                health_checks = []
                
                with self.lock:
                    nodes_to_check = list(self.nodes.values())
                
                for node in nodes_to_check:
                    if self.config.async_health_checks:
                        health_checks.append(self._check_and_update_node_health(node))
                    else:
                        await self._check_and_update_node_health(node)
                
                # Wait for all async health checks
                if health_checks:
                    await asyncio.gather(*health_checks, return_exceptions=True)
                
                # Clean up expired sessions
                self._cleanup_expired_sessions()
                
                await asyncio.sleep(self.config.health_check_interval)
                
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                await asyncio.sleep(self.config.health_check_interval * 2)
    
    async def _check_and_update_node_health(self, node: LoadBalancerNode) -> None:
        """Check and update health status of a node."""
        try:
            is_healthy = await self.health_checker.check_node_health(node)
            
            with self.lock:
                node.last_health_check = datetime.now()
                
                if is_healthy:
                    # Node is healthy
                    if node.state == NodeState.UNHEALTHY:
                        logger.info(f"Node {node.node_id} recovered")
                        node.state = NodeState.HEALTHY
                        node.health_check_failures = 0
                else:
                    # Node is unhealthy
                    node.health_check_failures += 1
                    
                    if (node.health_check_failures >= self.config.max_failures and 
                        node.state == NodeState.HEALTHY):
                        
                        logger.warning(f"Node {node.node_id} marked as unhealthy after {node.health_check_failures} failures")
                        node.state = NodeState.UNHEALTHY
                        
                        # Trigger alerts/notifications for unhealthy node
                        self._trigger_node_health_alert(node)
                
        except Exception as e:
            logger.error(f"Error checking health of node {node.node_id}: {e}")
    
    def _cleanup_expired_sessions(self) -> None:
        """Clean up expired session affinities."""
        if not self.config.enable_sticky_sessions:
            return
        
        current_time = datetime.now()
        expired_sessions = []
        
        for session_id, timeout_time in self.session_timeouts.items():
            if current_time > timeout_time:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            if session_id in self.session_affinity:
                del self.session_affinity[session_id]
            del self.session_timeouts[session_id]
        
        if expired_sessions:
            logger.debug(f"Cleaned up {len(expired_sessions)} expired sessions")
    
    def get_load_balancer_stats(self) -> Dict[str, Any]:
        """Get comprehensive load balancer statistics."""
        with self.lock:
            node_stats = {
                node_id: node.to_dict() 
                for node_id, node in self.nodes.items()
            }
            
            healthy_nodes = [n for n in self.nodes.values() if n.state == NodeState.HEALTHY]
            
            # Calculate aggregate metrics
            total_connections = sum(n.active_connections for n in self.nodes.values())
            avg_response_time = 0.0
            
            if self.performance_metrics['response_times']:
                avg_response_time = np.mean(list(self.performance_metrics['response_times']))
            
            success_rate = 0.0
            if self.routing_stats['total_requests'] > 0:
                success_rate = self.routing_stats['successful_routes'] / self.routing_stats['total_requests']
            
            return {
                'configuration': self.config.to_dict(),
                'routing_stats': self.routing_stats,
                'node_summary': {
                    'total_nodes': len(self.nodes),
                    'healthy_nodes': len(healthy_nodes),
                    'unhealthy_nodes': len(self.nodes) - len(healthy_nodes),
                    'total_connections': total_connections
                },
                'performance_metrics': {
                    'average_response_time_ms': avg_response_time,
                    'success_rate': success_rate,
                    'requests_per_second': len(self.routing_history) / 60.0 if self.routing_history else 0.0
                },
                'session_management': {
                    'active_sessions': len(self.session_affinity),
                    'sticky_sessions_enabled': self.config.enable_sticky_sessions
                },
                'nodes': node_stats
            }
    
    def export_load_balancer_report(self, output_path: str) -> None:
        """Export comprehensive load balancer report."""
        report = {
            'load_balancer_stats': self.get_load_balancer_stats(),
            'routing_history': [
                {
                    'timestamp': record['timestamp'].isoformat(),
                    'request_id': record['request_id'],
                    'selected_node': record['selected_node'],
                    'algorithm_used': record['algorithm_used'],
                    'node_load': record['node_load']
                }
                for record in list(self.routing_history)[-100:]
            ],
            'health_check_history': {
                node_id: self.health_checker.get_health_history(node_id)
                for node_id in self.nodes.keys()
            },
            'export_timestamp': datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Load balancer report exported to {output_path}")
    
    async def shutdown(self) -> None:
        """Shutdown load balancer."""
        logger.info("Shutting down load balancer...")
        
        await self.stop_health_monitoring()
        await self.health_checker.shutdown()
        self.executor.shutdown(wait=True)
        
        logger.info("Load balancer shutdown completed")


# Convenience functions and global instance
_global_load_balancer: Optional[IntelligentLoadBalancer] = None


def get_load_balancer(config: Optional[LoadBalancerConfig] = None) -> IntelligentLoadBalancer:
    """Get global load balancer instance."""
    global _global_load_balancer
    if _global_load_balancer is None:
        _global_load_balancer = IntelligentLoadBalancer(config)
    return _global_load_balancer


async def route_training_request(
    client_ip: str,
    training_step: Optional[int] = None,
    privacy_level: str = "standard",
    batch_size: int = 32,
    **kwargs
) -> Optional[LoadBalancerNode]:
    """Convenience function to route training request."""
    lb = get_load_balancer()
    
    request = RoutingRequest(
        request_id=f"train_{int(time.time())}_{random.randint(1000, 9999)}",
        client_ip=client_ip,
        request_type="training",
        training_step=training_step,
        privacy_level=privacy_level,
        batch_size=batch_size,
        **kwargs
    )
    
    return await lb.route_request(request)
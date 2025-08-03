"""Advanced query optimization and caching for privacy-preserving operations."""

import logging
import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from datetime import datetime, timedelta
from functools import wraps
from dataclasses import dataclass
from sqlalchemy.orm import Session, Query
from sqlalchemy import text, func, and_, or_
from sqlalchemy.sql import Select
import redis

logger = logging.getLogger(__name__)


@dataclass
class QueryMetrics:
    """Query performance metrics."""
    query_hash: str
    execution_time_ms: float
    rows_returned: int
    cache_hit: bool
    timestamp: datetime


class QueryOptimizer:
    """Advanced query optimization and caching system."""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        """Initialize query optimizer.
        
        Args:
            redis_client: Redis client for caching
        """
        self.redis_client = redis_client
        self.metrics: List[QueryMetrics] = []
        self.cache_ttl = 3600  # 1 hour default TTL
        self.max_cache_size = 1000
        
    def cache_query(
        self, 
        ttl: int = None,
        cache_key_prefix: str = "query_cache"
    ) -> Callable:
        """Decorator for caching query results.
        
        Args:
            ttl: Cache time-to-live in seconds
            cache_key_prefix: Prefix for cache keys
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                if not self.redis_client:
                    return func(*args, **kwargs)
                
                # Generate cache key from function name and arguments
                cache_key = self._generate_cache_key(
                    func.__name__, args, kwargs, cache_key_prefix
                )
                
                # Try to get from cache
                start_time = datetime.now()
                cached_result = self.redis_client.get(cache_key)
                
                if cached_result:
                    result = json.loads(cached_result)
                    self._record_metrics(
                        cache_key, 
                        (datetime.now() - start_time).total_seconds() * 1000,
                        len(result) if isinstance(result, list) else 1,
                        cache_hit=True
                    )
                    logger.debug(f"Cache hit for key: {cache_key}")
                    return result
                
                # Execute query and cache result
                result = func(*args, **kwargs)
                execution_time = (datetime.now() - start_time).total_seconds() * 1000
                
                # Cache the result
                cache_value = json.dumps(result, default=str, separators=(',', ':'))
                actual_ttl = ttl or self.cache_ttl
                self.redis_client.setex(cache_key, actual_ttl, cache_value)
                
                self._record_metrics(
                    cache_key,
                    execution_time,
                    len(result) if isinstance(result, list) else 1,
                    cache_hit=False
                )
                
                logger.debug(f"Query cached with key: {cache_key}")
                return result
            
            return wrapper
        return decorator
    
    def optimize_query(self, query: Query) -> Query:
        """Optimize SQLAlchemy query for better performance.
        
        Args:
            query: SQLAlchemy Query object
            
        Returns:
            Optimized query
        """
        # Add query hints and optimizations
        optimized = query
        
        # Add index hints for common patterns
        if hasattr(query.column_descriptions[0]['type'], '__tablename__'):
            table_name = query.column_descriptions[0]['type'].__tablename__
            
            # Add specific optimizations based on table
            if table_name == 'training_jobs':
                # Optimize training job queries with status and user indexes
                optimized = query.with_hint(
                    query.column_descriptions[0]['type'],
                    'USE INDEX (idx_training_job_user_status)'
                )
            elif table_name == 'privacy_budget_entries':
                # Optimize privacy budget queries
                optimized = query.with_hint(
                    query.column_descriptions[0]['type'],
                    'USE INDEX (idx_privacy_entry_job_step)'
                )
        
        return optimized
    
    def batch_query(
        self, 
        session: Session, 
        queries: List[Tuple[str, Dict[str, Any]]]
    ) -> List[Any]:
        """Execute multiple queries in a single database round trip.
        
        Args:
            session: Database session
            queries: List of (query_string, parameters) tuples
            
        Returns:
            List of query results
        """
        results = []
        
        try:
            # Build batch query using UNION ALL where possible
            if self._can_batch_queries(queries):
                batch_sql = self._build_batch_query(queries)
                result = session.execute(text(batch_sql))
                results = self._parse_batch_results(result, len(queries))
            else:
                # Execute individually but in same transaction
                for query_sql, params in queries:
                    result = session.execute(text(query_sql), params)
                    results.append(result.fetchall())
        
        except Exception as e:
            logger.error(f"Batch query execution failed: {e}")
            # Fallback to individual execution
            for query_sql, params in queries:
                try:
                    result = session.execute(text(query_sql), params)
                    results.append(result.fetchall())
                except Exception as individual_error:
                    logger.error(f"Individual query failed: {individual_error}")
                    results.append([])
        
        return results
    
    def explain_query(self, session: Session, query: Union[Query, str]) -> Dict[str, Any]:
        """Analyze query execution plan.
        
        Args:
            session: Database session
            query: SQLAlchemy Query or raw SQL string
            
        Returns:
            Query execution plan analysis
        """
        if isinstance(query, Query):
            query_sql = str(query.statement.compile(compile_kwargs={"literal_binds": True}))
        else:
            query_sql = query
        
        try:
            # Get query execution plan
            explain_result = session.execute(text(f"EXPLAIN (ANALYZE, BUFFERS) {query_sql}"))
            plan_lines = [row[0] for row in explain_result.fetchall()]
            
            # Parse execution plan
            analysis = self._parse_execution_plan(plan_lines)
            
            return {
                "query": query_sql,
                "execution_plan": plan_lines,
                "analysis": analysis,
                "recommendations": self._generate_query_recommendations(analysis)
            }
        
        except Exception as e:
            logger.error(f"Query explanation failed: {e}")
            return {"error": str(e), "query": query_sql}
    
    def get_slow_queries(self, threshold_ms: float = 1000) -> List[QueryMetrics]:
        """Get queries that exceeded performance threshold.
        
        Args:
            threshold_ms: Execution time threshold in milliseconds
            
        Returns:
            List of slow query metrics
        """
        return [
            metric for metric in self.metrics 
            if metric.execution_time_ms > threshold_ms and not metric.cache_hit
        ]
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        if not self.redis_client:
            return {"cache_enabled": False}
        
        try:
            info = self.redis_client.info()
            total_queries = len(self.metrics)
            cache_hits = sum(1 for m in self.metrics if m.cache_hit)
            
            return {
                "cache_enabled": True,
                "total_queries": total_queries,
                "cache_hits": cache_hits,
                "cache_hit_rate": (cache_hits / total_queries * 100) if total_queries > 0 else 0,
                "redis_memory_usage": info.get("used_memory_human", "unknown"),
                "redis_connected_clients": info.get("connected_clients", 0),
                "average_execution_time_ms": sum(m.execution_time_ms for m in self.metrics) / total_queries if total_queries > 0 else 0
            }
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {"cache_enabled": True, "error": str(e)}
    
    def clear_cache(self, pattern: str = "query_cache:*") -> int:
        """Clear cached queries matching pattern.
        
        Args:
            pattern: Redis key pattern to match
            
        Returns:
            Number of keys deleted
        """
        if not self.redis_client:
            return 0
        
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                return self.redis_client.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return 0
    
    def _generate_cache_key(
        self, 
        func_name: str, 
        args: tuple, 
        kwargs: dict,
        prefix: str
    ) -> str:
        """Generate deterministic cache key."""
        # Create a deterministic hash of function name and arguments
        key_data = {
            "function": func_name,
            "args": str(args),
            "kwargs": sorted(kwargs.items())
        }
        
        key_string = json.dumps(key_data, sort_keys=True, separators=(',', ':'))
        key_hash = hashlib.sha256(key_string.encode()).hexdigest()[:16]
        
        return f"{prefix}:{func_name}:{key_hash}"
    
    def _record_metrics(
        self, 
        query_hash: str, 
        execution_time: float, 
        rows_returned: int,
        cache_hit: bool
    ) -> None:
        """Record query performance metrics."""
        metric = QueryMetrics(
            query_hash=query_hash,
            execution_time_ms=execution_time,
            rows_returned=rows_returned,
            cache_hit=cache_hit,
            timestamp=datetime.now()
        )
        
        self.metrics.append(metric)
        
        # Keep only recent metrics to prevent memory bloat
        if len(self.metrics) > 10000:
            self.metrics = self.metrics[-5000:]  # Keep last 5000 metrics
    
    def _can_batch_queries(self, queries: List[Tuple[str, Dict[str, Any]]]) -> bool:
        """Check if queries can be batched together."""
        # Simple check - all queries should be SELECT statements
        return all(
            query_sql.strip().upper().startswith('SELECT') 
            for query_sql, _ in queries
        )
    
    def _build_batch_query(self, queries: List[Tuple[str, Dict[str, Any]]]) -> str:
        """Build a batched query using UNION ALL."""
        # This is a simplified implementation
        # In practice, you'd need more sophisticated logic
        union_parts = []
        for i, (query_sql, params) in enumerate(queries):
            # Add a query identifier to distinguish results
            modified_query = f"SELECT {i} as _query_id, * FROM ({query_sql}) as q{i}"
            union_parts.append(modified_query)
        
        return " UNION ALL ".join(union_parts)
    
    def _parse_batch_results(self, result: Any, num_queries: int) -> List[Any]:
        """Parse results from a batched query."""
        # Group results by query_id
        grouped_results = {}
        for row in result.fetchall():
            query_id = row[0]  # First column is _query_id
            actual_row = row[1:]  # Rest of the data
            
            if query_id not in grouped_results:
                grouped_results[query_id] = []
            grouped_results[query_id].append(actual_row)
        
        # Return results in original order
        return [grouped_results.get(i, []) for i in range(num_queries)]
    
    def _parse_execution_plan(self, plan_lines: List[str]) -> Dict[str, Any]:
        """Parse PostgreSQL execution plan."""
        analysis = {
            "total_cost": 0.0,
            "execution_time_ms": 0.0,
            "rows_returned": 0,
            "sequential_scans": 0,
            "index_scans": 0,
            "sorts": 0,
            "joins": 0
        }
        
        for line in plan_lines:
            line = line.strip()
            
            # Extract cost and timing information
            if "cost=" in line:
                try:
                    cost_part = line.split("cost=")[1].split()[0]
                    if ".." in cost_part:
                        analysis["total_cost"] = float(cost_part.split("..")[1])
                except (IndexError, ValueError):
                    pass
            
            # Extract execution time
            if "actual time=" in line:
                try:
                    time_part = line.split("actual time=")[1].split()[0]
                    if ".." in time_part:
                        analysis["execution_time_ms"] = float(time_part.split("..")[1])
                except (IndexError, ValueError):
                    pass
            
            # Count different operation types
            if "Seq Scan" in line:
                analysis["sequential_scans"] += 1
            elif "Index Scan" in line or "Index Only Scan" in line:
                analysis["index_scans"] += 1
            elif "Sort" in line:
                analysis["sorts"] += 1
            elif "Join" in line:
                analysis["joins"] += 1
        
        return analysis
    
    def _generate_query_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations based on execution plan."""
        recommendations = []
        
        if analysis["sequential_scans"] > 0:
            recommendations.append(
                f"Consider adding indexes to reduce {analysis['sequential_scans']} sequential scan(s)"
            )
        
        if analysis["execution_time_ms"] > 1000:
            recommendations.append(
                "Query execution time is high - consider query optimization or result caching"
            )
        
        if analysis["sorts"] > 2:
            recommendations.append(
                f"Multiple sorts detected ({analysis['sorts']}) - consider adding composite indexes"
            )
        
        if analysis["total_cost"] > 10000:
            recommendations.append(
                "High query cost detected - review query structure and data access patterns"
            )
        
        if not recommendations:
            recommendations.append("Query appears to be well-optimized")
        
        return recommendations


class PrivacyQueryMixin:
    """Mixin for privacy-aware database queries."""
    
    @staticmethod
    def apply_privacy_filters(
        query: Query, 
        user_id: Optional[str] = None,
        privacy_level: Optional[str] = None,
        data_classification: Optional[str] = None
    ) -> Query:
        """Apply privacy-based filtering to queries.
        
        Args:
            query: SQLAlchemy Query object
            user_id: User ID for ownership filtering
            privacy_level: Minimum privacy level required
            data_classification: Required data classification level
            
        Returns:
            Query with privacy filters applied
        """
        # Apply user ownership filter
        if user_id and hasattr(query.column_descriptions[0]['type'], 'owner_id'):
            query = query.filter(query.column_descriptions[0]['type'].owner_id == user_id)
        
        # Apply privacy level filter
        if privacy_level and hasattr(query.column_descriptions[0]['type'], 'privacy_level'):
            privacy_hierarchy = {"low": 1, "medium": 2, "high": 3}
            min_level = privacy_hierarchy.get(privacy_level, 1)
            
            query = query.filter(
                func.case(
                    [(query.column_descriptions[0]['type'].privacy_level == "low", 1),
                     (query.column_descriptions[0]['type'].privacy_level == "medium", 2),
                     (query.column_descriptions[0]['type'].privacy_level == "high", 3)],
                    else_=1
                ) >= min_level
            )
        
        # Apply data classification filter
        if data_classification and hasattr(query.column_descriptions[0]['type'], 'data_classification'):
            classification_hierarchy = {
                "public": 1, "internal": 2, "confidential": 3, "restricted": 4
            }
            max_level = classification_hierarchy.get(data_classification, 4)
            
            query = query.filter(
                func.case(
                    [(query.column_descriptions[0]['type'].data_classification == "public", 1),
                     (query.column_descriptions[0]['type'].data_classification == "internal", 2),
                     (query.column_descriptions[0]['type'].data_classification == "confidential", 3),
                     (query.column_descriptions[0]['type'].data_classification == "restricted", 4)],
                    else_=1
                ) <= max_level
            )
        
        return query
    
    @staticmethod
    def anonymize_results(results: List[Any], fields_to_anonymize: List[str]) -> List[Any]:
        """Anonymize sensitive fields in query results.
        
        Args:
            results: Query results
            fields_to_anonymize: List of field names to anonymize
            
        Returns:
            Results with specified fields anonymized
        """
        anonymized_results = []
        
        for result in results:
            if hasattr(result, '_asdict'):
                result_dict = result._asdict()
                for field in fields_to_anonymize:
                    if field in result_dict:
                        # Simple anonymization - replace with hash
                        original_value = str(result_dict[field])
                        hashed_value = hashlib.sha256(original_value.encode()).hexdigest()[:8]
                        result_dict[field] = f"***{hashed_value}"
                anonymized_results.append(result_dict)
            else:
                # Handle ORM objects
                anonymized_result = result
                for field in fields_to_anonymize:
                    if hasattr(result, field):
                        original_value = str(getattr(result, field))
                        hashed_value = hashlib.sha256(original_value.encode()).hexdigest()[:8]
                        setattr(anonymized_result, field, f"***{hashed_value}")
                anonymized_results.append(anonymized_result)
        
        return anonymized_results
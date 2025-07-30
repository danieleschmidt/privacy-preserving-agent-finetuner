"""Privacy budget monitoring and compliance tracking."""

from typing import Dict, List, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class PrivacyEvent:
    """Individual privacy budget consumption event."""
    timestamp: datetime
    epsilon_spent: float
    delta: float
    operation: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class PrivacyBudgetMonitor:
    """Monitor and track privacy budget consumption over time.
    
    Provides real-time monitoring of differential privacy budget usage
    with alerting and compliance reporting capabilities.
    """
    
    def __init__(self, total_epsilon: float, total_delta: float):
        """Initialize privacy budget monitor.
        
        Args:
            total_epsilon: Total privacy budget (epsilon)
            total_delta: Total privacy parameter (delta)
        """
        self.total_epsilon = total_epsilon
        self.total_delta = total_delta
        self.events: List[PrivacyEvent] = []
        self.alert_threshold = 0.8  # Alert when 80% budget consumed
        
        logger.info(f"Initialized privacy monitor: ε={total_epsilon}, δ={total_delta}")
    
    def record_event(self, epsilon_spent: float, delta: float, operation: str, **metadata):
        """Record a privacy budget consumption event.
        
        Args:
            epsilon_spent: Amount of epsilon budget consumed
            delta: Delta parameter for this operation
            operation: Description of the operation
            **metadata: Additional event metadata
        """
        event = PrivacyEvent(
            timestamp=datetime.now(),
            epsilon_spent=epsilon_spent,
            delta=delta,
            operation=operation,
            metadata=metadata
        )
        
        self.events.append(event)
        
        # Check if alert threshold exceeded
        total_spent = self.get_total_spent()
        if total_spent / self.total_epsilon > self.alert_threshold:
            self._trigger_budget_alert(total_spent)
        
        logger.info(f"Privacy event recorded: {operation} (ε={epsilon_spent})")
    
    def get_total_spent(self) -> float:
        """Calculate total privacy budget spent."""
        return sum(event.epsilon_spent for event in self.events)
    
    def get_remaining_budget(self) -> float:
        """Calculate remaining privacy budget."""
        return max(0, self.total_epsilon - self.get_total_spent())
    
    def is_budget_exhausted(self) -> bool:
        """Check if privacy budget is exhausted."""
        return self.get_remaining_budget() <= 0
    
    def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate comprehensive privacy compliance report."""
        total_spent = self.get_total_spent()
        
        return {
            "report_timestamp": datetime.now().isoformat(),
            "total_budget": self.total_epsilon,
            "total_spent": total_spent,
            "remaining_budget": self.get_remaining_budget(),
            "budget_utilization": total_spent / self.total_epsilon,
            "total_operations": len(self.events),
            "compliance_status": "compliant" if not self.is_budget_exhausted() else "exceeded",
            "recent_events": [
                {
                    "timestamp": event.timestamp.isoformat(),
                    "operation": event.operation,
                    "epsilon_spent": event.epsilon_spent
                }
                for event in self.events[-10:]  # Last 10 events
            ]
        }
    
    def _trigger_budget_alert(self, current_spent: float):
        """Trigger alert when budget threshold exceeded."""
        utilization = current_spent / self.total_epsilon
        logger.warning(f"Privacy budget alert: {utilization:.1%} consumed ({current_spent:.3f}/{self.total_epsilon})")
        
        # TODO: Implement actual alerting (Slack, email, etc.)
        # This could integrate with monitoring systems like Prometheus
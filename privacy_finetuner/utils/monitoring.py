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
        
        # Implement actual alerting integration
        self._send_alert_notification(utilization, current_spent)
    
    def _send_alert_notification(self, utilization: float, current_spent: float):
        """Send alert notification through configured channels."""
        alert_message = f"Privacy Budget Alert: {utilization:.1%} consumed ({current_spent:.3f}/{self.total_epsilon})"
        
        # Log alert
        logger.warning(alert_message)
        
        # Slack webhook integration
        self._send_slack_notification(alert_message, utilization)
        
        # Email notification integration
        self._send_email_notification(alert_message, utilization)
        
        # PagerDuty alert integration
        self._send_pagerduty_alert(alert_message, utilization)
        
        # Prometheus alertmanager integration
        self._send_prometheus_alert(utilization, current_spent)
        
        # Update Prometheus metrics
        try:
            from prometheus_client import Counter, Gauge
            privacy_alerts = Counter('privacy_budget_alerts_total', 'Privacy budget alerts')
            privacy_budget_utilization = Gauge('privacy_budget_utilization', 'Privacy budget utilization ratio')
            
            privacy_alerts.inc()
            privacy_budget_utilization.set(utilization)
        except ImportError:
            pass
    
    def _send_slack_notification(self, message: str, utilization: float):
        """Send alert to Slack webhook."""
        try:
            import requests
            import os
            
            webhook_url = os.getenv('SLACK_WEBHOOK_URL')
            if not webhook_url:
                return
            
            payload = {
                "text": f"🚨 {message}",
                "attachments": [{
                    "color": "danger" if utilization > 0.9 else "warning",
                    "fields": [
                        {"title": "Utilization", "value": f"{utilization:.1%}", "short": True},
                        {"title": "Timestamp", "value": datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "short": True}
                    ]
                }]
            }
            
            response = requests.post(webhook_url, json=payload, timeout=10)
            if response.status_code == 200:
                logger.info("Slack notification sent successfully")
            else:
                logger.error(f"Failed to send Slack notification: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error sending Slack notification: {e}")
    
    def _send_email_notification(self, message: str, utilization: float):
        """Send alert via email."""
        try:
            import smtplib
            import os
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            smtp_host = os.getenv('SMTP_HOST')
            smtp_port = int(os.getenv('SMTP_PORT', '587'))
            smtp_user = os.getenv('SMTP_USER')
            smtp_password = os.getenv('SMTP_PASSWORD')
            alert_emails = os.getenv('ALERT_EMAILS', '').split(',')
            
            if not all([smtp_host, smtp_user, smtp_password, alert_emails[0]]):
                return
            
            msg = MIMEMultipart()
            msg['From'] = smtp_user
            msg['To'] = ', '.join(alert_emails)
            msg['Subject'] = f"Privacy Budget Alert - {utilization:.1%} Consumed"
            
            body = f"""
Privacy Budget Alert

{message}

Details:
- Total Budget: {self.total_epsilon}
- Current Utilization: {utilization:.1%}
- Remaining Budget: {self.get_remaining_budget():.3f}
- Alert Timestamp: {datetime.now().isoformat()}

Please review privacy budget usage and take appropriate action if necessary.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP(smtp_host, smtp_port) as server:
                server.starttls()
                server.login(smtp_user, smtp_password)
                server.sendmail(smtp_user, alert_emails, msg.as_string())
                
            logger.info("Email notification sent successfully")
            
        except Exception as e:
            logger.error(f"Error sending email notification: {e}")
    
    def _send_pagerduty_alert(self, message: str, utilization: float):
        """Send alert to PagerDuty."""
        try:
            import requests
            import os
            
            integration_key = os.getenv('PAGERDUTY_INTEGRATION_KEY')
            if not integration_key:
                return
            
            severity = "critical" if utilization > 0.95 else "warning"
            
            payload = {
                "routing_key": integration_key,
                "event_action": "trigger",
                "dedup_key": f"privacy-budget-{datetime.now().strftime('%Y%m%d')}",
                "payload": {
                    "summary": message,
                    "source": "privacy-finetuner",
                    "severity": severity,
                    "component": "privacy-budget-monitor",
                    "group": "privacy-compliance",
                    "class": "budget-alert",
                    "custom_details": {
                        "utilization": utilization,
                        "total_budget": self.total_epsilon,
                        "remaining_budget": self.get_remaining_budget(),
                        "total_operations": len(self.events)
                    }
                }
            }
            
            response = requests.post(
                'https://events.pagerduty.com/v2/enqueue',
                json=payload,
                timeout=10
            )
            
            if response.status_code == 202:
                logger.info("PagerDuty alert sent successfully")
            else:
                logger.error(f"Failed to send PagerDuty alert: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error sending PagerDuty alert: {e}")
    
    def _send_prometheus_alert(self, utilization: float, current_spent: float):
        """Send alert to Prometheus Alertmanager."""
        try:
            import requests
            import os
            
            alertmanager_url = os.getenv('ALERTMANAGER_URL')
            if not alertmanager_url:
                return
            
            alert_payload = [{
                "labels": {
                    "alertname": "PrivacyBudgetThresholdExceeded",
                    "service": "privacy-finetuner",
                    "severity": "critical" if utilization > 0.95 else "warning",
                    "component": "privacy-budget-monitor"
                },
                "annotations": {
                    "summary": f"Privacy budget {utilization:.1%} consumed",
                    "description": f"Privacy budget utilization is {utilization:.1%} ({current_spent:.3f}/{self.total_epsilon})",
                    "runbook_url": "https://docs.privacy-finetuner.com/runbooks/privacy-budget"
                },
                "generatorURL": "http://privacy-finetuner/metrics"
            }]
            
            response = requests.post(
                f"{alertmanager_url}/api/v1/alerts",
                json=alert_payload,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info("Prometheus alert sent successfully")
            else:
                logger.error(f"Failed to send Prometheus alert: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error sending Prometheus alert: {e}")
    
    def export_events_json(self) -> str:
        """Export privacy events as JSON for audit purposes."""
        import json
        
        events_data = []
        for event in self.events:
            events_data.append({
                "timestamp": event.timestamp.isoformat(),
                "epsilon_spent": event.epsilon_spent,
                "delta": event.delta,
                "operation": event.operation,
                "metadata": event.metadata
            })
        
        return json.dumps({
            "export_timestamp": datetime.now().isoformat(),
            "total_events": len(events_data),
            "budget_summary": self.generate_compliance_report(),
            "events": events_data
        }, indent=2)
    
    def get_usage_by_operation(self) -> Dict[str, Dict[str, float]]:
        """Get privacy budget usage grouped by operation type."""
        operation_usage = {}
        
        for event in self.events:
            if event.operation not in operation_usage:
                operation_usage[event.operation] = {
                    "total_epsilon": 0.0,
                    "event_count": 0,
                    "avg_epsilon": 0.0
                }
            
            operation_usage[event.operation]["total_epsilon"] += event.epsilon_spent
            operation_usage[event.operation]["event_count"] += 1
        
        # Calculate averages
        for operation in operation_usage:
            data = operation_usage[operation]
            data["avg_epsilon"] = data["total_epsilon"] / data["event_count"]
        
        return operation_usage
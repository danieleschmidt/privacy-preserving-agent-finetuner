"""Notification services for email and Slack integration."""

import asyncio
import logging
from typing import Optional, Dict, List, Any, Union
from dataclasses import dataclass
from enum import Enum
import smtplib
import aiohttp
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class NotificationPriority(Enum):
    """Notification priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class NotificationChannel(Enum):
    """Available notification channels."""
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    SMS = "sms"


@dataclass
class NotificationMessage:
    """Notification message structure."""
    title: str
    content: str
    priority: NotificationPriority = NotificationPriority.NORMAL
    channels: List[NotificationChannel] = None
    metadata: Dict[str, Any] = None
    attachments: List[Dict[str, Any]] = None
    recipients: List[str] = None
    
    def __post_init__(self):
        if self.channels is None:
            self.channels = [NotificationChannel.EMAIL]
        if self.metadata is None:
            self.metadata = {}
        if self.attachments is None:
            self.attachments = []
        if self.recipients is None:
            self.recipients = []


class NotificationService:
    """Central notification service coordinator."""
    
    def __init__(self):
        """Initialize notification service."""
        self.providers = {}
        self.default_recipients = {}
        self.rate_limits = {}
        
    def register_provider(
        self,
        channel: NotificationChannel,
        provider: 'BaseNotificationProvider'
    ):
        """Register a notification provider.
        
        Args:
            channel: Notification channel
            provider: Provider instance
        """
        self.providers[channel] = provider
        logger.info(f"Registered {channel.value} notification provider")
        
    def set_default_recipients(
        self,
        channel: NotificationChannel,
        recipients: List[str]
    ):
        """Set default recipients for a channel.
        
        Args:
            channel: Notification channel
            recipients: List of recipient addresses
        """
        self.default_recipients[channel] = recipients
        
    async def send_notification(
        self,
        message: NotificationMessage,
        use_defaults: bool = True
    ) -> Dict[NotificationChannel, bool]:
        """Send notification through specified channels.
        
        Args:
            message: Notification message
            use_defaults: Whether to use default recipients
            
        Returns:
            Success status for each channel
        """
        results = {}
        
        for channel in message.channels:
            if channel not in self.providers:
                logger.warning(f"No provider registered for {channel.value}")
                results[channel] = False
                continue
                
            try:
                # Use default recipients if none specified
                recipients = message.recipients
                if use_defaults and not recipients:
                    recipients = self.default_recipients.get(channel, [])
                
                if not recipients:
                    logger.warning(f"No recipients configured for {channel.value}")
                    results[channel] = False
                    continue
                    
                # Apply rate limiting if configured
                if await self._check_rate_limit(channel):
                    provider = self.providers[channel]
                    success = await provider.send_message(message, recipients)
                    results[channel] = success
                    
                    if success:
                        logger.info(f"Successfully sent {channel.value} notification")
                    else:
                        logger.error(f"Failed to send {channel.value} notification")
                else:
                    logger.warning(f"Rate limit exceeded for {channel.value}")
                    results[channel] = False
                    
            except Exception as e:
                logger.error(f"Error sending {channel.value} notification: {e}")
                results[channel] = False
                
        return results
        
    async def send_privacy_alert(
        self,
        epsilon_spent: float,
        budget_remaining: float,
        threshold_exceeded: bool = False
    ):
        """Send privacy budget alert.
        
        Args:
            epsilon_spent: Privacy budget spent
            budget_remaining: Remaining budget
            threshold_exceeded: Whether threshold was exceeded
        """
        priority = NotificationPriority.CRITICAL if threshold_exceeded else NotificationPriority.HIGH
        
        title = "Privacy Budget Alert" if threshold_exceeded else "Privacy Budget Update"
        content = f"""
Privacy Budget Status:
- Epsilon Spent: {epsilon_spent:.4f}
- Budget Remaining: {budget_remaining:.4f}
- Utilization: {(epsilon_spent / (epsilon_spent + budget_remaining)) * 100:.1f}%

{'⚠️ THRESHOLD EXCEEDED - Immediate attention required!' if threshold_exceeded else 'Budget update for monitoring purposes.'}

Timestamp: {datetime.now(timezone.utc).isoformat()}
        """.strip()
        
        message = NotificationMessage(
            title=title,
            content=content,
            priority=priority,
            channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK],
            metadata={
                "epsilon_spent": epsilon_spent,
                "budget_remaining": budget_remaining,
                "threshold_exceeded": threshold_exceeded,
                "alert_type": "privacy_budget"
            }
        )
        
        await self.send_notification(message)
        
    async def send_training_completion(
        self,
        job_name: str,
        status: str,
        privacy_spent: float,
        model_path: str,
        metrics: Dict[str, float]
    ):
        """Send training completion notification.
        
        Args:
            job_name: Training job name
            status: Completion status
            privacy_spent: Privacy budget consumed
            model_path: Path to trained model
            metrics: Training metrics
        """
        emoji = "✅" if status == "completed" else "❌"
        
        title = f"{emoji} Training Job {status.title()}: {job_name}"
        content = f"""
Training Job Details:
- Job Name: {job_name}
- Status: {status}
- Privacy Spent: {privacy_spent:.4f} epsilon
- Model Path: {model_path}

Metrics:
""" + "\n".join([f"- {k}: {v}" for k, v in metrics.items()])
        
        message = NotificationMessage(
            title=title,
            content=content,
            priority=NotificationPriority.NORMAL,
            channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK],
            metadata={
                "job_name": job_name,
                "status": status,
                "privacy_spent": privacy_spent,
                "alert_type": "training_completion"
            }
        )
        
        await self.send_notification(message)
        
    async def send_security_alert(
        self,
        event_type: str,
        severity: str,
        details: Dict[str, Any]
    ):
        """Send security alert notification.
        
        Args:
            event_type: Type of security event
            severity: Severity level
            details: Event details
        """
        priority_map = {
            "low": NotificationPriority.LOW,
            "medium": NotificationPriority.NORMAL,
            "high": NotificationPriority.HIGH,
            "critical": NotificationPriority.CRITICAL
        }
        
        priority = priority_map.get(severity.lower(), NotificationPriority.NORMAL)
        
        title = f"🚨 Security Alert: {event_type}"
        content = f"""
Security Event Detected:
- Type: {event_type}
- Severity: {severity}
- Timestamp: {datetime.now(timezone.utc).isoformat()}

Details:
""" + "\n".join([f"- {k}: {v}" for k, v in details.items()])
        
        message = NotificationMessage(
            title=title,
            content=content,
            priority=priority,
            channels=[NotificationChannel.EMAIL, NotificationChannel.SLACK],
            metadata={
                "event_type": event_type,
                "severity": severity,
                "alert_type": "security",
                **details
            }
        )
        
        await self.send_notification(message)
        
    async def _check_rate_limit(self, channel: NotificationChannel) -> bool:
        """Check if rate limit allows sending.
        
        Args:
            channel: Notification channel
            
        Returns:
            True if sending is allowed
        """
        # Simple rate limiting implementation
        # In production, use Redis or similar for distributed rate limiting
        current_time = datetime.now(timezone.utc).timestamp()
        
        if channel not in self.rate_limits:
            self.rate_limits[channel] = {"count": 0, "window_start": current_time}
            
        rate_limit = self.rate_limits[channel]
        
        # Reset window if 1 hour has passed
        if current_time - rate_limit["window_start"] > 3600:
            rate_limit["count"] = 0
            rate_limit["window_start"] = current_time
            
        # Allow up to 100 notifications per hour per channel
        if rate_limit["count"] >= 100:
            return False
            
        rate_limit["count"] += 1
        return True


class BaseNotificationProvider:
    """Base class for notification providers."""
    
    async def send_message(
        self,
        message: NotificationMessage,
        recipients: List[str]
    ) -> bool:
        """Send notification message.
        
        Args:
            message: Notification message
            recipients: List of recipients
            
        Returns:
            True if successful
        """
        raise NotImplementedError


class EmailNotification(BaseNotificationProvider):
    """Email notification provider."""
    
    def __init__(
        self,
        smtp_host: str,
        smtp_port: int = 587,
        username: Optional[str] = None,
        password: Optional[str] = None,
        use_tls: bool = True,
        sender_email: Optional[str] = None,
        sender_name: str = "Privacy Finetuner"
    ):
        """Initialize email provider.
        
        Args:
            smtp_host: SMTP server host
            smtp_port: SMTP server port
            username: SMTP username
            password: SMTP password
            use_tls: Whether to use TLS
            sender_email: Sender email address
            sender_name: Sender display name
        """
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.use_tls = use_tls
        self.sender_email = sender_email or username
        self.sender_name = sender_name
        
    async def send_message(
        self,
        message: NotificationMessage,
        recipients: List[str]
    ) -> bool:
        """Send email notification.
        
        Args:
            message: Notification message
            recipients: Email recipients
            
        Returns:
            True if successful
        """
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = f"{self.sender_name} <{self.sender_email}>"
            msg['To'] = ", ".join(recipients)
            msg['Subject'] = message.title
            
            # Add priority headers
            if message.priority == NotificationPriority.CRITICAL:
                msg['X-Priority'] = '1'
                msg['X-MSMail-Priority'] = 'High'
            elif message.priority == NotificationPriority.HIGH:
                msg['X-Priority'] = '2'
                msg['X-MSMail-Priority'] = 'High'
                
            # Add body
            body = MIMEText(message.content, 'plain', 'utf-8')
            msg.attach(body)
            
            # Add attachments
            for attachment in message.attachments:
                if 'content' in attachment and 'filename' in attachment:
                    part = MIMEBase('application', 'octet-stream')
                    part.set_payload(attachment['content'])
                    encoders.encode_base64(part)
                    part.add_header(
                        'Content-Disposition',
                        f'attachment; filename= {attachment["filename"]}'
                    )
                    msg.attach(part)
                    
            # Send email
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                    
                if self.username and self.password:
                    server.login(self.username, self.password)
                    
                server.sendmail(self.sender_email, recipients, msg.as_string())
                
            logger.info(f"Email sent successfully to {len(recipients)} recipients")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
            return False


class SlackNotification(BaseNotificationProvider):
    """Slack notification provider."""
    
    def __init__(
        self,
        webhook_url: Optional[str] = None,
        bot_token: Optional[str] = None,
        default_channel: str = "#alerts"
    ):
        """Initialize Slack provider.
        
        Args:
            webhook_url: Slack webhook URL
            bot_token: Slack bot token
            default_channel: Default channel for notifications
        """
        self.webhook_url = webhook_url
        self.bot_token = bot_token
        self.default_channel = default_channel
        
        if not webhook_url and not bot_token:
            raise ValueError("Either webhook_url or bot_token must be provided")
            
    async def send_message(
        self,
        message: NotificationMessage,
        recipients: List[str]
    ) -> bool:
        """Send Slack notification.
        
        Args:
            message: Notification message
            recipients: Slack channels or user IDs
            
        Returns:
            True if successful
        """
        try:
            # Choose appropriate sending method
            if self.webhook_url:
                return await self._send_via_webhook(message, recipients)
            else:
                return await self._send_via_api(message, recipients)
                
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
            return False
            
    async def _send_via_webhook(
        self,
        message: NotificationMessage,
        recipients: List[str]
    ) -> bool:
        """Send notification via webhook."""
        # Priority emoji mapping
        priority_emoji = {
            NotificationPriority.LOW: "ℹ️",
            NotificationPriority.NORMAL: "📢", 
            NotificationPriority.HIGH: "⚠️",
            NotificationPriority.CRITICAL: "🚨"
        }
        
        emoji = priority_emoji.get(message.priority, "📢")
        
        # Build Slack message
        slack_message = {
            "text": f"{emoji} {message.title}",
            "attachments": [
                {
                    "color": self._get_color_for_priority(message.priority),
                    "fields": [
                        {
                            "title": "Details",
                            "value": message.content,
                            "short": False
                        }
                    ],
                    "footer": "Privacy Finetuner",
                    "ts": int(datetime.now(timezone.utc).timestamp())
                }
            ]
        }
        
        # Add metadata as fields
        if message.metadata:
            for key, value in message.metadata.items():
                slack_message["attachments"][0]["fields"].append({
                    "title": key.replace("_", " ").title(),
                    "value": str(value),
                    "short": True
                })
                
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.webhook_url,
                json=slack_message,
                headers={"Content-Type": "application/json"}
            ) as response:
                success = response.status == 200
                if not success:
                    logger.error(f"Slack webhook failed: {response.status}")
                return success
                
    async def _send_via_api(
        self,
        message: NotificationMessage,
        recipients: List[str]
    ) -> bool:
        """Send notification via Slack API."""
        headers = {
            "Authorization": f"Bearer {self.bot_token}",
            "Content-Type": "application/json"
        }
        
        success_count = 0
        
        async with aiohttp.ClientSession() as session:
            for channel in recipients or [self.default_channel]:
                payload = {
                    "channel": channel,
                    "text": f"{message.title}\n\n{message.content}",
                    "username": "Privacy Finetuner",
                    "icon_emoji": ":robot_face:"
                }
                
                async with session.post(
                    "https://slack.com/api/chat.postMessage",
                    json=payload,
                    headers=headers
                ) as response:
                    
                    if response.status == 200:
                        result = await response.json()
                        if result.get("ok"):
                            success_count += 1
                        else:
                            logger.error(f"Slack API error: {result.get('error')}")
                    else:
                        logger.error(f"Slack API request failed: {response.status}")
                        
        return success_count > 0
        
    def _get_color_for_priority(self, priority: NotificationPriority) -> str:
        """Get color for priority level."""
        color_map = {
            NotificationPriority.LOW: "#36a64f",      # Green
            NotificationPriority.NORMAL: "#2196F3",   # Blue
            NotificationPriority.HIGH: "#ff9800",     # Orange
            NotificationPriority.CRITICAL: "#f44336"  # Red
        }
        return color_map.get(priority, "#2196F3")


class WebhookNotification(BaseNotificationProvider):
    """Generic webhook notification provider."""
    
    def __init__(
        self,
        webhook_url: str,
        headers: Optional[Dict[str, str]] = None,
        secret: Optional[str] = None
    ):
        """Initialize webhook provider.
        
        Args:
            webhook_url: Webhook endpoint URL
            headers: Additional HTTP headers
            secret: Webhook secret for HMAC signing
        """
        self.webhook_url = webhook_url
        self.headers = headers or {}
        self.secret = secret
        
    async def send_message(
        self,
        message: NotificationMessage,
        recipients: List[str]
    ) -> bool:
        """Send webhook notification.
        
        Args:
            message: Notification message
            recipients: Not used for webhooks
            
        Returns:
            True if successful
        """
        try:
            payload = {
                "title": message.title,
                "content": message.content,
                "priority": message.priority.value,
                "metadata": message.metadata,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            headers = {
                "Content-Type": "application/json",
                **self.headers
            }
            
            # Add HMAC signature if secret is configured
            if self.secret:
                import hmac
                import hashlib
                
                payload_bytes = json.dumps(payload, sort_keys=True).encode('utf-8')
                signature = hmac.new(
                    self.secret.encode('utf-8'),
                    payload_bytes,
                    hashlib.sha256
                ).hexdigest()
                headers["X-Signature-SHA256"] = f"sha256={signature}"
                
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    headers=headers
                ) as response:
                    success = 200 <= response.status < 300
                    if not success:
                        logger.error(f"Webhook failed: {response.status}")
                    return success
                    
        except Exception as e:
            logger.error(f"Failed to send webhook notification: {e}")
            return False
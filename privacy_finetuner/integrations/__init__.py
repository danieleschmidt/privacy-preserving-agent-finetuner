"""External integrations module."""

from .github_client import GitHubClient
from .notifications import NotificationService, EmailNotification, SlackNotification
from .auth import AuthService, JWTManager, OAuthProvider

__all__ = [
    "GitHubClient",
    "NotificationService", 
    "EmailNotification",
    "SlackNotification",
    "AuthService",
    "JWTManager", 
    "OAuthProvider"
]
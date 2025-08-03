"""Authentication and authorization services including JWT and OAuth."""

import asyncio
import logging
from typing import Optional, Dict, List, Any, Union
from dataclasses import dataclass
from enum import Enum
import jwt
import hashlib
import secrets
import base64
from datetime import datetime, timedelta, timezone
import aiohttp
from urllib.parse import urlencode, parse_qs
import bcrypt

logger = logging.getLogger(__name__)


class TokenType(Enum):
    """JWT token types."""
    ACCESS = "access"
    REFRESH = "refresh"
    API = "api"
    RESET = "reset"


class OAuthProvider(Enum):
    """Supported OAuth providers."""
    GITHUB = "github"
    GOOGLE = "google"
    MICROSOFT = "microsoft"
    SLACK = "slack"


@dataclass
class TokenPayload:
    """JWT token payload structure."""
    user_id: str
    username: str
    email: str
    roles: List[str]
    permissions: List[str]
    token_type: TokenType
    issued_at: datetime
    expires_at: datetime
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class OAuthConfig:
    """OAuth provider configuration."""
    provider: OAuthProvider
    client_id: str
    client_secret: str
    redirect_uri: str
    scope: List[str]
    authorize_url: str
    token_url: str
    user_info_url: str
    additional_params: Dict[str, str] = None
    
    def __post_init__(self):
        if self.additional_params is None:
            self.additional_params = {}


class JWTManager:
    """JWT token management service."""
    
    def __init__(
        self,
        secret_key: str,
        algorithm: str = "HS256",
        access_token_expire_minutes: int = 30,
        refresh_token_expire_days: int = 7,
        issuer: str = "privacy-finetuner"
    ):
        """Initialize JWT manager.
        
        Args:
            secret_key: Secret key for token signing
            algorithm: JWT signing algorithm
            access_token_expire_minutes: Access token expiration
            refresh_token_expire_days: Refresh token expiration
            issuer: Token issuer
        """
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_token_expire_minutes = access_token_expire_minutes
        self.refresh_token_expire_days = refresh_token_expire_days
        self.issuer = issuer
        self.revoked_tokens = set()  # In production, use Redis
        
    def create_token(
        self,
        payload: TokenPayload,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create JWT token.
        
        Args:
            payload: Token payload
            expires_delta: Custom expiration delta
            
        Returns:
            Encoded JWT token
        """
        now = datetime.now(timezone.utc)
        
        # Set expiration based on token type or custom delta
        if expires_delta:
            expire = now + expires_delta
        elif payload.token_type == TokenType.ACCESS:
            expire = now + timedelta(minutes=self.access_token_expire_minutes)
        elif payload.token_type == TokenType.REFRESH:
            expire = now + timedelta(days=self.refresh_token_expire_days)
        elif payload.token_type == TokenType.API:
            expire = now + timedelta(days=365)  # Long-lived API tokens
        elif payload.token_type == TokenType.RESET:
            expire = now + timedelta(hours=1)  # Short-lived reset tokens
        else:
            expire = now + timedelta(minutes=self.access_token_expire_minutes)
            
        # Build JWT claims
        claims = {
            "sub": payload.user_id,
            "username": payload.username,
            "email": payload.email,
            "roles": payload.roles,
            "permissions": payload.permissions,
            "token_type": payload.token_type.value,
            "iat": now,
            "exp": expire,
            "iss": self.issuer,
            "jti": secrets.token_urlsafe(16),  # JWT ID for revocation
            **payload.metadata
        }
        
        return jwt.encode(claims, self.secret_key, algorithm=self.algorithm)
        
    def verify_token(self, token: str) -> Optional[TokenPayload]:
        """Verify and decode JWT token.
        
        Args:
            token: JWT token string
            
        Returns:
            Decoded token payload or None if invalid
        """
        try:
            # Check if token is revoked
            if token in self.revoked_tokens:
                logger.warning("Attempted to use revoked token")
                return None
                
            # Decode token
            claims = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                issuer=self.issuer
            )
            
            # Extract metadata (remove standard claims)
            metadata = {k: v for k, v in claims.items() 
                       if k not in ['sub', 'username', 'email', 'roles', 'permissions', 
                                   'token_type', 'iat', 'exp', 'iss', 'jti']}
            
            return TokenPayload(
                user_id=claims["sub"],
                username=claims["username"],
                email=claims["email"],
                roles=claims.get("roles", []),
                permissions=claims.get("permissions", []),
                token_type=TokenType(claims["token_type"]),
                issued_at=datetime.fromtimestamp(claims["iat"], tz=timezone.utc),
                expires_at=datetime.fromtimestamp(claims["exp"], tz=timezone.utc),
                metadata=metadata
            )
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {e}")
        except Exception as e:
            logger.error(f"Token verification error: {e}")
            
        return None
        
    def refresh_token(self, refresh_token: str) -> Optional[str]:
        """Create new access token from refresh token.
        
        Args:
            refresh_token: Valid refresh token
            
        Returns:
            New access token or None if invalid
        """
        payload = self.verify_token(refresh_token)
        
        if not payload or payload.token_type != TokenType.REFRESH:
            return None
            
        # Create new access token
        new_payload = TokenPayload(
            user_id=payload.user_id,
            username=payload.username,
            email=payload.email,
            roles=payload.roles,
            permissions=payload.permissions,
            token_type=TokenType.ACCESS,
            issued_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=self.access_token_expire_minutes),
            metadata=payload.metadata
        )
        
        return self.create_token(new_payload)
        
    def revoke_token(self, token: str):
        """Revoke a token.
        
        Args:
            token: Token to revoke
        """
        self.revoked_tokens.add(token)
        logger.info("Token revoked")
        
    def create_api_key(
        self,
        user_id: str,
        name: str,
        permissions: List[str],
        expires_days: Optional[int] = None
    ) -> str:
        """Create long-lived API key.
        
        Args:
            user_id: User ID
            name: API key name
            permissions: API key permissions
            expires_days: Custom expiration in days
            
        Returns:
            API key token
        """
        expires_delta = timedelta(days=expires_days) if expires_days else timedelta(days=365)
        
        payload = TokenPayload(
            user_id=user_id,
            username="api_key",
            email="",
            roles=["api_user"],
            permissions=permissions,
            token_type=TokenType.API,
            issued_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + expires_delta,
            metadata={"api_key_name": name}
        )
        
        return self.create_token(payload, expires_delta)


class PasswordManager:
    """Password hashing and verification."""
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password using bcrypt.
        
        Args:
            password: Plain text password
            
        Returns:
            Hashed password
        """
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
        
    @staticmethod
    def verify_password(password: str, hashed_password: str) -> bool:
        """Verify password against hash.
        
        Args:
            password: Plain text password
            hashed_password: Hashed password
            
        Returns:
            True if password matches
        """
        try:
            return bcrypt.checkpw(
                password.encode('utf-8'),
                hashed_password.encode('utf-8')
            )
        except Exception:
            return False
            
    @staticmethod
    def generate_secure_password(length: int = 16) -> str:
        """Generate secure random password.
        
        Args:
            length: Password length
            
        Returns:
            Random password
        """
        return secrets.token_urlsafe(length)


class OAuthManager:
    """OAuth 2.0 flow management."""
    
    def __init__(self, providers: Dict[OAuthProvider, OAuthConfig]):
        """Initialize OAuth manager.
        
        Args:
            providers: OAuth provider configurations
        """
        self.providers = providers
        self.state_store = {}  # In production, use Redis
        
    def get_authorization_url(
        self,
        provider: OAuthProvider,
        state: Optional[str] = None
    ) -> str:
        """Get OAuth authorization URL.
        
        Args:
            provider: OAuth provider
            state: Optional state parameter
            
        Returns:
            Authorization URL
        """
        if provider not in self.providers:
            raise ValueError(f"Provider {provider.value} not configured")
            
        config = self.providers[provider]
        
        # Generate state if not provided
        if not state:
            state = secrets.token_urlsafe(32)
            
        # Store state for verification
        self.state_store[state] = {
            "provider": provider,
            "created_at": datetime.now(timezone.utc),
            "expires_at": datetime.now(timezone.utc) + timedelta(minutes=10)
        }
        
        # Build authorization URL
        params = {
            "client_id": config.client_id,
            "redirect_uri": config.redirect_uri,
            "scope": " ".join(config.scope),
            "response_type": "code",
            "state": state,
            **config.additional_params
        }
        
        return f"{config.authorize_url}?{urlencode(params)}"
        
    async def handle_callback(
        self,
        provider: OAuthProvider,
        code: str,
        state: str
    ) -> Optional[Dict[str, Any]]:
        """Handle OAuth callback.
        
        Args:
            provider: OAuth provider
            code: Authorization code
            state: State parameter
            
        Returns:
            User info or None if failed
        """
        # Verify state
        if state not in self.state_store:
            logger.warning("Invalid OAuth state")
            return None
            
        state_info = self.state_store.pop(state)
        
        if state_info["provider"] != provider:
            logger.warning("OAuth state provider mismatch")
            return None
            
        if datetime.now(timezone.utc) > state_info["expires_at"]:
            logger.warning("OAuth state expired")
            return None
            
        config = self.providers[provider]
        
        try:
            # Exchange code for token
            token_data = await self._exchange_code_for_token(config, code)
            
            if not token_data:
                return None
                
            # Get user info
            user_info = await self._get_user_info(config, token_data["access_token"])
            
            if user_info:
                user_info["provider"] = provider.value
                user_info["access_token"] = token_data["access_token"]
                user_info["refresh_token"] = token_data.get("refresh_token")
                
            return user_info
            
        except Exception as e:
            logger.error(f"OAuth callback error: {e}")
            return None
            
    async def _exchange_code_for_token(
        self,
        config: OAuthConfig,
        code: str
    ) -> Optional[Dict[str, Any]]:
        """Exchange authorization code for access token."""
        data = {
            "client_id": config.client_id,
            "client_secret": config.client_secret,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": config.redirect_uri
        }
        
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                config.token_url,
                data=data,
                headers=headers
            ) as response:
                
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Token exchange failed: {response.status}")
                    return None
                    
    async def _get_user_info(
        self,
        config: OAuthConfig,
        access_token: str
    ) -> Optional[Dict[str, Any]]:
        """Get user information from provider."""
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(
                config.user_info_url,
                headers=headers
            ) as response:
                
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"User info request failed: {response.status}")
                    return None


class AuthService:
    """Comprehensive authentication service."""
    
    def __init__(
        self,
        jwt_manager: JWTManager,
        oauth_manager: Optional[OAuthManager] = None
    ):
        """Initialize auth service.
        
        Args:
            jwt_manager: JWT token manager
            oauth_manager: OAuth manager (optional)
        """
        self.jwt_manager = jwt_manager
        self.oauth_manager = oauth_manager
        self.login_attempts = {}  # In production, use Redis
        
    async def authenticate_user(
        self,
        username: str,
        password: str,
        ip_address: Optional[str] = None
    ) -> Optional[Dict[str, str]]:
        """Authenticate user with username/password.
        
        Args:
            username: Username or email
            password: Password
            ip_address: Client IP address
            
        Returns:
            Access and refresh tokens or None
        """
        # Check rate limiting
        if not self._check_rate_limit(username, ip_address):
            logger.warning(f"Rate limit exceeded for {username}")
            return None
            
        # In a real implementation, verify against database
        # This is a simplified example
        user_data = await self._verify_credentials(username, password)
        
        if not user_data:
            self._record_failed_attempt(username, ip_address)
            return None
            
        # Clear failed attempts on successful login
        self._clear_failed_attempts(username, ip_address)
        
        # Create tokens
        access_payload = TokenPayload(
            user_id=user_data["id"],
            username=user_data["username"],
            email=user_data["email"],
            roles=user_data.get("roles", []),
            permissions=user_data.get("permissions", []),
            token_type=TokenType.ACCESS,
            issued_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=30)
        )
        
        refresh_payload = TokenPayload(
            user_id=user_data["id"],
            username=user_data["username"],
            email=user_data["email"],
            roles=user_data.get("roles", []),
            permissions=user_data.get("permissions", []),
            token_type=TokenType.REFRESH,
            issued_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(days=7)
        )
        
        return {
            "access_token": self.jwt_manager.create_token(access_payload),
            "refresh_token": self.jwt_manager.create_token(refresh_payload),
            "token_type": "bearer"
        }
        
    async def authenticate_oauth(
        self,
        provider: OAuthProvider,
        code: str,
        state: str
    ) -> Optional[Dict[str, str]]:
        """Authenticate user via OAuth.
        
        Args:
            provider: OAuth provider
            code: Authorization code
            state: State parameter
            
        Returns:
            Access and refresh tokens or None
        """
        if not self.oauth_manager:
            raise ValueError("OAuth manager not configured")
            
        user_info = await self.oauth_manager.handle_callback(provider, code, state)
        
        if not user_info:
            return None
            
        # Create or update user in database
        user_data = await self._handle_oauth_user(user_info)
        
        if not user_data:
            return None
            
        # Create tokens
        access_payload = TokenPayload(
            user_id=user_data["id"],
            username=user_data["username"],
            email=user_data["email"],
            roles=user_data.get("roles", []),
            permissions=user_data.get("permissions", []),
            token_type=TokenType.ACCESS,
            issued_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=30),
            metadata={"oauth_provider": provider.value}
        )
        
        refresh_payload = TokenPayload(
            user_id=user_data["id"],
            username=user_data["username"],
            email=user_data["email"],
            roles=user_data.get("roles", []),
            permissions=user_data.get("permissions", []),
            token_type=TokenType.REFRESH,
            issued_at=datetime.now(timezone.utc),
            expires_at=datetime.now(timezone.utc) + timedelta(days=7),
            metadata={"oauth_provider": provider.value}
        )
        
        return {
            "access_token": self.jwt_manager.create_token(access_payload),
            "refresh_token": self.jwt_manager.create_token(refresh_payload),
            "token_type": "bearer"
        }
        
    def validate_token(self, token: str) -> Optional[TokenPayload]:
        """Validate JWT token.
        
        Args:
            token: JWT token
            
        Returns:
            Token payload or None if invalid
        """
        return self.jwt_manager.verify_token(token)
        
    def has_permission(self, token_payload: TokenPayload, permission: str) -> bool:
        """Check if user has specific permission.
        
        Args:
            token_payload: JWT token payload
            permission: Permission to check
            
        Returns:
            True if user has permission
        """
        return permission in token_payload.permissions
        
    def has_role(self, token_payload: TokenPayload, role: str) -> bool:
        """Check if user has specific role.
        
        Args:
            token_payload: JWT token payload
            role: Role to check
            
        Returns:
            True if user has role
        """
        return role in token_payload.roles
        
    def _check_rate_limit(self, username: str, ip_address: Optional[str]) -> bool:
        """Check authentication rate limit."""
        now = datetime.now(timezone.utc)
        key = f"{username}:{ip_address}" if ip_address else username
        
        if key not in self.login_attempts:
            return True
            
        attempts = self.login_attempts[key]
        
        # Remove old attempts (older than 1 hour)
        attempts["timestamps"] = [
            t for t in attempts["timestamps"]
            if now - t < timedelta(hours=1)
        ]
        
        # Allow up to 5 attempts per hour
        return len(attempts["timestamps"]) < 5
        
    def _record_failed_attempt(self, username: str, ip_address: Optional[str]):
        """Record failed login attempt."""
        now = datetime.now(timezone.utc)
        key = f"{username}:{ip_address}" if ip_address else username
        
        if key not in self.login_attempts:
            self.login_attempts[key] = {"timestamps": [], "count": 0}
            
        self.login_attempts[key]["timestamps"].append(now)
        self.login_attempts[key]["count"] += 1
        
    def _clear_failed_attempts(self, username: str, ip_address: Optional[str]):
        """Clear failed login attempts after successful login."""
        key = f"{username}:{ip_address}" if ip_address else username
        if key in self.login_attempts:
            del self.login_attempts[key]
            
    async def _verify_credentials(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Verify user credentials (mock implementation)."""
        # In production, query database and verify password hash
        # This is a simplified mock
        mock_users = {
            "admin": {
                "id": "user_1",
                "username": "admin",
                "email": "admin@example.com",
                "password_hash": PasswordManager.hash_password("admin123"),
                "roles": ["admin", "user"],
                "permissions": ["read", "write", "delete", "train", "manage"]
            }
        }
        
        user = mock_users.get(username)
        if user and PasswordManager.verify_password(password, user["password_hash"]):
            return user
            
        return None
        
    async def _handle_oauth_user(self, user_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle OAuth user creation/update (mock implementation)."""
        # In production, create or update user in database
        return {
            "id": user_info.get("id", "oauth_user"),
            "username": user_info.get("login", user_info.get("email", "oauth_user")),
            "email": user_info.get("email", ""),
            "roles": ["user"],
            "permissions": ["read", "train"]
        }
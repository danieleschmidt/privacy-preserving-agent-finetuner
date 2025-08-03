"""Database connection management and session handling."""

import os
import logging
from typing import Generator, Optional
from contextlib import contextmanager
from sqlalchemy import create_engine, event, Engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
import redis
from functools import lru_cache

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database connections and sessions."""
    
    def __init__(self, database_url: str, redis_url: Optional[str] = None):
        """Initialize database manager.
        
        Args:
            database_url: PostgreSQL connection URL
            redis_url: Redis connection URL for caching
        """
        self.database_url = database_url
        self.redis_url = redis_url
        
        # Create SQLAlchemy engine with connection pooling
        self.engine = create_engine(
            database_url,
            poolclass=QueuePool,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=os.getenv("DATABASE_DEBUG", "false").lower() == "true"
        )
        
        # Add connection event listeners
        event.listen(self.engine, "connect", self._on_connect)
        event.listen(self.engine, "checkout", self._on_checkout)
        
        # Create session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
        # Initialize Redis connection if URL provided
        self.redis_client = None
        if redis_url:
            try:
                self.redis_client = redis.from_url(redis_url, decode_responses=True)
                self.redis_client.ping()
                logger.info("Redis connection established")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}")
                self.redis_client = None
        
        logger.info("Database manager initialized")
    
    def _on_connect(self, dbapi_connection, connection_record):
        """Called when a new database connection is created."""
        # Set connection-level settings
        with dbapi_connection.cursor() as cursor:
            cursor.execute("SET timezone = 'UTC'")
            cursor.execute("SET statement_timeout = '300s'")
    
    def _on_checkout(self, dbapi_connection, connection_record, connection_proxy):
        """Called when a connection is checked out from the pool."""
        # Validate connection is still alive
        try:
            with dbapi_connection.cursor() as cursor:
                cursor.execute("SELECT 1")
        except Exception:
            # Connection is stale, invalidate it
            connection_record.invalidate()
            raise
    
    def get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()
    
    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """Provide a transactional scope around database operations."""
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    def create_all_tables(self):
        """Create all database tables."""
        from .models import Base
        Base.metadata.create_all(bind=self.engine)
        logger.info("Database tables created")
    
    def drop_all_tables(self):
        """Drop all database tables (use with caution)."""
        from .models import Base
        Base.metadata.drop_all(bind=self.engine)
        logger.warning("All database tables dropped")
    
    def get_cache_client(self) -> Optional[redis.Redis]:
        """Get Redis cache client."""
        return self.redis_client
    
    def health_check(self) -> dict:
        """Perform database health check."""
        try:
            with self.session_scope() as session:
                session.execute("SELECT 1")
            db_status = "healthy"
        except Exception as e:
            db_status = f"unhealthy: {str(e)}"
        
        # Check Redis if available
        redis_status = "not_configured"
        if self.redis_client:
            try:
                self.redis_client.ping()
                redis_status = "healthy"
            except Exception as e:
                redis_status = f"unhealthy: {str(e)}"
        
        return {
            "database": db_status,
            "redis": redis_status,
            "engine_pool_size": self.engine.pool.size(),
            "engine_checked_out": self.engine.pool.checkedout()
        }


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def initialize_database(database_url: str, redis_url: Optional[str] = None) -> DatabaseManager:
    """Initialize the global database manager."""
    global _db_manager
    _db_manager = DatabaseManager(database_url, redis_url)
    return _db_manager


@lru_cache()
def get_database() -> DatabaseManager:
    """Get the global database manager instance."""
    if _db_manager is None:
        # Initialize with environment variables
        database_url = os.getenv("DATABASE_URL", "sqlite:///./privacy_finetuner.db")
        redis_url = os.getenv("REDIS_URL")
        return initialize_database(database_url, redis_url)
    return _db_manager


def get_db_session() -> Generator[Session, None, None]:
    """Dependency for FastAPI to get database session."""
    db = get_database()
    with db.session_scope() as session:
        yield session


# Database migration utilities
def run_migrations():
    """Run database migrations using Alembic."""
    try:
        from alembic.config import Config
        from alembic import command
        
        alembic_cfg = Config("alembic.ini")
        command.upgrade(alembic_cfg, "head")
        logger.info("Database migrations completed")
    except ImportError:
        logger.warning("Alembic not available, skipping migrations")
    except Exception as e:
        logger.error(f"Migration failed: {e}")
        raise


def create_database_url(
    username: str,
    password: str,
    host: str,
    port: int,
    database: str,
    driver: str = "postgresql+psycopg2"
) -> str:
    """Create database URL from components."""
    return f"{driver}://{username}:{password}@{host}:{port}/{database}"
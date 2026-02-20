"""
Database connection and session management using SQLAlchemy.
Provides safe, ORM-based database operations with connection pooling.
"""
import os
from contextlib import contextmanager
from typing import Generator, Optional

from sqlalchemy import create_engine, event, pool
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError

from config import DATABASE_URL, DB_POOL_SIZE, DB_MAX_OVERFLOW, DB_POOL_TIMEOUT


class DatabaseManager:
    """Manages database connections and sessions with safety features."""

    def __init__(self):
        self.engine: Optional[Engine] = None
        self.SessionLocal: Optional[sessionmaker] = None

    def init(self):
        """Initialize database engine and session factory."""
        if not DATABASE_URL:
            raise ValueError("DATABASE_URL not set in environment")

        # Create engine with connection pooling
        self.engine = create_engine(
            DATABASE_URL,
            poolclass=pool.QueuePool,
            pool_size=DB_POOL_SIZE,
            max_overflow=DB_MAX_OVERFLOW,
            pool_timeout=DB_POOL_TIMEOUT,
            echo=False,  # Set to True for SQL debugging
        )

        # Add event listener for connection pool
        @event.listens_for(self.engine, "connect")
        def set_sqlite_pragma(dbapi_conn, connection_record):
            """Enable foreign keys for SQLite."""
            if "sqlite" in DATABASE_URL:
                cursor = dbapi_conn.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()

        self.SessionLocal = sessionmaker(bind=self.engine, expire_on_commit=False)

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """
        Context manager for database sessions.
        Ensures proper cleanup and error handling.
        """
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            raise RuntimeError(f"Database error: {str(e)}") from e
        finally:
            session.close()

    def get_schema_info(self) -> dict:
        """
        Retrieve database schema information.
        Returns tables, columns, and foreign key relationships.
        """
        if not self.engine:
            raise RuntimeError("Database not initialized")

        schema_info = {}
        try:
            with self.engine.connect() as conn:
                inspector = None
                if "sqlite" in DATABASE_URL:
                    from sqlalchemy import inspect
                    inspector = inspect(self.engine)
                    tables = inspector.get_table_names()
                    for table in tables:
                        columns = inspector.get_columns(table)
                        fks = inspector.get_foreign_keys(table)
                        schema_info[table] = {
                            "columns": [col["name"] for col in columns],
                            "foreign_keys": [(fk["constrained_columns"][0], fk["referred_table"]) for fk in fks],
                        }
        except Exception as e:
            raise RuntimeError(f"Failed to retrieve schema: {str(e)}") from e

        return schema_info

    def close(self):
        """Close database engine and connections."""
        if self.engine:
            self.engine.dispose()


# Global instance
db_manager = DatabaseManager()

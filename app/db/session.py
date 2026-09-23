"""SQLAlchemy engine management for read-only database access.

The engine is created lazily so that importing the package never opens a
connection. This keeps the service importable (and testable) without a live
database.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from app.core.config import get_settings

_engine: Engine | None = None


def _normalise_url(url: str) -> str:
    """Ensure the URL uses the psycopg (v3) driver."""

    if url.startswith("postgres://"):
        return "postgresql+psycopg://" + url[len("postgres://") :]
    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url[len("postgresql://") :]
    return url


def get_engine() -> Engine:
    """Return the process-wide engine, creating it on first use."""

    global _engine
    if _engine is None:
        settings = get_settings()
        connect_args: dict[str, Any] = {
            "connect_timeout": settings.db_connect_timeout,
        }
        if _normalise_url(settings.database_url).startswith("postgresql"):
            # Belt-and-braces read-only guard at the session level.
            connect_args["options"] = "-c default_transaction_read_only=on"

        _engine = create_engine(
            _normalise_url(settings.database_url),
            pool_pre_ping=True,
            pool_recycle=1800,
            future=True,
            connect_args=connect_args,
        )
    return _engine


def dispose_engine() -> None:
    """Dispose of the engine (used on shutdown)."""

    global _engine
    if _engine is not None:
        _engine.dispose()
        _engine = None

"""Application settings loaded from environment variables (prefix ``AI_``)."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration.

    Every field is populated from an environment variable prefixed with ``AI_``
    (for example ``AI_DATABASE_URL`` -> ``database_url``). Values may also be
    provided through a local ``.env`` file.
    """

    model_config = SettingsConfigDict(
        env_prefix="AI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Database (read-only access to the backend PostgreSQL database) ---
    database_url: str = Field(
        default="postgresql+psycopg://postgres:postgres@localhost:5432/maminco",
        description="SQLAlchemy URL pointing at the backend PostgreSQL database.",
    )
    db_connect_timeout: int = Field(default=5, ge=1, le=60)

    # --- Security ---
    internal_service_token: str = Field(
        default="change-me",
        description="Shared secret expected in the X-Internal-Token header.",
    )

    # --- Observability ---
    log_level: str = Field(default="INFO")

    # --- Model artifacts ---
    model_dir: str = Field(default="data/models")

    # --- Forecasting behaviour ---
    min_days_required: int = Field(default=14, ge=2)
    history_days: int = Field(default=180, ge=14)
    default_horizon: int = Field(default=7, ge=1, le=60)
    max_horizon: int = Field(default=60, ge=1, le=365)
    random_seed: int = Field(default=42)

    # --- CORS ---
    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://localhost:3001",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:3001",
        ]
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        """Allow a comma-separated string for ``AI_CORS_ORIGINS``."""
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return []
            if stripped.startswith("["):
                return value
            return [item.strip() for item in stripped.split(",") if item.strip()]
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()

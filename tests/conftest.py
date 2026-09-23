"""Shared pytest fixtures.

The tests never touch a live database: the query layer is monkeypatched to
return empty DataFrames by default. Individual tests may override the queries
to inject synthetic frames.
"""

from __future__ import annotations

import os

# Configure the service before any application module is imported.
os.environ["AI_INTERNAL_SERVICE_TOKEN"] = "test-token"
os.environ["AI_DATABASE_URL"] = "postgresql+psycopg://postgres:postgres@localhost:5432/test"
os.environ["AI_MODEL_DIR"] = "data/models"
os.environ.setdefault("AI_LOG_LEVEL", "WARNING")

import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.core.config import get_settings  # noqa: E402

get_settings.cache_clear()

AUTH_HEADERS = {"X-Internal-Token": "test-token"}


@pytest.fixture()
def client(monkeypatch):
    """A TestClient whose startup sees an empty (mocked) database."""

    import app.db.queries as queries

    empty = pd.DataFrame()
    monkeypatch.setattr(queries, "load_daily_sales", lambda *a, **k: empty)
    monkeypatch.setattr(queries, "load_product_demand", lambda *a, **k: empty)
    monkeypatch.setattr(queries, "load_hourly_demand", lambda *a, **k: empty)
    monkeypatch.setattr(queries, "load_daily_metrics", lambda *a, **k: empty)

    from app.main import app
    from app.models.engine import EngineState, engine

    engine.state = EngineState()

    with TestClient(app) as test_client:
        yield test_client

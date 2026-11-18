"""Engine behaviour: insufficient-data windows, error masking and timezone."""

from __future__ import annotations

import numpy as np
import pandas as pd

import app.db.queries as queries
from app.models.engine import engine
from tests.conftest import AUTH_HEADERS


def daily_frame(days: int) -> pd.DataFrame:
    dates = pd.date_range(end=pd.Timestamp("2026-03-31"), periods=days, freq="D")
    totals = np.where(dates.dayofweek >= 5, 1200.0, 800.0)
    return pd.DataFrame({"date": dates, "total_sales": totals, "total_orders": 30})


def test_forecast_insufficient_between_14_and_35_days(client, monkeypatch):
    for days in (14, 35):
        monkeypatch.setattr(queries, "load_daily_sales", lambda *a, _d=days, **k: daily_frame(_d))
        engine.refresh()
        body = client.get("/forecast/sales", headers=AUTH_HEADERS).json()
        assert body["insufficient_data"] is True
        assert body["forecast"] == []
        assert "36" in body["reason"]


def test_forecast_available_with_enough_days_and_metrics(client, monkeypatch):
    monkeypatch.setattr(queries, "load_daily_sales", lambda *a, **k: daily_frame(90))
    engine.refresh()
    body = client.get("/forecast/sales?horizon=7", headers=AUTH_HEADERS).json()

    assert body["insufficient_data"] is False
    assert body["baseline_mape"] is not None
    assert body["mape"] is not None
    assert body["backtest_folds"] >= 1

    summary = client.get("/metrics/summary", headers=AUTH_HEADERS).json()
    assert summary["sales_model_mape"] == body["mape"]
    assert summary["baseline_mape"] == body["baseline_mape"]


def test_database_errors_are_not_leaked(client, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("password authentication failed for user secretuser")

    monkeypatch.setattr(queries, "load_daily_sales", boom)
    engine.refresh()

    responses = [
        client.get("/health"),
        client.get("/forecast/sales", headers=AUTH_HEADERS),
        client.get("/forecast/products", headers=AUTH_HEADERS),
        client.get("/forecast/peak-hours", headers=AUTH_HEADERS),
        client.get("/insights/anomalies", headers=AUTH_HEADERS),
        client.get("/metrics/summary", headers=AUTH_HEADERS),
        client.post("/train", headers=AUTH_HEADERS),
    ]
    for response in responses:
        assert response.status_code == 200
        assert "secretuser" not in response.text
        assert "password" not in response.text.lower()

    assert client.get("/health").json()["database_error"] == "Database unavailable."
    assert client.get("/forecast/sales", headers=AUTH_HEADERS).json()["reason"] == "Database unavailable."


def test_queries_bucket_by_configured_timezone(monkeypatch):
    captured: list[dict] = []

    def fake_read(sql, params):
        captured.append(params)
        return pd.DataFrame()

    monkeypatch.setattr(queries, "_read", fake_read)
    queries.load_daily_sales(30)
    assert captured and captured[0]["days"] == 30

    sql = str(queries._DAILY_SALES_SQL)
    assert "AT TIME ZONE :tz" in sql
    assert "AT TIME ZONE :tz" in str(queries._HOURLY_DEMAND_SQL)
    assert "AT TIME ZONE :tz" in str(queries._PRODUCT_DEMAND_SQL)


def test_read_injects_timezone_parameter(monkeypatch):
    import app.core.config as config

    seen: dict = {}

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    class FakeEngine:
        def connect(self):
            return FakeConnection()

    def fake_read_sql(sql, connection, params):
        seen.update(params)
        return pd.DataFrame()

    monkeypatch.setenv("AI_TIMEZONE", "America/Lima")
    config.get_settings.cache_clear()
    monkeypatch.setattr(queries, "get_engine", lambda: FakeEngine())
    monkeypatch.setattr(pd, "read_sql", fake_read_sql)
    try:
        queries.load_daily_sales(10)
    finally:
        config.get_settings.cache_clear()

    assert seen["tz"] == "America/Lima"

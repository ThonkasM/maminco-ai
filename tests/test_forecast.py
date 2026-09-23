"""Forecast endpoint behaviour with empty and populated data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.models.engine import engine
from tests.conftest import AUTH_HEADERS


def test_sales_forecast_insufficient_when_empty(client):
    response = client.get("/forecast/sales", headers=AUTH_HEADERS)

    assert response.status_code == 200
    body = response.json()
    assert body["insufficient_data"] is True
    assert body["forecast"] == []
    assert body["reason"]


def test_products_forecast_insufficient_when_empty(client):
    response = client.get("/forecast/products", headers=AUTH_HEADERS)

    assert response.status_code == 200
    body = response.json()
    assert body["insufficient_data"] is True
    assert body["forecast"] == []


def test_peak_hours_insufficient_when_empty(client):
    response = client.get("/forecast/peak-hours", headers=AUTH_HEADERS)

    assert response.status_code == 200
    body = response.json()
    assert body["insufficient_data"] is True
    assert body["profile"] == []


def test_anomalies_insufficient_when_empty(client):
    response = client.get("/insights/anomalies", headers=AUTH_HEADERS)

    assert response.status_code == 200
    body = response.json()
    assert body["insufficient_data"] is True
    assert body["anomalies"] == []


def test_metrics_summary_insufficient_when_empty(client):
    response = client.get("/metrics/summary", headers=AUTH_HEADERS)

    assert response.status_code == 200
    body = response.json()
    assert body["insufficient_data"] is True


def test_sales_forecast_with_synthetic_history(client, monkeypatch):
    """The full pipeline works on in-memory data (no live DB needed)."""

    import app.db.queries as queries

    rng = np.random.default_rng(7)
    dates = pd.date_range(end=pd.Timestamp("2026-01-31"), periods=60, freq="D")
    weekday_factor = np.where(dates.dayofweek >= 5, 1.6, 1.0)
    totals = 800.0 * weekday_factor + rng.normal(0, 40, size=len(dates))
    daily = pd.DataFrame(
        {
            "date": dates,
            "total_sales": totals,
            "total_orders": (totals / 25).round(),
        }
    )
    monkeypatch.setattr(queries, "load_daily_sales", lambda *a, **k: daily)

    engine.refresh()
    response = client.get("/forecast/sales?horizon=7", headers=AUTH_HEADERS)

    assert response.status_code == 200
    body = response.json()
    assert body["insufficient_data"] is False
    assert len(body["forecast"]) == 7
    assert body["chosen"] in {"model", "baseline"}
    assert all(point["predicted_sales"] >= 0 for point in body["forecast"])


def test_horizon_validation(client):
    assert client.get("/forecast/sales?horizon=0", headers=AUTH_HEADERS).status_code == 422
    assert client.get("/forecast/sales?horizon=999", headers=AUTH_HEADERS).status_code == 422

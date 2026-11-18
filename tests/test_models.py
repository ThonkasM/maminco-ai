"""Unit tests for baselines, anomaly detection and the walk-forward backtest."""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.models import anomalies as anomaly_model
from app.models import sales as sales_model
from app.models.baseline import (
    baseline_forecast,
    mean_absolute_error,
    mean_absolute_percentage_error,
    moving_average_forecast,
    seasonal_naive_forecast,
)


def weekly_series(days: int, seed: int = 3) -> pd.Series:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2026-01-01", periods=days, freq="D")
    factor = np.where(index.dayofweek >= 5, 1.6, 1.0)
    values = 800.0 * factor + rng.normal(0, 30, size=days)
    return pd.Series(values, index=index)


def test_seasonal_naive_repeats_last_week():
    history = [1, 2, 3, 4, 5, 6, 7, 10, 20, 30, 40, 50, 60, 70]
    assert seasonal_naive_forecast(history, 3) == [10.0, 20.0, 30.0]


def test_moving_average_uses_window_and_is_non_negative():
    assert moving_average_forecast([2, 4, 6], 2, window=3)[0] == 4.0
    assert all(v >= 0 for v in moving_average_forecast([], 3))


def test_baseline_picks_seasonal_only_with_two_seasons():
    short = [10.0] * 10
    assert baseline_forecast(short, 2) == moving_average_forecast(short, 2, window=7)
    long = list(range(1, 15))
    assert baseline_forecast(long, 1) == seasonal_naive_forecast(long, 1)


def test_mape_ignores_zero_actuals():
    assert mean_absolute_percentage_error([0, 100], [50, 90]) == 10.0
    assert mean_absolute_percentage_error([0, 0], [1, 1]) == 0.0
    assert mean_absolute_error([1, 2, 3], [2, 2, 2]) == 2 / 3


def test_anomaly_detection_flags_spike():
    dates = pd.date_range("2026-01-01", periods=30, freq="D")
    totals = np.full(30, 500.0) + np.arange(30) % 5
    totals[10] = 5000.0
    frame = pd.DataFrame({"date": dates, "total_sales": totals})

    result = anomaly_model.detect_anomalies(frame, min_days=14)

    assert result["insufficient_data"] is False
    assert any(item["date"] == "2026-01-11" and item["direction"] == "high" for item in result["anomalies"])


def test_anomaly_detection_insufficient_below_minimum():
    frame = pd.DataFrame(
        {"date": pd.date_range("2026-01-01", periods=5), "total_sales": [1.0] * 5}
    )
    assert anomaly_model.detect_anomalies(frame, min_days=14)["insufficient_data"] is True


def test_fit_rejects_14_to_35_days_with_sales():
    for days in (14, 20, 35):
        assert sales_model.fit(weekly_series(days)) is None


def test_fit_counts_days_with_sales_not_calendar_span():
    series = weekly_series(60)
    series.iloc[::2] = 0.0
    assert sales_model.count_sales_days(series) == 30
    assert sales_model.fit(series) is None


def test_fit_with_minimum_days_falls_back_to_baseline():
    fitted = sales_model.fit(weekly_series(sales_model.MIN_SALES_DAYS))
    assert fitted is not None
    assert fitted.chosen == "baseline"
    assert fitted.model_mae is None
    assert fitted.baseline_mape >= 0


def test_walk_forward_exposes_model_and_baseline_metrics():
    fitted = sales_model.fit(weekly_series(90))
    assert fitted is not None
    body = fitted.forecast(7)

    assert body["backtest_folds"] >= 1
    assert body["model_mae"] is not None
    assert body["baseline_mape"] >= 0
    assert body["model_mape"] is not None
    expected = body["model_mape"] if body["chosen"] == "model" else body["baseline_mape"]
    assert body["mape"] == expected
    assert len(body["forecast"]) == 7


def test_walk_forward_is_deterministic():
    first = sales_model.fit(weekly_series(90)).forecast(7)
    second = sales_model.fit(weekly_series(90)).forecast(7)
    assert first["model_mae"] == second["model_mae"]
    assert first["forecast"] == second["forecast"]

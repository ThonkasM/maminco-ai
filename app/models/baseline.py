"""Baseline forecasting helpers.

Two classical, deterministic baselines are provided:

* seasonal-naive (same weekday from the previous week);
* moving average.

A Holt-Winters exponential smoothing baseline is also exposed when
``statsmodels`` is installed. Baselines are used both as a fallback and as the
reference point the machine-learning model must beat.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def moving_average_forecast(
    values: Sequence[float], horizon: int, window: int = 7
) -> list[float]:
    """Forecast ``horizon`` steps with a trailing moving average."""

    history = [float(v) for v in values]
    predictions: list[float] = []
    effective_window = max(1, min(window, len(history) or 1))

    for _ in range(horizon):
        if history:
            prediction = float(np.mean(history[-effective_window:]))
        else:
            prediction = 0.0
        predictions.append(max(prediction, 0.0))
        history.append(prediction)

    return predictions


def seasonal_naive_forecast(
    values: Sequence[float], horizon: int, season: int = 7
) -> list[float]:
    """Forecast by repeating the value observed ``season`` steps earlier."""

    history = [float(v) for v in values]
    predictions: list[float] = []
    fallback = float(np.mean(history)) if history else 0.0

    for _ in range(horizon):
        if len(history) >= season:
            prediction = history[-season]
        else:
            prediction = fallback
        prediction = max(prediction, 0.0)
        predictions.append(prediction)
        history.append(prediction)

    return predictions


def baseline_forecast(
    values: Sequence[float], horizon: int, season: int = 7
) -> list[float]:
    """Pick the most appropriate baseline given the available history."""

    if len(values) >= 2 * season:
        return seasonal_naive_forecast(values, horizon, season=season)
    return moving_average_forecast(values, horizon, window=min(7, max(1, len(values))))


def holt_winters_forecast(
    values: Sequence[float], horizon: int, season: int = 7
) -> list[float] | None:
    """Optional Holt-Winters baseline (returns ``None`` if unavailable)."""

    if len(values) < 2 * season:
        return None

    try:  # pragma: no cover - depends on optional dependency being installed
        from statsmodels.tsa.holtwinters import ExponentialSmoothing

        series = np.asarray([float(v) for v in values], dtype=float)
        model = ExponentialSmoothing(
            series,
            trend=None,
            seasonal="add",
            seasonal_periods=season,
            initialization_method="estimated",
        ).fit(optimized=True)
        forecast = model.forecast(horizon)
        return [max(float(v), 0.0) for v in forecast]
    except Exception:  # noqa: BLE001 - optional, never break the request
        return None


def mean_absolute_error(actual: Sequence[float], predicted: Sequence[float]) -> float:
    """MAE between two equal-length sequences."""

    actual_arr = np.asarray(actual, dtype=float)
    predicted_arr = np.asarray(predicted, dtype=float)
    if actual_arr.size == 0 or predicted_arr.size == 0:
        return 0.0
    size = min(actual_arr.size, predicted_arr.size)
    return float(np.mean(np.abs(actual_arr[:size] - predicted_arr[:size])))


def mean_absolute_percentage_error(
    actual: Sequence[float], predicted: Sequence[float]
) -> float:
    """MAPE (%), ignoring zero actuals to avoid division by zero."""

    actual_arr = np.asarray(actual, dtype=float)
    predicted_arr = np.asarray(predicted, dtype=float)
    size = min(actual_arr.size, predicted_arr.size)
    if size == 0:
        return 0.0
    actual_arr = actual_arr[:size]
    predicted_arr = predicted_arr[:size]
    mask = actual_arr != 0
    if not mask.any():
        return 0.0
    return float(
        np.mean(np.abs((actual_arr[mask] - predicted_arr[mask]) / actual_arr[mask]))
        * 100
    )

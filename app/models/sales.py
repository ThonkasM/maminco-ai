"""Daily sales forecasting.

Approach
--------
1. Reindex the daily series to a continuous calendar (missing days -> 0 sales,
   because a closed restaurant day really means zero revenue).
2. Build a supervised feature matrix: day-of-week, weekend flag, lag-1, lag-7,
   and trailing rolling means.
3. Score a scikit-learn ``GradientBoostingRegressor`` against the baseline with
   a walk-forward backtest: for several origins the model is refit on data
   before the origin and forecasts the next block recursively (feeding its own
   predictions back as lags, exactly as in production), so no future actuals
   leak into the comparison.
4. Keep whichever wins on backtest MAE and produce a recursive forecast.

The model is fitted **once** (see :func:`fit`) and reused for every horizon,
which keeps request handling free of model training. It is deterministic: a
fixed ``random_state`` is used and only CPU features are involved.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from app.models.baseline import (
    baseline_forecast,
    mean_absolute_error,
    mean_absolute_percentage_error,
)

MAX_LAG = 28
MIN_SALES_DAYS = MAX_LAG + 8
MIN_TRAIN_ROWS = 8
MIN_BASELINE_HISTORY = 14
BACKTEST_HORIZON = 7
MAX_BACKTEST_FOLDS = 3
FEATURES = ["dow", "is_weekend", "lag1", "lag7", "roll7", "roll14", "roll28"]
WEEKDAY_NAMES = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]


def required_days(min_days_required: int) -> int:
    """Minimum days with sales needed to fit, coherent with the 28-day lags."""

    return max(int(min_days_required), MIN_SALES_DAYS)


def count_sales_days(series: pd.Series) -> int:
    """Number of days with strictly positive values (zero-filled gaps excluded)."""

    if series is None or series.empty:
        return 0
    return int((series > 0).sum())


def to_daily_series(daily_sales: pd.DataFrame) -> pd.Series:
    """Convert a ``load_daily_sales`` frame into a continuous daily Series."""

    if daily_sales is None or daily_sales.empty:
        return pd.Series(dtype="float64")

    frame = daily_sales.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    series = frame.groupby("date")["total_sales"].sum().sort_index().astype("float64")
    if series.empty:
        return series

    full_range = pd.date_range(series.index.min(), series.index.max(), freq="D")
    return series.reindex(full_range, fill_value=0.0)


def _make_features(history: list[float], next_date: pd.Timestamp) -> dict[str, float]:
    """Feature vector for a single future day."""

    def lag(n: int) -> float:
        return float(history[-n]) if len(history) >= n else 0.0

    def roll(window: int) -> float:
        values = history[-window:] if history else [0.0]
        return float(np.mean(values)) if values else 0.0

    dow = int(next_date.dayofweek)
    return {
        "dow": dow,
        "is_weekend": 1.0 if dow >= 5 else 0.0,
        "lag1": lag(1),
        "lag7": lag(7),
        "roll7": roll(7),
        "roll14": roll(14),
        "roll28": roll(28),
    }


def _build_supervised(series: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Build the supervised learning matrix from a daily series."""

    frame = pd.DataFrame({"y": series.values}, index=series.index)
    frame["dow"] = frame.index.dayofweek
    frame["is_weekend"] = (frame["dow"] >= 5).astype(int)
    frame["lag1"] = frame["y"].shift(1)
    frame["lag7"] = frame["y"].shift(7)
    frame["roll7"] = frame["y"].shift(1).rolling(7).mean()
    frame["roll14"] = frame["y"].shift(1).rolling(14).mean()
    frame["roll28"] = frame["y"].shift(1).rolling(28).mean()
    frame = frame.dropna()

    if frame.empty:
        return frame.reindex(columns=FEATURES), frame.get("y", pd.Series(dtype=float))

    return frame[FEATURES], frame["y"]


def _fit_model(
    x_train: pd.DataFrame, y_train: pd.Series, seed: int
) -> GradientBoostingRegressor:
    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        random_state=seed,
    )
    model.fit(x_train, y_train)
    return model


def _recursive_forecast(
    model: GradientBoostingRegressor,
    history: list[float],
    last_date: pd.Timestamp,
    horizon: int,
) -> list[float]:
    predictions: list[float] = []
    working = list(history)
    for step in range(1, horizon + 1):
        next_date = last_date + pd.Timedelta(days=step)
        row = pd.DataFrame([_make_features(working, next_date)], columns=FEATURES)
        prediction = max(float(model.predict(row)[0]), 0.0)
        predictions.append(prediction)
        working.append(prediction)
    return predictions


@dataclass
class FittedSalesModel:
    """A fitted model plus the metadata needed to score and forecast it."""

    series: pd.Series
    model: GradientBoostingRegressor
    values: list[float] = field(default_factory=list)
    model_mae: float | None = None
    baseline_mae: float = 0.0
    model_mape: float | None = None
    baseline_mape: float = 0.0
    chosen: str = "baseline"
    history_days: int = 0
    backtest_folds: int = 0
    backtest_horizon: int = BACKTEST_HORIZON

    @property
    def mape(self) -> float:
        if self.chosen == "model" and self.model_mape is not None:
            return self.model_mape
        return self.baseline_mape

    def forecast(self, horizon: int) -> dict:
        """Produce a forecast of ``horizon`` days using the chosen method."""

        model_forecast = _recursive_forecast(
            self.model, self.values, self.series.index[-1], horizon
        )
        baseline_values = baseline_forecast(self.values, horizon, season=7)
        forecast_values = (
            model_forecast if self.chosen == "model" else baseline_values
        )

        forecast_dates = pd.date_range(
            self.series.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D"
        )
        forecast = [
            {
                "date": date.strftime("%Y-%m-%d"),
                "predicted_sales": round(float(value), 2),
                "weekday": WEEKDAY_NAMES[date.dayofweek],
            }
            for date, value in zip(forecast_dates, forecast_values)
        ]

        return {
            "insufficient_data": False,
            "reason": None,
            "horizon": horizon,
            "model_mae": self.model_mae,
            "baseline_mae": round(float(self.baseline_mae), 2),
            "mape": round(float(self.mape), 2),
            "model_mape": (
                round(float(self.model_mape), 2)
                if self.model_mape is not None
                else None
            ),
            "baseline_mape": round(float(self.baseline_mape), 2),
            "backtest_folds": self.backtest_folds,
            "backtest_horizon": self.backtest_horizon,
            "chosen": self.chosen,
            "history_days": self.history_days,
            "last_history_date": self.series.index[-1].strftime("%Y-%m-%d"),
            "forecast": forecast,
            "model_forecast": [round(float(v), 2) for v in model_forecast],
            "baseline_forecast": [round(float(v), 2) for v in baseline_values],
        }


def _backtest_cuts(n: int, horizon: int, max_folds: int) -> list[int]:
    cuts = [n - horizon * k for k in range(max_folds, 0, -1)]
    return [cut for cut in cuts if cut >= MIN_BASELINE_HISTORY]


def _walk_forward(
    series: pd.Series,
    values: list[float],
    x_all: pd.DataFrame,
    y_all: pd.Series,
    seed: int,
    horizon: int,
    max_folds: int,
) -> dict:
    """Recursive walk-forward backtest of the model and the baseline."""

    n = len(values)
    first_row = n - len(x_all)
    cuts = _backtest_cuts(n, horizon, max_folds)

    actual_all: list[float] = []
    baseline_all: list[float] = []
    model_actual: list[float] = []
    model_pred: list[float] = []
    model_baseline: list[float] = []
    model_folds = 0

    for cut in cuts:
        actual = values[cut : cut + horizon]
        if not actual:
            continue
        step = len(actual)
        baseline_pred = baseline_forecast(values[:cut], step, season=7)
        actual_all.extend(actual)
        baseline_all.extend(baseline_pred)

        train_rows = cut - first_row
        if train_rows < MIN_TRAIN_ROWS:
            continue
        fold_model = _fit_model(x_all.iloc[:train_rows], y_all.iloc[:train_rows], seed)
        predictions = _recursive_forecast(
            fold_model, values[:cut], series.index[cut - 1], step
        )
        model_actual.extend(actual)
        model_pred.extend(predictions)
        model_baseline.extend(baseline_pred)
        model_folds += 1

    return {
        "actual": actual_all,
        "baseline": baseline_all,
        "model_actual": model_actual,
        "model_pred": model_pred,
        "model_baseline": model_baseline,
        "model_folds": model_folds,
        "folds": len(cuts),
    }


def fit(
    series: pd.Series,
    seed: int = 42,
    min_days: int = MIN_SALES_DAYS,
    max_folds: int = MAX_BACKTEST_FOLDS,
) -> FittedSalesModel | None:
    """Fit the sales model once. Returns ``None`` when history is too short."""

    if series is None or series.empty:
        return None

    if count_sales_days(series) < required_days(min_days):
        return None

    values = [float(v) for v in series.values]
    x_all, y_all = _build_supervised(series)
    if len(x_all) < MIN_TRAIN_ROWS:
        return None

    report = _walk_forward(
        series, values, x_all, y_all, seed, BACKTEST_HORIZON, max_folds
    )

    model_mae: float | None = None
    model_mape: float | None = None
    if report["model_folds"] > 0:
        model_mae = float(
            mean_absolute_error(report["model_actual"], report["model_pred"])
        )
        model_mape = float(
            mean_absolute_percentage_error(
                report["model_actual"], report["model_pred"]
            )
        )
        compare_actual = report["model_actual"]
        compare_baseline = report["model_baseline"]
    else:
        compare_actual = report["actual"]
        compare_baseline = report["baseline"]

    baseline_mae = float(mean_absolute_error(compare_actual, compare_baseline))
    baseline_mape = float(
        mean_absolute_percentage_error(compare_actual, compare_baseline)
    )

    chosen = "model" if model_mae is not None and model_mae <= baseline_mae else "baseline"

    final_model = _fit_model(x_all, y_all, seed)

    return FittedSalesModel(
        series=series,
        model=final_model,
        values=values,
        model_mae=round(model_mae, 2) if model_mae is not None else None,
        baseline_mae=baseline_mae,
        model_mape=model_mape,
        baseline_mape=baseline_mape,
        chosen=chosen,
        history_days=len(values),
        backtest_folds=report["model_folds"] or report["folds"],
    )


def evaluate_and_forecast(
    series: pd.Series, horizon: int, seed: int = 42
) -> dict | None:
    """Convenience wrapper used by the offline training entrypoint."""

    fitted = fit(series, seed=seed)
    if fitted is None:
        return None
    return fitted.forecast(horizon)

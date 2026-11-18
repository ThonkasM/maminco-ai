"""Process-wide forecasting engine.

The engine is a module-level singleton. It loads the historical data and fits
every model **once** (at application startup, or on an explicit ``/train``
call) and then serves all requests from the cached, immutable state. Requests
never touch the database directly and never fit a model.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

from app.core.config import get_settings
from app.db import queries
from app.models import anomalies as anomaly_model
from app.models import peak_hours as peak_hours_model
from app.models import products as products_model
from app.models import sales as sales_model

logger = logging.getLogger(__name__)

DATABASE_UNAVAILABLE = "Database unavailable."


@dataclass
class EngineState:
    daily_sales: pd.DataFrame = field(default_factory=pd.DataFrame)
    product_demand: pd.DataFrame = field(default_factory=pd.DataFrame)
    hourly_demand: pd.DataFrame = field(default_factory=pd.DataFrame)
    daily_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    sales_series: pd.Series = field(default_factory=lambda: pd.Series(dtype="float64"))
    sales_fitted: sales_model.FittedSalesModel | None = None
    products_fitted: list[products_model.FittedProduct] = field(default_factory=list)
    loaded_at: datetime | None = None
    error: str | None = None
    refresh_count: int = 0

    @property
    def history_days(self) -> int:
        return int(len(self.sales_series))

    @property
    def sales_days(self) -> int:
        return sales_model.count_sales_days(self.sales_series)

    @property
    def has_sufficient_sales_data(self) -> bool:
        return self.sales_fitted is not None


class ForecastEngine:
    """Holds the cached data + fitted models for the whole process."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.state = EngineState()

    def refresh(self) -> EngineState:
        """(Re)load data and refit models. Safe to call at startup and /train."""

        settings = get_settings()
        with self._lock:
            try:
                daily_sales = queries.load_daily_sales(settings.history_days)
                product_demand = queries.load_product_demand(settings.history_days)
                hourly_demand = queries.load_hourly_demand(settings.history_days)
                daily_metrics = queries.load_daily_metrics(settings.history_days)

                series = sales_model.to_daily_series(daily_sales)
                sales_fitted = sales_model.fit(
                    series,
                    seed=settings.random_seed,
                    min_days=settings.min_days_required,
                )
                products_fitted = products_model.fit_products(
                    product_demand,
                    limit=50,
                    seed=settings.random_seed,
                    min_days=settings.min_days_required,
                )

                self.state = EngineState(
                    daily_sales=daily_sales,
                    product_demand=product_demand,
                    hourly_demand=hourly_demand,
                    daily_metrics=daily_metrics,
                    sales_series=series,
                    sales_fitted=sales_fitted,
                    products_fitted=products_fitted,
                    loaded_at=datetime.now(timezone.utc),
                    error=None,
                    refresh_count=self.state.refresh_count + 1,
                )
                logger.info(
                    "Forecast engine refreshed: %s history days, %s products fitted",
                    self.state.history_days,
                    len(self.state.products_fitted),
                )
            except Exception:  # noqa: BLE001 - degrade gracefully, never crash
                logger.exception("Forecast engine refresh failed")
                self.state = EngineState(
                    loaded_at=datetime.now(timezone.utc),
                    error=DATABASE_UNAVAILABLE,
                    refresh_count=self.state.refresh_count + 1,
                )

        return self.state

    # ------------------------------------------------------------------
    # Endpoint helpers
    # ------------------------------------------------------------------
    def sales_forecast(self, horizon: int) -> dict:
        settings = get_settings()
        state = self.state
        if state.error:
            return _insufficient(DATABASE_UNAVAILABLE, horizon)
        if not state.has_sufficient_sales_data or state.sales_fitted is None:
            return _insufficient(
                "Not enough closed-order history to forecast "
                f"(need at least {sales_model.required_days(settings.min_days_required)} "
                f"days with sales, have {state.sales_days}).",
                horizon,
            )
        return state.sales_fitted.forecast(horizon)

    def product_forecast(self, horizon: int, limit: int) -> dict:
        state = self.state
        if state.error:
            return _insufficient(DATABASE_UNAVAILABLE, horizon)
        selected = state.products_fitted[:limit]
        return products_model.forecast_products(
            selected, horizon, min_days=get_settings().min_days_required
        )

    def peak_hours(self) -> dict:
        state = self.state
        if state.error:
            return {
                "insufficient_data": True,
                "reason": DATABASE_UNAVAILABLE,
                "profile": [],
            }
        return peak_hours_model.build_peak_hours_profile(state.hourly_demand)

    def anomalies(self) -> dict:
        state = self.state
        if state.error:
            return {
                "insufficient_data": True,
                "reason": DATABASE_UNAVAILABLE,
                "anomalies": [],
            }
        return anomaly_model.detect_anomalies(
            state.daily_sales, min_days=get_settings().min_days_required
        )

    def metrics_summary(self) -> dict:
        state = self.state
        if state.error:
            return {
                "insufficient_data": True,
                "reason": DATABASE_UNAVAILABLE,
            }
        if state.daily_sales is None or state.daily_sales.empty:
            return {
                "insufficient_data": True,
                "reason": "No daily sales history is available.",
            }

        frame = state.daily_sales.copy()
        frame["date"] = pd.to_datetime(frame["date"])
        total_sales = float(frame["total_sales"].sum())
        total_orders = int(frame["total_orders"].sum())

        summary: dict = {
            "insufficient_data": False,
            "reason": None,
            "days": int(len(frame)),
            "first_date": frame["date"].min().strftime("%Y-%m-%d"),
            "last_date": frame["date"].max().strftime("%Y-%m-%d"),
            "total_sales": round(total_sales, 2),
            "total_orders": total_orders,
            "average_daily_sales": round(total_sales / max(len(frame), 1), 2),
            "average_order_value": round(
                total_sales / total_orders if total_orders else 0.0, 2
            ),
            "last_refresh": state.loaded_at.isoformat() if state.loaded_at else None,
            "products_fitted": len(state.products_fitted),
            "sales_model_chosen": (
                state.sales_fitted.chosen if state.sales_fitted else None
            ),
            "sales_model_mae": (
                state.sales_fitted.model_mae if state.sales_fitted else None
            ),
            "baseline_mae": (
                state.sales_fitted.baseline_mae if state.sales_fitted else None
            ),
            "sales_model_mape": (
                state.sales_fitted.mape if state.sales_fitted else None
            ),
            "baseline_mape": (
                state.sales_fitted.baseline_mape if state.sales_fitted else None
            ),
        }

        if not state.product_demand.empty:
            top = (
                state.product_demand.groupby("product_name")["quantity"]
                .sum()
                .sort_values(ascending=False)
                .head(5)
            )
            summary["top_products"] = [
                {"name": name, "quantity": int(quantity)}
                for name, quantity in top.items()
            ]

        return summary


def _insufficient(reason: str, horizon: int) -> dict:
    return {
        "insufficient_data": True,
        "reason": reason,
        "horizon": horizon,
        "forecast": [],
    }


engine = ForecastEngine()

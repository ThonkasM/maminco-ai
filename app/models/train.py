"""Offline training entrypoint.

Usage::

    python -m app.models.train

Loads history from the backend PostgreSQL database, fits the sales and
per-product models, scores them against the seasonal-naive baseline, and
writes JSON artifacts under ``AI_MODEL_DIR`` (default ``data/models``).
Artifacts are plain JSON — never pickle — so they are safe and portable.

The run is deterministic: a fixed random seed is used for every model.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.core.config import get_settings
from app.db import queries
from app.models import anomalies as anomaly_model
from app.models import peak_hours as peak_hours_model
from app.models import products as products_model
from app.models import sales as sales_model

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger("app.models.train")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    raise TypeError(f"Object of type {type(value)!r} is not JSON serialisable")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_json_default, ensure_ascii=False)
    logger.info("Wrote %s", path)


def train() -> int:
    settings = get_settings()
    model_dir = Path(settings.model_dir)
    seed = settings.random_seed

    logger.info("Loading history (%s days) from the database...", settings.history_days)
    try:
        daily_sales = queries.load_daily_sales(settings.history_days)
        product_demand = queries.load_product_demand(settings.history_days)
        hourly_demand = queries.load_hourly_demand(settings.history_days)
    except Exception as exc:  # noqa: BLE001
        logger.error("Could not read the database: %s", type(exc).__name__)
        logger.error("Training aborted; no artifacts were written.")
        return 1

    series = sales_model.to_daily_series(daily_sales)
    history_days = int(len(series))
    sales_days = sales_model.count_sales_days(series)
    required = sales_model.required_days(settings.min_days_required)

    if sales_days < required:
        logger.warning(
            "Insufficient data: %s days with sales available, %s required. "
            "No artifacts written.",
            sales_days,
            required,
        )
        return 0

    fitted = sales_model.fit(
        series, seed=seed, min_days=settings.min_days_required
    )
    if fitted is None:
        logger.warning("Sales model could not be fitted. No artifacts written.")
        return 0

    forecast = fitted.forecast(settings.default_horizon)

    logger.info("=" * 60)
    logger.info("Sales model evaluation (walk-forward, seed=%s)", seed)
    logger.info("  history days : %s", history_days)
    logger.info("  model MAE    : %s", forecast["model_mae"])
    logger.info("  baseline MAE : %s", forecast["baseline_mae"])
    logger.info("  model MAPE   : %s%%", forecast["model_mape"])
    logger.info("  baseline MAPE: %s%%", forecast["baseline_mape"])
    logger.info("  chosen       : %s", forecast["chosen"])
    logger.info("=" * 60)

    product_models = products_model.fit_products(
        product_demand,
        limit=50,
        seed=seed,
        min_days=settings.min_days_required,
    )
    product_forecast = products_model.forecast_products(
        product_models, settings.default_horizon, min_days=settings.min_days_required
    )
    peak_hours = peak_hours_model.build_peak_hours_profile(hourly_demand)
    anomalies = anomaly_model.detect_anomalies(
        daily_sales, min_days=settings.min_days_required
    )

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "history_days": history_days,
        "min_days_required": settings.min_days_required,
        "default_horizon": settings.default_horizon,
        "algorithm": "GradientBoostingRegressor + seasonal-naive baseline",
        "features": sales_model.FEATURES,
    }

    _write_json(model_dir / "metadata.json", metadata)
    _write_json(model_dir / "sales_metrics.json", forecast)
    _write_json(model_dir / "product_forecast.json", product_forecast)
    _write_json(model_dir / "peak_hours.json", peak_hours)
    _write_json(model_dir / "anomalies.json", anomalies)

    logger.info(
        "Training complete: %s products fitted, %s anomalies detected.",
        len(product_forecast.get("forecast", [])),
        anomalies.get("count", 0),
    )
    return 0


if __name__ == "__main__":
    sys.exit(train())

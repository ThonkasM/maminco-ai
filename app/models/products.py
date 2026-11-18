"""Per-product demand forecasting.

Each product is modelled as its own daily demand series. The most popular
products (by quantity over the loaded window) are fitted once and then reused
for any horizon, reusing the sales forecaster so behaviour stays consistent
between endpoints.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from app.models import sales as sales_model
from app.models.sales import to_daily_series


@dataclass
class FittedProduct:
    product_id: str
    product_name: str
    category_name: str | None
    model: sales_model.FittedSalesModel
    recent_quantity: float


def _product_series(product_frame: pd.DataFrame, product_id: str) -> pd.Series:
    subset = product_frame[product_frame["product_id"] == product_id]
    if subset.empty:
        return pd.Series(dtype="float64")
    grouped = subset.groupby("date")["quantity"].sum().sort_index().astype("float64")
    full_range = pd.date_range(grouped.index.min(), grouped.index.max(), freq="D")
    return grouped.reindex(full_range, fill_value=0.0)


def fit_products(
    product_demand: pd.DataFrame,
    limit: int = 20,
    seed: int = 42,
    min_days: int = 14,
) -> list[FittedProduct]:
    """Fit one model per top product (once, at startup)."""

    if product_demand is None or product_demand.empty:
        return []

    frame = product_demand.copy()
    frame["date"] = pd.to_datetime(frame["date"])

    popularity = (
        frame.groupby(["product_id", "product_name"], dropna=False)["quantity"]
        .sum()
        .reset_index()
        .sort_values("quantity", ascending=False)
    )
    category_lookup = (
        frame.dropna(subset=["product_name"])
        .groupby("product_id")["category_name"]
        .first()
        .to_dict()
    )

    fitted: list[FittedProduct] = []
    for _, row in popularity.head(limit).iterrows():
        product_id = row["product_id"]
        series = _product_series(frame, product_id)
        model = sales_model.fit(series, seed=seed, min_days=min_days, max_folds=2)
        if model is None:
            continue

        fitted.append(
            FittedProduct(
                product_id=product_id,
                product_name=row["product_name"],
                category_name=category_lookup.get(product_id),
                model=model,
                recent_quantity=float(row["quantity"]),
            )
        )

    return fitted


def forecast_products(
    fitted: list[FittedProduct],
    horizon: int,
    min_days: int = 14,
) -> dict:
    """Produce forecasts from already-fitted product models."""

    if not fitted:
        return {
            "insufficient_data": True,
            "reason": (
                "No product has enough days with sales to forecast "
                f"(minimum {sales_model.required_days(min_days)} days)."
            ),
            "horizon": horizon,
            "forecast": [],
        }

    results: list[dict] = []
    for product in fitted:
        forecast = product.model.forecast(horizon)
        predicted_daily = [point["predicted_sales"] for point in forecast["forecast"]]
        results.append(
            {
                "product_id": product.product_id,
                "product_name": product.product_name,
                "category_name": product.category_name,
                "predicted_quantity": round(float(sum(predicted_daily)), 2),
                "average_daily_quantity": round(
                    float(sum(predicted_daily) / max(len(predicted_daily), 1)), 2
                ),
                "chosen": forecast["chosen"],
                "daily": predicted_daily,
                "dates": [point["date"] for point in forecast["forecast"]],
            }
        )

    results.sort(key=lambda entry: entry["predicted_quantity"], reverse=True)

    return {
        "insufficient_data": False,
        "reason": None,
        "horizon": horizon,
        "forecast": results,
    }


__all__ = ["FittedProduct", "fit_products", "forecast_products", "to_daily_series"]

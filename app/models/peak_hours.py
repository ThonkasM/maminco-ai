"""Hour-of-day demand profile.

Aggregates historical CLOSED orders by hour of day to describe when the
restaurant is busiest. This is a descriptive profile, not a time-series
forecast: it answers "which hours matter" for staffing and prep.
"""

from __future__ import annotations

import pandas as pd

HOURS_IN_DAY = 24


def build_peak_hours_profile(hourly_demand: pd.DataFrame) -> dict:
    """Return the average hourly demand profile and its peak hour."""

    if hourly_demand is None or hourly_demand.empty:
        return {
            "insufficient_data": True,
            "reason": "No hourly order history is available.",
            "profile": [],
        }

    frame = hourly_demand.copy()
    frame["hour"] = frame["hour"].astype(int)
    frame["orders"] = frame["orders"].astype(float)
    frame["sales"] = frame["sales"].astype(float)

    n_days = int(frame["date"].nunique()) if "date" in frame else 0
    n_days = max(n_days, 1)

    grouped = frame.groupby("hour").agg(
        orders=("orders", "sum"),
        sales=("sales", "sum"),
    )

    total_orders = float(grouped["orders"].sum())
    total_sales = float(grouped["sales"].sum())

    profile: list[dict] = []
    for hour in range(HOURS_IN_DAY):
        orders = float(grouped["orders"].get(hour, 0.0))
        sales = float(grouped["sales"].get(hour, 0.0))
        profile.append(
            {
                "hour": hour,
                "avg_orders": round(orders / n_days, 2),
                "avg_sales": round(sales / n_days, 2),
                "order_share": round(orders / total_orders, 4) if total_orders else 0.0,
                "sales_share": round(sales / total_sales, 4) if total_sales else 0.0,
            }
        )

    active = [entry for entry in profile if entry["avg_orders"] > 0]
    if not active:
        return {
            "insufficient_data": True,
            "reason": "Hourly history exists but contains no orders.",
            "profile": [],
        }

    peak = max(profile, key=lambda entry: entry["avg_orders"])
    busiest = sorted(profile, key=lambda entry: entry["avg_orders"], reverse=True)[:3]

    return {
        "insufficient_data": False,
        "reason": None,
        "days_observed": n_days,
        "peak_hour": peak["hour"],
        "peak_avg_orders": peak["avg_orders"],
        "top_hours": [entry["hour"] for entry in busiest],
        "profile": profile,
    }

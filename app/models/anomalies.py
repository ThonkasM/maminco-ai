"""Anomaly detection on daily sales.

Two complementary, interpretable detectors are combined:

* robust z-score using the median and median absolute deviation (MAD);
* Tukey's IQR fences.

A day is flagged when either detector considers it an outlier. No model
artifacts are required, so this is cheap to run on every request.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

Z_THRESHOLD = 3.5
IQR_MULTIPLIER = 1.5


def detect_anomalies(daily_sales: pd.DataFrame, min_days: int = 14) -> dict:
    """Detect anomalous days in the daily sales series."""

    if daily_sales is None or daily_sales.empty:
        return {
            "insufficient_data": True,
            "reason": "No daily sales history is available.",
            "anomalies": [],
        }

    frame = daily_sales.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date").reset_index(drop=True)
    values = frame["total_sales"].astype(float)

    if len(frame) < min_days:
        return {
            "insufficient_data": True,
            "reason": (
                f"Only {len(frame)} days of history; at least {min_days} are "
                "required for anomaly detection."
            ),
            "anomalies": [],
        }

    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    # 1.4826 scales MAD to be a consistent estimator of the std dev.
    robust_sigma = 1.4826 * mad
    z_scores = (
        (values - median) / robust_sigma
        if robust_sigma > 0
        else pd.Series(np.zeros(len(values)), index=values.index)
    )

    q1 = float(values.quantile(0.25))
    q3 = float(values.quantile(0.75))
    iqr = q3 - q1
    lower_fence = q1 - IQR_MULTIPLIER * iqr
    upper_fence = q3 + IQR_MULTIPLIER * iqr

    anomalies: list[dict] = []
    for index, row in frame.iterrows():
        value = float(row["total_sales"])
        z_score = float(z_scores.iloc[index])
        is_z = abs(z_score) > Z_THRESHOLD
        is_iqr = value < lower_fence or value > upper_fence
        if not (is_z or is_iqr):
            continue

        methods = []
        if is_z:
            methods.append("zscore")
        if is_iqr:
            methods.append("iqr")

        anomalies.append(
            {
                "date": row["date"].strftime("%Y-%m-%d"),
                "value": round(value, 2),
                "z_score": round(z_score, 2),
                "direction": "high" if value > median else "low",
                "methods": methods,
            }
        )

    anomalies.sort(key=lambda entry: abs(entry["z_score"]), reverse=True)

    return {
        "insufficient_data": False,
        "reason": None,
        "method": "robust-zscore+iqr",
        "median": round(median, 2),
        "iqr_lower": round(lower_fence, 2),
        "iqr_upper": round(upper_fence, 2),
        "count": len(anomalies),
        "anomalies": anomalies,
    }

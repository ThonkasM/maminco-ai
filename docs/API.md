# API Reference

Base URL: `http://localhost:8000`. Interactive docs: `/docs`.

## Authentication

All routes except `GET /health` require:

```
X-Internal-Token: <AI_INTERNAL_SERVICE_TOKEN>
```

Missing or incorrect token -> `401 Unauthorized`:

```json
{ "detail": "Invalid or missing X-Internal-Token header." }
```

## Degradation contract

When there is not enough history (the sales forecast needs
`max(AI_MIN_DAYS_REQUIRED, 36)` days *with sales*; anomalies need
`AI_MIN_DAYS_REQUIRED`) or the database is unreachable, endpoints return **HTTP 200**
with:

```json
{ "insufficient_data": true, "reason": "...", "forecast": [] }
```

`forecast` is always present and empty on degradation. `reason` explains why
(a database failure is reported only as `"Database unavailable."`; details go to
the service log). No data is ever fabricated.

Days are bucketed in the `AI_TIMEZONE` time zone (default `America/La_Paz`).

---

## GET /health

No authentication. Always `200`.

```json
{
  "status": "ok",
  "service": "maminco-ai",
  "version": "0.1.0",
  "models_loaded": true,
  "history_days": 120,
  "products_fitted": 50,
  "database_error": null,
  "last_refresh": "2026-01-31T12:00:00+00:00"
}
```

## GET /forecast/sales

| Query | Type | Default | Constraints |
| --- | --- | --- | --- |
| `horizon` | int | 7 | 1..60 |

```json
{
  "insufficient_data": false,
  "reason": null,
  "horizon": 7,
  "model_mae": 42.13,
  "baseline_mae": 58.77,
  "mape": 5.21,
  "model_mape": 5.21,
  "baseline_mape": 8.4,
  "backtest_folds": 3,
  "backtest_horizon": 7,
  "chosen": "model",
  "history_days": 120,
  "last_history_date": "2026-01-31",
  "forecast": [
    { "date": "2026-02-01", "predicted_sales": 812.4, "weekday": "Sunday" }
  ],
  "model_forecast": [812.4],
  "baseline_forecast": [790.1],
  "generated_at": "2026-01-31T12:05:00+00:00"
}
```

## GET /forecast/products

| Query | Type | Default | Constraints |
| --- | --- | --- | --- |
| `horizon` | int | 7 | 1..60 |
| `limit` | int | 20 | 1..100 |

```json
{
  "insufficient_data": false,
  "reason": null,
  "horizon": 7,
  "forecast": [
    {
      "product_id": "uuid",
      "product_name": "Pato Criollo",
      "category_name": "PATO",
      "predicted_quantity": 63.0,
      "average_daily_quantity": 9.0,
      "chosen": "model",
      "daily": [9.0, 8.5, 10.0, 9.2, 8.8, 8.5, 9.0],
      "dates": ["2026-02-01", "..."],
      "generated_at": "2026-01-31T12:05:00+00:00"
    }
  ]
}
```

## GET /forecast/peak-hours

```json
{
  "insufficient_data": false,
  "reason": null,
  "days_observed": 120,
  "peak_hour": 13,
  "peak_avg_orders": 18.4,
  "top_hours": [13, 12, 19],
  "profile": [
    { "hour": 13, "avg_orders": 18.4, "avg_sales": 460.2, "order_share": 0.14, "sales_share": 0.13 }
  ],
  "generated_at": "2026-01-31T12:05:00+00:00"
}
```

## GET /insights/anomalies

```json
{
  "insufficient_data": false,
  "reason": null,
  "method": "robust-zscore+iqr",
  "median": 850.0,
  "iqr_lower": 300.0,
  "iqr_upper": 1500.0,
  "count": 2,
  "anomalies": [
    { "date": "2026-01-15", "value": 2400.0, "z_score": 4.1, "direction": "high", "methods": ["zscore", "iqr"] }
  ],
  "generated_at": "2026-01-31T12:05:00+00:00"
}
```

## GET /metrics/summary

```json
{
  "insufficient_data": false,
  "reason": null,
  "days": 120,
  "first_date": "2025-10-04",
  "last_date": "2026-01-31",
  "total_sales": 102340.5,
  "total_orders": 4093,
  "average_daily_sales": 852.84,
  "average_order_value": 25.0,
  "last_refresh": "2026-01-31T12:00:00+00:00",
  "products_fitted": 50,
  "sales_model_chosen": "model",
  "sales_model_mae": 42.13,
  "baseline_mae": 58.77,
  "sales_model_mape": 5.21,
  "baseline_mape": 8.4,
  "top_products": [{ "name": "Pato Criollo", "quantity": 812 }],
  "generated_at": "2026-01-31T12:05:00+00:00"
}
```

## POST /train

Reloads data and refits models in-process (equivalent to a startup refresh).

```json
{
  "ok": true,
  "message": "Models refreshed.",
  "refresh_count": 2,
  "history_days": 120,
  "products_fitted": 50,
  "sales_model_chosen": "model",
  "sales_model_mae": 42.13,
  "baseline_mae": 58.77,
  "sales_model_mape": 5.21,
  "baseline_mape": 8.4,
  "error": null,
  "generated_at": "2026-01-31T12:05:00+00:00"
}
```

On database failure it returns `ok: false` and the generic error `"Database unavailable."` (still HTTP 200
so the caller can distinguish "no data" from a transport failure).

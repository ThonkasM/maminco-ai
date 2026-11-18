"""Read-only SQL queries returning pandas DataFrames.

Table and column names follow the Prisma mapping in
``maminco-backend/prisma/schema.prisma`` (snake_case tables, camelCase columns
quoted as in the database, PostgreSQL ``OrderStatus`` enum).
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy import text

from app.core.config import get_settings
from app.db.session import get_engine

LOCAL_CLOSED_AT = """((o."closedAt" AT TIME ZONE 'UTC') AT TIME ZONE :tz)"""

CLOSED_ORDER_FILTER = """
    o.status = 'CLOSED'
    AND o."deletedAt" IS NULL
    AND o."closedAt" IS NOT NULL
"""

_DAILY_SALES_SQL = text(
    f"""
    SELECT
        date_trunc('day', {LOCAL_CLOSED_AT})::date AS date,
        COALESCE(SUM(o.total), 0)::float8               AS total_sales,
        COUNT(*)::int                                   AS total_orders,
        COALESCE(SUM(o."tipAmount"), 0)::float8          AS total_tips,
        COALESCE(SUM(o."discountAmount"), 0)::float8     AS total_discount,
        COALESCE(AVG(o.total), 0)::float8               AS average_order_value
    FROM orders o
    WHERE {CLOSED_ORDER_FILTER}
      AND o."closedAt" >= (NOW() AT TIME ZONE 'UTC') - make_interval(days => :days)
    GROUP BY 1
    ORDER BY 1
    """
)

_PRODUCT_DEMAND_SQL = text(
    f"""
    SELECT
        date_trunc('day', {LOCAL_CLOSED_AT})::date AS date,
        p.id                                            AS product_id,
        p.name                                          AS product_name,
        c.name                                          AS category_name,
        SUM(oi.quantity)::int                           AS quantity,
        COALESCE(SUM(oi.subtotal), 0)::float8           AS revenue
    FROM order_items oi
    JOIN orders o   ON o.id = oi."orderId"
    JOIN products p ON p.id = oi."productId"
    LEFT JOIN categories c ON c.id = p."categoryId"
    WHERE {CLOSED_ORDER_FILTER}
      AND o."closedAt" >= (NOW() AT TIME ZONE 'UTC') - make_interval(days => :days)
    GROUP BY 1, 2, 3, 4
    ORDER BY 1
    """
)

_HOURLY_DEMAND_SQL = text(
    f"""
    SELECT
        date_trunc('day', {LOCAL_CLOSED_AT})::date AS date,
        EXTRACT(HOUR FROM {LOCAL_CLOSED_AT})::int AS hour,
        COUNT(*)::int                                   AS orders,
        COALESCE(SUM(o.total), 0)::float8               AS sales
    FROM orders o
    WHERE {CLOSED_ORDER_FILTER}
      AND o."closedAt" >= (NOW() AT TIME ZONE 'UTC') - make_interval(days => :days)
    GROUP BY 1, 2
    ORDER BY 1, 2
    """
)

_DAILY_METRICS_SQL = text(
    """
    SELECT
        dm.date                                         AS date,
        dm."totalSales"::float8                          AS total_sales,
        dm."totalOrders"::int                            AS total_orders,
        dm."averageOrderValue"::float8                  AS average_order_value,
        dm."mesaOrders"::int                             AS mesa_orders,
        dm."deliveryOrders"::int                         AS delivery_orders,
        dm."recogerOrders"::int                          AS recoger_orders,
        dm."cashPayments"::float8                        AS cash_payments,
        dm."cardPayments"::float8                        AS card_payments,
        dm."transferPayments"::float8                        AS transfer_payments
    FROM daily_metrics dm
    WHERE dm.date >= ((NOW() AT TIME ZONE :tz)::date - make_interval(days => :days))
    ORDER BY dm.date
    """
)


def _read(sql, params: dict) -> pd.DataFrame:
    """Execute a read-only statement and return a DataFrame."""

    params = {**params, "tz": get_settings().timezone}
    engine = get_engine()
    with engine.connect() as connection:
        return pd.read_sql(sql, connection, params=params)


def load_daily_sales(days: int = 180) -> pd.DataFrame:
    """Daily aggregated sales for CLOSED orders.

    Returns columns ``date, total_sales, total_orders, total_tips,
    total_discount, average_order_value``. Empty when no data is available.
    """

    frame = _read(_DAILY_SALES_SQL, {"days": int(days)})
    if not frame.empty:
        frame["date"] = pd.to_datetime(frame["date"])
    return frame


def load_product_demand(days: int = 180) -> pd.DataFrame:
    """Daily demand per product for CLOSED orders.

    Returns columns ``date, product_id, product_name, category_name, quantity,
    revenue``.
    """

    frame = _read(_PRODUCT_DEMAND_SQL, {"days": int(days)})
    if not frame.empty:
        frame["date"] = pd.to_datetime(frame["date"])
    return frame


def load_hourly_demand(days: int = 180) -> pd.DataFrame:
    """Order counts and sales per hour of day.

    Returns columns ``date, hour, orders, sales``.
    """

    frame = _read(_HOURLY_DEMAND_SQL, {"days": int(days)})
    if not frame.empty:
        frame["date"] = pd.to_datetime(frame["date"])
    return frame


def load_daily_metrics(days: int = 180) -> pd.DataFrame:
    """Pre-aggregated ``daily_metrics`` rows when the backend has populated them."""

    frame = _read(_DAILY_METRICS_SQL, {"days": int(days)})
    if not frame.empty:
        frame["date"] = pd.to_datetime(frame["date"])
    return frame

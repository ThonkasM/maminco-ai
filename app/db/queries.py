"""Read-only SQL queries returning pandas DataFrames.

Table and column names follow the Prisma mapping in
``maminco-backend/prisma/schema.prisma`` (snake_case tables, camelCase columns
mapped to snake_case, PostgreSQL ``OrderStatus`` enum).
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy import text

from app.db.session import get_engine

CLOSED_ORDER_FILTER = """
    o.status = 'CLOSED'
    AND o.deleted_at IS NULL
    AND o.closed_at IS NOT NULL
"""

_DAILY_SALES_SQL = text(
    f"""
    SELECT
        date_trunc('day', o.closed_at)::date            AS date,
        COALESCE(SUM(o.total), 0)::float8               AS total_sales,
        COUNT(*)::int                                   AS total_orders,
        COALESCE(SUM(o.tip_amount), 0)::float8          AS total_tips,
        COALESCE(SUM(o.discount_amount), 0)::float8     AS total_discount,
        COALESCE(AVG(o.total), 0)::float8               AS average_order_value
    FROM orders o
    WHERE {CLOSED_ORDER_FILTER}
      AND o.closed_at >= NOW() - make_interval(days => :days)
    GROUP BY 1
    ORDER BY 1
    """
)

_PRODUCT_DEMAND_SQL = text(
    f"""
    SELECT
        date_trunc('day', o.closed_at)::date            AS date,
        p.id                                            AS product_id,
        p.name                                          AS product_name,
        c.name                                          AS category_name,
        SUM(oi.quantity)::int                           AS quantity,
        COALESCE(SUM(oi.subtotal), 0)::float8           AS revenue
    FROM order_items oi
    JOIN orders o   ON o.id = oi.order_id
    JOIN products p ON p.id = oi.product_id
    LEFT JOIN categories c ON c.id = p.category_id
    WHERE {CLOSED_ORDER_FILTER}
      AND o.closed_at >= NOW() - make_interval(days => :days)
    GROUP BY 1, 2, 3, 4
    ORDER BY 1
    """
)

_HOURLY_DEMAND_SQL = text(
    f"""
    SELECT
        date_trunc('day', o.closed_at)::date            AS date,
        EXTRACT(HOUR FROM o.closed_at)::int             AS hour,
        COUNT(*)::int                                   AS orders,
        COALESCE(SUM(o.total), 0)::float8               AS sales
    FROM orders o
    WHERE {CLOSED_ORDER_FILTER}
      AND o.closed_at >= NOW() - make_interval(days => :days)
    GROUP BY 1, 2
    ORDER BY 1, 2
    """
)

_DAILY_METRICS_SQL = text(
    """
    SELECT
        dm.date                                         AS date,
        dm.total_sales::float8                          AS total_sales,
        dm.total_orders::int                            AS total_orders,
        dm.average_order_value::float8                  AS average_order_value,
        dm.mesa_orders::int                             AS mesa_orders,
        dm.delivery_orders::int                         AS delivery_orders,
        dm.recoger_orders::int                          AS recoger_orders,
        dm.cash_payments::float8                        AS cash_payments,
        dm.card_payments::float8                        AS card_payments,
        dm.transfer_payments::float8                        AS transfer_payments
    FROM daily_metrics dm
    WHERE dm.date >= (CURRENT_DATE - make_interval(days => :days))
    ORDER BY dm.date
    """
)


def _read(sql, params: dict) -> pd.DataFrame:
    """Execute a read-only statement and return a DataFrame."""

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

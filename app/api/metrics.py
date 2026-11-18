"""``/metrics/summary`` — high-level descriptive metrics."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.core.security import verify_internal_token
from app.models.engine import engine

router = APIRouter(
    prefix="/metrics",
    tags=["metrics"],
    dependencies=[Depends(verify_internal_token)],
)


class TopProduct(BaseModel):
    name: str
    quantity: int


class MetricsSummaryResponse(BaseModel):
    insufficient_data: bool = False
    reason: str | None = None
    days: int | None = None
    first_date: str | None = None
    last_date: str | None = None
    total_sales: float | None = None
    total_orders: int | None = None
    average_daily_sales: float | None = None
    average_order_value: float | None = None
    last_refresh: str | None = None
    products_fitted: int | None = None
    sales_model_chosen: str | None = None
    sales_model_mae: float | None = None
    baseline_mae: float | None = None
    sales_model_mape: float | None = None
    baseline_mape: float | None = None
    top_products: list[TopProduct] = Field(default_factory=list)
    generated_at: str | None = None


@router.get("/summary", response_model=MetricsSummaryResponse)
def get_metrics_summary() -> MetricsSummaryResponse:
    data = engine.metrics_summary()
    data["generated_at"] = datetime.now(timezone.utc).isoformat()
    return MetricsSummaryResponse(**data)

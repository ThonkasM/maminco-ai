"""``/forecast/sales`` — daily sales forecast."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from app.core.security import verify_internal_token
from app.models.engine import engine

router = APIRouter(
    prefix="/forecast",
    tags=["forecast"],
    dependencies=[Depends(verify_internal_token)],
)


class SalesForecastPoint(BaseModel):
    date: str
    predicted_sales: float
    weekday: str


class SalesForecastResponse(BaseModel):
    insufficient_data: bool = False
    reason: str | None = None
    horizon: int = 0
    model_mae: float | None = None
    baseline_mae: float | None = None
    mape: float | None = None
    chosen: str | None = None
    history_days: int | None = None
    last_history_date: str | None = None
    forecast: list[SalesForecastPoint] = Field(default_factory=list)
    model_forecast: list[float] = Field(default_factory=list)
    baseline_forecast: list[float] = Field(default_factory=list)
    generated_at: str | None = None


@router.get("/sales", response_model=SalesForecastResponse)
def get_sales_forecast(
    horizon: int = Query(default=7, ge=1, le=60, description="Days to forecast."),
) -> SalesForecastResponse:
    data = engine.sales_forecast(horizon)
    data["generated_at"] = datetime.now(timezone.utc).isoformat()
    return SalesForecastResponse(**data)

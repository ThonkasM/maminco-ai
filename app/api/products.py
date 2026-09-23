"""``/forecast/products`` — per-product demand forecast."""

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


class ProductForecastPoint(BaseModel):
    product_id: str
    product_name: str
    category_name: str | None = None
    predicted_quantity: float
    average_daily_quantity: float
    chosen: str
    daily: list[float] = Field(default_factory=list)
    dates: list[str] = Field(default_factory=list)


class ProductForecastResponse(BaseModel):
    insufficient_data: bool = False
    reason: str | None = None
    horizon: int = 0
    forecast: list[ProductForecastPoint] = Field(default_factory=list)
    generated_at: str | None = None


@router.get("/products", response_model=ProductForecastResponse)
def get_products_forecast(
    horizon: int = Query(default=7, ge=1, le=60, description="Days to forecast."),
    limit: int = Query(default=20, ge=1, le=100, description="Number of products."),
) -> ProductForecastResponse:
    data = engine.product_forecast(horizon, limit)
    data["generated_at"] = datetime.now(timezone.utc).isoformat()
    return ProductForecastResponse(**data)

"""``/forecast/peak-hours`` — hour-of-day demand profile."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.core.security import verify_internal_token
from app.models.engine import engine

router = APIRouter(
    prefix="/forecast",
    tags=["forecast"],
    dependencies=[Depends(verify_internal_token)],
)


class PeakHourPoint(BaseModel):
    hour: int
    avg_orders: float
    avg_sales: float
    order_share: float
    sales_share: float


class PeakHoursResponse(BaseModel):
    insufficient_data: bool = False
    reason: str | None = None
    days_observed: int | None = None
    peak_hour: int | None = None
    peak_avg_orders: float | None = None
    top_hours: list[int] = Field(default_factory=list)
    profile: list[PeakHourPoint] = Field(default_factory=list)
    generated_at: str | None = None


@router.get("/peak-hours", response_model=PeakHoursResponse)
def get_peak_hours() -> PeakHoursResponse:
    data = engine.peak_hours()
    data["generated_at"] = datetime.now(timezone.utc).isoformat()
    return PeakHoursResponse(**data)

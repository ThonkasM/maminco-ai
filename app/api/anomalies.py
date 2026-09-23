"""``/insights/anomalies`` — anomalous sales days."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.core.security import verify_internal_token
from app.models.engine import engine

router = APIRouter(
    prefix="/insights",
    tags=["insights"],
    dependencies=[Depends(verify_internal_token)],
)


class AnomalyPoint(BaseModel):
    date: str
    value: float
    z_score: float
    direction: str
    methods: list[str] = Field(default_factory=list)


class AnomaliesResponse(BaseModel):
    insufficient_data: bool = False
    reason: str | None = None
    method: str | None = None
    median: float | None = None
    iqr_lower: float | None = None
    iqr_upper: float | None = None
    count: int = 0
    anomalies: list[AnomalyPoint] = Field(default_factory=list)
    generated_at: str | None = None


@router.get("/anomalies", response_model=AnomaliesResponse)
def get_anomalies() -> AnomaliesResponse:
    data = engine.anomalies()
    data["generated_at"] = datetime.now(timezone.utc).isoformat()
    return AnomaliesResponse(**data)

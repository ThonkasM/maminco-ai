"""``/train`` — trigger an in-process refresh + refit."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.core.security import verify_internal_token
from app.models.engine import engine

router = APIRouter(tags=["training"], dependencies=[Depends(verify_internal_token)])


class TrainResponse(BaseModel):
    ok: bool
    message: str
    refresh_count: int = 0
    history_days: int = 0
    products_fitted: int = 0
    sales_model_chosen: str | None = None
    sales_model_mae: float | None = None
    baseline_mae: float | None = None
    sales_model_mape: float | None = None
    baseline_mape: float | None = None
    error: str | None = None
    generated_at: str | None = None


@router.post("/train", response_model=TrainResponse)
def train() -> TrainResponse:
    state = engine.refresh()
    if state.error:
        return TrainResponse(
            ok=False,
            message="Training failed; the database may be unavailable.",
            refresh_count=state.refresh_count,
            history_days=state.history_days,
            error=state.error,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

    return TrainResponse(
        ok=True,
        message="Models refreshed.",
        refresh_count=state.refresh_count,
        history_days=state.history_days,
        products_fitted=len(state.products_fitted),
        sales_model_chosen=state.sales_fitted.chosen if state.sales_fitted else None,
        sales_model_mae=state.sales_fitted.model_mae if state.sales_fitted else None,
        baseline_mae=state.sales_fitted.baseline_mae if state.sales_fitted else None,
        sales_model_mape=state.sales_fitted.mape if state.sales_fitted else None,
        baseline_mape=(
            state.sales_fitted.baseline_mape if state.sales_fitted else None
        ),
        generated_at=datetime.now(timezone.utc).isoformat(),
    )

"""FastAPI application entrypoint.

Run locally with::

    uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api import anomalies, metrics, peak_hours, products, sales, training
from app.core.config import get_settings
from app.db.session import dispose_engine
from app.models.engine import engine

logger = logging.getLogger(__name__)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    logger.info("Starting maminco-ai v%s", __version__)

    # Load data and fit models exactly once for the lifetime of the process.
    # Failures are swallowed inside the engine so startup never crashes when
    # the database is unavailable.
    engine.refresh()

    yield

    dispose_engine()
    logger.info("Shutting down maminco-ai")


def create_app() -> FastAPI:
    settings = get_settings()

    application = FastAPI(
        title="Maminco AI",
        description="Sales forecasting and insights microservice for the Maminco POS.",
        version=__version__,
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    application.include_router(sales.router)
    application.include_router(products.router)
    application.include_router(peak_hours.router)
    application.include_router(anomalies.router)
    application.include_router(metrics.router)
    application.include_router(training.router)

    @application.get("/health", tags=["system"])
    def health() -> dict:
        """Liveness probe. Does not require the internal token."""

        state = engine.state
        return {
            "status": "ok",
            "service": "maminco-ai",
            "version": __version__,
            "models_loaded": state.has_sufficient_sales_data,
            "history_days": state.history_days,
            "products_fitted": len(state.products_fitted),
            "database_error": state.error,
            "last_refresh": state.loaded_at.isoformat() if state.loaded_at else None,
        }

    return application


app = create_app()

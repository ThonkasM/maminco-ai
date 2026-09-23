# AGENTS.md

FastAPI + scikit-learn forecasting microservice for Maminco. Read-only access
to the backend PostgreSQL database. Port **8000**.

## Commands

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt

uvicorn app.main:app --reload --port 8000   # dev server
pytest -q                                    # tests (no live DB required)
python -m app.models.train                   # offline fit + JSON artifacts
```

## Conventions

- Env vars use the `AI_` prefix (pydantic-settings, `env_file=".env"`).
  Key ones: `AI_DATABASE_URL`, `AI_INTERNAL_SERVICE_TOKEN`, `AI_LOG_LEVEL`,
  `AI_MODEL_DIR`.
- All routes except `/health` require header `X-Internal-Token`
  (`app/core/security.py`).
- Models are loaded/fitted **once** at startup in the `engine` singleton
  (`app/models/engine.py`). Never fit models per request.
- DB access only through `app/db/queries.py` (read-only, pandas DataFrames).
- Endpoints degrade gracefully: HTTP 200 + `insufficient_data: true` when data
  is missing. Never fabricate or seed data.
- Artifacts are JSON under `data/models/` (gitignored). **Never pickle.**
- snake_case modules; type hints everywhere; `from __future__ import annotations`.

## Layout

`app/core/`, `app/db/`, `app/models/`, `app/api/`; tests under `tests/`;
docs under `docs/`.

## Gotchas

- `daily_metrics` may be empty if the backend hasn't populated it; forecasts
  are computed from `orders` / `order_items` directly.
- Money columns are `numeric`; SQL casts them to `float8` for pandas.
- `OrderStatus` enum values are `DRAFT`, `CLOSED`, `CANCELLED`.

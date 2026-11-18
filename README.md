# maminco-ai

FastAPI microservice that provides **sales forecasting and insights** for the
Maminco restaurant POS. It reads the **same PostgreSQL database** owned by the
NestJS backend (`maminco-backend`) in **read-only** mode. It never writes and
never seeds data.

- **Port:** `8000`
- **Auth:** every route except `/health` requires the header `X-Internal-Token`
  matching `AI_INTERNAL_SERVICE_TOKEN`. The backend proxy attaches it. The token
  has no default: the service refuses to boot if it is missing, shorter than 32
  characters or a placeholder (`change-me`).
- **Time zone:** `AI_TIMEZONE` (default `America/La_Paz`, same as the backend
  `REPORT_TIMEZONE`) decides which business day/hour an order belongs to.
- **Graceful degradation:** with too little history (the sales forecast needs at
  least 36 days *with sales*, see below) or an unreachable database endpoints
  return HTTP `200` with
  `{"insufficient_data": true, "reason": "...", "forecast": []}`. No synthetic
  data is ever fabricated.

## Endpoints

| Method | Path | Description |
| --- | --- | --- |
| GET | `/health` | Liveness probe (no auth). |
| GET | `/forecast/sales?horizon=7` | Daily sales forecast. |
| GET | `/forecast/products?horizon=7&limit=20` | Per-product demand forecast. |
| GET | `/forecast/peak-hours` | Hour-of-day demand profile. |
| GET | `/insights/anomalies` | Anomalous sales days. |
| GET | `/metrics/summary` | Descriptive metrics summary. |
| POST | `/train` | Reload data and refit models in-process. |

See [`docs/API.md`](docs/API.md) for full request/response schemas and
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the design.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt

cp .env.example .env          # set AI_DATABASE_URL and AI_INTERNAL_SERVICE_TOKEN (openssl rand -hex 32)

uvicorn app.main:app --reload --port 8000
```

OpenAPI docs are served at <http://localhost:8000/docs>.

## Offline training

Fits the models, scores them against the seasonal-naive baseline, and writes
JSON artifacts (never pickle) under `AI_MODEL_DIR` (`data/models/`):

```bash
python -m app.models.train
```

The run is deterministic (fixed seed, CPU only).

## Tests

```bash
pytest -q
```

Tests never require a live database: the query layer is mocked.

## Docker

This repo has **no Dockerfile**: the single source of truth is
`maminco-infra/docker/ai.Dockerfile` (Python 3.12, multi-stage, non-root, tini,
healthcheck on `/health`), built with this repo as context:

```bash
docker build -f ../maminco-infra/docker/ai.Dockerfile -t maminco-ai .
docker run --rm -p 8000:8000 --env-file .env maminco-ai
```

`requirements.txt` is pinned; bump versions deliberately (Dependabot opens PRs).

## Forecast quality

The sales model is compared with the baseline using a **walk-forward recursive
backtest** (up to 3 folds of 7 days; the model is refit before each origin and
feeds its own predictions back as lags, like in production). `mape` is the MAPE
of the chosen method; `baseline_mape` and `model_mape` are reported separately.
The sales forecast needs at least 36 days with sales (28-day lags); below that
the endpoint answers `insufficient_data`.

## Layout

```
app/
  main.py            FastAPI app + lifespan (models loaded once)
  core/              config (pydantic-settings) + internal-token auth
  db/                SQLAlchemy engine + read-only pandas queries
  models/            baseline, sales, products, peak_hours, anomalies, train, engine
  api/               routers with Pydantic response models
tests/               pytest suite (no live DB)
docs/                architecture + API reference
```

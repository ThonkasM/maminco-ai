# maminco-ai

FastAPI microservice that provides **sales forecasting and insights** for the
Maminco restaurant POS. It reads the **same PostgreSQL database** owned by the
NestJS backend (`maminco-backend`) in **read-only** mode. It never writes and
never seeds data.

- **Port:** `8000`
- **Auth:** every route except `/health` requires the header `X-Internal-Token`
  matching `AI_INTERNAL_SERVICE_TOKEN`. The backend proxy attaches it.
- **Graceful degradation:** with fewer than ~14 days of `CLOSED` orders (or an
  unreachable database) endpoints return HTTP `200` with
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

cp .env.example .env          # then edit AI_DATABASE_URL / AI_INTERNAL_SERVICE_TOKEN

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

```bash
docker build -t maminco-ai .
docker run --rm -p 8000:8000 --env-file .env maminco-ai
```

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

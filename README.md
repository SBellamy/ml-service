# ml-service

A small MLOps-style service with two containers:
- `trainer`: runs the training pipeline against a CSV file.
- `api`: serves predictions from the promoted model.

Both services share a Docker volume for model artifacts.

## Architecture

- `pipeline/run.py` executes: validate -> train -> evaluate (F1) -> stage artifacts -> promote if score improves.
- `api/main.py` loads `models/production` artifacts and exposes inference endpoints.
- Shared volume: `artifacts`.

## Repository Layout

- `api/main.py`: FastAPI app and routes.
- `api/schemas.py`: request/response models.
- `api/model_store.py`: thread-safe model loading and prediction.
- `pipeline/validate.py`: input data validation.
- `pipeline/train.py`: train/test split and model training.
- `pipeline/evaluate.py`: F1 computation.
- `pipeline/run.py`: end-to-end training and promotion logic.
- `pipeline/promote.py`: placeholder for atomic promotion helper (currently unused).
- `docker/api.Dockerfile`: API image.
- `docker/trainer.Dockerfile`: trainer image.
- `docker-compose.yml`: service wiring and shared volume config.

## Data Contract

Training CSV must contain:
- `age`
- `income`
- `account_balance`
- `transactions_last_30d`
- `is_premium`
- `target`

Validation enforced by `pipeline/validate.py`:
- Required columns present.
- No nulls.
- `target` contains only `0` or `1`.

## Docker Compose Workflow

### 1. Build images

```bash
docker compose build
```

### 2. Run a training job

```bash
docker compose run --rm trainer --csv /data/day_01.csv
```

Notes:
- `trainer` mounts `./data` as read-only at `/data`.
- `pipeline.run` requires the `--csv` argument.
- Model artifacts are written to the shared `artifacts` volume at `/artifacts/models/...`.

### 3. Start API

```bash
docker compose up -d api
```

### 4. Check health/readiness

```bash
curl http://localhost:8000/healthz
curl http://localhost:8000/readyz
```

If `readyz` is `false`, run training first or reload the model after training:

```bash
curl -X POST http://localhost:8000/model/reload
```

### 5. Run prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 42,
    "income": 85000,
    "account_balance": 12000,
    "transactions_last_30d": 18,
    "is_premium": 1
  }'
```

Expected response shape:

```json
{
  "prediction": 1,
  "probability": 0.73
}
```

## Artifacts and Promotion

The trainer writes:
- `models/staging/model.joblib`
- `models/staging/metadata.json`

Then it compares staged F1 vs current production F1:
- Promote when `new_score > current_score`.
- Reject otherwise.

On promotion, production artifacts are replaced under:
- `models/production/model.joblib`
- `models/production/metadata.json`

## Useful Commands

Run training on a different dataset:

```bash
docker compose run --rm trainer --csv /data/day_12.csv
```

Follow API logs:

```bash
docker compose logs -f api
```

Stop services:

```bash
docker compose down
```

Stop and remove volume (clears saved models):

```bash
docker compose down -v
```

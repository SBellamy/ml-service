# ML Service

A small ML service that trains a tabular model, conditionally promotes it to production,
and serves predictions via an API.

Containerized with Docker Compose. API and trainer services share a Docker volume for model artifacts.

Quickstart:

```bash
make build && make train && make serve && make predict
```

## Architecture

- Trainer validates CSV data, trains/evaluates a model, and promotes only on improved F1.
- API loads the production model and serves `/healthz`, `/readyz`, `/model/reload`, and `/predict`.
- Both services use a shared `artifacts` volume for model handoff.

## Repository Layout

- `api/`: FastAPI app, schemas, and model loading logic.
- `pipeline/`: validation, training, evaluation, and promotion flow.
- `docker/`: Dockerfiles for API, trainer, and test containers.
- `data/`: sample CSV datasets.
- `docker-compose.yml`: service and volume wiring.
- `Makefile`: main entrypoint for build/train/serve/test/cleanup commands.

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

## Usage Workflow

`Makefile` is the primary interface for common tasks. Targets wrap the existing Docker Compose commands.

### 1. Build images

```bash
make build
```

### 2. Configure environment

`docker-compose.yml` loads `ml-service.env` for both `trainer` and `api`.

Example:

```env
ARTIFACTS_DIR=/artifacts
BALANCED_CLASS_WEIGHT=True
```

`BALANCED_CLASS_WEIGHT` behavior:
- `True`: trainer uses `LogisticRegression(class_weight="balanced")`.
- Any other value: trainer uses default class weights (`class_weight=None`).

### 3. Run a training job

```bash
make train
```

Notes:
- `trainer` mounts `./data` as read-only at `/data`.
- Model artifacts are written to the shared `artifacts` volume at `/artifacts/models/...`.
- Default dataset is `day_01.csv`.
- Override dataset with `make train DAY=day_12.csv`.

### 4. Start API

```bash
make serve
```

### 5. Check health/readiness

```bash
make health
make ready
```

If `readyz` is `false`, run training first or reload the model after training:

```bash
make reload
```

### 6. Run prediction

```bash
make predict
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

Run full test suite in container:

```bash
make test
```

Follow API logs:

```bash
make logs
```

Show service status:

```bash
make ps
```

Stop services:

```bash
make stop
```

Stop and remove containers/networks:

```bash
make cleanup
```

Stop and remove everything, including the artifacts volume:

```bash
make nuke
```

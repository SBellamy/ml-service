# ML Service

A small ML service that trains a tabular model, promotes better versions, and serves predictions via an API.

Everything runs in Docker Compose. Trainer and API share a volume for model artifacts.

Quickstart:

```bash
make build && make train && make serve && make predict
```

## Architecture

- Trainer validates CSV data, trains/evaluates a model, and promotes only when F1 improves.
- Promotions are atomic: models are written to `models/versions/<version_id>` and `models/CURRENT` is atomically swapped.
- API auto-refreshes to the latest promoted version by following `models/CURRENT`.

## Repo Layout

- `api/`: FastAPI app, schemas, model store.
- `pipeline/`: validate/train/evaluate/promote logic.
- `tests/`: API and pipeline tests.
- `docker/`: API, trainer, and test Dockerfiles.
- `data/`: sample CSV datasets.
- `docker-compose.yml`: services + shared volume wiring.
- `Makefile`: day-to-day commands.

## Data Contract

Training CSV must include:
- `age`
- `income`
- `account_balance`
- `transactions_last_30d`
- `is_premium`
- `target`

Validation rules:
- required columns present
- no nulls
- `target` is binary (`0/1`)

## Usage Workflow

### 1. Build

```bash
make build
```

### 2. Configure env

`docker-compose.yml` loads `ml-service.env` for `trainer` and `api`.

Example:

```env
ARTIFACTS_DIR=/artifacts
BALANCED_CLASS_WEIGHT=True
```

`BALANCED_CLASS_WEIGHT=True` enables `class_weight="balanced"` in logistic regression.

### 3. Train

```bash
make train
```

Notes:
- default dataset is `day_01.csv`
- override dataset with `make train DAY=day_12.csv`
- artifacts go to `/artifacts/models/...`

### 4. Serve API

```bash
make serve
```

### 5. Check status

```bash
make health
make ready
```

If `ready` is false, train a model first. `make reload` is available as a forced refresh, but normal promotions are picked up automatically.

### 6. Predict

```bash
make predict
```

Example response:

```json
{
  "prediction": 1,
  "probability": 0.73
}
```

## Model Metadata Schema

Each promoted version stores `metadata.json` with:
- `model_version_id`
- `metric.name` and `metric.value`
- `training_data_file`
- `trained_at`
- `feature_schema_version`
- `schema_version`

## Artifacts + Promotion

Layout:
- `models/CURRENT` (active version id)
- `models/versions/<version_id>/model.joblib`
- `models/versions/<version_id>/metadata.json`

Promotion policy:
- promote when `new_f1 > current_f1`
- reject otherwise
- rejected runs still keep their version directory for rollback/history

## Useful Commands

```bash
make test       # run full test suite in test container
make logs       # tail API logs
make ps         # show compose service state
make stop       # stop services
make cleanup    # stop + remove containers/networks
make nuke       # cleanup + remove artifacts volume
```

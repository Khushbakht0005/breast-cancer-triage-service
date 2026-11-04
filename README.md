# Breast Cancer Triage Service

FastAPI prediction service for the Meridian Diagnostics Lab scenario. It classifies
cell-nucleus measurements (30 features) as likely **malignant** (positive class,
target 0) or **benign**. The training data is versioned with DVC and the service is
packaged as a Docker image published on Docker Hub.
Educational project, not a medical device.

- GitHub: https://github.com/Khushbakht0005/breast-cancer-triage-service
- Docker Hub: https://hub.docker.com/r/khushbakht91/breast-cancer-triage-api

## Requirements
Docker (Compose v2, `docker compose`), Python 3.12, Git. No GPU or cloud account needed.

## 1. Clone and set up the environment
```bash
git clone https://github.com/Khushbakht0005/breast-cancer-triage-service.git
cd breast-cancer-triage-service
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements-dev.txt -r requirements-serve.txt
```

## 2. Restore the data (DVC)
The dataset is not stored in Git, only the pointer `data/raw/breast_cancer.csv.dvc`.
The local DVC remote is the committed folder `dvc-remote/`.
```bash
dvc pull
wc -l data/raw/breast_cancer.csv    # 540 lines = dataset v2 (539 rows + header)
```
Dataset versions: tag `data-v1` (569 rows) and `data-v2` (539 rows, used by everything else).
```bash
git checkout data-v1 -- data/raw/breast_cancer.csv.dvc && dvc checkout   # switch to v1
git checkout data-v2 -- data/raw/breast_cancer.csv.dvc && dvc checkout   # back to v2
```
To regenerate the raw CSV from scikit-learn: `python -m training.prepare_data`.

## 3. Train (DVC pipeline)
Seed 42 and all hyperparameters are in `params.yaml`.
```bash
dvc repro
dvc metrics show
```
Outputs: `artifacts/model.joblib`, `artifacts/model_card.json`, `artifacts/metrics.json`.
Training exits with a non-zero status if test malignant recall is below 0.95.

Run the training script directly:
```bash
python -m training.train --data-path data/raw/breast_cancer.csv \
  --output-dir artifacts --model-version 1.0.0
```

## 4. Run the API locally (no Docker)
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
Interactive docs: http://localhost:8000/docs

```bash
curl -s localhost:8000/health
curl -s -X POST localhost:8000/predict -H "Content-Type: application/json" -d @sample.json
curl -s localhost:8000/model/info
curl -s localhost:8000/stats
```

## 5. Build the images
Run `dvc pull` first: the training stage reads the dataset from the build context.
```bash
# Baseline (deliberately naive)
docker build -f Dockerfile.naive -t naive .

# Production (multi-stage, non-root, healthcheck)
docker build \
  --build-arg MODEL_VERSION=1.0.0 \
  --build-arg GIT_SHA=$(git rev-parse --short HEAD) \
  -t breast-cancer-triage-api:1.0.0 .
```
Run the production image on its own (Redis is optional):
```bash
docker run -d --name api -p 8000:8000 breast-cancer-triage-api:1.0.0
docker run -d --name api2 -p 8001:8000 -e DECISION_THRESHOLD=0.3 -e LOG_LEVEL=DEBUG \
  breast-cancer-triage-api:1.0.0
docker rm -f api api2
```

## 6. Docker Compose stack (api + Redis)
```bash
cp .env.example .env
# edit .env: set DOCKERHUB_USER=khushbakht91 and GIT_SHA=$(git rev-parse --short HEAD)
docker compose up -d --build
docker compose ps
curl -s -X POST localhost:8000/predict -H "Content-Type: application/json" -d @sample.json
curl -s localhost:8000/stats
docker compose down        # keeps the redis-data volume (counter persists)
docker compose down -v     # also removes the volume (counter resets to 0)
```
Only `api` publishes a port (`HOST_PORT`). It reaches Redis by the service name `redis`.

## 7. Push to Docker Hub
```bash
docker login -u khushbakht91
docker tag breast-cancer-triage-api:1.0.0 khushbakht91/breast-cancer-triage-api:1.0.0
docker tag breast-cancer-triage-api:1.0.0 khushbakht91/breast-cancer-triage-api:latest
docker push khushbakht91/breast-cancer-triage-api:1.0.0
docker push khushbakht91/breast-cancer-triage-api:latest
```

## 8. Run the published image (clean room)
In an empty directory containing only `docker-compose.hub.yml` and a `.env`
(`DOCKERHUB_USER`, `IMAGE_TAG`, `HOST_PORT`, `DECISION_THRESHOLD`, `LOG_LEVEL`):
```bash
docker compose -f docker-compose.hub.yml up -d
```

## Configuration (environment variables)
| Variable | Default | Meaning |
|---|---|---|
| MODEL_PATH | artifacts/model.joblib | model artifact location |
| DECISION_THRESHOLD | 0.5 | malignant probability cutoff |
| LOG_LEVEL | INFO | logging verbosity (stdout only) |
| REDIS_HOST / REDIS_PORT | unset / 6379 | request counting only when REDIS_HOST is set |
| MODEL_VERSION, GIT_SHA | build args | reported by /model/info |

## API contract
`GET /health`, `POST /predict` (30 non-negative numeric features, 422 otherwise),
`GET /model/info`, `GET /stats`.

## Notes
- Never commit `.env`. `.env.example` documents every variable.
- `pathspec==0.12.1` and `scipy==1.14.1` are pinned for DVC and scikit-learn compatibility.
- scikit-learn 1.5.2 is used for both training and serving.

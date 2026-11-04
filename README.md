# Breast Cancer Triage API

FastAPI service that classifies breast-cancer cell-nucleus measurements (30 features) as likely **malignant** or **benign**. Malignant is the positive class. Educational project, not a medical device.

## Tags
- `1.0.0`: first release (scikit-learn 1.5.2, LogisticRegression pipeline)
- `latest`: same image as `1.0.0`

## Run with Docker
```bash
docker run -d -p 8000:8000 khushbakht91/breast-cancer-triage-api:1.0.0
curl localhost:8000/health
```
Docs: http://localhost:8000/docs

## Run with Docker Compose (with Redis request counter)
Create a `.env` file (`DOCKERHUB_USER=khushbakht91`, `IMAGE_TAG=1.0.0`, `HOST_PORT=8000`, `DECISION_THRESHOLD=0.5`, `LOG_LEVEL=INFO`) and use the `docker-compose.hub.yml` from the GitHub repository, then run `docker compose up -d`.

## Endpoints
`GET /health`, `POST /predict`, `GET /model/info`, `GET /stats`

## Environment variables
| Variable | Default | Meaning |
|---|---|---|
| MODEL_PATH | artifacts/model.joblib | model artifact |
| DECISION_THRESHOLD | 0.5 | malignant probability cutoff |
| LOG_LEVEL | INFO | logging verbosity (stdout) |
| REDIS_HOST / REDIS_PORT | unset / 6379 | counter enabled only when REDIS_HOST is set |
| MODEL_VERSION, GIT_SHA | build args | shown in /model/info |

## Port
Exposes **8000/tcp**. Runs as non-root user `appuser`.

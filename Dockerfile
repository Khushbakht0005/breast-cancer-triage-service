# ---------- Stage 1: train ----------
FROM python:3.12.7-slim AS trainer
WORKDIR /build
COPY requirements-train.txt .
RUN pip install --no-cache-dir -r requirements-train.txt
COPY params.yaml .
COPY training/ training/
COPY data/raw/breast_cancer.csv data/raw/breast_cancer.csv
ARG MODEL_VERSION=1.0.0
RUN python -m training.train \
    --data-path data/raw/breast_cancer.csv \
    --output-dir /build/artifacts \
    --model-version ${MODEL_VERSION}

# ---------- Stage 2: serve ----------
FROM python:3.12.7-slim
ARG MODEL_VERSION=1.0.0
ARG GIT_SHA=unknown
LABEL org.opencontainers.image.version=$MODEL_VERSION \
      org.opencontainers.image.revision=$GIT_SHA
ENV MODEL_VERSION=$MODEL_VERSION \
    GIT_SHA=$GIT_SHA \
    MODEL_PATH=artifacts/model.joblib \
    DECISION_THRESHOLD=0.5 \
    LOG_LEVEL=INFO \
    PYTHONUNBUFFERED=1

RUN useradd --create-home --uid 10001 appuser
WORKDIR /srv

COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY --from=trainer /build/artifacts artifacts/
COPY app/ app/

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

ENTRYPOINT ["uvicorn", "app.main:app"]
CMD ["--host", "0.0.0.0", "--port", "8000"]

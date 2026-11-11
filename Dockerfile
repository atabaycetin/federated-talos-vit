# syntax=docker/dockerfile:1.7
# Default target `onnx`: INT8 ONNX Runtime serving, no torch (~400 MB image).
# Target `taskvector`: shared PyTorch backbone with per-request TaLoS task-vector swapping.
#
#   talos package --config configs/talos_cifar100.yaml --store model_store
#   docker build -t talos-api .
#   docker build --target taskvector -t talos-api:taskvector .
#   docker buildx build --platform linux/amd64 --target lambda --provenance=false ...   # AWS Lambda

# Base image pinned by digest (multi-arch index), dependencies installed from hashed lock files:
# the same commit always builds the same image, and a tampered package fails the hash check.
# Dependabot bumps both; regenerate locks with scripts/lock_requirements.sh.
FROM python:3.15-slim@sha256:65a65fc7b30b06cb6df4930fd4650697cb23f4c2698ab81fc87e4b031bc36928 AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
RUN useradd --uid 10001 --create-home app
COPY requirements/serve.txt requirements/
RUN pip install --require-hashes -r requirements/serve.txt
# Run from source instead of `pip install .`, which would fetch an unpinned build backend.
# COPY keeps host file modes; make the code readable by the non-root user whatever they were.
COPY src ./src
RUN chmod -R a+rX src
ENV PYTHONPATH=/app/src MODEL_STORE=/models PORT=8000 WORKERS=1 THREADS=0
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"
CMD ["sh", "-c", "exec uvicorn talos_mlops.serving.app:app --host 0.0.0.0 --port ${PORT} --workers ${WORKERS} --no-access-log"]

FROM base AS taskvector
COPY requirements/taskvector.txt requirements/
RUN pip install --require-hashes -r requirements/taskvector.txt
ENV BACKEND=taskvector HF_HOME=/opt/hf
COPY model_store /models
# Bake the public DINO backbone weights into the image so startup needs no network.
RUN python - <<'PY'
import json, pathlib, timm
for p in pathlib.Path("/models").glob("*/meta.json"):
    meta = json.loads(p.read_text())
    if meta.get("pretrained", True):
        timm.create_model(meta["backbone"], pretrained=True)
PY
RUN mkdir -p /opt/hf && chmod -R a+rX /opt/hf
USER app

# AWS Lambda: the onnx image plus the Lambda Web Adapter, which turns Lambda invocations into
# HTTP requests to uvicorn. Must stay above `onnx`: the last stage is the default build target.
FROM base AS lambda
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:1.1.0@sha256:17cfd08eff1dfea3f6a9a1e9c65fdac80aa4919b6085e746615530f43f57d2f1 \
  /lambda-adapter /opt/extensions/lambda-adapter
ENV BACKEND=onnx AWS_LWA_PORT=8000 AWS_LWA_READINESS_CHECK_PATH=/health
COPY model_store /models
USER app

FROM base AS onnx
ENV BACKEND=onnx
COPY model_store /models
USER app

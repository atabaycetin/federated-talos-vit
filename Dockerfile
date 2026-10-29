# syntax=docker/dockerfile:1.7
# Default target `onnx`: INT8 ONNX Runtime serving, no torch (~400 MB image).
# Target `taskvector`: shared PyTorch backbone with per-request TaLoS task-vector swapping.
#
#   talos package --config configs/talos_cifar100.yaml --store model_store
#   docker build -t talos-api .
#   docker build --target taskvector -t talos-api:taskvector .

# Base image pinned by digest (multi-arch index), dependencies installed from hashed lock files:
# the same commit always builds the same image, and a tampered package fails the hash check.
# Dependabot bumps both; regenerate locks with scripts/lock_requirements.sh.
FROM python:3.12-slim@sha256:aa943ce411423c7582288f18707ff66eddcbaeb510502d747a72aeb34f30d159 AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
RUN useradd --uid 10001 --create-home app
COPY requirements/serve.txt requirements/
RUN pip install --require-hashes -r requirements/serve.txt
# Run from source instead of `pip install .`, which would fetch an unpinned build backend.
COPY src ./src
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

FROM base AS onnx
ENV BACKEND=onnx
COPY model_store /models
USER app

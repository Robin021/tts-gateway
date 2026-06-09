# tts-gateway: WebSocket + OpenAI-compatible HTTP gateway for vllm-omini.
#
# Build:
#   docker build -t tts-gateway:latest .
#
# Run:
#   docker run -d --name tts-gateway --restart always \
#     --network host \
#     -e VLLM_OMINI_URL=http://localhost:8091 \
#     -e GATEWAY_AUTH_TOKEN=your-secret-key \
#     -e VLLM_OMINI_VOICE=female \
#     tts-gateway:latest
#
# Or with explicit port (without --network host):
#   docker run -d --name tts-gateway --restart always \
#     -p 8000:8000 \
#     -e VLLM_OMINI_URL=http://<vllm-host>:8091 \
#     -e GATEWAY_AUTH_TOKEN=your-secret-key \
#     tts-gateway:latest

# Use a base image that's reachable from China. The Dockerfile defaults
# to the public Python image, which is firewalled in many regions. Pin
# a CN-accessible mirror via ARG:
#   docker build --build-arg PYTHON_IMAGE=python:3.11-slim .     # default (DockerHub)
#   docker build --build-arg PYTHON_IMAGE=docker.m.daocloud.io/library/python:3.11-slim .
#   docker build --build-arg PYTHON_IMAGE=registry.cn-hangzhou.aliyuncs.com/library/python:3.11-slim .
ARG PYTHON_IMAGE=docker.m.daocloud.io/library/python:3.11-slim
FROM ${PYTHON_IMAGE} AS base

# No apt-get install needed — python:3.11-slim already has everything
# this gateway needs (Python + pip + ssl/certs). We do the health
# check with `python -c` instead of curl, which saves ~30s of apt
# downloads and ~30MB of image size.

WORKDIR /app

# Install Python deps first so layer caches well across code changes.
# Use a CN-accessible PyPI mirror if PIP_INDEX_URL is passed in via
# --build-arg.
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn
ENV PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_TRUSTED_HOST=${PIP_TRUSTED_HOST}

COPY pyproject.toml /app/pyproject.toml
RUN pip install --no-cache-dir \
        'fastapi>=0.115' \
        'uvicorn[standard]>=0.30' \
        'httpx>=0.27' \
        'httpx-ws>=0.7' \
        'websockets>=12' \
        'wsproto>=1.2' \
        'starlette>=0.40' \
        'python-multipart>=0.0.9'

# Copy package source.
COPY tts_gateway/ /app/tts_gateway/

# Default config — override at runtime via -e flags.
# We deliberately do NOT bake a default GATEWAY_AUTH_TOKEN into the
# image. The container reads it from the runtime environment
# (`docker run -e GATEWAY_AUTH_TOKEN=...`), which is the correct
# place for secrets. If unset at runtime, the gateway falls back to
# 'test-token' (only useful for local dev).
ENV VLLM_OMINI_URL=http://localhost:8091 \
    VLLM_OMINI_MODEL=cosyvoice3 \
    VLLM_OMINI_VOICE=female \
    PYTHONUNBUFFERED=1

EXPOSE 8000

# Health check via Python (no curl needed). Calls the unauth /docs
# route which always returns 200 if uvicorn is up.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; \
        sys.exit(0 if urllib.request.urlopen('http://localhost:8000/docs', timeout=3).status==200 else 1)" \
        || exit 1

# uvicorn binds 0.0.0.0:8000 by default (we set --host below).
# Single worker is correct: the gateway is async and shares state in
# memory (ACTIVE_SESSIONS, scheduler, etc). For more capacity, scale
# horizontally with multiple containers behind a load balancer that
# supports WebSocket sticky sessions.
CMD ["uvicorn", "tts_gateway.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

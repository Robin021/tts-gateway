#!/usr/bin/env bash
# deploy_h20.sh — one-shot deploy to the H20 box.
#
# Run from your laptop:
#   bash scripts/deploy_h20.sh
#
# What it does:
#   1. rsync the project to /opt/tts-gateway on H20 (excludes test
#      artifacts, venvs, etc.)
#   2. ssh to H20, build the docker image, run/restart the container
#   3. health-check it
#
# Re-run this any time you push code changes — it'll rebuild and
# restart in place.

set -euo pipefail

REMOTE="${REMOTE:-root@221.194.152.20}"
REMOTE_DIR="${REMOTE_DIR:-/opt/tts-gateway}"
GATEWAY_AUTH_TOKEN="${GATEWAY_AUTH_TOKEN:-}"
VLLM_OMINI_URL="${VLLM_OMINI_URL:-http://localhost:8091}"
VLLM_OMINI_VOICE="${VLLM_OMINI_VOICE:-female}"

# Override these if Docker Hub is not reachable from the build host.
# Default uses daocloud (CN-friendly mirror of Docker Hub library).
PYTHON_IMAGE="${PYTHON_IMAGE:-docker.m.daocloud.io/library/python:3.11-slim}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-pypi.tuna.tsinghua.edu.cn}"

if [ -z "$GATEWAY_AUTH_TOKEN" ]; then
    echo "GATEWAY_AUTH_TOKEN is not set."
    echo "Generate one and re-run:"
    echo "    GATEWAY_AUTH_TOKEN=\$(openssl rand -hex 32) bash $0"
    exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
echo "=== syncing $PROJECT_ROOT -> $REMOTE:$REMOTE_DIR ==="
rsync -az --delete \
    --exclude='__pycache__' \
    --exclude='.pytest_cache' \
    --exclude='.venv' \
    --exclude='venv' \
    --exclude='.git' \
    --exclude='install_h20.sh' \
    --exclude='e2e_run_*.log' \
    --exclude='voice_upload_*.log' \
    --exclude='*.pcm' \
    --exclude='female_test.wav' \
    --exclude='male_test.wav' \
    --exclude='linzhiling_test.wav' \
    --exclude='.DS_Store' \
    "$PROJECT_ROOT/" \
    "$REMOTE:$REMOTE_DIR/"

echo
echo "=== building + (re)starting container on H20 ==="
ssh "$REMOTE" \
    GATEWAY_AUTH_TOKEN="$GATEWAY_AUTH_TOKEN" \
    VLLM_OMINI_URL="$VLLM_OMINI_URL" \
    VLLM_OMINI_VOICE="$VLLM_OMINI_VOICE" \
    REMOTE_DIR="$REMOTE_DIR" \
    PYTHON_IMAGE="$PYTHON_IMAGE" \
    PIP_INDEX_URL="$PIP_INDEX_URL" \
    PIP_TRUSTED_HOST="$PIP_TRUSTED_HOST" \
    'bash -s' <<'REMOTE_EOF'
set -euo pipefail
cd "$REMOTE_DIR"

# Build with mirrors that work from China.
docker build \
    --build-arg PYTHON_IMAGE="$PYTHON_IMAGE" \
    --build-arg PIP_INDEX_URL="$PIP_INDEX_URL" \
    --build-arg PIP_TRUSTED_HOST="$PIP_TRUSTED_HOST" \
    -t tts-gateway:latest .

# Stop & remove any existing container (idempotent).
docker rm -f tts-gateway 2>/dev/null || true

# Run with host networking so localhost:8091 reaches the vllm-omini
# container that's already running on this box.
docker run -d \
    --name tts-gateway \
    --restart always \
    --network host \
    -e VLLM_OMINI_URL="$VLLM_OMINI_URL" \
    -e VLLM_OMINI_VOICE="$VLLM_OMINI_VOICE" \
    -e GATEWAY_AUTH_TOKEN="$GATEWAY_AUTH_TOKEN" \
    tts-gateway:latest

echo
echo "container status:"
docker ps --filter name=tts-gateway --format "  {{.Status}}  {{.Image}}  {{.Names}}"
REMOTE_EOF

echo
echo "=== smoke test (waiting up to 30s for service to come up) ==="
for i in $(seq 1 15); do
    if ssh "$REMOTE" \
        "curl -fsS -o /dev/null -w '%{http_code}' \
         -H 'Authorization: Bearer $GATEWAY_AUTH_TOKEN' \
         http://localhost:8000/v1/audio/voices" 2>/dev/null | grep -q 200; then
        echo "  ✓ service healthy"
        break
    fi
    echo "  waiting... ($i/15)"
    sleep 2
done

echo
echo "=== last 20 log lines ==="
ssh "$REMOTE" 'docker logs --tail=20 tts-gateway'

echo
echo "=== DONE ==="
echo "WebSocket:  ws://221.194.152.20:8000/ws/tts"
echo "HTTP TTS:   POST http://221.194.152.20:8000/v1/audio/speech"
echo "Auth:       Authorization: Bearer <GATEWAY_AUTH_TOKEN>"

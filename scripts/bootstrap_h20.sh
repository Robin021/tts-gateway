#!/usr/bin/env bash
# bootstrap_h20.sh — run after the project files are extracted on h20.

set -euo pipefail

TARGET="${TARGET:-$(pwd)}"
ENGINE_URL="${ENGINE_URL:-http://localhost:8091}"
VOICE="${VOICE:-test_voice_v3}"
SAMPLE_RATE="${SAMPLE_RATE:-24000}"

cd "$TARGET"

echo "=== install deps ==="
python3 -m pip install -q --no-cache-dir \
    'fastapi>=0.115' \
    'uvicorn[standard]>=0.30' \
    'httpx>=0.27' \
    'httpx-ws>=0.7' \
    'websockets>=12' \
    'wsproto>=1.2' \
    'starlette>=0.40' \
    'pytest>=7' || true

echo
echo "=== verify voice exists ==="
python3 - <<PY
import asyncio, sys
from tts_gateway.engine_vllm_omini import VllmOminiEngine
async def main():
    e = VllmOminiEngine(base_url="$ENGINE_URL")
    await e.load()
    voices = await e.list_voices()
    print("voices:", voices)
    if "$VOICE" not in voices.get("voices", []):
        print("FATAL: voice '$VOICE' not found, re-run upload"); sys.exit(2)
    await e.shutdown()
    print("OK voice present")
asyncio.run(main())
PY

echo
echo "=== unit tests on h20 ==="
python3 -m pytest tests/test_sentence_buffer.py tests/test_scheduler.py \
    tests/test_session.py tests/test_engine.py -q 2>&1 | tail -10

echo
echo "=== END-TO-END against real vllm-omini ==="
ENGINE_URL="$ENGINE_URL" VOICE="$VOICE" SAMPLE_RATE="$SAMPLE_RATE" \
    python3 -m tests.run_e2e_real --max-concurrent 32

echo
echo "=== DONE — paste this entire output back ==="

#!/usr/bin/env bash
# probe_vllm_omini.sh
#
# Run on the h20 box that hosts vllm-omini at localhost:8091.
# Produces a single block of output that's enough for me to write the
# VllmOminiEngine adapter.
#
# Steps:
#   1. Detect tooling.
#   2. Generate a 4s synthetic-vowel ref.wav using only Python stdlib
#      (no ffmpeg/sox/espeak required). Audio quality of the cloned
#      voice will be terrible — we only need it to make the link work
#      so we can measure TTFB and abort_latency.
#   3. Delete any leftover test voices, upload a fresh one with ref_text.
#   4. Confirm it's listed.
#   5. Non-stream synthesis -> /tmp/full.pcm. Verify byte count.
#   6. Stream synthesis -> /tmp/stream.pcm with curl, record TTFB.
#   7. Python: 20 runs of stream-then-aclose at cancel_after_ms=500.
#      Reports TTFB / aclose / tail_audio_ms percentiles.
#
# Bail on first failure so we don't waste time after a fundamental break.

set -uo pipefail

BASE_URL="${BASE_URL:-http://localhost:8091}"
MODEL="${MODEL:-cosyvoice3}"
VOICE_NAME="${VOICE_NAME:-test_voice_v3}"
REF_TEXT="${REF_TEXT:-啊啊啊啊。}"

echo "============================================================"
echo "STEP 1: tooling"
echo "============================================================"
for cmd in python3 curl jq ffmpeg sox espeak-ng espeak; do
    if path=$(command -v "$cmd" 2>/dev/null); then
        echo "  $cmd -> $path"
    else
        echo "  $cmd -> (not found)"
    fi
done

echo
echo "============================================================"
echo "STEP 2: generate /tmp/ref.wav (synthetic-vowel, 4s @ 16kHz)"
echo "============================================================"
python3 - <<'PY'
import wave, math, struct, os
sr, dur = 16000, 4
n = sr * dur
samples = []
for i in range(n):
    t = i / sr
    env = 0.5 + 0.5 * math.sin(2 * math.pi * 2 * t)
    s = 0.3 * env * (
        math.sin(2*math.pi*220*t)
        + 0.5*math.sin(2*math.pi*400*t)
        + 0.3*math.sin(2*math.pi*1700*t)
        + 0.2*math.sin(2*math.pi*2400*t)
    )
    samples.append(int(max(-1, min(1, s)) * 30000))
with wave.open('/tmp/ref.wav', 'wb') as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
    w.writeframes(struct.pack(f'<{n}h', *samples))
print(f"  ref.wav size = {os.path.getsize('/tmp/ref.wav')}")
PY
[ -s /tmp/ref.wav ] || { echo "FAIL: /tmp/ref.wav not created"; exit 1; }

echo
echo "============================================================"
echo "STEP 3: delete leftover voices + upload fresh one"
echo "============================================================"
for v in test_voice_v1 test_voice_v2 "$VOICE_NAME"; do
    code=$(curl -s -o /tmp/del.json -w "%{http_code}" -X DELETE "$BASE_URL/v1/audio/voices/$v")
    echo "  DELETE $v -> $code"
done

echo "  uploading $VOICE_NAME ..."
curl -s -o /tmp/upload.json -w "  upload http=%{http_code}\n" \
    -X POST "$BASE_URL/v1/audio/voices" \
    -F "audio_sample=@/tmp/ref.wav" \
    -F "name=$VOICE_NAME" \
    -F "consent=acknowledged" \
    -F "ref_text=$REF_TEXT"
cat /tmp/upload.json; echo

echo
echo "============================================================"
echo "STEP 4: list voices"
echo "============================================================"
curl -s "$BASE_URL/v1/audio/voices" | (jq . 2>/dev/null || cat); echo

echo
echo "============================================================"
echo "STEP 5: NON-STREAM synthesis"
echo "============================================================"
curl -s -o /tmp/full.pcm -w "  http=%{http_code} bytes=%{size_download} ct=%{content_type} time_total=%{time_total}\n" \
    -X POST "$BASE_URL/v1/audio/speech" \
    -H 'Content-Type: application/json' \
    -d "{
        \"model\":\"$MODEL\",
        \"input\":\"你好,这是一个测试,我们正在验证流式合成。\",
        \"voice\":\"$VOICE_NAME\",
        \"task_type\":\"CustomVoice\",
        \"response_format\":\"pcm\"
    }"
ls -l /tmp/full.pcm
echo "  first 32 bytes of /tmp/full.pcm:"
xxd /tmp/full.pcm 2>/dev/null | head -2 || head -c 64 /tmp/full.pcm | od -An -tx1 | head -2

# Try to derive sample rate. We compute audio duration assuming 16k/22.05k/24k/32k/44.1k
# pcm16 mono and let user eyeball which is plausible if they listen.
python3 - <<PY
import os
b = os.path.getsize("/tmp/full.pcm")
print(f"  bytes={b}")
for sr in (16000, 22050, 24000, 32000, 44100, 48000):
    secs = b / 2 / sr
    print(f"    if sr={sr}: duration ~ {secs:.2f}s")
PY

echo
echo "============================================================"
echo "STEP 6: STREAM synthesis (curl, record TTFB)"
echo "============================================================"
curl -N -s --no-buffer -o /tmp/stream.pcm \
    -w "  http=%{http_code} ttfb=%{time_starttransfer} total=%{time_total} bytes=%{size_download}\n" \
    -X POST "$BASE_URL/v1/audio/speech" \
    -H 'Content-Type: application/json' \
    -d "{
        \"model\":\"$MODEL\",
        \"input\":\"你好,这是一个测试,我们正在验证流式合成。\",
        \"voice\":\"$VOICE_NAME\",
        \"task_type\":\"CustomVoice\",
        \"response_format\":\"pcm\",
        \"stream\":true,
        \"stream_format\":\"audio\"
    }"
ls -l /tmp/stream.pcm

echo
echo "============================================================"
echo "STEP 7: TTFB + abort_latency (Python, 20 runs)"
echo "============================================================"
BASE_URL="$BASE_URL" MODEL="$MODEL" VOICE_NAME="$VOICE_NAME" python3 - <<'PY'
import asyncio, os, time, sys
import httpx

BASE_URL = os.environ["BASE_URL"]
MODEL = os.environ["MODEL"]
VOICE = os.environ["VOICE_NAME"]
LONG_TEXT = "我们来测一段长文本," + "测试服务端的abort行为," * 30 + "看尾音多长。"

async def one_run(cancel_after_ms: int) -> dict:
    payload = {
        "model": MODEL,
        "input": LONG_TEXT,
        "voice": VOICE,
        "task_type": "CustomVoice",
        "response_format": "pcm",
        "stream": True,
        "stream_format": "audio",
    }
    bytes_received = 0
    bytes_after_cancel = 0
    cancelled_at = None
    aclose_returned_at = None
    first_chunk_at = None
    t0 = time.monotonic()
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", f"{BASE_URL}/v1/audio/speech", json=payload) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                return {"error": f"HTTP {resp.status_code}: {body[:300]!r}"}
            async for chunk in resp.aiter_bytes():
                now = time.monotonic()
                if first_chunk_at is None:
                    first_chunk_at = now
                bytes_received += len(chunk)
                if cancelled_at is not None:
                    bytes_after_cancel += len(chunk)
                if cancelled_at is None and (now - t0) * 1000 >= cancel_after_ms:
                    cancelled_at = now
                    await resp.aclose()
                    aclose_returned_at = time.monotonic()
                    break
    return {
        "ttfb_ms": (first_chunk_at - t0) * 1000 if first_chunk_at else None,
        "bytes_received": bytes_received,
        "bytes_after_cancel": bytes_after_cancel,
        "aclose_dur_ms": (aclose_returned_at - cancelled_at) * 1000
                          if (cancelled_at and aclose_returned_at) else None,
    }

def pct(values, p):
    if not values: return None
    s = sorted(values); k = (len(s)-1)*p; f = int(k); c = min(f+1, len(s)-1)
    return s[f] if f == c else s[f] + (s[c]-s[f])*(k-f)

async def main():
    runs = 20
    cancel_after_ms = 500
    print(f"  runs={runs}, cancel_after_ms={cancel_after_ms}")
    results = []
    for i in range(runs):
        try:
            r = await one_run(cancel_after_ms)
            print(f"  [{i:2d}] {r}")
            if "error" not in r:
                results.append(r)
        except Exception as e:
            print(f"  [{i:2d}] EXC {type(e).__name__}: {e}")
    if not results:
        print("  no successful runs")
        sys.exit(1)
    ttfb = [r["ttfb_ms"] for r in results if r.get("ttfb_ms") is not None]
    bytes_after = [r["bytes_after_cancel"] for r in results]
    aclose = [r["aclose_dur_ms"] for r in results if r.get("aclose_dur_ms") is not None]
    print()
    print("  === SUMMARY ===")
    if ttfb:
        print(f"  TTFB ms        P50={pct(ttfb,.5):8.1f}  P95={pct(ttfb,.95):8.1f}  P99={pct(ttfb,.99):8.1f}  max={max(ttfb):8.1f}")
    if aclose:
        print(f"  aclose() ms    P50={pct(aclose,.5):8.1f}  P95={pct(aclose,.95):8.1f}  P99={pct(aclose,.99):8.1f}  max={max(aclose):8.1f}")
    print(f"  bytes_after    P50={pct(bytes_after,.5):8.0f}  P95={pct(bytes_after,.95):8.0f}  P99={pct(bytes_after,.99):8.0f}  max={max(bytes_after):8.0f}")
    print()
    print("  tail audio ms (= bytes_after / bytes_per_second):")
    for sr in (16000, 22050, 24000):
        bps = sr * 2  # pcm16 mono
        tail = [b * 1000 / bps for b in bytes_after]
        print(f"    if sr={sr}: P50={pct(tail,.5):8.1f}  P95={pct(tail,.95):8.1f}  P99={pct(tail,.99):8.1f}")

asyncio.run(main())
PY

echo
echo "============================================================"
echo "DONE — copy this entire output and paste it back to me."
echo "============================================================"

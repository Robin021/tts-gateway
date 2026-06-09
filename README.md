# tts-gateway

WebSocket TTS gateway that wraps vllm-omini / Fun-CosyVoice3 behind a
low-latency, cancellable, multi-client streaming protocol.

> **Looking to call this service from your app?**
> See [`CLIENT_GUIDE.md`](./CLIENT_GUIDE.md) — that's the document to
> hand to client-side developers. They don't need to read this README.

The gateway sits between your clients and vllm-omini:

```
[Client (browser/app)]
        ↕ WebSocket  (protocol described in PROTOCOL.md)
[tts-gateway]   ← this project
        ↕ HTTP  /v1/audio/speech
[vllm-omini + cosyvoice3]
```

**Why a gateway?** vllm-omini's `/v1/audio/speech` is a one-shot HTTP
request — no streaming text input, no mid-flight cancel from the
client's perspective, no auth, no queueing telemetry. The gateway adds
all of that so a real client can use it for an interactive
conversational UX.

## Quickstart

### 1. Upload voices to vllm-omini

CosyVoice3 has **no built-in voices** — it's a zero-shot voice-cloning
model. You upload a reference audio + transcript ("ref_text"), and the
model clones that speaker. The gateway forwards the client's
`prompt_audio_id` to vllm-omini's `voice` parameter.

**Critical**: CosyVoice3 requires the ref_text to start with the
prefix `You are a helpful assistant.<|endofprompt|>` (this comes
straight from the official README). Without the prefix, voice cloning
quality is significantly worse.

For convenience there's a one-shot upload script in `scripts/`:

```bash
# scp the script + voice samples to the vllm-omini host, then:
bash scripts/upload_voices.sh
```

This uploads three named voices (`female`, `male`, `linzhiling`) from
the bundled `scripts/voices.tar.gz`. Replace these with your own
recordings before going to production. Each voice needs:

- 3-15 second clean speech sample (mono wav)
- Exact transcript of what's said
- The `You are a helpful assistant.<|endofprompt|>` prefix on the
  transcript

To upload manually:

```bash
curl -X POST http://<vllm-host>:8091/v1/audio/voices \
  -F "audio_sample=@reference.wav" \
  -F "name=my_voice" \
  -F "consent=acknowledged" \
  -F 'ref_text=You are a helpful assistant.<|endofprompt|>录音里念的话'
```

**Voice persistence**: vllm-omini stores voices at
`/root/.cache/vllm-omni/speakers/`. By default this is inside the
container and lost on restart. Mount it out with `docker run -v
/data/vllm-omni-speakers:/root/.cache/vllm-omni/speakers` to keep
voices across restarts.

Verify uploaded voices:

```bash
curl http://<vllm-host>:8091/v1/audio/voices
```

### 2. Run the gateway

The gateway is a normal Python service. Put it on the same machine as
vllm-omini, or on a separate machine that can reach vllm-omini over
HTTP — it doesn't need a GPU.

```bash
pip install fastapi 'uvicorn[standard]' httpx httpx-ws websockets wsproto starlette

# Same machine as vllm-omini:
VLLM_OMINI_URL=http://localhost:8091 \
GATEWAY_AUTH_TOKEN="prod-secret-1,prod-secret-2" \
    uvicorn tts_gateway.app:app --host 0.0.0.0 --port 8000

# Separate machine (vllm-omini at 10.0.0.5):
VLLM_OMINI_URL=http://10.0.0.5:8091 \
VLLM_OMINI_VOICE=my_voice \
GATEWAY_AUTH_TOKEN="prod-secret" \
    uvicorn tts_gateway.app:app --host 0.0.0.0 --port 8000
```

### 3. Connect from a client

The gateway exposes **two interfaces**. Pick the one that fits your client:

#### A) WebSocket (recommended for interactive / real-time use)

Full streaming, mid-flight cancel, low latency. See `PROTOCOL.md` for
the full wire protocol. Minimal JS example:

```js
const ws = new WebSocket("ws://gateway-host:8000/ws/tts", "tts.v1");
ws.binaryType = "arraybuffer";
ws.onopen = () => {
  ws.send(JSON.stringify({type:"auth", token:"prod-secret", client_id:"u1"}));
  ws.send(JSON.stringify({type:"tts.start", request_id:"r1", voice:"female"}));
  ws.send(JSON.stringify({type:"tts.text", text:"你好,这是一个测试。"}));
  ws.send(JSON.stringify({type:"tts.end"}));
};
ws.onmessage = (e) => {
  if (typeof e.data === "string") console.log(JSON.parse(e.data));
  else /* play e.data as pcm16 mono @ 24kHz */;
};
```

To interrupt mid-synthesis:

```js
ws.send(JSON.stringify({type:"tts.cancel"}));
```

#### B) OpenAI-compatible HTTP (drop-in for OpenAI SDK clients)

`POST /v1/audio/speech` mirrors OpenAI's TTS API. Clients written
against the OpenAI SDK can point `base_url` at this gateway and Just
Work. The `voice` field is the vllm-omini speaker file name (`female`,
`male`, `linzhiling`, or any other voice you've uploaded).

```bash
curl -X POST http://gateway-host:8000/v1/audio/speech \
  -H "Authorization: Bearer prod-secret" \
  -H "Content-Type: application/json" \
  -o output.wav \
  -d '{
    "model": "cosyvoice3",
    "input": "你好,这是一个测试。",
    "voice": "female",
    "response_format": "wav"
  }'
```

Or in Python via the OpenAI SDK:

```python
from openai import OpenAI
client = OpenAI(api_key="prod-secret", base_url="http://gateway-host:8000/v1")
resp = client.audio.speech.create(
    model="cosyvoice3",
    voice="female",         # vllm-omini speaker file name
    input="你好,这是一个测试。",
    response_format="wav",  # or "pcm"
)
resp.stream_to_file("output.wav")
```

For low-latency streaming (chunked audio bytes back as they're
synthesized), pass `"stream": true` in the request body. Response is
chunked HTTP with `Content-Type: audio/pcm` (or `audio/wav` with a
streaming-friendly header).

**Note**: The HTTP endpoint does not support mid-flight cancel — if
you need interruption, use the WebSocket interface. To list available
voices: `GET /v1/audio/voices` (also Bearer-auth'd).

## Configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `VLLM_OMINI_URL` | `http://localhost:8091` | Where vllm-omini is. If unset → MockEngine (silence) for dev/demo only |
| `VLLM_OMINI_MODEL` | `cosyvoice3` | Model id, must match what vllm-omini serves |
| `VLLM_OMINI_VOICE` | `female` | Default voice when client doesn't specify `prompt_audio_id`. Must already be uploaded to vllm-omini. |
| `GATEWAY_AUTH_TOKEN` | `test-token` | Comma-separated list of accepted auth tokens. **Replace before exposing publicly.** |

For finer control (timeouts, queue sizes, concurrency limits), construct
the app programmatically:

```python
from tts_gateway.app import create_app, GatewaySettings
from tts_gateway.engine_vllm_omini import VllmOminiEngine

engine = VllmOminiEngine(base_url="http://10.0.0.5:8091", default_voice="my_voice")
settings = GatewaySettings(
    max_concurrent_synthesis=32,
    auth_timeout_s=5.0,
    idle_timeout_s=30.0,
    max_session_s=300.0,
    valid_tokens=("k1","k2"),
)
app = create_app(engine=engine, settings=settings)
# uvicorn this app
```

## Measured behavior (real vllm-omini + cosyvoice3-0.5B)

End-to-end through the gateway, with `cosyvoice3` running on H20:

| Metric | Value | Notes |
|---|---|---|
| TTFB (`tts.start`→first audio byte) | ~600ms | This is vllm-omini's prefill latency. The gateway adds **<5ms** on top. |
| `tts.cancelled` ack latency | ~1ms | Gateway protocol layer is fast. |
| Tail audio after `tts.cancel` | 600–1400ms (one chunk) | **vllm-omini limitation** — it emits in sentence-level segments. When the client cancels, the segment already in flight (in TCP/HTTP buffers) still arrives. The gateway cannot truncate this without changes inside vllm-omini. |
| Concurrent sessions | Limited by vllm-omini's `--max-num-seqs` | Gateway defaults `max_concurrent_synthesis=32`. If vllm itself can only batch 1 request at a time, the second client waits. |
| RTF (per session) | ~0.5x realtime | Once streaming starts. |

## Deployment

Three deployment shapes from simplest to most production-y. **Start
with shape A**; only graduate to B/C when you actually have the
problems they solve.

### A) Docker on the same host as vllm-omini (recommended for now)

This is the simplest path. The gateway runs as a sidecar Docker
container on the H20 box that already runs `cosyvoice3-tts`. Uses
host networking so localhost:8091 reaches vllm-omini directly.

**One-shot deploy from your laptop:**

```bash
# Generate a real auth token first
export GATEWAY_AUTH_TOKEN=$(openssl rand -hex 32)
echo "save this token, give it to clients:"
echo "  $GATEWAY_AUTH_TOKEN"

# Deploy (rsync code → build → run on H20)
GATEWAY_AUTH_TOKEN="$GATEWAY_AUTH_TOKEN" bash scripts/deploy_h20.sh
```

`deploy_h20.sh` rsyncs the project to `/opt/tts-gateway` on the
remote, builds the Docker image, and (re)starts the container with
`--restart always`. Re-run any time you push code; it's idempotent.

**On H20 directly** (if you don't have the laptop-side toolchain):

```bash
cd /opt/tts-gateway   # whatever the rsynced or git-cloned dir is
docker build -t tts-gateway:latest .
docker rm -f tts-gateway 2>/dev/null
docker run -d \
    --name tts-gateway --restart always \
    --network host \
    -e GATEWAY_AUTH_TOKEN=<your-token> \
    -e VLLM_OMINI_URL=http://localhost:8091 \
    -e VLLM_OMINI_VOICE=female \
    tts-gateway:latest
```

**Or via docker compose:**

```bash
cp .env.example .env
# edit .env to set GATEWAY_AUTH_TOKEN
docker compose up -d
docker compose logs -f tts-gateway
```

After deployment, clients connect to `ws://h20-ip:8000/ws/tts` (or
`POST http://h20-ip:8000/v1/audio/speech` for the OpenAI-compat path).

### B) Docker on a separate gateway machine

Same image, just runs on a different machine that can HTTP-reach H20:

```bash
docker run -d \
    --name tts-gateway --restart always \
    -p 8000:8000 \
    -e GATEWAY_AUTH_TOKEN=<your-token> \
    -e VLLM_OMINI_URL=http://221.194.152.20:8091 \
    -e VLLM_OMINI_VOICE=female \
    tts-gateway:latest
```

Useful if H20's CPU is hot, or if you want to put the gateway in a
different network zone (e.g. public-facing, while H20 stays internal).

### C) Kubernetes / EKS

When real load shows up, move to k8s. Outline (no manifests in this
repo yet — add when you actually deploy):

- **Deployment** (replicas: 2-4 to start, behind HPA on connection
  count or CPU). Single uvicorn worker per pod (the gateway shares
  in-memory state per process; horizontal scaling is via more pods).
- **Service** type `ClusterIP`, port 8000.
- **Ingress** with WebSocket-aware load balancer (ALB with
  `alb.ingress.kubernetes.io/target-type: ip` and connection draining;
  or an NLB for L4). Sticky sessions are NOT required — every WS
  connection is fully self-contained.
- **TLS**: terminate at ALB (ACM cert) or cert-manager + ingress-nginx.
- **Config**: `GATEWAY_AUTH_TOKEN` in Secret, other env in ConfigMap.
- **Voices**: vllm-omini stays where it is. Or, if you also want to
  containerize vllm-omini in EKS, mount voices via PVC.
- **Observability**: scrape `/metrics` (TODO: not implemented yet —
  Prometheus client is a future addition). Or front with Sentry's
  WebSocket tracing.

Performance ceiling is set by vllm-omini's `--max-num-seqs`, not the
gateway. Scale gateway pods only to match incoming connection counts;
synthesis throughput is bound by GPU.

## Production caveats

These are **not** in the gateway, they're in the layers around it:

1. **Tail audio on cancel**: A chunk already in flight (typically
   600–1400ms of audio) will reach the client even after `tts.cancel`.
   The client should be prepared to discard audio after it has shown
   the "stopped" UI state. To reduce the tail at the engine level,
   you'd need vllm-omini to emit smaller chunks or expose a
   step-level abort API — neither exists today.

2. **vllm-omini concurrency**: Set `--max-num-seqs` on vllm-omini to
   the number of concurrent synthesis sessions you want. Gateway's
   `max_concurrent_synthesis` should be ≥ that number; gateway adds no
   extra serialization.

3. **Auth tokens**: `GATEWAY_AUTH_TOKEN` is a static-list demo. Replace
   `_do_first_frame_auth` with JWT/OIDC validation before exposing
   publicly. The first-frame auth pattern itself is fine — you just
   need a real verifier.

4. **WebSocket through nginx / LB**: Set `proxy_read_timeout 3600s`
   and `proxy_send_timeout 3600s`, otherwise idle synthesis sessions
   get killed mid-stream. The gateway sends a `tts.keepalive` every
   20s to help, but LB timeouts override that.

5. **Voices**: Voice management (upload / list / delete) goes directly
   to vllm-omini's `/v1/audio/voices` HTTP endpoints. The gateway does
   not proxy these. If you need a unified API surface, add an HTTP
   endpoint to the gateway that forwards to
   `VllmOminiEngine.upload_voice / list_voices / delete_voice`.

6. **TLS**: Run uvicorn behind nginx/Caddy/ALB with TLS termination.
   The gateway speaks plain `ws://` — clients should use `wss://` via
   the reverse proxy.

## Repo layout

| Path | Purpose |
|---|---|
| `tts_gateway/protocol.py` | Wire types, `SessionState`, `ErrorCode`, `AudioFormat` |
| `tts_gateway/sentence_buffer.py` | Streamed-text → sentence batching |
| `tts_gateway/scheduler.py` | Two-stage scheduler (`reserve()` / `wait()`) |
| `tts_gateway/engine.py` | `TtsBackend` ABC + `MockEngine` |
| `tts_gateway/engine_vllm_omini.py` | Real backend → vllm-omini HTTP |
| `tts_gateway/session.py` | Per-WS state + the single `close()` path |
| `tts_gateway/loops.py` | sender / heartbeat / watchdog / run_tts / supervisor |
| `tts_gateway/app.py` | FastAPI WS endpoint + lifespan + auth |
| `tests/` | 40 unit + integration tests, mock-based |
| `tests/run_e2e_real.py` | End-to-end against a real vllm-omini |
| `tests/stress_run.py` | Concurrent-load harness |
| `PROTOCOL.md` | The full wire-protocol contract |

## Tests

```bash
# Unit + integration (no engine needed):
pytest tests/

# End-to-end against real vllm-omini:
ENGINE_URL=http://your-vllm:8091 VOICE=my_voice \
    python -m tests.run_e2e_real

# Concurrent stress (in-process ASGI, no real socket bind):
python -m tests.stress_run --connections 50 --duration 30 --cancel-rate 0.05
```

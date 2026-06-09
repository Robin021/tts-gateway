"""End-to-end test against the real vllm-omini at localhost:8091.

Run on the h20 box:

    cd /tmp/tts_gateway
    pip install -q fastapi uvicorn httpx httpx_ws websockets wsproto
    python -m tests.run_e2e_real

What it does:

  1. Builds a VllmOminiEngine pointed at http://localhost:8091
  2. Wraps it in our gateway app
  3. Boots the gateway in-process (no extra port)
  4. Drives end-to-end scenarios over the gateway WebSocket:
       a) happy path: auth -> start -> text -> end -> audio -> done
       b) mid-flight cancel: start -> text -> end -> audio... -> cancel
       c) two concurrent sessions to verify isolation
  5. Reports TTFB and abort_latency from the CLIENT'S point of view
     (i.e. through the gateway, which is what production sees)

This is the moment of truth: numbers from MockEngine were ~5ms abort
latency. Now we see what the real cosyvoice3 engine + gateway combo
delivers.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

# Make the package importable when run as a module.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx
import wsproto
from httpx_ws import WebSocketDisconnect, aconnect_ws
from httpx_ws.transport import ASGIWebSocketTransport

from tts_gateway.app import GatewaySettings, create_app
from tts_gateway.engine_vllm_omini import VllmOminiEngine


END_OF_STREAM = (asyncio.TimeoutError, TimeoutError, WebSocketDisconnect)


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------


@asynccontextmanager
async def make_gateway(
    engine: VllmOminiEngine,
    *,
    max_concurrent: int = 2,
) -> AsyncIterator[tuple[httpx.AsyncClient, "AsyncSession"]]:
    settings = GatewaySettings(
        max_concurrent_synthesis=max_concurrent,
        auth_timeout_s=10.0,
        start_timeout_s=10.0,
        idle_timeout_s=120.0,
        max_session_s=300.0,
    )
    app = create_app(engine=engine, settings=settings)

    # ASGIWebSocketTransport doesn't reliably drive ASGI lifespan, so we
    # manually invoke startup here and shutdown on exit. Without this,
    # engine.load() never runs and stream_tts() raises "load() was not called".
    lifespan_recv: asyncio.Queue = asyncio.Queue()
    lifespan_sent: list = []

    async def lifespan_send(msg):
        lifespan_sent.append(msg)

    async def lifespan_receive():
        return await lifespan_recv.get()

    lifespan_task = asyncio.create_task(
        app({"type": "lifespan", "asgi": {"version": "3.0"}},
            lifespan_receive, lifespan_send)
    )
    await lifespan_recv.put({"type": "lifespan.startup"})
    # Wait until startup.complete arrives.
    for _ in range(500):
        if any(m.get("type") == "lifespan.startup.complete" for m in lifespan_sent):
            break
        await asyncio.sleep(0.02)
    else:
        raise RuntimeError("lifespan startup did not complete in 10s")

    transport = ASGIWebSocketTransport(app=app)
    try:
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://e2e",
            timeout=httpx.Timeout(None),
        ) as client:
            yield client, app
    finally:
        await lifespan_recv.put({"type": "lifespan.shutdown"})
        try:
            await asyncio.wait_for(lifespan_task, timeout=10.0)
        except asyncio.TimeoutError:
            lifespan_task.cancel()


async def auth(ws, client_id: str = "e2e-client") -> dict:
    await ws.send_text(json.dumps({
        "type": "auth", "token": "test-token", "client_id": client_id,
    }))
    msg = await asyncio.wait_for(ws.receive_text(), timeout=5.0)
    return json.loads(msg)


async def recv(ws, timeout=10.0):
    evt = await ws.receive(timeout=timeout)
    if isinstance(evt, wsproto.events.TextMessage):
        return ("text", evt.data)
    if isinstance(evt, wsproto.events.BytesMessage):
        return ("bytes", evt.data)
    if isinstance(evt, wsproto.events.CloseConnection):
        return ("close", evt)
    return ("other", evt)


# ---------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------


async def scenario_happy_path(client: httpx.AsyncClient, voice: str, text: str) -> dict:
    """Auth → start → push text → end → collect audio → done.

    Returns timing metrics.
    """
    print(f"\n=== scenario: HAPPY PATH (voice={voice}) ===")
    metrics = {
        "ttfb_ms": None,
        "first_audio_at": None,
        "audio_chunks": 0,
        "audio_bytes": 0,
        "duration_ms": None,
        "wall_ms": None,
        "queued_position": None,
    }
    t0 = time.monotonic()
    async with aconnect_ws(
        "http://e2e/ws/tts",
        client=client,
        subprotocols=["tts.v1"],
        keepalive_ping_interval_seconds=None,
    ) as ws:
        ack = await auth(ws)
        assert ack["type"] == "auth.ok", f"bad auth: {ack}"

        await ws.send_text(json.dumps({
            "type": "tts.start",
            "request_id": "req-happy",
            "prompt_audio_id": voice,
        }))
        start_at = time.monotonic()

        # Wait for tts.processing before sending text (be a polite client).
        while True:
            kind, payload = await recv(ws)
            if kind == "text":
                d = json.loads(payload)
                if d["type"] == "tts.queued":
                    metrics["queued_position"] = d.get("position")
                elif d["type"] == "tts.processing":
                    break
                elif d["type"] == "tts.error":
                    raise RuntimeError(f"error before processing: {d}")

        # Push text + end.
        await ws.send_text(json.dumps({"type": "tts.text", "text": text}))
        await ws.send_text(json.dumps({"type": "tts.end"}))
        text_sent_at = time.monotonic()

        while True:
            kind, payload = await recv(ws, timeout=30.0)
            if kind == "bytes":
                if metrics["first_audio_at"] is None:
                    metrics["first_audio_at"] = time.monotonic()
                    metrics["ttfb_ms"] = (metrics["first_audio_at"] - text_sent_at) * 1000
                metrics["audio_chunks"] += 1
                metrics["audio_bytes"] += len(payload)
                continue
            if kind == "close":
                break
            d = json.loads(payload)
            if d["type"] == "tts.done":
                metrics["duration_ms"] = d["generated_duration_ms"]
                break
            if d["type"] == "tts.error":
                raise RuntimeError(f"error during synth: {d}")

    metrics["wall_ms"] = (time.monotonic() - t0) * 1000
    return metrics


async def scenario_cancel_midstream(
    client: httpx.AsyncClient,
    voice: str,
    text: str,
    cancel_after_first_audio_ms: float,
) -> dict:
    """Auth → start → push text → end → wait for audio → cancel."""
    print(f"\n=== scenario: CANCEL @ {cancel_after_first_audio_ms:.0f}ms after first audio ===")
    metrics = {
        "ttfb_ms": None,
        "audio_chunks": 0,
        "audio_bytes": 0,
        "audio_after_cancel_bytes": 0,
        "audio_after_cancel_chunks": 0,
        "abort_latency_ms": None,
        "got_cancelled": False,
    }
    async with aconnect_ws(
        "http://e2e/ws/tts",
        client=client,
        subprotocols=["tts.v1"],
        keepalive_ping_interval_seconds=None,
    ) as ws:
        await auth(ws)
        await ws.send_text(json.dumps({
            "type": "tts.start",
            "request_id": "req-cancel",
            "prompt_audio_id": voice,
        }))

        while True:
            kind, payload = await recv(ws)
            if kind == "text":
                d = json.loads(payload)
                if d["type"] == "tts.processing":
                    break
                if d["type"] == "tts.error":
                    raise RuntimeError(d)

        await ws.send_text(json.dumps({"type": "tts.text", "text": text}))
        await ws.send_text(json.dumps({"type": "tts.end"}))
        text_sent_at = time.monotonic()

        cancel_sent_at = None
        cancelled_received_at = None
        first_audio_at = None

        while True:
            try:
                kind, payload = await recv(ws, timeout=10.0)
            except END_OF_STREAM:
                break
            if kind == "bytes":
                if first_audio_at is None:
                    first_audio_at = time.monotonic()
                    metrics["ttfb_ms"] = (first_audio_at - text_sent_at) * 1000
                if cancel_sent_at:
                    metrics["audio_after_cancel_chunks"] += 1
                    metrics["audio_after_cancel_bytes"] += len(payload)
                else:
                    metrics["audio_chunks"] += 1
                    metrics["audio_bytes"] += len(payload)

                # Trigger cancel after configured time post-first-audio.
                if (
                    cancel_sent_at is None
                    and (time.monotonic() - first_audio_at) * 1000 >= cancel_after_first_audio_ms
                ):
                    cancel_sent_at = time.monotonic()
                    await ws.send_text(json.dumps({"type": "tts.cancel"}))
                continue
            if kind == "close":
                break
            d = json.loads(payload)
            if d["type"] == "tts.cancelled":
                cancelled_received_at = time.monotonic()
                metrics["got_cancelled"] = True
                break
            if d["type"] == "tts.done":
                # Engine finished before we could cancel.
                break
            if d["type"] == "tts.error":
                metrics["error"] = d
                break

        if cancel_sent_at and cancelled_received_at:
            metrics["abort_latency_ms"] = (cancelled_received_at - cancel_sent_at) * 1000
    return metrics


async def scenario_two_concurrent(
    client: httpx.AsyncClient, voice: str, text: str,
) -> dict:
    """Two sessions in parallel through the same gateway / engine."""
    print(f"\n=== scenario: 2 CONCURRENT ===")

    async def one(rid: str) -> dict:
        return await scenario_happy_path(
            client=client, voice=voice, text=text + f" [{rid}]",
        )

    a, b = await asyncio.gather(one("A"), one("B"))
    return {"A": a, "B": b}


# ---------------------------------------------------------------
# Driver
# ---------------------------------------------------------------


async def run_all(args: argparse.Namespace) -> None:
    engine = VllmOminiEngine(
        base_url=args.engine_url,
        model=args.model,
        default_voice=args.voice,
        sample_rate=args.sample_rate,
    )

    short_text = "你好,这是一个端到端测试。"
    long_text = (
        "你好,我们正在通过 gateway 串联到真实的 vllm-omini。"
        + "测试服务端的abort行为," * 30
        + "看尾音长度。"
    )

    async with make_gateway(engine, max_concurrent=args.max_concurrent) as (client, app):
        # 1. Happy path
        m1 = await scenario_happy_path(client, args.voice, short_text)
        print(f"  TTFB={m1['ttfb_ms']:.1f}ms  audio={m1['audio_bytes']} bytes "
              f"chunks={m1['audio_chunks']} duration={m1['duration_ms']}ms "
              f"wall={m1['wall_ms']:.0f}ms")

        # 2. Cancel @ 200ms
        m2 = await scenario_cancel_midstream(client, args.voice, long_text, 200.0)
        print(f"  TTFB={m2.get('ttfb_ms') or 0:.1f}ms")
        print(f"  before_cancel: {m2['audio_bytes']} bytes / {m2['audio_chunks']} chunks")
        print(f"  AFTER cancel:  {m2['audio_after_cancel_bytes']} bytes / "
              f"{m2['audio_after_cancel_chunks']} chunks")
        print(f"  abort_latency: {m2.get('abort_latency_ms') or 0:.1f}ms  "
              f"got_cancelled={m2['got_cancelled']}")

        # 3. Cancel @ 500ms
        m3 = await scenario_cancel_midstream(client, args.voice, long_text, 500.0)
        print(f"  TTFB={m3.get('ttfb_ms') or 0:.1f}ms")
        print(f"  before_cancel: {m3['audio_bytes']} bytes / {m3['audio_chunks']} chunks")
        print(f"  AFTER cancel:  {m3['audio_after_cancel_bytes']} bytes / "
              f"{m3['audio_after_cancel_chunks']} chunks")
        print(f"  abort_latency: {m3.get('abort_latency_ms') or 0:.1f}ms  "
              f"got_cancelled={m3['got_cancelled']}")

        # 4. Two concurrent
        m4 = await scenario_two_concurrent(client, args.voice, short_text)
        print(f"  A: TTFB={m4['A']['ttfb_ms']:.1f}ms wall={m4['A']['wall_ms']:.0f}ms "
              f"queued_pos={m4['A']['queued_position']}")
        print(f"  B: TTFB={m4['B']['ttfb_ms']:.1f}ms wall={m4['B']['wall_ms']:.0f}ms "
              f"queued_pos={m4['B']['queued_position']}")

    # Compute tail-audio in ms for the cancel scenarios.
    bps = args.sample_rate * 2  # pcm16 mono
    print("\n========== SUMMARY ==========")
    print(f"engine: {args.engine_url} model={args.model} voice={args.voice}")
    print(f"sample_rate (assumed): {args.sample_rate}Hz  -> bytes_per_sec={bps}")
    print(f"\nHappy path:")
    print(f"  TTFB: {m1['ttfb_ms']:.0f}ms")
    print(f"  generated audio: {m1['duration_ms']}ms (gateway-reported), "
          f"received {m1['audio_bytes']} bytes ({m1['audio_bytes']*1000//bps}ms @ {args.sample_rate}Hz)")
    print(f"  wall: {m1['wall_ms']:.0f}ms")

    for tag, m in (("cancel@200ms", m2), ("cancel@500ms", m3)):
        tail_ms = m["audio_after_cancel_bytes"] * 1000 / bps if bps else 0
        print(f"\n{tag}:")
        print(f"  TTFB: {(m.get('ttfb_ms') or 0):.0f}ms")
        print(f"  abort_latency: {(m.get('abort_latency_ms') or 0):.1f}ms")
        print(f"  audio AFTER cancel: {m['audio_after_cancel_bytes']} bytes "
              f"= {tail_ms:.1f}ms tail audio @ {args.sample_rate}Hz")
        print(f"  got tts.cancelled: {m['got_cancelled']}")

    print(f"\n2 concurrent (max_concurrent={args.max_concurrent}):")
    print(f"  A: TTFB={m4['A']['ttfb_ms']:.0f}ms  pos={m4['A']['queued_position']}")
    print(f"  B: TTFB={m4['B']['ttfb_ms']:.0f}ms  pos={m4['B']['queued_position']}")
    if args.max_concurrent >= 2:
        print(f"  (no queueing expected -- both should have similar TTFB)")
    else:
        print(f"  (max_concurrent=1, B should pay queue wait + own TTFB)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--engine-url", default=os.environ.get("ENGINE_URL", "http://localhost:8091"))
    p.add_argument("--model", default=os.environ.get("MODEL", "cosyvoice3"))
    p.add_argument("--voice", default=os.environ.get("VOICE", "test_voice_v3"))
    p.add_argument("--sample-rate", type=int, default=int(os.environ.get("SAMPLE_RATE", "24000")))
    p.add_argument("--max-concurrent", type=int, default=32)
    args = p.parse_args()
    asyncio.run(run_all(args))


if __name__ == "__main__":
    main()

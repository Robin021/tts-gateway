"""Stress harness: validates the gateway under realistic load.

Scenario:
  - N concurrent WebSocket clients (default 50)
  - Each pushes a tts.text every ~200ms with sentence-ending punctuation
  - 5% of sessions cancel after a uniform random delay in [0, 500ms]
  - Run for DURATION seconds (default 60)

We measure (and dump as JSON):
  - TTFB: time from tts.start sent to first audio byte received
  - abort_latency: time from tts.cancel sent to tts.cancelled received
  - synthesis_rtf: wall_time / generated_audio_duration_ms
  - backpressure_count: how many sessions were terminated for BACKPRESSURE
  - scheduler_max_waiting: peak queue depth (observed at sample)
  - errors: any unexpected error codes seen
  - active_tasks_at_end: sanity check for task leaks

Run with:
  python -m tests.stress_run --connections 50 --duration 60

This uses the in-process ASGI transport (no real socket) so it works in
sandboxes. For end-to-end stress against a deployed engine, swap in a
real `ws://host/ws/tts` URL via httpx_ws.aconnect_ws (no transport).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Make the package importable when run as a module.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx
import wsproto
from httpx_ws import WebSocketDisconnect, aconnect_ws
from httpx_ws.transport import ASGIWebSocketTransport

from tts_gateway.app import GatewaySettings, create_app
from tts_gateway.engine import MockEngine
from tts_gateway.session import ACTIVE_SESSIONS


END_OF_STREAM = (asyncio.TimeoutError, TimeoutError, WebSocketDisconnect)


@dataclass
class SessionMetrics:
    started: bool = False
    completed: bool = False
    cancelled: bool = False
    error_code: Optional[str] = None

    tts_start_at: Optional[float] = None
    first_audio_at: Optional[float] = None
    cancel_sent_at: Optional[float] = None
    cancelled_received_at: Optional[float] = None
    done_received_at: Optional[float] = None

    audio_chunks: int = 0
    generated_duration_ms: int = 0


@dataclass
class StressMetrics:
    sessions: list[SessionMetrics] = field(default_factory=list)
    scheduler_max_waiting: int = 0
    active_sessions_peak: int = 0
    start_at: float = field(default_factory=time.monotonic)
    end_at: Optional[float] = None

    def add(self, sm: SessionMetrics) -> None:
        self.sessions.append(sm)


# ---------------------------------------------------------------
# Per-client driver
# ---------------------------------------------------------------


SAMPLE_TEXTS = [
    "你好,这是一个测试。",
    "今天天气真不错,我们去散步吧。",
    "请问最近在忙什么？",
    "代码已经写完了,正在跑压测。",
    "服务端的延迟看起来还可以。",
]


async def drive_session(
    app,
    client_id: str,
    request_id: str,
    duration_s: float,
    cancel_after_s: Optional[float],
) -> SessionMetrics:
    sm = SessionMetrics()
    deadline = time.monotonic() + duration_s

    # Each driver gets its own httpx client + transport. Sharing across
    # tasks trips anyio's "different task" cancel-scope check inside
    # ASGIWebSocketTransport. The app, scheduler, and engine are still
    # shared because we pass `app` directly into ASGIWebSocketTransport.
    transport = ASGIWebSocketTransport(app=app)
    try:
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://stress",
            timeout=httpx.Timeout(None),  # WS sessions can sit idle while queued
        ) as client:
            async with aconnect_ws(
                "http://stress/ws/tts",
                client=client,
                subprotocols=["tts.v1"],
                keepalive_ping_interval_seconds=None,
            ) as ws:
                await _drive_one(ws, client_id, request_id, deadline, cancel_after_s, sm)
    except END_OF_STREAM:
        pass
    except asyncio.CancelledError:
        # ASGIWebSocketTransport's anyio cancel scope can leak CancelledError
        # when the server closes the socket while we're still in the context
        # manager. Treat as a normal early termination — if we got an error
        # code from the gateway, it's already in sm.
        if sm.error_code is None and not sm.completed and not sm.cancelled:
            sm.error_code = "TRANSPORT_CANCELLED"
    except Exception as exc:
        sm.error_code = f"CLIENT_EXC:{type(exc).__name__}"
    return sm


async def _drive_one(
    ws,
    client_id: str,
    request_id: str,
    deadline: float,
    cancel_after_s: Optional[float],
    sm: SessionMetrics,
) -> None:
    # Auth.
    await ws.send_text(json.dumps({
        "type": "auth", "token": "test-token", "client_id": client_id,
    }))
    ack_raw = await ws.receive(timeout=5.0)
    ack = (
        json.loads(ack_raw.data)
        if isinstance(ack_raw, wsproto.events.TextMessage)
        else {}
    )
    if ack.get("type") != "auth.ok":
        sm.error_code = "AUTH_FAILED"
        return

    # Start.
    await ws.send_text(json.dumps({
        "type": "tts.start", "request_id": request_id,
    }))
    sm.tts_start_at = time.monotonic()

    # Don't push text until tts.processing arrives. A real client would do
    # this too — there's no point queueing audio synthesis input while the
    # session is waiting for a GPU slot, since vllm-omini's batcher hasn't
    # seen the request yet either.
    processing_event = asyncio.Event()

    feeder_done = asyncio.Event()

    async def feeder():
        try:
            await processing_event.wait()
            while time.monotonic() < deadline:
                await asyncio.sleep(0.2)
                if cancel_after_s is not None and sm.tts_start_at and \
                        time.monotonic() - sm.tts_start_at > cancel_after_s:
                    return
                text = random.choice(SAMPLE_TEXTS)
                try:
                    await ws.send_text(json.dumps({
                        "type": "tts.text", "text": text,
                    }))
                except Exception:
                    return
            try:
                await ws.send_text(json.dumps({"type": "tts.end"}))
            except Exception:
                pass
        finally:
            feeder_done.set()

    feeder_task = asyncio.create_task(feeder())

    if cancel_after_s is not None:
        async def canceller():
            await asyncio.sleep(cancel_after_s)
            sm.cancel_sent_at = time.monotonic()
            try:
                await ws.send_text(json.dumps({"type": "tts.cancel"}))
            except Exception:
                pass

        asyncio.create_task(canceller())

    try:
        while True:
            try:
                evt = await ws.receive(timeout=10.0)
            except END_OF_STREAM:
                break
            if isinstance(evt, wsproto.events.BytesMessage):
                if sm.first_audio_at is None:
                    sm.first_audio_at = time.monotonic()
                sm.audio_chunks += 1
                continue
            if isinstance(evt, wsproto.events.CloseConnection):
                break
            if not isinstance(evt, wsproto.events.TextMessage):
                continue
            data = json.loads(evt.data)
            t = data.get("type")
            if t == "tts.started":
                sm.started = True
            elif t == "tts.processing":
                processing_event.set()
            elif t == "tts.done":
                sm.completed = True
                sm.done_received_at = time.monotonic()
                sm.generated_duration_ms = int(data.get("generated_duration_ms", 0))
                break
            elif t == "tts.cancelled":
                sm.cancelled = True
                sm.cancelled_received_at = time.monotonic()
                break
            elif t == "tts.error":
                sm.error_code = data.get("code")
                break
    finally:
        feeder_task.cancel()
        try:
            await feeder_task
        except Exception:
            pass


# ---------------------------------------------------------------
# Sampler: poll scheduler.waiting and ACTIVE_SESSIONS
# ---------------------------------------------------------------


async def sampler(
    metrics: StressMetrics,
    scheduler,
    interval_s: float,
    stop: asyncio.Event,
) -> None:
    while not stop.is_set():
        try:
            metrics.scheduler_max_waiting = max(
                metrics.scheduler_max_waiting, scheduler.waiting,
            )
            metrics.active_sessions_peak = max(
                metrics.active_sessions_peak, len(ACTIVE_SESSIONS),
            )
        except Exception:
            pass
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval_s)
        except asyncio.TimeoutError:
            pass


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------


async def run_stress(
    *,
    connections: int,
    duration_s: float,
    cancel_rate: float,
    cancel_max_delay_s: float,
    max_concurrent_synthesis: int,
    chunks_per_sentence: int,
    per_chunk_delay_s: float,
) -> dict:
    engine = MockEngine(
        chunks_per_sentence=chunks_per_sentence,
        per_chunk_delay_s=per_chunk_delay_s,
        first_chunk_extra_delay_s=0.02,
    )
    settings = GatewaySettings(
        max_concurrent_synthesis=max_concurrent_synthesis,
        # Generous timeouts for stress — we don't want false IDLE_TIMEOUT.
        auth_timeout_s=10.0,
        start_timeout_s=10.0,
        idle_timeout_s=max(60.0, duration_s + 30.0),
        max_session_s=max(120.0, duration_s + 60.0),
    )

    app = create_app(engine=engine, settings=settings)
    scheduler = app.state.scheduler

    # Manually drive ASGI lifespan startup so engine.load() runs before
    # any client opens. We also call shutdown at the end for clean teardown.
    lifespan_send: list = []
    lifespan_recv = asyncio.Queue()

    async def lifespan_app_send(msg):
        lifespan_send.append(msg)

    async def lifespan_app_recv():
        return await lifespan_recv.get()

    lifespan_task = asyncio.create_task(
        app({"type": "lifespan", "asgi": {"version": "3.0"}}, lifespan_app_recv, lifespan_app_send)
    )
    await lifespan_recv.put({"type": "lifespan.startup"})
    # Wait for startup.complete
    for _ in range(100):
        if any(m.get("type") == "lifespan.startup.complete" for m in lifespan_send):
            break
        await asyncio.sleep(0.02)

    metrics = StressMetrics()
    stop = asyncio.Event()
    sampler_task = asyncio.create_task(sampler(metrics, scheduler, 0.05, stop))

    async def one_client(i: int):
        # Stagger connections slightly — slamming N at once trips
        # ASGIWebSocketTransport's anyio scope handling.
        await asyncio.sleep(i * 0.01)
        cancel_after = (
            random.uniform(0.0, cancel_max_delay_s)
            if random.random() < cancel_rate
            else None
        )
        try:
            sm = await drive_session(
                app=app,
                client_id=f"c-{i}",
                request_id=f"req-{i}",
                duration_s=duration_s,
                cancel_after_s=cancel_after,
            )
        except Exception as exc:
            sm = SessionMetrics(error_code=f"DRIVER_EXC:{type(exc).__name__}:{exc}")
        metrics.add(sm)

    tasks = [asyncio.create_task(one_client(i)) for i in range(connections)]
    await asyncio.gather(*tasks, return_exceptions=True)

    stop.set()
    await sampler_task

    # Lifespan shutdown.
    await lifespan_recv.put({"type": "lifespan.shutdown"})
    try:
        await asyncio.wait_for(lifespan_task, timeout=10.0)
    except asyncio.TimeoutError:
        lifespan_task.cancel()

    metrics.end_at = time.monotonic()
    return summarize(metrics, scheduler_max_concurrent=max_concurrent_synthesis)


def _percentile(values: list[float], p: float) -> Optional[float]:
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def summarize(metrics: StressMetrics, scheduler_max_concurrent: int) -> dict:
    ttfbs = [
        sm.first_audio_at - sm.tts_start_at
        for sm in metrics.sessions
        if sm.first_audio_at and sm.tts_start_at
    ]
    abort_lats = [
        sm.cancelled_received_at - sm.cancel_sent_at
        for sm in metrics.sessions
        if sm.cancelled_received_at and sm.cancel_sent_at
    ]
    rtfs = [
        (sm.done_received_at - sm.tts_start_at) / (sm.generated_duration_ms / 1000.0)
        for sm in metrics.sessions
        if sm.completed and sm.generated_duration_ms > 0 and sm.done_received_at and sm.tts_start_at
    ]

    error_codes: dict[str, int] = {}
    for sm in metrics.sessions:
        if sm.error_code:
            error_codes[sm.error_code] = error_codes.get(sm.error_code, 0) + 1

    completed = sum(1 for sm in metrics.sessions if sm.completed)
    cancelled = sum(1 for sm in metrics.sessions if sm.cancelled)

    return {
        "wall_time_s": (metrics.end_at or time.monotonic()) - metrics.start_at,
        "sessions": len(metrics.sessions),
        "completed": completed,
        "cancelled": cancelled,
        "errors": error_codes,
        "ttfb_s": {
            "n": len(ttfbs),
            "p50": _percentile(ttfbs, 0.5),
            "p95": _percentile(ttfbs, 0.95),
            "p99": _percentile(ttfbs, 0.99),
            "max": max(ttfbs) if ttfbs else None,
        },
        "abort_latency_s": {
            "n": len(abort_lats),
            "p50": _percentile(abort_lats, 0.5),
            "p95": _percentile(abort_lats, 0.95),
            "p99": _percentile(abort_lats, 0.99),
            "max": max(abort_lats) if abort_lats else None,
        },
        "rtf": {
            "n": len(rtfs),
            "p50": _percentile(rtfs, 0.5),
            "p95": _percentile(rtfs, 0.95),
            "max": max(rtfs) if rtfs else None,
        },
        "scheduler": {
            "max_concurrent": scheduler_max_concurrent,
            "max_waiting": metrics.scheduler_max_waiting,
        },
        "active_sessions_peak": metrics.active_sessions_peak,
        "active_sessions_at_end": len(ACTIVE_SESSIONS),
        "active_tasks_at_end": len([t for t in asyncio.all_tasks() if not t.done()]),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--connections", type=int, default=50)
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--cancel-rate", type=float, default=0.05)
    p.add_argument("--cancel-max-delay", type=float, default=0.5)
    p.add_argument("--max-concurrent-synthesis", type=int, default=8)
    p.add_argument("--chunks-per-sentence", type=int, default=10)
    p.add_argument("--per-chunk-delay", type=float, default=0.04)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    summary = asyncio.run(run_stress(
        connections=args.connections,
        duration_s=args.duration,
        cancel_rate=args.cancel_rate,
        cancel_max_delay_s=args.cancel_max_delay,
        max_concurrent_synthesis=args.max_concurrent_synthesis,
        chunks_per_sentence=args.chunks_per_sentence,
        per_chunk_delay_s=args.per_chunk_delay,
    ))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()

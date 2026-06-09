"""End-to-end integration tests using httpx_ws + ASGIWebSocketTransport.

ASGIWebSocketTransport runs the ASGI app in-process — no socket bind, so
this works in restricted sandboxes. The WS API is fully async, which means
we can use asyncio.wait_for for timeouts, unlike TestClient.
"""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx
import wsproto
from httpx_ws import WebSocketDisconnect, aconnect_ws
from httpx_ws.transport import ASGIWebSocketTransport

from tts_gateway.app import GatewaySettings, create_app
from tts_gateway.engine import MockEngine


# Exceptions that mean "no more frames coming" — wrap calls in this set.
END_OF_STREAM = (asyncio.TimeoutError, TimeoutError, WebSocketDisconnect)


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------


@asynccontextmanager
async def make_ws(
    engine: MockEngine | None = None,
    settings: GatewaySettings | None = None,
) -> AsyncIterator[tuple[httpx.AsyncClient, "AsyncWebSocketSession"]]:
    app = create_app(engine=engine, settings=settings or GatewaySettings())
    transport = ASGIWebSocketTransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        # ASGI lifespan startup
        async with aconnect_ws(
            "http://testserver/ws/tts",
            client=client,
            subprotocols=["tts.v1"],
            keepalive_ping_interval_seconds=None,  # we test heartbeat behavior separately
        ) as ws:
            yield client, ws


async def auth(ws, token: str = "test-token", client_id: str = "client-A") -> dict:
    await ws.send_text(json.dumps({"type": "auth", "token": token, "client_id": client_id}))
    msg = await asyncio.wait_for(ws.receive_text(), timeout=2.0)
    return json.loads(msg)


async def recv_either(ws, timeout: float = 5.0):
    """Receive next text or bytes frame. Returns (kind, payload).

    httpx_ws exposes `receive()` which returns a wsproto event we can switch on.
    On server-initiated close it raises WebSocketDisconnect — callers should
    catch that explicitly when they expect a close.
    """
    event = await ws.receive(timeout=timeout)
    if isinstance(event, wsproto.events.TextMessage):
        return ("text", event.data)
    if isinstance(event, wsproto.events.BytesMessage):
        return ("bytes", event.data)
    if isinstance(event, wsproto.events.CloseConnection):
        return ("close", event)
    return ("other", event)


# ---------------------------------------------------------------
# Tests
# ---------------------------------------------------------------


def test_happy_path_streams_audio_and_done():
    async def _():
        engine = MockEngine(
            chunks_per_sentence=5,
            per_chunk_delay_s=0.005,
            first_chunk_extra_delay_s=0.0,
        )
        async with make_ws(engine=engine) as (_client, ws):
            ack = await auth(ws)
            assert ack["type"] == "auth.ok"

            await ws.send_text(json.dumps({"type": "tts.start", "request_id": "req-1"}))

            seen = set()
            audio_chunks = 0
            duration_ms = None
            done = False
            started_at = None
            first_audio_at = None

            while not done:
                kind, payload = await recv_either(ws)
                if kind == "bytes":
                    if first_audio_at is None:
                        first_audio_at = time.monotonic()
                    audio_chunks += 1
                    continue
                if kind == "close":
                    break
                data = json.loads(payload)
                seen.add(data["type"])
                if data["type"] == "tts.started":
                    assert data["sample_rate"] == 24000
                    await ws.send_text(json.dumps({"type": "tts.text", "text": "你好。"}))
                    await ws.send_text(json.dumps({"type": "tts.end"}))
                    started_at = time.monotonic()
                elif data["type"] == "tts.done":
                    duration_ms = data["generated_duration_ms"]
                    done = True
                elif data["type"] == "tts.error":
                    raise AssertionError(f"unexpected error: {data}")

            assert "tts.queued" in seen
            assert "tts.processing" in seen
            assert audio_chunks == 5
            assert duration_ms and duration_ms > 0
            assert first_audio_at - started_at < 1.0
    asyncio.run(_())


def test_cancel_terminates_stream():
    async def _():
        engine = MockEngine(
            chunks_per_sentence=200,
            per_chunk_delay_s=0.05,
            first_chunk_extra_delay_s=0.0,
        )
        async with make_ws(engine=engine) as (_client, ws):
            await auth(ws)
            await ws.send_text(json.dumps({"type": "tts.start", "request_id": "req-c"}))

            saw_audio = False
            while not saw_audio:
                kind, payload = await recv_either(ws)
                if kind == "bytes":
                    saw_audio = True
                elif kind == "text":
                    data = json.loads(payload)
                    if data["type"] == "tts.started":
                        await ws.send_text(json.dumps({
                            "type": "tts.text",
                            "text": "这是一个非常长的句子。",
                        }))
                        await ws.send_text(json.dumps({"type": "tts.end"}))

            cancel_sent = time.monotonic()
            await ws.send_text(json.dumps({"type": "tts.cancel"}))

            got_cancelled = False
            cancelled_at = None
            for _ in range(300):
                try:
                    kind, payload = await recv_either(ws, timeout=3.0)
                except END_OF_STREAM:
                    break
                if kind == "close":
                    break
                if kind == "bytes":
                    continue
                data = json.loads(payload)
                if data["type"] == "tts.cancelled":
                    got_cancelled = True
                    cancelled_at = time.monotonic()
                    break
            assert got_cancelled
            assert cancelled_at - cancel_sent < 1.5
    asyncio.run(_())


def test_unauthorized_token_closes_socket():
    async def _():
        from httpx_ws import WebSocketDisconnect
        async with make_ws(settings=GatewaySettings(valid_tokens=("good",))) as (_client, ws):
            await ws.send_text(json.dumps({"type": "auth", "token": "bad"}))
            got_error = False
            for _ in range(5):
                try:
                    kind, payload = await recv_either(ws, timeout=2.0)
                except END_OF_STREAM:
                    break
                if kind == "close":
                    break
                if kind == "text":
                    data = json.loads(payload)
                    if data.get("code") == "UNAUTHORIZED":
                        got_error = True
            assert got_error
    asyncio.run(_())


def test_text_before_start_is_bad_state():
    async def _():
        async with make_ws() as (_client, ws):
            await auth(ws)
            await ws.send_text(json.dumps({"type": "tts.text", "text": "hi"}))
            kind, payload = await recv_either(ws, timeout=2.0)
            assert kind == "text"
            data = json.loads(payload)
            assert data["type"] == "tts.error"
            assert data["code"] == "BAD_STATE"
    asyncio.run(_())


def test_unknown_event_does_not_kill_session():
    async def _():
        async with make_ws() as (_client, ws):
            await auth(ws)
            await ws.send_text(json.dumps({"type": "tts.bogus"}))
            kind, payload = await recv_either(ws, timeout=2.0)
            data = json.loads(payload)
            assert data["type"] == "tts.error"
            assert data["code"] == "UNKNOWN_EVENT"

            await ws.send_text(json.dumps({"type": "tts.start", "request_id": "x"}))
            got_started = False
            for _ in range(15):
                kind, payload = await recv_either(ws, timeout=2.0)
                if kind == "text" and json.loads(payload)["type"] == "tts.started":
                    got_started = True
                    break
            assert got_started
    asyncio.run(_())


def test_done_arrives_after_audio():
    async def _():
        engine = MockEngine(
            chunks_per_sentence=10,
            per_chunk_delay_s=0.005,
            first_chunk_extra_delay_s=0.0,
        )
        async with make_ws(engine=engine) as (_client, ws):
            await auth(ws)
            await ws.send_text(json.dumps({"type": "tts.start", "request_id": "r"}))

            while True:
                kind, payload = await recv_either(ws, timeout=2.0)
                if kind == "text" and json.loads(payload)["type"] == "tts.started":
                    break

            await ws.send_text(json.dumps({"type": "tts.text", "text": "你好。"}))
            await ws.send_text(json.dumps({"type": "tts.end"}))

            order: list[str] = []
            while True:
                kind, payload = await recv_either(ws, timeout=5.0)
                if kind == "close":
                    break
                if kind == "bytes":
                    order.append("audio")
                else:
                    data = json.loads(payload)
                    order.append(data["type"])
                    if data["type"] == "tts.done":
                        break
            last_audio = max(i for i, x in enumerate(order) if x == "audio")
            done_idx = order.index("tts.done")
            assert last_audio < done_idx
    asyncio.run(_())


def test_idle_timeout_after_start_with_no_text():
    async def _():
        async with make_ws(settings=GatewaySettings(
            idle_timeout_s=1.0, max_session_s=60.0,
        )) as (_client, ws):
            await auth(ws)
            await ws.send_text(json.dumps({"type": "tts.start", "request_id": "i"}))
            got_idle = False
            for _ in range(30):
                try:
                    kind, payload = await recv_either(ws, timeout=3.0)
                except END_OF_STREAM:
                    break
                if kind == "close":
                    break
                if kind == "bytes":
                    continue
                data = json.loads(payload)
                if data["type"] == "tts.error" and data["code"] == "IDLE_TIMEOUT":
                    got_idle = True
                    break
            assert got_idle
    asyncio.run(_())


def test_engine_error_yields_error_terminal_frame():
    async def _():
        engine = MockEngine(
            chunks_per_sentence=1,
            per_chunk_delay_s=0.001,
            fail_on_text="boom",
        )
        async with make_ws(engine=engine) as (_client, ws):
            await auth(ws)
            await ws.send_text(json.dumps({"type": "tts.start", "request_id": "e"}))
            while True:
                kind, payload = await recv_either(ws, timeout=2.0)
                if kind == "text" and json.loads(payload)["type"] == "tts.started":
                    break
            await ws.send_text(json.dumps({"type": "tts.text", "text": "boom。"}))
            await ws.send_text(json.dumps({"type": "tts.end"}))

            got_engine_error = False
            for _ in range(30):
                try:
                    kind, payload = await recv_either(ws, timeout=3.0)
                except END_OF_STREAM:
                    break
                if kind == "close":
                    break
                if kind == "bytes":
                    continue
                data = json.loads(payload)
                if data["type"] == "tts.error" and data["code"] == "ENGINE_ERROR":
                    got_engine_error = True
                    break
            assert got_engine_error
    asyncio.run(_())

"""FastAPI app: WebSocket endpoint, lifespan, auth, request_id factory.

The endpoint is a thin protocol adapter. All real logic lives in
session.py / loops.py. The endpoint's job is:

  1. Accept the WS, run first-frame auth.
  2. Spawn sender / heartbeat / idle_watchdog tasks.
  3. Read JSON frames in a loop, dispatch to control plane (cancel/end)
     or data plane (text). Control plane never blocks on a queue.
  4. On exit (disconnect or any exception) call close() once.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Literal, Optional

from fastapi import FastAPI, Header, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .engine import MockEngine, TtsBackend
from .loops import (
    heartbeat_loop,
    idle_watchdog_loop,
    run_tts,
    sender_loop,
    supervisor_loop,
)
from .protocol import ErrorCode, PCM16_24K_MONO, SessionState
from .scheduler import EngineScheduler
from .sentence_buffer import SentenceBuffer
from .session import (
    ACTIVE_SESSIONS,
    TTSSession,
    cancel_and_close,
    fail_and_close,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------
# Settings
# ---------------------------------------------------------------


@dataclass
class GatewaySettings:
    # max_concurrent should be >= vllm-omini's max_num_seqs. The gateway
    # serializes nothing in the synthesis path other than this semaphore,
    # and vllm-omini does its own batching internally. Setting this too
    # low makes the gateway artificially queue requests that the engine
    # could happily run in parallel.
    max_concurrent_synthesis: int = 32
    auth_timeout_s: float = 5.0
    start_timeout_s: float = 10.0
    idle_timeout_s: float = 30.0
    max_session_s: float = 300.0
    heartbeat_interval_s: float = 20.0
    # Tokens accepted by the demo auth. Replace with JWT/API-key in prod.
    valid_tokens: tuple[str, ...] = ("test-token",)


# ---------------------------------------------------------------
# Pydantic models for OpenAPI / Swagger documentation
# ---------------------------------------------------------------


class SpeechRequest(BaseModel):
    """Request body for `POST /v1/audio/speech`. OpenAI-compatible."""

    input: str = Field(
        ...,
        description="Text to synthesize. UTF-8, no length limit enforced here "
                    "(vllm-omini may truncate to its `max_model_len`).",
        examples=["你好,这是一个测试。"],
        min_length=1,
    )
    voice: Optional[str] = Field(
        default=None,
        description="Speaker file name. Must already be uploaded to "
                    "vllm-omini (see `GET /v1/audio/voices`). If omitted, "
                    "the gateway's configured default voice is used.",
        examples=["female", "male", "linzhiling"],
    )
    model: Optional[str] = Field(
        default=None,
        description="Ignored — kept for OpenAI SDK compatibility. The "
                    "actual model served is configured at deploy time.",
        examples=["cosyvoice3", "tts-1"],
    )
    response_format: Literal["pcm", "wav"] = Field(
        default="pcm",
        description=(
            "Audio container format.\n"
            "- `pcm`: raw little-endian 16-bit signed mono samples at the "
            "  engine's sample rate (24kHz for cosyvoice3).\n"
            "- `wav`: standard WAV header + same PCM payload."
        ),
    )
    stream: bool = Field(
        default=False,
        description="When true, response body is sent as chunked HTTP, "
                    "delivering audio bytes to the client as they're "
                    "synthesized. When false (default), the server "
                    "buffers the full audio and returns it in one body.",
    )
    speed: Optional[float] = Field(
        default=None,
        description="Ignored — kept for OpenAI SDK compatibility.",
        ge=0.25, le=4.0,
    )
    request_id: Optional[str] = Field(
        default=None,
        description="Optional client-provided id. If absent, the gateway "
                    "generates one. Echoed in error responses.",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "input": "你好,世界。",
                    "voice": "female",
                    "response_format": "wav",
                },
                {
                    "input": "Streaming low-latency synthesis.",
                    "voice": "male",
                    "response_format": "pcm",
                    "stream": True,
                },
            ]
        }
    }


class VoiceInfo(BaseModel):
    """Information about a single uploaded voice."""

    name: str = Field(..., description="Voice file name; pass this in `voice` field.")
    consent: Optional[str] = None
    created_at: Optional[int] = None
    file_size: Optional[int] = None
    mime_type: Optional[str] = None
    embedding_source: Optional[str] = None
    ref_text: Optional[str] = Field(
        default=None,
        description="Transcript of the reference audio. Includes the "
                    "`<|endofprompt|>` prompt prefix required by CosyVoice3.",
    )


class VoicesListResponse(BaseModel):
    voices: list[str] = Field(
        ...,
        description="Names available for use in the `voice` field.",
        examples=[["female", "male", "linzhiling"]],
    )
    uploaded_voices: list[VoiceInfo] = Field(
        ...,
        description="Detailed metadata per voice.",
    )


# ---------------------------------------------------------------
# Engine request-id factory: client_id : client_request_id : seq
# ---------------------------------------------------------------


class RequestIdFactory:
    def __init__(self) -> None:
        self._seq = 0
        self._lock = asyncio.Lock()

    async def make(self, client_id: str, client_request_id: str) -> str:
        async with self._lock:
            self._seq += 1
            return f"{client_id}:{client_request_id}:{self._seq}"


# ---------------------------------------------------------------
# Auth (first-frame, post-accept)
# ---------------------------------------------------------------


@dataclass
class AuthContext:
    client_id: str


async def _do_first_frame_auth(
    websocket: WebSocket,
    settings: GatewaySettings,
) -> Optional[AuthContext]:
    """Read exactly one JSON frame, expect {type: 'auth', token: '...'}.

    Returns AuthContext on success, None on failure (caller closes).
    Bounded by settings.auth_timeout_s; idle_watchdog also enforces
    the same deadline as a backstop.
    """
    try:
        raw = await asyncio.wait_for(
            websocket.receive(),
            timeout=settings.auth_timeout_s,
        )
    except asyncio.TimeoutError:
        return None

    if raw.get("type") == "websocket.disconnect":
        return None

    text = raw.get("text")
    if text is None:
        return None

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None

    if data.get("type") != "auth":
        return None

    token = data.get("token")
    if not isinstance(token, str) or token not in settings.valid_tokens:
        return None

    client_id = data.get("client_id")
    if not isinstance(client_id, str) or not client_id:
        client_id = f"anon-{int(time.time() * 1000)}"

    return AuthContext(client_id=client_id)


# ---------------------------------------------------------------
# App factory
# ---------------------------------------------------------------


def create_app(
    engine: Optional[TtsBackend] = None,
    settings: Optional[GatewaySettings] = None,
) -> FastAPI:
    settings = settings or GatewaySettings()
    engine = engine or MockEngine()
    scheduler = EngineScheduler(max_concurrent=settings.max_concurrent_synthesis)
    request_id_factory = RequestIdFactory()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await engine.load()
        try:
            yield
        finally:
            # Parallel shutdown: 100 sessions x serial drain would
            # blow past any reasonable graceful-stop deadline.
            sessions = list(ACTIVE_SESSIONS)
            for s in sessions:
                s.set_terminal_frame_once(
                    {
                        "type": "tts.error",
                        "request_id": s.request_id,
                        "code": ErrorCode.SERVER_SHUTDOWN,
                        "message": "server is shutting down",
                    },
                    kind="error",
                )
            if sessions:
                await asyncio.gather(
                    *[
                        s.close(drain=True, reason="server_shutdown")
                        for s in sessions
                    ],
                    return_exceptions=True,
                )
            await engine.shutdown()

    app = FastAPI(
        lifespan=lifespan,
        title="tts-gateway",
        version="0.1.0",
        summary="WebSocket + OpenAI-compatible HTTP gateway for vllm-omini / Fun-CosyVoice3",
        description=(
            "Two interfaces for the same TTS backend:\n\n"
            "**WebSocket `/ws/tts`** (recommended): subprotocol `tts.v1`, "
            "supports streaming text input, mid-flight cancel, queueing telemetry. "
            "See the project's `PROTOCOL.md` for the full event protocol. "
            "WebSocket endpoints aren't reflected in OpenAPI — only the HTTP "
            "endpoints below are documented here.\n\n"
            "**HTTP `/v1/audio/speech`**: OpenAI-compatible TTS endpoint. "
            "Drop-in replacement for OpenAI's `client.audio.speech.create()`.\n\n"
            "All endpoints require `Authorization: Bearer <token>`. "
            "Tokens are configured at deploy time via `GATEWAY_AUTH_TOKEN`."
        ),
        contact={"name": "tts-gateway"},
        openapi_tags=[
            {"name": "tts", "description": "Text-to-speech synthesis."},
            {"name": "voices", "description": "Voice (speaker) management."},
        ],
    )
    app.state.engine = engine
    app.state.scheduler = scheduler
    app.state.settings = settings
    app.state.request_id_factory = request_id_factory

    @app.websocket("/ws/tts")
    async def tts_ws(websocket: WebSocket) -> None:
        await websocket.accept(subprotocol="tts.v1")

        # First-frame auth. We must accept first because subprotocol
        # selection requires it, but we hard-bound the auth window so
        # an unauthenticated socket can't sit forever.
        auth = await _do_first_frame_auth(websocket, settings)
        if auth is None:
            with contextlib.suppress(Exception):
                await websocket.send_json(
                    {"type": "tts.error", "code": ErrorCode.UNAUTHORIZED}
                )
            with contextlib.suppress(Exception):
                await websocket.close(code=4401)
            return

        session = TTSSession(
            websocket=websocket,
            engine=engine,
            client_id=auth.client_id,
            audio_format=PCM16_24K_MONO,
        )
        session.state = SessionState.AUTHENTICATED
        ACTIVE_SESSIONS.add(session)
        sentence_buffer = SentenceBuffer()

        # Spin up sender / heartbeat / watchdog. tts_task starts on
        # tts.start; supervisor starts then too.
        session.sender_task = asyncio.create_task(
            sender_loop(session), name=f"sender-{auth.client_id}"
        )
        session.heartbeat_task = asyncio.create_task(
            heartbeat_loop(session, settings.heartbeat_interval_s),
            name=f"heartbeat-{auth.client_id}",
        )
        session.idle_watchdog_task = asyncio.create_task(
            idle_watchdog_loop(
                session,
                auth_timeout_s=settings.auth_timeout_s,
                start_timeout_s=settings.start_timeout_s,
                idle_timeout_s=settings.idle_timeout_s,
                max_session_s=settings.max_session_s,
            ),
            name=f"watchdog-{auth.client_id}",
        )

        session.send_json(
            {
                "type": "auth.ok",
                "client_id": auth.client_id,
            }
        )

        try:
            await _receive_loop(
                session=session,
                sentence_buffer=sentence_buffer,
                scheduler=scheduler,
                request_id_factory=request_id_factory,
            )
        except WebSocketDisconnect:
            await session.close(drain=False, reason="client_disconnected")
        except Exception as exc:
            logger.exception(
                "ws handler error request_id=%s", session.request_id
            )
            await fail_and_close(
                session,
                code=ErrorCode.SERVER_ERROR,
                message=str(exc),
                drain=True,
            )
        finally:
            # Belt-and-suspenders. close() is idempotent.
            await session.close(drain=False, reason="endpoint_exit")

    # ---------------------------------------------------------------
    # OpenAI-compatible HTTP endpoint
    # ---------------------------------------------------------------
    #
    # POST /v1/audio/speech
    #
    # Drop-in replacement for OpenAI's TTS endpoint. Clients written
    # against the OpenAI Python SDK can point `base_url` at this gateway
    # and Just Work. The `voice` field is a vllm-omini speaker file name
    # (same names as the WebSocket protocol) — supports `female`, `male`,
    # `linzhiling`, or any other voice you've uploaded.
    #
    # Differences from OpenAI:
    #   - `model` accepts whatever vllm-omini serves (e.g. "cosyvoice3"),
    #     not OpenAI's "tts-1" etc. We don't validate it here; we just
    #     forward to the engine.
    #   - `response_format`: pcm | wav (other formats not exposed yet).
    #   - `stream`: true → chunked HTTP body of raw audio bytes.
    #     false (default) → single Response with full audio.
    #
    # Auth: same `GATEWAY_AUTH_TOKEN` accepted via `Authorization: Bearer`.

    @app.post(
        "/v1/audio/speech",
        tags=["tts"],
        summary="Synthesize speech from text (OpenAI-compatible)",
        description=(
            "Drop-in replacement for OpenAI's TTS endpoint.\n\n"
            "**Request body** is JSON. The `voice` field is the vllm-omini "
            "speaker file name — see `GET /v1/audio/voices` for the list of "
            "uploaded voices.\n\n"
            "**Response body** is raw audio bytes (`audio/pcm` or `audio/wav` "
            "per `response_format`). When `stream=true` the body is sent as "
            "chunked HTTP and clients should consume it as it arrives.\n\n"
            "For interactive use cases (mid-flight cancel, streaming text "
            "input) use the WebSocket interface at `/ws/tts` instead — see "
            "`PROTOCOL.md`."
        ),
        responses={
            200: {
                "description": "Audio stream",
                "content": {
                    "audio/pcm": {"schema": {"type": "string", "format": "binary"}},
                    "audio/wav": {"schema": {"type": "string", "format": "binary"}},
                },
            },
            400: {"description": "Invalid request body"},
            401: {"description": "Missing or invalid Authorization header"},
            502: {"description": "Engine error"},
        },
    )
    async def create_speech(
        body: SpeechRequest,
        authorization: Optional[str] = Header(
            default=None,
            description="Bearer token. e.g. `Authorization: Bearer your-secret`",
        ),
    ):
        # Auth: Authorization: Bearer <token>
        token = None
        if authorization and authorization.startswith("Bearer "):
            token = authorization[len("Bearer "):]
        if token not in settings.valid_tokens:
            raise HTTPException(status_code=401, detail="invalid token")

        text = body.input
        voice = body.voice
        response_format = body.response_format
        stream = body.stream

        # Resolve concurrency slot the same way the WS path does, so HTTP
        # and WS clients share fairly. We don't use the queue-position
        # telemetry here.
        await scheduler.reserve()
        try:
            await scheduler.wait()
        except BaseException:
            await scheduler.cancel_reservation()
            raise

        # Build a one-shot text iterator and call engine.stream_tts.
        async def _text_iter():
            yield text

        # Synthesize a unique engine_request_id so concurrent HTTP
        # callers don't collide. Reuse the same factory to keep IDs
        # globally unique across HTTP + WS.
        client_request_id = body.request_id or f"http-{int(time.time()*1000)}"
        engine_req_id = await request_id_factory.make("http", client_request_id)

        async def _audio_stream():
            try:
                if response_format == "wav":
                    # Streaming WAV with unknown length: use the
                    # "size=0xFFFFFFFF" trick so decoders read until EOF.
                    yield _wav_header_streaming(
                        sample_rate=engine.sample_rate
                        if hasattr(engine, "sample_rate")
                        else 24000,
                        channels=1,
                        bytes_per_sample=2,
                    )
                async for chunk in engine.stream_tts(
                    request_id=engine_req_id,
                    text_iter=_text_iter(),
                    voice=voice or "default",
                    prompt_audio_id=voice,  # treat voice as the speaker name
                ):
                    if chunk:
                        yield chunk
            finally:
                scheduler.release()

        media_type = "audio/wav" if response_format == "wav" else "audio/pcm"

        if stream:
            return StreamingResponse(_audio_stream(), media_type=media_type)

        # Non-stream: collect everything, return single body.
        chunks: list[bytes] = []
        async for c in _audio_stream():
            chunks.append(c)
        return StreamingResponse(
            iter([b"".join(chunks)]),
            media_type=media_type,
        )

    @app.get(
        "/v1/audio/voices",
        tags=["voices"],
        summary="List available voices (speakers)",
        description=(
            "Forwards to vllm-omini's voice listing. Returns the names of "
            "uploaded reference speakers — these are the values clients "
            "pass in the `voice` field. To upload a new voice, use vllm-omini's "
            "`POST /v1/audio/voices` directly (see `scripts/upload_voices.sh`)."
        ),
        response_model=VoicesListResponse,
        responses={
            401: {"description": "Missing or invalid Authorization header"},
            501: {"description": "Engine doesn't support voice listing"},
            502: {"description": "Engine error"},
        },
    )
    async def list_voices(
        authorization: Optional[str] = Header(default=None),
    ):
        """Forward to vllm-omini's voice listing. Lets clients discover
        what voice names are available without going to the engine
        directly."""
        token = None
        if authorization and authorization.startswith("Bearer "):
            token = authorization[len("Bearer "):]
        if token not in settings.valid_tokens:
            raise HTTPException(status_code=401, detail="invalid token")

        if not hasattr(engine, "list_voices"):
            raise HTTPException(
                status_code=501,
                detail="engine does not support voice listing",
            )
        try:
            return await engine.list_voices()
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc))

    return app


def _wav_header_streaming(
    sample_rate: int, channels: int, bytes_per_sample: int
) -> bytes:
    """Build a WAV header that says 'unknown length' so we can stream.

    Decoders that respect RIFF chunk sizes will read until EOF when
    they see 0xFFFFFFFF. This is a well-known streaming trick.
    """
    import struct
    riff_size = 0xFFFFFFFF
    data_size = 0xFFFFFFFF
    byte_rate = sample_rate * channels * bytes_per_sample
    block_align = channels * bytes_per_sample
    bits_per_sample = bytes_per_sample * 8
    return (
        b"RIFF"
        + struct.pack("<I", riff_size)
        + b"WAVEfmt "
        + struct.pack(
            "<IHHIIHH",
            16,                # fmt chunk size
            1,                 # PCM
            channels,
            sample_rate,
            byte_rate,
            block_align,
            bits_per_sample,
        )
        + b"data"
        + struct.pack("<I", data_size)
    )


# ---------------------------------------------------------------
# receive_loop: control plane never blocks; data plane uses put_nowait
# ---------------------------------------------------------------


def _require_str(data: dict, key: str) -> Optional[str]:
    v = data.get(key)
    if v is None:
        return None
    if not isinstance(v, str):
        raise ValueError(f"{key} must be string")
    return v


async def _receive_loop(
    *,
    session: TTSSession,
    sentence_buffer: SentenceBuffer,
    scheduler: EngineScheduler,
    request_id_factory: RequestIdFactory,
) -> None:
    while True:
        try:
            raw = await session.websocket.receive()
        except (RuntimeError, WebSocketDisconnect):
            # Starlette raises RuntimeError if the client closes after a
            # close frame has already been received — treat it as disconnect.
            raise WebSocketDisconnect()
        session.last_client_msg_at = time.monotonic()

        if raw.get("type") == "websocket.disconnect":
            raise WebSocketDisconnect()

        # Reject binary frames — this endpoint is text-only.
        if raw.get("bytes") is not None:
            session.send_json(
                {
                    "type": "tts.error",
                    "request_id": session.request_id,
                    "code": ErrorCode.UNEXPECTED_BINARY,
                    "message": "binary frames not supported",
                }
            )
            continue

        text = raw.get("text")
        if text is None:
            continue

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            session.send_json(
                {
                    "type": "tts.error",
                    "request_id": session.request_id,
                    "code": ErrorCode.BAD_JSON,
                    "message": "invalid json",
                }
            )
            continue

        if not isinstance(data, dict):
            session.send_json(
                {
                    "type": "tts.error",
                    "request_id": session.request_id,
                    "code": ErrorCode.BAD_REQUEST,
                    "message": "frame must be a JSON object",
                }
            )
            continue

        msg_type = data.get("type")

        # ---------------- Control plane (never blocks) ----------------

        if msg_type == "tts.cancel":
            session.cancel_received_at = time.monotonic()
            await cancel_and_close(session, reason="client_cancelled")
            return

        if msg_type == "tts.start":
            await _handle_start(
                data=data,
                session=session,
                scheduler=scheduler,
                request_id_factory=request_id_factory,
            )
            continue

        if msg_type == "tts.end":
            await _handle_end(data=data, session=session, sentence_buffer=sentence_buffer)
            if session.state in (SessionState.CLOSING, SessionState.CLOSED):
                return
            continue

        if msg_type == "tts.flush":
            await _handle_flush(session=session, sentence_buffer=sentence_buffer)
            if session.state in (SessionState.CLOSING, SessionState.CLOSED):
                return
            continue

        # ---------------- Data plane (put_nowait, never await) -------

        if msg_type == "tts.text":
            await _handle_text(
                data=data,
                session=session,
                sentence_buffer=sentence_buffer,
            )
            if session.state in (SessionState.CLOSING, SessionState.CLOSED):
                return
            continue

        session.send_json(
            {
                "type": "tts.error",
                "request_id": session.request_id,
                "code": ErrorCode.UNKNOWN_EVENT,
                "message": f"unknown event type: {msg_type!r}",
            }
        )


async def _handle_start(
    *,
    data: dict,
    session: TTSSession,
    scheduler: EngineScheduler,
    request_id_factory: RequestIdFactory,
) -> None:
    if session.state != SessionState.AUTHENTICATED:
        session.send_json(
            {
                "type": "tts.error",
                "request_id": session.request_id,
                "code": ErrorCode.BAD_STATE,
                "message": "tts.start is only allowed once after auth",
            }
        )
        return

    try:
        client_request_id = _require_str(data, "request_id")
        # Two equivalent names: 'voice' (recommended, matches OpenAI TTS API
        # naming) and 'prompt_audio_id' (legacy). Either resolves to the
        # vllm-omini speaker file name.
        voice = _require_str(data, "voice")
        prompt_audio_id = _require_str(data, "prompt_audio_id")
    except ValueError as e:
        session.send_json(
            {
                "type": "tts.error",
                "request_id": None,
                "code": ErrorCode.BAD_REQUEST,
                "message": str(e),
            }
        )
        return

    if not client_request_id:
        session.send_json(
            {
                "type": "tts.error",
                "request_id": None,
                "code": ErrorCode.MISSING_REQUEST_ID,
                "message": "request_id is required",
            }
        )
        return

    assert session.client_id is not None
    session.request_id = client_request_id
    # Either field may carry the speaker name. Engine resolves priority.
    session.voice = voice or "default"
    session.prompt_audio_id = prompt_audio_id or voice
    session.engine_request_id = await request_id_factory.make(
        session.client_id, client_request_id
    )
    session.state = SessionState.STARTED
    session.started_at = time.monotonic()
    session.last_client_msg_at = session.started_at

    session.send_json(
        {
            "type": "tts.started",
            "request_id": session.request_id,
            "sample_rate": session.audio_format.sample_rate,
            "format": session.audio_format.name,
            "channels": session.audio_format.channels,
        }
    )

    # Spawn run_tts and supervisor. Supervisor watches run_tts +
    # idle_watchdog and triggers close() when either exits.
    session.tts_task = asyncio.create_task(
        run_tts(session, scheduler),
        name=f"tts-{session.engine_request_id}",
    )
    session.supervisor_task = asyncio.create_task(
        supervisor_loop(session),
        name=f"supervisor-{session.engine_request_id}",
    )


async def _handle_text(
    *,
    data: dict,
    session: TTSSession,
    sentence_buffer: SentenceBuffer,
) -> None:
    if session.state != SessionState.STARTED:
        session.send_json(
            {
                "type": "tts.error",
                "request_id": session.request_id,
                "code": ErrorCode.BAD_STATE,
                "message": "tts.text requires an active session",
            }
        )
        return

    try:
        text_value = _require_str(data, "text")
    except ValueError as exc:
        session.send_json(
            {
                "type": "tts.error",
                "request_id": session.request_id,
                "code": ErrorCode.BAD_REQUEST,
                "message": str(exc),
            }
        )
        return

    if not text_value:
        return

    flush_after = bool(data.get("flush", False))
    chunks = sentence_buffer.push(text_value)
    if flush_after:
        chunks.extend(sentence_buffer.flush())

    for chunk in chunks:
        if not session.put_text_nowait(chunk):
            # text_queue full = engine is way behind. Don't silently
            # drop audio chunks; bail out — voice with random gaps is
            # worse than a clear error.
            await fail_and_close(
                session,
                code=ErrorCode.BACKPRESSURE,
                message="server text queue is full; slow down input",
                drain=True,
            )
            return


async def _handle_flush(
    *,
    session: TTSSession,
    sentence_buffer: SentenceBuffer,
) -> None:
    if session.state != SessionState.STARTED:
        session.send_json(
            {
                "type": "tts.error",
                "request_id": session.request_id,
                "code": ErrorCode.BAD_STATE,
                "message": "tts.flush requires an active session",
            }
        )
        return

    for chunk in sentence_buffer.flush():
        if not session.put_text_nowait(chunk):
            await fail_and_close(
                session,
                code=ErrorCode.BACKPRESSURE,
                message="server text queue is full",
                drain=True,
            )
            return

    session.send_json(
        {
            "type": "tts.flushed",
            "request_id": session.request_id,
        }
    )


async def _handle_end(
    *,
    data: dict,
    session: TTSSession,
    sentence_buffer: SentenceBuffer,
) -> None:
    if session.state != SessionState.STARTED:
        session.send_json(
            {
                "type": "tts.error",
                "request_id": session.request_id,
                "code": ErrorCode.BAD_STATE,
                "message": "tts.end requires an active session",
            }
        )
        return

    session.state = SessionState.ENDING

    for chunk in sentence_buffer.flush():
        if not session.put_text_nowait(chunk):
            await fail_and_close(
                session,
                code=ErrorCode.BACKPRESSURE,
                message="server text queue is full",
                drain=True,
            )
            return

    if not session.put_text_end_nowait():
        await fail_and_close(
            session,
            code=ErrorCode.BACKPRESSURE,
            message="server text queue is full (could not enqueue end-of-text)",
            drain=True,
        )
        return


# ---------------------------------------------------------------
# Default app (used by `uvicorn tts_gateway.app:app`)
# ---------------------------------------------------------------
#
# Configuration via environment variables:
#
#   VLLM_OMINI_URL   — http://hostname:8091  (where vllm-omini is running)
#                      Default: http://localhost:8091
#   VLLM_OMINI_MODEL — model id (default: cosyvoice3)
#   VLLM_OMINI_VOICE — default voice name (default: test_voice_v3)
#   GATEWAY_AUTH_TOKEN — comma-separated valid tokens
#                        Default: test-token (DEMO ONLY, replace in prod)
#
# Examples:
#
#   # Same machine as vllm-omini:
#   uvicorn tts_gateway.app:app --host 0.0.0.0 --port 8000
#
#   # Separate gateway machine, vllm-omini on 10.0.0.5:
#   VLLM_OMINI_URL=http://10.0.0.5:8091 \
#   GATEWAY_AUTH_TOKEN="prod-key-1,prod-key-2" \
#       uvicorn tts_gateway.app:app --host 0.0.0.0 --port 8000


def _build_default_app() -> FastAPI:
    backend_url = os.environ.get("VLLM_OMINI_URL")
    if backend_url:
        # Real backend configured -> wire up VllmOminiEngine.
        from .engine_vllm_omini import VllmOminiEngine
        engine: TtsBackend = VllmOminiEngine(
            base_url=backend_url,
            model=os.environ.get("VLLM_OMINI_MODEL", "cosyvoice3"),
            default_voice=os.environ.get("VLLM_OMINI_VOICE", "female"),
        )
    else:
        # Dev / demo mode -> silence-emitting MockEngine.
        engine = MockEngine()

    tokens = os.environ.get("GATEWAY_AUTH_TOKEN", "test-token")
    settings = GatewaySettings(
        valid_tokens=tuple(t.strip() for t in tokens.split(",") if t.strip()),
    )
    return create_app(engine=engine, settings=settings)


app = _build_default_app()

"""TTSSession: per-connection state + close() — the single termination path.

Design invariants this module enforces:

  1. close() is the ONLY way a session terminates. fail_and_close /
     cancel_and_close are convenience wrappers that set a terminal_frame
     and call close().

  2. close() is idempotent and serialized via close_lock. It also skips
     asyncio.current_task() when cancelling/awaiting, which is what lets
     run_tts and watchdog be safe even though they don't call close()
     themselves — the supervisor does.

  3. terminal_frame is a priority slot, NOT enqueued in out_queue. If
     out_queue is full when an error fires, we still want the client to
     get the error frame on the way out. sender_loop sends it right
     before exiting on SENTINEL.

  4. _put_outbound_nowait checks state synchronously. As soon as
     sender_loop sets state=CLOSING, no producer can enqueue anything
     new (no race with close() teardown).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from fastapi import WebSocket

from .engine import TtsBackend
from .protocol import (
    SENTINEL,
    AudioFormat,
    BytesOut,
    ErrorCode,
    JsonOut,
    Outbound,
    PCM16_24K_MONO,
    SessionState,
)

logger = logging.getLogger(__name__)


# Module-level registry. Lifespan iterates this on shutdown to fan out
# close() calls in parallel.
ACTIVE_SESSIONS: set["TTSSession"] = set()


@dataclass(eq=False)
class TTSSession:
    websocket: WebSocket
    engine: TtsBackend

    # Identity
    client_id: Optional[str] = None
    request_id: Optional[str] = None  # client-supplied, echoed in events
    engine_request_id: Optional[str] = None  # gateway-derived, used for abort

    # Lifecycle
    state: str = SessionState.CONNECTED
    close_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    close_event: asyncio.Event = field(default_factory=asyncio.Event)
    sender_failed: bool = False

    # Synthesis params
    voice: str = "default"
    prompt_audio_id: Optional[str] = None
    instructions: Optional[str] = None
    audio_format: AudioFormat = field(default_factory=lambda: PCM16_24K_MONO)

    # Queues
    text_queue: asyncio.Queue = field(
        default_factory=lambda: asyncio.Queue(maxsize=32)
    )
    out_queue: asyncio.Queue = field(
        default_factory=lambda: asyncio.Queue(maxsize=128)
    )

    # Tasks (set by the endpoint right after accept)
    sender_task: Optional[asyncio.Task] = None
    heartbeat_task: Optional[asyncio.Task] = None
    idle_watchdog_task: Optional[asyncio.Task] = None
    tts_task: Optional[asyncio.Task] = None
    supervisor_task: Optional[asyncio.Task] = None

    # Terminal frame priority slot — sent by sender_loop just before exit.
    terminal_frame: Optional[JsonOut] = None
    terminal_kind: Optional[str] = None  # "done" | "cancelled" | "error"

    # Telemetry
    connected_at: float = field(default_factory=time.monotonic)
    started_at: float = field(default_factory=time.monotonic)
    last_client_msg_at: float = field(default_factory=time.monotonic)
    total_generated_samples: int = 0
    first_audio_sent_at: Optional[float] = None
    cancel_received_at: Optional[float] = None

    # ---------------------------------------------------------------
    # Outbound queue access (called from any producer)
    # ---------------------------------------------------------------

    def _put_outbound_nowait(self, item: Outbound) -> bool:
        """Synchronous, non-blocking enqueue. Returns False if rejected.

        Triggers a close on out_queue full so the session can't deadlock
        with a slow client. Safe to call from any context — never awaits.
        """
        if self.state in (SessionState.CLOSING, SessionState.CLOSED):
            return False
        try:
            self.out_queue.put_nowait(item)
            return True
        except asyncio.QueueFull:
            # Slow client. Set terminal frame, flip state synchronously
            # (so subsequent _put_outbound_nowait calls return False
            # immediately), and schedule the actual close async.
            self.set_terminal_frame_once(
                {
                    "type": "tts.error",
                    "request_id": self.request_id,
                    "code": ErrorCode.CLIENT_SLOW,
                    "message": "client is too slow to consume audio",
                },
                kind="error",
            )
            self.state = SessionState.CLOSING
            asyncio.create_task(
                self.close(drain=False, reason="client_slow"),
                name=f"close-client-slow-{self.request_id}",
            )
            return False

    def send_json(self, data: dict) -> bool:
        return self._put_outbound_nowait(JsonOut(data))

    def send_bytes(self, data: bytes) -> bool:
        if self.first_audio_sent_at is None and data:
            self.first_audio_sent_at = time.monotonic()
        return self._put_outbound_nowait(BytesOut(data))

    def put_text_nowait(self, text: str) -> bool:
        try:
            self.text_queue.put_nowait(text)
            return True
        except asyncio.QueueFull:
            return False

    def put_text_end_nowait(self) -> bool:
        try:
            self.text_queue.put_nowait(None)
            return True
        except asyncio.QueueFull:
            return False

    # ---------------------------------------------------------------
    # Terminal frame (priority slot)
    # ---------------------------------------------------------------

    def set_terminal_frame_once(self, frame: dict, kind: str) -> None:
        """Set the priority terminal frame. First writer wins.

        We rely on single-event-loop semantics: there is no `await` between
        the `is None` check and the assignment, so two concurrent callers
        cannot both succeed.
        """
        if self.terminal_frame is None:
            self.terminal_frame = JsonOut(frame)
            self.terminal_kind = kind

    # ---------------------------------------------------------------
    # close: the single termination path
    # ---------------------------------------------------------------

    async def close(self, *, drain: bool, reason: str = "close",
                    purge_queue: bool = False) -> None:
        """Tear down the session. Idempotent. Safe to call from any task,
        including the producer tasks themselves — current_task is skipped
        when cancelling/awaiting siblings.

        purge_queue=True empties out_queue of any pending audio BEFORE
        sender drains. Use this on cancel — without it, sender keeps
        flushing buffered tail audio after the user said "stop".
        """
        async with self.close_lock:
            if self.state == SessionState.CLOSED:
                return

            self.state = SessionState.CLOSING

            # Tell the engine to stop ASAP. abort() must be idempotent
            # and silent on unknown request_id (see engine contract).
            if self.engine_request_id:
                try:
                    await self.engine.abort(self.engine_request_id)
                except Exception:
                    logger.exception(
                        "engine.abort failed request_id=%s",
                        self.engine_request_id,
                    )

            current = asyncio.current_task()

            # Stop producers first so nothing else lands in out_queue.
            producers = [
                self.tts_task,
                self.heartbeat_task,
                self.idle_watchdog_task,
                self.supervisor_task,
            ]
            for task in producers:
                if task and not task.done() and task is not current:
                    task.cancel()
            for task in producers:
                if task and task is not current:
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await task

            # Discard buffered audio (cancel path) — keep terminal frame.
            # The terminal frame lives in self.terminal_frame, NOT in
            # out_queue, so it's still delivered.
            if purge_queue:
                try:
                    while True:
                        self.out_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass

            # If sender died (e.g., client disconnected mid-send) drain
            # is meaningless — the socket can't accept more bytes anyway.
            if self.sender_failed:
                drain = False

            if self.sender_task and not self.sender_task.done() and self.sender_task is not current:
                if drain:
                    await self._stop_sender_with_drain()
                else:
                    self.sender_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await self.sender_task

            self.state = SessionState.CLOSED
            self.close_event.set()
            ACTIVE_SESSIONS.discard(self)

            with contextlib.suppress(Exception):
                await self.websocket.close()

    async def _stop_sender_with_drain(self) -> None:
        """Best-effort drain of out_queue. Timeout scales with queue depth.

        If the client is consuming normally, the sender will burn through
        the backlog and exit on SENTINEL within timeout. If the client is
        slow, we cancel after timeout — the docs say drain is best-effort
        and tail audio may be lost.
        """
        assert self.sender_task is not None

        qsize = self.out_queue.qsize()
        # 50ms per chunk + 0.5s headroom; capped at 10s so shutdown
        # doesn't stall on a stuck client.
        timeout = max(2.0, min(10.0, qsize * 0.05 + 0.5))

        try:
            await asyncio.wait_for(self.out_queue.put(SENTINEL), timeout=0.2)
        except asyncio.TimeoutError:
            # out_queue is so full we can't even append SENTINEL; just cancel.
            self.sender_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.sender_task
            return

        try:
            await asyncio.wait_for(self.sender_task, timeout=timeout)
        except asyncio.TimeoutError:
            self.sender_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.sender_task


# ---------------------------------------------------------------
# Convenience wrappers — these only set the terminal frame and call close()
# ---------------------------------------------------------------


async def fail_and_close(
    session: TTSSession,
    *,
    code: str,
    message: str,
    drain: bool,
) -> None:
    session.set_terminal_frame_once(
        {
            "type": "tts.error",
            "request_id": session.request_id,
            "code": code,
            "message": message,
        },
        kind="error",
    )
    await session.close(drain=drain, reason=code)


async def cancel_and_close(session: TTSSession, *, reason: str) -> None:
    # Cancel: cut sender immediately, even mid-send. The sender's
    # CancelledError handler still flushes the terminal tts.cancelled
    # frame, so the client gets the ack — just no more audio.
    # This is what makes the user-visible abort_latency match
    # engine.abort() latency (~1ms) instead of getting a 1+ second
    # tail of buffered or in-flight audio.
    session.set_terminal_frame_once(
        {
            "type": "tts.cancelled",
            "request_id": session.request_id,
            "reason": reason,
        },
        kind="cancelled",
    )
    await session.close(drain=False, reason=reason)

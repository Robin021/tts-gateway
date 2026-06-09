"""All the long-running tasks attached to a session.

Pipeline:

  receiver_loop (in endpoint)  ->  text_queue  ->  text_iterator  ->
      engine.stream_tts  ->  audio bytes  ->  out_queue  ->  sender_loop  ->  ws

Plus three supervision tasks: heartbeat, idle_watchdog, supervisor.

Critical rules:

  - Only sender_loop calls websocket.send_*.
  - Producers (run_tts, idle_watchdog) NEVER call session.close() on
    themselves. They set terminal_frame and exit. supervisor_loop
    notices and calls close().
  - sender_loop on exception: synchronously sets state=CLOSING and
    sender_failed=True before scheduling close(). This prevents
    producers from continuing to enqueue.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import AsyncIterator

from .engine import TtsBackend
from .protocol import SENTINEL, BytesOut, ErrorCode, JsonOut, SessionState
from .scheduler import EngineScheduler
from .session import TTSSession

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------
# sender_loop: the only place websocket.send_* runs
# ---------------------------------------------------------------


async def sender_loop(session: TTSSession) -> None:
    """The only place websocket.send_* runs.

    Two normal exits:
      - SENTINEL arrives in out_queue (drain path): we send terminal_frame
        if set, then return.
      - We get cancelled (cancel path): the except CancelledError clause
        does a best-effort send of terminal_frame so cancel_and_close can
        still deliver tts.cancelled even though it cut us off mid-send.

    On any other exception we synchronously flip state -> CLOSING (so
    producers stop enqueuing) and schedule a close() to clean up.
    """
    sent_terminal = False

    async def _send_terminal_once() -> None:
        nonlocal sent_terminal
        if sent_terminal or session.terminal_frame is None:
            return
        sent_terminal = True
        with contextlib.suppress(Exception):
            await session.websocket.send_json(session.terminal_frame.data)

    try:
        while True:
            item = await session.out_queue.get()

            if item is SENTINEL:
                await _send_terminal_once()
                return

            # Drop any buffered audio that arrived while the session was
            # being torn down. close() flips state -> CLOSING before it
            # cancels us, so this acts as a fast-purge for cancel paths
            # that left audio in flight.
            if (
                isinstance(item, BytesOut)
                and session.state in (SessionState.CLOSING, SessionState.CLOSED)
            ):
                continue

            if isinstance(item, JsonOut):
                await session.websocket.send_json(item.data)
            elif isinstance(item, BytesOut):
                await session.websocket.send_bytes(item.data)

            await asyncio.sleep(0)

    except asyncio.CancelledError:
        # Cancel path: try to deliver tts.cancelled / tts.error before exit.
        await _send_terminal_once()
        raise
    except Exception as exc:
        logger.warning(
            "sender failed request_id=%s error=%r",
            session.request_id,
            exc,
        )
        session.sender_failed = True
        session.state = SessionState.CLOSING
        asyncio.create_task(
            session.close(drain=False, reason="sender_failed"),
            name=f"close-sender-failed-{session.request_id}",
        )


# ---------------------------------------------------------------
# heartbeat_loop: keep NAT/LB happy
# ---------------------------------------------------------------


async def heartbeat_loop(session: TTSSession, interval_s: float = 20.0) -> None:
    while True:
        await asyncio.sleep(interval_s)
        if session.state in (SessionState.CLOSING, SessionState.CLOSED):
            return
        ok = session.send_json(
            {
                "type": "tts.keepalive",
                "request_id": session.request_id,
                "ts": int(time.time() * 1000),
            }
        )
        if not ok:
            return


# ---------------------------------------------------------------
# idle_watchdog_loop: enforce timeouts. Sets terminal frame and exits;
# supervisor closes the session.
# ---------------------------------------------------------------


async def idle_watchdog_loop(
    session: TTSSession,
    *,
    auth_timeout_s: float = 5.0,
    start_timeout_s: float = 10.0,
    idle_timeout_s: float = 30.0,
    max_session_s: float = 300.0,
    tick_s: float = 1.0,
) -> None:
    auth_deadline = session.connected_at + auth_timeout_s
    start_deadline: float | None = None

    while True:
        await asyncio.sleep(tick_s)
        if session.state in (SessionState.CLOSING, SessionState.CLOSED):
            return

        now = time.monotonic()

        if session.state == SessionState.CONNECTED:
            if now > auth_deadline:
                session.set_terminal_frame_once(
                    {
                        "type": "tts.error",
                        "request_id": None,
                        "code": ErrorCode.AUTH_TIMEOUT,
                        "message": "auth was not received in time",
                    },
                    kind="error",
                )
                return

        elif session.state == SessionState.AUTHENTICATED:
            if start_deadline is None:
                start_deadline = now + start_timeout_s
            if now > start_deadline:
                session.set_terminal_frame_once(
                    {
                        "type": "tts.error",
                        "request_id": None,
                        "code": ErrorCode.START_TIMEOUT,
                        "message": "tts.start was not received in time",
                    },
                    kind="error",
                )
                return

        elif session.state in (SessionState.STARTED, SessionState.ENDING):
            if now - session.started_at > max_session_s:
                session.set_terminal_frame_once(
                    {
                        "type": "tts.error",
                        "request_id": session.request_id,
                        "code": ErrorCode.SESSION_TIMEOUT,
                        "message": "max session duration exceeded",
                    },
                    kind="error",
                )
                return

            # Idle only checks STARTED — once the client said tts.end,
            # they're allowed to wait silently for audio to finish.
            if (
                session.state == SessionState.STARTED
                and now - session.last_client_msg_at > idle_timeout_s
            ):
                session.set_terminal_frame_once(
                    {
                        "type": "tts.error",
                        "request_id": session.request_id,
                        "code": ErrorCode.IDLE_TIMEOUT,
                        "message": "no client input received before timeout",
                    },
                    kind="error",
                )
                return


# ---------------------------------------------------------------
# text_iterator: bridges text_queue -> engine.stream_tts
# ---------------------------------------------------------------


async def text_iterator(session: TTSSession) -> AsyncIterator[str]:
    while True:
        item = await session.text_queue.get()
        if item is None:
            return
        yield item


# ---------------------------------------------------------------
# run_tts: drives engine.stream_tts. Sets terminal_frame on done/error.
# Does NOT call close — supervisor does.
# ---------------------------------------------------------------


async def run_tts(
    session: TTSSession,
    scheduler: EngineScheduler,
) -> None:
    assert session.engine_request_id is not None

    reserved = False
    acquired = False

    try:
        position = await scheduler.reserve()
        reserved = True

        session.send_json(
            {
                "type": "tts.queued",
                "request_id": session.request_id,
                "position": position,
            }
        )

        # Ownership transfer: from this point, wait()'s finally owns the
        # _waiting decrement. run_tts.finally must NOT cancel_reservation.
        reserved = False
        await scheduler.wait()
        acquired = True

        session.send_json(
            {
                "type": "tts.processing",
                "request_id": session.request_id,
            }
        )

        async for audio_chunk in session.engine.stream_tts(
            request_id=session.engine_request_id,
            text_iter=text_iterator(session),
            voice=session.voice,
            prompt_audio_id=session.prompt_audio_id,
            instructions=session.instructions,
        ):
            if not audio_chunk:
                continue

            session.total_generated_samples += session.audio_format.samples_in(
                audio_chunk
            )

            ok = session.send_bytes(audio_chunk)
            if not ok:
                # out_queue full or session closing. Stop pulling from
                # engine — the gateway is already tearing down.
                break

        duration_ms = int(
            session.total_generated_samples
            / session.audio_format.sample_rate
            * 1000
        )
        session.set_terminal_frame_once(
            {
                "type": "tts.done",
                "request_id": session.request_id,
                "generated_samples": session.total_generated_samples,
                "generated_duration_ms": duration_ms,
            },
            kind="done",
        )

    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.exception("tts failed request_id=%s", session.request_id)
        session.set_terminal_frame_once(
            {
                "type": "tts.error",
                "request_id": session.request_id,
                "code": ErrorCode.ENGINE_ERROR,
                "message": str(exc),
            },
            kind="error",
        )
    finally:
        if reserved:
            await scheduler.cancel_reservation()
        if acquired:
            scheduler.release()


# ---------------------------------------------------------------
# supervisor_loop: watches producers, calls close() on first one done
# ---------------------------------------------------------------


async def supervisor_loop(session: TTSSession) -> None:
    """Watch the producers that drive session lifecycle:

      - tts_task: normal completion or engine error
      - idle_watchdog_task: timeout

    heartbeat_task is intentionally NOT watched — it's purely advisory
    and only exits when the session is already closing. The receiver
    loop in the endpoint handles client-driven termination directly.
    """
    watched = [
        t
        for t in (
            session.tts_task,
            session.idle_watchdog_task,
        )
        if t is not None
    ]
    if not watched:
        return

    done, _pending = await asyncio.wait(
        watched,
        return_when=asyncio.FIRST_COMPLETED,
    )

    if session.state in (SessionState.CLOSING, SessionState.CLOSED):
        return

    # Surface unexpected exceptions as a terminal error frame.
    # asyncio.wait guarantees these tasks are done, so .exception() is safe.
    for task in done:
        try:
            exc = task.exception()
        except asyncio.CancelledError:
            exc = None
        if exc:
            session.set_terminal_frame_once(
                {
                    "type": "tts.error",
                    "request_id": session.request_id,
                    "code": ErrorCode.TASK_ERROR,
                    "message": str(exc),
                },
                kind="error",
            )

    await session.close(drain=True, reason="producer_done")

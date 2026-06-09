"""TTSSession unit tests — focus on close() invariants and the priority
terminal frame slot. We use a minimal fake websocket + fake engine because
the real WS dance happens in the integration tests.
"""

from __future__ import annotations

import asyncio
from typing import Any

from tests.conftest import run_async
from tts_gateway.engine import MockEngine
from tts_gateway.protocol import SENTINEL, BytesOut, JsonOut, SessionState
from tts_gateway.session import TTSSession, fail_and_close, cancel_and_close


class FakeWebSocket:
    """Captures send_json/send_bytes calls for assertions."""

    def __init__(self):
        self.json_sent: list[dict] = []
        self.bytes_sent: list[bytes] = []
        self.closed = False

    async def send_json(self, data):
        self.json_sent.append(data)

    async def send_bytes(self, data):
        self.bytes_sent.append(data)

    async def close(self):
        self.closed = True


def _make_session() -> tuple[TTSSession, FakeWebSocket]:
    ws = FakeWebSocket()
    engine = MockEngine()
    s = TTSSession(websocket=ws, engine=engine)  # type: ignore[arg-type]
    return s, ws


def test_close_is_idempotent():
    async def _():
        s, _ws = _make_session()
        # No tasks attached — close should still complete.
        await s.close(drain=False, reason="test")
        await s.close(drain=False, reason="test2")
        assert s.state == SessionState.CLOSED
    run_async(_)


def test_terminal_frame_first_writer_wins():
    async def _():
        s, _ws = _make_session()
        s.set_terminal_frame_once({"type": "tts.done"}, kind="done")
        s.set_terminal_frame_once({"type": "tts.error"}, kind="error")
        assert s.terminal_kind == "done"
        assert s.terminal_frame is not None
        assert s.terminal_frame.data == {"type": "tts.done"}
    run_async(_)


def test_put_outbound_rejects_when_closing():
    async def _():
        s, _ws = _make_session()
        s.state = SessionState.CLOSING
        assert s.send_json({"type": "x"}) is False
        assert s.send_bytes(b"a") is False
    run_async(_)


def test_put_outbound_full_triggers_close():
    """When out_queue is full, _put_outbound_nowait must:
      1. Set the terminal frame to CLIENT_SLOW.
      2. Flip state to CLOSING synchronously.
      3. Schedule a close task.
    Subsequent enqueues must fail immediately."""
    async def _():
        s, _ws = _make_session()
        # Fill the out_queue to capacity.
        for _ in range(s.out_queue.maxsize):
            s.out_queue.put_nowait(JsonOut({"type": "filler"}))

        # Next put triggers the slow-client path.
        ok = s.send_json({"type": "important"})
        assert ok is False
        assert s.state == SessionState.CLOSING
        assert s.terminal_kind == "error"
        assert s.terminal_frame.data["code"] == "CLIENT_SLOW"

        # Subsequent puts get rejected without scheduling another close.
        ok2 = s.send_bytes(b"more")
        assert ok2 is False

        # Drain the scheduled close task.
        await asyncio.sleep(0.05)
        # Cancel any lingering tasks for clean test exit.
        for t in asyncio.all_tasks() - {asyncio.current_task()}:
            t.cancel()
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
    run_async(_)


def test_first_audio_timestamp_recorded():
    async def _():
        s, _ws = _make_session()
        assert s.first_audio_sent_at is None
        s.send_bytes(b"\x00\x00")
        assert s.first_audio_sent_at is not None
        first_ts = s.first_audio_sent_at
        s.send_bytes(b"\x00\x00")
        # Doesn't update on subsequent sends.
        assert s.first_audio_sent_at == first_ts
    run_async(_)


def test_fail_and_close_sets_error_terminal_frame():
    async def _():
        s, _ws = _make_session()
        await fail_and_close(s, code="TEST", message="m", drain=False)
        assert s.terminal_kind == "error"
        assert s.terminal_frame.data["code"] == "TEST"
        assert s.state == SessionState.CLOSED
    run_async(_)


def test_cancel_and_close_sets_cancelled_terminal_frame():
    async def _():
        s, _ws = _make_session()
        # No tasks => drain still works (sender_task is None).
        await cancel_and_close(s, reason="user")
        assert s.terminal_kind == "cancelled"
        assert s.terminal_frame.data["reason"] == "user"
        assert s.state == SessionState.CLOSED
    run_async(_)


def test_close_skips_current_task():
    """If close() is called from inside a producer, it must NOT cancel-await
    that producer (which would deadlock self-await). It should still succeed."""
    async def _():
        s, _ws = _make_session()

        async def producer_that_calls_close():
            # Set ourselves as the tts_task.
            s.tts_task = asyncio.current_task()
            # Calling close() from inside should not deadlock.
            await s.close(drain=False, reason="self")

        await asyncio.create_task(producer_that_calls_close())
        assert s.state == SessionState.CLOSED
    run_async(_)


def test_sender_failed_disables_drain():
    """If sender_failed is set, close() must not attempt drain — there's no
    healthy socket to drain to."""
    async def _():
        s, ws = _make_session()
        s.sender_failed = True

        # Pretend we have a sender task that's already done.
        async def dead_sender():
            return

        s.sender_task = asyncio.create_task(dead_sender())
        await asyncio.sleep(0)  # let it complete

        # close should not block trying to drain.
        await asyncio.wait_for(s.close(drain=True, reason="test"), timeout=1.0)
        assert s.state == SessionState.CLOSED
    run_async(_)

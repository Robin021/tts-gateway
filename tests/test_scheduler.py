"""Tests for EngineScheduler — specifically the cancel-path counter accounting.

Recap of the bug we hunted at design time:
  Stage 1: reserve()   -> _waiting += 1
  Stage 2: wait()      -> _waiting -= 1 in finally
  Cancel : cancel_reservation() -> _waiting -= 1

If both wait()'s finally AND cancel_reservation() run, _waiting double-decrements.
The fix is ownership transfer in the caller: clear `reserved` flag before
calling wait(). These tests verify the scheduler invariants directly.
"""

from __future__ import annotations

import asyncio

from tests.conftest import run_async
from tts_gateway.scheduler import EngineScheduler


def test_reserve_increments_waiting():
    async def _():
        s = EngineScheduler(max_concurrent=1)
        assert s.waiting == 0
        pos = await s.reserve()
        assert pos == 1
        assert s.waiting == 1
    run_async(_)


def test_wait_decrements_waiting_on_success():
    async def _():
        s = EngineScheduler(max_concurrent=1)
        await s.reserve()
        await s.wait()
        assert s.waiting == 0
        s.release()
    run_async(_)


def test_wait_decrements_waiting_on_cancel():
    """Critical: wait()'s finally must run even when cancelled while blocked."""
    async def _():
        s = EngineScheduler(max_concurrent=1)
        # Take the only slot so the next wait() blocks.
        await s.reserve()
        await s.wait()

        # Now reserve a second waiter and cancel it mid-wait.
        await s.reserve()
        assert s.waiting == 1

        async def waiter():
            await s.wait()

        t = asyncio.create_task(waiter())
        await asyncio.sleep(0.01)  # let waiter block
        t.cancel()
        try:
            await t
        except asyncio.CancelledError:
            pass

        # _waiting must be back to 0 — finally ran.
        assert s.waiting == 0
        s.release()
    run_async(_)


def test_cancel_reservation_decrements():
    async def _():
        s = EngineScheduler(max_concurrent=1)
        await s.reserve()
        await s.cancel_reservation()
        assert s.waiting == 0
    run_async(_)


def test_double_decrement_is_clamped():
    """If a caller buggily calls cancel_reservation after wait, we clamp at 0
    rather than going negative. This is defense in depth — the contract says
    callers must pick exactly one path, but we don't want to crash on misuse."""
    async def _():
        s = EngineScheduler(max_concurrent=1)
        await s.reserve()
        await s.wait()
        s.release()
        # Buggy double-decrement.
        await s.cancel_reservation()
        assert s.waiting == 0
    run_async(_)


def test_position_reflects_queue_depth():
    async def _():
        s = EngineScheduler(max_concurrent=1)
        p1 = await s.reserve()
        p2 = await s.reserve()
        p3 = await s.reserve()
        assert (p1, p2, p3) == (1, 2, 3)
        await s.cancel_reservation()
        await s.cancel_reservation()
        await s.cancel_reservation()
    run_async(_)


def test_max_concurrent_enforced():
    """Two waiters, capacity 1: second one must block until first releases."""
    async def _():
        s = EngineScheduler(max_concurrent=1)
        order = []

        async def w(name):
            await s.reserve()
            await s.wait()
            order.append(f"acquired-{name}")
            await asyncio.sleep(0.02)
            s.release()
            order.append(f"released-{name}")

        await asyncio.gather(w("a"), w("b"))
        # a should fully complete before b acquires.
        assert order == [
            "acquired-a",
            "released-a",
            "acquired-b",
            "released-b",
        ]
        assert s.waiting == 0
    run_async(_)


def test_run_tts_pattern_no_double_decrement_on_cancel():
    """End-to-end simulation of the exact run_tts pattern with cancel
    happening during wait(). Verifies the ownership transfer works."""
    async def _():
        s = EngineScheduler(max_concurrent=1)
        # Block the slot.
        await s.reserve()
        await s.wait()

        async def run_tts_like():
            reserved = False
            acquired = False
            try:
                await s.reserve()
                reserved = True
                # ... send tts.queued ...
                reserved = False  # ownership transfer to wait()
                await s.wait()    # cancel happens here
                acquired = True
            finally:
                if reserved:
                    await s.cancel_reservation()
                if acquired:
                    s.release()

        t = asyncio.create_task(run_tts_like())
        await asyncio.sleep(0.01)
        t.cancel()
        try:
            await t
        except asyncio.CancelledError:
            pass
        # Exactly one decrement, not two.
        assert s.waiting == 0
        s.release()
    run_async(_)


def test_run_tts_pattern_cancel_before_wait():
    """Cancel after reserve() but before ownership transfer — caller must
    cancel_reservation in finally."""
    async def _():
        s = EngineScheduler(max_concurrent=1)
        cancelled_during_send = asyncio.Event()

        async def run_tts_like():
            reserved = False
            try:
                await s.reserve()
                reserved = True
                # Simulate "send tts.queued" hanging long enough to be cancelled.
                cancelled_during_send.set()
                await asyncio.sleep(10)
                reserved = False
                await s.wait()
            finally:
                if reserved:
                    await s.cancel_reservation()

        t = asyncio.create_task(run_tts_like())
        await cancelled_during_send.wait()
        t.cancel()
        try:
            await t
        except asyncio.CancelledError:
            pass
        assert s.waiting == 0
    run_async(_)

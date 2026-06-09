"""MockEngine sanity tests — abort idempotency, unknown id silence,
basic streaming behavior."""

from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator

from tests.conftest import run_async
from tts_gateway.engine import MockEngine


async def _text_iter(items):
    for x in items:
        yield x


def test_abort_unknown_request_id_is_silent():
    async def _():
        engine = MockEngine()
        await engine.load()
        # Must not raise.
        await engine.abort("nonexistent-id")
        await engine.abort("nonexistent-id")  # idempotent
    run_async(_)


def test_streams_silence_for_text():
    async def _():
        engine = MockEngine(
            chunks_per_sentence=3,
            per_chunk_delay_s=0.005,
            first_chunk_extra_delay_s=0.0,
        )
        await engine.load()

        chunks = []
        async for c in engine.stream_tts(
            request_id="r1",
            text_iter=_text_iter(["hello"]),
            voice="default",
            prompt_audio_id=None,
        ):
            chunks.append(c)
        assert len(chunks) == 3
        assert all(len(c) == engine.chunk_bytes for c in chunks)
    run_async(_)


def test_abort_mid_synthesis_terminates_quickly():
    """Critical test: abort latency must be roughly per_chunk_delay,
    not chunks_per_sentence * per_chunk_delay. This is the gateway's
    main user-facing latency knob."""
    async def _():
        engine = MockEngine(
            chunks_per_sentence=100,  # would take 10s without abort
            per_chunk_delay_s=0.1,
            first_chunk_extra_delay_s=0.0,
        )
        await engine.load()

        async def consume():
            count = 0
            async for _c in engine.stream_tts(
                request_id="r1",
                text_iter=_text_iter(["hello"]),
                voice="default",
                prompt_audio_id=None,
            ):
                count += 1
            return count

        t = asyncio.create_task(consume())
        await asyncio.sleep(0.15)  # let 1-2 chunks come through
        abort_start = time.monotonic()
        await engine.abort("r1")
        count = await asyncio.wait_for(t, timeout=1.0)
        abort_latency = time.monotonic() - abort_start

        assert count < 100  # didn't synthesize the full sentence
        # abort should land within ~per_chunk_delay (the wait timeout).
        assert abort_latency < 0.5, f"abort_latency={abort_latency:.3f}s"
    run_async(_)


def test_engine_error_propagates():
    async def _():
        engine = MockEngine(
            chunks_per_sentence=1,
            per_chunk_delay_s=0.001,
            fail_on_text="boom",
        )
        await engine.load()

        raised = False
        try:
            async for _c in engine.stream_tts(
                request_id="r1",
                text_iter=_text_iter(["boom"]),
                voice="default",
                prompt_audio_id=None,
            ):
                pass
        except RuntimeError:
            raised = True
        assert raised
    run_async(_)

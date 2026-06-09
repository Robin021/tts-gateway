"""Engine abstraction + Mock implementation.

Real backend (vllm-omini / Fun-CosyVoice3) goes behind the same interface.
The Mock is what we run the gateway against during stress testing — it
simulates per-chunk latency, supports abort, and tracks abort_latency so
we can sanity-check the gateway's close path before the real engine is
plugged in.

Contract for any TtsBackend implementation:

  - load() and shutdown() are called once each at lifespan boundaries.
  - stream_tts(request_id=..., text_iter=..., ...) yields raw audio bytes.
    Each yield is a complete frame in the session's audio_format.
  - abort(request_id) MUST be:
      1. idempotent (calling twice is fine)
      2. silent on unknown request_id (no KeyError)
      3. cause stream_tts to terminate ASAP — ideally mid-step, not
         only between sentences
  - The engine takes an opaque request_id string. The gateway derives
    engine_request_id from (client_id, client_request_id, gateway_seq)
    so two clients can't collide.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional

logger = logging.getLogger(__name__)


class TtsBackend(ABC):
    @abstractmethod
    async def load(self) -> None: ...

    @abstractmethod
    async def shutdown(self) -> None: ...

    @abstractmethod
    def stream_tts(
        self,
        *,
        request_id: str,
        text_iter: AsyncIterator[str],
        voice: str,
        prompt_audio_id: Optional[str],
    ) -> AsyncIterator[bytes]:
        """Async generator yielding raw audio bytes."""
        ...

    @abstractmethod
    async def abort(self, request_id: str) -> None: ...


# =========================
# MockEngine
# =========================
#
# Generates silence at a configurable rate per text chunk so the gateway
# sees realistic streaming. Abort is implemented as an asyncio.Event
# checked between every yielded chunk — close enough to vllm's
# step-boundary abort to validate gateway behavior. Real abort_latency
# on the production engine is the number that ultimately matters.


class MockEngine(TtsBackend):
    def __init__(
        self,
        *,
        sample_rate: int = 24000,
        bytes_per_sample: int = 2,
        chunk_ms: int = 40,
        chunks_per_sentence: int = 25,
        per_chunk_delay_s: float = 0.04,
        first_chunk_extra_delay_s: float = 0.05,
        fail_on_text: Optional[str] = None,
    ):
        self._loaded = False

        self.sample_rate = sample_rate
        self.bytes_per_sample = bytes_per_sample
        self.chunk_ms = chunk_ms
        self.chunks_per_sentence = chunks_per_sentence
        self.per_chunk_delay_s = per_chunk_delay_s
        self.first_chunk_extra_delay_s = first_chunk_extra_delay_s
        self.fail_on_text = fail_on_text

        # request_id -> abort Event
        self._aborts: dict[str, asyncio.Event] = {}
        # request_id -> wall clock at which abort() was called (for telemetry)
        self.abort_called_at: dict[str, float] = {}
        # request_id -> wall clock at which stream_tts actually returned
        self.abort_observed_at: dict[str, float] = {}

    @property
    def chunk_bytes(self) -> int:
        samples = int(self.sample_rate * self.chunk_ms / 1000)
        return samples * self.bytes_per_sample

    async def load(self) -> None:
        if self._loaded:
            return
        # Real engine would: load model weights, warm up, etc.
        await asyncio.sleep(0)
        self._loaded = True
        logger.info("MockEngine loaded")

    async def shutdown(self) -> None:
        # Best-effort abort of any in-flight requests.
        for rid, ev in list(self._aborts.items()):
            ev.set()
        logger.info("MockEngine shutdown")

    async def stream_tts(
        self,
        *,
        request_id: str,
        text_iter: AsyncIterator[str],
        voice: str,
        prompt_audio_id: Optional[str],
    ) -> AsyncIterator[bytes]:
        abort_ev = asyncio.Event()
        self._aborts[request_id] = abort_ev

        try:
            first = True
            async for sentence in text_iter:
                if abort_ev.is_set():
                    break

                if self.fail_on_text and self.fail_on_text in sentence:
                    raise RuntimeError(f"injected failure on text: {sentence!r}")

                if first:
                    await asyncio.sleep(self.first_chunk_extra_delay_s)
                    first = False

                for _ in range(self.chunks_per_sentence):
                    if abort_ev.is_set():
                        break
                    # Yield first, then sleep — this models a typical
                    # decoder that produces a chunk and then waits for
                    # the next one. Putting abort check before yield
                    # gives sub-chunk abort latency.
                    yield b"\x00" * self.chunk_bytes
                    try:
                        await asyncio.wait_for(
                            abort_ev.wait(),
                            timeout=self.per_chunk_delay_s,
                        )
                        # If we got here, abort fired during the wait.
                        break
                    except asyncio.TimeoutError:
                        pass

                if abort_ev.is_set():
                    break
        finally:
            if abort_ev.is_set():
                self.abort_observed_at[request_id] = time.monotonic()
            self._aborts.pop(request_id, None)

    async def abort(self, request_id: str) -> None:
        # Idempotent + silent on unknown id.
        ev = self._aborts.get(request_id)
        if ev is not None and not ev.is_set():
            self.abort_called_at[request_id] = time.monotonic()
            ev.set()

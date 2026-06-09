"""Two-stage scheduler: reserve a queue slot, then wait for a GPU slot.

The split exists so we can emit `tts.queued {position}` before blocking on
the semaphore. A merged acquire() would only return after the slot is free,
by which point queueing telemetry is meaningless.

Ownership model (this is the bug we hunted at the end of the design phase):

  Stage 1 - reserve():
      Increments _waiting and returns the snapshot position.
      Caller "owns" the reservation until it either:
        (a) hands ownership to wait() by calling wait(), OR
        (b) calls cancel_reservation() to release it.

  Stage 2 - wait():
      Blocks on the semaphore. Its finally clause unconditionally
      decrements _waiting. From the moment wait() is called, the
      caller must NOT call cancel_reservation() — even if cancelled
      during the await, wait()'s finally has already run.

The canonical run_tts pattern:

      reserved = False
      try:
          await scheduler.reserve()
          reserved = True
          await session.send_json({"type": "tts.queued", ...})
          reserved = False           # <-- ownership transfer
          await scheduler.wait()
          # ... use slot ...
      finally:
          if reserved:
              await scheduler.cancel_reservation()
          if acquired:
              scheduler.release()

This covers all three cancel paths without double-decrement.
"""

from __future__ import annotations

import asyncio


class EngineScheduler:
    def __init__(self, max_concurrent: int):
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1")
        self._sem = asyncio.Semaphore(max_concurrent)
        self._waiting = 0
        self._lock = asyncio.Lock()
        self._max_concurrent = max_concurrent

    async def reserve(self) -> int:
        """Stage 1: register a queue position. Caller owns the reservation."""
        async with self._lock:
            self._waiting += 1
            return self._waiting

    async def cancel_reservation(self) -> None:
        """Release a reservation that was made but not yet handed to wait()."""
        async with self._lock:
            self._waiting = max(0, self._waiting - 1)

    async def wait(self) -> None:
        """Stage 2: block until a slot is available. Always decrements _waiting."""
        try:
            await self._sem.acquire()
        finally:
            async with self._lock:
                self._waiting = max(0, self._waiting - 1)

    def release(self) -> None:
        """Return a slot to the pool. Mirrors a successful wait()."""
        self._sem.release()

    @property
    def waiting(self) -> int:
        return self._waiting

    @property
    def max_concurrent(self) -> int:
        return self._max_concurrent

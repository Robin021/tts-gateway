"""RoutingEngine — pick a backend per request.

We have two backends with complementary strengths:

  - vllm-omini  (:8091): fast, batched, low TTFB. But its cosyvoice3
    reimplementation does NOT support instruct2 (dialect/emotion).
  - cosyvoice-svc (:8092): the official CosyVoice library. Supports
    instruct2 reliably, but plain pytorch is slower / unbatched.

Routing rule:

    instructions present  -> cosyvoice-svc  (needs instruct2)
    instructions absent   -> vllm-omini     (fast plain cloning)

Both backends speak the same OpenAI-ish /v1/audio/speech protocol, so we
reuse VllmOminiEngine (an HTTP-streaming client) for each, just with a
different base_url.

If only one backend is configured, all requests go to it.
"""

from __future__ import annotations

import logging
from typing import AsyncIterator, Optional

from .engine import TtsBackend
from .engine_vllm_omini import VllmOminiEngine

logger = logging.getLogger(__name__)


class RoutingEngine(TtsBackend):
    def __init__(
        self,
        *,
        fast: Optional[VllmOminiEngine] = None,
        instruct: Optional[VllmOminiEngine] = None,
    ):
        """
        fast     — backend for requests WITHOUT instructions (vllm-omini)
        instruct — backend for requests WITH instructions (cosyvoice-svc)

        At least one must be provided. If only one is set, it handles
        everything.
        """
        if fast is None and instruct is None:
            raise ValueError("RoutingEngine needs at least one backend")
        self.fast = fast
        self.instruct = instruct
        # request_id -> which backend handled it (so abort routes correctly)
        self._routed: dict[str, VllmOminiEngine] = {}

    def _pick(self, instructions: Optional[str]) -> VllmOminiEngine:
        if instructions and self.instruct is not None:
            return self.instruct
        if self.fast is not None:
            return self.fast
        # fast not configured but instruct is — use instruct for everything
        assert self.instruct is not None
        return self.instruct

    async def load(self) -> None:
        if self.fast is not None:
            await self.fast.load()
        if self.instruct is not None:
            await self.instruct.load()

    async def shutdown(self) -> None:
        if self.fast is not None:
            await self.fast.shutdown()
        if self.instruct is not None:
            await self.instruct.shutdown()

    async def stream_tts(
        self,
        *,
        request_id: str,
        text_iter: AsyncIterator[str],
        voice: str,
        prompt_audio_id: Optional[str],
        instructions: Optional[str] = None,
    ) -> AsyncIterator[bytes]:
        backend = self._pick(instructions)
        self._routed[request_id] = backend
        logger.info(
            "route request_id=%s -> %s (instructions=%s)",
            request_id,
            "instruct" if backend is self.instruct else "fast",
            bool(instructions),
        )
        try:
            async for chunk in backend.stream_tts(
                request_id=request_id,
                text_iter=text_iter,
                voice=voice,
                prompt_audio_id=prompt_audio_id,
                instructions=instructions,
            ):
                yield chunk
        finally:
            self._routed.pop(request_id, None)

    async def abort(self, request_id: str) -> None:
        backend = self._routed.get(request_id)
        if backend is not None:
            await backend.abort(request_id)
        else:
            # Unknown / already finished — abort on both is harmless
            # (idempotent + silent on unknown id).
            if self.fast is not None:
                await self.fast.abort(request_id)
            if self.instruct is not None:
                await self.instruct.abort(request_id)

    async def list_voices(self) -> dict:
        """Return available voices from a configured backend.

        Prefer the fast backend because vllm-omini owns voice upload and
        persistence in the normal deployment. If only cosyvoice-svc is
        configured, or if the fast backend lacks voice listing, fall back
        to the instruct backend.
        """
        last_exc: Exception | None = None
        for backend in (self.fast, self.instruct):
            if backend is None or not hasattr(backend, "list_voices"):
                continue
            try:
                return await backend.list_voices()
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.warning("voice listing failed on %r: %r", backend, exc)
        if last_exc is not None:
            raise last_exc
        raise NotImplementedError("no configured backend supports voice listing")

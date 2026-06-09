"""VllmOminiEngine — TtsBackend implementation against vllm-omini.

Probed contract (h20 box, vllm-omini at :8091, model "cosyvoice3"):

  POST /v1/audio/speech
    body: {
      "model": "cosyvoice3",
      "input": "<full text>",
      "voice": "<uploaded voice name>",
      "task_type": "CustomVoice",
      "response_format": "pcm",
      "stream": true,
      "stream_format": "audio"
    }
    response:
      200 audio/pcm  (chunked)
      pcm16, mono, 24000Hz (configurable)
      First audio chunk arrives at ~640ms (TTFB).
      Server emits in sentence-level segments separated by ~1-2s gaps.

  abort:
      resp.aclose() on the in-flight response.
      Measured: ~0.2ms to return, bytes_after_cancel == 0 across 45 runs.
      Effectively perfect — once aclose() is called, no more audio is
      delivered to the client.

Voice lifecycle:

  POST /v1/audio/voices
    multipart: audio_sample (wav), name, consent, ref_text
    -> {"success":true,"voice":{"name":"...", "ref_text":"..."}}

  GET  /v1/audio/voices  -> list
  DELETE /v1/audio/voices/{name}

The gateway's `prompt_audio_id` field is mapped 1:1 to the vllm-omini
voice name. If you need indirection (cache embeddings, namespace per
client, etc.), wrap this engine.

Design choice: ONE HTTP request per synthesis session.
  We accumulate the entire text from `text_iter` before issuing the
  POST. vllm-omini is sentence-level and pays a fixed ~640ms first-token
  latency per request — issuing one request per gateway sentence would
  multiply that latency. Better to send the full text and let the
  server stream sentence-level audio.

  Trade-off: TTFB is bounded by "client finishes streaming text" + 640ms.
  For LLM-driven scenarios where text dribbles in over seconds, this
  hurts. The fix is to wait for `tts.end` (or first sentence boundary)
  before issuing the POST. We currently wait for `tts.end`. A future
  optimization: if the first complete sentence is available and >N
  characters, issue the POST early with that sentence, then issue more
  requests for subsequent sentences.
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterator, Optional

import httpx

from .engine import TtsBackend

logger = logging.getLogger(__name__)


DEFAULT_BASE_URL = "http://localhost:8091"
DEFAULT_MODEL = "cosyvoice3"
DEFAULT_TASK_TYPE = "CustomVoice"
DEFAULT_RESPONSE_FORMAT = "pcm"
DEFAULT_SAMPLE_RATE = 24000  # Update once we confirm by listening
DEFAULT_TIMEOUT_S = 60.0


class VllmOminiEngine(TtsBackend):
    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        default_voice: str = "female",
        task_type: str = DEFAULT_TASK_TYPE,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        request_timeout_s: float = DEFAULT_TIMEOUT_S,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.default_voice = default_voice
        self.task_type = task_type
        self.sample_rate = sample_rate
        self.request_timeout_s = request_timeout_s

        # request_id -> in-flight httpx.Response. abort() pulls from here.
        self._active: dict[str, httpx.Response] = {}
        self._client: Optional[httpx.AsyncClient] = None

    # ------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------

    async def load(self) -> None:
        if self._client is not None:
            return
        # No connection limit on httpx — the gateway's EngineScheduler
        # already bounds concurrency. timeout=None: streaming sessions
        # legitimately stay open for the duration of synthesis.
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(connect=5.0, read=None, write=10.0, pool=None),
            limits=httpx.Limits(max_connections=64, max_keepalive_connections=16),
        )
        # Sanity check the engine is reachable. We don't hard-fail on
        # error — let synthesis attempts surface the real cause.
        try:
            r = await self._client.get("/v1/models", timeout=5.0)
            if r.status_code == 200:
                logger.info("vllm-omini reachable: %s", r.json())
            else:
                logger.warning("vllm-omini /v1/models returned %d", r.status_code)
        except Exception as exc:
            logger.warning("vllm-omini health check failed: %r", exc)

    async def shutdown(self) -> None:
        # Abort any in-flight requests before closing the pool.
        for rid in list(self._active.keys()):
            await self.abort(rid)
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------

    async def stream_tts(
        self,
        *,
        request_id: str,
        text_iter: AsyncIterator[str],
        voice: str,
        prompt_audio_id: Optional[str],
    ) -> AsyncIterator[bytes]:
        if self._client is None:
            raise RuntimeError("VllmOminiEngine.load() was not called")

        # Drain text_iter into a single request body. The gateway closes
        # text_iter on tts.end / cancel.
        chunks: list[str] = []
        async for sentence in text_iter:
            if sentence:
                chunks.append(sentence)
        text = "".join(chunks).strip()
        if not text:
            return  # No text -> no audio. Caller emits tts.done with 0 samples.

        # prompt_audio_id maps 1:1 to a vllm-omini voice name.
        # voice param ("default" by default) is the fallback.
        resolved_voice = (
            prompt_audio_id
            or (voice if voice and voice != "default" else None)
            or self.default_voice
        )

        payload = {
            "model": self.model,
            "input": text,
            "voice": resolved_voice,
            "task_type": self.task_type,
            "response_format": DEFAULT_RESPONSE_FORMAT,
            "stream": True,
            "stream_format": "audio",
        }

        logger.info(
            "vllm-omini synth request_id=%s voice=%s text_len=%d",
            request_id, resolved_voice, len(text),
        )

        # Use httpx streaming. resp.aclose() in abort() will cause the
        # iter_bytes() loop to end (closes the underlying connection).
        req = self._client.build_request("POST", "/v1/audio/speech", json=payload)
        resp = await self._client.send(req, stream=True)
        self._active[request_id] = resp
        try:
            if resp.status_code != 200:
                body = await resp.aread()
                raise RuntimeError(
                    f"vllm-omini HTTP {resp.status_code}: {body[:300]!r}"
                )
            async for chunk in resp.aiter_bytes():
                if chunk:
                    yield chunk
        except (httpx.ReadError, httpx.RemoteProtocolError, httpx.StreamClosed) as exc:
            # Expected when abort() closes the response mid-stream.
            logger.debug(
                "vllm-omini stream closed request_id=%s: %r",
                request_id, exc,
            )
        finally:
            self._active.pop(request_id, None)
            # Make sure the response/connection is released even if the
            # consumer didn't drain to completion.
            try:
                await resp.aclose()
            except Exception:
                pass

    # ------------------------------------------------------------
    # Abort
    # ------------------------------------------------------------

    async def abort(self, request_id: str) -> None:
        """Idempotent. Silent on unknown request_id.

        Closes the in-flight response, which causes the underlying TCP
        connection to drop and stops vllm-omini from sending more audio
        bytes to us. The server may continue running the synthesis
        for a few hundred ms (its problem, not ours), but the client
        sees zero additional bytes.
        """
        resp = self._active.get(request_id)
        if resp is None:
            return
        try:
            await resp.aclose()
        except Exception as exc:
            # Already closed / connection dead — fine.
            logger.debug(
                "vllm-omini abort: aclose raised request_id=%s: %r",
                request_id, exc,
            )

    # ------------------------------------------------------------
    # Voice management (used by the prompt-audio HTTP endpoint, not
    # by the WebSocket synthesis path)
    # ------------------------------------------------------------

    async def upload_voice(
        self,
        *,
        name: str,
        audio_bytes: bytes,
        ref_text: str,
        consent: str = "acknowledged",
        speaker_description: Optional[str] = None,
    ) -> dict:
        if self._client is None:
            raise RuntimeError("VllmOminiEngine.load() was not called")
        files = {"audio_sample": (f"{name}.wav", audio_bytes, "audio/wav")}
        data = {"name": name, "consent": consent, "ref_text": ref_text}
        if speaker_description:
            data["speaker_description"] = speaker_description
        r = await self._client.post(
            "/v1/audio/voices",
            files=files,
            data=data,
            timeout=30.0,
        )
        if r.status_code != 200:
            raise RuntimeError(f"upload_voice HTTP {r.status_code}: {r.text[:300]}")
        return r.json()

    async def list_voices(self) -> dict:
        if self._client is None:
            raise RuntimeError("VllmOminiEngine.load() was not called")
        r = await self._client.get("/v1/audio/voices", timeout=5.0)
        r.raise_for_status()
        return r.json()

    async def delete_voice(self, name: str) -> dict:
        if self._client is None:
            raise RuntimeError("VllmOminiEngine.load() was not called")
        r = await self._client.delete(f"/v1/audio/voices/{name}", timeout=5.0)
        # 404 is fine — already deleted.
        if r.status_code not in (200, 404):
            raise RuntimeError(f"delete_voice HTTP {r.status_code}: {r.text[:300]}")
        return r.json() if r.status_code == 200 else {"success": False, "code": 404}

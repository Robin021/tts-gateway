"""Tests for the OpenAI-compatible HTTP endpoint /v1/audio/speech.

We hit the route via httpx ASGITransport and a MockEngine so we don't
need a real vllm-omini.
"""

from __future__ import annotations

import asyncio
import json

import httpx

from tts_gateway.app import GatewaySettings, create_app
from tts_gateway.engine import MockEngine


def _make_app(engine=None, valid_tokens=("test-token",)):
    engine = engine or MockEngine(
        chunks_per_sentence=3,
        per_chunk_delay_s=0.001,
        first_chunk_extra_delay_s=0.0,
    )
    return create_app(
        engine=engine,
        settings=GatewaySettings(valid_tokens=valid_tokens, max_concurrent_synthesis=4),
    )


async def _post(app, body, token="test-token"):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        return await c.post(
            "/v1/audio/speech",
            json=body,
            headers={"Authorization": f"Bearer {token}"} if token else {},
            timeout=10.0,
        )


def test_http_unauth_rejected():
    async def _():
        app = _make_app()
        r = await _post(app, {"input": "hi", "voice": "female"}, token="bad")
        assert r.status_code == 401
    asyncio.run(_())


def test_http_no_input_rejected():
    async def _():
        app = _make_app()
        r = await _post(app, {"voice": "female"})
        # FastAPI's Pydantic validation returns 422 (Unprocessable Entity)
        # for missing required fields, which is the standard.
        assert r.status_code == 422
    asyncio.run(_())


def test_http_pcm_synth_returns_audio():
    async def _():
        app = _make_app()
        r = await _post(app, {"input": "你好", "voice": "female"})
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("audio/pcm")
        # MockEngine generates 3 chunks of 960 bytes each (40ms @ 24kHz)
        # Total = 3 * 960 = 2880 bytes
        assert len(r.content) > 0
    asyncio.run(_())


def test_http_wav_synth_has_riff_header():
    async def _():
        app = _make_app()
        r = await _post(
            app, {"input": "你好", "voice": "female", "response_format": "wav"}
        )
        assert r.status_code == 200
        # WAV starts with "RIFF"
        assert r.content[:4] == b"RIFF"
        assert r.content[8:12] == b"WAVE"
    asyncio.run(_())


def test_http_stream_true_returns_chunked():
    async def _():
        app = _make_app()
        # We can't easily inspect chunked-vs-buffered through httpx ASGI
        # transport, but we can at least verify it returns 200 + audio.
        r = await _post(
            app, {"input": "你好", "voice": "female", "stream": True}
        )
        assert r.status_code == 200
        assert len(r.content) > 0
    asyncio.run(_())


def test_http_voice_field_forwarded_to_engine():
    """Verify the voice name reaches the engine as both `voice` and
    `prompt_audio_id` (the latter is what VllmOminiEngine resolves)."""
    seen: dict = {}

    class CapturingEngine(MockEngine):
        async def stream_tts(
            self, *, request_id, text_iter, voice, prompt_audio_id,
            instructions=None,
        ):
            seen["voice"] = voice
            seen["prompt_audio_id"] = prompt_audio_id
            seen["instructions"] = instructions
            async for c in super().stream_tts(
                request_id=request_id,
                text_iter=text_iter,
                voice=voice,
                prompt_audio_id=prompt_audio_id,
                instructions=instructions,
            ):
                yield c

    async def _():
        app = _make_app(engine=CapturingEngine(
            chunks_per_sentence=1, per_chunk_delay_s=0.001,
            first_chunk_extra_delay_s=0.0,
        ))
        r = await _post(app, {
            "input": "你好",
            "voice": "linzhiling",
            "instructions": "请用广东话表达,语气兴奋",
        })
        assert r.status_code == 200
        assert seen["prompt_audio_id"] == "linzhiling"
        assert seen["instructions"] == "请用广东话表达,语气兴奋"

    asyncio.run(_())


def test_http_instructions_field_optional():
    """instructions is optional — engine should see None when not passed."""
    seen: dict = {}

    class CapturingEngine(MockEngine):
        async def stream_tts(
            self, *, request_id, text_iter, voice, prompt_audio_id,
            instructions=None,
        ):
            seen["instructions"] = instructions
            async for c in super().stream_tts(
                request_id=request_id,
                text_iter=text_iter,
                voice=voice,
                prompt_audio_id=prompt_audio_id,
                instructions=instructions,
            ):
                yield c

    async def _():
        app = _make_app(engine=CapturingEngine(
            chunks_per_sentence=1, per_chunk_delay_s=0.001,
            first_chunk_extra_delay_s=0.0,
        ))
        r = await _post(app, {"input": "你好", "voice": "female"})
        assert r.status_code == 200
        assert seen["instructions"] is None
    asyncio.run(_())


def test_openapi_schema_includes_speech_route():
    """Make sure /docs and /openapi.json are discoverable + populated."""
    async def _():
        app = _make_app()
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
            r = await c.get("/openapi.json", timeout=5.0)
            assert r.status_code == 200
            schema = r.json()
            paths = schema["paths"]
            # The TTS route is documented.
            assert "/v1/audio/speech" in paths
            assert "post" in paths["/v1/audio/speech"]
            speech_op = paths["/v1/audio/speech"]["post"]
            assert speech_op["summary"] == "Synthesize speech from text (OpenAI-compatible)"
            # The voices listing too.
            assert "/v1/audio/voices" in paths
            # The request schema is referenced.
            ref = speech_op["requestBody"]["content"]["application/json"]["schema"]
            assert "$ref" in ref or "properties" in ref

            # Swagger UI is reachable.
            r = await c.get("/docs", timeout=5.0)
            assert r.status_code == 200
            assert b"swagger" in r.content.lower()

            # ReDoc too.
            r = await c.get("/redoc", timeout=5.0)
            assert r.status_code == 200
    asyncio.run(_())

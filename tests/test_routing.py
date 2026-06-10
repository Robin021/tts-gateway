"""Tests for RoutingEngine — picks backend by presence of instructions."""

from __future__ import annotations

import asyncio

from tests.conftest import run_async  # noqa: F401
from tts_gateway.engine import MockEngine
from tts_gateway.engine_routing import RoutingEngine


class TaggedMock(MockEngine):
    """MockEngine that records which instance handled a request."""

    def __init__(self, tag, **kw):
        super().__init__(**kw)
        self.tag = tag
        self.handled: list[str] = []

    async def stream_tts(self, *, request_id, text_iter, voice,
                         prompt_audio_id, instructions=None):
        self.handled.append(request_id)
        async for c in super().stream_tts(
            request_id=request_id, text_iter=text_iter, voice=voice,
            prompt_audio_id=prompt_audio_id, instructions=instructions,
        ):
            yield c


async def _text():
    yield "你好"


def _mk(**kw):
    return TaggedMock(chunks_per_sentence=1, per_chunk_delay_s=0.001,
                      first_chunk_extra_delay_s=0.0, **kw)


def test_routes_instructions_to_instruct_backend():
    async def _():
        fast = _mk(tag="fast")
        instruct = _mk(tag="instruct")
        eng = RoutingEngine(fast=fast, instruct=instruct)

        # With instructions -> instruct backend
        async for _c in eng.stream_tts(
            request_id="r1", text_iter=_text(), voice="female",
            prompt_audio_id=None, instructions="请用广东话",
        ):
            pass
        assert instruct.handled == ["r1"]
        assert fast.handled == []
    run_async(_)


def test_routes_no_instructions_to_fast_backend():
    async def _():
        fast = _mk(tag="fast")
        instruct = _mk(tag="instruct")
        eng = RoutingEngine(fast=fast, instruct=instruct)

        async for _c in eng.stream_tts(
            request_id="r2", text_iter=_text(), voice="female",
            prompt_audio_id=None, instructions=None,
        ):
            pass
        assert fast.handled == ["r2"]
        assert instruct.handled == []
    run_async(_)


def test_single_backend_handles_all():
    async def _():
        instruct = _mk(tag="instruct")
        eng = RoutingEngine(instruct=instruct)  # no fast backend

        # Even without instructions, falls through to the only backend
        async for _c in eng.stream_tts(
            request_id="r3", text_iter=_text(), voice="female",
            prompt_audio_id=None, instructions=None,
        ):
            pass
        assert instruct.handled == ["r3"]
    run_async(_)


def test_requires_at_least_one_backend():
    try:
        RoutingEngine()
        assert False, "should have raised"
    except ValueError:
        pass


def test_list_voices_uses_fast_backend_first():
    async def _():
        class VoiceBackend(TaggedMock):
            async def list_voices(self):
                return {"voices": [self.tag], "uploaded_voices": []}

        fast = VoiceBackend(tag="fast")
        instruct = VoiceBackend(tag="instruct")
        eng = RoutingEngine(fast=fast, instruct=instruct)

        assert await eng.list_voices() == {
            "voices": ["fast"],
            "uploaded_voices": [],
        }
    run_async(_)

"""Protocol types: outbound message wrappers, audio format, session state, error codes.

The wire protocol is documented in PROTOCOL.md. This module only defines the
in-process types we use to enforce the protocol contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Union


# =========================
# Outbound message wrappers
# =========================
#
# All websocket.send_* calls happen in exactly one place: sender_loop.
# Producers wrap their payloads in JsonOut/BytesOut and put them in
# session.out_queue. The SENTINEL signals "drain done, you may exit".


@dataclass
class JsonOut:
    data: dict


@dataclass
class BytesOut:
    data: bytes


class _Sentinel:
    """Marker class so SENTINEL has a stable identity for `is` checks."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "SENTINEL"


SENTINEL = _Sentinel()

Outbound = Union[JsonOut, BytesOut, _Sentinel]


# =========================
# Session state
# =========================
#
# Linear progression: CONNECTED -> AUTHENTICATED -> INIT -> STARTED -> ENDING -> CLOSING -> CLOSED
# Any error/cancel jumps directly to CLOSING.
#
# Critical invariant: once state is CLOSING, _put_outbound_nowait must
# return False (no more enqueue). sender_loop checks this synchronously
# in its except path.


class SessionState:
    CONNECTED = "CONNECTED"  # WS accepted, awaiting auth frame
    AUTHENTICATED = "AUTHENTICATED"  # auth ok, awaiting tts.start
    INIT = "INIT"  # alias for AUTHENTICATED used in some checks
    STARTED = "STARTED"  # tts.start received, run_tts spawned
    ENDING = "ENDING"  # tts.end received, draining text_queue
    CLOSING = "CLOSING"  # close() in progress, no new outbound
    CLOSED = "CLOSED"  # all tasks done, ws closed


# =========================
# Audio format
# =========================
#
# Hard-coded to pcm16 mono for v1. When we add opus/float32 we extend this
# class instead of sprinkling magic numbers across run_tts.


@dataclass(frozen=True)
class AudioFormat:
    name: Literal["pcm16"]
    sample_rate: int
    channels: int
    bytes_per_sample: int

    def samples_in(self, audio_bytes: bytes) -> int:
        return len(audio_bytes) // self.bytes_per_sample // self.channels


PCM16_24K_MONO = AudioFormat(
    name="pcm16",
    sample_rate=24000,
    channels=1,
    bytes_per_sample=2,
)


# =========================
# Error codes
# =========================
#
# Stable strings clients can switch on. Keep them flat and add new codes
# rather than overloading existing ones.


class ErrorCode:
    BAD_JSON = "BAD_JSON"
    BAD_REQUEST = "BAD_REQUEST"
    BAD_STATE = "BAD_STATE"
    UNKNOWN_EVENT = "UNKNOWN_EVENT"
    UNEXPECTED_BINARY = "UNEXPECTED_BINARY"

    UNAUTHORIZED = "UNAUTHORIZED"
    AUTH_TIMEOUT = "AUTH_TIMEOUT"
    START_TIMEOUT = "START_TIMEOUT"
    IDLE_TIMEOUT = "IDLE_TIMEOUT"
    SESSION_TIMEOUT = "SESSION_TIMEOUT"

    MISSING_REQUEST_ID = "MISSING_REQUEST_ID"
    BACKPRESSURE = "BACKPRESSURE"
    CLIENT_SLOW = "CLIENT_SLOW"
    ENGINE_ERROR = "ENGINE_ERROR"
    TASK_ERROR = "TASK_ERROR"
    SERVER_ERROR = "SERVER_ERROR"
    SERVER_SHUTDOWN = "SERVER_SHUTDOWN"

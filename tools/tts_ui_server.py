#!/usr/bin/env python3
"""Local browser UI for exercising the TTS gateway.

The browser talks to this local helper for HTTP requests so CORS never
gets in the way. WebSocket tests connect from the browser directly to
the gateway, which lets the Cancel button send `tts.cancel` in real time.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
from pathlib import Path
from typing import Optional

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parent
HTML_PATH = ROOT / "tts_ui.html"
DEFAULT_GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://221.194.152.20:8000")
DEFAULT_GATEWAY_TOKEN = os.environ.get("GATEWAY_TOKEN", "")
SAMPLE_RATE = 24000


app = FastAPI(title="Local TTS Test Console")


class GatewaySettings(BaseModel):
    gateway_url: str = Field(default=DEFAULT_GATEWAY_URL)
    token: str = Field(default=DEFAULT_GATEWAY_TOKEN)


class SpeechBody(GatewaySettings):
    input: str
    voice: Optional[str] = None
    instructions: Optional[str] = None
    response_format: str = "wav"
    stream: bool = False
    model: str = "cosyvoice3"
    request_id: Optional[str] = None


def _wav_header(payload_len: int, sample_rate: int = SAMPLE_RATE) -> bytes:
    return (
        b"RIFF"
        + struct.pack("<I", 36 + payload_len)
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
        + b"data"
        + struct.pack("<I", payload_len)
    )


def _upstream_url(base: str, path: str) -> str:
    return base.rstrip("/") + path


def _auth_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"} if token else {}


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return HTML_PATH.read_text(encoding="utf-8")


@app.get("/api/defaults")
async def defaults() -> dict:
    return {
        "gateway_url": DEFAULT_GATEWAY_URL,
        "token": DEFAULT_GATEWAY_TOKEN,
        "sample_rate": SAMPLE_RATE,
    }


@app.post("/api/voices")
async def voices(settings: GatewaySettings) -> dict:
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                _upstream_url(settings.gateway_url, "/v1/audio/voices"),
                headers=_auth_headers(settings.token),
            )
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text[:500])
        return resp.json()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/api/speech")
async def speech(body: SpeechBody) -> Response:
    payload = {
        "model": body.model,
        "input": body.input,
        "response_format": body.response_format,
        "stream": body.stream,
    }
    if body.voice:
        payload["voice"] = body.voice
    if body.instructions:
        payload["instructions"] = body.instructions
    if body.request_id:
        payload["request_id"] = body.request_id

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(90.0, read=None)) as client:
            async with client.stream(
                "POST",
                _upstream_url(body.gateway_url, "/v1/audio/speech"),
                headers={
                    **_auth_headers(body.token),
                    "Content-Type": "application/json",
                },
                json=payload,
            ) as resp:
                chunks = [chunk async for chunk in resp.aiter_bytes() if chunk]
                data = b"".join(chunks)
                if resp.status_code != 200:
                    raise HTTPException(
                        status_code=resp.status_code,
                        detail=data.decode("utf-8", errors="replace")[:500],
                    )
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    media_type = resp.headers.get("content-type", "")
    if body.response_format == "pcm":
        data = _wav_header(len(data)) + data
        media_type = "audio/wav"

    return Response(
        content=data,
        media_type=media_type or "audio/wav",
        headers={
            "X-Upstream-Bytes": str(len(data)),
            "X-Upstream-Format": body.response_format,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local TTS test UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("TTS_UI_PORT", "8787")))
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

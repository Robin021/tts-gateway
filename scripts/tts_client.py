#!/usr/bin/env python3
"""tts-gateway CLI client — minimal, single-file, no extra deps.

Usage examples
--------------

    # Basic HTTP synth, play it
    ./tts_client.py "你好,世界"

    # Different voice + dialect
    ./tts_client.py -v male -i "请用东北话" "今天天气真好啊老铁"

    # Use WebSocket (true streaming, mid-flight cancel possible)
    ./tts_client.py --ws -v linzhiling "希望你以后能够做的比我还好哟"

    # Save audio to a file, don't play
    ./tts_client.py "测试一下" --out /tmp/hi.wav --no-play

    # List available voices
    ./tts_client.py --list-voices

    # Override server / token (or set env GATEWAY_URL / GATEWAY_TOKEN)
    ./tts_client.py --url http://1.2.3.4:8000 --token my-secret "测试"

Configuration
-------------

Two env vars (matches the deploy script conventions):

    GATEWAY_URL    e.g. http://221.194.152.20:8000
    GATEWAY_TOKEN  the Bearer token

You can put these in ~/.tts-gateway.env and source it.

Dependencies
------------

HTTP mode: only the stdlib (`urllib`).
WebSocket mode: `pip install websockets`.

Audio playback
--------------

Mac: uses `afplay` (built-in).
Linux: tries `aplay` then `paplay` then `play` (sox).
Windows: falls back to `os.startfile`.
Or just use --no-play and play the saved file yourself.
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path
from typing import Optional


DEFAULT_URL = os.environ.get("GATEWAY_URL", "http://localhost:8000")
DEFAULT_TOKEN = os.environ.get("GATEWAY_TOKEN", "test-token")


# -------------------- HTTP --------------------


def http_synth(
    *,
    url: str,
    token: str,
    text: str,
    voice: Optional[str],
    instructions: Optional[str],
    response_format: str = "wav",
    stream: bool = False,
    timeout: float = 60.0,
) -> bytes:
    body: dict = {"input": text, "response_format": response_format, "stream": stream}
    if voice:
        body["voice"] = voice
    if instructions:
        body["instructions"] = instructions
    data = json.dumps(body).encode("utf-8")

    req = urllib.request.Request(
        url.rstrip("/") + "/v1/audio/speech",
        method="POST",
        data=data,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if not stream:
                audio = resp.read()
            else:
                buf = io.BytesIO()
                first_byte_at: Optional[float] = None
                while True:
                    chunk = resp.read(4096)
                    if not chunk:
                        break
                    if first_byte_at is None:
                        first_byte_at = time.monotonic()
                        print(
                            f"  [http] first audio chunk at "
                            f"{(first_byte_at - t0) * 1000:.0f}ms",
                            file=sys.stderr,
                        )
                    buf.write(chunk)
                audio = buf.getvalue()
    except urllib.error.HTTPError as e:
        try:
            err = e.read().decode("utf-8", errors="replace")
        except Exception:
            err = ""
        die(f"HTTP {e.code} from {url}: {err[:400]}")
    except urllib.error.URLError as e:
        die(f"network error reaching {url}: {e}")

    print(
        f"  [http] {len(audio)} bytes in {(time.monotonic()-t0)*1000:.0f}ms",
        file=sys.stderr,
    )
    return audio


def http_list_voices(url: str, token: str) -> list[str]:
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/audio/voices",
        headers={"Authorization": f"Bearer {token}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        die(f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}")
    except urllib.error.URLError as e:
        die(f"network error: {e}")
    return sorted(data.get("voices", []))


# -------------------- WebSocket --------------------


async def ws_synth(
    *,
    url: str,
    token: str,
    text: str,
    voice: Optional[str],
    instructions: Optional[str],
    timeout: float = 60.0,
) -> tuple[bytes, dict]:
    try:
        import websockets
    except ImportError:
        die(
            "WebSocket mode needs the `websockets` package.\n"
            "  pip install websockets"
        )

    ws_url = url.rstrip("/")
    if ws_url.startswith("https://"):
        ws_url = "wss://" + ws_url[len("https://"):]
    elif ws_url.startswith("http://"):
        ws_url = "ws://" + ws_url[len("http://"):]
    ws_url += "/ws/tts"

    t0 = time.monotonic()
    audio = io.BytesIO()
    meta: dict = {
        "ttfb_ms": None,
        "sample_rate": 24000,
        "channels": 1,
        "generated_duration_ms": None,
        "events": [],
    }

    async with websockets.connect(ws_url, subprotocols=["tts.v1"]) as ws:
        await ws.send(json.dumps({"type": "auth", "token": token, "client_id": "cli"}))
        ack = json.loads(await asyncio.wait_for(ws.recv(), timeout=5))
        if ack.get("type") != "auth.ok":
            die(f"auth failed: {ack}")

        start_msg: dict = {"type": "tts.start", "request_id": f"cli-{int(time.time())}"}
        if voice:
            start_msg["voice"] = voice
        if instructions:
            start_msg["instructions"] = instructions
        await ws.send(json.dumps(start_msg))

        await ws.send(json.dumps({"type": "tts.text", "text": text}))
        await ws.send(json.dumps({"type": "tts.end"}))
        text_sent_at = time.monotonic()

        first_audio_at: Optional[float] = None
        while True:
            try:
                evt = await asyncio.wait_for(ws.recv(), timeout=timeout)
            except asyncio.TimeoutError:
                die("ws receive timeout")
            except websockets.ConnectionClosed:
                break

            if isinstance(evt, (bytes, bytearray)):
                if first_audio_at is None:
                    first_audio_at = time.monotonic()
                    meta["ttfb_ms"] = (first_audio_at - text_sent_at) * 1000
                    print(f"  [ws] TTFB {meta['ttfb_ms']:.0f}ms", file=sys.stderr)
                audio.write(evt)
                continue

            data = json.loads(evt)
            kind = data.get("type")
            meta["events"].append(kind)
            if kind == "tts.started":
                meta["sample_rate"] = int(data.get("sample_rate", 24000))
                meta["channels"] = int(data.get("channels", 1))
            elif kind == "tts.done":
                meta["generated_duration_ms"] = data.get("generated_duration_ms")
                break
            elif kind == "tts.error":
                die(f"server error: {data}")
            elif kind == "tts.cancelled":
                break

    print(
        f"  [ws] {audio.tell()} bytes total, "
        f"{(time.monotonic()-t0)*1000:.0f}ms wall",
        file=sys.stderr,
    )
    return audio.getvalue(), meta


# -------------------- Audio helpers --------------------


def pcm_to_wav(pcm: bytes, sample_rate: int, channels: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


def play(path: str) -> None:
    system = platform.system()
    if system == "Darwin":
        candidates = ["afplay"]
    elif system == "Linux":
        candidates = ["aplay", "paplay", "play", "ffplay"]
    elif system == "Windows":
        candidates = []
    else:
        candidates = ["play", "ffplay"]

    for cmd in candidates:
        if shutil.which(cmd):
            try:
                if cmd == "ffplay":
                    subprocess.run(
                        [cmd, "-autoexit", "-nodisp", "-loglevel", "error", path],
                        check=False,
                    )
                else:
                    subprocess.run([cmd, path], check=False)
                return
            except Exception as e:
                print(f"  [play] {cmd} failed: {e}", file=sys.stderr)

    if system == "Windows":
        try:
            os.startfile(path)  # type: ignore[attr-defined]
            return
        except Exception:
            pass

    print(f"  [play] no audio player found. File saved: {path}", file=sys.stderr)


def die(msg: str, code: int = 1) -> None:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(code)


# -------------------- CLI --------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Local CLI for tts-gateway. HTTP or WebSocket.",
        epilog="Env vars: GATEWAY_URL, GATEWAY_TOKEN.",
    )
    p.add_argument("text", nargs="?", help="text to synthesize")
    p.add_argument("--url", default=DEFAULT_URL, help=f"gateway URL (default {DEFAULT_URL})")
    p.add_argument("--token", default=DEFAULT_TOKEN, help="Bearer token")
    p.add_argument("-v", "--voice", default=None, help="voice name (female/male/linzhiling/...)")
    p.add_argument(
        "-i", "--instructions", default=None,
        help='style/dialect/emotion, e.g. "请用广东话表达"',
    )
    p.add_argument(
        "--ws", action="store_true",
        help="use WebSocket (true streaming + cancel-capable). Default is HTTP.",
    )
    p.add_argument(
        "--stream-http", action="store_true",
        help="HTTP mode but with stream=true (chunked transfer)",
    )
    p.add_argument(
        "--out", default=None,
        help='save to this file (default: tempfile). "-" = stdout.',
    )
    p.add_argument("--no-play", action="store_true", help="don't autoplay")
    p.add_argument("--list-voices", action="store_true", help="list available voices and exit")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_voices:
        voices = http_list_voices(args.url, args.token)
        for v in voices:
            print(v)
        return

    if not args.text:
        die("missing text. e.g. tts_client.py '你好,世界'")

    if args.ws:
        pcm, meta = asyncio.run(ws_synth(
            url=args.url, token=args.token, text=args.text,
            voice=args.voice, instructions=args.instructions,
        ))
        audio = pcm_to_wav(pcm, meta["sample_rate"], meta["channels"])
    else:
        audio = http_synth(
            url=args.url, token=args.token, text=args.text,
            voice=args.voice, instructions=args.instructions,
            response_format="wav",
            stream=args.stream_http,
        )

    if args.out == "-":
        sys.stdout.buffer.write(audio)
        return

    if args.out:
        path = args.out
        Path(path).write_bytes(audio)
        print(f"saved: {path}", file=sys.stderr)
    else:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.write(audio)
        tmp.close()
        path = tmp.name
        print(f"saved: {path}", file=sys.stderr)

    if not args.no_play:
        play(path)


if __name__ == "__main__":
    main()

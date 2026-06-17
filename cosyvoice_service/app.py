#!/usr/bin/env python3
"""cosyvoice-svc — thin FastAPI wrapper around the official CosyVoice library.

WHY THIS EXISTS
---------------
vllm-omini's reimplementation of CosyVoice3 does NOT support instruct2
(dialect/emotion/speed control) reliably — confirmed by testing: the
`instructions` field is ignored, and the ref_text-injection workaround
garbles output ~2/3 of the time.

The OFFICIAL CosyVoice library DOES support it via
`inference_instruct2()`. This service runs that library directly and
exposes an HTTP API compatible with our tts-gateway.

ARCHITECTURE
------------
    tts-gateway  --HTTP-->  cosyvoice-svc (this)  --in-process-->  CosyVoice AutoModel
                                                                     |- inference_zero_shot   (clone)
                                                                     |- inference_instruct2   (clone + style)

API
---
  POST /v1/audio/speech
    body: {input, voice, instructions?, response_format?, stream?}
    - voice resolves to (prompt_wav_path, prompt_text) via voices.json
    - instructions present -> inference_instruct2(input, instruct_text, prompt_wav)
    - instructions absent   -> inference_zero_shot(input, prompt_text, prompt_wav)
    returns raw pcm16 (or wav) bytes, optionally chunked when stream=true

  GET /v1/audio/voices
    -> {"voices": [...names...]}

  GET /health -> {"status": "ok", "model_loaded": bool}

VOICE REGISTRY (voices.json)
----------------------------
    {
      "female":  {"wav": "voices/female.wav",  "prompt_text": "今天上海晴空万里，蓝天白云，体感温度很舒服。..."},
      "male":    {"wav": "voices/male.wav",    "prompt_text": "..."},
      ...
    }
  prompt_text is the *actual transcript* of the wav (used by zero_shot).
  For instruct2 we don't need prompt_text — only the wav + the instruction.

CONFIG (env)
------------
  COSYVOICE_MODEL_DIR   default /models/Fun-CosyVoice3-0.5B
  COSYVOICE_VOICES      default ./voices.json
  COSYVOICE_LOAD_VLLM   default 0   (set 1 to enable vllm LLM accel — needs extra setup)
  COSYVOICE_LOAD_TRT    default 0
  COSYVOICE_FP16        default 0
  PORT                  default 8092

IMPORTANT — INSTRUCT PROMPT FORMAT
----------------------------------
CosyVoice3's instruct2 expects the instruction wrapped a specific way.
Per the official README example:

    cosyvoice.inference_instruct2(
        '好少咯，一般系放嗰啲国庆啊...',
        'You are a helpful assistant. 请用广东话表达。<|endofprompt|>',
        prompt_speech_16k, stream=False)

We replicate that in `_format_instruct()`. If the colleague's working
script uses a different wrapper, change that ONE function.
"""

from __future__ import annotations

import io
import json
import os
import sys
import time
import wave
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# CosyVoice library imports are deferred to load time so this file can at
# least be imported / linted without the heavy deps present.


# ---------------------------------------------------------------
# Config
# ---------------------------------------------------------------

MODEL_DIR = os.environ.get("COSYVOICE_MODEL_DIR", "/models/Fun-CosyVoice3-0.5B")
VOICES_PATH = os.environ.get("COSYVOICE_VOICES", str(Path(__file__).parent / "voices.json"))
LOAD_VLLM = os.environ.get("COSYVOICE_LOAD_VLLM", "0") == "1"
LOAD_TRT = os.environ.get("COSYVOICE_LOAD_TRT", "0") == "1"
FP16 = os.environ.get("COSYVOICE_FP16", "0") == "1"
PORT = int(os.environ.get("PORT", "8092"))
AUTH_TOKEN = os.environ.get("COSYVOICE_AUTH_TOKEN")  # optional; gateway is the real auth layer


# ---------------------------------------------------------------
# Global model state (loaded once at startup)
# ---------------------------------------------------------------


class _State:
    model = None              # cosyvoice AutoModel
    sample_rate = 24000       # set from model after load
    voices: dict = {}         # name -> {"wav": path, "prompt_text": str}


STATE = _State()


def _format_instruct(instructions: str) -> str:
    """Wrap a raw instruction into CosyVoice3 instruct2 prompt format.

    Official README uses:  'You are a helpful assistant. <instr>。<|endofprompt|>'

    >>> _format_instruct('请用广东话表达')
    'You are a helpful assistant. 请用广东话表达。<|endofprompt|>'

    If the colleague's working script uses a different format, change here.
    """
    instr = instructions.strip()
    if "<|endofprompt|>" in instr:
        return instr  # caller already formatted it
    if not instr.endswith(("。", ".", "!", "！", "?", "?")):
        instr = instr + "。"
    return f"You are a helpful assistant. {instr}<|endofprompt|>"


def _load_model():
    """Load CosyVoice AutoModel + voices. Called once at startup."""
    # Heavy imports here.
    sys.path.append("/opt/CosyVoice/third_party/Matcha-TTS")

    # When using the vllm-accelerated LLM stage, the official example
    # registers the custom model class before constructing AutoModel.
    if LOAD_VLLM:
        try:
            from vllm import ModelRegistry
            from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
            ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)
            print("[cosyvoice-svc] registered CosyVoice2ForCausalLM for vllm", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[cosyvoice-svc] WARN vllm registration failed: {e!r}", flush=True)

    from cosyvoice.cli.cosyvoice import AutoModel

    print(f"[cosyvoice-svc] loading model from {MODEL_DIR} "
          f"(vllm={LOAD_VLLM}, trt={LOAD_TRT}, fp16={FP16})", flush=True)
    t0 = time.monotonic()
    # CosyVoice3.__init__ does NOT accept load_jit (that's CosyVoice2 only).
    # The official cosyvoice3 example uses: model_dir, load_trt, load_vllm, fp16.
    model = AutoModel(
        model_dir=MODEL_DIR,
        load_trt=LOAD_TRT,
        load_vllm=LOAD_VLLM,
        fp16=FP16,
    )
    STATE.model = model
    STATE.sample_rate = int(getattr(model, "sample_rate", 24000))
    print(f"[cosyvoice-svc] model loaded in {time.monotonic()-t0:.1f}s, "
          f"sample_rate={STATE.sample_rate}", flush=True)

    # Load voice registry. The current official CosyVoice API expects
    # prompt_wav as a file path and loads/resamples it inside the frontend.
    with open(VOICES_PATH, "r", encoding="utf-8") as f:
        registry = json.load(f)
    voices = {}
    for name, cfg in registry.items():
        wav_path = cfg["wav"]
        if not os.path.isabs(wav_path):
            wav_path = str(Path(VOICES_PATH).parent / wav_path)
        if not os.path.exists(wav_path):
            print(f"[cosyvoice-svc] WARN voice '{name}': wav not found at {wav_path}", flush=True)
            continue
        voices[name] = {
            "wav": wav_path,
            "prompt_text": cfg.get("prompt_text", ""),
        }
        print(f"[cosyvoice-svc] voice '{name}' ready ({wav_path})", flush=True)
    STATE.voices = voices
    print(f"[cosyvoice-svc] {len(voices)} voices loaded: {sorted(voices)}", flush=True)


# ---------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------


def _tensor_to_pcm16(tts_speech) -> bytes:
    """CosyVoice yields a float32 torch tensor in [-1, 1]; convert to pcm16 bytes."""
    import torch
    audio = tts_speech
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().reshape(-1)
        audio = torch.clamp(audio, -1.0, 1.0)
        pcm = (audio * 32767.0).to(torch.int16).numpy().tobytes()
        return pcm
    raise TypeError(f"unexpected tts_speech type: {type(audio)}")


def _synthesize(text: str, voice: str, instructions: Optional[str], stream: bool):
    """Generator yielding pcm16 byte chunks."""
    if STATE.model is None:
        raise RuntimeError("model not loaded")
    if voice not in STATE.voices:
        raise KeyError(f"unknown voice '{voice}'. available: {sorted(STATE.voices)}")

    v = STATE.voices[voice]
    prompt_wav = v["wav"]

    if instructions:
        instruct_text = _format_instruct(instructions)
        gen = STATE.model.inference_instruct2(
            text, instruct_text, prompt_wav, stream=stream,
        )
    else:
        # CosyVoice3 zero_shot wants the prompt_text prefixed with the
        # assistant system prompt, per the official README:
        #   'You are a helpful assistant.<|endofprompt|>{transcript}'
        prompt_text = v["prompt_text"]
        if "<|endofprompt|>" not in prompt_text:
            prompt_text = f"You are a helpful assistant.<|endofprompt|>{prompt_text}"
        gen = STATE.model.inference_zero_shot(
            text, prompt_text, prompt_wav, stream=stream,
        )

    for out in gen:
        # out is a dict like {"tts_speech": tensor}
        chunk = _tensor_to_pcm16(out["tts_speech"])
        if chunk:
            yield chunk


def _wav_header(sample_rate: int, channels: int = 1, bytes_per_sample: int = 2) -> bytes:
    import struct
    return (
        b"RIFF" + struct.pack("<I", 0xFFFFFFFF) + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, channels, sample_rate,
                      sample_rate * channels * bytes_per_sample,
                      channels * bytes_per_sample, bytes_per_sample * 8)
        + b"data" + struct.pack("<I", 0xFFFFFFFF)
    )


# ---------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield


app = FastAPI(title="cosyvoice-svc", lifespan=lifespan)


class SpeechRequest(BaseModel):
    input: str = Field(..., min_length=1)
    voice: str = Field(default="female")
    instructions: Optional[str] = None
    response_format: str = Field(default="pcm")  # pcm | wav
    stream: bool = False
    model: Optional[str] = None  # ignored, OpenAI compat


def _check_auth(authorization: Optional[str]):
    if AUTH_TOKEN is None:
        return  # auth disabled at this layer (gateway handles it)
    tok = None
    if authorization and authorization.startswith("Bearer "):
        tok = authorization[len("Bearer "):]
    if tok != AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="invalid token")


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": STATE.model is not None,
            "sample_rate": STATE.sample_rate, "voices": sorted(STATE.voices)}


@app.get("/v1/audio/voices")
async def list_voices(authorization: Optional[str] = Header(default=None)):
    _check_auth(authorization)
    return {"voices": sorted(STATE.voices)}


@app.post("/v1/audio/speech")
async def create_speech(
    body: SpeechRequest,
    authorization: Optional[str] = Header(default=None),
):
    _check_auth(authorization)
    if body.response_format not in ("pcm", "wav"):
        raise HTTPException(status_code=400, detail="response_format must be pcm or wav")

    try:
        # Synthesis is blocking (pytorch). Run it in a thread so the event
        # loop isn't blocked. We collect into a generator-backed response.
        import anyio

        async def agen():
            if body.response_format == "wav":
                yield _wav_header(STATE.sample_rate)
            # Bridge the blocking generator to async by running it in a worker thread,
            # pushing chunks through a memory queue.
            import queue as _queue
            import threading
            q: "_queue.Queue" = _queue.Queue(maxsize=64)
            SENTINEL = object()
            err = {}

            def worker():
                try:
                    for chunk in _synthesize(
                        body.input, body.voice, body.instructions, body.stream
                    ):
                        q.put(chunk)
                except Exception as e:  # noqa: BLE001
                    err["e"] = e
                finally:
                    q.put(SENTINEL)

            t = threading.Thread(target=worker, daemon=True)
            t.start()
            while True:
                chunk = await anyio.to_thread.run_sync(q.get)
                if chunk is SENTINEL:
                    break
                yield chunk
            if "e" in err:
                # Can't change status mid-stream; log. Client sees truncated audio.
                print(f"[cosyvoice-svc] synth error: {err['e']!r}", flush=True)

        media = "audio/wav" if body.response_format == "wav" else "audio/pcm"
        return StreamingResponse(agen(), media_type=media)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)

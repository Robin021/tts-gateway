#!/usr/bin/env python3
"""upload_custom_voices.py — 批量上传 wav+txt 音色到 vllm-omini。

用法
----
把一个目录里成对的 <name>.wav + <name>.txt 全部上传成 voice。
每个 voice 的 ref_text = "You are a helpful assistant.<|endofprompt|>" + txt内容。

    # 在能访问 vllm-omini 的机器上(通常是 H20 本机):
    python3 upload_custom_voices.py /path/to/音频样品

    # 指定引擎地址(默认 localhost:8091):
    ENGINE_URL=http://localhost:8091 python3 upload_custom_voices.py ./音频样品

    # 上传前先删同名旧 voice:
    python3 upload_custom_voices.py ./音频样品 --replace

    # 只看会做什么,不真上传:
    python3 upload_custom_voices.py ./音频样品 --dry-run

仅用 Python 标准库(urllib),不需要装任何东西。
"""

from __future__ import annotations

import argparse
import io
import json
import mimetypes
import os
import re
import sys
import urllib.request
import urllib.error
import uuid
import wave
from pathlib import Path

ENGINE_URL = os.environ.get("ENGINE_URL", "http://localhost:8091")
PREFIX = "You are a helpful assistant.<|endofprompt|>"


def _trim_wav_and_text(
    wav_path: Path, transcript: str, max_seconds: float
) -> tuple[bytes, str, float]:
    """Return (wav_bytes, transcript, duration_seconds), trimming if needed.

    Robust against bogus WAV headers: files produced by streaming
    pipelines often carry a data-chunk size of 0xFFFFFFFF ("unknown
    length"), which makes wave report an absurd nframes (~25 hours).
    We therefore read the ACTUAL frames (clamped at EOF), compute the
    true duration from what we read, and always re-wrap into a clean
    header so the engine never sees the bogus one.

    If true duration exceeds max_seconds, both audio and transcript are
    trimmed (transcript proportionally, snapped to punctuation).
    CosyVoice clones best from short clean clips (~5-15s), so trimming
    is not a quality loss.
    """
    with wave.open(str(wav_path), "rb") as w:
        n_channels = w.getnchannels()
        sampwidth = w.getsampwidth()
        framerate = w.getframerate()
        header_frames = w.getnframes()
        # readframes clamps at EOF, so this yields the REAL payload even
        # when the header lies about the length.
        frames = w.readframes(header_frames)

    bytes_per_frame = n_channels * sampwidth
    true_frames = len(frames) // bytes_per_frame
    true_dur = true_frames / framerate

    if header_frames != true_frames:
        print(f"  [note] {wav_path.name}: header claims "
              f"{header_frames/framerate:.0f}s but payload is "
              f"{true_dur:.1f}s — rewrapping with a clean header")

    trimmed = False
    if true_dur > max_seconds:
        keep = int(max_seconds * framerate) * bytes_per_frame
        frames = frames[:keep]
        trimmed = True

    # Always re-wrap: fixes bogus headers, harmless for good files.
    buf = io.BytesIO()
    with wave.open(buf, "wb") as out:
        out.setnchannels(n_channels)
        out.setsampwidth(sampwidth)
        out.setframerate(framerate)
        out.writeframes(frames)
    wav_bytes = buf.getvalue()

    if not trimmed:
        return wav_bytes, transcript, true_dur

    # Truncate transcript proportionally to the TRUE duration, snapped
    # to a sentence boundary.
    ratio = max_seconds / true_dur
    approx_chars = max(1, int(len(transcript) * ratio))
    head = transcript[:approx_chars]
    m = list(re.finditer(r"[。！？!?.；;，,]", head))
    if m:
        head = head[: m[-1].end()]
    trimmed_text = head.strip() or transcript[:approx_chars].strip()
    return wav_bytes, trimmed_text, max_seconds


def _multipart(fields: dict, files: dict) -> tuple[bytes, str]:
    """Build a multipart/form-data body. files: name -> (filename, bytes, mime)."""
    boundary = "----voiceupload" + uuid.uuid4().hex
    out = bytearray()
    for k, v in fields.items():
        out += f"--{boundary}\r\n".encode()
        out += f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode()
        out += str(v).encode("utf-8")
        out += b"\r\n"
    for k, (fname, data, mime) in files.items():
        out += f"--{boundary}\r\n".encode()
        out += (
            f'Content-Disposition: form-data; name="{k}"; filename="{fname}"\r\n'
            f"Content-Type: {mime}\r\n\r\n"
        ).encode()
        out += data
        out += b"\r\n"
    out += f"--{boundary}--\r\n".encode()
    return bytes(out), boundary


def delete_voice(name: str) -> None:
    req = urllib.request.Request(
        f"{ENGINE_URL}/v1/audio/voices/{name}", method="DELETE"
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            print(f"  DELETE {name} -> {r.status}")
    except urllib.error.HTTPError as e:
        print(f"  DELETE {name} -> {e.code}")
    except Exception as e:
        print(f"  DELETE {name} -> err {e}")


def upload_voice(name: str, wav_bytes: bytes, wav_filename: str, ref_text: str) -> bool:
    mime = mimetypes.guess_type(wav_filename)[0] or "audio/wav"
    body, boundary = _multipart(
        fields={"name": name, "consent": "acknowledged", "ref_text": ref_text},
        files={"audio_sample": (wav_filename, wav_bytes, mime)},
    )
    req = urllib.request.Request(
        f"{ENGINE_URL}/v1/audio/voices",
        method="POST",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            resp = r.read().decode("utf-8", "replace")
            # vllm-omini sometimes returns HTTP 200 with an error body —
            # detect that so we don't falsely report success.
            if r.status == 200 and '"error"' not in resp and '"success":true' in resp:
                print(f"  UPLOAD {name} -> OK")
                return True
            print(f"  UPLOAD {name} -> FAILED  {resp[:200]}")
            return False
    except urllib.error.HTTPError as e:
        print(f"  UPLOAD {name} -> {e.code}  {e.read().decode('utf-8','replace')[:200]}")
        return False
    except Exception as e:
        print(f"  UPLOAD {name} -> err {e}")
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", help="目录,内含成对的 <name>.wav + <name>.txt")
    ap.add_argument("--replace", action="store_true", help="上传前先删同名旧 voice")
    ap.add_argument("--dry-run", action="store_true", help="只打印,不上传")
    ap.add_argument("--max-seconds", type=float, default=28.0,
                    help="参考音频最大时长(秒),超过自动裁剪+截断文字稿。vllm-omini 上限 30s。")
    ap.add_argument("--no-prefix", action="store_true",
                    help="不给 ref_text 加 'You are a helpful assistant.<|endofprompt|>' 前缀。"
                         "vllm-omni >= v0.24 (PR #4756) 服务端会自动加,再手动加会双重前缀,"
                         "必须用本参数上传裸转写。v0.22 及更早则不要加本参数。")
    args = ap.parse_args()

    d = Path(args.dir)
    if not d.is_dir():
        sys.exit(f"not a directory: {d}")

    wavs = sorted(d.glob("*.wav"))
    if not wavs:
        sys.exit(f"no .wav files in {d}")

    print(f"engine: {ENGINE_URL}")
    print(f"found {len(wavs)} wav files in {d}  (max_seconds={args.max_seconds})\n")

    ok, skipped, failed = 0, 0, 0
    for wav in wavs:
        name = wav.stem
        txt = wav.with_suffix(".txt")
        if not txt.exists():
            print(f"[skip] {name}: 缺少 {txt.name}(没有文字稿,跳过)")
            skipped += 1
            continue
        transcript = txt.read_text(encoding="utf-8").strip().strip('"').strip()
        if not transcript:
            print(f"[skip] {name}: 文字稿为空")
            skipped += 1
            continue

        wav_bytes, transcript, dur = _trim_wav_and_text(
            wav, transcript, args.max_seconds
        )
        ref_text = transcript if args.no_prefix else PREFIX + transcript
        trimmed_note = "" if dur < args.max_seconds - 0.01 else f"(裁剪到 {dur:.0f}s)"
        print(f"[{name}]  dur={dur:.1f}s {trimmed_note}  ref_text_len={len(ref_text)}")
        if args.dry_run:
            print(f"  (dry-run) ref_text = {ref_text[:80]}...")
            continue
        if args.replace:
            delete_voice(name)
        if upload_voice(name, wav_bytes, wav.name, ref_text):
            ok += 1
        else:
            failed += 1
        print()

    if not args.dry_run:
        print(f"\n=== done: {ok} ok, {failed} failed, {skipped} skipped ===")
        print("查看已上传列表:")
        print(f"  curl -s {ENGINE_URL}/v1/audio/voices")


if __name__ == "__main__":
    main()

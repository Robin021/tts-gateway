#!/usr/bin/env python3
"""transcribe_trimmed.py — 修复"裁剪后文字稿不匹配"的 voice。

问题: upload_custom_voices.py 把 >30s 的参考音频裁到 28s,但文字稿是按
字数比例截断的 — 语速不均匀导致文字和音频对不上,CosyVoice 的 ref_text
对齐机制错位,合成输出丢前半句。

修法: 对裁剪后的 28s 音频用 whisper 重新转写,得到与音频完全对应的文字稿。

用法(在 H20 上,用 cosyvoice-svc 镜像跑,它有 whisper+GPU):

    docker run --rm --runtime nvidia --gpus '"device=2"' \
      -v /opt/custom-voices:/data \
      -v /opt/transcribe_trimmed.py:/opt/transcribe.py:ro \
      cosyvoice-svc:latest \
      python /opt/transcribe.py /data --max-seconds 28

产物: 对每个超长的 <name>.wav 生成
    /data/trimmed/<name>.wav   (28s 裁剪版)
    /data/trimmed/<name>.txt   (whisper 转写,与裁剪版精确对应)

然后在宿主机重新上传:
    ENGINE_URL=http://localhost:8091 \
    python3 /opt/upload_custom_voices.py /data/trimmed --replace
    (注意 upload 脚本读的是 /opt/custom-voices/trimmed 宿主机路径)
"""

from __future__ import annotations

import argparse
import io
import sys
import wave
from pathlib import Path


def trim_wav(src: Path, dst: Path, max_seconds: float) -> float:
    with wave.open(str(src), "rb") as w:
        nch, sw, sr = w.getnchannels(), w.getsampwidth(), w.getframerate()
        nframes = w.getnframes()
        dur = nframes / sr
        keep = min(nframes, int(max_seconds * sr))
        w.rewind()
        frames = w.readframes(keep)
    with wave.open(str(dst), "wb") as out:
        out.setnchannels(nch)
        out.setsampwidth(sw)
        out.setframerate(sr)
        out.writeframes(frames)
    return min(dur, max_seconds)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", help="含 .wav/.txt 的目录 (容器内路径)")
    ap.add_argument("--max-seconds", type=float, default=28.0)
    ap.add_argument("--model", default="base",
                    help="whisper 模型: tiny/base/small/medium (大=准但慢)")
    args = ap.parse_args()

    d = Path(args.dir)
    out_dir = d / "trimmed"
    out_dir.mkdir(exist_ok=True)

    # 找出超长的 wav
    long_wavs = []
    for wav in sorted(d.glob("*.wav")):
        with wave.open(str(wav), "rb") as w:
            dur = w.getnframes() / w.getframerate()
        if dur > args.max_seconds:
            long_wavs.append((wav, dur))
        else:
            print(f"[skip] {wav.name}: {dur:.1f}s <= {args.max_seconds}s,无需处理")

    if not long_wavs:
        print("没有超长音频,无事可做")
        return

    print(f"\n{len(long_wavs)} 个超长音频需要裁剪+转写: "
          f"{[w.name for w, _ in long_wavs]}")
    print(f"加载 whisper 模型 '{args.model}' (第一次会下载) ...", flush=True)

    import whisper  # heavy import
    model = whisper.load_model(args.model)
    print("whisper 就绪\n", flush=True)

    for wav, orig_dur in long_wavs:
        name = wav.stem
        dst_wav = out_dir / wav.name
        dur = trim_wav(wav, dst_wav, args.max_seconds)
        print(f"[{name}] {orig_dur:.1f}s -> {dur:.1f}s,转写中 ...", flush=True)

        result = model.transcribe(
            str(dst_wav), language="zh", fp16=True,
        )
        text = result["text"].strip()
        (out_dir / f"{name}.txt").write_text(text, encoding="utf-8")
        print(f"  转写: {text}\n", flush=True)

    print(f"完成。裁剪版 + 精确文字稿在 {out_dir}/")
    print("下一步在宿主机跑:")
    print("  python3 /opt/upload_custom_voices.py <宿主机对应的 trimmed 目录> --replace")


if __name__ == "__main__":
    main()

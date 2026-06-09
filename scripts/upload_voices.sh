#!/usr/bin/env bash
# 一键上传 3 个 voice 到 vllm-omini + 测试合成
#
# 用法 (在 h20 上):
#   bash /tmp/voicepkg/upload_voices.sh
#
# ref_text 已经按 CosyVoice3 官方格式带 "You are a helpful assistant.<|endofprompt|>" 前缀

set -euo pipefail

PKG_DIR="${PKG_DIR:-/tmp/voicepkg}"
ENGINE_URL="${ENGINE_URL:-http://localhost:8091}"

cd "$PKG_DIR"

# 解压 wav (如果还没解压)
[ -f female.wav ] || tar xzf voices.tar.gz

echo "=== files ==="
ls -lh *.wav

# CosyVoice3 ref_text 必须有这个前缀 (来自官方 README 代码示例)
PFX='You are a helpful assistant.<|endofprompt|>'

REF_FEMALE="${PFX}希望你以后能够做的比我还好哟"
REF_MALE="${PFX}在那之后完全收购那家公司,因此保持管理层的一致性,利益与即将加入家族的资产保持一致。这就是我们有时不买下全部的原因。"
REF_LZL="${PFX}现在这通是甜蜜的来电哦!现在请先把所有的不愉快抛开,穿起美丽的衣服,带着幸福的微笑,还有你最愉快的心情,接电话咯!"

# 删旧的 (如果存在)
echo
echo "=== cleanup old voices (if any) ==="
for v in female male linzhiling test_voice_v3; do
  curl -s -o /dev/null -w "  DELETE $v -> %{http_code}\n" \
    -X DELETE "$ENGINE_URL/v1/audio/voices/$v"
done

echo
echo "=== upload female (官方原始 float32 wav) ==="
curl -s -X POST "$ENGINE_URL/v1/audio/voices" \
  -F "audio_sample=@female.wav" \
  -F "name=female" \
  -F "consent=acknowledged" \
  -F "ref_text=$REF_FEMALE"
echo

echo
echo "=== upload male ==="
curl -s -X POST "$ENGINE_URL/v1/audio/voices" \
  -F "audio_sample=@male.wav" \
  -F "name=male" \
  -F "consent=acknowledged" \
  -F "ref_text=$REF_MALE"
echo

echo
echo "=== upload linzhiling ==="
curl -s -X POST "$ENGINE_URL/v1/audio/voices" \
  -F "audio_sample=@linzhiling.wav" \
  -F "name=linzhiling" \
  -F "consent=acknowledged" \
  -F "ref_text=$REF_LZL"
echo

echo
echo "=== voices list ==="
curl -s "$ENGINE_URL/v1/audio/voices" | python3 -m json.tool

echo
echo "=== synth test (each voice) ==="
TEXT="你好,这是一段测试,我们正在验证语音合成的效果。"
for v in female male linzhiling; do
  echo "--- voice=$v ---"
  curl -s -o "/tmp/${v}_test.pcm" \
    -w "  http=%{http_code}  bytes=%{size_download}  ttfb=%{time_starttransfer}s  total=%{time_total}s\n" \
    -X POST "$ENGINE_URL/v1/audio/speech" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"cosyvoice3\",\"input\":\"$TEXT\",\"voice\":\"$v\",\"task_type\":\"CustomVoice\",\"response_format\":\"pcm\"}"
  sz=$(stat -c%s "/tmp/${v}_test.pcm" 2>/dev/null || echo 0)
  if [ "$sz" -gt 1000 ]; then
    python3 - <<PY
import wave
data = open("/tmp/${v}_test.pcm","rb").read()
with wave.open("/tmp/${v}_test.wav","wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000)
    w.writeframes(data)
print(f"  saved /tmp/${v}_test.wav  ({len(data)} bytes, {len(data)/2/24000:.2f}s @ 24kHz)")
PY
  else
    echo "  ❌ synth failed:"
    cat "/tmp/${v}_test.pcm"
    echo
  fi
done

echo
echo "=== ALL DONE ==="
echo "测试 wav 在 /tmp/{female,male,linzhiling}_test.wav"
echo "scp 回 Mac 听:"
echo "  scp 'root@221.194.152.20:/tmp/*_test.wav' ~/Downloads/"

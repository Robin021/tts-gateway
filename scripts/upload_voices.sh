#!/usr/bin/env bash
# 一键上传 base voices + style/dialect/emotion 变体到 vllm-omini.
#
# 关键发现 (2026-06-09):
#   vllm-omini 的 cosyvoice3 实现 NOT support `instructions` field —
#   字段被接受但不传到模型。但是 ref_text 是按 CosyVoice3 官方
#   instruct2 格式存的 ("You are a helpful assistant. <instruction>。<|endofprompt|>"),
#   因此可以通过给同一段录音上传多个 voice、每个的 ref_text 注入
#   不同 instruction,实现"voice 变体 = 风格切换"。
#
#   实测有效:female_cantonese 用粤语合成。
#
# 用法 (在 h20 上):
#   bash /tmp/voicepkg/upload_voices.sh
#
# 添加新变体: 在 VARIANTS_FEMALE / VARIANTS_MALE 里加一行
#   "<voice_name>|<instruction (会被包进 instruct prompt)>"

set -euo pipefail

PKG_DIR="${PKG_DIR:-/tmp/voicepkg}"
ENGINE_URL="${ENGINE_URL:-http://localhost:8091}"

cd "$PKG_DIR"
[ -f female.wav ] || tar xzf voices.tar.gz

echo "=== files ==="
ls -lh *.wav

# CosyVoice3 ref_text 必须以这个开头 (官方 README 给的格式)
SYS='You are a helpful assistant.'
EOP='<|endofprompt|>'

# 普通 zero_shot (ref_text = 系统前缀 + 原始转写)
REF_FEMALE="${SYS}${EOP}希望你以后能够做的比我还好哟"
REF_MALE="${SYS}${EOP}在那之后完全收购那家公司,因此保持管理层的一致性,利益与即将加入家族的资产保持一致。这就是我们有时不买下全部的原因。"
REF_LZL="${SYS}${EOP}现在这通是甜蜜的来电哦!现在请先把所有的不愉快抛开,穿起美丽的衣服,带着幸福的微笑,还有你最愉快的心情,接电话咯!"

# instruct2 hack (ref_text = 系统前缀 + " " + instruction + "<|endofprompt|>", 不带原始转写)
# 格式: "<voice_name>|<instruction>"
VARIANTS_FEMALE=(
  "female_cantonese|请用广东话表达"
  "female_sichuan|请用四川话表达"
  "female_excited|请用兴奋的语气说"
  "female_sad|请用悲伤的语气说"
  "female_fast|请用尽可能快的语速说"
  "female_slow|请用缓慢的语速说"
)

VARIANTS_MALE=(
  # male_cantonese: 实测合成失败 (no output) — 已移除
  # male_sichuan: 实测听不出四川话 — 已移除
  # male_fast: 实测听不清 — 已移除
  "male_excited|请用兴奋的语气说"
  "male_sad|请用悲伤的语气说"
  "male_calm|请用平静沉稳的语气说"
)

# All voice names we manage — used for cleanup.
# 包括历史上用过但现在不再上传的(确保旧的失败变体被清理掉)
ALL_VOICES=(
  female male linzhiling
  test_voice_v3 female_cantonese_test
  # 历史失败变体 (合成不出 / 听不懂),保留在 cleanup 列表确保被删
  male_cantonese male_sichuan male_fast
)
for line in "${VARIANTS_FEMALE[@]}" "${VARIANTS_MALE[@]}"; do
  ALL_VOICES+=("${line%%|*}")
done

echo
echo "=== cleanup ==="
for v in "${ALL_VOICES[@]}"; do
  curl -s -o /dev/null -w "  DELETE $v -> %{http_code}\n" \
    -X DELETE "$ENGINE_URL/v1/audio/voices/$v"
done

upload() {
  local name="$1" wav="$2" ref_text="$3"
  echo "  upload $name ..."
  curl -s -X POST "$ENGINE_URL/v1/audio/voices" \
    -F "audio_sample=@${wav}" \
    -F "name=${name}" \
    -F "consent=acknowledged" \
    -F "ref_text=${ref_text}" \
    -w "    -> http=%{http_code}\n"
}

echo
echo "=== upload base voices (zero_shot mode, normal cloning) ==="
upload female      female.wav      "$REF_FEMALE"
upload male        male.wav        "$REF_MALE"
upload linzhiling  linzhiling.wav  "$REF_LZL"

echo
echo "=== upload female variants (instruct2 hack via ref_text) ==="
for line in "${VARIANTS_FEMALE[@]}"; do
  name="${line%%|*}"
  instr="${line#*|}"
  ref="${SYS} ${instr}。${EOP}"
  upload "$name" female.wav "$ref"
done

echo
echo "=== upload male variants ==="
for line in "${VARIANTS_MALE[@]}"; do
  name="${line%%|*}"
  instr="${line#*|}"
  ref="${SYS} ${instr}。${EOP}"
  upload "$name" male.wav "$ref"
done

echo
echo "=== voices list ==="
curl -s "$ENGINE_URL/v1/audio/voices" | python3 -m json.tool

echo
echo "=== synth tests (one per voice) ==="
TEXT="你好,这是一段测试,我们正在验证语音合成的效果。"

# Test all voices
ALL_TEST=(female male linzhiling)
for line in "${VARIANTS_FEMALE[@]}" "${VARIANTS_MALE[@]}"; do
  ALL_TEST+=("${line%%|*}")
done

for v in "${ALL_TEST[@]}"; do
  echo "--- voice=$v ---"
  curl -s -o "/tmp/${v}_test.pcm" \
    -w "  http=%{http_code}  bytes=%{size_download}  ttfb=%{time_starttransfer}s\n" \
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
print(f"  saved /tmp/${v}_test.wav  ({len(data)/2/24000:.2f}s)")
PY
  else
    echo "  ❌ synth failed:"
    cat "/tmp/${v}_test.pcm"; echo
  fi
done

echo
echo "=== DONE ==="
echo "测试 wav 在 /tmp/<voice>_test.wav"
echo "拉回 Mac 听全部:"
echo "  scp 'root@<gateway-host>:/tmp/*_test.wav' ~/Downloads/"
echo "听单个:"
echo "  scp root@<gateway-host>:/tmp/female_cantonese_test.wav ~/Downloads/ && afplay ~/Downloads/female_cantonese_test.wav"

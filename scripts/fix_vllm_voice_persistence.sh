#!/usr/bin/env bash
# fix_vllm_voice_persistence.sh
#
# 给现有的 cosyvoice3-tts 容器加 voice 持久化挂载。
#
# 在 H20 上跑(从 Mac 也行,会自己 ssh 过去):
#   bash scripts/fix_vllm_voice_persistence.sh
#
# 做什么:
#   1. 创建宿主机持久化目录 /data/model/vllm-omini-speakers
#   2. 如果容器在跑,先把当前的 voices 从容器里拷出来 → 持久化目录
#   3. 停 + 删旧 cosyvoice3-tts 容器
#   4. 用新的启动命令重启(加 -v 挂载,保留原有 GPU/IPC 等设置)
#   5. 等容器起来 + 验证 voices 都还在

set -euo pipefail

REMOTE="${REMOTE:?REMOTE env var must be set, e.g. REMOTE=root@<gateway-host> ...}"

# 如果你已经在 H20 上跑这个脚本,设 LOCAL=1 跳过 ssh
LOCAL="${LOCAL:-0}"

run() {
    if [ "$LOCAL" = "1" ]; then
        bash -c "$1"
    else
        ssh "$REMOTE" "$1"
    fi
}

echo "=== 当前 cosyvoice3-tts 容器 ==="
run 'docker ps --filter name=cosyvoice3-tts --format "  {{.Status}}  {{.Image}}  {{.Names}}"'

echo
echo "=== 1. 创建持久化目录 + 备份现有 voice (如果有) ==="
run 'mkdir -p /data/model/vllm-omini-speakers
if docker ps --filter name=cosyvoice3-tts --format "{{.Names}}" | grep -q cosyvoice3-tts; then
    echo "  备份容器内的 voices..."
    docker cp cosyvoice3-tts:/root/.cache/vllm-omni/speakers/. /data/model/vllm-omini-speakers/ 2>/dev/null \
        || echo "  (容器里没有 voices 目录,跳过)"
fi
ls -la /data/model/vllm-omini-speakers/ | head -10'

echo
echo "=== 2. 停 + 删旧容器 ==="
run 'docker stop cosyvoice3-tts 2>/dev/null || true
docker rm cosyvoice3-tts 2>/dev/null || true'

echo
echo "=== 3. 重启容器(加挂载) ==="
run '
docker run -d \
  --name cosyvoice3-tts \
  --runtime nvidia \
  --gpus "\"device=1\"" \
  --shm-size=32g \
  -p 8091:8091 \
  -e HF_ENDPOINT=https://hf-mirror.com \
  -e VLLM_USE_MODELSCOPE=True \
  -v /data/model/huggingface:/root/.cache/huggingface \
  -v /data/model/modelscope:/root/.cache/modelscope \
  -v /data/model/vllm-omini-speakers:/root/.cache/vllm-omni/speakers \
  --restart always \
  742678245098.dkr.ecr.cn-north-1.amazonaws.com.cn/vllm/vllm-omni:v0.22.0 \
  vllm serve FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
    --served-model-name cosyvoice3 \
    --omni \
    --port 8091 \
    --host 0.0.0.0 \
    --trust-remote-code \
    --enforce-eager \
    --max-num-seqs 16 \
    --gpu-memory-utilization 0.45 \
    --max-model-len 2048
docker ps --filter name=cosyvoice3-tts --format "  {{.Status}}  {{.Image}}  {{.Names}}"'

echo
echo "=== 4. 等容器健康(约 60-150s 加载模型) ==="
# vllm-omni 加载完才会响应 /v1/audio/voices, 用它做 ready 信号
for i in $(seq 1 75); do
    if run 'curl -fsS http://localhost:8091/v1/audio/voices 2>/dev/null | grep -q voices'; then
        echo "  ✓ vllm-omini ready (took ~$((i*2))s)"
        break
    fi
    sleep 2
    if [ $((i % 15)) = 0 ]; then
        echo "  waiting ($((i*2))s) ..."
    fi
done

echo
echo "=== 5. 验证 voices 还在 ==="
run 'curl -s http://localhost:8091/v1/audio/voices'
echo

echo
echo "=== DONE ==="
echo "下次容器重启 voice 不会丢了。宿主机持久化目录: /data/model/vllm-omini-speakers/"
echo
echo "(如果 voices 列表是空的,说明备份失败 / 容器原本就没有 voice。"
echo " 跑 bash scripts/upload_voices.sh 重新上传)"

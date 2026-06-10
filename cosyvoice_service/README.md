# cosyvoice-svc

Official CosyVoice library + FastAPI, in Docker. Provides **reliable**
`instructions` (dialect / emotion / speed) support via the model
author's own `inference_instruct2()` — which vllm-omini's reimplementation
does not.

This sits behind `tts-gateway` as an alternative backend to vllm-omini.

```
[client] → [tts-gateway] → [cosyvoice-svc :8092]  (this — instruct2 works)
                       ↘ [vllm-omini   :8091]      (fast, but no instruct2)
```

## Why a separate service

vllm-omini reimplemented CosyVoice3 from scratch for serving speed, but
its `instructions` path is broken (ignored / garbled). The official
CosyVoice Python library has a working `inference_instruct2()`. We can't
get both speed and instruct from one backend today, so we run both:

- vllm-omini for plain cloning (low latency, batched)
- cosyvoice-svc for instruct2 (dialect/emotion), accepting higher latency

The gateway can route per-request (instructions present → cosyvoice-svc).

## ⚠️ Before building: match the colleague's working env

CosyVoice's environment is finicky. The colleague already has it working
with **local pytorch**. To avoid days of dependency hell, get these from
his setup and bake them into the Dockerfile ARGs:

```bash
# On the colleague's machine / wherever inference_instruct2 works:
python --version
python -c "import torch; print(torch.__version__, torch.version.cuda)"
cd /path/to/CosyVoice && git rev-parse HEAD
# Also: how does he load the model + call instruct2? (the exact script)
# And: what instruct_text format does he pass? (with or without the
#      'You are a helpful assistant. ... <|endofprompt|>' wrapper)
```

Plug those into:
- `Dockerfile` ARG `CUDA_IMAGE` (match torch CUDA)
- `Dockerfile` ARG `TORCH_VERSION`, `TORCHAUDIO_VERSION`, `PYTORCH_INDEX_URL`
  (pin exact torch build)
- `Dockerfile` ARG `COSYVOICE_REF` (pin the commit)
- `app.py` `_format_instruct()` (match his instruct_text format)

Known-good default pins in this repo are aligned with the current
official CosyVoice requirements:

- `torch==2.3.1`, `torchaudio==2.3.1`, CUDA 12.1 wheel index
- `lightning==2.2.4`
- `matplotlib==3.7.5`

The Dockerfile also runs a build-time import smoke test for the exact
chain that previously failed:
`cosyvoice.flow.flow_matching -> matcha -> matplotlib`.

## Build + run on H20

```bash
# 1. Put the 3 voice wavs where the container can mount them
mkdir -p /opt/cosyvoice-voices
# copy female.wav / male.wav / linzhiling.wav here
#   (they're in the gateway repo's scripts/voices.tar.gz)

# 2. Build (GPU 2 assumed free; vllm-omini is on GPU 1)
cd cosyvoice_service
docker build -t cosyvoice-svc:latest .

# 3. Run
docker run -d --name cosyvoice-svc --restart always \
  --runtime nvidia --gpus '"device=2"' \
  --shm-size=16g \
  -p 8092:8092 \
  -v /data/model/modelscope/hub/models/FunAudioLLM/Fun-CosyVoice3-0___5B-2512:/models/Fun-CosyVoice3-0.5B:ro \
  -v /opt/cosyvoice-voices:/app/voices:ro \
  cosyvoice-svc:latest

# 4. Wait for model load (~1-2 min), then health check
sleep 90
curl http://localhost:8092/health

# 5. Test instruct2 directly (the moment of truth)
curl -s -o /tmp/c.pcm -X POST http://localhost:8092/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"今天我们一起去喝早茶,顺便聊聊最近发生的事情吧","voice":"female","instructions":"请用广东话表达","response_format":"pcm"}'
python3 -c "import wave;d=open('/tmp/c.pcm','rb').read();w=wave.open('/tmp/c.wav','wb');w.setnchannels(1);w.setsampwidth(2);w.setframerate(24000);w.writeframes(d);w.close();print(len(d)/2/24000,'s')"
# scp /tmp/c.wav back and listen — should be CLEAN full-sentence Cantonese
```

If the H20 box is already inside a half-fixed container and you just
need to unblock the current `ModuleNotFoundError: No module named
'matplotlib'`, the immediate hotfix is:

```bash
pip install --no-cache-dir matplotlib==3.7.5 gdown==5.1.0 wget==3.2 \
  omegaconf==2.3.0 rich==13.7.1 hydra-core==1.3.2 \
  tensorboard==2.14.0 librosa==0.10.2 soundfile==0.12.1

PYTHONPATH=/opt/CosyVoice:/opt/CosyVoice/third_party/Matcha-TTS \
python -c "from cosyvoice.flow.flow_matching import ConditionalCFM; print('import ok')"
```

That is only a container hotfix. Rebuild the image from the Dockerfile
afterwards so a restart does not lose the fix.

If that test produces clean Cantonese **consistently** (run it 3-5×),
this backend is good and we point the gateway at it.

## Config (env vars)

| var | default | meaning |
|---|---|---|
| `COSYVOICE_MODEL_DIR` | `/models/Fun-CosyVoice3-0.5B` | model weights (mount) |
| `COSYVOICE_VOICES` | `/app/voices.json` | voice registry |
| `COSYVOICE_LOAD_VLLM` | `0` | enable vllm LLM accel (extra setup) |
| `COSYVOICE_LOAD_TRT` | `0` | enable tensorrt |
| `COSYVOICE_FP16` | `0` | fp16 inference |
| `PORT` | `8092` | listen port |

Start with all accel off (plain pytorch) to get it WORKING, then turn on
vllm/trt/fp16 one at a time if latency is too high.

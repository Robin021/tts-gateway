# cosyvoice-svc — official CosyVoice library + FastAPI, in Docker.
#
# ⚠️ CosyVoice 的环境很挑(Matcha-TTS 子模块 / torch 版本 / ttsfrd / onnxruntime-gpu)。
# 这个 Dockerfile 是"尽力一版",大概率需要照你同事跑通的本地 pytorch 环境
# 校准 1-2 处(torch 版本、cuda 版本、requirements 冲突)。
#
# 构建前确认:
#   1. 你同事的 python 版本 (python --version)
#   2. 他的 torch 版本 (python -c "import torch;print(torch.__version__, torch.version.cuda)")
#   3. CosyVoice repo 的 commit (cd CosyVoice && git rev-parse HEAD)
#   把这三个填进下面的 ARG,匹配上最稳。
#
# 构建:
#   cd cosyvoice_service
#   docker build -t cosyvoice-svc:latest .
#
# 运行 (模型 + voice wav 通过挂载注入,不烤进镜像):
#   docker run -d --name cosyvoice-svc --restart always \
#     --runtime nvidia --gpus '"device=2"' \
#     --shm-size=16g \
#     -p 8092:8092 \
#     -v /data/model/modelscope/hub/models/FunAudioLLM/Fun-CosyVoice3-0___5B-2512:/models/Fun-CosyVoice3-0.5B:ro \
#     -v /opt/cosyvoice-voices:/app/voices:ro \
#     cosyvoice-svc:latest

# CUDA + cuDNN 运行时基础镜像。版本要和你同事的 torch CUDA 版本对齐。
# 默认 12.1;如果同事用 11.8,改 ARG。
ARG CUDA_IMAGE=docker.m.daocloud.io/nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
FROM ${CUDA_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=5

# 系统依赖: python3.10, git, sox/libsox(CosyVoice 需要), ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3-pip \
        git git-lfs \
        sox libsox-dev libsox-fmt-all \
        ffmpeg \
        build-essential \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && ln -sf /usr/bin/pip3 /usr/bin/pip

# 克隆 CosyVoice (含 Matcha-TTS 子模块)
#   github 在国内常抖 (curl 16 HTTP2 framing)。对策:
#   - 关 HTTP2 走 HTTP/1.1
#   - 加大 postBuffer
#   - 浅克隆 --depth 1 (不要历史,快很多)
#   - 重试 5 次
ARG COSYVOICE_REPO=https://github.com/FunAudioLLM/CosyVoice.git
ARG COSYVOICE_REF=main
RUN git config --global http.version HTTP/1.1 \
    && git config --global http.postBuffer 524288000 \
    && for i in 1 2 3 4 5; do \
         echo "clone attempt $i ..."; \
         git clone --recursive --depth 1 --shallow-submodules "${COSYVOICE_REPO}" /opt/CosyVoice && break; \
         echo "attempt $i failed, retrying in 5s"; rm -rf /opt/CosyVoice; sleep 5; \
       done \
    && test -d /opt/CosyVoice/cosyvoice \
    && if [ "${COSYVOICE_REF}" != "main" ]; then \
         cd /opt/CosyVoice \
         && git fetch --depth 1 origin "${COSYVOICE_REF}" \
         && git checkout FETCH_HEAD; \
       fi \
    && (cd /opt/CosyVoice && git submodule update --init --recursive --depth 1 || true)

# 装 torch (CUDA 12.1 轮子),版本对齐 CosyVoice requirements.txt (torch==2.3.1)
# numpy 先装,避免某些包 build 时找不到 numpy
ARG TORCH_VERSION=2.3.1
ARG TORCHAUDIO_VERSION=2.3.1
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu121
RUN pip install --no-cache-dir numpy==1.26.4
RUN pip install --no-cache-dir \
        torch==${TORCH_VERSION} torchaudio==${TORCHAUDIO_VERSION} \
        --index-url ${PYTORCH_INDEX_URL} \
    || pip install --no-cache-dir torch==${TORCH_VERSION} torchaudio==${TORCHAUDIO_VERSION}

# pyworld (CosyVoice 依赖)要编译 C++,需要 Python 头文件 Python.h。
# 放在 torch 安装之后,保证上面 torch 那层缓存不失效(不重下 2.5GB)。
RUN apt-get update && apt-get install -y --no-install-recommends python3.10-dev \
    && rm -rf /var/lib/apt/lists/*

# 单独装 pyworld:用 --no-build-isolation 让它用已装好的 numpy/cython,
# 否则它的 build 隔离环境会绕过清华镜像去 pythonhosted.org 重下 numpy(被墙超时)。
RUN pip install --no-cache-dir cython "numpy==1.26.4" setuptools wheel \
    && pip install --no-cache-dir --no-build-isolation pyworld==0.3.4

# 装 CosyVoice 推理依赖。
# 只过滤掉真正会破坏 build 的包(其余全装,避免模块级 import 缺包的打地鼠):
#   deepspeed       - 训练用,要 nvcc 编译(runtime 镜像没有)→ build 失败元凶
#   tensorrt-cu12*  - 仅 load_trt=True 才需要,体积巨大,默认关。如要开 TRT 自行装回
#   pyworld         - 上面已单独用 --no-build-isolation 装好
#   torch/torchaudio/numpy - 上面已固定装好,避免重装
# 注意: openai-whisper / lightning / matplotlib 等必须留 —— CosyVoice/Matcha-TTS
#       会在模块级 import 它们,删了会启动报 ModuleNotFoundError。
RUN grep -vE '^(deepspeed|tensorrt-cu12|pyworld|torch==|torchaudio==|numpy==)' \
        /opt/CosyVoice/requirements.txt > /tmp/req.inference.txt \
    && echo "=== filtered requirements ===" && cat /tmp/req.inference.txt \
    && pip install --no-cache-dir -r /tmp/req.inference.txt

# Matcha-TTS is pulled in as a CosyVoice submodule and has module-level
# imports that are easy to miss when upstream requirements change. Pin the
# small runtime set explicitly so startup does not become a missing-package
# whack-a-mole after a long Docker build.
RUN pip install --no-cache-dir \
        matplotlib==3.7.5 \
        gdown==5.1.0 \
        wget==3.2 \
        omegaconf==2.3.0 \
        rich==13.7.1 \
        hydra-core==1.3.2 \
        tensorboard==2.14.0 \
        librosa==0.10.2 \
        soundfile==0.12.1

# 我们服务自己的依赖 (fastapi/uvicorn/pydantic 已在 CosyVoice requirements 里;补 anyio)
RUN pip install --no-cache-dir 'anyio>=4'

# 让 cosyvoice 包可被 import
ENV PYTHONPATH=/opt/CosyVoice:/opt/CosyVoice/third_party/Matcha-TTS

# Build-time smoke test for the exact import chain that failed in production:
# cosyvoice.flow.flow_matching -> matcha -> matplotlib/rich/omegaconf/etc.
RUN python -c "from cosyvoice.flow.flow_matching import ConditionalCFM; from cosyvoice.cli.cosyvoice import AutoModel; from cosyvoice.utils.file_utils import load_wav; print('cosyvoice import smoke test OK')"

WORKDIR /app
COPY app.py /app/app.py
COPY voices.json /app/voices.json

# 模型 + voice wav 通过挂载注入 (见文件头注释)
ENV COSYVOICE_MODEL_DIR=/models/Fun-CosyVoice3-0.5B \
    COSYVOICE_VOICES=/app/voices.json \
    COSYVOICE_LOAD_VLLM=0 \
    COSYVOICE_LOAD_TRT=0 \
    COSYVOICE_FP16=0 \
    PORT=8092

EXPOSE 8092

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request,sys;sys.exit(0 if urllib.request.urlopen('http://localhost:8092/health',timeout=3).status==200 else 1)" || exit 1

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8092"]

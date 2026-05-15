FROM nvidia/cuda:13.2.1-base-ubuntu24.04

RUN apt-get update -y \
    && apt-get install -y python3-pip curl \
    && curl -LsSf https://astral.sh/uv/install.sh  | sh

ENV PATH="/root/.local/bin:$PATH"

# Ubuntu 24.04 PEP 668: allow uv to install into the system Python
ENV UV_SYSTEM_PYTHON=1 \
    UV_BREAK_SYSTEM_PACKAGES=1

RUN ldconfig /usr/local/cuda-13.2/compat/ 2>/dev/null || ldconfig

# Install vLLM 0.21.0 with FlashInfer - native TurboQuant + Blackwell (sm_120) support
RUN uv pip install --system "packaging>=24.2" && \
    uv pip install --system "vllm[flashinfer]==0.21.0" --extra-index-url https://download.pytorch.org/whl/cu130

# Install additional Python dependencies (after vLLM to avoid PyTorch version conflicts)
COPY builder/requirements.txt /requirements.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r /requirements.txt

# Setup for Option 2: Building the Image with the Model included
ARG MODEL_NAME=""
ARG TOKENIZER_NAME=""
ARG BASE_PATH="/runpod-volume"
ARG QUANTIZATION=""
ARG MODEL_REVISION=""
ARG TOKENIZER_REVISION=""
ARG VLLM_NIGHTLY="false"

ENV MODEL_NAME=$MODEL_NAME \
    MODEL_REVISION=$MODEL_REVISION \
    TOKENIZER_NAME=$TOKENIZER_NAME \
    TOKENIZER_REVISION=$TOKENIZER_REVISION \
    BASE_PATH=$BASE_PATH \
    QUANTIZATION=$QUANTIZATION \
    HF_DATASETS_CACHE="${BASE_PATH}/huggingface-cache/datasets" \
    HUGGINGFACE_HUB_CACHE="${BASE_PATH}/huggingface-cache/hub" \
    HF_HOME="${BASE_PATH}/huggingface-cache/hub" \
    HF_HUB_ENABLE_HF_TRANSFER=0 \
    # Suppress Ray metrics agent warnings (not needed in containerized environments)
    RAY_METRICS_EXPORT_ENABLED=0 \
    RAY_DISABLE_USAGE_STATS=1 \
    # Prevent rayon thread pool panic in containers where ulimit -u < nproc
    # (tokenizers uses Rust's rayon which tries to spawn threads = CPU cores)
    TOKENIZERS_PARALLELISM=false \
    RAYON_NUM_THREADS=4

# Blackwell (sm_120) compatibility + long context support
ENV VLLM_FLASH_ATTN_VERSION=2 \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    # Required for context lengths > 262K tokens (e.g. 1M)
    VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

ENV PYTHONPATH="/:/vllm-workspace"

RUN if [ "${VLLM_NIGHTLY}" = "true" ]; then \
    uv pip install --system -U vllm --pre --index-url https://pypi.org/simple --extra-index-url https://wheels.vllm.ai/nightly && \
    apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/* && \
    uv pip install --system git+https://github.com/huggingface/transformers.git; \
fi

COPY src /src
RUN chmod +x /src/start.sh
RUN --mount=type=secret,id=HF_TOKEN,required=false \
    if [ -f /run/secrets/HF_TOKEN ]; then \
    export HF_TOKEN=$(cat /run/secrets/HF_TOKEN); \
    fi && \
    if [ -n "$MODEL_NAME" ]; then \
    python3 /src/download_model.py; \
    fi

# Start the handler
CMD ["/bin/bash", "/src/start.sh"]

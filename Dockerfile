# NOVA Gateway (GPU box): models via Ollama, memory, sessions, speech-to-text, voice recognition, cloud.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/data/hf
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app

COPY requirements-gpu.txt .
# Gateway + Whisper with its CUDA 12 libraries (the GPU itself comes from the NVIDIA container toolkit).
RUN pip install -r requirements-gpu.txt

# Voice recognition (who's talking): runs on the CPU, so the small CPU build of PyTorch is enough.
ARG VOICE_ID=1
RUN if [ "$VOICE_ID" = "1" ]; then \
      pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu && pip install "speechbrain>=1.0"; \
    fi

COPY nova_common ./nova_common
COPY gateway ./gateway
COPY docker ./docker

EXPOSE 8800
VOLUME ["/data"]
HEALTHCHECK --interval=30s --timeout=6s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8800/health', timeout=5)" || exit 1
ENTRYPOINT ["python", "/app/docker/entrypoint.py"]

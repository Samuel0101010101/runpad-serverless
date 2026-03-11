FROM runpod/pytorch:1.0.3-cu1281-torch290-ubuntu2204

WORKDIR /app

# Prevent basicsr from compiling CUDA extensions (uses PyTorch fallbacks at runtime)
ENV BASICSR_EXT=False

# Layer 1: system deps (cached unless Dockerfile base changes)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libsm6 libxext6 libgl1 git libsndfile1 \
    && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# Layer 2: pip requirements (cached unless requirements.txt changes)
COPY requirements.txt .
COPY xformers-stub/ /tmp/xformers-stub/
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt \
    && pip install --no-cache-dir "diffusers @ git+https://github.com/huggingface/diffusers.git@e747fe4a942ce379d73a975a82f9e4c484c74ba2" \
    && pip install --no-cache-dir --no-deps basicsr realesrgan gfpgan \
    && python -c "import importlib, pathlib, torchvision; tv=pathlib.Path(torchvision.__file__).parent/'transforms'/'functional_tensor.py'; tv.exists() or tv.write_text('from torchvision.transforms.functional import *\n')" \
    && pip install --no-cache-dir --no-deps /tmp/xformers-stub \
    && pip install --no-cache-dir --no-deps demucs openai-whisper \
    && pip install --no-cache-dir --no-deps "audiocraft @ git+https://github.com/facebookresearch/audiocraft.git@v1.3.0" \
    && pip install --no-cache-dir --no-deps "insightface>=0.7.3" \
    && rm -rf /tmp/* /root/.cache \
    # ── Trim ~500MB+ of unnecessary files from site-packages ──
    && SITE=$(python -c "import site; print(site.getsitepackages()[0])") \
    && find "$SITE" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; true \
    && find "$SITE" -type d -name "tests" -o -name "test" | xargs rm -rf 2>/dev/null; true \
    && find "$SITE" -name "*.pyc" -delete 2>/dev/null; true \
    && rm -rf "$SITE"/nvidia/*/lib/*.a 2>/dev/null; true

# Layer 3: Pre-download Wan 2.2 TI2V-5B weights so cold starts skip HuggingFace
# This adds ~12GB to the image but saves 5-10 min on every cold start.
# The layer is cached by Docker + RunPod, so rebuilds are fast unless the model changes.
ENV HF_HUB_ENABLE_HF_TRANSFER=0
ENV HF_HUB_DISABLE_XET=1
RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('Wan-AI/Wan2.2-TI2V-5B-Diffusers', cache_dir='/app/models', ignore_patterns=['*.md','*.txt'])"

# Layer 4: application code (rebuilds in seconds on code changes)
COPY handler.py wan.py realesrgan_backend.py wav2lip_backend.py whisper_model.py audiocraft_backend.py faceswap_backend.py ./

# Backend env vars
ENV FACESWAP_BACKEND=faceswap_backend:load_model
# Tell handler where the baked-in models live
ENV BAKED_MODEL_DIR=/app/models

CMD ["python", "-u", "handler.py"]
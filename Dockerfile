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
    && rm -rf /tmp/* /root/.cache

# Layer 3: application code (rebuilds in seconds on code changes)
COPY handler.py wan.py realesrgan_backend.py wav2lip_backend.py whisper_model.py audiocraft_backend.py faceswap_backend.py ./

# Backend env vars
ENV FACESWAP_BACKEND=faceswap_backend:load_model

CMD ["python", "-u", "handler.py"]
FROM runpod/pytorch:1.0.3-cu1290-torch290-ubuntu2204

WORKDIR /app

# Single layer: system deps + all pip installs + cleanup
# BASICSR_EXT=False skips 15-min CUDA extension compilation (uses PyTorch fallbacks at runtime)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libsm6 libxext6 libgl1 git \
    && BASICSR_EXT=False pip install --no-cache-dir --prefer-binary \
        runpod requests boto3 \
        diffusers transformers accelerate sentencepiece Pillow \
        opencv-python-headless openai-whisper \
        av encodec einops flashy lameenc num2words spacy \
        basicsr realesrgan gfpgan \
    && pip install --no-cache-dir --no-deps "audiocraft @ git+https://github.com/facebookresearch/audiocraft.git" \
    && apt-get autoremove -y && rm -rf /var/lib/apt/lists/* /tmp/* /root/.cache

# Copy handler and model backends
COPY handler.py wan.py realesrgan_backend.py wav2lip_backend.py whisper_model.py audiocraft_backend.py ./

CMD ["python", "-u", "handler.py"]
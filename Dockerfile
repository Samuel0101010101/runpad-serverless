FROM runpod/pytorch:1.0.3-cu1290-torch290-ubuntu2204

WORKDIR /app

# System dependencies (runtime + build)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    git \
    git-lfs \
    pkg-config \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libavfilter-dev \
    libswscale-dev \
    libswresample-dev \
    && rm -rf /var/lib/apt/lists/*

# Stable Python dependencies (no torch – already in base image)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# PyAV from wheel (needed by audiocraft at runtime)
RUN pip install --no-cache-dir av

# basicsr / realesrgan / gfpgan – install without letting them pull their own torch
RUN pip install --no-cache-dir --no-deps basicsr && \
    pip install --no-cache-dir --no-deps realesrgan && \
    pip install --no-cache-dir --no-deps gfpgan

# audiocraft – install from GitHub HEAD with --no-deps to avoid torch conflict,
# then install its non-torch runtime deps separately
RUN pip install --no-cache-dir --no-deps "audiocraft @ git+https://github.com/facebookresearch/audiocraft.git" && \
    pip install --no-cache-dir encodec einops flashy lameenc num2words spacy demucs

# Copy handler and model backends
COPY handler.py .
COPY wan.py .
COPY realesrgan_backend.py .
COPY wav2lip_backend.py .
COPY whisper_model.py .
COPY audiocraft_backend.py .

# RunPod entrypoint
CMD ["python", "-u", "handler.py"]
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
    # Build deps for PyAV (audiocraft dependency)
    pkg-config \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libavfilter-dev \
    libswscale-dev \
    libswresample-dev \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy handler and model backends
COPY handler.py .
COPY wan.py .
COPY realesrgan_backend.py .
COPY wav2lip_backend.py .
COPY whisper_model.py .
COPY audiocraft_backend.py .

# RunPod entrypoint
CMD ["python", "-u", "handler.py"]
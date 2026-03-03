FROM runpod/pytorch:1.0.3-cu1290-torch290-ubuntu2204

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy handler
COPY handler.py .

# RunPod entrypoint
CMD ["python", "-u", "handler.py"]
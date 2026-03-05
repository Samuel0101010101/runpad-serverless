FROM runpod/pytorch:1.0.3-cu1290-torch290-ubuntu2204

WORKDIR /app

# Single layer: system deps + all pip installs + cleanup
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libsm6 libxext6 libgl1 git pkg-config \
        libavformat-dev libavcodec-dev libavdevice-dev \
        libavutil-dev libavfilter-dev libswscale-dev libswresample-dev \
    && pip install --no-cache-dir \
        runpod requests boto3 \
        diffusers transformers accelerate sentencepiece Pillow \
        opencv-python-headless openai-whisper \
        av encodec einops flashy lameenc num2words spacy \
    && pip install --no-cache-dir --no-deps basicsr realesrgan gfpgan \
    && pip install --no-cache-dir --no-deps "audiocraft @ git+https://github.com/facebookresearch/audiocraft.git" \
    && apt-get purge -y pkg-config libavformat-dev libavcodec-dev libavdevice-dev \
        libavutil-dev libavfilter-dev libswscale-dev libswresample-dev \
    && apt-get autoremove -y && rm -rf /var/lib/apt/lists/* /tmp/* /root/.cache

# Copy handler and model backends
COPY handler.py wan.py realesrgan_backend.py wav2lip_backend.py whisper_model.py audiocraft_backend.py ./

CMD ["python", "-u", "handler.py"]
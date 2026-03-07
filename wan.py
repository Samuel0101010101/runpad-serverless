"""Wan 2.2 TI2V-5B backend for Tarik pipeline.

Uses WanImageToVideoPipeline for image-to-video generation at 720P 24fps.
Fits on RTX 4090 (24 GB) with CPU offload.

Provides load_wan_model(cache_dir) -> WanModel with .generate_video() interface
expected by handler.py's run_generate_video().
"""

import logging
import os
from pathlib import Path

import torch

# Disable hf_transfer and xet downloaders — both crash on large sharded models
# with "File Reconstruction Error: receiver dropped".  Fall back to the reliable
# default Python downloader.
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from diffusers import WanImageToVideoPipeline, AutoencoderKLWan
from diffusers.utils import export_to_video
from PIL import Image

logger = logging.getLogger("tarik-handler.wan")

MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
DEFAULT_FPS = 24
RESOLUTION_MAP = {
    "480p": (848, 480),
    "720p": (1280, 704),
}

NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，"
    "整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，"
    "画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，"
    "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


class WanModel:
    """Thin wrapper around Wan TI2V pipeline matching handler interface."""

    def __init__(self, pipe):
        self.pipe = pipe
        self.name = "wan_ti2v_5b"

    def generate_video(
        self,
        image_path: str,
        motion_prompt: str,
        duration_seconds: int = 5,
        resolution: str = "720p",
        output_path: str = "/tmp/output.mp4",
    ) -> None:
        width, height = RESOLUTION_MAP.get(resolution, RESOLUTION_MAP["720p"])
        num_frames = duration_seconds * DEFAULT_FPS + 1

        image = Image.open(image_path).convert("RGB").resize((width, height))

        logger.info(
            "Generating video: prompt=%r resolution=%s frames=%d",
            motion_prompt[:80],
            resolution,
            num_frames,
        )

        result = self.pipe(
            image=image,
            prompt=motion_prompt,
            negative_prompt=NEGATIVE_PROMPT,
            num_frames=num_frames,
            width=width,
            height=height,
            guidance_scale=5.0,
            num_inference_steps=50,
        )

        frames = result.frames[0]
        export_to_video(frames, output_path, fps=DEFAULT_FPS)
        logger.info("Video saved to %s (%d frames)", output_path, len(frames))


def load_wan_model(cache_dir: str) -> WanModel:
    """Entry point called by handler via WAN_TI2V_5B_BACKEND=wan:load_wan_model."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for Wan TI2V but not available on this worker"
        )

    logger.info("Loading Wan TI2V-5B pipeline from %s (cache=%s)", MODEL_ID, cache_dir)

    logger.info(
        "GPU: %s, VRAM: %.1f GB",
        torch.cuda.get_device_name(0),
        torch.cuda.get_device_properties(0).total_memory / (1024**3),
    )

    # VAE must be loaded in float32 per model card
    vae = AutoencoderKLWan.from_pretrained(
        MODEL_ID,
        subfolder="vae",
        torch_dtype=torch.float32,
        cache_dir=cache_dir,
    )

    pipe = WanImageToVideoPipeline.from_pretrained(
        MODEL_ID,
        vae=vae,
        cache_dir=cache_dir,
        torch_dtype=torch.bfloat16,
    )
    pipe.enable_model_cpu_offload(gpu_id=0)
    pipe.vae.enable_tiling()
    logger.info("Wan TI2V-5B I2V pipeline loaded with CPU offload + VAE tiling")
    return WanModel(pipe)

"""Wan 2.1 I2V-5B backend for Tarik pipeline.

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

from diffusers import WanImageToVideoPipeline
from diffusers.utils import export_to_video
from PIL import Image

logger = logging.getLogger("tarik-handler.wan")

MODEL_ID = "Wan-AI/Wan2.1-I2V-5B-480P-Diffusers"
DEFAULT_NUM_FRAMES = 81  # ~5s at 16fps
DEFAULT_FPS = 16
RESOLUTION_MAP = {
    "480p": (848, 480),
    "720p": (1280, 720),
}


class WanModel:
    """Thin wrapper around Wan I2V pipeline matching handler interface."""

    def __init__(self, pipe):
        self.pipe = pipe
        self.name = "wan_i2v_5b"

    def generate_video(
        self,
        image_path: str,
        motion_prompt: str,
        duration_seconds: int = 5,
        resolution: str = "480p",
        output_path: str = "/tmp/output.mp4",
    ) -> None:
        width, height = RESOLUTION_MAP.get(resolution, RESOLUTION_MAP["480p"])
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
            num_frames=num_frames,
            width=width,
            height=height,
            guidance_scale=5.0,
            num_inference_steps=40,
        )

        frames = result.frames[0]
        export_to_video(frames, output_path, fps=DEFAULT_FPS)
        logger.info("Video saved to %s (%d frames)", output_path, len(frames))


def load_wan_model(cache_dir: str) -> WanModel:
    """Entry point called by handler via WAN_I2V_5B_BACKEND=wan:load_wan_model."""
    logger.info("Loading Wan I2V pipeline from %s (cache=%s)", MODEL_ID, cache_dir)
    pipe = WanImageToVideoPipeline.from_pretrained(
        MODEL_ID,
        cache_dir=cache_dir,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    pipe.enable_model_cpu_offload()
    logger.info("Wan I2V pipeline loaded with CPU offload")
    return WanModel(pipe)

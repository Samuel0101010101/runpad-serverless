"""Real-ESRGAN x4 backend for Tarik pipeline.

Provides load_model(cache_dir) -> RealESRGANModel with .upscale() interface
expected by handler.py's run_upscale().
"""

import logging
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

logger = logging.getLogger("tarik-handler.realesrgan")

MODEL_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
MODEL_FILENAME = "RealESRGAN_x4plus.pth"


def _ensure_weights(cache_dir: str) -> str:
    """Download model weights if not already cached."""
    weights_path = os.path.join(cache_dir, MODEL_FILENAME)
    if not os.path.exists(weights_path):
        logger.info("Downloading Real-ESRGAN weights to %s", weights_path)
        os.makedirs(cache_dir, exist_ok=True)
        import requests

        resp = requests.get(MODEL_URL, timeout=300, stream=True)
        resp.raise_for_status()
        with open(weights_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        logger.info("Weights downloaded: %s", weights_path)
    return weights_path


class RealESRGANModel:
    """Wrapper matching handler interface: .upscale(source_path, output_path, scale)."""

    def __init__(self, upsampler: RealESRGANer):
        self.upsampler = upsampler
        self.name = "realesrgan_x4"

    def _upscale_image(self, input_path: str, output_path: str, scale: int) -> None:
        img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Failed to read image: {input_path}")
        output, _ = self.upsampler.enhance(img, outscale=scale)
        cv2.imwrite(output_path, output)

    def upscale(self, source_path: str, output_path: str, scale: int = 4) -> None:
        """Upscale a video file frame-by-frame from 480p to 1080p."""
        source = Path(source_path)
        out = Path(output_path)

        # Extract frames
        frames_dir = source.parent / f"frames_{source.stem}"
        frames_dir.mkdir(exist_ok=True)
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(source),
                "-qscale:v", "2",
                str(frames_dir / "frame_%06d.png"),
            ],
            capture_output=True,
            check=True,
        )

        # Upscale each frame
        upscaled_dir = source.parent / f"upscaled_{source.stem}"
        upscaled_dir.mkdir(exist_ok=True)
        frame_files = sorted(frames_dir.glob("frame_*.png"))
        logger.info("Upscaling %d frames at scale=%d", len(frame_files), scale)

        for frame_file in frame_files:
            out_frame = upscaled_dir / frame_file.name
            self._upscale_image(str(frame_file), str(out_frame), scale)

        # Get original video fps
        probe = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate",
                "-of", "csv=p=0",
                str(source),
            ],
            capture_output=True,
            text=True,
        )
        fps_str = probe.stdout.strip()
        if "/" in fps_str:
            num, den = fps_str.split("/")
            fps = float(num) / float(den)
        else:
            fps = float(fps_str) if fps_str else 24.0

        # Re-encode upscaled frames to video
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", str(upscaled_dir / "frame_%06d.png"),
                "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                "-pix_fmt", "yuv420p",
                str(out),
            ],
            capture_output=True,
            check=True,
        )
        logger.info("Upscaled video saved to %s", out)


def load_model(cache_dir: str) -> RealESRGANModel:
    """Entry point: REALESRGAN_X4_BACKEND=realesrgan:load_model."""
    weights_path = _ensure_weights(cache_dir)

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(
        scale=4,
        model_path=weights_path,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=True,
        device="cuda",
    )
    logger.info("Real-ESRGAN x4 loaded on CUDA")
    return RealESRGANModel(upsampler)

"""Wav2Lip + GFPGAN backend for Tarik pipeline.

Provides load_model(cache_dir) -> Wav2LipWithGFPGAN with .lipsync() interface
expected by handler.py's run_lipsync().
"""

import logging
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

logger = logging.getLogger("tarik-handler.wav2lip")

WAV2LIP_REPO = "https://github.com/Rudrabha/Wav2Lip.git"
WAV2LIP_WEIGHTS_URL = "https://huggingface.co/camenduru/Wav2Lip/resolve/main/checkpoints/wav2lip_gan.pth"
WAV2LIP_WEIGHTS_FILENAME = "wav2lip_gan.pth"

GFPGAN_WEIGHTS_URL = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
GFPGAN_WEIGHTS_FILENAME = "GFPGANv1.4.pth"


def _ensure_wav2lip_repo(cache_dir: str) -> str:
    """Clone Wav2Lip repo if needed and return its path."""
    repo_dir = os.path.join(cache_dir, "Wav2Lip")
    if not os.path.exists(os.path.join(repo_dir, "inference.py")):
        logger.info("Cloning Wav2Lip repo to %s", repo_dir)
        subprocess.run(
            ["git", "clone", WAV2LIP_REPO, repo_dir],
            check=True,
            capture_output=True,
        )
    return repo_dir


def _download_file(url: str, dest: str) -> None:
    if os.path.exists(dest):
        return
    logger.info("Downloading %s → %s", url[:80], dest)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    import requests

    resp = requests.get(url, timeout=600, stream=True, allow_redirects=True)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)


def _ensure_weights(cache_dir: str) -> tuple:
    wav2lip_path = os.path.join(cache_dir, WAV2LIP_WEIGHTS_FILENAME)
    gfpgan_path = os.path.join(cache_dir, GFPGAN_WEIGHTS_FILENAME)
    _download_file(WAV2LIP_WEIGHTS_URL, wav2lip_path)
    _download_file(GFPGAN_WEIGHTS_URL, gfpgan_path)
    return wav2lip_path, gfpgan_path


class Wav2LipWithGFPGAN:
    """Wrapper matching handler interface: .lipsync(video_path, audio_path, output_path)."""

    def __init__(self, wav2lip_repo: str, wav2lip_weights: str, gfpgan_weights: str):
        self.wav2lip_repo = wav2lip_repo
        self.wav2lip_weights = wav2lip_weights
        self.gfpgan_weights = gfpgan_weights
        self.name = "wav2lip_gfpgan"

        # Lazy-load GFPGAN restorer
        self._gfpgan_restorer = None

    def _get_gfpgan(self):
        if self._gfpgan_restorer is None:
            from gfpgan import GFPGANer

            self._gfpgan_restorer = GFPGANer(
                model_path=self.gfpgan_weights,
                upscale=1,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=None,
            )
        return self._gfpgan_restorer

    def lipsync(self, video_path: str, audio_path: str, output_path: str) -> None:
        """Run Wav2Lip inference, then GFPGAN face restoration on output."""
        raw_output = output_path + ".raw.mp4"

        # Step 1: Wav2Lip inference via subprocess (its imports are messy)
        cmd = [
            sys.executable,
            os.path.join(self.wav2lip_repo, "inference.py"),
            "--checkpoint_path", self.wav2lip_weights,
            "--face", video_path,
            "--audio", audio_path,
            "--outfile", raw_output,
            "--nosmooth",
        ]
        logger.info("Running Wav2Lip inference")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.wav2lip_repo)
        if result.returncode != 0:
            raise RuntimeError(f"Wav2Lip failed: {result.stderr[-2000:]}")

        # Step 2: GFPGAN face restoration frame-by-frame
        logger.info("Applying GFPGAN face restoration")
        self._enhance_faces(raw_output, output_path)
        logger.info("Lip-synced video saved to %s", output_path)

    def _enhance_faces(self, input_video: str, output_video: str) -> None:
        """Read video, enhance each frame with GFPGAN, write output."""
        cap = cv2.VideoCapture(input_video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

        restorer = self._get_gfpgan()
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            _, _, restored = restorer.enhance(frame, has_aligned=False, only_center_face=True, paste_back=True)
            writer.write(restored)
            frame_count += 1

        cap.release()
        writer.release()
        logger.info("GFPGAN enhanced %d frames", frame_count)

        # Re-mux with ffmpeg for proper mp4
        temp = output_video + ".tmp.mp4"
        os.rename(output_video, temp)
        subprocess.run(
            ["ffmpeg", "-y", "-i", temp, "-c:v", "libx264", "-preset", "fast", "-crf", "20", output_video],
            capture_output=True,
            check=True,
        )
        os.remove(temp)


def load_model(cache_dir: str) -> Wav2LipWithGFPGAN:
    """Entry point: WAV2LIP_GFPGAN_BACKEND=wav2lip:load_model."""
    wav2lip_repo = _ensure_wav2lip_repo(cache_dir)
    wav2lip_weights, gfpgan_weights = _ensure_weights(cache_dir)
    logger.info("Wav2Lip + GFPGAN ready (repo=%s)", wav2lip_repo)
    return Wav2LipWithGFPGAN(wav2lip_repo, wav2lip_weights, gfpgan_weights)

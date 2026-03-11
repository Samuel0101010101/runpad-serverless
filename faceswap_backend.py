"""InsightFace face-swap backend for Tarik pipeline.

Provides load_model(cache_dir) -> FaceSwapper with .swap_face() interface
expected by handler.py's process_face_swap().

Uses insightface + onnxruntime-gpu for face detection/recognition,
and the inswapper_128 model for high-quality single-face replacement.
"""

import logging
import os
import shutil
import subprocess
import uuid
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger("tarik-handler.faceswap")

INSWAPPER_URL = "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"


class FaceSwapper:
    """Wraps insightface for frame-by-frame face swap on video."""

    def __init__(self, app, swapper):
        self.app = app
        self.swapper = swapper
        self.name = "faceswap"

    def swap_face(
        self,
        video_path: str,
        reference_face_path: str,
        output_path: str,
    ) -> None:
        """Replace all faces in video with the reference face."""
        # Get reference face embedding
        ref_img = cv2.imread(reference_face_path)
        if ref_img is None:
            raise ValueError(f"Could not read reference face image: {reference_face_path}")

        ref_faces = self.app.get(ref_img)
        if not ref_faces:
            raise ValueError("No face detected in reference image")
        # Use the largest face in the reference image
        ref_face = max(ref_faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

        # Process video frame by frame
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 24
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(
            "Processing %d frames (%dx%d @ %.1ffps)",
            total_frames, width, height, fps,
        )

        # Write to temp file, then mux audio from original
        tmp_noaudio = output_path + ".noaudio.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(tmp_noaudio, fourcc, fps, (width, height))

        frame_idx = 0
        swapped_count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_idx += 1

                faces = self.app.get(frame)
                if faces:
                    for face in faces:
                        frame = self.swapper.get(frame, face, ref_face, paste_back=True)
                    swapped_count += 1

                writer.write(frame)

                if frame_idx % 24 == 0:
                    logger.info("Frame %d/%d (%d swapped)", frame_idx, total_frames, swapped_count)
        finally:
            cap.release()
            writer.release()

        logger.info("Swapped faces in %d/%d frames", swapped_count, frame_idx)

        # Mux audio from original video back in
        _mux_audio(video_path, tmp_noaudio, output_path)

        # Cleanup temp
        if os.path.exists(tmp_noaudio):
            os.remove(tmp_noaudio)


def _mux_audio(original_video: str, processed_video: str, output_path: str) -> None:
    """Copy audio from original and video from processed into output."""
    cmd = [
        "ffmpeg", "-y",
        "-i", processed_video,
        "-i", original_video,
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-c:a", "aac",
        "-shortest",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning("FFmpeg mux failed, falling back to video-only: %s", result.stderr[-500:])
        # Fallback: just re-encode video without audio
        shutil.move(processed_video, output_path)


def _download_inswapper(cache_dir: str) -> str:
    """Download inswapper_128.onnx if not cached."""
    model_path = os.path.join(cache_dir, "inswapper_128.onnx")
    if os.path.exists(model_path):
        logger.info("inswapper_128.onnx found in cache")
        return model_path

    logger.info("Downloading inswapper_128.onnx...")
    import urllib.request
    os.makedirs(cache_dir, exist_ok=True)
    urllib.request.urlretrieve(INSWAPPER_URL, model_path)
    logger.info("inswapper_128.onnx downloaded to %s", model_path)
    return model_path


def load_model(cache_dir: str) -> FaceSwapper:
    """Entry point: FACESWAP_BACKEND=faceswap_backend:load_model."""
    import insightface
    from insightface.app import FaceAnalysis

    logger.info("Loading InsightFace FaceAnalysis (cache=%s)", cache_dir)
    app = FaceAnalysis(
        name="buffalo_l",
        root=cache_dir,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(640, 640))

    inswapper_path = _download_inswapper(cache_dir)
    swapper = insightface.model_zoo.get_model(inswapper_path, providers=["CUDAExecutionProvider"])

    logger.info("FaceSwapper loaded on CUDA")
    return FaceSwapper(app, swapper)

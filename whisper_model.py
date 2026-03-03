"""Whisper Large v3 backend for Tarik pipeline.

Provides load_model(cache_dir) -> WhisperTranscriber with .transcribe() interface
expected by handler.py's run_transcribe().
"""

import logging
import os
from pathlib import Path

import whisper

logger = logging.getLogger("tarik-handler.whisper")

MODEL_NAME = "large-v3"


class WhisperTranscriber:
    """Wrapper matching handler interface: .transcribe(audio_path, srt_output_path) -> str."""

    def __init__(self, model):
        self.model = model
        self.name = "whisper_large_v3"

    def transcribe(self, audio_path: str, srt_output_path: str) -> str:
        """Transcribe audio to SRT and return plain text."""
        logger.info("Transcribing %s", audio_path)
        result = self.model.transcribe(audio_path, task="transcribe", verbose=False)

        segments = result.get("segments", [])
        srt_lines = []
        full_text_parts = []

        for idx, seg in enumerate(segments, 1):
            start = self._format_timestamp(seg["start"])
            end = self._format_timestamp(seg["end"])
            text = seg["text"].strip()
            srt_lines.append(f"{idx}")
            srt_lines.append(f"{start} --> {end}")
            srt_lines.append(text)
            srt_lines.append("")
            full_text_parts.append(text)

        srt_content = "\n".join(srt_lines)
        Path(srt_output_path).write_text(srt_content, encoding="utf-8")
        logger.info("SRT saved to %s (%d segments)", srt_output_path, len(segments))

        return " ".join(full_text_parts)

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        """Convert seconds to SRT timestamp format: HH:MM:SS,mmm."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def load_model(cache_dir: str) -> WhisperTranscriber:
    """Entry point: WHISPER_LARGE_V3_BACKEND=whisper_model:load_model."""
    download_root = os.path.join(cache_dir, "whisper")
    os.makedirs(download_root, exist_ok=True)
    logger.info("Loading Whisper %s (cache=%s)", MODEL_NAME, download_root)
    model = whisper.load_model(MODEL_NAME, download_root=download_root, device="cuda")
    logger.info("Whisper %s loaded on CUDA", MODEL_NAME)
    return WhisperTranscriber(model)

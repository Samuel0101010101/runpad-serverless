"""AudioCraft MusicGen + AudioGen backends for Tarik pipeline.

Provides:
  load_music_model(cache_dir) -> MusicGenModel with .generate_music()
  load_sfx_model(cache_dir)   -> AudioGenModel with .generate_sfx()

Matches interfaces expected by handler.py's run_music() and run_sfx().
"""

import logging
import os

import soundfile as sf
import torch
from audiocraft.models import MusicGen, AudioGen

logger = logging.getLogger("tarik-handler.audiocraft")

MUSICGEN_MODEL_ID = "facebook/musicgen-large"
AUDIOGEN_MODEL_ID = "facebook/audiogen-medium"


class MusicGenModel:
    """Wrapper matching handler interface: .generate_music(prompt, duration_seconds, output_path)."""

    def __init__(self, model: MusicGen):
        self.model = model
        self.name = "musicgen_large"

    def generate_music(self, prompt: str, duration_seconds: int = 30, output_path: str = "/tmp/music.wav") -> None:
        logger.info("Generating %ds music: %r", duration_seconds, prompt[:80])
        self.model.set_generation_params(duration=duration_seconds)
        wav = self.model.generate([prompt])  # shape: (1, channels, samples)
        audio = wav[0].cpu()
        # soundfile expects (samples, channels)
        sf.write(output_path, audio.numpy().T, samplerate=32000)
        logger.info("Music saved to %s", output_path)


class AudioGenModel:
    """Wrapper matching handler interface: .generate_sfx(prompt, output_path)."""

    def __init__(self, model: AudioGen):
        self.model = model
        self.name = "audiogen"

    def generate_sfx(self, prompt: str, output_path: str = "/tmp/sfx.wav") -> None:
        logger.info("Generating SFX: %r", prompt[:80])
        self.model.set_generation_params(duration=5)
        wav = self.model.generate([prompt])  # shape: (1, channels, samples)
        audio = wav[0].cpu()
        sf.write(output_path, audio.numpy().T, samplerate=16000)
        logger.info("SFX saved to %s", output_path)


def load_music_model(cache_dir: str) -> MusicGenModel:
    """Entry point: MUSICGEN_LARGE_BACKEND=audiocraft_backend:load_music_model."""
    os.environ.setdefault("AUDIOCRAFT_CACHE_DIR", cache_dir)
    logger.info("Loading MusicGen Large (cache=%s)", cache_dir)
    model = MusicGen.get_pretrained(MUSICGEN_MODEL_ID, device="cuda")
    logger.info("MusicGen Large loaded on CUDA")
    return MusicGenModel(model)


def load_sfx_model(cache_dir: str) -> AudioGenModel:
    """Entry point: AUDIOGEN_BACKEND=audiocraft_backend:load_sfx_model."""
    os.environ.setdefault("AUDIOCRAFT_CACHE_DIR", cache_dir)
    logger.info("Loading AudioGen Medium (cache=%s)", cache_dir)
    model = AudioGen.get_pretrained(AUDIOGEN_MODEL_ID, device="cuda")
    logger.info("AudioGen Medium loaded on CUDA")
    return AudioGenModel(model)

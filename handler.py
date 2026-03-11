import gc
import importlib
import json
import logging
import os
import subprocess
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Disable HuggingFace Xet and hf_transfer downloaders BEFORE any HF imports.
# Both crash on large sharded models ("receiver dropped").
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import requests
import runpod

try:
    import boto3
except Exception:  # pragma: no cover - runtime dependency in RunPod image
    boto3 = None

try:
    import torch
except Exception as _torch_err:  # pragma: no cover - runtime dependency in RunPod image
    torch = None
    # Log import failure so we can diagnose CUDA issues on workers
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    logging.getLogger("tarik-handler").error(
        "torch import failed: %s", _torch_err
    )


logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("tarik-handler")


# Use /workspace/models if a network volume is mounted (has >50GB free),
# otherwise fall back to /tmp/models to avoid "No space left on device".
def _is_volume_mounted() -> bool:
    """Check if /workspace is a real volume (different device from /)."""
    try:
        root_dev = os.stat("/").st_dev
        ws_dev = os.stat("/workspace").st_dev
        return ws_dev != root_dev
    except OSError:
        return False


def _pick_cache_dir() -> Path:
    if _is_volume_mounted():
        ws = Path("/workspace/models")
        ws.mkdir(parents=True, exist_ok=True)
        logger.info("Network volume detected at /workspace")
        return ws
    logger.warning("No network volume detected — falling back to /tmp/models")
    fallback = Path("/tmp/models")
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


MODEL_CACHE_DIR = _pick_cache_dir()
TMP_DIR = Path("/tmp")
TMP_DIR.mkdir(parents=True, exist_ok=True)

# Point all temp/cache dirs to the volume so large model loads don't
# fill up the tiny container root disk.
OFFLOAD_DIR = MODEL_CACHE_DIR / "offload"
OFFLOAD_DIR.mkdir(parents=True, exist_ok=True)
if str(MODEL_CACHE_DIR).startswith("/workspace"):
    os.environ.setdefault("TMPDIR", str(MODEL_CACHE_DIR / "tmp"))
    os.environ.setdefault("TORCH_HOME", str(MODEL_CACHE_DIR / "torch"))
    Path(os.environ["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["TORCH_HOME"]).mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(MODEL_CACHE_DIR / "hf"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(MODEL_CACHE_DIR / "transformers"))
logger.info("Model cache: %s", MODEL_CACHE_DIR)


WAN = "wan_ti2v_5b"
REALESRGAN = "realesrgan_x4"
WAV2LIP = "wav2lip_gfpgan"
WHISPER = "whisper_large_v3"
MUSICGEN = "musicgen_large"
AUDIOGEN = "audiogen"
FACESWAP = "faceswap"


MODEL_VRAM_HINTS_GB = {
    WAN: 20,
    REALESRGAN: 2,
    WAV2LIP: 8,
    WHISPER: 10,
    MUSICGEN: 4,
    AUDIOGEN: 3,
    FACESWAP: 2,
}


MODELS: Dict[str, Any] = {
    WAN: None,
    REALESRGAN: None,
    WAV2LIP: None,
    WHISPER: None,
    MUSICGEN: None,
    AUDIOGEN: None,
    FACESWAP: None,
}


class ModelLoadError(RuntimeError):
    pass


@dataclass
class StepResult:
    output_urls: List[str]
    credits_used: int
    failed_indices: Optional[List[int]] = None
    payload: Optional[Dict[str, Any]] = None


def is_cuda_oom(error: Exception) -> bool:
    text = str(error).lower()
    return "cuda out of memory" in text or "out of memory" in text


def log_vram_usage(context: str) -> None:
    if torch is None or not torch.cuda.is_available():
        logger.info("[%s] CUDA unavailable; VRAM usage not reported.", context)
        return
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    logger.info(
        "[%s] VRAM allocated=%.2fGB reserved=%.2fGB", context, allocated, reserved
    )


def unload_all_models(except_name: Optional[str] = None) -> None:
    for model_name in MODELS:
        if model_name == except_name:
            continue
        if MODELS[model_name] is not None:
            logger.info("Unloading model: %s", model_name)
            MODELS[model_name] = None
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
    log_vram_usage("after-unload")


class MockModel:
    def __init__(self, name: str):
        self.name = name


def _load_backend_or_mock(model_name: str) -> Any:
    env_key = f"{model_name.upper()}_BACKEND"
    backend_path = os.getenv(env_key, "").strip()
    if not backend_path:
        logger.warning(
            "No backend configured for %s (%s). Falling back to MockModel.",
            model_name,
            env_key,
        )
        return MockModel(model_name)

    module_name, sep, attr_name = backend_path.partition(":")
    if not sep:
        raise ModelLoadError(
            f"Invalid backend format for {env_key}: {backend_path}. Expected module:function"
        )
    module = importlib.import_module(module_name)
    loader = getattr(module, attr_name)
    return loader(cache_dir=str(MODEL_CACHE_DIR))


def load_model(name: str) -> Any:
    if name not in MODELS:
        raise ModelLoadError(f"Unknown model: {name}")

    unload_all_models(except_name=name)
    if MODELS[name] is None:
        logger.info(
            "Loading model: %s (~%sGB VRAM)", name, MODEL_VRAM_HINTS_GB.get(name, "?")
        )
        try:
            MODELS[name] = _load_backend_or_mock(name)
        except Exception as exc:
            unload_all_models()
            raise ModelLoadError(f"Failed to load model '{name}': {exc}") from exc
        log_vram_usage(f"after-load-{name}")
    else:
        logger.info("Model already loaded: %s", name)
        log_vram_usage(f"reuse-{name}")

    return MODELS[name]


def _get_r2_client():
    if boto3 is None:
        raise RuntimeError("boto3 is required for R2 upload support")

    endpoint_url = os.getenv("R2_ENDPOINT_URL")
    access_key = os.getenv("R2_ACCESS_KEY_ID")
    secret_key = os.getenv("R2_SECRET_ACCESS_KEY")
    if not endpoint_url or not access_key or not secret_key:
        raise RuntimeError("R2 credentials are not configured")

    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=os.getenv("R2_REGION", "auto"),
    )


def download_to_tmp(url: str, suffix: str) -> Path:
    file_name = f"{uuid.uuid4().hex}_{suffix}"
    target = TMP_DIR / file_name
    with requests.get(url, headers={"User-Agent": "Tarik-RunPod/1.0"}, timeout=120, stream=True) as response:
        response.raise_for_status()
        with target.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    return target


def upload_file_to_r2(path: Path, key_prefix: str) -> str:
    bucket = os.getenv("R2_BUCKET")
    if not bucket:
        raise RuntimeError("R2_BUCKET is not configured")

    key = f"{key_prefix}/{uuid.uuid4().hex}_{path.name}"
    s3 = _get_r2_client()
    s3.upload_file(str(path), bucket, key)

    # Prefer public URL if R2_PUBLIC_URL is set (presigned URLs 403 on R2)
    public_base = os.getenv("R2_PUBLIC_URL", "").rstrip("/")
    if public_base:
        return f"{public_base}/{key}"

    url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=int(os.getenv("R2_SIGNED_URL_TTL_SEC", "86400")),
    )
    return url


def write_mock_binary(path: Path, label: str, meta: Dict[str, Any]) -> None:
    payload = {"label": label, "meta": meta}
    path.write_bytes(json.dumps(payload).encode("utf-8"))


def run_generate_video(model: Any, image_path: Path, motion_prompt: str, output_path: Path) -> None:
    if hasattr(model, "generate_video"):
        model.generate_video(
            image_path=str(image_path),
            motion_prompt=motion_prompt,
            duration_seconds=5,
            resolution="720p",
            output_path=str(output_path),
        )
        return
    write_mock_binary(
        output_path,
        "generate_video",
        {
            "image": str(image_path),
            "motion_prompt": motion_prompt,
            "duration_seconds": 5,
            "resolution": "720p",
            "model": getattr(model, "name", "unknown"),
        },
    )


def run_upscale(model: Any, source_path: Path, output_path: Path) -> None:
    if hasattr(model, "upscale"):
        model.upscale(source_path=str(source_path), output_path=str(output_path), scale=4)
        return
    write_mock_binary(
        output_path,
        "upscale",
        {
            "source": str(source_path),
            "scale": "480p->1080p",
            "model": getattr(model, "name", "unknown"),
        },
    )


def run_lipsync(model: Any, video_path: Path, audio_path: Path, output_path: Path) -> None:
    if hasattr(model, "lipsync"):
        model.lipsync(video_path=str(video_path), audio_path=str(audio_path), output_path=str(output_path))
        return
    write_mock_binary(
        output_path,
        "lipsync",
        {
            "video": str(video_path),
            "audio": str(audio_path),
            "model": getattr(model, "name", "unknown"),
        },
    )


def run_transcribe(model: Any, audio_path: Path, srt_path: Path) -> str:
    if hasattr(model, "transcribe"):
        return model.transcribe(audio_path=str(audio_path), srt_output_path=str(srt_path))
    srt = "1\n00:00:00,000 --> 00:00:01,500\n[Mock transcription]\n"
    srt_path.write_text(srt, encoding="utf-8")
    return "[Mock transcription]"


def run_music(model: Any, prompt: str, duration_seconds: int, output_path: Path) -> None:
    if hasattr(model, "generate_music"):
        model.generate_music(prompt=prompt, duration_seconds=duration_seconds, output_path=str(output_path))
        return
    write_mock_binary(
        output_path,
        "music",
        {"prompt": prompt, "duration_seconds": duration_seconds, "model": getattr(model, "name", "unknown")},
    )


def run_sfx(model: Any, prompt: str, duration_seconds: int, output_path: Path) -> None:
    if hasattr(model, "generate_sfx"):
        model.generate_sfx(prompt=prompt, duration_seconds=duration_seconds, output_path=str(output_path))
        return
    write_mock_binary(
        output_path,
        "sfx",
        {"prompt": prompt, "duration_seconds": duration_seconds, "model": getattr(model, "name", "unknown")},
    )


def run_ffmpeg(command: List[str]) -> None:
    logger.info("Running FFmpeg command: %s", " ".join(command))
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        stderr_tail = (completed.stderr or "")[-4000:]
        raise RuntimeError(f"FFmpeg failed (code {completed.returncode}): {stderr_tail}")


def normalize_audio_track_urls(job_input: Dict[str, Any]) -> List[str]:
    audio_urls: List[str] = []
    audio_tracks = job_input.get("audio_tracks", {})

    if isinstance(audio_tracks, dict):
        for key in ("dialogue", "music", "background_music"):
            value = audio_tracks.get(key)
            if isinstance(value, str) and value:
                audio_urls.append(value)
        sfx_values = audio_tracks.get("sfx")
        if isinstance(sfx_values, str) and sfx_values:
            audio_urls.append(sfx_values)
        elif isinstance(sfx_values, list):
            audio_urls.extend([item for item in sfx_values if isinstance(item, str) and item])
    elif isinstance(audio_tracks, list):
        audio_urls.extend([item for item in audio_tracks if isinstance(item, str) and item])

    for key in (
        "dialogue_audio_url",
        "music_audio_url",
        "background_music_url",
        "audio_url",
    ):
        value = job_input.get(key)
        if isinstance(value, str) and value:
            audio_urls.append(value)

    sfx_urls = job_input.get("sfx_audio_urls", [])
    if isinstance(sfx_urls, str) and sfx_urls:
        audio_urls.append(sfx_urls)
    elif isinstance(sfx_urls, list):
        audio_urls.extend([item for item in sfx_urls if isinstance(item, str) and item])

    deduped: List[str] = []
    seen = set()
    for url in audio_urls:
        if url not in seen:
            seen.add(url)
            deduped.append(url)
    return deduped


def ffmpeg_subtitles_filter_part(srt_path: Optional[Path]) -> str:
    if srt_path is None:
        return ""
    subtitle_path = str(srt_path).replace("\\", "\\\\").replace(":", "\\:")
    return f",subtitles={subtitle_path}"


def response(
    status: str,
    output_urls: Optional[List[str]] = None,
    credits_used: int = 0,
    error: Optional[str] = None,
    retry_recommended: bool = False,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    urls = output_urls or []
    payload = {
        "status": status,
        "output_urls": urls,
        "output_url": urls[0] if urls else None,
        "credits_used": credits_used,
        "error": error,
        "retry_recommended": retry_recommended,
    }
    if extra:
        payload.update(extra)
    return payload


def process_generate_video(job_input: Dict[str, Any]) -> StepResult:
    # Fail fast if no GPU — Wan TI2V-5B requires CUDA
    if torch is None or not torch.cuda.is_available():
        raise ModelLoadError(
            "CUDA is not available on this worker. "
            "generate_video requires a GPU. "
            f"torch={'missing' if torch is None else 'loaded'}, "
            f"nvidia-smi: {_nvidia_smi_summary()}"
        )
    image_url = job_input["image_url"]
    motion_prompt = job_input.get("motion_prompt", "")
    model = load_model(WAN)

    image_path = download_to_tmp(image_url, "input_image")
    output_path = TMP_DIR / "output.mp4"
    run_generate_video(model, image_path, motion_prompt, output_path)
    output_url = upload_file_to_r2(output_path, "tarik/generate_video")
    return StepResult(output_urls=[output_url], credits_used=30)


def _upscale_batch(
    model: Any, clip_urls: List[str], start_index: int
) -> Tuple[List[str], List[int]]:
    output_urls: List[str] = []
    failed_indices: List[int] = []
    for idx, clip_url in enumerate(clip_urls):
        absolute_idx = start_index + idx
        try:
            source_path = download_to_tmp(clip_url, f"clip_{absolute_idx}.mp4")
            output_path = TMP_DIR / f"upscaled_{absolute_idx}.mp4"
            run_upscale(model, source_path, output_path)
            output_urls.append(upload_file_to_r2(output_path, "tarik/upscale"))
        except Exception as exc:
            if is_cuda_oom(exc):
                raise
            logger.exception("Upscale failed for index %s: %s", absolute_idx, exc)
            failed_indices.append(absolute_idx)
    return output_urls, failed_indices


def process_upscale(job_input: Dict[str, Any]) -> StepResult:
    # Accept single video_url (Lovable) or batch clips (internal)
    clip_urls = list(job_input.get("clips", []))
    if not clip_urls and job_input.get("video_url"):
        clip_urls = [job_input["video_url"]]
    if not clip_urls:
        return StepResult(output_urls=[], credits_used=0)

    initial_batch_size = max(1, int(job_input.get("batch_size", 4)))
    model = load_model(REALESRGAN)

    output_urls: List[str] = []
    failed_indices: List[int] = []
    pointer = 0
    batch_size = initial_batch_size

    while pointer < len(clip_urls):
        batch = clip_urls[pointer : pointer + batch_size]
        try:
            urls, failed = _upscale_batch(model, batch, pointer)
            output_urls.extend(urls)
            failed_indices.extend(failed)
            pointer += len(batch)
        except Exception as exc:
            if not is_cuda_oom(exc):
                raise
            logger.warning("CUDA OOM during upscale batch. Retrying with smaller batch.")
            unload_all_models()
            batch_size = max(1, batch_size // 2)
            model = load_model(REALESRGAN)
            if batch_size == 1:
                logger.warning("Upscale already at batch_size=1 after OOM.")

    return StepResult(
        output_urls=output_urls,
        credits_used=len(output_urls) * 3,
        failed_indices=failed_indices,
    )


def process_lipsync(job_input: Dict[str, Any]) -> StepResult:
    # Accept single video_url (Lovable) or batch clips (internal)
    clip_urls = list(job_input.get("clips", []))
    if not clip_urls and job_input.get("video_url"):
        clip_urls = [job_input["video_url"]]
    audio_url = job_input["audio_url"]
    model = load_model(WAV2LIP)
    audio_path = download_to_tmp(audio_url, "dialog_audio")

    output_urls: List[str] = []
    failed_indices: List[int] = []
    errors: List[str] = []
    for idx, clip_url in enumerate(clip_urls):
        try:
            clip_path = download_to_tmp(clip_url, f"scene_{idx}.mp4")
            output_path = TMP_DIR / f"lipsynced_{idx}.mp4"
            run_lipsync(model, clip_path, audio_path, output_path)
            output_urls.append(upload_file_to_r2(output_path, "tarik/lipsync"))
        except Exception as exc:
            logger.exception("Lipsync failed for clip index %s: %s", idx, exc)
            failed_indices.append(idx)
            errors.append(f"clip_{idx}: {str(exc)[-500:]}")

    return StepResult(
        output_urls=output_urls,
        credits_used=len(output_urls) * 4,
        failed_indices=failed_indices,
        payload={"errors": errors} if errors else None,
    )


def _parse_srt_segments(srt_path: Path) -> List[Dict[str, Any]]:
    """Parse SRT file into [{start, end, text}] segments for Lovable."""
    segments: List[Dict[str, Any]] = []
    content = srt_path.read_text(encoding="utf-8").strip()
    if not content:
        return segments
    blocks = content.split("\n\n")
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue
        timestamp_line = lines[1]
        text = " ".join(lines[2:])
        parts = timestamp_line.split(" --> ")
        if len(parts) != 2:
            continue
        segments.append({
            "start": _srt_ts_to_seconds(parts[0].strip()),
            "end": _srt_ts_to_seconds(parts[1].strip()),
            "text": text,
        })
    return segments


def _srt_ts_to_seconds(ts: str) -> float:
    """Convert SRT timestamp HH:MM:SS,mmm to float seconds."""
    ts = ts.replace(",", ".")
    parts = ts.split(":")
    return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])


def process_transcribe(job_input: Dict[str, Any]) -> StepResult:
    audio_url = job_input["audio_url"]
    model = load_model(WHISPER)
    audio_path = download_to_tmp(audio_url, "transcribe_audio")

    srt_path = TMP_DIR / f"transcript_{uuid.uuid4().hex}.srt"
    transcript_text = run_transcribe(model, audio_path, srt_path)
    srt_url = upload_file_to_r2(srt_path, "tarik/transcribe")
    segments = _parse_srt_segments(srt_path)
    return StepResult(
        output_urls=[srt_url],
        credits_used=2,
        payload={
            "text": transcript_text,
            "subtitle_text": transcript_text,
            "segments": segments,
            "srt_url": srt_url,
        },
    )


def process_generate_music(job_input: Dict[str, Any]) -> StepResult:
    prompt = job_input["prompt"]
    duration_seconds = int(job_input.get("duration_sec", job_input.get("duration_seconds", 30)))
    model = load_model(MUSICGEN)
    output_path = TMP_DIR / f"music_{uuid.uuid4().hex}.wav"
    run_music(model, prompt, duration_seconds, output_path)
    output_url = upload_file_to_r2(output_path, "tarik/music")
    return StepResult(output_urls=[output_url], credits_used=6)


def process_generate_sfx(job_input: Dict[str, Any]) -> StepResult:
    # Accept single prompt (Lovable) or batch prompts (internal)
    prompts = list(job_input.get("prompts", []))
    if not prompts and job_input.get("prompt"):
        prompts = [job_input["prompt"]]
    duration_seconds = int(job_input.get("duration_sec", job_input.get("duration_seconds", 5)))
    model = load_model(AUDIOGEN)

    output_urls: List[str] = []
    failed_indices: List[int] = []
    for idx, prompt in enumerate(prompts):
        try:
            output_path = TMP_DIR / f"sfx_{idx}_{uuid.uuid4().hex}.wav"
            run_sfx(model, prompt, duration_seconds, output_path)
            output_urls.append(upload_file_to_r2(output_path, "tarik/sfx"))
        except Exception as exc:
            logger.exception("SFX generation failed for index %s: %s", idx, exc)
            failed_indices.append(idx)

    return StepResult(
        output_urls=output_urls,
        credits_used=len(output_urls),
        failed_indices=failed_indices,
    )


ASSEMBLE_FORMAT_SPECS = {
    "youtube_16x9": (1920, 1080),
    "tiktok_9x16": (1080, 1920),
    "instagram_1x1": (1080, 1080),
    "reel_4x5": (1080, 1350),
    # Lovable-friendly aliases
    "16:9": (1920, 1080),
    "9:16": (1080, 1920),
    "1:1": (1080, 1080),
    "4:5": (1080, 1350),
    "youtube": (1920, 1080),
    "tiktok": (1080, 1920),
    "instagram": (1080, 1080),
    "reels": (1080, 1350),
}

ALL_DEFAULT_EXPORTS = [
    ("youtube_16x9", 1920, 1080),
    ("tiktok_9x16", 1080, 1920),
    ("instagram_1x1", 1080, 1080),
    ("reel_4x5", 1080, 1350),
]


def _collect_audio_urls_from_scenes(scenes: List[Dict[str, Any]]) -> List[str]:
    """Extract all unique audio track URLs from Lovable-style scenes array."""
    urls: List[str] = []
    seen: set = set()
    for scene in scenes:
        audio_tracks = scene.get("audio_tracks", {})
        if isinstance(audio_tracks, dict):
            for val in audio_tracks.values():
                if isinstance(val, str) and val and val not in seen:
                    seen.add(val)
                    urls.append(val)
                elif isinstance(val, list):
                    for item in val:
                        if isinstance(item, str) and item and item not in seen:
                            seen.add(item)
                            urls.append(item)
        elif isinstance(audio_tracks, list):
            for item in audio_tracks:
                if isinstance(item, str) and item and item not in seen:
                    seen.add(item)
                    urls.append(item)
    return urls


def _collect_subtitle_url_from_scenes(scenes: List[Dict[str, Any]]) -> Optional[str]:
    """Extract first subtitle URL from scenes array."""
    for scene in scenes:
        sub = scene.get("subtitles")
        if isinstance(sub, str) and sub:
            return sub
    return None


def _build_atempo_filter(speed: float) -> str:
    """Build FFmpeg atempo filter chain for a given speed multiplier.

    atempo accepts 0.5..100.0 per stage, so values below 0.5
    require chaining (e.g., 0.25x = atempo=0.5,atempo=0.5).
    """
    if speed >= 0.5:
        return f"atempo={speed}"
    # Chain multiple atempo stages for very slow speeds
    parts: List[str] = []
    remaining = speed
    while remaining < 0.5:
        parts.append("atempo=0.5")
        remaining /= 0.5
    parts.append(f"atempo={remaining:.4f}")
    return ",".join(parts)


def process_assemble(job_input: Dict[str, Any]) -> StepResult:
    """Assemble video clips with audio tracks.

    Accepts two input formats:
      Lovable format:   scenes=[{video_url, audio_tracks, subtitles}], format="youtube_16x9"
      Legacy format:    clips=[url, ...], audio_tracks={music: url, sfx: url}

    If 'format' is specified, exports only that format and returns video_url.
    If 'format' is omitted, exports all 4 formats and returns exports dict.
    """
    unload_all_models()

    # --- Normalize inputs: Lovable scenes vs legacy clips ---
    scenes = job_input.get("scenes", [])
    clip_urls: List[str] = []
    audio_urls: List[str] = []
    subtitle_url: Optional[str] = None

    if scenes:
        # Lovable format: scenes=[{video_url, audio_tracks, subtitles}]
        for scene in scenes:
            vurl = scene.get("video_url")
            if vurl:
                clip_urls.append(vurl)
        audio_urls = _collect_audio_urls_from_scenes(scenes)
        subtitle_url = _collect_subtitle_url_from_scenes(scenes)
    else:
        # Legacy format: clips=[...], audio_tracks={...}
        clip_urls = list(job_input.get("clips", []))
        audio_urls = normalize_audio_track_urls(job_input)
        subtitle_url = job_input.get("subtitle_srt_url") or job_input.get("srt_url")

    if not clip_urls:
        raise ValueError("assemble requires non-empty 'clips' or 'scenes' list")

    # --- Per-scene speed values (from scenes array or global default) ---
    scene_speeds: List[float] = []
    if scenes:
        for scene in scenes:
            speed = float(scene.get("speed", 1.0))
            speed = max(0.25, min(4.0, speed))  # clamp to safe range
            scene_speeds.append(speed)
    else:
        scene_speeds = [1.0] * len(clip_urls)

    # --- Download and optionally speed-adjust clips ---
    clip_paths: List[Path] = []
    for idx, clip_url in enumerate(clip_urls):
        raw_path = download_to_tmp(clip_url, f"assemble_clip_{idx}.mp4")
        speed = scene_speeds[idx] if idx < len(scene_speeds) else 1.0
        if speed != 1.0:
            adjusted_path = TMP_DIR / f"speed_{speed}x_{idx}_{uuid.uuid4().hex}.mp4"
            vf_speed = f"setpts=PTS/{speed}"
            af_speed = _build_atempo_filter(speed)
            speed_cmd = [
                "ffmpeg", "-y", "-i", str(raw_path),
                "-vf", vf_speed,
                "-af", af_speed,
                "-c:v", "libx264", "-preset", "medium", "-crf", "20",
                "-c:a", "aac",
                str(adjusted_path),
            ]
            run_ffmpeg(speed_cmd)
            clip_paths.append(adjusted_path)
        else:
            clip_paths.append(raw_path)

    concat_list_path = TMP_DIR / f"concat_{uuid.uuid4().hex}.txt"
    concat_entries = []
    for clip_path in clip_paths:
        escaped = str(clip_path).replace("'", "'\\''")
        concat_entries.append(f"file '{escaped}'")
    concat_list_path.write_text("\n".join(concat_entries) + "\n", encoding="utf-8")

    concatenated_video_path = TMP_DIR / f"concatenated_{uuid.uuid4().hex}.mp4"
    run_ffmpeg(
        [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(concat_list_path),
            "-c:v", "libx264", "-preset", "medium", "-crf", "20",
            "-c:a", "aac",
            str(concatenated_video_path),
        ]
    )

    # --- Mix audio tracks ---
    mixed_audio_path: Optional[Path] = None
    if audio_urls:
        audio_paths = [
            download_to_tmp(audio_url, f"assemble_audio_{idx}.wav")
            for idx, audio_url in enumerate(audio_urls)
        ]
        mixed_audio_path = TMP_DIR / f"mixed_audio_{uuid.uuid4().hex}.m4a"

        if len(audio_paths) == 1:
            run_ffmpeg([
                "ffmpeg", "-y", "-i", str(audio_paths[0]),
                "-c:a", "aac", str(mixed_audio_path),
            ])
        else:
            ffmpeg_cmd = ["ffmpeg", "-y"]
            for audio_path in audio_paths:
                ffmpeg_cmd.extend(["-i", str(audio_path)])
            amix_inputs = "".join([f"[{idx}:a]" for idx in range(len(audio_paths))])
            ffmpeg_cmd.extend([
                "-filter_complex",
                f"{amix_inputs}amix=inputs={len(audio_paths)}:duration=longest:normalize=0[aout]",
                "-map", "[aout]", "-c:a", "aac",
                str(mixed_audio_path),
            ])
            run_ffmpeg(ffmpeg_cmd)

    # --- Subtitles ---
    subtitle_path: Optional[Path] = None
    if isinstance(subtitle_url, str) and subtitle_url:
        subtitle_path = download_to_tmp(subtitle_url, "subtitles.srt")

    # --- Determine export formats ---
    requested_format = job_input.get("format")
    if requested_format:
        spec = ASSEMBLE_FORMAT_SPECS.get(requested_format)
        if spec is None:
            raise ValueError(
                f"Unknown format '{requested_format}'. "
                f"Valid: {list(ASSEMBLE_FORMAT_SPECS.keys())}"
            )
        export_specs = [(requested_format, spec[0], spec[1])]
    else:
        export_specs = ALL_DEFAULT_EXPORTS

    # --- Export ---
    output_urls: List[str] = []
    export_map: Dict[str, str] = {}
    subtitle_filter = ffmpeg_subtitles_filter_part(subtitle_path)

    for name, width, height in export_specs:
        output_path = TMP_DIR / f"assembled_{name}_{uuid.uuid4().hex}.mp4"
        vf = (
            f"scale={width}:{height}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height}{subtitle_filter}"
        )

        ffmpeg_cmd = [
            "ffmpeg", "-y", "-i", str(concatenated_video_path),
        ]
        if mixed_audio_path is not None:
            ffmpeg_cmd.extend(["-i", str(mixed_audio_path), "-map", "0:v:0", "-map", "1:a:0"])
        ffmpeg_cmd.extend([
            "-vf", vf,
            "-c:v", "libx264", "-preset", "medium", "-crf", "20",
            "-c:a", "aac", "-b:a", "192k", "-shortest",
            str(output_path),
        ])
        run_ffmpeg(ffmpeg_cmd)

        signed_url = upload_file_to_r2(output_path, f"tarik/assemble/{name}")
        output_urls.append(signed_url)
        export_map[name] = signed_url

    # Lovable expects video_url (singular) when format is specified
    payload: Dict[str, Any] = {"exports": export_map}
    if requested_format and output_urls:
        payload["video_url"] = output_urls[0]

    return StepResult(
        output_urls=output_urls,
        credits_used=len(export_specs) * 3,
        payload=payload,
    )


REFORMAT_SPECS = {
    "16:9": (1920, 1080),
    "9:16": (1080, 1920),
    "1:1": (1080, 1080),
    "4:5": (1080, 1350),
    "youtube": (1920, 1080),
    "tiktok": (1080, 1920),
    "instagram": (1080, 1080),
    "reels": (1080, 1350),
}


def process_reformat(job_input: Dict[str, Any]) -> StepResult:
    """Reformat a video to a target aspect ratio / platform."""
    source_url = job_input["source_url"]
    target_aspect = job_input.get("target_aspect", "16:9")

    spec = REFORMAT_SPECS.get(target_aspect)
    if spec is None:
        raise ValueError(
            f"Unknown target_aspect '{target_aspect}'. "
            f"Valid: {list(REFORMAT_SPECS.keys())}"
        )
    width, height = spec
    source_path = download_to_tmp(source_url, "reformat_source.mp4")
    output_path = TMP_DIR / f"reformatted_{target_aspect.replace(':', 'x')}_{uuid.uuid4().hex}.mp4"

    vf = (
        f"scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height}"
    )
    run_ffmpeg([
        "ffmpeg", "-y", "-i", str(source_path),
        "-vf", vf,
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k",
        str(output_path),
    ])

    signed_url = upload_file_to_r2(output_path, f"tarik/reformat/{target_aspect.replace(':', 'x')}")
    return StepResult(
        output_urls=[signed_url],
        credits_used=2,
        payload={"target_aspect": target_aspect, "width": width, "height": height},
    )


def _nvidia_smi_summary() -> str:
    """Run nvidia-smi and return a short summary string."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        return r.stdout.strip() or r.stderr.strip() or "(no output)"
    except Exception as exc:
        return f"(nvidia-smi failed: {exc})"


def _disk_usage_summary() -> Dict[str, Any]:
    """Return free/total disk space for key mount points."""
    info = {}
    for mount in ("/", "/workspace", "/tmp"):
        try:
            st = os.statvfs(mount)
            total_gb = round((st.f_blocks * st.f_frsize) / (1024**3), 1)
            free_gb = round((st.f_bavail * st.f_frsize) / (1024**3), 1)
            info[mount] = {"total_gb": total_gb, "free_gb": free_gb}
        except OSError:
            info[mount] = "not mounted"
    return info


def process_health_check(job_input: Dict[str, Any]) -> StepResult:
    """Lightweight smoke test: verifies CUDA, R2, and handler wiring."""
    checks: Dict[str, Any] = {"build": "89ec366-v2"}

    # CUDA availability
    if torch is not None and torch.cuda.is_available():
        checks["cuda"] = {
            "available": True,
            "device": torch.cuda.get_device_name(0),
            "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1),
        }
    else:
        checks["cuda"] = {
            "available": False,
            "torch_loaded": torch is not None,
            "nvidia_smi": _nvidia_smi_summary(),
        }

    # Disk space
    checks["disk"] = _disk_usage_summary()

    # Volume diagnostics
    checks["volume"] = {
        "mounted": _is_volume_mounted(),
    }
    try:
        ws_contents = os.listdir("/workspace")
        checks["volume"]["workspace_contents"] = ws_contents[:30]
    except OSError as exc:
        checks["volume"]["workspace_contents"] = str(exc)
    try:
        model_contents = os.listdir("/workspace/models")
        checks["volume"]["models_contents"] = model_contents[:30]
    except OSError as exc:
        checks["volume"]["models_contents"] = str(exc)

    # Env diagnostics
    checks["env"] = {
        "MODEL_CACHE_DIR": str(MODEL_CACHE_DIR),
        "TMPDIR": os.environ.get("TMPDIR", "(not set)"),
        "TORCH_HOME": os.environ.get("TORCH_HOME", "(not set)"),
        "HF_HOME": os.environ.get("HF_HOME", "(not set)"),
        "OFFLOAD_DIR": str(OFFLOAD_DIR),
    }

    # R2 upload smoke test
    try:
        probe_path = TMP_DIR / f"health_probe_{uuid.uuid4().hex}.txt"
        probe_path.write_text("health_check", encoding="utf-8")
        probe_url = upload_file_to_r2(probe_path, "tarik/health")
        checks["r2"] = {"ok": True, "url_https": probe_url.startswith("https://"), "url": probe_url}
        logger.info("Health check R2 probe URL: %s", probe_url)
    except Exception as exc:
        checks["r2"] = {"ok": False, "error": str(exc)}

    # Registered steps
    checks["steps"] = list(STEP_HANDLERS.keys())

    # Import diagnostics for backends
    import_checks = {}
    for mod_name in ["scipy", "soundfile", "cv2", "audiocraft", "audiocraft.models"]:
        try:
            __import__(mod_name)
            import_checks[mod_name] = "ok"
        except Exception as exc:
            import_checks[mod_name] = str(exc)
    checks["imports"] = import_checks

    return StepResult(
        output_urls=[],
        credits_used=0,
        payload=checks,
    )


def process_generate_tts(job_input: Dict[str, Any]) -> StepResult:
    """Stub for TTS generation. Returns error until a TTS backend is configured."""
    raise NotImplementedError(
        "generate_tts is not yet implemented. "
        "Configure a TTS backend (Coqui, Piper, or ElevenLabs) to enable this step."
    )


def process_face_swap(job_input: Dict[str, Any]) -> StepResult:
    """Swap face in video to match a character reference image.

    Input:
        video_url: URL of the video to process
        reference_face_url: URL of the character reference face image
    Output:
        output_url: URL of the face-swapped video
    """
    video_url = job_input["video_url"]
    reference_face_url = job_input["reference_face_url"]

    model = load_model(FACESWAP)
    video_path = download_to_tmp(video_url, "faceswap_video.mp4")
    ref_path = download_to_tmp(reference_face_url, "faceswap_ref.jpg")
    output_path = TMP_DIR / f"faceswapped_{uuid.uuid4().hex}.mp4"

    if hasattr(model, "swap_face"):
        model.swap_face(
            video_path=str(video_path),
            reference_face_path=str(ref_path),
            output_path=str(output_path),
        )
    else:
        raise NotImplementedError("Face swap backend not configured")

    output_url = upload_file_to_r2(output_path, "tarik/faceswap")
    return StepResult(output_urls=[output_url], credits_used=3)


STEP_HANDLERS: Dict[str, Callable[[Dict[str, Any]], StepResult]] = {
    "generate_video": process_generate_video,
    "upscale": process_upscale,
    "lipsync": process_lipsync,
    "transcribe": process_transcribe,
    # Lovable names
    "generate_music": process_generate_music,
    "generate_sfx": process_generate_sfx,
    "generate_tts": process_generate_tts,
    # Legacy aliases (backward compat)
    "music": process_generate_music,
    "sfx": process_generate_sfx,
    "assemble": process_assemble,
    "reformat": process_reformat,
    "face_swap": process_face_swap,
    "health_check": process_health_check,
}


def _handler_impl(job_input: Dict[str, Any]) -> Dict[str, Any]:
    step = job_input.get("step")
    logger.info("Processing step=%s", step)

    if step not in STEP_HANDLERS:
        return response(
            status="failed",
            error=f"Unknown or missing step: {step}",
            retry_recommended=False,
        )

    try:
        result = STEP_HANDLERS[step](job_input)
        extra = result.payload or {}
        if result.failed_indices:
            extra["failed_indices"] = result.failed_indices
        status = "partial_success" if result.failed_indices else "completed"
        return response(
            status=status,
            output_urls=result.output_urls,
            credits_used=result.credits_used,
            retry_recommended=False,
            extra=extra,
        )
    except ModelLoadError as exc:
        logger.exception("Model load failure for step=%s", step)
        return response(
            status="failed",
            error=str(exc),
            retry_recommended=True,
        )
    except Exception as exc:
        if is_cuda_oom(exc):
            logger.warning("CUDA OOM in step=%s. Unloading all models.", step)
            unload_all_models()
            return response(
                status="failed",
                error="CUDA OOM encountered. Models unloaded. Retry with smaller batch.",
                retry_recommended=True,
            )
        logger.error("Unhandled error in step=%s: %s", step, traceback.format_exc())
        return response(
            status="failed",
            error=str(exc),
            retry_recommended=False,
        )


def handler(event):
    """RunPod serverless handler with resilient event unwrapping."""
    logger.info("RAW EVENT TYPE: %s", type(event).__name__)
    try:
        logger.info("RAW EVENT: %s", json.dumps(event) if isinstance(event, dict) else str(event)[:500])
    except Exception:
        logger.info("RAW EVENT (repr): %s", repr(event)[:500])

    job_input = None
    if isinstance(event, dict):
        if "input" in event:
            job_input = event["input"]
            logger.info("Extracted job_input from event['input']")
        elif "step" in event:
            job_input = event
            logger.info("Using event directly as job_input (flat format)")

    if job_input is None:
        keys = list(event.keys()) if isinstance(event, dict) else type(event).__name__
        logger.error("Could not extract job_input. Event keys: %s", keys)
        return response(
            status="failed",
            error=f"Invalid event format. Expected 'input' or 'step' key, got keys: {keys}",
            retry_recommended=False,
        )

    return _handler_impl(job_input)


if __name__ == "__main__":
    import sys

    if "--test" in sys.argv:
        # Local testing only
        test_event = {"input": {"step": "health_check"}}
        print(handler(test_event))
    else:
        # Production: start the RunPod serverless polling loop
        runpod.serverless.start({"handler": handler})
import gc
import importlib
import json
import logging
import os
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests
import runpod

try:
    import boto3
except Exception:  # pragma: no cover - runtime dependency in RunPod image
    boto3 = None

try:
    import torch
except Exception:  # pragma: no cover - runtime dependency in RunPod image
    torch = None


logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("tarik-handler")


MODEL_CACHE_DIR = Path("/workspace/models")
TMP_DIR = Path("/tmp")
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_HOME", str(MODEL_CACHE_DIR / "hf"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(MODEL_CACHE_DIR / "transformers"))


WAN = "wan_i2v_14b"
REALESRGAN = "realesrgan_x4"
WAV2LIP = "wav2lip_gfpgan"
WHISPER = "whisper_large_v3"
MUSICGEN = "musicgen_large"
AUDIOGEN = "audiogen"


MODEL_VRAM_HINTS_GB = {
    WAN: 75,
    REALESRGAN: 2,
    WAV2LIP: 8,
    WHISPER: 10,
    MUSICGEN: 4,
    AUDIOGEN: 3,
}


MODELS: Dict[str, Any] = {
    WAN: None,
    REALESRGAN: None,
    WAV2LIP: None,
    WHISPER: None,
    MUSICGEN: None,
    AUDIOGEN: None,
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
    with requests.get(url, timeout=120, stream=True) as response:
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
            resolution="480p",
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
            "resolution": "480p",
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


def run_music(model: Any, prompt: str, output_path: Path) -> None:
    if hasattr(model, "generate_music"):
        model.generate_music(prompt=prompt, duration_seconds=30, output_path=str(output_path))
        return
    write_mock_binary(
        output_path,
        "music",
        {"prompt": prompt, "duration_seconds": 30, "model": getattr(model, "name", "unknown")},
    )


def run_sfx(model: Any, prompt: str, output_path: Path) -> None:
    if hasattr(model, "generate_sfx"):
        model.generate_sfx(prompt=prompt, output_path=str(output_path))
        return
    write_mock_binary(
        output_path,
        "sfx",
        {"prompt": prompt, "model": getattr(model, "name", "unknown")},
    )


def response(
    status: str,
    output_urls: Optional[List[str]] = None,
    credits_used: int = 0,
    error: Optional[str] = None,
    retry_recommended: bool = False,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = {
        "status": status,
        "output_urls": output_urls or [],
        "credits_used": credits_used,
        "error": error,
        "retry_recommended": retry_recommended,
    }
    if extra:
        payload.update(extra)
    return payload


def process_generate_video(job_input: Dict[str, Any]) -> StepResult:
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
            logger.exception("Upscale failed for index %s: %s", absolute_idx, exc)
            failed_indices.append(absolute_idx)
    return output_urls, failed_indices


def process_upscale(job_input: Dict[str, Any]) -> StepResult:
    clip_urls = list(job_input.get("clips", []))
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
    clip_urls = list(job_input.get("clips", []))
    audio_url = job_input["audio_url"]
    model = load_model(WAV2LIP)
    audio_path = download_to_tmp(audio_url, "dialog_audio")

    output_urls: List[str] = []
    failed_indices: List[int] = []
    for idx, clip_url in enumerate(clip_urls):
        try:
            clip_path = download_to_tmp(clip_url, f"scene_{idx}.mp4")
            output_path = TMP_DIR / f"lipsynced_{idx}.mp4"
            run_lipsync(model, clip_path, audio_path, output_path)
            output_urls.append(upload_file_to_r2(output_path, "tarik/lipsync"))
        except Exception as exc:
            logger.exception("Lipsync failed for clip index %s: %s", idx, exc)
            failed_indices.append(idx)

    return StepResult(
        output_urls=output_urls,
        credits_used=len(output_urls) * 4,
        failed_indices=failed_indices,
    )


def process_transcribe(job_input: Dict[str, Any]) -> StepResult:
    audio_url = job_input["audio_url"]
    model = load_model(WHISPER)
    audio_path = download_to_tmp(audio_url, "transcribe_audio")

    srt_path = TMP_DIR / f"transcript_{uuid.uuid4().hex}.srt"
    transcript_text = run_transcribe(model, audio_path, srt_path)
    srt_url = upload_file_to_r2(srt_path, "tarik/transcribe")
    return StepResult(
        output_urls=[srt_url],
        credits_used=2,
        payload={"subtitle_text": transcript_text},
    )


def process_music(job_input: Dict[str, Any]) -> StepResult:
    prompt = job_input["prompt"]
    model = load_model(MUSICGEN)
    output_path = TMP_DIR / f"music_{uuid.uuid4().hex}.wav"
    run_music(model, prompt, output_path)
    output_url = upload_file_to_r2(output_path, "tarik/music")
    return StepResult(output_urls=[output_url], credits_used=6)


def process_sfx(job_input: Dict[str, Any]) -> StepResult:
    prompts = list(job_input.get("prompts", []))
    model = load_model(AUDIOGEN)

    output_urls: List[str] = []
    failed_indices: List[int] = []
    for idx, prompt in enumerate(prompts):
        try:
            output_path = TMP_DIR / f"sfx_{idx}_{uuid.uuid4().hex}.wav"
            run_sfx(model, prompt, output_path)
            output_urls.append(upload_file_to_r2(output_path, "tarik/sfx"))
        except Exception as exc:
            logger.exception("SFX generation failed for index %s: %s", idx, exc)
            failed_indices.append(idx)

    return StepResult(
        output_urls=output_urls,
        credits_used=len(output_urls),
        failed_indices=failed_indices,
    )


STEP_HANDLERS: Dict[str, Callable[[Dict[str, Any]], StepResult]] = {
    "generate_video": process_generate_video,
    "upscale": process_upscale,
    "lipsync": process_lipsync,
    "transcribe": process_transcribe,
    "music": process_music,
    "sfx": process_sfx,
}


def _handler_impl(event: Dict[str, Any]) -> Dict[str, Any]:
    job_input = event.get("input", {})
    step = job_input.get("step")
    logger.info("Received step=%s", step)

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
    return _handler_impl(event)


runpod.serverless.start({"handler": handler})
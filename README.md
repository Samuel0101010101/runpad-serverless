# runpad-serverless

RunPod serverless handler for the Tarik AI film production pipeline.

## What is included

- `handler.py`: single-GPU sequential orchestration for `generate_video`, `upscale`, `lipsync`, `transcribe`, `music`, and `sfx`
- `requirements.txt`: runtime dependencies

The handler enforces a strict **one-heavy-model-at-a-time** policy and unloads all other models before loading the requested one.

## Runtime contract

Input shape:

```json
{
	"input": {
		"step": "generate_video"
	}
}
```

Output shape:

```json
{
	"status": "completed|partial_success|failed",
	"output_urls": [],
	"credits_used": 0,
	"error": null,
	"retry_recommended": false
}
```

Additional payloads may include `failed_indices` and `subtitle_text`.

## Environment variables

Required for file upload/download:

- `R2_ENDPOINT_URL`
- `R2_ACCESS_KEY_ID`
- `R2_SECRET_ACCESS_KEY`
- `R2_BUCKET`

Optional:

- `R2_REGION` (default: `auto`)
- `R2_SIGNED_URL_TTL_SEC` (default: `86400`)
- `LOG_LEVEL` (default: `INFO`)

Model backend hooks (optional, but needed for real inference):

- `WAN_TI2V_5B_BACKEND`
- `REALESRGAN_X4_BACKEND`
- `WAV2LIP_GFPGAN_BACKEND`
- `WHISPER_LARGE_V3_BACKEND`
- `MUSICGEN_LARGE_BACKEND`
- `AUDIOGEN_BACKEND`

Each backend must be `module:function`, where the function accepts `cache_dir` and returns a model object implementing the expected method:

- Wan: `generate_video(...)`
- Real-ESRGAN: `upscale(...)`
- Wav2Lip: `lipsync(...)`
- Whisper: `transcribe(...)`
- MusicGen: `generate_music(...)`
- AudioGen: `generate_sfx(...)`

Weights are cached under `/workspace/models`.

## Local sanity check

```bash
pip install -r requirements.txt
python -m py_compile handler.py
```

## Deploy to RunPod

### Option A: GitHub repo deploy (recommended)

1. Push this repo to GitHub.
2. In RunPod, create a **Serverless Endpoint** with an A100 80GB worker.
3. Set source to this repository and start command:

```bash
python handler.py
```

4. Add all environment variables above.

### Option B: Custom container deploy

Use a CUDA-enabled image, install `requirements.txt`, copy `handler.py`, and run:

```bash
python handler.py
```

## Notes

- If a model backend env var is missing, the handler uses mock behavior so pipeline wiring can still be tested.
- CUDA OOM triggers model unload + retry guidance (`retry_recommended=true`).

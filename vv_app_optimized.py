import base64
import io
import os
import threading
import time
import uuid
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Lazy imports of VibeVoice to speed up process start
_vv_loader = None
_MODEL_ID: Optional[str] = None


class TtsRequest(BaseModel):
    text: str
    speaker_names: Optional[List[str]] = None  # e.g., ["Alice"]
    voice_sample_paths: Optional[List[str]] = None  # absolute paths inside container
    inference_steps: Optional[int] = 10
    cfg_scale: Optional[float] = 1.3
    sample_rate: Optional[int] = 24000
    do_sample: Optional[bool] = False
    use_eager: Optional[bool] = False


class TtsResponse(BaseModel):
    audio_wav_base64: str
    sample_rate: int
    duration_sec: float
    log: str


class TtsJobStatus(BaseModel):
    job_id: str
    status: str  # queued|in_progress|completed|failed|canceled
    percent: Optional[float] = None
    eta_seconds: Optional[float] = None
    elapsed_seconds: float
    message: Optional[str] = None
    audio_wav_base64: Optional[str] = None
    sample_rate: Optional[int] = None
    duration_sec: Optional[float] = None


def _require_api_key(x_api_key: Optional[str]):
    expected = os.getenv("API_KEY")
    if expected:
        if not x_api_key or x_api_key != expected:
            raise HTTPException(status_code=401, detail="Invalid API key")


def _load_vibevoice():
    global _vv_loader
    if _vv_loader is not None:
        return _vv_loader

    # Import here to avoid import time on cold start if just /health is used
    from vibevoice.modular.modeling_vibevoice_inference import (
        VibeVoiceForConditionalGenerationInference,
    )
    from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

    model_path = os.getenv("VIBEVOICE_MODEL_PATH", "microsoft/VibeVoice-1.5B")
    device = os.getenv("VIBEVOICE_DEVICE", "cuda")
    inference_steps_env = os.getenv("VIBEVOICE_INFERENCE_STEPS")
    default_steps = int(inference_steps_env) if inference_steps_env else 10
    voices_dir = os.getenv("VIBEVOICE_VOICES_DIR", "/app/voices")
    attn_impl_env = os.getenv("VIBEVOICE_ATTN_IMPL", "flash_attention_2")
    # Some runtimes have issues with accelerate's meta tensors + device_map=auto.
    # Allow disabling device_map via env to avoid "Cannot copy out of meta tensor" errors.
    device_map_env = os.getenv("VIBEVOICE_DEVICE_MAP", "").lower()
    use_device_map = device_map_env in ("1", "true", "auto")

    # derive a friendly model id for reporting
    global _MODEL_ID
    try:
        lower = (model_path or "").lower()
        if "large" in lower or "7b" in lower:
            _MODEL_ID = "vibevoice-large"
        elif "1.5b" in lower or "1_5b" in lower or "1-5b" in lower:
            _MODEL_ID = "vibevoice-1.5b"
        else:
            _MODEL_ID = os.path.basename(model_path) or "vibevoice"
    except Exception:
        _MODEL_ID = "vibevoice"

    class Loader:
        def __init__(self):
            # Load processor and model
            self.processor = VibeVoiceProcessor.from_pretrained(model_path)
            attn_impl = attn_impl_env
            # Respect VIBEVOICE_DEVICE_MAP; default is disabled to avoid meta tensor copy errors.
            dev_map = "auto" if (device == "cuda" and use_device_map) else None
            try:
                self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    model_path,
                    torch_dtype="auto",
                    device_map=dev_map,
                    attn_implementation=attn_impl,
                )
            except Exception:
                # Fallback to minimal args
                self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    model_path
                )
            try:
                self.model.to(device)
            except Exception:
                # Fallback to CPU if CUDA not available
                self.model.to("cpu")

            # === TIER 2 OPTIMIZATIONS ===

            # 1. Enable eval mode (disables dropout, batchnorm training)
            self.model.eval()

            # 2. torch.compile() — DISABLED: hurts perf on dynamic-shape autoregressive generation
            # Autoregressive models change tensor shapes each step, causing constant recompilation.
            # Keeping this code for reference if VibeVoice adds static-shape mode later.
            use_compile = os.getenv("VIBEVOICE_TORCH_COMPILE", "0").lower() in ("1", "true", "yes")
            if use_compile and device == "cuda":
                try:
                    import torch
                    compile_mode = os.getenv("VIBEVOICE_COMPILE_MODE", "reduce-overhead")
                    self.model = torch.compile(self.model, mode=compile_mode, fullgraph=False)
                    print(f"[OPT] torch.compile() applied with mode={compile_mode}", flush=True)
                except Exception as e:
                    print(f"[OPT] torch.compile() failed: {e}", flush=True)

            # 3. Enable CUDA optimizations
            if device == "cuda":
                import torch
                # Enable TF32 for matmul (already set via env but enforce)
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                # Enable cuDNN benchmark for auto-tuned convolutions
                torch.backends.cudnn.benchmark = True
                # Enable flash SDP attention
                torch.backends.cuda.enable_flash_sdp(True)
                torch.backends.cuda.enable_mem_efficient_sdp(True)
                # Pre-allocate CUDA memory pool to avoid allocation overhead during inference
                torch.cuda.empty_cache()
                torch.cuda.memory.set_per_process_memory_fraction(0.95)
                print("[OPT] CUDA optimizations enabled: TF32, cuDNN benchmark, Flash SDP, 95% mem pool", flush=True)

            self.device = device
            self.default_steps = default_steps
            self.voices_map = self._scan_voices(voices_dir)

        def _scan_voices(self, dir_path: str) -> Dict[str, str]:
            mapping: Dict[str, str] = {}
            try:
                if os.path.isdir(dir_path):
                    for fn in os.listdir(dir_path):
                        if fn.lower().endswith(".wav"):
                            name = os.path.splitext(fn)[0]
                            full = os.path.join(dir_path, fn)
                            mapping[name] = full
                            # Provide simplified aliases (e.g., en-Alice_woman -> Alice)
                            simple = name
                            if "_" in simple:
                                simple = simple.split("_")[0]
                            if "-" in simple:
                                simple = simple.split("-")[-1]
                            mapping.setdefault(simple, full)
            except Exception:
                pass
            return mapping

        def _scriptify(self, text: str) -> str:
            # Ensure text fits expected "Speaker X: ..." format
            t = (text or "").strip()
            if not t:
                return "Speaker 1: "
            lowered = t.lower()
            if lowered.startswith("speaker ") and ":" in lowered:
                return text
            return f"Speaker 1: {text}"

        def _map_speakers(self, speaker_names: Optional[List[str]], override_paths: Optional[List[str]]) -> List[str]:
            if override_paths:
                return override_paths
            speakers = speaker_names or ["Alice"]
            samples: List[str] = []
            for name in speakers:
                # exact match or fallback to alias
                if name in self.voices_map:
                    samples.append(self.voices_map[name])
                else:
                    # try case-insensitive contains
                    lower = name.lower()
                    match = None
                    for k, v in self.voices_map.items():
                        if k.lower() == lower or k.lower() in lower or lower in k.lower():
                            match = v
                            break
                    samples.append(match or next(iter(self.voices_map.values()), ""))
            return samples

        def tts(self, *, text: str, speaker_names: Optional[List[str]], voice_sample_paths: Optional[List[str]], steps: Optional[int], cfg_scale: Optional[float], sample_rate: Optional[int], do_sample: bool, use_eager: bool, stop_check_fn=None):
            import torch

            # Prepare inputs
            script = self._scriptify(text)
            voice_samples = self._map_speakers(speaker_names, voice_sample_paths)

            inputs = self.processor(
                text=[script],
                voice_samples=[voice_samples],
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )
            # to device
            inputs = {k: (v.to(self.model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

            # Configure model
            steps_to_use = int(steps or self.default_steps)
            self.model.set_ddpm_inference_steps(num_steps=steps_to_use)

            # Generate with inference_mode for better perf (no grad tracking overhead)
            # Note: skip autocast — model already loads as bfloat16, double-casting adds overhead
            with torch.inference_mode():
                try:
                    outputs = self.model.generate(
                        **inputs,
                        cfg_scale=float(cfg_scale or 1.3),
                        tokenizer=self.processor.tokenizer,
                        generation_config={"do_sample": bool(do_sample)},
                    )
                except TypeError:
                    # Fallback: minimal inputs only
                    outputs = self.model.generate(**inputs)

            # Extract audio
            audio_tensor = None
            if hasattr(outputs, "speech_outputs") and outputs.speech_outputs:
                audio_tensor = outputs.speech_outputs[0]
            elif hasattr(outputs, "detach"):
                audio_tensor = outputs

            if audio_tensor is None:
                raise RuntimeError("No audio produced by model.generate")

            # Convert to numpy
            if hasattr(audio_tensor, "detach"):
                try:
                    import torch  # type: ignore
                    if isinstance(audio_tensor, torch.Tensor):
                        audio_tensor = audio_tensor.to(torch.float32).detach().cpu()
                        audio_np = audio_tensor.numpy()
                    else:
                        audio_np = np.asarray(audio_tensor)
                except Exception:
                    audio_np = np.asarray(audio_tensor)
            else:
                audio_np = np.asarray(audio_tensor)

            # Squeeze batch/channel dims if present
            audio_np = np.squeeze(audio_np)

            sr = int(sample_rate or 24000)
            # Encode to WAV in-memory
            buf = io.BytesIO()
            sf.write(buf, audio_np, sr, format="WAV")
            buf.seek(0)
            b64 = base64.b64encode(buf.read()).decode("ascii")
            duration = float(len(audio_np) / sr) if sr > 0 else 0.0
            return b64, sr, duration

    _vv_loader = Loader()
    return _vv_loader


import logging
from collections import deque

_LOG_BUFFER = deque(maxlen=1000)

class _DequeLogHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
        except Exception:
            msg = str(record.getMessage())
        _LOG_BUFFER.append(msg)

_handler = _DequeLogHandler()
_handler.setLevel(logging.INFO)
_handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s"))
logging.getLogger().addHandler(_handler)

app = FastAPI(title="VibeVoice API", version="0.3.0")

# CORS for browser-based E-Worker-App calls (toggleable)
cors_enabled = os.getenv("CORS_ENABLED", "true").lower() in ("1","true","yes","on")
if cors_enabled:
    allowed_origins = os.getenv("ALLOWED_ORIGINS", "*")
    origins_list = [o.strip() for o in allowed_origins.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins_list if origins_list else ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/voices")
def list_voices(x_api_key: Optional[str] = Header(None)):
    _require_api_key(x_api_key)
    # Lightweight scan without loading the model
    dir_path = os.getenv("VIBEVOICE_VOICES_DIR", "/app/voices")
    voices = []
    try:
        if os.path.isdir(dir_path):
            for fn in os.listdir(dir_path):
                if fn.lower().endswith(".wav"):
                    name = os.path.splitext(fn)[0]
                    full = os.path.join(dir_path, fn)
                    voices.append({"name": name, "path": full})
    except Exception:
        voices = []
    return {"count": len(voices), "voices": voices}


@app.get("/logs")
def tail_logs(limit: int = 200, x_api_key: Optional[str] = Header(None)):
    _require_api_key(x_api_key)
    try:
        n = max(1, min(int(limit), 1000))
    except Exception:
        n = 200
    lines = list(_LOG_BUFFER)[-n:]
    return {"count": len(lines), "lines": lines}


@app.get("/models")
def list_models(x_api_key: Optional[str] = Header(None)):
    _require_api_key(x_api_key)
    # For now, expose a single model based on env; can be extended
    mp = os.getenv("VIBEVOICE_MODEL_PATH", "microsoft/VibeVoice-1.5B").lower()
    if "large" in mp or "7b" in mp:
        return [{"id": "vibevoice-large", "name": "VibeVoice Large"}]
    return [{"id": "vibevoice-1.5b", "name": "VibeVoice 1.5B"}]


@app.get("/models/{model_id}")
def get_model_schema(model_id: str, x_api_key: Optional[str] = Header(None)):
    _require_api_key(x_api_key)
    mid = model_id.lower()
    is_large = ("large" in mid) or ("7b" in mid)
    defaults = {
        "id": model_id,
        "name": "VibeVoice Large" if is_large else "VibeVoice 1.5B",
        "defaults": {
            "guidance": { "cfg_scale": 6.0 if is_large else 4.5, "inference_steps": 40 if is_large else 28 },
            "sampling": { "do_sample": False, "temperature": 1.0, "top_p": 0.9, "top_k": 50 },
            "audio": { "sample_rate_hz": 24000, "format": "wav" },
            "text_prep": { "enforce_speaker_format": True, "normalize_punctuation": True, "chunk_long_sentences": True, "max_chars_per_turn": 240, "warn_trigger_phrases": True }
        }
    }
    return defaults


@app.post("/tts", response_model=TtsResponse)
def tts(req: TtsRequest, x_api_key: Optional[str] = Header(None)):
    _require_api_key(x_api_key)
    start = time.time()
    loader = _load_vibevoice()
    audio_b64, sr, dur = loader.tts(
        text=req.text,
        speaker_names=req.speaker_names,
        voice_sample_paths=req.voice_sample_paths,
        steps=req.inference_steps,
        cfg_scale=req.cfg_scale,
        sample_rate=req.sample_rate,
        do_sample=bool(req.do_sample),
        use_eager=bool(req.use_eager),
    )
    log = f"device={loader.device} steps={req.inference_steps or loader.default_steps} sr={sr} latency_ms={(time.time()-start)*1000:.1f}"
    return TtsResponse(audio_wav_base64=audio_b64, sample_rate=sr, duration_sec=dur, log=log)


# Minimal async job system for progress + cancel
_JOBS: Dict[str, Dict] = {}
_JOBS_LOCK = threading.Lock()


def _estimate_duration_seconds(text: str) -> float:
    # Rough estimate: ~2.5 words/sec
    words = max(1, len((text or "").split()))
    return words / 2.5


def _job_percent_eta(job: Dict):
    now = time.time()
    if job["status"] == "queued":
        return 0.0, job.get("eta") or None, 0.0
    started = job.get("started") or job.get("created") or now
    elapsed = max(0.0, now - started)
    eta = job.get("eta") or 0.0
    percent = None
    if job["status"] in ("queued", "in_progress") and eta > 0:
        percent = min(90.0, max(0.0, (elapsed / eta) * 100.0))
    if job["status"] == "completed":
        percent = 100.0
        eta = 0.0
    return percent, (eta if eta else None), elapsed



@app.post("/tts/start", response_model=TtsJobStatus)
def tts_start(req: TtsRequest, x_api_key: Optional[str] = Header(None)):
    _require_api_key(x_api_key)
    job_id = str(uuid.uuid4())
    created = time.time()
    eta = _estimate_duration_seconds(req.text)

    with _JOBS_LOCK:
        _JOBS[job_id] = {
            "status": "queued",
            "created": created,
            "started": None,
            "finished": None,
            "cancel": False,
            "eta": eta,
            "req": req,
            "result": None,
            "error": None,
        }

    def run_job():
        with _JOBS_LOCK:
            job = _JOBS.get(job_id)
            if job is None:
                return
            job["status"] = "in_progress"
            job["started"] = time.time()

        try:
            loader = _load_vibevoice()
            def _stop():
                with _JOBS_LOCK:
                    j = _JOBS.get(job_id)
                    return bool(j and j.get("cancel"))

            audio_b64, sr, dur = loader.tts(
                text=req.text,
                speaker_names=req.speaker_names,
                voice_sample_paths=req.voice_sample_paths,
                steps=req.inference_steps,
                cfg_scale=req.cfg_scale,
                sample_rate=req.sample_rate,
                do_sample=bool(req.do_sample),
                use_eager=bool(req.use_eager),
                stop_check_fn=_stop,
            )
            with _JOBS_LOCK:
                job = _JOBS.get(job_id)
                if job is None or job.get("cancel"):
                    # Treat as canceled even if finished
                    if job is not None:
                        job["status"] = "canceled"
                        job["finished"] = time.time()
                    return
                job["status"] = "completed"
                job["finished"] = time.time()
                job["result"] = {"audio_b64": audio_b64, "sr": sr, "dur": dur}
        except Exception as e:
            with _JOBS_LOCK:
                job = _JOBS.get(job_id)
                if job is not None:
                    job["status"] = "failed"
                    job["finished"] = time.time()
                    job["error"] = str(e)

    threading.Thread(target=run_job, daemon=True).start()

    return TtsJobStatus(
        job_id=job_id,
        status="queued",
        percent=0.0,
        eta_seconds=eta,
        elapsed_seconds=0.0,
        message="Job queued",
    )


@app.get("/tts/jobs")
def tts_jobs(status: Optional[str] = None, x_api_key: Optional[str] = Header(None)):
    _require_api_key(x_api_key)
    items = []
    allowed = None
    if status:
        try:
            allowed = {s.strip().lower() for s in status.split(',') if s.strip()}
        except Exception:
            allowed = None
    with _JOBS_LOCK:
        for jid, job in _JOBS.items():
            percent, eta, elapsed = _job_percent_eta(job)
            row = {
                "job_id": jid,
                "model": _MODEL_ID,
                "status": job.get("status"),
                "percent": percent,
                "eta_seconds": eta,
                "elapsed_seconds": elapsed,
            }
            if allowed and (row["status"] or "").lower() not in allowed:
                continue
            items.append(row)
    items.sort(key=lambda x: _JOBS.get(x["job_id"], {}).get("created", 0), reverse=True)
    return items


@app.get("/tts/jobs/metrics")
def tts_jobs_metrics(x_api_key: Optional[str] = Header(None)):
    _require_api_key(x_api_key)
    counts: Dict[str, int] = {}
    total = 0
    with _JOBS_LOCK:
        for job in _JOBS.values():
            s = job.get("status") or "unknown"
            counts[s] = counts.get(s, 0) + 1
            total += 1
    return {
        "model": _MODEL_ID,
        "total": total,
        "queued": counts.get("queued", 0),
        "in_progress": counts.get("in_progress", 0),
        "completed": counts.get("completed", 0),
        "failed": counts.get("failed", 0),
        "canceled": counts.get("canceled", 0),
        "by_status": counts,
    }


@app.get("/tts/status/{job_id}", response_model=TtsJobStatus)
def tts_status(job_id: str, x_api_key: Optional[str] = Header(None)):
    _require_api_key(x_api_key)
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")

        now = time.time()
        percent, eta, elapsed = _job_percent_eta(job)

        if job["status"] == "completed" and job.get("result"):
            return TtsJobStatus(
                job_id=job_id,
                status=job["status"],
                percent=100.0,
                eta_seconds=0.0,
                elapsed_seconds=float((job["finished"] or now) - job["started"]),
                message=None,
                audio_wav_base64=job["result"]["audio_b64"],
                sample_rate=int(job["result"]["sr"]),
                duration_sec=float(job["result"]["dur"]),
            )

        if job["status"] == "failed":
            return TtsJobStatus(
                job_id=job_id,
                status="failed",
                percent=percent,
                eta_seconds=max(0.0, eta - elapsed) if eta else None,
                elapsed_seconds=elapsed,
                message=job.get("error") or "failed",
            )

        return TtsJobStatus(
            job_id=job_id,
            status=job["status"],
            percent=percent,
            eta_seconds=max(0.0, (eta or 0.0) - elapsed) if eta is not None else None,
            elapsed_seconds=elapsed,
            message=None,
        )


@app.post("/tts/cancel/{job_id}")
def tts_cancel(job_id: str, x_api_key: Optional[str] = Header(None)):
    _require_api_key(x_api_key)
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        if job["status"] in ("completed", "failed", "canceled"):
            return {"job_id": job_id, "status": job["status"]}
        job["cancel"] = True
        job["status"] = "canceled"
        job["finished"] = time.time()
        return {"job_id": job_id, "status": "canceled"}

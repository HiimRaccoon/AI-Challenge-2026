import argparse
import json
import logging
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

# Bounded cache for parquet reads. Keyed on (path, mtime, size) so a file
# that gets overwritten by a later pipeline step is reloaded automatically
# instead of serving stale data forever (the previous lru_cache(path_str)
# never invalidated on file changes).
_PARQUET_CACHE: "OrderedDict[tuple, pd.DataFrame]" = OrderedDict()
_PARQUET_CACHE_MAX = 16

def read_parquet_cached(path_str: str) -> pd.DataFrame:
    """Return a copy of a cached DataFrame; invalidates when the file changes."""
    path = Path(path_str)
    stat = path.stat()
    key = (str(path.resolve()), stat.st_mtime_ns, stat.st_size)

    cached = _PARQUET_CACHE.get(key)
    if cached is not None:
        _PARQUET_CACHE.move_to_end(key)
        return cached.copy()

    df = pd.read_parquet(path)
    _PARQUET_CACHE[key] = df
    _PARQUET_CACHE.move_to_end(key)
    while len(_PARQUET_CACHE) > _PARQUET_CACHE_MAX:
        _PARQUET_CACHE.popitem(last=False)
    return df.copy()
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

# =============================================================================
# PART 1: VALIDATION ENGINE
# =============================================================================

@dataclass
class ValidationIssue:
    severity: str  # P0 | P1 | P2
    module: str    # wp00 | wp01 | wp02 | wp05 | embeddings | ocr | asr
    video_id: str | None
    message: str
    details: str | None = None

def validate_keyframe_mapping(run_dir: Path) -> list[ValidationIssue]:
    issues = []
    frames_path = run_dir / "frames.parquet"
    if not frames_path.exists():
        issues.append(ValidationIssue("P0", "wp02", None, "frames.parquet missing"))
        return issues
    
    try:
        df = read_parquet_cached(str(frames_path))
    except Exception as e:
        issues.append(ValidationIssue("P0", "wp02", None, f"Failed to read frames.parquet: {e}"))
        return issues

    required_cols = ["video_id", "frame_id", "timestamp_ms", "keyframe_seq", "keyframe_path"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        issues.append(ValidationIssue("P0", "wp02", None, f"frames.parquet missing columns: {missing}"))
        return issues

    invalid_frames = df[(df["frame_id"] < 0) | (df["timestamp_ms"] < 0)]
    for _, row in invalid_frames.iterrows():
        vid = row["video_id"]
        issues.append(ValidationIssue("P0", "wp02", vid, "frame_id or timestamp_ms < 0", f"frame_id: {row['frame_id']}, ts: {row['timestamp_ms']}"))

    def check_image(row_dict):
        vid = row_dict["video_id"]
        kf_path_str = str(row_dict["keyframe_path"]).lstrip("/")
        kf_path = run_dir / kf_path_str
        if not kf_path.exists():
            return ValidationIssue("P0", "wp02", vid, "Keyframe file missing", str(kf_path))
        try:
            with Image.open(kf_path) as img:
                img.verify()
        except Exception:
            return ValidationIssue("P0", "wp02", vid, "Keyframe image invalid", str(kf_path))
        return None

    with ThreadPoolExecutor() as executor:
        results = executor.map(check_image, df.to_dict('records'))
    issues.extend(filter(None, results))
    
    if df.duplicated(subset=["video_id", "frame_id"]).any():
        issues.append(ValidationIssue("P0", "wp02", None, "Duplicate (video_id, frame_id) found"))
        
    if df.duplicated(subset=["video_id", "keyframe_seq"]).any():
        issues.append(ValidationIssue("P0", "wp02", None, "Duplicate (video_id, keyframe_seq) found"))

    return issues

def validate_frame_records(run_dir: Path) -> list[ValidationIssue]:
    issues = []
    frames_path = run_dir / "frames.parquet"
    if not frames_path.exists():
        return issues

    try:
        df = read_parquet_cached(str(frames_path))
    except Exception:
        return issues

    if "timestamp_ms" in df.columns:
        for vid, group in df.groupby("video_id"):
            if not group["timestamp_ms"].is_monotonic_increasing:
                issues.append(ValidationIssue("P0", "wp02", vid, "timestamp_ms is not monotonic"))

    if "keyframe_seq" in df.columns:
        for vid, group in df.groupby("video_id"):
            if not group["keyframe_seq"].is_monotonic_increasing:
                issues.append(ValidationIssue("P1", "wp02", vid, "keyframe_seq is not sequentially monotonic"))
                
    if "shot_id" in df.columns:
        for vid, group in df.groupby("video_id"):
            shot_file = run_dir / "shots" / f"{vid}.parquet"
            if not shot_file.exists():
                issues.append(ValidationIssue("P1", "wp02", vid, "Missing shots file", str(shot_file)))
                continue
            
            try:
                shots_df = read_parquet_cached(str(shot_file))
                valid_shot_ids = set(shots_df["shot_id"])
                missing_shots = set(group["shot_id"].dropna()) - valid_shot_ids
                if missing_shots:
                    issues.append(ValidationIssue("P1", "wp02", vid, "Frames reference missing shot_ids", str(missing_shots)))
            except Exception as e:
                issues.append(ValidationIssue("P1", "wp02", vid, f"Failed to read shot file: {e}", str(shot_file)))
                
    return issues

def validate_embeddings(run_dir: Path) -> list[ValidationIssue]:
    issues = []
    emb_maps_dir = run_dir / "embedding_maps"
    if not emb_maps_dir.exists() or not list(emb_maps_dir.glob("*.parquet")):
        issues.append(ValidationIssue("P1", "embeddings", None, "Missing embeddings (optional modality)"))
        return issues
        
    frames_path = run_dir / "frames.parquet"
    valid_frames = set()
    if frames_path.exists():
        try:
            valid_frames = set(read_parquet_cached(str(frames_path))["frame_id"])
        except Exception:
            pass

    for pq_file in emb_maps_dir.glob("*.parquet"):
        try:
            df = read_parquet_cached(str(pq_file))
        except Exception as e:
            issues.append(ValidationIssue("P1", "embeddings", None, f"Failed to read {pq_file.name}", str(e)))
            continue
            
        if "vector_id" in df.columns:
            invalid_vids = df[~df["vector_id"].isin(valid_frames)]
            for _, row in invalid_vids.iterrows():
                issues.append(ValidationIssue("P0", "embeddings", row.get("video_id"), "vector_id does not map to valid frame_id", str(row["vector_id"])))

        def check_embedding(row_dict):
            vid = row_dict.get("video_id")
            npy_path_str = str(row_dict["embedding_path"]).lstrip("/")
            npy_path = run_dir / npy_path_str
            if not npy_path.exists():
                return ValidationIssue("P1", "embeddings", vid, "Referenced shard .npy file missing", str(npy_path))
            try:
                arr = np.load(npy_path)
                if np.isnan(arr).any() or np.isinf(arr).any():
                    return ValidationIssue("P0", "embeddings", vid, "NaN or Inf in vectors", str(npy_path))
            except Exception:
                return ValidationIssue("P0", "embeddings", vid, "Failed to load .npy", str(npy_path))
            return None

        with ThreadPoolExecutor() as executor:
            results = executor.map(check_embedding, df.to_dict('records'))
        issues.extend(filter(None, results))

    return issues

def validate_ocr_asr(run_dir: Path) -> list[ValidationIssue]:
    issues = []
    frames_path = run_dir / "frames.parquet"
    valid_frames = set()
    if frames_path.exists():
        try:
            valid_frames = set(read_parquet_cached(str(frames_path))["frame_id"])
        except Exception:
            pass
            
    ocr_path = run_dir / "ocr" / "ocr.parquet"
    if ocr_path.exists():
        try:
            ocr_df = read_parquet_cached(str(ocr_path))
            if "frame_id" in ocr_df.columns:
                invalid_frames = set(ocr_df["frame_id"].dropna()) - valid_frames
                if invalid_frames:
                    issues.append(ValidationIssue("P1", "ocr", None, "OCR references invalid frame_ids"))
        except Exception as e:
            issues.append(ValidationIssue("P1", "ocr", None, "Failed to read ocr.parquet", str(e)))
    else:
        issues.append(ValidationIssue("P2", "ocr", None, "Missing OCR (optional modality)"))
        
    asr_path = run_dir / "asr" / "asr.parquet"
    media_path = run_dir / "media" / "media.parquet"
    media_durations = {}
    if media_path.exists():
        try:
            media_df = read_parquet_cached(str(media_path))
            if "video_id" in media_df.columns and "duration_ms" in media_df.columns:
                media_durations = media_df.set_index("video_id")["duration_ms"].to_dict()
        except Exception:
            pass
            
    if asr_path.exists():
        try:
            asr_df = read_parquet_cached(str(asr_path))
            if "start_ms" in asr_df.columns and "end_ms" in asr_df.columns:
                invalid_start_end = asr_df[asr_df["start_ms"] >= asr_df["end_ms"]]
                for _, row in invalid_start_end.iterrows():
                    issues.append(ValidationIssue("P1", "asr", row.get("video_id"), "start_ms >= end_ms"))
                
                if media_durations:
                    media_dur_series = asr_df['video_id'].map(media_durations)
                    invalid_duration = asr_df[asr_df["end_ms"] > media_dur_series]
                    for _, row in invalid_duration.iterrows():
                        issues.append(ValidationIssue("P1", "asr", row.get("video_id"), "end_ms > media duration"))
        except Exception as e:
            issues.append(ValidationIssue("P1", "asr", None, "Failed to read asr.parquet", str(e)))
    else:
        issues.append(ValidationIssue("P2", "asr", None, "Missing ASR (optional modality)"))

    return issues

def run_full_validation(run_dir: Path) -> list[ValidationIssue]:
    issues = []
    issues.extend(validate_keyframe_mapping(run_dir))
    issues.extend(validate_frame_records(run_dir))
    issues.extend(validate_embeddings(run_dir))
    issues.extend(validate_ocr_asr(run_dir))
    
    report_dir = run_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "validation.json"
    
    report_data = [asdict(i) for i in issues]
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
        
    return issues


# =============================================================================
# PART 2: PREPROCESSING REGISTRY
# =============================================================================

@dataclass
class PreprocessingRun:
    schema_version: str = "1.0.0"
    pipeline_version: str = "0.2.0"
    preprocess_run_id: str = ""
    source_manifest_sha256: str = ""
    config_sha256: str = ""
    code_commit: str = ""
    status: str = "running"  # running|partial|validated|failed|stable
    video_count: int = 0
    keyframe_count: int = 0
    started_at_utc: str = ""
    finished_at_utc: str | None = None
    artifact_root: str = ""
    validation_report_path: str | None = None

def create_preprocessing_run(preprocess_run_id: str, artifact_root: str, **kwargs) -> PreprocessingRun:
    run = PreprocessingRun(
        preprocess_run_id=preprocess_run_id,
        artifact_root=artifact_root,
        started_at_utc=datetime.now(timezone.utc).isoformat(),
        **kwargs
    )
    return run

def save_preprocessing_run(run: PreprocessingRun, run_dir: Path):
    manifest_path = run_dir / "manifest.json"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(asdict(run), f, indent=2)

def load_preprocessing_run(run_dir: Path) -> PreprocessingRun:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest.json in {run_dir}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return PreprocessingRun(**data)

def update_run_status(run_dir: Path, status: str, **kwargs):
    run = load_preprocessing_run(run_dir)
    if run.status == "stable":
        raise ValueError("Cannot update status of a stable run")
        
    valid_transitions = {
        "running": ["partial", "failed"],
        "partial": ["validated", "failed"],
        "validated": ["stable"]
    }
    
    if status not in valid_transitions.get(run.status, []):
        raise ValueError(f"Invalid status transition from {run.status} to {status}")
        
    run.status = status
    for k, v in kwargs.items():
        if hasattr(run, k):
            setattr(run, k, v)
            
    if status in ("failed", "stable"):
        run.finished_at_utc = datetime.now(timezone.utc).isoformat()
        
    save_preprocessing_run(run, run_dir)


# =============================================================================
# PART 3: API SERVER
# =============================================================================

app = FastAPI(title="AIC Preprocessing API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger(__name__)
GLOBAL_RUN_DIR = Path(".")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = datetime.now()
    response = await call_next(request)
    process_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"{request.method} {request.url.path} - {response.status_code} ({process_time:.3f}s)")
    return response

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}

@app.get("/manifest")
def get_manifest():
    p = GLOBAL_RUN_DIR / "manifest" / "corpus_manifest.parquet"
    if not p.exists():
        raise HTTPException(404, "Manifest not found")
    df = read_parquet_cached(str(p))
    return json.loads(df.to_json(orient="records"))

@app.get("/media")
def get_media():
    p = GLOBAL_RUN_DIR / "media" / "media.parquet"
    if not p.exists():
        raise HTTPException(404, "Media records not found")
    df = read_parquet_cached(str(p))
    return json.loads(df.to_json(orient="records"))

@app.get("/frames")
def get_frames(video_id: Optional[str] = None):
    p = GLOBAL_RUN_DIR / "frames.parquet"
    if not p.exists():
        raise HTTPException(404, "Frames records not found")
    df = read_parquet_cached(str(p))
    if video_id:
        df = df[df["video_id"] == video_id]
        if df.empty:
            raise HTTPException(404, "Video ID not found")
    return json.loads(df.to_json(orient="records"))

@app.get("/frames/{video_id}")
def get_frames_by_video(video_id: str):
    p = GLOBAL_RUN_DIR / "frames.parquet"
    if not p.exists():
        raise HTTPException(404, "Frames records not found")
    df = read_parquet_cached(str(p))
    df = df[df["video_id"] == video_id]
    if df.empty:
        raise HTTPException(404, "Video ID not found")
    return json.loads(df.to_json(orient="records"))

@app.get("/shots/{video_id}")
def get_shots(video_id: str):
    p = GLOBAL_RUN_DIR / "shots" / f"{video_id}.parquet"
    if not p.exists():
        raise HTTPException(404, "Shots not found for video")
    df = read_parquet_cached(str(p))
    if df.empty:
        raise HTTPException(404, "Shots not found for video")
    return json.loads(df.to_json(orient="records"))

@app.get("/temporal/{video_id}")
def get_temporal(video_id: str):
    p = GLOBAL_RUN_DIR / "temporal" / "temporal_frames.parquet"
    if not p.exists():
        raise HTTPException(404, "Temporal frames not found")
    df = read_parquet_cached(str(p))
    df = df[df["video_id"] == video_id]
    if df.empty:
        raise HTTPException(404, "Temporal frames not found for video")
    return json.loads(df.to_json(orient="records"))

@app.get("/keyframe-image/{video_id}/{filename}")
def get_keyframe_image(video_id: str, filename: str):
    if not re.match(r'^[a-zA-Z0-9_\-]+$', video_id):
        raise HTTPException(400, 'Invalid video_id')
    if not re.match(r'^[0-9]+\.jpg$', filename):
        raise HTTPException(400, 'Invalid filename')
        
    kf_dir = GLOBAL_RUN_DIR / "keyframes"
    img_path = kf_dir / video_id / filename
    
    if not img_path.resolve().is_relative_to(kf_dir.resolve()):
        raise HTTPException(403, 'Access denied')
        
    if not img_path.exists():
        raise HTTPException(404, 'Keyframe image not found')
        
    return FileResponse(str(img_path))

@app.get("/audio")
def get_audio():
    p = GLOBAL_RUN_DIR / "media" / "audio.parquet"
    if not p.exists():
        raise HTTPException(404, "Audio records not found")
    df = read_parquet_cached(str(p))
    return json.loads(df.to_json(orient="records"))

@app.get("/audio/{video_id}")
def get_audio_file(video_id: str):
    if not re.match(r'^[a-zA-Z0-9_\-]+$', video_id):
        raise HTTPException(400, 'Invalid video_id')
        
    audio_path = GLOBAL_RUN_DIR / "media" / f"{video_id}.wav"
    
    if not audio_path.exists():
        raise HTTPException(404, 'Audio file not found')
        
    return FileResponse(str(audio_path))

@app.get("/runs")
def list_runs():
    try:
        run = load_preprocessing_run(GLOBAL_RUN_DIR)
        return [asdict(run)]
    except FileNotFoundError:
        return []

@app.get("/runs/{run_id}/validate")
def validate_run(run_id: str):
    issues = run_full_validation(GLOBAL_RUN_DIR)
    return [asdict(i) for i in issues]

@app.get("/summary")
def get_summary():
    media_path = GLOBAL_RUN_DIR / "media" / "media.parquet"
    total_size_gb = 0.0
    video_count = 0
    if media_path.exists():
        media_df = read_parquet_cached(str(media_path))
        video_count = len(media_df)
        if "file_size_bytes" in media_df.columns:
            total_size_gb = float(media_df["file_size_bytes"].sum() / (1024**3))

    frames_path = GLOBAL_RUN_DIR / "frames.parquet"
    frame_count = 0
    if frames_path.exists():
        frames_df = read_parquet_cached(str(frames_path))
        frame_count = len(frames_df)

    return {
        "total_videos": video_count,
        "total_frames": frame_count,
        "total_size_gb": total_size_gb
    }

def main():
    global GLOBAL_RUN_DIR
    parser = argparse.ArgumentParser(description="AIC Preprocessing API Server")
    parser.add_argument("--run_dir", type=str, required=True, help="Path to preprocessing run directory")
    parser.add_argument("--port", type=int, default=8000, help="Port to run server on")
    args = parser.parse_args()
    
    GLOBAL_RUN_DIR = Path(args.run_dir)
    logging.basicConfig(level=logging.INFO)
    
    uvicorn.run(app, host="0.0.0.0", port=args.port)

if __name__ == "__main__":
    main()
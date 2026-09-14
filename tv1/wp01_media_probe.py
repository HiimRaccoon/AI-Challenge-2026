import argparse
import datetime
import hashlib
import json
import logging
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def setup_logger(name: str, log_file: Path) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    
    if not logger.handlers:
        fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # File handler
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        
        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        logger.addHandler(ch)
        
    return logger


def get_utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def calculate_sha256(filepath: str | Path) -> str:
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


class MediaProbe:
    def __init__(self, run_dir: Path, preprocess_run_id: str, raw_dir: Path):
        self.run_dir = run_dir
        self.preprocess_run_id = preprocess_run_id
        self.raw_dir = raw_dir
        self.media_dir = self.run_dir / "media"
        self.media_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = setup_logger("MediaProbe", self.media_dir / "wp01_media_probe.log")
        
        self.media_parquet = self.media_dir / "media.parquet"
        self.audio_parquet = self.media_dir / "audio.parquet"
        self.rejected_jsonl = self.media_dir / "rejected_videos.jsonl"
        self._file_lock = threading.Lock()
        
        self.media_schema = pa.schema([
            ('schema_version', pa.string()),
            ('preprocess_run_id', pa.string()),
            ('video_id', pa.string()),
            ('original_video_path', pa.string()),
            ('remux_path', pa.string()),
            ('proxy_path', pa.string()),
            ('source_sha256', pa.string()),
            ('time_base', pa.string()),
            ('fps_nominal', pa.float32()),
            ('fps_average', pa.float32()),
            ('is_variable_frame_rate', pa.bool_()),
            ('frame_count', pa.int64()),
            ('duration_ms', pa.int64()),
            ('width_px', pa.int32()),
            ('height_px', pa.int32()),
            ('codec', pa.string()),
            ('has_audio', pa.bool_()),
            ('created_at_utc', pa.string())
        ])
        
        self.audio_schema = pa.schema([
            ('schema_version', pa.string()),
            ('preprocess_run_id', pa.string()),
            ('video_id', pa.string()),
            ('audio_path', pa.string()),
            ('audio_sha256', pa.string()),
            ('sample_rate_hz', pa.int32()),
            ('channels', pa.int32()),
            ('duration_ms', pa.int64()),
            ('status', pa.string()),
            ('created_at_utc', pa.string())
        ])

    def parse_fraction(self, frac_str: Optional[str]) -> Optional[float]:
        try:
            if not frac_str or '/' not in frac_str:
                return float(frac_str) if frac_str else None
            num, den = frac_str.split('/')
            if int(den) == 0:
                return None
            return float(num) / float(den)
        except Exception:
            return None

    def reject_video(self, video_id: str, filename: str, reason_code: str, reason_message: str):
        record = {
            "video_id": video_id,
            "filename": filename,
            "reason_code": reason_code,
            "reason_message": reason_message,
            "created_at_utc": get_utc_now()
        }
        with self._file_lock:
            with open(self.rejected_jsonl, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record) + '\n')
        self.logger.warning(f"Rejected {video_id}: {reason_code} - {reason_message}")

    def probe_media(self, video_path: str) -> Optional[Dict[str, Any]]:
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_format', '-show_streams', str(video_path)
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                return None
            return json.loads(result.stdout)
        except Exception as e:
            self.logger.error(f"Error probing {video_path}: {e}")
            return None

    def decode_probe(self, video_path: str, duration_sec: float) -> bool:
        points = [0.0, 0.25, 0.5, 0.75, 0.95]
        for p in points:
            t = duration_sec * p
            cmd = ['ffmpeg', '-ss', str(t), '-i', str(video_path), '-frames:v', '1', '-f', 'null', '-']
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                if result.returncode != 0:
                    self.logger.error(f"Decode failed at {t}s for {video_path}")
                    return False
            except Exception as e:
                self.logger.error(f"Decode probe error at {t}s for {video_path}: {e}")
                return False
        return True

    def extract_audio(self, video_id: str, video_path: str) -> Dict[str, Any]:
        audio_filename = f"{video_id}.wav"
        audio_path = self.media_dir / audio_filename
        
        cmd = [
            'ffmpeg', '-y', '-i', str(video_path),
            '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', str(audio_path)
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0 and audio_path.exists():
                audio_sha = calculate_sha256(audio_path)
                
                # Get audio duration
                probe_res = self.probe_media(str(audio_path))
                duration_ms = None
                if probe_res and 'format' in probe_res and 'duration' in probe_res['format']:
                    duration_ms = int(float(probe_res['format']['duration']) * 1000)
                
                return {
                    "audio_path": str(audio_path),
                    "audio_sha256": audio_sha,
                    "sample_rate_hz": 16000,
                    "channels": 1,
                    "duration_ms": duration_ms,
                    "status": "ready"
                }
            else:
                return {"status": "failed"}
        except Exception as e:
            self.logger.error(f"Audio extraction failed for {video_id}: {e}")
            if audio_path.exists():
                try:
                    audio_path.unlink()
                except:
                    pass
            return {"status": "failed"}

    def _save_records(self, media_records: List[Dict[str, Any]], audio_records: List[Dict[str, Any]]):
        with self._file_lock:
            # Save media
            m_table = pa.Table.from_pylist(media_records, schema=self.media_schema)
            if self.media_parquet.exists():
                existing_table = pq.read_table(self.media_parquet)
                m_table = pa.concat_tables([existing_table, m_table])
                
            tmp_m = self.media_parquet.with_suffix('.tmp')
            pq.write_table(m_table, tmp_m)
            tmp_m.replace(self.media_parquet)
            
            # Save audio
            a_table = pa.Table.from_pylist(audio_records, schema=self.audio_schema)
            if self.audio_parquet.exists():
                existing_table = pq.read_table(self.audio_parquet)
                a_table = pa.concat_tables([existing_table, a_table])
                
            tmp_a = self.audio_parquet.with_suffix('.tmp')
            pq.write_table(a_table, tmp_a)
            tmp_a.replace(self.audio_parquet)

    def process_single_video(self, row: pd.Series, existing_video_ids: set) -> Optional[Dict[str, Any]]:
        video_id = row['video_id']
        original_path = str(self.raw_dir / row['original_video_path'])
        source_sha256 = row['source_sha256']
        filename = os.path.basename(original_path)
        
        if video_id in existing_video_ids:
            self.logger.info(f"Skipping {video_id} (already processed)")
            return None
            
        self.logger.info(f"Processing {video_id}")
        
        probe_info = self.probe_media(original_path)
        if not probe_info:
            self.reject_video(video_id, filename, "PROBE_FAILED", "ffprobe failed or returned nothing")
            return None
            
        video_stream = next((s for s in probe_info.get('streams', []) if s.get('codec_type') == 'video'), None)
        if not video_stream:
            self.reject_video(video_id, filename, "NO_VIDEO_STREAM", "No video stream found")
            return None
            
        has_audio = any(s.get('codec_type') == 'audio' for s in probe_info.get('streams', []))
        
        duration_sec = 0.0
        if 'format' in probe_info and 'duration' in probe_info['format']:
            duration_sec = float(probe_info['format']['duration'])
        else:
            self.reject_video(video_id, filename, "CORRUPT", "No duration found in format")
            return None
            
        if not self.decode_probe(original_path, duration_sec):
            self.reject_video(video_id, filename, "DECODE_FAILED", "Failed decoding frames at specific points")
            return None
            
        fps_nominal = self.parse_fraction(video_stream.get('r_frame_rate'))
        fps_average = self.parse_fraction(video_stream.get('avg_frame_rate'))
        
        is_vfr = False
        if fps_nominal and fps_average:
            if abs(fps_nominal - fps_average) > 0.1:
                is_vfr = True
        
        frame_count = None
        if 'nb_frames' in video_stream:
            try:
                frame_count = int(video_stream['nb_frames'])
            except:
                pass
                
        m_record = {
            "schema_version": "1.0.0",
            "preprocess_run_id": self.preprocess_run_id,
            "video_id": video_id,
            "original_video_path": original_path,
            "remux_path": None,
            "proxy_path": None,
            "source_sha256": source_sha256,
            "time_base": video_stream.get('time_base'),
            "fps_nominal": fps_nominal,
            "fps_average": fps_average,
            "is_variable_frame_rate": is_vfr,
            "frame_count": frame_count,
            "duration_ms": int(duration_sec * 1000),
            "width_px": int(video_stream.get('width', 0)),
            "height_px": int(video_stream.get('height', 0)),
            "codec": video_stream.get('codec_name'),
            "has_audio": has_audio,
            "created_at_utc": get_utc_now()
        }
        
        a_record = {
            "schema_version": "1.0.0",
            "preprocess_run_id": self.preprocess_run_id,
            "video_id": video_id,
            "audio_path": None,
            "audio_sha256": None,
            "sample_rate_hz": None,
            "channels": None,
            "duration_ms": None,
            "status": "no_audio",
            "created_at_utc": get_utc_now()
        }
        
        if has_audio:
            audio_info = self.extract_audio(video_id, original_path)
            a_record.update(audio_info)
            
        return {"media": m_record, "audio": a_record}

    def run(self, manifest_path: Path):
        self.logger.info(f"Starting media probe using manifest {manifest_path}")
        
        try:
            df_manifest = pd.read_parquet(manifest_path)
            
            # Filter rejected/duplicates
            if 'ingest_status' in df_manifest.columns:
                df_manifest = df_manifest[df_manifest['ingest_status'] == 'accepted']
                
        except Exception as e:
            self.logger.error(f"Failed to load manifest: {e}")
            return
            
        existing_video_ids = set()
        if self.media_parquet.exists():
            try:
                existing_df = pd.read_parquet(self.media_parquet)
                existing_video_ids = frozenset(existing_df['video_id'].tolist())
            except Exception as e:
                self.logger.error(f"Failed to read existing media.parquet: {e}")

        media_records = []
        audio_records = []

        try:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            max_workers = min(16, (os.cpu_count() or 1) * 2)
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for _, row in df_manifest.iterrows():
                    futures.append(executor.submit(self.process_single_video, row, existing_video_ids))
                    
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        media_records.append(result["media"])
                        audio_records.append(result["audio"])
                        
                        # Save in batches of 10 to reduce memory and allow resuming gracefully
                        if len(media_records) >= 10:
                            self._save_records(media_records, audio_records)
                            media_records = []
                            audio_records = []
                            
            if media_records:
                self._save_records(media_records, audio_records)
                
        finally:
            self.logger.info("Probe completed.")


def main():
    parser = argparse.ArgumentParser(description="WP01 Media Probe")
    parser.add_argument(
        "--manifest", 
        type=str, 
        default="data/runs/run_v1_batch1/manifest/corpus_manifest.parquet", 
        help="Path to input manifest (corpus_manifest.parquet)"
    )
    parser.add_argument(
        "--run_dir", 
        type=str, 
        default="data/runs/run_v1_batch1", 
        help="Run directory to output data"
    )
    parser.add_argument(
        "--preprocess_run_id", 
        type=str, 
        default="run_v1_batch1", 
        help="Preprocess run ID"
    )
    parser.add_argument(
        "--raw_dir", 
        type=str, 
        default="data/raw", 
        help="Raw video directory"
    )
    
    args = parser.parse_args()
    
    probe = MediaProbe(Path(args.run_dir), args.preprocess_run_id, Path(args.raw_dir))
    probe.run(Path(args.manifest))


if __name__ == "__main__":
    main()
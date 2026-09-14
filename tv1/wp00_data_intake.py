import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
import uuid
import yaml
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple, Any

import pyarrow as pa
import pyarrow.parquet as pq

# --- Logging Setup ---
def setup_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("DataIntake")
    logger.setLevel(logging.DEBUG)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # File handler (DEBUG)
    fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console handler (INFO)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger

# --- Schema Definition ---
@dataclass
class CorpusManifestRecord:
    schema_version: str
    video_id: str
    source_archive: Optional[str]
    original_video_path: str
    source_sha256: str
    file_size_bytes: int
    batch_id: Optional[str]
    duplicate_of_video_id: Optional[str]
    ingest_status: str
    created_at_utc: str

def get_pyarrow_schema() -> pa.Schema:
    return pa.schema([
        pa.field('schema_version', pa.string()),
        pa.field('video_id', pa.string()),
        pa.field('source_archive', pa.string()),
        pa.field('original_video_path', pa.string()),
        pa.field('source_sha256', pa.string()),
        pa.field('file_size_bytes', pa.int64()),
        pa.field('batch_id', pa.string()),
        pa.field('duplicate_of_video_id', pa.string()),
        pa.field('ingest_status', pa.string()),
        pa.field('created_at_utc', pa.string()),
    ])

# --- Core Class ---
class DataIntake:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Paths
        corpus_cfg = config.get('corpus', {})
        prep_cfg = config.get('preprocessing', {})
        
        self.archives_dir = Path(corpus_cfg.get('archives_dir', 'zip_video'))
        self.raw_dir = Path(corpus_cfg.get('raw_dir', 'data/raw'))
        self.run_dir = Path(prep_cfg.get('run_dir', 'data/runs/run_v1_batch1'))
        
        self.video_glob = corpus_cfg.get('video_glob', '**/*.mp4')
        self.video_id_rule = corpus_cfg.get('video_id_rule', 'stem')
        self.batch_id = corpus_cfg.get('batch_id', None)
        self.schema_version = "1.0.0"
        
        # Output paths
        self.manifest_dir = self.run_dir / "manifest"
        self.logs_dir = self.run_dir / "logs"
        self.manifest_parquet = self.manifest_dir / "corpus_manifest.parquet"
        self.manifest_json = self.manifest_dir / "corpus_manifest.json"
        self.duplicates_jsonl = self.run_dir / "duplicate_videos.jsonl"
        self.rejected_jsonl = self.run_dir / "rejected_files.jsonl"
        self.log_file = self.logs_dir / "wp00_data_intake.log"
        
        # State
        self.records: List[CorpusManifestRecord] = []
        self.duplicates: List[Dict[str, Any]] = []
        self.rejected: List[Dict[str, Any]] = []
        self.seen_hashes: Dict[str, str] = {}  # sha256 -> video_id
        
    def _prepare_directories(self):
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def _compute_sha256(self, file_path: Path) -> str:
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(8192), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception as e:
            self.logger.error(f"Error computing hash for {file_path}: {e}")
            raise

    def _is_safe_path(self, basedir: Path, path: str) -> bool:
        try:
            resolved_base = basedir.resolve()
            resolved_path = (basedir / path).resolve()
            resolved_path.relative_to(resolved_base)
            return True
        except ValueError:
            return False

    def _is_ignored_file(self, filename: str) -> bool:
        ignored = ['__MACOSX', '.DS_Store', 'Thumbs.db', 'desktop.ini']
        return any(ig in filename for ig in ignored)

    def _generate_video_id(self, path: Path) -> str:
        if self.video_id_rule == 'parent_stem':
            return f"{path.parent.name}_{path.stem}"
        return path.stem

    def _extract_zips(self):
        if not self.archives_dir.exists():
            self.logger.warning(f"Archives directory not found: {self.archives_dir}")
            return

        zip_files = list(self.archives_dir.glob("*.zip"))
        self.logger.info(f"Found {len(zip_files)} zip files in {self.archives_dir}")
        
        for zip_path in zip_files:
            self.logger.info(f"Processing zip: {zip_path.name}")
            try:
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    for info in zf.infolist():
                        if info.is_dir():
                            continue
                            
                        if not info.filename.lower().endswith('.mp4'):
                            continue
                            
                        if self._is_ignored_file(info.filename):
                            continue
                            
                        if not self._is_safe_path(self.raw_dir, info.filename):
                            self.logger.warning(f"Skipping unsafe path in zip: {info.filename}")
                            continue

                        target_path = self.raw_dir / info.filename
                        target_path.parent.mkdir(parents=True, exist_ok=True)
                        tmp_path = target_path.with_suffix(f'.tmp_{uuid.uuid4().hex[:8]}')
                        
                        try:
                            with zf.open(info) as source, open(tmp_path, 'wb') as target:
                                shutil.copyfileobj(source, target)
                                
                            if target_path.exists():
                                existing_hash = self._compute_sha256(target_path)
                                new_hash = self._compute_sha256(tmp_path)
                                if existing_hash == new_hash:
                                    self.logger.debug(f"File {info.filename} already extracted and content matches, skipping.")
                                    tmp_path.unlink()
                                    continue
                                else:
                                    target_path = target_path.with_name(f"{target_path.stem}_{new_hash[:8]}{target_path.suffix}")
                                    self.logger.warning(f"File {info.filename} content differs, saving as {target_path.name}")
                                    
                            tmp_path.rename(target_path)
                            self.logger.debug(f"Extracted: {target_path.name}")
                        except Exception as e:
                            self.logger.error(f"Error extracting {info.filename}: {e}")
                            if tmp_path.exists():
                                tmp_path.unlink()
            except zipfile.BadZipFile:
                self.logger.error(f"Bad zip file: {zip_path}")
            except Exception as e:
                self.logger.error(f"Error processing zip {zip_path}: {e}")

    def _process_files(self):
        if not self.raw_dir.exists():
            self.logger.warning(f"Raw directory not found: {self.raw_dir}")
            return

        mp4_files = list(self.raw_dir.glob(self.video_glob))
        self.logger.info(f"Found {len(mp4_files)} mp4 files to process in {self.raw_dir}")
        
        for idx, file_path in enumerate(mp4_files):
            if idx % 100 == 0 and idx > 0:
                self.logger.info(f"Processed {idx} / {len(mp4_files)} files...")
                
            try:
                video_id = self._generate_video_id(file_path)
                if not video_id:
                    raise ValueError("Generated video_id is empty")
                    
                file_size = file_path.stat().st_size
                sha256 = self._compute_sha256(file_path)
                
                # Check duplicates
                if sha256 in self.seen_hashes:
                    dup_of = self.seen_hashes[sha256]
                    status = "duplicate"
                    
                    record = CorpusManifestRecord(
                        schema_version=self.schema_version,
                        video_id=video_id,
                        source_archive=None,  # could derive from path if needed
                        original_video_path=str(file_path.relative_to(self.raw_dir)),
                        source_sha256=sha256,
                        file_size_bytes=file_size,
                        batch_id=self.batch_id,
                        duplicate_of_video_id=dup_of,
                        ingest_status=status,
                        created_at_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                    )
                    self.duplicates.append(asdict(record))
                    self.logger.debug(f"Duplicate found: {video_id} is dup of {dup_of}")
                    continue  # Do not add to main records
                else:
                    self.seen_hashes[sha256] = video_id
                    status = "accepted"
                    record = CorpusManifestRecord(
                        schema_version=self.schema_version,
                        video_id=video_id,
                        source_archive=None,
                        original_video_path=str(file_path.relative_to(self.raw_dir)),
                        source_sha256=sha256,
                        file_size_bytes=file_size,
                        batch_id=self.batch_id,
                        duplicate_of_video_id=None,
                        ingest_status=status,
                        created_at_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                    )
                    self.records.append(record)
                
            except Exception as e:
                self.logger.error(f"Error processing {file_path}: {e}")
                self.rejected.append({
                    "file_path": str(file_path),
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                })

    def _write_outputs(self):
        self.logger.info("Writing outputs...")
        
        # 1. Parquet
        if self.records:
            table = pa.Table.from_pylist([asdict(r) for r in self.records], schema=get_pyarrow_schema())
            pq.write_table(table, self.manifest_parquet)
        else:
            self.logger.warning("No records to write to parquet.")
            
        # 2. JSON
        with open(self.manifest_json, 'w', encoding='utf-8') as f:
            json.dump([asdict(r) for r in self.records], f, indent=2)
            
        # 3. Duplicates
        with open(self.duplicates_jsonl, 'w', encoding='utf-8') as f:
            for d in self.duplicates:
                f.write(json.dumps(d) + '\n')
                
        # 4. Rejected
        with open(self.rejected_jsonl, 'w', encoding='utf-8') as f:
            for r in self.rejected:
                f.write(json.dumps(r) + '\n')

    def run(self):
        self._prepare_directories()
        self.logger = setup_logger(self.log_file)
        self.logger.info(f"Starting DataIntake run in {self.run_dir}")
        
        self.logger.info("Extracting archives...")
        self._extract_zips()
        
        self.logger.info("Processing files...")
        self._process_files()
        
        self._write_outputs()
        
        self.logger.info("\n=== Summary Report ===")
        self.logger.info(f"Total processed: {len(self.records)}")
        self.logger.info(f"Accepted: {len([r for r in self.records if r.ingest_status == 'accepted'])}")
        self.logger.info(f"Duplicates: {len(self.duplicates)}")
        self.logger.info(f"Rejected: {len(self.rejected)}")
        self.logger.info(f"Outputs written to: {self.run_dir}")
        self.logger.info("======================")

def parse_args():
    parser = argparse.ArgumentParser(description="Data Intake Pipeline")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--archives_dir", type=str, help="Directory containing zip files")
    parser.add_argument("--raw_dir", type=str, help="Directory to extract raw videos")
    parser.add_argument("--run_dir", type=str, help="Directory for this run outputs")
    return parser.parse_args()

def main():
    args = parse_args()
    
    config = {}
    if args.config:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        # Defaults
        archives = args.archives_dir or "zip_video"
        raw = args.raw_dir or "data/raw"
        run = args.run_dir or "data/runs/run_v1_batch1"
        
        config = {
            'corpus': {
                'archives_dir': archives,
                'raw_dir': raw,
                'video_glob': '**/*.mp4',
                'video_id_rule': 'stem',
            },
            'preprocessing': {
                'run_dir': run
            }
        }
        
    intake = DataIntake(config)
    intake.run()

if __name__ == "__main__":
    main()

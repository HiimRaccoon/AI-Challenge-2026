import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def setup_logging(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("TemporalAdjacency")
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        logger.handlers.clear()
        
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def lookup_by_timestamp(records_df: pd.DataFrame, video_id: str, timestamp_ms: int) -> dict | None:
    """Binary search (or vectorized) to find frame whose window contains the timestamp."""
    vid_df = records_df[records_df['video_id'] == video_id]
    if vid_df.empty:
        return None
    
    starts = vid_df['window_start_ms'].values
    idx = np.searchsorted(starts, timestamp_ms, side='right') - 1
    
    if idx < 0 or idx >= len(vid_df):
        return None
        
    row = vid_df.iloc[idx]
    if row['window_start_ms'] <= timestamp_ms <= row['window_end_ms']:
        return row.to_dict()
    return None


def get_neighbors(records_df: pd.DataFrame, video_id: str, frame_id: int, window_ms: int) -> pd.DataFrame:
    """Return all frames for video within window_ms of the specified frame."""
    vid_df = records_df[records_df['video_id'] == video_id]
    if vid_df.empty:
        return pd.DataFrame()
        
    target_rows = vid_df[vid_df['frame_id'] == frame_id]
    if target_rows.empty:
        return pd.DataFrame()
        
    target_ts = target_rows.iloc[0]['timestamp_ms']
    
    start_ts = target_ts - window_ms
    end_ts = target_ts + window_ms
    
    return vid_df[(vid_df['timestamp_ms'] >= start_ts) & (vid_df['timestamp_ms'] <= end_ts)].copy()


def link_asr_segments(temporal_df: pd.DataFrame, asr_parquet_path: Path | None) -> pd.DataFrame:
    """If asr_parquet_path exists, read ASR segments and link to temporal windows."""
    logger = logging.getLogger("TemporalAdjacency")
    if asr_parquet_path and asr_parquet_path.exists():
        logger.info(f"ASR linking requested using {asr_parquet_path}, but not fully implemented yet.")
        # Future implementation goes here
    else:
        logger.info("No ASR parquet path provided or file missing. Skipping ASR linking.")
    return temporal_df


class TemporalAdjacency:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.frames_path = run_dir / "frames.parquet"
        self.media_path = run_dir / "media" / "media.parquet"
        self.out_dir = run_dir / "temporal"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logging(self.out_dir / "temporal_adj.log")

    def run(self):
        self.logger.info("Starting TemporalAdjacency process...")
        if not self.frames_path.exists():
            self.logger.error(f"Frames parquet not found: {self.frames_path}")
            return
            
        if not self.media_path.exists():
            self.logger.error(f"Media parquet not found: {self.media_path}")
            return
            
        frames_df = pd.read_parquet(self.frames_path)
        media_df = pd.read_parquet(self.media_path)
        
        self.logger.info(f"Loaded {len(frames_df)} frames and {len(media_df)} media records.")
        
        duration_map = {}
        if 'duration_ms' in media_df.columns and 'video_id' in media_df.columns:
            duration_map = media_df.set_index('video_id')['duration_ms'].to_dict()
        elif 'duration' in media_df.columns and 'video_id' in media_df.columns:
            duration_map = media_df.set_index('video_id')['duration'].to_dict()
            
        processed_frames = []
        
        for video_id, group in frames_df.groupby('video_id'):
            vid_df = group.sort_values(by='timestamp_ms').copy()
            vid_df.reset_index(drop=True, inplace=True)
            
            if not vid_df['timestamp_ms'].is_monotonic_increasing:
                err_msg = f"Video {video_id} timestamps not monotonic."
                self.logger.error(f"Monotonicity assertion failed for video {video_id} after sorting by timestamp_ms.")
                raise ValueError(err_msg)
            
            duration_ms = duration_map.get(video_id)
            if duration_ms is None:
                if len(vid_df) > 1:
                    avg_gap = (vid_df['timestamp_ms'].iloc[-1] - vid_df['timestamp_ms'].iloc[0]) / (len(vid_df) - 1)
                    duration_ms = vid_df['timestamp_ms'].iloc[-1] + avg_gap
                else:
                    duration_ms = vid_df['timestamp_ms'].iloc[0] + 1000
                    
            vid_df['prev_frame_id'] = vid_df['frame_id'].shift(1).astype(object)
            vid_df['prev_frame_id'] = vid_df['prev_frame_id'].where(vid_df['prev_frame_id'].notnull(), None)
            
            vid_df['next_frame_id'] = vid_df['frame_id'].shift(-1).astype(object)
            vid_df['next_frame_id'] = vid_df['next_frame_id'].where(vid_df['next_frame_id'].notnull(), None)
            
            vid_df['prev_timestamp_ms'] = vid_df['timestamp_ms'].shift(1).astype(object)
            vid_df['prev_timestamp_ms'] = vid_df['prev_timestamp_ms'].where(vid_df['prev_timestamp_ms'].notnull(), None)
            
            vid_df['next_timestamp_ms'] = vid_df['timestamp_ms'].shift(-1).astype(object)
            vid_df['next_timestamp_ms'] = vid_df['next_timestamp_ms'].where(vid_df['next_timestamp_ms'].notnull(), None)
            
            starts = []
            ends = []
            for i in range(len(vid_df)):
                row = vid_df.iloc[i]
                ts = row['timestamp_ms']
                
                if i == 0:
                    starts.append(0)
                else:
                    prev_ts = vid_df.iloc[i-1]['timestamp_ms']
                    starts.append(int((ts + prev_ts) // 2))
                    
                if i == len(vid_df) - 1:
                    ends.append(int(duration_ms))
                else:
                    next_ts = vid_df.iloc[i+1]['timestamp_ms']
                    ends.append(int((ts + next_ts) // 2))
                    
            vid_df['window_start_ms'] = starts
            vid_df['window_end_ms'] = ends
            
            processed_frames.append(vid_df)
            
        if not processed_frames:
            self.logger.warning("No frames to process.")
            return
            
        out_df = pd.concat(processed_frames, ignore_index=True)
        
        asr_path = self.run_dir / "asr" / "asr.parquet"
        out_df = link_asr_segments(out_df, asr_path if asr_path.exists() else None)
        
        now_str = datetime.now(timezone.utc).isoformat()
        
        final_records = []
        for _, row in out_df.iterrows():
            record = {
                "schema_version": "1.0.0",
                "preprocess_run_id": row.get('preprocess_run_id', 'unknown_run'),
                "video_id": row['video_id'],
                "frame_id": int(row['frame_id']),
                "keyframe_seq": int(row['keyframe_seq']),
                "timestamp_ms": int(row['timestamp_ms']),
                "shot_id": row['shot_id'] if 'shot_id' in row and pd.notna(row['shot_id']) else "",
                "prev_frame_id": int(row['prev_frame_id']) if pd.notna(row['prev_frame_id']) else None,
                "next_frame_id": int(row['next_frame_id']) if pd.notna(row['next_frame_id']) else None,
                "prev_timestamp_ms": int(row['prev_timestamp_ms']) if pd.notna(row['prev_timestamp_ms']) else None,
                "next_timestamp_ms": int(row['next_timestamp_ms']) if pd.notna(row['next_timestamp_ms']) else None,
                "window_start_ms": int(row['window_start_ms']),
                "window_end_ms": int(row['window_end_ms']),
                "created_at_utc": now_str
            }
            final_records.append(record)
            
        final_df = pd.DataFrame(final_records)
        
        parquet_out = self.out_dir / "temporal_frames.parquet"
        
        schema = pa.schema([
            pa.field('schema_version', pa.string()),
            pa.field('preprocess_run_id', pa.string()),
            pa.field('video_id', pa.string()),
            pa.field('frame_id', pa.int64()),
            pa.field('keyframe_seq', pa.int32()),
            pa.field('timestamp_ms', pa.int64()),
            pa.field('shot_id', pa.string()),
            pa.field('prev_frame_id', pa.int64()),
            pa.field('next_frame_id', pa.int64()),
            pa.field('prev_timestamp_ms', pa.int64()),
            pa.field('next_timestamp_ms', pa.int64()),
            pa.field('window_start_ms', pa.int64()),
            pa.field('window_end_ms', pa.int64()),
            pa.field('created_at_utc', pa.string()),
        ])
        
        table = pa.Table.from_pandas(final_df, schema=schema, preserve_index=False)
        pq.write_table(table, parquet_out)
        self.logger.info(f"Saved {len(final_df)} records to {parquet_out}")
        
        json_out = self.out_dir / "temporal_frames.json"
        with open(json_out, 'w', encoding='utf-8') as f:
            json.dump(final_records, f, indent=2)
        self.logger.info(f"Saved {len(final_records)} records to {json_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Temporal Adjacency Processing (WP05)")
    parser.add_argument("--run_dir", type=str, required=True, help="Path to the run directory")
    
    args = parser.parse_args()
    
    processor = TemporalAdjacency(Path(args.run_dir))
    processor.run()

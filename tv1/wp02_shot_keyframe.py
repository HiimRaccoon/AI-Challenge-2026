import os
import sys
import json
import logging
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from dataclasses import asdict, dataclass
import traceback

import av
import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


@dataclass(frozen=True)
class KeyframePolicy:
    """Validated policy for retrieval-keyframe coverage and image output."""

    threshold: float = 0.65
    min_gap_ms: int = 250
    max_gap_ms: int = 1000
    dedup_threshold: int = 12
    target_eval_fps: float = 10.0
    thumbnail_width: int = 240
    thumbnail_height: int = 180
    jpeg_quality: int = 95
    thumbnail_quality: int = 85

    @classmethod
    def from_config(cls, config: dict) -> "KeyframePolicy":
        policy = cls(
            threshold=float(config.get("threshold", cls.threshold)),
            min_gap_ms=int(config.get("min_gap_ms", cls.min_gap_ms)),
            max_gap_ms=int(config.get("max_gap_ms", cls.max_gap_ms)),
            dedup_threshold=int(config.get("dedup_threshold", cls.dedup_threshold)),
            target_eval_fps=float(config.get("target_eval_fps", cls.target_eval_fps)),
            thumbnail_width=int(config.get("thumbnail_width", cls.thumbnail_width)),
            thumbnail_height=int(config.get("thumbnail_height", cls.thumbnail_height)),
            jpeg_quality=int(config.get("jpeg_quality", cls.jpeg_quality)),
            thumbnail_quality=int(config.get("thumbnail_quality", cls.thumbnail_quality)),
        )
        policy.validate()
        return policy

    def validate(self) -> None:
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        if self.min_gap_ms <= 0:
            raise ValueError("min_gap_ms must be positive")
        if self.max_gap_ms < self.min_gap_ms:
            raise ValueError("max_gap_ms must be greater than or equal to min_gap_ms")
        if self.dedup_threshold < 0:
            raise ValueError("dedup_threshold must be non-negative")
        if self.target_eval_fps <= 0:
            raise ValueError("target_eval_fps must be positive")
        if 1000.0 / self.target_eval_fps > self.max_gap_ms:
            raise ValueError("target_eval_fps is too low to satisfy max_gap_ms")
        if self.thumbnail_width <= 0 or self.thumbnail_height <= 0:
            raise ValueError("thumbnail dimensions must be positive")
        if not 1 <= self.jpeg_quality <= 100 or not 1 <= self.thumbnail_quality <= 100:
            raise ValueError("JPEG qualities must be in [1, 100]")


def selection_reason_for_frame(
    *, gap_ms: int, boundary_detected: bool, min_gap_ms: int, max_gap_ms: int,
) -> str | None:
    """Return why the current best candidate should be emitted, if at all."""
    if gap_ms >= max_gap_ms:
        return "max_gap_support"
    if boundary_detected and gap_ms >= min_gap_ms:
        return "boundary_guard"
    return None


def keep_after_dedup(*, reason: str, distance: int | None, threshold: int) -> bool:
    """Do not let perceptual dedup violate the mandatory max-gap coverage."""
    if reason == "max_gap_support":
        return True
    return distance is None or distance > threshold


# --- CORE FUNCTIONS ---
def phash(image: np.ndarray, hash_size: int = 8, highfreq_factor: int = 4) -> np.ndarray:
    img_size = hash_size * highfreq_factor
    resized = cv2.resize(image, (img_size, img_size), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(resized.shape) == 3 else resized
    dct = cv2.dct(np.float32(gray))
    dctlowfreq = dct[0:hash_size, 0:hash_size]
    med = np.median(dctlowfreq)
    return (dctlowfreq > med).flatten()


def hamming_distance(hash1: np.ndarray, hash2: np.ndarray) -> int:
    return np.count_nonzero(hash1 != hash2)


def compute_histogram(image: np.ndarray) -> np.ndarray:
    # Resize nhỏ lại để tính Histogram siêu tốc
    small = cv2.resize(image, (320, 240))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist


def compute_sharpness(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    return cv2.Laplacian(gray, cv2.CV_64F).var()


# --- XỬ LÝ 1 VIDEO TỐI ƯU HÓA ---
def process_single_video(video_id: str, video_path: Path, run_dir: Path, preprocess_run_id: str, config: dict) -> dict:
    keyframes_dir = run_dir / "keyframes" / video_id
    thumbnails_dir = run_dir / "thumbnails" / video_id
    mappings_dir = run_dir / "mappings"
    shots_dir = run_dir / "shots"

    for d in [keyframes_dir, thumbnails_dir, mappings_dir, shots_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # --- THÔNG SỐ TỐI ƯU CHO TV2 & TV3 ---
    policy = KeyframePolicy.from_config(config)
    threshold = policy.threshold
    min_gap_ms = policy.min_gap_ms
    max_gap_ms = policy.max_gap_ms
    dedup_threshold = policy.dedup_threshold
    target_eval_fps = policy.target_eval_fps

    frame_records, shot_records = [], []

    container = None
    try:
        container = av.open(str(video_path))
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else 30.0

        # Evaluate a configurable number of frames per second. The mandatory
        # max-gap selection below still controls how many images are persisted.
        frame_skip = max(1, int(fps / target_eval_fps))

        prev_hist = None
        prev_hash = None

        # Keep one pending candidate in RAM. Selected images are persisted
        # immediately, so denser sampling does not retain a full-video image list.
        best_candidate = None
        last_extracted_ms = -min_gap_ms

        shot_count = 0
        current_shot_id = f"{video_id}_shot_{shot_count:04d}"
        shot_start_frame_id = 0
        shot_start_ms = 0

        keyframe_seq = 0
        frame_idx_in_stream = -1

        def save_candidate(candidate: dict, reason: str) -> None:
            nonlocal keyframe_seq
            kf_rel_path = f"keyframes/{video_id}/{keyframe_seq}.jpg"
            thumb_rel_path = f"thumbnails/{video_id}/{keyframe_seq}.jpg"
            image = candidate["img"]
            if not cv2.imwrite(
                str(run_dir / kf_rel_path), image,
                [int(cv2.IMWRITE_JPEG_QUALITY), policy.jpeg_quality],
            ):
                raise OSError(f"failed to write keyframe {kf_rel_path}")
            thumbnail = cv2.resize(
                image,
                (policy.thumbnail_width, policy.thumbnail_height),
                interpolation=cv2.INTER_AREA,
            )
            if not cv2.imwrite(
                str(run_dir / thumb_rel_path), thumbnail,
                [int(cv2.IMWRITE_JPEG_QUALITY), policy.thumbnail_quality],
            ):
                raise OSError(f"failed to write thumbnail {thumb_rel_path}")
            frame_records.append({
                "schema_version": "1.0.0", "preprocess_run_id": preprocess_run_id,
                "video_id": video_id, "frame_id": int(candidate["frame_id"]),
                "keyframe_seq": int(keyframe_seq), "timestamp_ms": int(candidate["timestamp_ms"]),
                "pts": int(candidate["pts"]) if candidate["pts"] is not None else None,
                "shot_id": candidate["shot_id"], "keyframe_path": kf_rel_path,
                "thumbnail_path": thumb_rel_path, "selection_reason": reason,
                "sharpness_score": float(candidate["sharpness"]),
                "blur_score": float(candidate["blur_score"]),
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            })
            keyframe_seq += 1

        for frame in container.decode(video=0):
            frame_idx_in_stream += 1
            if frame_idx_in_stream % frame_skip != 0:
                continue

            pts = frame.pts
            if frame.time is not None:
                timestamp_ms = int(frame.time * 1000)
            elif pts is not None and stream.time_base:
                timestamp_ms = int(pts * stream.time_base * 1000)
            else:
                timestamp_ms = int((frame_idx_in_stream / fps) * 1000)

            frame_id = frame.index if hasattr(frame, 'index') else frame_idx_in_stream

            # Giải nén ảnh
            img = frame.to_ndarray(format='bgr24')
            hist = compute_histogram(img)
            sharpness = compute_sharpness(img)

            # --- TÌM SHOT BOUNDARY ---
            boundary_detected = False
            if prev_hist is not None:
                correlation = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                if correlation < threshold:
                    boundary_detected = True

            gap_ms = timestamp_ms - last_extracted_ms

            # --- QUYẾT ĐỊNH CẮT ẢNH ---
            reason = selection_reason_for_frame(
                gap_ms=gap_ms,
                boundary_detected=boundary_detected,
                min_gap_ms=min_gap_ms,
                max_gap_ms=max_gap_ms,
            )

            if reason is not None and best_candidate is not None:
                # Kiểm tra lọc trùng lặp (Dedup)
                distance = (
                    hamming_distance(prev_hash, best_candidate["hash"])
                    if prev_hash is not None else None
                )
                if keep_after_dedup(
                    reason=reason,
                    distance=distance,
                    threshold=dedup_threshold,
                ):
                    save_candidate(best_candidate, reason)
                    prev_hash = best_candidate["hash"]
                    last_extracted_ms = best_candidate["timestamp_ms"]

                # Lưu thông tin Shot
                if boundary_detected:
                    shot_records.append({
                        "video_id": video_id, "shot_id": current_shot_id,
                        "start_frame_id": shot_start_frame_id, "end_frame_id": frame_id,
                        "start_ms": shot_start_ms, "end_ms": timestamp_ms,
                        "duration_ms": timestamp_ms - shot_start_ms
                    })
                    shot_count += 1
                    current_shot_id = f"{video_id}_shot_{shot_count:04d}"
                    shot_start_frame_id, shot_start_ms = frame_id, timestamp_ms

                # Reset Best Candidate cho cảnh mới
                best_candidate = None

            # --- CẬP NHẬT BEST CANDIDATE TRONG CẢNH HIỆN TẠI ---
            if best_candidate is None or sharpness > best_candidate["sharpness"]:
                best_candidate = {
                    "frame_id": frame_id, "timestamp_ms": timestamp_ms, "pts": pts,
                    "sharpness": sharpness, "blur_score": 1.0 / max(sharpness, 1e-6),
                    "hash": phash(img), "img": img, "shot_id": current_shot_id
                }

            prev_hist = hist

        # Cuối video: Lấy nốt candidate còn sót lại
        if best_candidate is not None:
            if not frame_records or frame_records[-1]["frame_id"] != best_candidate["frame_id"]:
                save_candidate(best_candidate, "end_of_video")
            shot_records.append({
                "video_id": video_id, "shot_id": current_shot_id,
                "start_frame_id": shot_start_frame_id, "end_frame_id": best_candidate["frame_id"],
                "start_ms": shot_start_ms, "end_ms": best_candidate["timestamp_ms"],
                "duration_ms": best_candidate["timestamp_ms"] - shot_start_ms
            })

        if frame_records:
            pq.write_table(pa.Table.from_pandas(pd.DataFrame(frame_records)), mappings_dir / f"{video_id}.parquet")
        if shot_records:
            pq.write_table(pa.Table.from_pandas(pd.DataFrame(shot_records)), shots_dir / f"{video_id}.parquet")

        return {"video_id": video_id, "status": "success", "keyframes": len(frame_records)}

    except Exception as e:
        traceback.print_exc()
        return {"video_id": video_id, "status": "error", "error": str(e)}
    finally:
        try:
            if container is not None:
                container.close()
        except Exception:
            pass


# --- QUẢN LÝ ĐA LUỒNG (ORCHESTRATOR) ---
class ShotKeyframeExtractor:
    def __init__(self, raw_dir: Path, run_dir: Path, config: dict = None):
        self.raw_dir = Path(raw_dir)
        self.run_dir = Path(run_dir)
        self.config = config or {}
        self.preprocess_run_id = self.config.get("preprocess_run_id", "run_v1")
        # Giới hạn max_workers để tránh lỗi Out Of Memory (OOM)
        default_workers = min(4, max(1, (os.cpu_count() or 2) - 1))
        self.max_workers = self.config.get("max_workers") or default_workers

        self.logger = logging.getLogger("WP02")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            ch = logging.StreamHandler(sys.stdout)
            ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(ch)

    def _prepare_run_policy(self) -> None:
        """Pin the content-affecting policy and reject mixed preprocess runs."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        policy = KeyframePolicy.from_config(self.config)
        payload = {
            "schema_version": "1.0.0",
            "preprocess_run_id": self.preprocess_run_id,
            "keyframes": asdict(policy),
        }
        policy_path = self.run_dir / "preprocess_policy.json"
        if policy_path.is_file():
            try:
                existing = json.loads(policy_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"invalid preprocess policy: {policy_path}") from exc
            if existing != payload:
                raise RuntimeError(
                    "preprocess policy differs from the existing run; use a new run directory"
                )
            return
        mappings_dir = self.run_dir / "mappings"
        if mappings_dir.is_dir() and any(mappings_dir.glob("*.parquet")):
            raise RuntimeError(
                "legacy mappings exist without preprocess_policy.json; use a new run directory"
            )
        temp_path = policy_path.with_suffix(".json.tmp")
        temp_path.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temp_path, policy_path)

    def run(self):
        self._prepare_run_policy()
        self.logger.info(f"Starting Multi-core Extraction (Max workers: {self.max_workers})")
        videos = [(p.stem, p) for p in self.raw_dir.rglob("*.mp4")]

        futures = []
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            for video_id, v_path in videos:
                mapping_path = self.run_dir / "mappings" / f"{video_id}.parquet"
                if mapping_path.is_file():
                    self.logger.info(f"Skipping {video_id}, already processed.")
                    continue

                futures.append(executor.submit(
                    process_single_video, video_id, v_path, self.run_dir, self.preprocess_run_id, self.config
                ))

            for future in as_completed(futures):
                res = future.result()
                if res["status"] == "success":
                    self.logger.info(f"✅ {res['video_id']} - Trích xuất: {res['keyframes']} ảnh siêu nét.")
                else:
                    self.logger.error(f"❌ Lỗi {res['video_id']}: {res.get('error')}")

        # Merge mappings
        self.logger.info("Merging mappings into frames.parquet...")
        all_frames = [pd.read_parquet(p) for p in (self.run_dir / "mappings").glob("*.parquet")]
        if all_frames:
            merged_df = pd.concat(all_frames, ignore_index=True)
            pq.write_table(pa.Table.from_pandas(merged_df), self.run_dir / "frames.parquet")
            self.logger.info(f"🎉 Xong! Tổng cộng có {len(merged_df)} khung hình chất lượng cao.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, default="data/raw")
    parser.add_argument("--run_dir", type=str, default="data/runs/run_v3_ultradense")
    parser.add_argument("--preprocess_run_id", type=str, default="run_v3_ultradense")
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--min_gap_ms", type=int, default=250)
    parser.add_argument("--max_gap_ms", type=int, default=1000)
    parser.add_argument("--dedup_threshold", type=int, default=12)
    parser.add_argument("--target_eval_fps", type=float, default=10.0)
    parser.add_argument("--jpeg_quality", type=int, default=95)
    parser.add_argument("--thumbnail_quality", type=int, default=85)
    parser.add_argument("--thumbnail_width", type=int, default=240)
    parser.add_argument("--thumbnail_height", type=int, default=180)
    parser.add_argument("--max_workers", type=int, default=None)
    args = parser.parse_args()

    config = {
        "preprocess_run_id": args.preprocess_run_id,
        "threshold": args.threshold,
        "min_gap_ms": args.min_gap_ms,
        "max_gap_ms": args.max_gap_ms,
        "dedup_threshold": args.dedup_threshold,
        "target_eval_fps": args.target_eval_fps,
        "jpeg_quality": args.jpeg_quality,
        "thumbnail_quality": args.thumbnail_quality,
        "thumbnail_width": args.thumbnail_width,
        "thumbnail_height": args.thumbnail_height,
        "max_workers": args.max_workers
    }

    KeyframePolicy.from_config(config)
    extractor = ShotKeyframeExtractor(Path(args.raw_dir), Path(args.run_dir), config)
    extractor.run()

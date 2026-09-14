"""
Long-lived local model server cho WP04 (OCR / ASR / Object detection).

Vì sao cần file này
--------------------
wp04.runtime_adapters gọi 3 biến môi trường WP04_OCR_COMMAND /
WP04_CHUNKFORMER_COMMAND / WP04_RFDETR_COMMAND như một câu lệnh shell,
chạy MỘT LẦN cho MỖI keyframe / MỖI audio segment (xem
_CommandAdapter._run trong runtime_adapters.py). Nếu câu lệnh đó tự
import torch + load checkpoint mỗi lần được gọi, bạn sẽ trả giá "cold
start" (vài giây) x (hàng chục nghìn keyframe) -> đây là nguyên nhân
chính khiến pipeline chạy hàng chục giờ thay vì vài giờ.

Giải pháp: tách làm 2 phần.
  1. model_server.py (file này)  -> chạy 1 lần, load 3 model lên GPU,
     đứng nghe HTTP trên localhost, xử lý xong không thoát.
  2. model_client.py             -> chương trình "launcher" thật sự mà
     WP04_*_COMMAND trỏ tới. Nó CHỈ gửi 1 request HTTP tới server và in
     JSON ra stdout đúng format WP04 yêu cầu. Không import torch, nên
     khởi động trong vài chục mili-giây thay vì vài giây.

Cách chạy
---------
    python model_server.py --port 8765

Server sẽ load model theo config bên dưới (sửa lại đường dẫn checkpoint
cho khớp máy bạn, hoặc truyền qua biến môi trường / file .env).
"""

from __future__ import annotations

import io
import json
import logging
import math
import os
import threading
import torch
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Sequence

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("wp04-model-server")


# ---------------------------------------------------------------------------
# Cấu hình checkpoint / device — sửa theo máy bạn hoặc để trống và set qua env
# ---------------------------------------------------------------------------

DEVICE = os.environ.get("WP04_DEVICE", "cuda:0")

MODEL_ROOT = Path(__file__).resolve().parents[2] / "weights"
PARSEQ_CHECKPOINT = Path(
    os.environ.get("PARSEQ_CHECKPOINT", str(MODEL_ROOT / "parseq" / "best-parseq.ckpt"))
)
EASYOCR_MODEL_DIR = Path(
    os.environ.get("EASYOCR_MODEL_DIR", str(MODEL_ROOT / "easyocr"))
)
EASYOCR_NETWORK_DIR = Path(
    os.environ.get("EASYOCR_NETWORK_DIR", str(EASYOCR_MODEL_DIR / "user_network"))
)
EASYOCR_DOWNLOAD_ENABLED = os.environ.get("EASYOCR_DOWNLOAD_ENABLED", "0") == "1"
OCR_BATCH_SIZE = int(os.environ.get("WP04_OCR_BATCH_SIZE", "64"))

CHUNKFORMER_REPO = os.environ.get("CHUNKFORMER_REPO", str(MODEL_ROOT / "chunkformer"))
CHUNKFORMER_REVISION = os.environ.get(
    "CHUNKFORMER_REVISION", "311fc03558a895dc2b32957f2fb4236c7fb1455b"
)

RFDETR_MODEL = os.environ.get("RFDETR_MODEL", "RFDETRMedium")
RFDETR_CHECKPOINT = os.environ.get("RFDETR_CHECKPOINT", str(MODEL_ROOT / "rf-detr" / "rf-detr-medium.pth"))
RFDETR_OPTIMIZE = os.environ.get("WP04_RFDETR_OPTIMIZE", "1") == "1"
RFDETR_DTYPE = os.environ.get("WP04_RFDETR_DTYPE", "float16")


# ---------------------------------------------------------------------------
# Từng model được load 1 LẦN DUY NHẤT khi server khởi động (hoặc lazy khi
# request đầu tiên tới, tuỳ bạn chọn EAGER_LOAD).
# ---------------------------------------------------------------------------

EAGER_LOAD = os.environ.get("WP04_EAGER_LOAD", "1") == "1"


class OCREngine:
    """CRAFT detector (EasyOCR) plus the local Vietnamese PARSeq checkpoint."""

    def __init__(self) -> None:
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._detector = None
        self._recognizer = None
        self._tokenizer = None
        self._transform = None

    def _ensure_loaded(self) -> None:
        if (
            self._detector is not None
            and self._recognizer is not None
            and self._tokenizer is not None
            and self._transform is not None
        ):
            return

        with self._load_lock:
            if (
                self._detector is not None
                and self._recognizer is not None
                and self._tokenizer is not None
                and self._transform is not None
            ):
                return
            if not PARSEQ_CHECKPOINT.is_file():
                raise FileNotFoundError(f"PARSeq checkpoint not found: {PARSEQ_CHECKPOINT}")
            if DEVICE.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError(f"CUDA requested ({DEVICE}) but torch.cuda.is_available() is false")

            try:
                import easyocr
                from torchvision import transforms as T
                from strhub.data.utils import Tokenizer
                from strhub.models.parseq.model import PARSeq
            except ImportError as exc:
                raise RuntimeError(
                    "OCR runtime missing. Install easyocr, timm, and the local PARSeq checkout; "
                    "see deploy/windows-rtx5070ti/setup.ps1."
                ) from exc

            log.info("Loading CRAFT detector from %s ...", EASYOCR_MODEL_DIR)
            detector = easyocr.Reader(
                ["en"],
                gpu=DEVICE,
                model_storage_directory=str(EASYOCR_MODEL_DIR),
                user_network_directory=str(EASYOCR_NETWORK_DIR),
                detector=True,
                recognizer=False,
                download_enabled=EASYOCR_DOWNLOAD_ENABLED,
                verbose=False,
            )

            # This team-owned checkpoint contains Lightning metadata, which requires
            # weights_only=False on PyTorch >= 2.6. Do not point this variable at an
            # untrusted checkpoint.
            log.info("Loading PARSeq-VN recognizer from %s ...", PARSEQ_CHECKPOINT)
            checkpoint = torch.load(PARSEQ_CHECKPOINT, map_location="cpu", weights_only=False)
            hparams = checkpoint.get("hyper_parameters")
            state_dict = checkpoint.get("state_dict")
            if not isinstance(hparams, dict) or not isinstance(state_dict, dict):
                raise RuntimeError("PARSeq checkpoint is missing hyper_parameters or state_dict")

            required = (
                "charset_train",
                "max_label_length",
                "img_size",
                "patch_size",
                "embed_dim",
                "enc_num_heads",
                "enc_mlp_ratio",
                "enc_depth",
                "dec_num_heads",
                "dec_mlp_ratio",
                "dec_depth",
                "decode_ar",
                "refine_iters",
                "dropout",
            )
            missing = [name for name in required if name not in hparams]
            if missing:
                raise RuntimeError(f"PARSeq checkpoint is missing hyperparameters: {', '.join(missing)}")

            tokenizer = Tokenizer(hparams["charset_train"])
            recognizer = PARSeq(
                num_tokens=len(tokenizer),
                max_label_length=hparams["max_label_length"],
                img_size=hparams["img_size"],
                patch_size=hparams["patch_size"],
                embed_dim=hparams["embed_dim"],
                enc_num_heads=hparams["enc_num_heads"],
                enc_mlp_ratio=hparams["enc_mlp_ratio"],
                enc_depth=hparams["enc_depth"],
                dec_num_heads=hparams["dec_num_heads"],
                dec_mlp_ratio=hparams["dec_mlp_ratio"],
                dec_depth=hparams["dec_depth"],
                decode_ar=hparams["decode_ar"],
                refine_iters=hparams["refine_iters"],
                dropout=hparams["dropout"],
            )
            recognizer.load_state_dict(state_dict, strict=True)
            recognizer.to(DEVICE).eval()

            self._detector = detector
            self._recognizer = recognizer
            self._tokenizer = tokenizer
            self._transform = T.Compose(
                [
                    T.Resize(tuple(hparams["img_size"]), T.InterpolationMode.BICUBIC),
                    T.ToTensor(),
                    T.Normalize(0.5, 0.5),
                ]
            )
            log.info("OCR engine ready.")

    @staticmethod
    def _boxes_from_detector_output(horizontal: Sequence[Any], free: Sequence[Any]) -> list[tuple[float, float, float, float]]:
        """Normalize one EasyOCR detector result to axis-aligned boxes."""
        boxes: list[tuple[float, float, float, float]] = []

        for raw in horizontal:
            if len(raw) == 4:
                x0, x1, y0, y1 = map(float, raw)
                boxes.append((x0, y0, x1, y1))

        for polygon in free:
            if len(polygon) < 4:
                continue
            xs = [float(point[0]) for point in polygon]
            ys = [float(point[1]) for point in polygon]
            boxes.append((min(xs), min(ys), max(xs), max(ys)))
        return boxes

    def _detect_boxes(self, image_bgr: Any) -> list[tuple[float, float, float, float]]:
        """Return axis-aligned CRAFT boxes as (x0, y0, x1, y1)."""
        assert self._detector is not None
        horizontal, free = self._detector.detect(image_bgr)
        return self._boxes_from_detector_output(
            horizontal[0] if horizontal else [],
            free[0] if free else [],
        )

    def _detect_boxes_many(self, images_bgr: Sequence[Any]) -> list[list[tuple[float, float, float, float]]]:
        """Batch CRAFT for equally sized images; preserve order and fall back safely otherwise."""
        assert self._detector is not None
        if not images_bgr:
            return []

        import numpy as np

        grouped: dict[tuple[int, ...], list[tuple[int, Any]]] = {}
        for index, image_bgr in enumerate(images_bgr):
            grouped.setdefault(tuple(image_bgr.shape), []).append((index, image_bgr))

        boxes_by_image: list[list[tuple[float, float, float, float]]] = [[] for _ in images_bgr]
        for group in grouped.values():
            if len(group) == 1:
                index, image_bgr = group[0]
                boxes_by_image[index] = self._detect_boxes(image_bgr)
                continue
            indices, image_group = zip(*group)
            horizontal, free = self._detector.detect(np.stack(image_group), reformat=False)
            if len(horizontal) != len(group) or len(free) != len(group):
                raise RuntimeError("CRAFT batch returned an unexpected number of detector results")
            for position, image_index in enumerate(indices):
                boxes_by_image[image_index] = self._boxes_from_detector_output(
                    horizontal[position], free[position],
                )
        return boxes_by_image

    def infer(self, keyframe_path: str) -> list[dict[str, Any]]:
        return self.infer_many([keyframe_path])[0]

    def infer_many(self, keyframe_paths: Sequence[str]) -> list[list[dict[str, Any]]]:
        self._ensure_loaded()
        image_paths = [Path(path) for path in keyframe_paths]
        if not image_paths:
            return []
        missing = next((path for path in image_paths if not path.is_file()), None)
        if missing is not None:
            raise FileNotFoundError(f"keyframe not found: {missing}")

        assert self._recognizer is not None
        assert self._tokenizer is not None
        assert self._transform is not None

        # The HTTP server is threaded but all OCR calls share one CUDA device.
        # Serialise this section to avoid unsafe concurrent model use.
        with self._inference_lock:
            import cv2
            import numpy as np
            from PIL import Image

            detections_by_image: list[list[dict[str, Any]]] = [[] for _ in image_paths]
            crops: list[Any] = []
            crop_locations: list[tuple[int, int, int, int, int, int, int]] = []
            images: list[tuple[Any, Any, int, int]] = []
            for image_index, image_path in enumerate(image_paths):
                with Image.open(image_path) as source:
                    image = source.convert("RGB")
                width, height = image.size
                image_bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
                images.append((image, image_bgr, width, height))
            try:
                boxes_by_image = self._detect_boxes_many([image_bgr for _, image_bgr, _, _ in images])
                for image_index, ((image, _, width, height), boxes) in enumerate(zip(images, boxes_by_image)):
                    for x0, y0, x1, y1 in boxes:
                        left = max(0, min(width - 1, math.floor(x0)))
                        top = max(0, min(height - 1, math.floor(y0)))
                        right = max(0, min(width, math.ceil(x1)))
                        bottom = max(0, min(height, math.ceil(y1)))
                        if right <= left or bottom <= top:
                            continue
                        crops.append(image.crop((left, top, right, bottom)))
                        crop_locations.append((image_index, left, top, right, bottom, width, height))
            finally:
                for image, _, _, _ in images:
                    image.close()

            batch_size = max(1, OCR_BATCH_SIZE)
            for start in range(0, len(crops), batch_size):
                batch_crops = crops[start : start + batch_size]
                try:
                    batch = torch.stack([self._transform(crop) for crop in batch_crops]).to(DEVICE)
                finally:
                    for crop in batch_crops:
                        crop.close()
                with torch.inference_mode():
                    logits = self._recognizer(self._tokenizer, batch)
                    texts, token_probs = self._tokenizer.decode(logits.softmax(-1))

                for text, probabilities, (image_index, left, top, right, bottom, width, height) in zip(
                    texts, token_probs, crop_locations[start : start + batch_size]
                ):
                    confidence = (
                        float(probabilities.clamp_min(1e-9).log().mean().exp().item())
                        if probabilities.numel()
                        else 0.0
                    )
                    detections_by_image[image_index].append(
                        {
                            "text": text,
                            "bbox_xyxy_norm": [left / width, top / height, right / width, bottom / height],
                            "confidence": confidence,
                        }
                    )
            return detections_by_image


class ASREngine:
    """ChunkFormer-CTC-Large-Vie, resident tr?n GPU."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        with self._lock:
            if self._model is not None:
                return

            log.info("Loading ChunkFormer from %s ...", CHUNKFORMER_REPO)

            try:
                from chunkformer import ChunkFormerModel

                if DEVICE.startswith("cuda") and not torch.cuda.is_available():
                    raise RuntimeError(
                        f"WP04_DEVICE={DEVICE} nh?ng CUDA kh?ng kh? d?ng."
                    )

                self._model = ChunkFormerModel.from_pretrained(CHUNKFORMER_REPO)
                self._model = self._model.to(DEVICE)
                self._model.eval()

                log.info(
                    "ChunkFormer loaded: %s",
                    next(self._model.parameters()).device,
                )

            except Exception as exc:
                self._model = None
                raise RuntimeError(
                    f"Kh?ng load ???c ChunkFormer: {type(exc).__name__}: {exc}"
                ) from exc

    @staticmethod
    def _timecode_to_ms(value: str) -> int:
        parts = str(value).split(":")
        if len(parts) != 4:
            raise ValueError(f"ChunkFormer timestamp kh?ng h?p l?: {value}")

        hh, mm, ss, ms = map(int, parts)
        return ((hh * 60 + mm) * 60 + ss) * 1000 + ms

    def infer(self, audio_path: str) -> list[dict[str, Any]]:
        self._ensure_loaded()

        path = Path(audio_path)
        if not path.is_file():
            raise FileNotFoundError(f"audio not found: {audio_path}")

        log.info("ChunkFormer inference: %s", path)

        raw = self._model.endless_decode(
            audio_path=str(path),
            chunk_size=64,
            left_context_size=128,
            right_context_size=128,
            total_batch_duration=60,
            return_timestamps=True,
        )

        results: list[dict[str, Any]] = []

        for row in raw or []:
            text = str(row.get("decode", row.get("text", ""))).strip()
            if not text:
                continue

            results.append(
                {
                    "start_ms": self._timecode_to_ms(row["start"]),
                    "end_ms": self._timecode_to_ms(row["end"]),
                    "text": text,
                    "confidence": float(row.get("confidence", 1.0)),
                }
            )

        return results


class ObjectEngine:
    """RF-DETR Medium v?i checkpoint local trong weights/rf-detr."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        with self._lock:
            if self._model is not None:
                return

            log.info(
                "Loading RF-DETR (%s) from %s ...",
                RFDETR_MODEL,
                RFDETR_CHECKPOINT,
            )

            try:
                import rfdetr

                model_class = getattr(rfdetr, RFDETR_MODEL)
                model = model_class(
                    pretrain_weights=RFDETR_CHECKPOINT
                )
                if RFDETR_OPTIMIZE:
                    try:
                        model.inference(
                            compile=False,
                            dtype=RFDETR_DTYPE,
                            inplace=True,
                        )
                        log.info("RF-DETR inference optimized: dtype=%s", RFDETR_DTYPE)
                    except Exception as optimization_error:
                        log.warning(
                            "RF-DETR optimization failed (%s); reloading safe FP32 model",
                            optimization_error,
                        )
                        model = model_class(pretrain_weights=RFDETR_CHECKPOINT)
                self._model = model

                log.info("RF-DETR loaded successfully.")

            except ImportError as exc:
                raise RuntimeError(
                    "Ch?a c?i rfdetr."
                ) from exc

    def infer(self, keyframe_path: str) -> list[dict[str, Any]]:
        return self.infer_many([keyframe_path])[0]

    def infer_many(self, keyframe_paths: Sequence[str]) -> list[list[dict[str, Any]]]:
        self._ensure_loaded()
        image_paths = [Path(path) for path in keyframe_paths]
        if not image_paths:
            return []
        missing = next((path for path in image_paths if not path.is_file()), None)
        if missing is not None:
            raise FileNotFoundError(f"keyframe not found: {missing}")
        with self._inference_lock:
            output = self._model.predict(
                [str(path) for path in image_paths],
                include_source_image=False,
            )
        outputs = output if isinstance(output, (list, tuple)) else [output]
        if len(outputs) != len(image_paths):
            raise RuntimeError(
                f"RF-DETR batch returned {len(outputs)} results for {len(image_paths)} keyframes"
            )
        return [
            _parse_rfdetr_output(row, image_path, class_names=self._model.class_names)
            for row, image_path in zip(outputs, image_paths)
        ]


def _parse_rfdetr_output(
    output: Any,
    image_path: Path,
    *,
    class_names: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    try:
        from PIL import Image

        with Image.open(image_path) as img:
            width, height = img.size
    except Exception as exc:
        raise RuntimeError(
            f"Kh?ng ??c ???c k?ch th??c ?nh {image_path}: {exc}"
        ) from exc

    if isinstance(output, (list, tuple)):
        output = output[0] if output else None

    if output is None:
        return []

    boxes = getattr(output, "xyxy", None)
    confidences = getattr(output, "confidence", None)
    class_ids = getattr(output, "class_id", None)

    if boxes is None and isinstance(output, dict):
        boxes = output.get("xyxy")
        confidences = output.get("confidence")
        class_ids = output.get("class_id")

    if boxes is None:
        return []

    results: list[dict[str, Any]] = []

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [float(v) for v in box]
        class_id = int(class_ids[i]) if class_ids is not None else None
        label = "unknown"
        if class_id is not None:
            label = (
                str(class_names[class_id])
                if class_names is not None and 0 <= class_id < len(class_names)
                else str(class_id)
            )

        results.append(
            {
                "label": label,
                "bbox_xyxy_norm": [
                    max(0.0, min(1.0, x1 / width)),
                    max(0.0, min(1.0, y1 / height)),
                    max(0.0, min(1.0, x2 / width)),
                    max(0.0, min(1.0, y2 / height)),
                ],
                "confidence": (
                    float(confidences[i])
                    if confidences is not None
                    else 0.0
                ),
            }
        )

    return results


# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------

ocr_engine = OCREngine()
asr_engine = ASREngine()
object_engine = ObjectEngine()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: Any) -> None:  # quiet default logging
        log.debug(fmt, *args)

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # health check
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
            return
        self._send_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid JSON body"})
            return

        started = time.monotonic()
        try:
            if self.path == "/ocr":
                detections = ocr_engine.infer(payload["keyframe_path"])
                self._send_json(200, {"detections": detections})
            elif self.path == "/ocr/batch":
                paths = payload.get("keyframe_paths")
                if not isinstance(paths, list) or not paths or any(not isinstance(path, str) for path in paths):
                    raise ValueError("keyframe_paths must be a non-empty list of paths")
                detections = ocr_engine.infer_many(paths)
                self._send_json(200, {"detections": detections})
            elif self.path == "/asr":
                segments = asr_engine.infer(payload["audio_path"])
                self._send_json(200, {"segments": segments})
            elif self.path == "/object":
                detections = object_engine.infer(payload["keyframe_path"])
                self._send_json(200, {"detections": detections})
            elif self.path == "/object/batch":
                paths = payload.get("keyframe_paths")
                if not isinstance(paths, list) or not paths or any(not isinstance(path, str) for path in paths):
                    raise ValueError("keyframe_paths must be a non-empty list of paths")
                detections = object_engine.infer_many(paths)
                self._send_json(200, {"detections": detections})
            else:
                self._send_json(404, {"error": f"unknown route: {self.path}"})
                return
        except FileNotFoundError as exc:
            self._send_json(400, {"error": str(exc)})
        except Exception as exc:  # noqa: BLE001 - trả lỗi rõ ràng cho client
            log.exception("inference failed on %s", self.path)
            self._send_json(500, {"error": f"{type(exc).__name__}: {exc}"})
        finally:
            elapsed_ms = (time.monotonic() - started) * 1000
            log.info("%s -> %.1f ms", self.path, elapsed_ms)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="WP04 long-lived model server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    if EAGER_LOAD:
        log.info("EAGER_LOAD=1 -> loading all 3 models now (server sẽ chậm khởi động 1 lần)")
        for name, engine in (("ocr", ocr_engine), ("asr", asr_engine), ("object", object_engine)):
            try:
                engine._ensure_loaded()  # type: ignore[attr-defined]
            except Exception as exc:  # noqa: BLE001
                log.warning("Model %s chưa load được lúc khởi động: %s", name, exc)
                log.warning("-> vẫn chạy server, model sẽ thử load lại khi có request đầu tiên")

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    log.info("WP04 model server đang chạy tại http://%s:%s (Ctrl+C để dừng)", args.host, args.port)
    log.info("Routes: POST /ocr {keyframe_path}  POST /asr {audio_path}  POST /object {keyframe_path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Đang tắt server...")
        server.shutdown()


if __name__ == "__main__":
    main()

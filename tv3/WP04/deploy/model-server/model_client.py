r"""
Client mỏng dùng làm giá trị cho WP04_OCR_COMMAND / WP04_CHUNKFORMER_COMMAND
/ WP04_RFDETR_COMMAND.

wp04.runtime_adapters._CommandAdapter._run() sẽ gọi:

    command.format(keyframe_path=...)   # cho OCR / object
    command.format(audio_path=...)      # cho ASR

qua `subprocess.run(command, shell=True, capture_output=True, text=True)`
rồi `json.loads(completed.stdout)` — nghĩa là script này CHỈ ĐƯỢC in JSON
ra stdout, không được in gì khác (log phải đi stderr hoặc tắt hẳn).

Vì file này không import torch / rfdetr / transformers gì cả (chỉ dùng
urllib có sẵn trong Python), thời gian khởi động của mỗi lần gọi chỉ còn
vài chục mili-giây thay vì vài giây — model thật nằm bên trong
model_server.py, đã load sẵn từ trước.

Cách trỏ WP04_*_COMMAND vào file này (trong .env hoặc set biến môi trường
trước khi chạy run.ps1):

    WP04_OCR_COMMAND=python C:\path\to\model_client.py --route ocr --keyframe-path {keyframe_path}
    WP04_CHUNKFORMER_COMMAND=python C:\path\to\model_client.py --route asr --audio-path {audio_path}
    WP04_RFDETR_COMMAND=python C:\path\to\model_client.py --route object --keyframe-path {keyframe_path}

Lưu ý: _CommandAdapter dùng str.format nên {keyframe_path} / {audio_path}
phải giữ nguyên literal trong .env, WP04 sẽ tự thay giá trị thật vào.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

# The OCR result can contain Vietnamese text. Windows PowerShell may start
# Python with a legacy CP1252 stdout encoding, which cannot print that JSON.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_SERVER = "http://127.0.0.1:8765"


def call_server(base_url: str, route: str, payload: dict) -> dict:
    url = f"{base_url}/{route}"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = resp.read()
            return json.loads(body.decode("utf-8"))
    except urllib.error.HTTPError as exc:
        # Server trả JSON lỗi có ý nghĩa -> đẩy nguyên văn ra để WP04 log lại được.
        body = exc.read().decode("utf-8", errors="replace")
        try:
            error_payload = json.loads(body)
        except json.JSONDecodeError:
            error_payload = {"error": body or str(exc)}
        print(json.dumps(error_payload, ensure_ascii=False), file=sys.stdout)
        sys.exit(1)
    except urllib.error.URLError as exc:
        print(
            json.dumps(
                {
                    "error": (
                        f"không kết nối được model_server.py tại {base_url} "
                        f"({exc.reason}). Đảm bảo đã chạy: python model_server.py"
                    )
                },
                ensure_ascii=False,
            ),
            file=sys.stdout,
        )
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--route", required=True, choices=["ocr", "asr", "object"])
    parser.add_argument("--keyframe-path", default=None)
    parser.add_argument("--audio-path", default=None)
    parser.add_argument("--server", default=DEFAULT_SERVER)
    args = parser.parse_args()

    if args.route in ("ocr", "object"):
        if not args.keyframe_path:
            print(json.dumps({"error": "--keyframe-path required"}), file=sys.stdout)
            sys.exit(1)
        payload = {"keyframe_path": args.keyframe_path}
    else:  # asr
        if not args.audio_path:
            print(json.dumps({"error": "--audio-path required"}), file=sys.stdout)
            sys.exit(1)
        payload = {"audio_path": args.audio_path}

    result = call_server(args.server, args.route, payload)
    # WP04 chỉ đọc stdout và json.loads() nguyên văn -> in đúng 1 dòng JSON.
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

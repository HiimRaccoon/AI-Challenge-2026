"""
Kiểm tra nhanh nội dung best-parseq.ckpt trước khi cắm vào model_server.py.

File chỉ 280KB là rất nhỏ so với PARSeq đầy đủ (thường vài chục MB), nên
cần xác nhận đây có đúng là full checkpoint hay chỉ một phần trước khi
tốn công wire vào pipeline chính.

Cách chạy (trên máy có torch, không cần GPU cho bước kiểm tra này):
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    python check_parseq_checkpoint.py "D:\\weights\\rec\\best-parseq.ckpt"
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python check_parseq_checkpoint.py <path-to-ckpt>")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.is_file():
        print(f"Không tìm thấy file: {path}")
        sys.exit(1)

    size_kb = path.stat().st_size / 1024
    print(f"File: {path}")
    print(f"Kích thước: {size_kb:.1f} KB")

    import torch

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        print(f"\n[!] torch.load thất bại: {type(exc).__name__}: {exc}")
        print("    -> file có thể hỏng, hoặc không phải checkpoint PyTorch chuẩn.")
        sys.exit(1)

    print(f"\nKiểu dữ liệu sau khi load: {type(checkpoint)}")

    if isinstance(checkpoint, dict):
        print(f"Các key cấp 1: {list(checkpoint.keys())}")

        # PyTorch Lightning checkpoint thường có các key này
        lightning_keys = {"state_dict", "hyper_parameters", "hparams_name", "pytorch-lightning_version"}
        found_lightning = lightning_keys & set(checkpoint.keys())
        if found_lightning:
            print(f"\n[✓] Trông giống checkpoint PyTorch Lightning (baudm/parseq dùng format này).")
            print(f"    Các key Lightning tìm thấy: {found_lightning}")
            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
                n_params = sum(v.numel() for v in state_dict.values() if hasattr(v, "numel"))
                print(f"    Số tensor trong state_dict: {len(state_dict)}")
                print(f"    Tổng số tham số: {n_params:,}")
                if n_params < 1_000_000:
                    print(
                        "    [!] Số tham số quá ít so với PARSeq thật (thường 20-30 triệu). "
                        "Nghi ngờ đây KHÔNG PHẢI full checkpoint."
                    )
                else:
                    print("    -> số lượng tham số hợp lý cho 1 model PARSeq đầy đủ.")
            if "hyper_parameters" in checkpoint:
                print(f"    hyper_parameters: {checkpoint['hyper_parameters']}")
        else:
            # Có thể là state_dict thuần (không bọc trong dict Lightning)
            n_params = sum(v.numel() for v in checkpoint.values() if hasattr(v, "numel"))
            print(f"\nSố tensor: {len(checkpoint)}")
            print(f"Tổng số tham số: {n_params:,}")
            if n_params < 1_000_000:
                print(
                    "[!] Số tham số quá ít so với PARSeq thật (thường 20-30 triệu). "
                    "Nghi ngờ đây KHÔNG PHẢI full checkpoint — có thể chỉ là 1 phần "
                    "(ví dụ optimizer state, hoặc lưu nhầm)."
                )
    else:
        print("Checkpoint không phải dict — cấu trúc khác thường, cần xem thủ công.")

    print("\n--- Kết luận ---")
    print("Nếu số tham số < 1 triệu: rất có thể file bị thiếu/hỏng, nên tải lại")
    print("hoặc hỏi lại người giữ checkpoint gốc trong team.")
    print("Nếu số tham số hợp lý (~20-30 triệu): file OK, có thể cắm vào model_server.py.")


if __name__ == "__main__":
    main()

# Kiểm tra bàn giao — 2026-09-14

Đã thực hiện tại workspace Windows, dùng các môi trường Python và dependency
đang cài. Kết quả này không thay cho thử nghiệm cài hoàn toàn mới trên máy nhận.

| Kiểm tra | Kết quả |
| --- | --- |
| PowerShell parser cho `setup_handoff.ps1` | Không có lỗi cú pháp |
| Ranh giới bản xuất/đường dẫn restore | 7 test đạt trong `scripts/tests/test_publish.py` |
| Mẫu token GitHub/HF, AWS access key, private-key header trên file xuất | Không phát hiện; đây là scan mẫu, không phải bảo đảm kiểm toán toàn diện |
| Video tổng hợp 4 giây → ZIP → WP00 → WP01 → WP02 → WP05 | Thành công, một video, 6 keyframe |
| WP03 validate strict và MetaCLIP2 GPU build | Hoàn tất, 6 vector, index HNSW, dimension 1024 |
| WP03 inspect | status ok, model complete, hash index/mapping khớp |
| WP04 export → metadata preprocess → validate | valid true, errors rỗng |
| API TV1/TV3 trên cổng thử riêng | Health HTTP 200 |
| TV4 qua TestClient, upstream TV1/TV3 thật | Health 200; POST /kis/search trả 200, status ok, candidate DEMO_V001, visual provenance, không degraded |
| `npm run build` | Thành công sau khi tách `tsconfig.build.json` cho mã ứng dụng |
| Model bundle | 35 file runtime, khoảng 12.00 GB, thêm model card; verify đạt 36 file và SHA-256 |

`npm run build` trước đó lỗi vì `tsconfig.json` bao gồm fixture test cũ không khớp
contract mới. Build hiện dùng `tsconfig.build.json` (kế thừa strict options, chỉ
include src). `tsconfig.json` và lệnh Vitest vẫn giữ để xử lý test riêng. Không
khẳng định toàn bộ test UI hay typecheck test hiện đạt. Vite còn cảnh báo plugin
React dùng tùy chọn đã deprecated; cảnh báo không làm build thất bại.

Chưa thực hiện trong lượt bàn giao này: cài dependency vào toàn bộ venv mới,
dựng lại tất cả video trong DATASETS_ZIP, chạy lại ensemble bốn model trên toàn
corpus, suy luận OCR/ASR/object đầy đủ, kiểm tra UI bằng trình duyệt tương tác,
restore model trên máy thứ hai hoặc tải lại từ repo HF mới.

Không có thao tác push GitHub hoặc upload Hugging Face. Người bàn giao tự điền
repo public, link nguồn dataset, SHA commit model và hoàn tất nguồn/điều kiện
phân phối checkpoint riêng trước khi xuất bản. Dữ liệu tổng hợp và log kiểm tra
đặt trong `_handoff_check/`, được loại khỏi cả hai gói xuất.

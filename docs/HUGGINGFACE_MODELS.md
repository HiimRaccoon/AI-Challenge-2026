# Model và trọng số lên Hugging Face

## 1. Nội dung chính xác

Gói chuẩn chứa tám thành phần trong [model card](MODEL_CARD.md). Script đọc đúng
checkpoint mà code gọi dưới `tv2/WP03/model-cache`, `tv3/WP04/weights` và ba snapshot
HF được pin trong worker. Không upload toàn bộ `weights/` hoặc `.cache/`.
Các file ở `weights/rec/VietOCR-best.pth`, `weights/detect/model_yolov8.pt`,
`weights/detect/model_0033999.pth` không được cấu hình hiện tại sử dụng.

Dung lượng dự kiến khoảng 12 GB; cần thêm chỗ cho bản sao chuẩn bị upload.
Model public vẫn có điều kiện sử dụng riêng. Bổ sung nguồn và điều kiện phân phối
checkpoint PARSeq-VN của nhóm vào `docs/MODEL_CARD.md` trước khi xuất bản.
Giữ các thông báo bản quyền đi kèm model. Nguồn:
[MetaCLIP2](https://huggingface.co/facebook/metaclip-2-worldwide-huge-quickgelu),
[ChunkFormer](https://huggingface.co/khanhld/chunkformer-ctc-large-vie).

## 2. Cài công cụ upload riêng

PowerShell ở gốc dự án. Tách môi trường này khỏi Transformers của worker.

```powershell
py -3.12 -m venv .venv-handoff
.\.venv-handoff\Scripts\python.exe -m pip install --upgrade huggingface_hub
.\.venv-handoff\Scripts\hf.exe auth login
.\.venv-handoff\Scripts\hf.exe auth whoami
```

Nhập token có quyền write ở màn hình đăng nhập của CLI, không ghi vào code/chat.

## 3. Xuất và kiểm tra gói model

```powershell
.\.venv-handoff\Scripts\python.exe scripts/model_bundle.py prepare --list
.\.venv-handoff\Scripts\python.exe scripts/model_bundle.py prepare
.\.venv-handoff\Scripts\python.exe scripts/model_bundle.py verify
```

Script mặc định chỉ đọc cache đã có, không tự tải model hay gửi file đi.
Nếu một snapshot thiếu, tải đúng revision rồi chạy lại (ví dụ MetaCLIP2):

```powershell
.\.venv-handoff\Scripts\hf.exe download facebook/metaclip-2-worldwide-huge-quickgelu --revision c139061af7b10fdb2e754b60d2b1182a3d5526c2
```

Revision các model còn lại nằm trong model card. Nếu báo thiếu checkpoint local,
đặt bản checkpoint tương ứng vào **đúng đường dẫn báo lỗi**; không đổi tên một
model khác để thay thế. Thư mục xuất có sẵn thì chọn `--bundle _hf_export_v2`
cho cả prepare/verify và dùng đúng thư mục đó ở bước upload.

`verify` kiểm tra hash và file thừa. Xem `bundle-manifest.json`, `README.md`,
`project/`, `hub/` trước upload. Chỉ gói này được gửi lên HF.

## 4. Tạo model repository public và upload

Hugging Face → New Model → điền owner/tên → Public → Create model.
Thay biến sau bằng `owner/repo` của bạn, không dùng URL đầy đủ:

```powershell
$ModelRepo = "nguyenviet21/AIC26-models"
.\.venv-handoff\Scripts\hf.exe upload $ModelRepo .\_hf_export . --repo-type model
```

Sau khi xong, mở Files and versions, lấy SHA commit và ghi vào hướng dẫn chạy
trong GitHub. Repo model mới chỉ nên chứa model card, manifest và hai thư mục
model; không kéo cả thư mục dự án vào giao diện web. HF có thể thêm `.gitattributes`
cho hạ tầng lưu trữ; khi tải chỉ chọn các file bundle như bên dưới.

## 5. Người nhận tải và khôi phục

Ở gốc repo code, sau khi tạo `.venv-handoff` như bước 2:

```powershell
$ModelRepo = "nguyenviet21/AIC26-models"
$ModelRevision = "a968a07f481690729db9bf3b811c6a41768356e5"
.\.venv-handoff\Scripts\hf.exe download $ModelRepo --revision $ModelRevision --local-dir _hf_export --include "project/*" "hub/*" "README.md" "bundle-manifest.json"
.\.venv-handoff\Scripts\python.exe scripts/model_bundle.py restore
$env:HF_HUB_CACHE = Join-Path (Get-Location).Path ".cache\huggingface\hub"
```

Restore kiểm tra SHA-256, từ chối ghi đè file khác nội dung và giữ nguyên ID/revision
HF gốc trong cache. Không sửa `model_id` thành ID của repo bundle: worker còn pin
ID trong mã nguồn, còn repo bundle chứa nhiều model với cấu trúc khác một model
Transformers độc lập. Đặt `HF_HUB_CACHE` ở mọi terminal build TV2 và chạy TV4.

BEiT-3 còn cần tạo lock bằng `wp03 lock-model` sau restore; xem hướng dẫn chạy đầy
đủ. Mã nguồn của UniLM/PARSeq/Perception được lấy theo commit trong hướng dẫn;
code wrapper của dự án nằm trên GitHub, BGE-VL loader đi cùng snapshot trong HF.

## 6. Model tùy chọn Qwen cho VQA

Code dùng Ollama tag `qwen2.5:3b-instruct`. Hướng dẫn chuẩn tải bằng `ollama pull`.
Nếu cần đưa cả bản Ollama đang dùng lên HF, tạo **model repo riêng**: dùng
`ollama show qwen2.5:3b-instruct --modelfile` để lấy chính xác đường dẫn GGUF từ
dòng `FROM`, sao chép riêng file đó vào thư mục mới, thêm model card/nguồn và
SHA-256, rồi upload thư mục mới bằng `hf upload`. Người nhận tải GGUF, tạo
`Modelfile` với `FROM ./ten-model.gguf`, chạy
`ollama create aic-qwen -f Modelfile`, đặt `TV4_OLLAMA_MODEL=aic-qwen` và
`TV4_EXTERNAL_ASSIST_MODEL=aic-qwen` trước khi chạy TV4. Kiểm tra template/parameters
bằng `ollama show` để giữ tương đương tag gốc. Không upload toàn bộ `.ollama/`.
Gói tám model không tự thu gom model Ollama hay trọng số dịch vụ VLM bên ngoài.
Semantic transcript embeddings qua `WP04_EMBEDDING_*` cũng là tích hợp tùy chọn,
chưa có model mặc định được chọn trong profile handoff.

Tài liệu CLI chính thức: [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli),
[upload file](https://huggingface.co/docs/huggingface_hub/guides/upload),
[Ollama Modelfile/GGUF](https://docs.ollama.com/modelfile).

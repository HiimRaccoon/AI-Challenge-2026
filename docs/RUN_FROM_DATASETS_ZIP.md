# Từ DATASETS_ZIP đến UI và truy vấn thử

Các lệnh dùng Windows PowerShell, chạy theo thứ tự. Thư mục gốc chứa `tv1/` đến
`tv5/`. Quy ước `run_handoff_v1` là run mới, tách khỏi dữ liệu xử lý sẵn của tác giả.
Không cần gói `tv1_data.zip`, `tv3_data.zip` hay index có sẵn.

## 1. Lấy code và ZIP đầu vào

Người xuất bản điền bốn thông tin:

- GitHub: `https://github.com/VietNguyen-21/AIC26.git`
- Nguồn tải bộ ZIP đã có: `https://huggingface.co/datasets/nguyenviet21/AIC26-dataset/tree/630afbf5c58fac1c6cb28f2c833c754aa9fae9aa`
- Model bundle: `nguyenviet21/AIC26-models`
- Commit model bundle: `a968a07f481690729db9bf3b811c6a41768356e5`

```powershell
git clone https://github.com/VietNguyen-21/AIC26.git final_aic
cd final_aic
```

Tải bộ ZIP từ link nguồn vào `final_aic/DATASETS_ZIP/`. Nếu bên ngoài là một gói
`datasetzip.zip`, giải nén lớp bao ngoài để `Videos_*.zip` nằm trực tiếp trong
`DATASETS_ZIP/`. Nếu thư mục của bạn tên `datasetzip`, thay đường dẫn tương ứng
ở lệnh WP00. Không giải nén lại từng video ZIP bằng tay.

Bộ trên máy tác giả có `Videos_L21_a.zip` … `Videos_L30_a.zip`, các phần
`Videos_L26_a.zip` … `Videos_L26_e.zip`, cùng ZIP Keyframes/map/media-info/objects/
clip-features. Pipeline WP00 chỉ lấy `.mp4`; pipeline sau tự tạo keyframe, metadata,
audio và index. Các ZIP feature/keyframe có sẵn không dùng trong luồng này.
Nếu chỉ có keyframe ZIP mà thiếu video ZIP thì chưa đủ cho giải mã/stream video.

```powershell
Get-ChildItem DATASETS_ZIP -Filter 'Videos*.zip'
```

Để thử nhanh, bắt đầu với bộ ZIP video nhỏ trong thư mục riêng và truyền thư mục
đó cho WP00. Khi mở rộng corpus, dùng run mới và sửa đồng bộ đường dẫn/config;
không trộn index của hai bộ frame khác nhau.

## 2. Phần mềm và dung lượng

Cài Python **3.12**, Git, Node.js **22.12 trở lên trong nhánh 22** (hoặc bản mới
tương thích), FFmpeg + ffprobe vào PATH. UI đang khóa Vite 8 trong package-lock;
không dùng Node 18. Các worker trong hướng dẫn dùng PyTorch 2.11.0/cu128 và
torchvision 0.26.0, theo môi trường đang có trong dự án.

Luồng GPU cần NVIDIA và driver tương thích CUDA 12.8. MetaCLIP2 checkpoint trên
đĩa khoảng 7.5 GB; VRAM/RAM và thời gian phụ thuộc số video, batch và model.
Máy tác giả dùng GPU 12 GB. Không coi đây là bảo đảm mọi cấu hình đều đủ bộ nhớ.
Chừa chỗ cho ZIP + MP4 giải nén + audio + keyframe + embedding/index + cache model
và bản tải model. Toàn bộ corpus có thể mất nhiều giờ; không cần dựng lại mỗi lần mở UI.

```powershell
py -3.12 --version
node --version
git --version
ffmpeg -version
ffprobe -version
nvidia-smi
```

## 3. Thiết lập môi trường

Ở gốc repo:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup_handoff.ps1
```

Script tạo venv riêng cho TV1, TV3, TV4, TV2 coordinator, MetaCLIP2, chạy `npm ci`
và build UI. Script dừng nếu một lệnh cài đặt lỗi. Không cần activate venv.
Các pin trong worker lấy từ môi trường hiện có; đây chưa phải lock toàn bộ
dependency cho mọi hệ điều hành. Nếu pip/GPU báo lỗi, giải quyết lỗi đó trước
bước dựng index. Bản `-Full` được mô tả ở bước 10.

## 4. Lấy model

### Cách A: bundle Hugging Face của người bàn giao

Làm bước tải/restore trong [HUGGINGFACE_MODELS.md](HUGGINGFACE_MODELS.md), sau đó:

```powershell
$env:HF_HUB_CACHE = Join-Path (Get-Location).Path '.cache\huggingface\hub'
```

Đặt lại biến trên trong mỗi terminal build TV2/chạy TV4. Bundle khôi phục model
dưới ID/revision gốc; không thay ID trong worker.

### Cách B: chỉ cần khởi động visual trước khi có bundle

```powershell
$env:HF_HUB_CACHE = Join-Path (Get-Location).Path '.cache\huggingface\hub'
.\tv2\WP03\.venvs\metaclip2\Scripts\python.exe -c "from huggingface_hub import snapshot_download; print(snapshot_download('facebook/metaclip-2-worldwide-huge-quickgelu', revision='c139061af7b10fdb2e754b60d2b1182a3d5526c2'))"
```

Bước này cần Internet và tải cả cấu hình/tokenizer. Khi chạy, MetaCLIP2 mặc định
chỉ đọc cache. Không dùng `--local-dir weights/...` tùy ý rồi mong worker tự tìm thấy.

## 5. Tạo dữ liệu từ video ZIP (TV1)

Tất cả lệnh dưới chạy tại **gốc repo**. Không bỏ `--manifest`: mặc định của WP01
trỏ về run cũ. Dùng cùng một run ID trong cả chuỗi.

```powershell
.\tv1\.venv\Scripts\python.exe tv1/wp00_data_intake.py --archives_dir DATASETS_ZIP --raw_dir tv1/data/raw --run_dir tv1/data/runs/run_handoff_v1
.\tv1\.venv\Scripts\python.exe tv1/wp01_media_probe.py --manifest tv1/data/runs/run_handoff_v1/manifest/corpus_manifest.parquet --raw_dir tv1/data/raw --run_dir tv1/data/runs/run_handoff_v1 --preprocess_run_id run_handoff_v1
.\tv1\.venv\Scripts\python.exe tv1/wp02_shot_keyframe.py --raw_dir tv1/data/raw --run_dir tv1/data/runs/run_handoff_v1 --preprocess_run_id run_handoff_v1 --target_eval_fps 10 --min_gap_ms 250 --max_gap_ms 1000 --max_workers 2
.\tv1\.venv\Scripts\python.exe tv1/wp05_temporal_adj.py --run_dir tv1/data/runs/run_handoff_v1
```

Chỉ chạy lệnh tiếp theo khi lệnh trước hoàn tất; kiểm tra log trong run nếu video
bị reject/decode lỗi. WP00 có thể ghi lỗi ZIP vào log mà không trả exit code lỗi,
nên cần kiểm tra manifest có video thật và không bỏ sót ZIP mong muốn.

```powershell
.\tv1\.venv\Scripts\python.exe -c "import pyarrow.parquet as p; from pathlib import Path; r=Path('tv1/data/runs/run_handoff_v1'); m=p.read_table(r/'manifest/corpus_manifest.parquet'); f=p.read_table(r/'frames.parquet'); print('videos',m.num_rows,'frames',f.num_rows); assert m.num_rows>0 and f.num_rows>0"
```

Các đầu ra cần có: `manifest/corpus_manifest.json`, `frames.parquet`, keyframe,
audio và bảng temporal trong `tv1/data/runs/run_handoff_v1/`. Giữ nguyên video tại
`tv1/data/raw/`; nếu chuyển dự án sang máy khác, dựng lại manifest từ ZIP ở máy mới
để tránh đường dẫn máy cũ.

## 6. Dựng visual index thật (TV2)

Ở gốc repo đặt cache rồi chuyển tới WP03:

```powershell
$env:HF_HUB_CACHE = Join-Path (Get-Location).Path '.cache\huggingface\hub'
cd tv2/WP03
.\.venvs\coordinator\Scripts\python.exe -m wp03 validate --data-root ../../tv1/data/runs/run_handoff_v1 --frames frames.parquet
.\.venvs\coordinator\Scripts\python.exe -m wp03 build --data-root ../../tv1/data/runs/run_handoff_v1 --frames frames.parquet --run-id run_handoff_v1 --config configs/handoff-metaclip2.yaml --runtime-root . --runtime-profile configs/runtime.windows.yaml --content-validation strict --artifact-root artifacts/handoff-metaclip2 --code-version handoff-v1 --resume
.\.venvs\coordinator\Scripts\python.exe -m wp03 inspect --artifact-root artifacts/handoff-metaclip2
cd ../..
```

`--frames` là đường dẫn **tương đối dưới data-root**, không phải đường dẫn từ cwd.
`inspect` phải trả `status: ok`, model complete và vector_count > 0. Khi bị ngắt,
chạy lại build với `--resume`, giữ nguyên dữ liệu/config/code. Khi thay code sản
xuất, dùng commit Git cho `--code-version` và rebuild vào thư mục artifact mới.

## 7. Chuẩn bị TV3 metadata

Luồng đầu tiên bật metadata, tắt OCR/ASR/object. Nó đủ để mở UI và tìm kiếm visual
trên video thật; chưa tạo bằng chứng chữ/tiếng nói cho VQA. Nâng cấp ở bước 10.

```powershell
cd tv3/WP04
.\.venv\Scripts\python.exe -m wp04.cli export-tv1-manifests --tv1-run-dir ../../tv1/data/runs/run_handoff_v1 --output-dir data/inputs/run_handoff_v1
.\.venv\Scripts\python.exe -m wp04.cli preprocess --run-dir data/runs/run_handoff_v1 --preprocess-run-id run_handoff_v1 --artifact-set-id metadata-v1 --frames-json data/inputs/run_handoff_v1/frames.json --audio-json data/inputs/run_handoff_v1/audio.json --config configs/handoff-metadata.yaml
.\.venv\Scripts\python.exe -m wp04.cli validate --run-dir data/runs/run_handoff_v1 --preprocess-run-id run_handoff_v1 --artifact-set-id metadata-v1 --frames-json data/inputs/run_handoff_v1/frames.json
cd ../..
```

Lệnh export từ chối ghi đè JSON cũ; chỉ thêm `--overwrite` nếu đang chủ động cập
nhật manifest của đúng run. Validation kiểm tra artifact; không tạo OCR/ASR khi
modality đang tắt. Không xem health HTTP 200 là bằng chứng mọi modality đầy đủ.

## 8. Chạy bốn terminal để mở UI

Mỗi terminal bắt đầu tại **gốc repo**. Giữ các terminal chạy trong lúc dùng UI.
TV2 được TV4 gọi qua coordinator và worker, không cần mở HTTP server TV2 riêng.

**Terminal 1 — TV1, cổng 8000**

```powershell
.\tv1\.venv\Scripts\python.exe tv1/wp06_api_server.py --run_dir tv1/data/runs/run_handoff_v1 --port 8000
```

**Terminal 2 — TV3, cổng 8100**

```powershell
$env:WP04_API_RUN_DIR = 'tv3/WP04/data/runs/run_handoff_v1'
$env:WP04_API_PREPROCESS_RUN_ID = 'run_handoff_v1'
$env:WP04_API_ARTIFACT_SET_ID = 'metadata-v1'
$env:WP04_API_FRAMES_JSON = 'tv3/WP04/data/inputs/run_handoff_v1/frames.json'
.\tv3\WP04\.venv\Scripts\python.exe -m uvicorn wp04.api:app --host 127.0.0.1 --port 8100
```

**Terminal 3 — TV4, cổng 8200**

```powershell
$env:HF_HUB_CACHE = Join-Path (Get-Location).Path '.cache\huggingface\hub'
$env:TV4_CONFIG = 'tv4/configs/handoff.yaml'
Remove-Item Env:TV4_FIXTURE_MODE -ErrorAction SilentlyContinue
$env:TV4_EXTERNAL_ASSIST_ENABLED = '0'
$env:TV4_EXTERNAL_VLM_ENABLED = '0'
.\tv4\.venv\Scripts\python.exe -m uvicorn tv4.api:app --host 127.0.0.1 --port 8200
```

**Terminal 4 — TV5, cổng 5173**

```powershell
cd tv5
npm.cmd run dev -- --strictPort
```

Mở **http://127.0.0.1:5173**. Kiểm tra trong terminal thứ năm:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/summary
Invoke-RestMethod http://127.0.0.1:8100/health
Invoke-RestMethod http://127.0.0.1:8200/health | ConvertTo-Json -Depth 12
```

Đợi TV4 `services.visual.available=true`, warmup `ready`. Warmup thất bại thì xem
terminal TV4, kiểm tra cache/GPU/index; không bật fixture để che lỗi dữ liệu thật.
API schema: http://127.0.0.1:8200/docs. Khi kết thúc, Ctrl+C trong từng terminal.
Lần mở sau chỉ làm bước 8; không cần chạy lại tiền xử lý.

## 9. Truy vấn thử và kiểm tra kết quả

### KIS (tìm một cảnh)

Ở ô `Search & Retrieval`, chọn Results Limit 20, nhập
`một người đang phát biểu trong phòng họp` hoặc
`a person speaking in a meeting room`, nhấn Enter/Search. Có thể thử
`xe máy di chuyển trên đường phố`. Đây là ví dụ tự tạo, không phải đáp án đề thi;
corpus nhỏ có thể không chứa cảnh mô tả. Chọn mô tả của video bạn đã nhập để kiểm
tra có chủ đích. Mở candidate vào Inspection, kiểm tra ảnh, video, timestamp và
frame ID cùng khớp. UI hiện video/ảnh thật mới xác nhận luồng media hoạt động.

Thử API độc lập UI (gốc repo, PowerShell):

```powershell
$body = @{ query_id='demo-kis-001'; query_text='a person speaking in a meeting room'; top_k=20; weights=@{visual=1.0; ocr=0.0; asr=0.0; object=0.0} } | ConvertTo-Json -Depth 4
$result = Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8200/kis/search -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($body)) -TimeoutSec 180
$result | ConvertTo-Json -Depth 10
```

Kiểm tra candidates có `video_id`, `frame_id`, `timestamp_ms` và visual provenance.
Index không rỗng có thể trả cảnh không liên quan với một truy vấn không có trong
corpus; HTTP thành công chưa chứng minh độ chính xác tìm kiếm.

### VQA và TRAKE

VQA: sau khi bật OCR/ASR và Ollama ở bước 10, chọn task VQA/QA, nhập mô tả cảnh,
rồi câu hỏi như `Dòng chữ trên biển hiệu là gì?`. Kiểm tra phần evidence và
grounding status, không coi câu trả lời thiếu bằng chứng là đáp án đã xác minh.

TRAKE: chọn task TRAKE, mô tả chuỗi như `một người bước vào phòng, sau đó ngồi
xuống ghế`, xem timeline và thứ tự frame trong cùng video. Thử API:

```powershell
$body = @{ query_text='a person enters a room then sits on a chair'; events=@('a person enters a room','the person sits on a chair'); top_k_sequences=3 } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8200/trake/align -ContentType 'application/json' -Body $body -TimeoutSec 300 | ConvertTo-Json -Depth 10
```

Profile handoff tắt WP09 exact refinement/feedback/dense QA. Chọn và xem keyframe
có sẵn hoạt động; bước sang frame exact lân cận đòi hỏi WP09 và chứng nhận cho
**chính run hiện tại**, không tái dùng chứng nhận `run_v1_batch1` cũ. Kiểm tra
frame trước khi xuất CSV/basket; hướng dẫn thử UI không khẳng định readiness thi đấu.

## 10. Nâng cấp cấu hình đầy đủ

### TV2 ensemble: BEiT-3 + BGE-VL + MetaCLIP2 + Perception

Khôi phục bundle ở bước 4A và chạy từ gốc repo:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup_handoff.ps1 -Full
git clone https://github.com/microsoft/unilm.git tv2/WP03/third_party/unilm
git -C tv2/WP03/third_party/unilm checkout 833df7e7832e5064a281131ee64a481afa8e5b95
$env:HF_HUB_CACHE = Join-Path (Get-Location).Path '.cache\huggingface\hub'
cd tv2/WP03
.\.venvs\coordinator\Scripts\python.exe -m wp03 lock-model --model beit3 --checkpoint model-cache/beit3/beit3_base_patch16_384_coco_retrieval.pth --lock-path model-locks/beit3.json
.\.venvs\coordinator\Scripts\python.exe -m wp03 build --data-root ../../tv1/data/runs/run_handoff_v1 --frames frames.parquet --run-id run_handoff_v1 --config configs/full.yaml --runtime-root . --runtime-profile configs/runtime.windows.yaml --content-validation strict --artifact-root artifacts/handoff-full --code-version handoff-v1 --resume
.\.venvs\coordinator\Scripts\python.exe -m wp03 inspect --artifact-root artifacts/handoff-full
cd ../..
```

Nếu checkout đã có, kiểm tra `git status`/commit trong checkout trước, không clone
đè. Sửa bản sao `tv4/configs/handoff.local.yaml` từ `handoff.yaml`:
`artifact_root: ../../tv2/WP03/artifacts/handoff-full`,
`online_models` và `temporal_online_models: [bge_vl, metaclip2]`,
`multi_moment_online_models: [beit3]`, `warmup_models: [bge_vl, metaclip2, beit3]`.
Perception nằm trong index đầy đủ, không mặc định tham gia mọi truy vấn online.
Chạy TV4 với `TV4_CONFIG=tv4/configs/handoff.local.yaml`. Không dùng default.yaml
cũ chứa đường dẫn F:/ của máy tác giả.

### TV3: CRAFT/PARSeq-VN, ChunkFormer, RF-DETR

Restore bundle trước. Chạy từ gốc repo, môi trường riêng để tránh xung đột timm:

```powershell
py -3.12 -m venv tv3/WP04/.venv-gpu
.\tv3\WP04\.venv-gpu\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.\tv3\WP04\.venv-gpu\Scripts\python.exe -m pip install -e './tv3/WP04[api]' rfdetr==1.9.3 easyocr==1.7.2 timm==0.9.16 pytorch-lightning==2.2.0.post0 transformers==5.15.1 huggingface_hub==1.28.0 nltk pillow soundfile python-dotenv jiwer pandas colorama pydub tqdm sentencepiece librosa langid
.\tv3\WP04\.venv-gpu\Scripts\python.exe -m pip install --no-deps chunkformer==1.2.2
git clone https://github.com/baudm/parseq.git tv3/WP04/third_party/parseq
git -C tv3/WP04/third_party/parseq checkout 1572487d138cdf0340fb2a2f1437cc0cca05eabf
.\tv3\WP04\.venv-gpu\Scripts\python.exe -m pip install --no-deps -e tv3/WP04/third_party/parseq
.\tv3\WP04\.venv-gpu\Scripts\python.exe tv3/WP04/deploy/model-server/check_parseq_checkpoint.py tv3/WP04/weights/parseq/best-parseq.ckpt
```

ChunkFormer được cài `--no-deps` sau các dependency inference để không kéo
DeepSpeed dùng cho training (môi trường đang chạy cũng không cài DeepSpeed).
Kiểm tra checkpoint có state_dict/hyper_parameters phù hợp; không thay PARSeq-VN
bằng checkpoint PARSeq tiếng Anh. Mở terminal model server từ gốc repo:

```powershell
.\tv3\WP04\.venv-gpu\Scripts\python.exe tv3/WP04/deploy/model-server/model_server.py --host 127.0.0.1 --port 8765
```

Model server tự đọc đường dẫn dưới `tv3/WP04/weights`; không cần `.env` chứa đường
dẫn máy tác giả. Xem log xác nhận từng engine nạp được: server có thể vẫn mở port
khi một model nạp lỗi. Trong terminal khác từ gốc repo:

```powershell
cd tv3/WP04
.\.venv-gpu\Scripts\python.exe -m wp04.cli doctor --config configs/default.yaml
.\.venv-gpu\Scripts\python.exe -m wp04.cli preprocess --run-dir data/runs/run_handoff_full --preprocess-run-id run_handoff_v1 --artifact-set-id multimodal-v1 --frames-json data/inputs/run_handoff_v1/frames.json --audio-json data/inputs/run_handoff_v1/audio.json --config configs/default.yaml
.\.venv-gpu\Scripts\python.exe -m wp04.cli validate --run-dir data/runs/run_handoff_full --preprocess-run-id run_handoff_v1 --artifact-set-id multimodal-v1 --frames-json data/inputs/run_handoff_v1/frames.json
cd ../..
```

Khi thành công, tắt model server 8765 để giải phóng GPU trước khi chạy TV2/TV4.
API TV3 đọc artifact đã tính, không cần giữ model server nếu dense QA đang tắt.
Đổi terminal TV3 ở bước 8 sang `WP04_API_RUN_DIR=tv3/WP04/data/runs/run_handoff_full`,
`WP04_API_ARTIFACT_SET_ID=multimodal-v1`, giữ `preprocess_run_id=run_handoff_v1` và
đường dẫn frames JSON. Sau đó restart TV3 và TV4, thử truy vấn OCR/ASR thật.

### VQA với Ollama (tùy chọn)

Cài Ollama, chạy `ollama pull qwen2.5:3b-instruct`, kiểm tra `ollama list`.
Đảm bảo Ollama phục vụ ở `http://127.0.0.1:11434` (`ollama serve` nếu chưa chạy).
Trước khi khởi động TV4:

```powershell
$env:TV4_OLLAMA_MODEL = 'qwen2.5:3b-instruct'
$env:TV4_EXTERNAL_ASSIST_MODEL = 'qwen2.5:3b-instruct'
$env:TV4_EXTERNAL_ASSIST_ENABLED = '1'
```

Giữ external VLM tắt nếu chưa cấu hình một model nhìn ảnh riêng. Qwen tag trên là
model văn bản; OCR/ASR evidence cung cấp thông tin cho trả lời. Qwen/OCR server và
visual worker có thể tranh VRAM, nên không chạy đồng thời bước tiền xử lý nặng.

## 11. Lỗi thường gặp

| Triệu chứng | Cách xử lý |
| --- | --- |
| Không có video/frame | Kiểm tra Videos ZIP và log WP00/WP01/WP02; truyền đúng manifest/run |
| `CUDA unavailable` | Kiểm tra nvidia-smi, đúng venv, wheel cu128 và driver |
| Model thiếu trong cache | Tải/restore đúng revision; đặt HF_HUB_CACHE ở terminal TV4 và TV2 |
| Index không complete | Chờ build hoàn tất, chạy inspect; không dùng index của run khác |
| OCR/ASR không có dữ liệu | Baseline tắt hai modality; làm bước 10 và đổi artifact-set TV3 |
| Video/ảnh 404 | Kiểm tra TV1 run, raw media và registry trong handoff.yaml |
| UI báo offline | Kiểm tra ba health, config TV4 và Vite proxy; dùng port 5173 đúng |
| Exact neighbor unavailable | Chưa có WP09 certification của run hiện tại; xem keyframe có sẵn |
| Hết VRAM | Dừng model server tiền xử lý/Ollama không cần dùng, giảm batch build |

Các lệnh cài dependency đã được đối chiếu với code/môi trường tại thời điểm bàn
giao; xem [VALIDATION.md](VALIDATION.md) để biết kiểm tra thực tế và giới hạn.
Nguồn công cụ: [PyTorch CUDA wheels](https://pytorch.org/get-started/previous-versions/),
[Ollama Modelfile](https://docs.ollama.com/modelfile).

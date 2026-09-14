# Đưa mã nguồn lên GitHub public

## 1. Chuẩn bị bản xuất

Mở PowerShell tại thư mục gốc dự án. Lệnh dưới chỉ đọc và sao chép các đường dẫn
mã nguồn được chọn trong script; không commit, không upload, không xóa dữ liệu.

```powershell
python scripts/export_github.py --list
python scripts/export_github.py
```

Kết quả: `_github_export/` và `_github_export_manifest.json` (danh sách, dung lượng,
SHA-256 từng file). Nếu thư mục đã tồn tại, dùng `--output _github_export_v2`.
Mỗi lần sửa code/hướng dẫn phải xuất lại vào thư mục mới trước khi push.

| Đích | Nội dung |
| --- | --- |
| GitHub | Code TV1–TV5, test và fixture tổng hợp đi kèm test, cấu hình, dependency, `docs/`, `scripts/` |
| Hugging Face model repo | Các model trong [hướng dẫn model](HUGGINGFACE_MODELS.md), tokenizer, cấu hình model, mã nạp cần thiết, manifest |
| Giữ tại máy | DATASETS_ZIP/datasetzip, video, keyframe, audio, parquet, FAISS/embedding sinh ra, môi trường ảo, cache, token, bài nộp, đề thi, báo cáo cá nhân |

Không dùng `git add .` tại repo làm việc gốc: repo này đã theo dõi một số cache,
artifact và output từ trước. `.gitignore` không bỏ các file đã được Git theo dõi,
và một commit xóa file không xóa file đó khỏi lịch sử. Bản xuất tạo lịch sử mới
cho repo public riêng, không mang `.git` cũ theo. Không copy thêm toàn bộ `tv*/`
hay `weights/` vào bản xuất.

Các README triển khai cũ, `start_all.bat` dùng dữ liệu cũ và chứng nhận frame cho
run cũ không thuộc bản xuất. Điểm bắt đầu của người nhận là `README.md` mới.

## 2. Điền link trước khi xuất bản

Trong `docs/RUN_FROM_DATASETS_ZIP.md`, thay `https://github.com/VietNguyen-21/AIC26.git`,
`https://huggingface.co/datasets/nguyenviet21/AIC26-dataset/tree/630afbf5c58fac1c6cb28f2c833c754aa9fae9aa`, `nguyenviet21/AIC26-models` và
`a968a07f481690729db9bf3b811c6a41768356e5` bằng thông tin của bạn. Link dataset là nguồn tải ZIP đã có;
không có bước upload lại dataset lên GitHub/Hugging Face.

Làm phần model ở [HUGGINGFACE_MODELS.md](HUGGINGFACE_MODELS.md) trước để lấy commit
SHA của gói model, rồi chạy lại bước xuất mã nguồn.

## 3. Tạo repository trên GitHub

Đăng nhập GitHub → New repository → tên bạn muốn → Public → Create repository.
Tạo repo **rỗng**, không thêm README, LICENSE hoặc .gitignore tự động ở bước này.
Quyền sử dụng mã nguồn và quyền của từng model cần được ghi riêng; public không
đồng nghĩa với việc tất cả thành phần có cùng một giấy phép.

## 4. Commit và push từ bản xuất

Thay URL bên dưới bằng repo mới của bạn. Git có thể mở trình duyệt để đăng nhập.
Không đặt token trong URL hoặc trong file dự án.

```powershell
cd _github_export
git init -b main
git add .
git diff --cached --stat
git diff --cached --name-only
git status --short
```

Kiểm tra danh sách chỉ chứa nội dung GitHub trong bảng ở bước 1. Sau đó:

```powershell
git commit -m "Publish AIC source and reproducible dataset-to-UI guide"
git remote add origin https://github.com/VietNguyen-21/AIC26.git
git push -u origin main
```

Nếu Git yêu cầu danh tính, đặt `git config user.name "Tên của bạn"` và
`git config user.email "Email GitHub của bạn"` trong bản xuất, rồi commit lại.
Nếu bị `non-fast-forward`, repo đích không rỗng: chọn repo rỗng riêng hoặc kiểm tra
lịch sử repo đích; không dùng force push theo hướng dẫn này.

## 5. Kiểm tra người nhận

Clone repo public vào thư mục mới, làm theo [luồng ZIP → UI](RUN_FROM_DATASETS_ZIP.md).
Kiểm tra link tài liệu trên GitHub, tải model theo commit cố định, gọi health và
truy vấn KIS trên dữ liệu thật. Các bước cài đặt/tải model cần Internet; runtime
visual dùng cache đã tải. Không cần gửi `tv1_data.zip`, `tv3_data.zip` hay index ZIP.

Tham khảo chính thức: [đưa code cục bộ lên GitHub](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github).

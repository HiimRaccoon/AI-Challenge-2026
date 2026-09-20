# AIC2026 – Multimodal Video Retrieval & Verification System

<p align="center">
  <b>AI Challenge Thành phố Hồ Chí Minh 2026</b><br>
  Đội <b>SS009.Q24</b>
</p>

<p align="center">
  <b>Score: 72.2 / 84</b>
</p>

---

## About the Competition

Đây là hệ thống **truy xuất và kiểm chứng video đa phương thức** được phát triển bởi đội **SS009.Q24** để tham gia **AI Challenge Thành phố Hồ Chí Minh 2026**.

Hệ thống hỗ trợ ba dạng tác vụ chính:

- **KIS – Known-Item Search**
- **VQA – Visual Question Answering**
- **TRAKE – Temporal Retrieval and Alignment of Key Events**

Trong quá trình thi, đội **SS009.Q24** đạt **72.2 / 84 điểm**.

Mục tiêu của hệ thống là hỗ trợ người dùng tìm nhanh video/frame phù hợp từ tập dữ liệu lớn, sau đó kiểm tra lại kết quả trên video gốc, timeline và canonical frame trước khi lựa chọn kết quả cuối cùng.

---

## System Overview

Hệ thống được thiết kế theo hướng **multimodal video retrieval kết hợp human-in-the-loop verification**.

Thay vì chỉ sử dụng một nguồn thông tin, hệ thống kết hợp nhiều tín hiệu khác nhau:

- Visual information
- OCR
- ASR
- Object information
- Metadata và temporal information

Các kết quả từ nhiều nhánh retrieval được hợp nhất, xếp hạng và đưa lên giao diện để người dùng tiếp tục kiểm chứng.

Một nguyên tắc quan trọng của hệ thống là mọi kết quả đều được ánh xạ về:

```text
video_id + frame_id + timestamp_ms
```

nhằm đảm bảo retrieval, timeline và frame cuối cùng sử dụng cùng một hệ quy chiếu.

---

## Methodology

Pipeline tổng quát:

```text
Video Dataset
      │
      ▼
Preprocessing & Canonical Frame Mapping
      │
      ├── Visual Embedding
      ├── OCR
      ├── ASR
      ├── Object Detection
      └── Metadata
      │
      ▼
Multimodal Retrieval
      │
      ▼
Weighted Reciprocal Rank Fusion
      │
      ▼
Temporal Deduplication
      │
      ▼
KIS / VQA / TRAKE
      │
      ▼
Human-in-the-loop Verification
      │
      ▼
Final Submission
```

### Visual Retrieval

Hệ thống sử dụng nhiều vision-language model để biểu diễn nội dung video dưới dạng embedding, bao gồm:

- **BGE-VL**
- **MetaCLIP2**
- **BEiT-3**
- **Perception Encoder**

Các embedding được lập chỉ mục bằng **FAISS-HNSW** để hỗ trợ truy xuất nhanh các frame có nội dung gần với query.

Tùy loại truy vấn và runtime configuration, hệ thống có thể lựa chọn các model phù hợp thay vì bắt buộc chạy toàn bộ model cùng lúc.

### Multimodal Evidence

Visual retrieval được bổ sung bằng nhiều nguồn evidence:

- **OCR:** CRAFT + PARSeq-VN
- **ASR:** Chunkformer-CTC-Large-Vie + Silero VAD
- **Object Detection:** RF-DETR
- **Metadata:** thông tin bổ trợ cho việc lọc và kiểm chứng candidate

Các nguồn evidence này giúp tăng khả năng phân biệt giữa những candidate có nội dung thị giác tương tự nhau.

### Rank Fusion

Các retrieval branch có thang điểm khác nhau nên hệ thống sử dụng **Weighted Reciprocal Rank Fusion (Weighted RRF)** để hợp nhất kết quả.

```text
Score(c) = Σ [ w_m × q_m(c) / (k + rank_m(c)) ]
```

Trong đó:

- `w_m`: trọng số của modality
- `q_m(c)`: độ tin cậy của candidate
- `rank_m(c)`: thứ hạng candidate trong modality tương ứng
- `k = 60`

Sau bước fusion, hệ thống tiếp tục temporal deduplication trước khi trả danh sách candidate cho người dùng.

---

## Backend System

Backend được xây dựng bằng **FastAPI** và đóng vai trò trung tâm trong quá trình truy xuất.

Backend nhận query từ giao diện, phân tích loại thông tin cần tìm và gọi các retrieval branch phù hợp như Visual, OCR, ASR, Object hoặc Metadata. Sau đó, kết quả từ các nhánh được hợp nhất bằng Weighted RRF, loại bỏ các candidate trùng nhau theo thời gian và trả danh sách kết quả về frontend.

Luồng xử lý chính:

```text
Frontend Query
      │
      ▼
FastAPI Backend
      │
      ▼
Query Router
      │
      ├── Visual Retrieval
      ├── OCR / ASR Retrieval
      └── Object / Metadata
      │
      ▼
Weighted RRF
      │
      ▼
Temporal Deduplication
      │
      ▼
Top Candidates
      │
      ▼
Frontend
```

Backend cũng chịu trách nhiệm giữ thống nhất thông tin `video_id`, `frame_id` và `timestamp_ms` để frontend có thể mở đúng video và frame tương ứng.

---

## Task Processing

### KIS – Known-Item Search

```text
Query
  ↓
Multimodal Retrieval
  ↓
Rank Fusion
  ↓
Candidate Inspection
  ↓
Canonical Frame Selection
```

KIS tập trung vào việc tìm đúng video hoặc frame dựa trên mô tả của người dùng.

Sau retrieval, người dùng kiểm tra lại video và timeline trước khi lựa chọn frame cuối cùng.

### VQA – Visual Question Answering

```text
Question
  ↓
Multimodal Evidence Retrieval
  ↓
Evidence Pack
  ↓
Answer Verification
  ↓
Supporting Canonical Frame
```

VQA không chỉ yêu cầu tìm đúng video mà còn cần xác định evidence liên quan tới câu hỏi và supporting frame tương ứng.

### TRAKE – Temporal Retrieval and Alignment of Key Events

```text
Multi-event Query
  ↓
Event Decomposition
  ↓
Candidate Retrieval
  ↓
Temporal Sequence Matching
  ↓
Chronological Verification
```

TRAKE xử lý truy vấn gồm nhiều sự kiện có thứ tự thời gian.

Các candidate cần được kiểm tra để đảm bảo các event thuộc cùng video và xuất hiện đúng thứ tự.

---

## Human-in-the-loop Verification

Hệ thống áp dụng chiến lược:

> **Tăng recall ở tầng retrieval, giảm false positive ở tầng inspection.**

Candidate do hệ thống tìm được chưa được xem ngay là kết quả cuối cùng.

Người dùng có thể tiếp tục kiểm tra bằng:

- Video gốc
- Timeline
- Neighboring keyframes
- Canonical frame
- OCR / ASR / Object evidence
- Metadata

Điều này đặc biệt hữu ích với những candidate gần nhau về nội dung hoặc thời gian.

---

## System Interface

Giao diện được xây dựng bằng **React + TypeScript** theo hướng một **inspection cockpit**, giúp người dùng truy vấn, xem candidate, kiểm tra video và quản lý kết quả nộp trong cùng một hệ thống.

> **Khu vực bên dưới được chừa để bổ sung hình ảnh UI của hệ thống.**

### Query & Retrieval Workspace

Cho phép người dùng:

- Nhập truy vấn
- Tìm kiếm video/frame
- Điều chỉnh trọng số các modality
- Xem danh sách candidate
- Pin candidate để tiếp tục kiểm tra

<!--
<p align="center">
  <img src="assets/ui/retrieval-workspace.png" width="900">
</p>
-->

### Inspection Workspace

Cho phép người dùng:

- Xem video gốc
- Di chuyển trên timeline
- Kiểm tra các keyframe lân cận
- Đối chiếu canonical frame
- Chọn frame cuối cùng

<!--
<p align="center">
  <img src="assets/ui/inspection-workspace.png" width="900">
</p>
-->

### Evidence & Submission Management

Hỗ trợ:

- Quản lý evidence
- Lưu candidate đã chọn
- Gom kết quả vào Submission Basket
- Kiểm tra kết quả trước khi nộp

<!--
<p align="center">
  <img src="assets/ui/evidence-submission.png" width="900">
</p>
-->

### Submission Verifier

Hỗ trợ kiểm tra lại frame trước khi xuất kết quả nộp, bao gồm các thao tác vi chỉnh:

```text
±1 frame
±5 frames
±25 frames
```

<!--
<p align="center">
  <img src="assets/ui/submission-verifier.png" width="900">
</p>
-->

---

## Tech Stack

| Thành phần | Công nghệ / Mô hình |
|---|---|
| Backend | FastAPI |
| Frontend | React, TypeScript |
| Video Processing | PyAV |
| Vector Search | FAISS-HNSW |
| Visual Models | BGE-VL, MetaCLIP2, BEiT-3, Perception Encoder |
| OCR | CRAFT, PARSeq-VN |
| ASR | Chunkformer-CTC-Large-Vie, Silero VAD |
| Object Detection | RF-DETR |
| Data Storage | Apache Parquet |
| Rank Fusion | Weighted Reciprocal Rank Fusion |

---

## Installation & Setup

README này chỉ tập trung giới thiệu tổng quan hệ thống và phương pháp.

Hướng dẫn chi tiết về:

- Chuẩn bị môi trường
- Cài đặt dependency
- Cấu hình dataset
- Cấu hình model
- Environment variables
- Preprocessing
- Build index
- Chạy backend
- Chạy frontend
- Troubleshooting

được đặt trong:

### ➡️ [SETUP.md](SETUP.md)

---

## Notes

Trong quá trình phát triển có nhiều module thử nghiệm và không phải tất cả đều được bật trong cùng một runtime configuration.

Khi tái lập hệ thống cần chú ý đến:

- Source code version
- Runtime configuration
- Model đang được kích hoạt
- Artifact/index tương ứng

Với những candidate gần nhau về nội dung hoặc thời gian, bước kiểm tra thủ công trên video và canonical frame vẫn đóng vai trò quan trọng.

---

## Achievement

**AI Challenge Thành phố Hồ Chí Minh 2026**

- **Team:** SS009.Q24
- **Score:** **72.2 / 84**

---

## Team

**SS009.Q24**

AI Challenge Thành phố Hồ Chí Minh 2026

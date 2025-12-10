# VNPT AI Hackathon - Track 2 (The Builder) - Team Underdog

Pipeline giải bài trắc nghiệm sử dụng LangGraph với Router-based architecture.

## Cấu trúc Project

```
project/
├── data/                  # Dữ liệu đầu vào
│   ├── val.json           # File validation
│   └── private_test.json  # File test chính thức
├── output/                # Kết quả đầu ra
├── src/
│   ├── agent/
│   │   ├── modules/       # Các solver modules
│   │   │   ├── math/      # Giải toán
│   │   │   ├── rag/       # Tra cứu kiến thức
│   │   │   ├── reading/   # Đọc hiểu văn bản
│   │   │   └── toxic/     # Phát hiện câu hỏi toxic
│   │   ├── router.py      # Phân loại câu hỏi
│   │   ├── graph.py       # LangGraph workflow
│   │   └── state.py       # Agent state
│   ├── client.py          # VNPT API client
│   ├── config.py          # Configuration
│   └── utils.py           # Utilities
├── predict.py             # Entry point
├── .env
├── pyproject.toml         # uv package manager
└── Dockerfile             # Docker build
```

## Cài đặt

### 1. Clone và cài đặt dependencies

```bash
# clone project 
git clone https://github.com/baeGil/vnptAI-Hackathon-2025.git

# vào thư mục project
cd project

# Cài đặt dependencies
uv sync
```

### 2. Cấu hình API Keys

Tạo file `.env` với nội dung:

```env
# Embedding model
VNPT_EMBEDDING_API_KEY=Bearer <your_token>
VNPT_EMBEDDING_TOKEN_KEY=<your_key>
VNPT_EMBEDDING_TOKEN_ID=<your_id>

# Large model
VNPT_LARGE_API_KEY=Bearer <your_token>
VNPT_LARGE_TOKEN_KEY=<your_key>
VNPT_LARGE_TOKEN_ID=<your_id>

# Small model
VNPT_SMALL_API_KEY=Bearer <your_token>
VNPT_SMALL_TOKEN_KEY=<your_key>
VNPT_SMALL_TOKEN_ID=<your_id>

# API Base URL
VNPT_API_BASE_URL=https://api.idg.vnpt.vn/data-service/v1/chat/completions

# Paths
DATA_DIR=./data
OUTPUT_DIR=./output
```

## Chạy Pipeline

### Chạy Local

```bash
uv run python predict.py
```

### Chạy với Docker

```bash
# Build image
docker build -t team_submission .

# Run
docker run --gpus all \
  -v $(pwd)/data:/code/data \
  -v $(pwd)/output:/output \
  team_submission
```

## Output

- `output/submission.csv` - File kết quả để nộp
- `output/inference_log.jsonl` - Log chi tiết từng câu hỏi

## Tính năng Checkpointing & Resume

Pipeline hỗ trợ **tự động resume** khi bị gián đoạn:

### Khi bị Rate Limit (429/401):
1. Pipeline dừng ngay lập tức
2. Tạo `submission_emergency.csv` với câu trả lời đã xử lý
3. Hiển thị hướng dẫn resume

### Để Resume:
1. Đợi quota API reset (hoặc đổi tokens trong `.env`)
2. Chạy lại: `python predict.py`
3. Hệ thống tự động detect `inference_log.jsonl` và chỉ xử lý câu chưa làm

### Model Usage:
| Node | Model | Mô tả |
|------|-------|-------|
| Router | `vnpt_small` | Phân loại câu hỏi |
| Math | `vnpt_large` → `vnpt_small` | Sinh code → Chọn đáp án |
| RAG | `vnpt_large` | Tra cứu + trả lời |
| Reading | `vnpt_large` | Đọc hiểu văn bản |
| Toxic | - | Trích xuất đáp án từ chối |
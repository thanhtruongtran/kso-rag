## Installation

1. Clone the repository and run the uv installation script:

   ```shell
   # clone this repo (replace with your own repository URL)
   git clone "link_repo"
   cd kso-rag

   # run the uv installation script (installs uv automatically if not present)
   bash scripts/run_uv.sh
   ```

   This script will:
   - Install uv package manager if not present
   - Create a virtual environment with Python 3.10
   - Install all dependencies using uv (significantly faster than conda/pip)
   - Set up PDF.js viewer
   - Launch the application

## Evaluation

The project uses **[DeepEval](https://deepeval.com/docs/getting-started-rag)** for RAG evaluation. Five metrics are supported:

| Metric | Đo lường | Cần `ground_truth_answer`? |
|---|---|---|
| `answer_relevancy` | Câu trả lời có đúng trọng tâm câu hỏi không | Không |
| `faithfulness` | Câu trả lời có bám vào context không | Không |
| `contextual_relevancy` | Context retrieved có liên quan câu hỏi không | Không |
| `contextual_precision` | Context liên quan có được xếp lên đầu không | **Có** |
| `contextual_recall` | Context có đủ thông tin để trả lời không | **Có** |

### 1. Cài DeepEval

```shell
# Kích hoạt venv của dự án trước
source .venv/bin/activate
pip install deepeval
```

### 2. Cấu hình API key (không cần gõ key vào lệnh chạy)

Chọn **một trong hai cách**; script sẽ đọc key từ biến môi trường hoặc file `.env`.

| Provider | Biến môi trường |
|---|---|
| `gemini` (mặc định) | `GOOGLE_API_KEY` |
| `openai` | `OPENAI_API_KEY` |
| `anthropic` | `ANTHROPIC_API_KEY` |
| `ollama` | Không cần (chạy local) |
| `azure` | `AZURE_OPENAI_API_KEY` + `AZURE_OPENAI_ENDPOINT` |

**Cách 1 – Gõ trong terminal mỗi lần mở session:**
```shell
export GOOGLE_API_KEY="AIza..."
```

**Cách 2 – Lưu vào file `.env` ở thư mục gốc repo (khuyến nghị):**
```shell
# Tạo/sửa file .env
echo 'GOOGLE_API_KEY=AIza...' >> .env
echo 'THEFLOW_SETTINGS=flowsettings' >> .env
```
Sau đó chỉ cần chạy `python scripts/run_evaluation.py ...` mà không cần gõ key vào lệnh.

### 3. Chuẩn bị dataset

Dataset là file **JSONL** (mỗi dòng một JSON) hoặc **JSON array**. Ví dụ (`scripts/evaluation_example_dataset.jsonl`):

```jsonl
{"question": "What is the main purpose of this system?", "ground_truth_answer": "A RAG Q&A system..."}
{"question": "What file formats are supported?"}
```

Các field:
- `question` (bắt buộc)
- `ground_truth_answer` (tùy chọn – cần thiết cho `contextual_precision` và `contextual_recall`)

### 4. Chạy evaluation

Sau khi đã set `GOOGLE_API_KEY` (export hoặc .env):

```shell
cd kso-rag
python scripts/run_evaluation.py \
  --dataset scripts/evaluation_example_dataset.jsonl \
  --output  evaluation_results.json
```

Tùy chọn:

```shell
# Chỉ chạy một số metrics
python scripts/run_evaluation.py \
  --dataset scripts/evaluation_example_dataset.jsonl \
  --metrics answer_relevancy faithfulness contextual_relevancy

# Dùng OpenAI làm judge
python scripts/run_evaluation.py \
  --provider openai --model gpt-4o-mini \
  --dataset scripts/evaluation_example_dataset.jsonl

# Dùng Ollama (local, không cần API key)
python scripts/run_evaluation.py \
  --provider ollama --model llama3 \
  --dataset scripts/evaluation_example_dataset.jsonl

# Đổi threshold
python scripts/run_evaluation.py \
  --dataset scripts/evaluation_example_dataset.jsonl \
  --threshold 0.7
```

### 5. Xem kết quả

Kết quả được ghi ra file `evaluation_results.json`:

```json
{
  "summary": {
    "total": 5,
    "failed": 0,
    "metrics_avg": {
      "Answer Relevancy": 0.82,
      "Faithfulness": 0.91,
      "Contextual Relevancy": 0.74
    }
  },
  "results": [
    {
      "question": "...",
      "answer": "...",
      "contexts": ["..."],
      "metrics": {
        "Answer Relevancy": {"score": 0.9, "passed": true, "reason": "..."}
      }
    }
  ]
}
```

### 6. Đánh giá bằng RAGBench (public dataset)

Bạn có thể dùng [RAGBench](https://huggingface.co/datasets/galileo-ai/ragbench) để benchmark hệ thống RAG:

1. Cài thư viện datasets:

```bash
uv pip install datasets
```

2. Tạo dataset eval từ một subset, ví dụ `hotpotqa` test split (lấy 200 mẫu):

```bash
cd kso-rag
python scripts/build_ragbench_dataset.py \
  --subset hotpotqa --split test --limit 200 \
  --output scripts/eval_ragbench_hotpotqa_test.jsonl
```

3. Chạy DeepEval trên dataset vừa tạo:

```bash
python scripts/run_evaluation.py \
  --dataset scripts/eval_ragbench_hotpotqa_test.jsonl \
  --output evaluation_results_hotpotqa.json
```

Cấu trúc file JSONL này giống hệt ví dụ ở trên (`question` + `ground_truth_answer`), nên có thể dùng lại cho nhiều lần chạy eval với cấu hình model / metric khác nhau.

#!/usr/bin/env python3
"""Run RAG evaluation using DeepEval on a question dataset.

DeepEval metrics supported:
  answer_relevancy      - Answer is relevant to the question
  faithfulness          - Answer is grounded in retrieved contexts
  contextual_relevancy  - Retrieved contexts are relevant to the question
  contextual_precision  - Relevant contexts ranked higher (needs ground_truth_answer)
  contextual_recall     - Contexts cover the expected answer (needs ground_truth_answer)

Prerequisites:
  1. Install deepeval:  uv pip install deepeval
  2. Cấu hình API key (chọn 1 trong 2 cách):
       Cách 1 – Biến môi trường trong terminal (mỗi lần mở terminal):
         export GOOGLE_API_KEY="AIza..."     # Gemini (mặc định)
         export OPENAI_API_KEY="sk-..."      # nếu dùng OpenAI
         export ANTHROPIC_API_KEY="..."      # nếu dùng Anthropic
       Cách 2 – File .env ở thư mục gốc repo (không cần gõ lại):
         Tạo file .env với nội dung:  GOOGLE_API_KEY=AIza...
         Script tự đọc .env khi chạy.
  3. Vector store + doc store đã được index (chạy app trước).
  4. THEFLOW_SETTINGS=flowsettings (có thể thêm vào .env).

Usage (sau khi đã set key bằng export hoặc .env):
  python scripts/run_evaluation.py \\
    --dataset scripts/evaluation_example_dataset.jsonl \\
    --output  evaluation_results.json

  # Dùng OpenAI làm judge:
  python scripts/run_evaluation.py --provider openai \\
    --dataset scripts/evaluation_example_dataset.jsonl

  # Ollama (local, không cần API key):
  python scripts/run_evaluation.py --provider ollama --model llama3 \\
    --dataset scripts/evaluation_example_dataset.jsonl

  # Chỉ chạy một số metrics:
  python scripts/run_evaluation.py \\
    --dataset scripts/evaluation_example_dataset.jsonl \\
    --metrics answer_relevancy faithfulness contextual_relevancy

Dataset format: JSONL or JSON array. Each record:
  {"question": "...", "ground_truth_answer": "...(optional)..."}

Reference: https://deepeval.com/docs/getting-started-rag
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)
os.environ.setdefault("THEFLOW_SETTINGS", "flowsettings")

# Load .env from repo root so GOOGLE_API_KEY / OPENAI_API_KEY etc. can be set there
try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

# DeepEval Settings validate Azure URL; set placeholder when not using Azure to avoid ValidationError
# If AZURE_OPENAI_ENDPOINT is missing or empty (e.g. in .env template),
# override with a placeholder so DeepEval's Settings validation passes.
if not os.environ.get("AZURE_OPENAI_ENDPOINT"):
    os.environ["AZURE_OPENAI_ENDPOINT"] = "https://not-used.invalid"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _build_rag_callables():
    """Return get_docs and get_answer for eval.

    - get_docs: dùng VectorRetrieval của app (giống RAG thật).
    - get_answer: gọi trực tiếp default ChatLLM với prompt đơn giản,
      tránh phụ thuộc vào AnswerWithContextPipeline.invoke (NotImplemented).
    """
    from kso_rag_core.base import HumanMessage
    from kso_rag_core.indices import VectorRetrieval
    from kso_rag_ui.components import get_docstore, get_vectorstore
    from kso_rag_ui.embeddings.manager import embedding_models_manager
    from kso_rag_ui.llms.manager import llms

    vs = get_vectorstore()
    ds = get_docstore()
    embedding = embedding_models_manager.get_default()
    llm = llms.get_default()

    retrieval = VectorRetrieval(
        embedding=embedding,
        vector_store=vs,
        doc_store=ds,
        top_k=10,
    )

    def get_docs(question: str):
        return retrieval.run(question)

    def get_answer(question: str, context: str) -> str:
        # Nếu có context, thêm vào prompt để mô phỏng RAG; nếu không, trả lời theo kiến thức LLM.
        if context.strip():
            prompt = (
                "Use the following context to answer the question. "
                "If the context is not sufficient, say you don't know.\n\n"
                f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
            )
        else:
            prompt = question

        resp = llm([HumanMessage(content=prompt)])
        return (resp.text or "").strip()

    return get_docs, get_answer


def main():
    from kso_rag_core.evaluation.deepeval_runner import ALL_METRICS

    parser = argparse.ArgumentParser(
        description="Run RAG evaluation with DeepEval.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to JSONL or JSON dataset (EvalRecord format).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation_results.json"),
        help="Output JSON file path (default: evaluation_results.json).",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        choices=ALL_METRICS,
        metavar="METRIC",
        help=(
            f"Metrics to compute. Choices: {ALL_METRICS}. "
            "Default: all 5 (contextual_precision and contextual_recall "
            "require ground_truth_answer in dataset)."
        ),
    )
    parser.add_argument(
        "--provider",
        default="gemini",
        choices=["gemini", "openai", "azure", "ollama", "anthropic"],
        help=(
            "LLM provider for DeepEval judge (default: gemini). "
            "gemini→GOOGLE_API_KEY, openai→OPENAI_API_KEY, "
            "anthropic→ANTHROPIC_API_KEY, ollama→no key needed."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Model name for the judge (optional). Defaults: "
            "gemini→gemini-2.0-flash, openai→gpt-4o-mini, "
            "anthropic→claude-3-haiku-20240307, ollama→llama3."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Pass/fail threshold per metric, 0–1 (default: 0.5).",
    )
    args = parser.parse_args()

    # Apply per-provider default model names
    _default_models = {
        "gemini": "gemini-2.0-flash",
        "openai": "gpt-4o-mini",
        "azure": "gpt-4o-mini",
        "anthropic": "claude-3-haiku-20240307",
        "ollama": "llama3",
    }
    if args.model is None:
        args.model = _default_models[args.provider]

    if not args.dataset.exists():
        logger.error("Dataset not found: %s", args.dataset)
        sys.exit(1)

    logger.info("Loading app components...")
    get_docs, get_answer = _build_rag_callables()

    from kso_rag_core.evaluation import run_deepeval

    logger.info("Starting DeepEval evaluation on: %s", args.dataset)
    logger.info("Judge: provider=%s  model=%s", args.provider, args.model)
    results, summary = run_deepeval(
        args.dataset,
        get_docs=get_docs,
        get_answer=get_answer,
        metrics=args.metrics,
        threshold=args.threshold,
        model=args.model,
        provider=args.provider,
    )

    out = {
        "summary": summary.model_dump(),
        "results": [r.model_dump() for r in results],
    }
    args.output.write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    logger.info("=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info("Total: %d | Failed: %d", summary.total, summary.failed)
    for metric, avg in summary.metrics_avg.items():
        passed = avg >= args.threshold
        status = "PASS" if passed else "FAIL"
        logger.info("  %-30s avg=%.3f  [%s]", metric, avg, status)
    logger.info("Results written to: %s", args.output)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

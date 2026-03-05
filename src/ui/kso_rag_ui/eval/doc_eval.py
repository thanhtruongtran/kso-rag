from __future__ import annotations

"""
Per-document RAG evaluation helpers (DeepEval-based).

This module is UI-layer glue that:
- Restricts retrieval to a single uploaded file (Source.id)
- Uses the app's default embedding + LLM
- Delegates metric computation to kso_rag_core.evaluation.run_deepeval

Usage (backend):
    summary, results = run_doc_eval(
        index=file_index_instance,
        source_id="<Source.id>",
        dataset_path="path/to/eval_<source_id>.jsonl",
        provider="gemini",
        model="gemini-2.0-flash",
    )
"""

from pathlib import Path
from typing import List, Tuple

import json
import os
from sqlalchemy import select
from sqlalchemy.orm import Session

from kso_rag_core.evaluation import EvalResult, EvalSummary, run_deepeval
from kso_rag_core.indices import VectorRetrieval
from kso_rag_ui.db.engine import engine
from kso_rag_ui.embeddings.manager import embedding_models_manager
from kso_rag_ui.llms.manager import llms

# Ensure GOOGLE_API_KEY is present in os.environ for DeepEval (GeminiModel)
try:  # pragma: no cover - best-effort env wiring
    if not os.environ.get("GOOGLE_API_KEY"):
        # Parse repo-root .env manually to avoid cwd issues
        from pathlib import Path as _Path

        _HERE = _Path(__file__).resolve()
        # src/ui/kso_rag_ui/eval/doc_eval.py -> repo root is 4 levels up
        _REPO_ROOT = _HERE.parents[4]
        env_path = _REPO_ROOT / ".env"
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                if key == "GOOGLE_API_KEY":
                    val = val.strip().strip('"').strip("'")
                    if val:
                        os.environ.setdefault("GOOGLE_API_KEY", val)
                    break
except Exception:
    # If this fails, user can still export GOOGLE_API_KEY manually
    pass


def _get_doc_ids_for_source(index, source_id: str) -> List[str]:
    """
    Return all doc_ids (targets in Index table) that belong to a given Source.id.
    """
    Index = index._resources["Index"]

    with Session(engine) as session:
        matches = session.execute(
            select(Index).where(
                Index.source_id == source_id,
                Index.relation_type == "document",
            )
        )
        return [row.target_id for (row,) in matches]


def build_eval_dataset_for_source(
    index,
    source_id: str,
    out_path: str | Path,
    num_questions: int = 10,
    max_chars: int = 8000,
) -> Path:
    """
    Auto-generate a small eval dataset (JSONL) for a single document.

    The dataset contains `num_questions` Q&A pairs created by the app's default LLM,
    based solely on the content of the selected document.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    doc_ids = _get_doc_ids_for_source(index, source_id)
    if not doc_ids:
        raise ValueError(
            f"No document chunks found for Source.id={source_id}; "
            "index may not be built yet."
        )

    ds = index._resources["DocStore"]
    docs = ds.get(doc_ids)

    # Sort docs in a deterministic order (page_label if present)
    docs = sorted(
        docs,
        key=lambda x: x.metadata.get("page_label", float("inf")),
    )

    full_text_parts: list[str] = []
    current_len = 0
    for d in docs:
        t = (d.text or "").strip()
        if not t:
            continue
        if current_len + len(t) > max_chars:
            # Cut off when exceeding max_chars to keep prompt manageable
            remaining = max_chars - current_len
            if remaining <= 0:
                break
            t = t[:remaining]
        full_text_parts.append(t)
        current_len += len(t)
        if current_len >= max_chars:
            break

    if not full_text_parts:
        raise ValueError(
            f"Document {source_id} has no textual content to build eval dataset."
        )

    full_text = "\n\n".join(full_text_parts)

    llm = llms.get_default()
    from kso_rag_core.base import HumanMessage

    prompt = f"""
You are generating an evaluation dataset for a Retrieval-Augmented Generation (RAG) system.

Given the document content below, create {num_questions} diverse question-answer pairs
in the SAME LANGUAGE as the document.

Requirements:
- Each question must be answerable using ONLY the given document content.
- Answers must be concise but precise (1–3 sentences).
- Cover different parts of the document (not all questions on the same sentence).
- Do NOT ask meta-questions (like "What is this document about?") only; mix fact and reasoning.

Return ONLY a JSON array (no extra text) of objects with the exact keys:
- "question": string
- "answer": string

Example of the expected JSON format:
[
  {{"question": "Q1?", "answer": "A1."}},
  {{"question": "Q2?", "answer": "A2."}}
]

Document content:
----------------
{full_text}
----------------
"""

    resp = llm([HumanMessage(content=prompt)])
    text = (resp.text or "").strip()

    # Some models may wrap JSON in ```json ... ``` fences. Try to be robust.
    def _extract_json_payload(raw: str) -> str:
        s = raw.strip()
        # Strip markdown code fences if present
        if s.startswith("```"):
            lines = s.splitlines()
            # remove first line (``` or ```json) and any trailing ``` line
            if len(lines) >= 3:
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines[-1].startswith("```"):
                    lines = lines[:-1]
                s = "\n".join(lines).strip()
        # Fallback: take content between first '[' and last ']'
        if "[" in s and "]" in s:
            start = s.find("[")
            end = s.rfind("]")
            if start != -1 and end != -1 and end > start:
                s = s[start : end + 1]
        return s

    payload = _extract_json_payload(text)

    try:
        data = json.loads(payload)
    except json.JSONDecodeError as e:  # pragma: no cover - defensive
        raise ValueError(
            "Failed to parse LLM output as JSON when generating eval dataset.\n"
            f"Output was:\n{text}"
        ) from e

    if not isinstance(data, list):
        raise ValueError("LLM output must be a JSON array of objects.")

    n_written = 0
    with out.open("w", encoding="utf-8") as f:
        for item in data:
            if not isinstance(item, dict):
                continue
            q = (item.get("question") or "").strip()
            a = (item.get("answer") or "").strip()
            if not q or not a:
                continue
            record = {"question": q, "ground_truth_answer": a}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1

    if n_written == 0:
        raise ValueError(
            "Generated eval dataset is empty; LLM did not produce valid Q&A pairs."
        )

    return out


def run_doc_eval(
    index,
    source_id: str,
    dataset_path: str | Path,
    *,
    settings: dict,
    user_id: str | int = "default",
    provider: str = "gemini",
    model: str = "gemini-2.0-flash",
) -> Tuple[EvalSummary, List[EvalResult]]:
    """
    Run DeepEval metrics for a single uploaded file (Source.id).

    Args:
        index: FileIndex instance (from IndexManager.indices).
        source_id: Source.id of the file to evaluate.
        dataset_path: JSON/JSONL EvalRecord dataset for this file only.
        provider: DeepEval provider (e.g. "gemini", "openai", "ollama").
        model: Judge model name for the provider.

    Returns:
        (summary, results) from kso_rag_core.evaluation.run_deepeval.
    """
    ds_path = Path(dataset_path)
    if not ds_path.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found for this document: {ds_path}"
        )

    # Ensure this Source actually has indexed chunks; reused from chunk viewer logic.
    doc_ids = _get_doc_ids_for_source(index, source_id)
    if not doc_ids:
        raise ValueError(
            f"No document chunks found for Source.id={source_id}; "
            "index may not be built yet."
        )

    # Build a retriever pipeline using the same settings as the app (top_k, mmr,
    # reranking, retrieval_mode, etc.), and restrict it to this file only.
    from kso_rag_ui.index.file.pipelines import DocumentRetrievalPipeline

    prefix = f"index.options.{index.id}."
    user_settings: dict = {}
    for key, value in settings.items():
        if key.startswith(prefix):
            user_settings[key[len(prefix) :]] = value

    # IMPORTANT:
    # DocumentRetrievalPipeline expects `selected` to be a list of Source IDs
    # (file_ids). It will internally map them to chunk_ids via the Index table.
    # Passing chunk-level doc_ids here would result in empty retrieval.
    retriever = DocumentRetrievalPipeline.get_pipeline(
        user_settings=user_settings,
        index_settings=index.config,
        selected=[source_id],
    )
    # Wire core resources (same pattern as FileIndex.get_retriever_pipelines)
    retriever.Source = index._resources["Source"]
    retriever.Index = index._resources["Index"]
    retriever.VS = index._resources["VectorStore"]
    retriever.DS = index._resources["DocStore"]
    retriever.FSPath = index._resources["FileStoragePath"]
    retriever.user_id = user_id

    # Use app's default chat LLM as answer generator (RAG-style: question + retrieved context)
    chat_llm = llms.get_default()

    def get_docs(question: str):
        # DocumentRetrievalPipeline already knows which doc_ids belong to this file
        # via its internal configuration (set in get_pipeline).
        return retriever(text=question)

    def get_answer(question: str, context: str) -> str:
        # Simple RAG-style prompt: use retrieved context to answer the question
        if context.strip():
            prompt = (
                "Use the following context to answer the question. "
                "If the context is not sufficient, say you don't know.\n\n"
                f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
            )
        else:
            # No context retrieved – still ask the question so DeepEval can judge
            prompt = question

        from kso_rag_core.base import HumanMessage

        resp = chat_llm([HumanMessage(content=prompt)])
        return (resp.text or "").strip()

    # Use DeepEval metrics via the custom runner that calls metric.measure()
    # directly (no dependency on TestRun / cloud APIs).
    results, summary = run_deepeval(
        dataset_path=ds_path,
        get_docs=get_docs,
        get_answer=get_answer,
        provider=provider,
        model=model,
    )

    return summary, results


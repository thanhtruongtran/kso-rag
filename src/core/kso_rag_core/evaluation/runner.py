"""Evaluation runner: runs retriever + answer pipeline on a dataset and computes metrics."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Callable, Optional

from .metrics import compute_answer_relevancy, compute_faithfulness
from .schema import EvalRecord, EvalResult, EvalSummary

logger = logging.getLogger(__name__)


def _load_dataset(path: Path) -> list[EvalRecord]:
    """Load EvalRecord list from JSON array or JSONL file."""
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        records = []
        for line in text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            records.append(EvalRecord.model_validate(json.loads(line)))
        return records
    data = json.loads(text)
    if isinstance(data, list):
        return [EvalRecord.model_validate(r) for r in data]
    if isinstance(data, dict) and "questions" in data:
        return [EvalRecord.model_validate(r) for r in data["questions"]]
    return [EvalRecord.model_validate(data)]


def run_evaluation(
    dataset_path: str | Path,
    get_docs: Callable[[str], list],
    get_answer: Callable[[str, str], str],
    llm_fn: Optional[Callable[[str], str]] = None,
    *,
    compute_faithfulness_metric: bool = True,
    compute_relevancy_metric: bool = True,
    context_join: str = "\n\n",
    max_context_chars: int = 32_000,
) -> tuple[list[EvalResult], EvalSummary]:
    """Run evaluation on a dataset.

    Args:
        dataset_path: Path to JSON or JSONL file of EvalRecords.
        get_docs: Callable(question: str) -> list of objects with .text (e.g. RetrievedDocument).
        get_answer: Callable(question: str, context: str) -> answer string.
        llm_fn: Callable(prompt: str) -> str for LLM-as-judge metrics. If None, skips faithfulness/relevancy.
        compute_faithfulness_metric: Whether to compute faithfulness (requires llm_fn).
        compute_relevancy_metric: Whether to compute answer relevancy (requires llm_fn).
        context_join: String to join document texts into context.
        max_context_chars: Truncate context to this length for answer generation.

    Returns:
        (list of EvalResult per record, EvalSummary with aggregates).
    """
    path = Path(dataset_path)
    records = _load_dataset(path)
    results: list[EvalResult] = []
    failed = 0

    for i, record in enumerate(records):
        logger.info("Eval %d/%d: %s", i + 1, len(records), record.question[:60] + "...")
        try:
            docs = get_docs(record.question)
            context_parts = []
            for d in docs:
                text = getattr(d, "text", None) or str(d)
                if text:
                    context_parts.append(text)
            context_str = context_join.join(context_parts)
            if len(context_str) > max_context_chars:
                context_str = context_str[:max_context_chars] + "..."

            answer = get_answer(record.question, context_str)

            metrics: dict = {}
            if llm_fn and compute_faithfulness_metric and context_str.strip():
                faith = compute_faithfulness(
                    record.question, answer, context_str, llm_fn
                )
                if faith is not None:
                    metrics["faithfulness"] = faith
            if llm_fn and compute_relevancy_metric:
                rel = compute_answer_relevancy(record.question, answer, llm_fn)
                if rel is not None:
                    metrics["answer_relevancy"] = rel

            results.append(
                EvalResult(
                    question=record.question,
                    answer=answer,
                    contexts=context_parts[:10],
                    metrics=metrics,
                )
            )
        except Exception as e:
            failed += 1
            logger.exception("Eval failed for: %s", record.question)
            results.append(
                EvalResult(
                    question=record.question,
                    answer="",
                    contexts=[],
                    metrics={},
                    error=str(e),
                )
            )

    # Aggregate
    metric_keys = set()
    for r in results:
        if not r.error:
            metric_keys.update(r.metrics.keys())
    metrics_avg = {}
    for k in metric_keys:
        values = [r.metrics[k] for r in results if not r.error and k in r.metrics]
        metrics_avg[k] = sum(values) / len(values) if values else 0.0

    summary = EvalSummary(
        total=len(records),
        failed=failed,
        metrics_avg=metrics_avg,
    )
    return results, summary

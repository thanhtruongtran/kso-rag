"""DeepEval-based evaluation runner for the RAG Q&A pipeline.

Supports 5 RAG metrics from DeepEval:
- AnswerRelevancy:    Generated answer is relevant to the question.
- Faithfulness:      Generated answer is grounded in the retrieved contexts.
- ContextualRelevancy:  Retrieved contexts are relevant to the question.
- ContextualPrecision:  Relevant contexts are ranked higher (requires expected_output).
- ContextualRecall:     Retrieved contexts cover the expected answer (requires expected_output).

Reference: https://deepeval.com/docs/getting-started-rag
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import os
from typing import Callable, Optional

from .schema import EvalRecord, EvalResult, EvalSummary

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric names constants – use these strings in the --metrics CLI arg.
# ---------------------------------------------------------------------------
METRIC_ANSWER_RELEVANCY = "answer_relevancy"
METRIC_FAITHFULNESS = "faithfulness"
METRIC_CONTEXTUAL_RELEVANCY = "contextual_relevancy"
METRIC_CONTEXTUAL_PRECISION = "contextual_precision"   # requires expected_output
METRIC_CONTEXTUAL_RECALL = "contextual_recall"         # requires expected_output

ALL_METRICS = [
    METRIC_ANSWER_RELEVANCY,
    METRIC_FAITHFULNESS,
    METRIC_CONTEXTUAL_RELEVANCY,
    METRIC_CONTEXTUAL_PRECISION,
    METRIC_CONTEXTUAL_RECALL,
]


def _load_dataset(path: Path) -> list[EvalRecord]:
    """Load EvalRecord list from JSON array or JSONL file."""
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        records = []
        for line in text.strip().split("\n"):
            line = line.strip()
            if line:
                records.append(EvalRecord.model_validate(json.loads(line)))
        return records
    data = json.loads(text)
    if isinstance(data, list):
        return [EvalRecord.model_validate(r) for r in data]
    if isinstance(data, dict) and "questions" in data:
        return [EvalRecord.model_validate(r) for r in data["questions"]]
    return [EvalRecord.model_validate(data)]


PROVIDER_OPENAI = "openai"
PROVIDER_GEMINI = "gemini"
PROVIDER_AZURE = "azure"
PROVIDER_OLLAMA = "ollama"
PROVIDER_ANTHROPIC = "anthropic"

ALL_PROVIDERS = [PROVIDER_OPENAI, PROVIDER_GEMINI, PROVIDER_AZURE, PROVIDER_OLLAMA, PROVIDER_ANTHROPIC]


def _read_google_api_key_from_env_file() -> str | None:
    """
    Best-effort helper to read GOOGLE_API_KEY from the project's .env file,
    in case it is not present in os.environ (e.g. when DeepEval is invoked
    from a different working directory).
    """
    if os.environ.get("GOOGLE_API_KEY"):
        return os.environ["GOOGLE_API_KEY"]

    # Walk up from this file to find repo root (directory containing pyproject.toml)
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            env_path = parent / ".env"
            if env_path.exists():
                try:
                    for line in env_path.read_text(encoding="utf-8").splitlines():
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        key, val = line.split("=", 1)
                        key = key.strip()
                        if key == "GOOGLE_API_KEY":
                            val = val.strip().strip('"').strip("'")
                            if val:
                                os.environ.setdefault("GOOGLE_API_KEY", val)
                                return val
                except Exception:
                    # Ignore parse errors; caller will handle missing key
                    return None
            break
    return os.environ.get("GOOGLE_API_KEY")


def _build_judge_model(provider: str, model: str):
    """Build a DeepEval judge model object for the given provider.

    Args:
        provider: One of ALL_PROVIDERS (e.g. "gemini", "openai").
        model:    Model name (e.g. "gemini-2.0-flash", "gpt-4o-mini").

    Returns:
        A DeepEval model object, or the model name string (for OpenAI default).
    """
    try:
        if provider == PROVIDER_GEMINI:
            from deepeval.models import GeminiModel

            api_key = _read_google_api_key_from_env_file()
            # Pass api_key explicitly when available to avoid relying purely on env
            if api_key:
                return GeminiModel(model, api_key=api_key)
            return GeminiModel(model)

        elif provider == PROVIDER_AZURE:
            import os
            from deepeval.models import AzureOpenAIModel
            return AzureOpenAIModel(
                model=model,
                deployment_name=os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT", model),
                api_key=os.environ.get("AZURE_OPENAI_API_KEY", ""),
                api_version=os.environ.get("OPENAI_API_VERSION", "2024-02-15-preview"),
                base_url=os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
            )

        elif provider == PROVIDER_OLLAMA:
            from deepeval.models import OllamaModel
            return OllamaModel(model)

        elif provider == PROVIDER_ANTHROPIC:
            from deepeval.models import AnthropicModel
            return AnthropicModel(model)

        else:
            # openai: deepeval accepts plain string model name
            return model

    except ImportError as e:
        raise ImportError(
            f"Could not build judge model for provider '{provider}'. "
            "Make sure deepeval and its provider dependencies are installed."
        ) from e


def _build_deepeval_metrics(
    metric_names: list[str],
    threshold: float,
    judge_model,
):
    """Instantiate DeepEval metric objects.

    Args:
        metric_names: Subset of ALL_METRICS to compute.
        threshold:    Pass/fail threshold (0–1).
        judge_model:  DeepEval model object or plain string (OpenAI).
    """
    try:
        from deepeval.metrics import (
            AnswerRelevancyMetric,
            ContextualPrecisionMetric,
            ContextualRecallMetric,
            ContextualRelevancyMetric,
            FaithfulnessMetric,
        )
    except ImportError as e:
        raise ImportError(
            "DeepEval is not installed. Run: uv pip install deepeval"
        ) from e

    metrics = []
    for name in metric_names:
        if name == METRIC_ANSWER_RELEVANCY:
            metrics.append(AnswerRelevancyMetric(threshold=threshold, model=judge_model))
        elif name == METRIC_FAITHFULNESS:
            metrics.append(FaithfulnessMetric(threshold=threshold, model=judge_model))
        elif name == METRIC_CONTEXTUAL_RELEVANCY:
            metrics.append(ContextualRelevancyMetric(threshold=threshold, model=judge_model))
        elif name == METRIC_CONTEXTUAL_PRECISION:
            metrics.append(ContextualPrecisionMetric(threshold=threshold, model=judge_model))
        elif name == METRIC_CONTEXTUAL_RECALL:
            metrics.append(ContextualRecallMetric(threshold=threshold, model=judge_model))
        else:
            logger.warning("Unknown metric name ignored: %s", name)
    return metrics


def run_deepeval(
    dataset_path: str | Path,
    get_docs: Callable[[str], list],
    get_answer: Callable[[str, str], str],
    *,
    metrics: Optional[list[str]] = None,
    threshold: float = 0.5,
    model: str = "gemini-2.0-flash",
    provider: str = PROVIDER_GEMINI,
    context_join: str = "\n\n",
    max_context_chars: int = 32_000,
) -> tuple[list[EvalResult], EvalSummary]:
    """Run evaluation using DeepEval RAG metrics.

    Args:
        dataset_path: Path to JSON or JSONL file with EvalRecords.
        get_docs:     Callable(question) -> list of objects with .text attribute.
        get_answer:   Callable(question, context_str) -> answer string.
        metrics:      List of metric name strings to compute (default: all 5).
                      ContextualPrecision and ContextualRecall require
                      `ground_truth_answer` in the dataset records.
        threshold:    Pass/fail threshold for each metric (0–1, default 0.5).
        model:        LLM model name for DeepEval judge (default "gemini-2.0-flash").
        provider:     Model provider: "gemini", "openai", "azure", "ollama", "anthropic".
        context_join: String to join doc texts into one context string.
        max_context_chars: Max characters for context passed to answer pipeline.

    Returns:
        (list[EvalResult], EvalSummary)
    """
    try:
        from deepeval.test_case import LLMTestCase
    except ImportError as e:
        raise ImportError(
            "DeepEval is not installed. Run: pip install 'kso-rag-core[eval]' "
            "or: pip install deepeval"
        ) from e

    path = Path(dataset_path)
    records = _load_dataset(path)

    if metrics is None:
        metrics = ALL_METRICS
    # Metrics that need expected_output – skip them if not in dataset
    needs_expected = {METRIC_CONTEXTUAL_PRECISION, METRIC_CONTEXTUAL_RECALL}
    has_expected = any(r.ground_truth_answer for r in records)
    if not has_expected:
        filtered = [m for m in metrics if m not in needs_expected]
        if len(filtered) < len(metrics):
            logger.warning(
                "Skipping %s because no `ground_truth_answer` found in dataset.",
                needs_expected & set(metrics),
            )
        metrics = filtered

    judge_model = _build_judge_model(provider, model)
    deepeval_metrics = _build_deepeval_metrics(metrics, threshold, judge_model)

    raw_results: list[EvalResult] = []
    failed = 0

    for i, record in enumerate(records):
        logger.info("Preparing %d/%d: %s", i + 1, len(records), record.question[:60])
        try:
            docs = get_docs(record.question)
            context_parts: list[str] = []
            for d in docs:
                text = getattr(d, "text", None) or str(d)
                if text:
                    context_parts.append(text)

            context_str = context_join.join(context_parts)
            if len(context_str) > max_context_chars:
                context_str = context_str[:max_context_chars] + "..."

            answer = get_answer(record.question, context_str)
            actual_output = answer or "(no answer)"

            test_case = LLMTestCase(
                input=record.question,
                actual_output=actual_output,
                retrieval_context=context_parts[:20],
                expected_output=record.ground_truth_answer or "",
            )

            metric_scores: dict[str, dict] = {}
            for metric in deepeval_metrics:
                # Each metric will call its underlying judge model as needed.
                metric.measure(test_case)
                metric_name = getattr(metric, "name", metric.__class__.__name__)
                metric_scores[metric_name] = {
                    "score": float(getattr(metric, "score", 0.0)),
                    "passed": bool(getattr(metric, "success", False)),
                    "reason": str(getattr(metric, "reason", "")),
                }

            raw_results.append(
                EvalResult(
                    question=record.question,
                    answer=answer,
                    contexts=context_parts[:10],
                    metrics=metric_scores,
                )
            )
        except Exception as e:
            failed += 1
            logger.exception("Failed to prepare record: %s", record.question)
            raw_results.append(
                EvalResult(
                    question=record.question,
                    answer="",
                    contexts=[],
                    metrics={},
                    error=str(e),
                )
            )

    if not raw_results:
        summary = EvalSummary(total=len(records), failed=failed, metrics_avg={})
        return raw_results, summary

    # Aggregate average scores
    score_sums: dict[str, list[float]] = {}
    for r in raw_results:
        if r.error:
            continue
        for metric_name, data in r.metrics.items():
            score = data.get("score") if isinstance(data, dict) else data
            if score is not None:
                score_sums.setdefault(metric_name, []).append(float(score))
    metrics_avg = {k: sum(v) / len(v) for k, v in score_sums.items() if v}

    summary = EvalSummary(
        total=len(records),
        failed=failed,
        metrics_avg=metrics_avg,
    )
    return raw_results, summary

"""Evaluation module for RAG (Retrieval-Augmented Generation) pipelines.

Provides:
- EvalRecord, EvalResult, EvalSummary: dataset and result schemas
- run_evaluation: custom LLM-as-judge runner (no extra deps)
- run_deepeval: DeepEval-based runner (requires `pip install deepeval`)
- compute_faithfulness, compute_answer_relevancy: standalone LLM-as-judge helpers
"""

from .runner import run_evaluation
from .schema import EvalRecord, EvalResult, EvalSummary
from .metrics import compute_faithfulness, compute_answer_relevancy
from .deepeval_runner import (
    run_deepeval,
    ALL_METRICS,
    ALL_PROVIDERS,
    METRIC_ANSWER_RELEVANCY,
    METRIC_FAITHFULNESS,
    METRIC_CONTEXTUAL_RELEVANCY,
    METRIC_CONTEXTUAL_PRECISION,
    METRIC_CONTEXTUAL_RECALL,
    PROVIDER_GEMINI,
    PROVIDER_OPENAI,
    PROVIDER_AZURE,
    PROVIDER_OLLAMA,
    PROVIDER_ANTHROPIC,
)

__all__ = [
    # schemas
    "EvalRecord",
    "EvalResult",
    "EvalSummary",
    # runners
    "run_evaluation",
    "run_deepeval",
    # deepeval metric name constants
    "ALL_METRICS",
    "ALL_PROVIDERS",
    "METRIC_ANSWER_RELEVANCY",
    "METRIC_FAITHFULNESS",
    "METRIC_CONTEXTUAL_RELEVANCY",
    "METRIC_CONTEXTUAL_PRECISION",
    "METRIC_CONTEXTUAL_RECALL",
    # deepeval provider constants
    "PROVIDER_GEMINI",
    "PROVIDER_OPENAI",
    "PROVIDER_AZURE",
    "PROVIDER_OLLAMA",
    "PROVIDER_ANTHROPIC",
    # standalone helpers
    "compute_faithfulness",
    "compute_answer_relevancy",
]

"""Schema for RAG evaluation dataset and results."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class EvalRecord(BaseModel):
    """One row in the evaluation dataset.

    Attributes:
        question: The user question to evaluate.
        ground_truth_answer: Optional reference answer (for comparison or retrieval eval).
        ground_truth_contexts: Optional list of context strings that should be retrieved.
    """

    question: str = Field(..., description="User question")
    ground_truth_answer: Optional[str] = Field(
        default=None,
        description="Optional reference answer",
    )
    ground_truth_contexts: Optional[list[str]] = Field(
        default=None,
        description="Optional list of context strings that are relevant",
    )


class EvalResult(BaseModel):
    """Result for a single evaluation record.

    Attributes:
        question: The question that was evaluated.
        answer: The generated answer.
        contexts: List of retrieved context strings (e.g. document texts).
        metrics: Dict of metric name -> value (float or dict).
        error: If an exception occurred, message is stored here.
    """

    question: str = Field(..., description="Evaluated question")
    answer: str = Field(default="", description="Generated answer")
    contexts: list[str] = Field(default_factory=list, description="Retrieved contexts")
    metrics: dict[str, Any] = Field(default_factory=dict, description="Metric scores")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class EvalSummary(BaseModel):
    """Aggregate summary of an evaluation run.

    Attributes:
        total: Number of records evaluated.
        failed: Number of records that raised an error.
        metrics_avg: Average (or aggregate) of each metric across records.
    """

    total: int = Field(..., description="Total records evaluated")
    failed: int = Field(default=0, description="Number of failed records")
    metrics_avg: dict[str, float] = Field(
        default_factory=dict,
        description="Average value per metric",
    )

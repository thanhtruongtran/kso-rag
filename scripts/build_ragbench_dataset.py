#!/usr/bin/env python3
"""Build evaluation datasets from the public RAGBench dataset.

Usage examples:

  # HotpotQA subset, test split, 200 samples
  python scripts/build_ragbench_dataset.py \
    --subset hotpotqa --split test --limit 200 \
    --output scripts/eval_ragbench_hotpotqa_test.jsonl

  # Full covidqa test split (no limit)
  python scripts/build_ragbench_dataset.py \
    --subset covidqa --split test \
    --output scripts/eval_ragbench_covidqa_test.jsonl

The output format is compatible with scripts/run_evaluation.py:
  {"question": "...", "ground_truth_answer": "..."}

Reference dataset: https://huggingface.co/datasets/galileo-ai/ragbench
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_dataset(subset: str, split: str, limit: int | None, output: Path) -> None:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as e:
        raise SystemExit(
            "The 'datasets' library is required. Install it with: `uv pip install datasets`"
        ) from e

    print(f"Loading RAGBench subset='{subset}', split='{split}'...")
    ds = load_dataset("galileo-ai/ragbench", subset, split=split)

    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
        print(f"Using first {len(ds)} examples (limit={limit}).")
    else:
        print(f"Using all {len(ds)} examples.")

    output.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0

    with output.open("w", encoding="utf-8") as f:
        for row in ds:
            question = row.get("question")
            if not question:
                continue

            # RAGBench has 'response' as the model's answer field; we use it as ground truth
            gt_answer = row.get("response") or ""

            record = {
                "question": question,
                "ground_truth_answer": gt_answer,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"Wrote {n_written} records to {output}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert RAGBench subset to kso-rag evaluation dataset.",
    )
    parser.add_argument(
        "--subset",
        required=True,
        help=(
            "RAGBench subset name, e.g. covidqa, hotpotqa, ms-marco, "
            "pubmedqa, tatqa, techqa, emanual, finqa, etc."
        ),
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Split name (train/validation/test). Default: test.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of examples to include.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL file path (EvalRecord format).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_dataset(args.subset, args.split, args.limit, args.output)


if __name__ == "__main__":
    main()

"""Feedback-style reward score dispatcher."""

from __future__ import annotations

from typing import Any

from verl.utils.reward_score.feedback import codeio


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if data_source == "codeio":
        return codeio.compute_score(solution_str, ground_truth, extra_info)

    raise ValueError(f"Reward data source {data_source!r} is not supported.")

"""Reward function for the reasoning_gym codeio task.

The model is given a Python function and inputs, then must predict the output as
a JSON value. Scoring uses reasoning_gym's built-in codeio verifier, with a
small exact-match fallback for large JSON values where tree-edit distance is too
expensive.
"""

from __future__ import annotations

import json
import re
from typing import Any

from reasoning_gym import get_score_answer_fn

_score_fn = get_score_answer_fn("codeio")

# reasoning_gym's tree-edit-distance scorer can hang on large JSON trees.
_MAX_SCORE_INPUT_CHARS = 5000


def _loads(value: str) -> Any:
    return json.loads(value)


def _is_valid_json(value: str) -> bool:
    try:
        _loads(value)
    except Exception:
        return False
    return True


def _find_matching_json_end(text: str, start: int, open_char: str, close_char: str) -> int | None:
    depth = 0
    in_string = False
    escape = False

    for index in range(start, len(text)):
        char = text[index]
        if escape:
            escape = False
            continue
        if char == "\\" and in_string:
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                return index + 1

    return None


def _extract_balanced_json_candidates(text: str, open_char: str, close_char: str) -> list[tuple[int, int, str]]:
    candidates = []
    for start, char in enumerate(text):
        if char != open_char:
            continue
        end = _find_matching_json_end(text, start, open_char, close_char)
        if end is None:
            continue
        candidate = text[start:end]
        if _is_valid_json(candidate):
            candidates.append((start, end, candidate))

    return candidates


def _extract_json(text: str) -> str | None:
    """Extract the last valid JSON value from model output."""
    fence_matches = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text)
    for candidate in reversed(fence_matches):
        candidate = candidate.strip()
        if _is_valid_json(candidate):
            return candidate

    balanced_candidates = []
    for open_char, close_char in [("{", "}"), ("[", "]")]:
        balanced_candidates.extend(_extract_balanced_json_candidates(text, open_char, close_char))
    if balanced_candidates:
        return max(balanced_candidates, key=lambda item: (item[1], -item[0]))[2]

    for line in reversed(text.splitlines()):
        candidate = line.strip()
        if candidate and _is_valid_json(candidate):
            return candidate

    return None


def _build_feedback(pred_str: str | None, ground_truth: str) -> str:
    if pred_str is None:
        return f"No JSON found in your response. Provide the answer as a JSON value, for example: {ground_truth}"

    try:
        pred = _loads(pred_str)
        expected = _loads(ground_truth)
    except Exception:
        return f"Your answer could not be parsed as JSON. Expected: {ground_truth}"

    if pred == expected:
        return ""

    if type(pred) is not type(expected):
        return (
            f"Your answer has the wrong type: expected {type(expected).__name__}, "
            f"got {type(pred).__name__}. Expected: {ground_truth}"
        )

    if isinstance(expected, dict) and isinstance(pred, dict):
        lines = [f"Your answer {pred_str} is incorrect:"]
        missing = set(expected) - set(pred)
        extra = set(pred) - set(expected)
        wrong = {key for key in expected if key in pred and pred[key] != expected[key]}
        for key in sorted(missing):
            lines.append(f"  - missing key '{key}' (expected value: {json.dumps(expected[key])})")
        for key in sorted(extra):
            lines.append(f"  - unexpected key '{key}' (value: {json.dumps(pred[key])})")
        for key in sorted(wrong):
            lines.append(f"  - '{key}': expected {json.dumps(expected[key])}, got {json.dumps(pred[key])}")
        return "\n".join(lines)

    return f"Your answer {pred_str} is incorrect. Expected: {ground_truth}"


def compute_score(solution_str: str, ground_truth: str, extra_info: dict[str, Any] | None = None) -> dict[str, Any]:
    """Compute reward metrics for a codeio prediction."""
    pred_str = _extract_json(solution_str)
    incorrect_format = 1 if pred_str is None else 0

    if pred_str is None:
        score = 0.0
    elif len(pred_str) > _MAX_SCORE_INPUT_CHARS or len(ground_truth) > _MAX_SCORE_INPUT_CHARS:
        try:
            score = 1.0 if _loads(pred_str) == _loads(ground_truth) else 0.0
        except Exception:
            score = 0.0
    else:
        score = _score_fn(pred_str, {"answer": ground_truth})

    acc = 1.0 if score >= 1.0 else 0.0
    feedback = "" if score >= 1.0 else _build_feedback(pred_str, ground_truth)

    return {
        "score": score,
        "acc": acc,
        "pred": pred_str or "",
        "feedback": feedback,
        "incorrect_format": incorrect_format,
    }

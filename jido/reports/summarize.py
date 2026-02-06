from __future__ import annotations

from statistics import mean
from typing import Any, Dict, List


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean_ms(entry: Dict[str, Any]) -> float | None:
    return _safe_float(entry.get("stats", {}).get("mean_ms"))


def summarize_benchmark(payload: Dict[str, Any], top: int = 10) -> Dict[str, Any]:
    results = payload.get("results", []) or []
    valid = [item for item in results if _mean_ms(item) is not None]
    passed = [
        item
        for item in valid
        if bool(item.get("correctness", {}).get("passed", True))
    ]

    top_fastest = sorted(passed, key=lambda item: float(_mean_ms(item) or 0.0))[:top]

    per_operation: Dict[str, List[float]] = {}
    for item in passed:
        op = str(item.get("operation", "unknown"))
        per_operation.setdefault(op, []).append(float(_mean_ms(item) or 0.0))

    op_summary: List[Dict[str, Any]] = []
    for operation, means in sorted(per_operation.items()):
        op_summary.append(
            {
                "operation": operation,
                "count": len(means),
                "best_mean_ms": min(means) if means else None,
                "avg_mean_ms": mean(means) if means else None,
            }
        )

    return {
        "source_command": payload.get("metadata", {}).get("command"),
        "total_results": len(results),
        "valid_results": len(valid),
        "correct_results": len(passed),
        "top_fastest": top_fastest,
        "operations": op_summary,
    }

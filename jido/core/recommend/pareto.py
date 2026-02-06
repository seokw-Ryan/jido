from __future__ import annotations

from typing import Any, Dict, List


def _dominates(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
    left_mean = float(left.get("stats", {}).get("mean_ms") or 0.0)
    right_mean = float(right.get("stats", {}).get("mean_ms") or 0.0)
    left_tflops = float(left.get("tflops") or 0.0)
    right_tflops = float(right.get("tflops") or 0.0)

    return (
        left_mean <= right_mean
        and left_tflops >= right_tflops
        and (left_mean < right_mean or left_tflops > right_tflops)
    )


def pareto_front(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    front: List[Dict[str, Any]] = []
    for candidate in entries:
        dominated = False
        for other in entries:
            if other is candidate:
                continue
            if _dominates(other, candidate):
                dominated = True
                break
        if not dominated:
            front.append(candidate)
    return front

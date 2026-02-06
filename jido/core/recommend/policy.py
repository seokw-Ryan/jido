from __future__ import annotations

from typing import Any, Dict, List

from jido.core.recommend.pareto import pareto_front


def _correct_entries(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    entries = payload.get("results", []) or []
    return [
        entry
        for entry in entries
        if bool(entry.get("correctness", {}).get("passed", True))
        and entry.get("stats", {}).get("mean_ms") is not None
    ]


def _scored(entries: List[Dict[str, Any]], objective: str) -> List[Dict[str, Any]]:
    if objective == "throughput":
        return sorted(entries, key=lambda e: float(e.get("tflops") or 0.0), reverse=True)
    if objective == "balanced":
        means = [float(e.get("stats", {}).get("mean_ms") or 0.0) for e in entries]
        tflops = [float(e.get("tflops") or 0.0) for e in entries]
        max_mean = max(means) if means else 1.0
        max_tflops = max(tflops) if tflops else 1.0

        def score(entry: Dict[str, Any]) -> float:
            mean_ms = float(entry.get("stats", {}).get("mean_ms") or 0.0)
            tflop_value = float(entry.get("tflops") or 0.0)
            latency_score = 1.0 - (mean_ms / max_mean if max_mean else 0.0)
            throughput_score = tflop_value / max_tflops if max_tflops else 0.0
            return (latency_score + throughput_score) / 2.0

        return sorted(entries, key=score, reverse=True)

    return sorted(entries, key=lambda e: float(e.get("stats", {}).get("mean_ms") or 0.0))


def recommend(
    payload: Dict[str, Any],
    objective: str = "latency",
    top: int = 5,
    use_pareto: bool = True,
) -> List[Dict[str, Any]]:
    entries = _correct_entries(payload)
    if use_pareto:
        entries = pareto_front(entries)
    return _scored(entries, objective=objective)[:top]

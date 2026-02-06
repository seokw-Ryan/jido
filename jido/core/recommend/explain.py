from __future__ import annotations

from typing import Any, Dict


def explain_recommendation(entry: Dict[str, Any], objective: str) -> str:
    operation = entry.get("operation", "unknown")
    kernel = entry.get("kernel", "unknown")
    mean_ms = float(entry.get("stats", {}).get("mean_ms") or 0.0)
    tflops = entry.get("tflops")
    if objective == "throughput":
        return f"{operation}/{kernel} selected for highest throughput ({tflops} TFLOPS)."
    if objective == "balanced":
        return (
            f"{operation}/{kernel} balances latency ({mean_ms:.3f} ms) "
            f"and throughput ({tflops} TFLOPS)."
        )
    return f"{operation}/{kernel} selected for lowest mean latency ({mean_ms:.3f} ms)."

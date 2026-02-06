from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def flatten_results(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    metadata = payload.get("metadata", {}) or {}
    for entry in payload.get("results", []) or []:
        stats = entry.get("stats", {}) or {}
        correctness = entry.get("correctness", {}) or {}
        rows.append(
            {
                "timestamp": metadata.get("timestamp"),
                "command": metadata.get("command"),
                "operation": entry.get("operation"),
                "kernel": entry.get("kernel"),
                "backend": entry.get("backend"),
                "kernel_type": entry.get("kernel_type"),
                "size": entry.get("size"),
                "dtype": entry.get("dtype"),
                "mean_ms": stats.get("mean_ms"),
                "p95_ms": stats.get("p95_ms"),
                "p99_ms": stats.get("p99_ms"),
                "std_ms": stats.get("std_ms"),
                "tflops": entry.get("tflops"),
                "speedup_vs_reference": entry.get("speedup_vs_reference"),
                "correct": correctness.get("passed"),
                "max_error": correctness.get("max_error"),
                "mean_error": correctness.get("mean_error"),
            }
        )
    return rows


def export_rows(rows: Iterable[Dict[str, Any]], output_path: str, fmt: str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "json":
        path.write_text(json.dumps(list(rows), indent=2), encoding="utf-8")
        return path

    if fmt != "csv":
        raise ValueError(f"Unsupported export format: {fmt}")

    rows_list = list(rows)
    fieldnames = list(rows_list[0].keys()) if rows_list else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_list:
            writer.writerow(row)
    return path

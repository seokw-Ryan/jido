from __future__ import annotations

from jido.core.recommend.policy import recommend
from jido.reports.export import flatten_results
from jido.reports.summarize import summarize_benchmark


def _sample_payload():
    return {
        "metadata": {"command": "benchmark"},
        "results": [
            {
                "operation": "matmul",
                "kernel": "reference",
                "size": "m=256,n=256,k=256,batch=1",
                "dtype": "fp32",
                "stats": {"mean_ms": 2.0, "p95_ms": 2.2, "p99_ms": 2.3, "std_ms": 0.1},
                "tflops": 1.0,
                "correctness": {"passed": True, "max_error": 0.0, "mean_error": 0.0},
            },
            {
                "operation": "matmul",
                "kernel": "torch_einsum",
                "size": "m=256,n=256,k=256,batch=1",
                "dtype": "fp32",
                "stats": {"mean_ms": 1.0, "p95_ms": 1.1, "p99_ms": 1.2, "std_ms": 0.1},
                "tflops": 2.0,
                "correctness": {"passed": True, "max_error": 0.0, "mean_error": 0.0},
            },
        ],
    }


def test_summarize_benchmark_reports_counts():
    summary = summarize_benchmark(_sample_payload(), top=1)

    assert summary["total_results"] == 2
    assert summary["correct_results"] == 2
    assert len(summary["top_fastest"]) == 1
    assert summary["top_fastest"][0]["kernel"] == "torch_einsum"


def test_recommend_latency_prefers_lowest_mean():
    choices = recommend(_sample_payload(), objective="latency", top=1, use_pareto=False)
    assert len(choices) == 1
    assert choices[0]["kernel"] == "torch_einsum"


def test_flatten_results_contains_rows():
    rows = flatten_results(_sample_payload())
    assert len(rows) == 2
    assert rows[0]["operation"] == "matmul"

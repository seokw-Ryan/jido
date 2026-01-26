from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from jido.benchmark import BenchmarkRunner
from jido.core.datastore.store import write_run_outputs
from jido.core.hardware import detect
from jido.operations import AttentionOperation, ConvolutionOperation, MatMulOperation, Operation
from jido.registry import KernelWrapper, get_kernels


def _safe_import_rich():
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
        from rich.table import Table

        return Console, Panel, Table, Progress, BarColumn, TextColumn, TimeElapsedColumn
    except Exception:
        return None, None, None, None, None, None, None


def _render_scan_output(
    hardware: Dict[str, Any],
    env: Dict[str, Any],
    hints: List[str],
    show_hardware: bool = True,
    show_software: bool = True,
) -> None:
    Console, Panel, Table, _, _, _, _ = _safe_import_rich()
    if Console is None:
        print("JIDO Scan")
        if show_hardware:
            print(hardware)
        if show_software:
            print(env)
        if show_software and hints:
            print("Install hints:")
            for hint in hints:
                print(f"- {hint}")
        return

    console = Console()
    console.print(Panel.fit("[bold]JIDO Scan[/bold]", border_style="cyan"))

    if show_hardware:
        cpu = hardware.get("cpu", {})
        memory = hardware.get("memory", {})
        gpus = hardware.get("gpus", [])

        cpu_table = Table(title="CPU", show_header=False)
        cpu_table.add_row("Brand", str(cpu.get("brand", "unknown")))
        cpu_table.add_row("Physical cores", str(cpu.get("physical_cores", "unknown")))
        cpu_table.add_row("Logical cores", str(cpu.get("logical_cores", "unknown")))
        cpu_table.add_row("Flags", str(len(cpu.get("flags", []) or [])))
        console.print(cpu_table)

        mem_table = Table(title="Memory", show_header=False)
        mem_table.add_row("Total (GB)", str(memory.get("total_gb", "unknown")))
        console.print(mem_table)

        gpu_table = Table(title="GPUs")
        gpu_table.add_column("Vendor")
        gpu_table.add_column("Name")
        gpu_table.add_column("VRAM (MB)")
        gpu_table.add_column("Driver")
        gpu_table.add_column("Runtimes")
        if gpus:
            for gpu in gpus:
                runtime = gpu.get("runtime", {})
                runtime_flags = [
                    name.upper()
                    for name, available in runtime.items()
                    if available
                ]
                gpu_table.add_row(
                    str(gpu.get("vendor", "unknown")),
                    str(gpu.get("name", "unknown")),
                    str(gpu.get("vram_total_mb", "unknown")),
                    str(gpu.get("driver", "unknown")),
                    ", ".join(runtime_flags) if runtime_flags else "none",
                )
        else:
            gpu_table.add_row("none", "none", "none", "none", "none")
        console.print(gpu_table)

    if show_software:
        runtime_table = Table(title="Available Runtimes")
        runtime_table.add_column("Runtime")
        runtime_table.add_column("Available")
        runtimes = detect.aggregate_runtime_flags(hardware.get("gpus", []))
        for name, available in runtimes.items():
            runtime_table.add_row(name.upper(), "yes" if available else "no")
        console.print(runtime_table)

        frameworks = env.get("frameworks", {})
        framework_table = Table(title="Installed ML Frameworks")
        framework_table.add_column("Framework")
        framework_table.add_column("Available")
        framework_table.add_column("Version")
        for framework in ["torch", "onnxruntime", "transformers"]:
            entry = frameworks.get(framework, {})
            framework_table.add_row(
                framework,
                "yes" if entry.get("available") else "no",
                str(entry.get("version") or "unknown"),
            )
        console.print(framework_table)

        if hints:
            hint_panel = Panel(
                "\n".join(f"- {hint}" for hint in hints),
                title="Install Hints",
                border_style="yellow",
            )
            console.print(hint_panel)


def _handle_scan(args: argparse.Namespace) -> int:
    hardware, env, hints, machine_id = detect.scan(
        deep=args.deep,
        install_hints=args.install_hints,
    )

    show_hardware = args.hardware or not args.software
    show_software = args.software or not args.hardware

    write_run_outputs(
        out_dir=args.out,
        machine_id=machine_id,
        hardware=hardware if show_hardware else None,
        env=env if show_software else None,
    )

    if not args.json_only:
        _render_scan_output(hardware, env, hints, show_hardware, show_software)
    return 0


def _operation_map() -> Dict[str, Operation]:
    return {
        "matmul": MatMulOperation(),
        "attention": AttentionOperation(),
        "conv2d": ConvolutionOperation(),
    }


def _parse_size_arg(value: str) -> Dict[str, int]:
    size: Dict[str, int] = {}
    for part in value.split(","):
        if not part.strip():
            continue
        if "=" not in part:
            raise ValueError(f"Invalid size fragment: {part}")
        key, raw = part.split("=", 1)
        size[key.strip()] = int(raw.strip())
    if not size:
        raise ValueError("Size must include at least one key=value pair")
    return size


def _parse_dtypes(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def _resolve_dtypes(operation: Operation, dtype_args: List[str]) -> List[Any]:
    torch = _require_torch()
    mapping = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    if not dtype_args:
        return list(operation.dtypes())
    resolved = []
    for name in dtype_args:
        if name not in mapping:
            raise ValueError(f"Unsupported dtype: {name}")
        resolved.append(mapping[name])
    return resolved


def _require_torch():
    try:
        import torch

        return torch
    except Exception as exc:
        raise RuntimeError("PyTorch is required for benchmark operations") from exc


def _kernel_candidates(operation_name: str, operation: Operation) -> Dict[str, KernelWrapper]:
    reference = KernelWrapper(
        operation=operation_name,
        name="reference",
        kernel=operation.reference_kernel(),
        backend="torch",
        kernel_type="reference",
    )
    candidates = {reference.name: reference}
    for kernel in get_kernels(operation_name):
        candidates[kernel.name] = kernel
    return candidates


def _filter_kernels(
    candidates: Dict[str, KernelWrapper], kernels_arg: Optional[str]
) -> List[KernelWrapper]:
    if not kernels_arg or kernels_arg == "all":
        return list(candidates.values())
    requested = [item.strip() for item in kernels_arg.split(",") if item.strip()]
    kernels = []
    for name in requested:
        if name not in candidates:
            raise ValueError(f"Unknown kernel: {name}")
        kernels.append(candidates[name])
    return kernels


def _dtype_label(dtype: Any) -> str:
    try:
        import torch

        if dtype == torch.float16:
            return "fp16"
        if dtype == torch.bfloat16:
            return "bf16"
        if dtype == torch.float32:
            return "fp32"
    except Exception:
        pass
    return str(dtype)git push --set-upstream origin feature/git-benchmark


def _format_size(size: Dict[str, int]) -> str:
    return ",".join(f"{key}={value}" for key, value in size.items())


def _load_baseline(path: str) -> Dict[str, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    baseline_map: Dict[str, Dict[str, Any]] = {}
    for entry in payload.get("results", []):
        key = "|".join(
            [
                entry.get("operation", ""),
                entry.get("kernel", ""),
                entry.get("dtype", ""),
                entry.get("size", ""),
            ]
        )
        baseline_map[key] = entry
    return baseline_map


def _render_benchmark_table(results: List[Dict[str, Any]]) -> None:
    Console, _, Table, _, _, _, _ = _safe_import_rich()
    if Console is None:
        for entry in results:
            print(entry)
        return

    console = Console()
    table = Table(title="JIDO Benchmark Results")
    table.add_column("Operation")
    table.add_column("Kernel")
    table.add_column("Size")
    table.add_column("Dtype")
    table.add_column("Mean (ms)", justify="right")
    table.add_column("P95 (ms)", justify="right")
    table.add_column("P99 (ms)", justify="right")
    table.add_column("TFLOPS", justify="right")
    table.add_column("Speedup", justify="right")
    table.add_column("Correct", justify="center")

    for entry in results:
        stats = entry.get("stats", {})
        correctness = entry.get("correctness", {})
        mean_ms = stats.get("mean_ms", 0.0)
        p95_ms = stats.get("p95_ms", 0.0)
        p99_ms = stats.get("p99_ms", 0.0)
        tflops = entry.get("tflops") or "-"
        speedup = entry.get("speedup_vs_reference") or "-"
        correct = "✅" if correctness.get("passed") else "⚠️"
        table.add_row(
            entry.get("operation", ""),
            entry.get("kernel", ""),
            entry.get("size", ""),
            entry.get("dtype", ""),
            f"{mean_ms:.3f}",
            f"{p95_ms:.3f}",
            f"{p99_ms:.3f}",
            f"{tflops:.3f}" if isinstance(tflops, float) else "-",
            f"{speedup:.2f}x" if isinstance(speedup, float) else "-",
            correct,
        )
    console.print(table)


def _handle_benchmark(args: argparse.Namespace) -> int:
    operations = _operation_map()
    if args.operation == "list":
        for name in sorted(operations):
            print(name)
        return 0

    selected_ops: Iterable[Tuple[str, Operation]]
    if args.operation:
        if args.operation not in operations:
            raise ValueError(f"Unknown operation: {args.operation}")
        selected_ops = [(args.operation, operations[args.operation])]
    else:
        selected_ops = list(operations.items())

    hardware, env, hints, machine_id = detect.scan(
        deep=False,
        install_hints=False,
    )

    all_results: List[Dict[str, Any]] = []
    baseline_map = _load_baseline(args.compare_against) if args.compare_against else {}

    runner = BenchmarkRunner(
        device=args.hardware,
        warmup=args.warmup,
        iterations=args.iterations,
        include_system_info=False,
    )

    Console, _, _, Progress, BarColumn, TextColumn, TimeElapsedColumn = _safe_import_rich()

    def _run_benchmarks(progress: Optional[Any] = None, task_id: Optional[int] = None) -> None:
        nonlocal all_results
        for op_name, operation in selected_ops:
            size_overrides = (
                [_parse_size_arg(args.size)] if args.size else list(operation.sizes())
            )
            dtype_args = _parse_dtypes(args.dtype)
            dtypes = _resolve_dtypes(operation, dtype_args)
            kernels = _filter_kernels(_kernel_candidates(op_name, operation), args.kernels)

            for size in size_overrides:
                for dtype in dtypes:
                    inputs = operation.generate_inputs(size, dtype, args.hardware)
                    for kernel in kernels:
                        result = runner.run_multiple_iterations(kernel, *inputs)
                        correctness = operation.validate_correctness(
                            kernel, inputs, atol=None, rtol=None
                        )

                        stats = result.stats
                        flops = operation.flops(size)
                        mean_seconds = stats.mean_ms / 1000.0 if stats.mean_ms else 0.0
                        tflops = None
                        if flops and mean_seconds > 0:
                            tflops = (flops / mean_seconds) / 1e12

                        size_label = _format_size(size)
                        dtype_label = _dtype_label(dtype)
                        entry = {
                            "operation": op_name,
                            "kernel": kernel.name,
                            "backend": kernel.backend,
                            "kernel_type": kernel.kernel_type,
                            "size": size_label,
                            "dtype": dtype_label,
                            "stats": {
                                "mean_ms": stats.mean_ms,
                                "std_ms": stats.std_ms,
                                "min_ms": stats.min_ms,
                                "max_ms": stats.max_ms,
                                "p50_ms": stats.p50_ms,
                                "p95_ms": stats.p95_ms,
                                "p99_ms": stats.p99_ms,
                            },
                            "timings_ms": result.timings_ms,
                            "memory_bytes": {
                                "peak": max(result.memory_bytes or [0]),
                                "mean": (
                                    float(sum(result.memory_bytes))
                                    / len(result.memory_bytes)
                                    if result.memory_bytes
                                    else 0.0
                                ),
                            },
                            "flops": flops,
                            "tflops": tflops,
                            "correctness": {
                                "passed": correctness.passed,
                                "max_error": correctness.max_error,
                                "mean_error": correctness.mean_error,
                                "atol": correctness.atol,
                                "rtol": correctness.rtol,
                            },
                        }

                        baseline_key = "|".join(
                            [op_name, kernel.name, dtype_label, size_label]
                        )
                        baseline_entry = baseline_map.get(baseline_key)
                        if baseline_entry:
                            baseline_mean = (
                                baseline_entry.get("stats", {}).get("mean_ms") or 0.0
                            )
                            if baseline_mean:
                                entry["speedup"] = baseline_mean / stats.mean_ms

                        all_results.append(entry)
                        if progress is not None and task_id is not None:
                            progress.advance(task_id)

    if Progress is not None:
        total = 0
        for op_name, operation in selected_ops:
            size_overrides = (
                [_parse_size_arg(args.size)] if args.size else list(operation.sizes())
            )
            dtype_args = _parse_dtypes(args.dtype)
            dtypes = _resolve_dtypes(operation, dtype_args)
            kernels = _filter_kernels(_kernel_candidates(op_name, operation), args.kernels)
            total += len(size_overrides) * len(dtypes) * len(kernels)

        with Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=Console() if Console else None,
        ) as progress:
            task_id = progress.add_task("Running benchmarks", total=total)
            _run_benchmarks(progress, task_id)
    else:
        _run_benchmarks()

    reference_means: Dict[str, float] = {}
    for entry in all_results:
        if entry.get("kernel") != "reference":
            continue
        key = "|".join(
            [
                entry.get("operation", ""),
                entry.get("size", ""),
                entry.get("dtype", ""),
            ]
        )
        reference_means[key] = float(entry.get("stats", {}).get("mean_ms") or 0.0)

    for entry in all_results:
        key = "|".join(
            [
                entry.get("operation", ""),
                entry.get("size", ""),
                entry.get("dtype", ""),
            ]
        )
        reference_mean = reference_means.get(key)
        if reference_mean and entry.get("stats", {}).get("mean_ms"):
            entry["speedup_vs_reference"] = reference_mean / entry["stats"]["mean_ms"]

    payload = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": "0.1.0",
            "command": "benchmark",
        },
        "config": {
            "operation": args.operation or "all",
            "size": args.size,
            "dtype": args.dtype,
            "kernels": args.kernels,
            "hardware": args.hardware,
            "iterations": args.iterations,
            "warmup": args.warmup,
        },
        "hardware": hardware,
        "env": env,
        "scan_hints": hints,
        "machine_id": machine_id,
        "results": all_results,
    }

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    if args.format in ("table", None):
        _render_benchmark_table(all_results)
    elif args.format == "json":
        print(json.dumps(payload, indent=2))
    else:
        raise ValueError(f"Unsupported format: {args.format}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jido")
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan_parser = subparsers.add_parser(
        "scan", help="Detect hardware and software info", add_help=False,
    )
    scan_parser.add_argument("--help", action="help", help="Show this help message and exit")
    scan_parser.add_argument(
        "-h", "--hardware", action="store_true",
        help="Scan hardware only (CPU, memory, GPUs)",
    )
    scan_parser.add_argument(
        "-s", "--software", action="store_true",
        help="Scan software only (frameworks, runtimes, vendor tools)",
    )
    scan_parser.add_argument("--out", default="runs", help="Output directory")
    scan_parser.add_argument("--json-only", action="store_true", help="Skip Rich output")
    scan_parser.add_argument(
        "--deep",
        action="store_true",
        help="Attempt deeper vendor tool detection",
    )
    scan_parser.add_argument(
        "--install-hints",
        action="store_true",
        help="Print install hints prominently",
    )
    scan_parser.set_defaults(func=_handle_scan)

    benchmark_parser = subparsers.add_parser(
        "benchmark", help="Run kernel benchmarks", add_help=False,
    )
    benchmark_parser.add_argument("--help", action="help", help="Show this help message and exit")
    benchmark_parser.add_argument(
        "operation", nargs="?", help="Operation name or 'list' to show available ops"
    )
    benchmark_parser.add_argument("--size", help="Override size, e.g. m=1024,n=1024,k=1024")
    benchmark_parser.add_argument("--dtype", help="Comma-separated dtypes (fp32, fp16, bf16)")
    benchmark_parser.add_argument("--kernels", default="all", help="Kernel names or 'all'")
    benchmark_parser.add_argument("--hardware", default="cpu", help="Device, e.g. cpu or cuda:0")
    benchmark_parser.add_argument("--iterations", type=int, default=50, help="Benchmark iterations")
    benchmark_parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    benchmark_parser.add_argument("--output", help="Write JSON output to path")
    benchmark_parser.add_argument(
        "--format", choices=["table", "json"], default="table", help="Output format"
    )
    benchmark_parser.add_argument(
        "--compare-against", help="Baseline JSON file to compare"
    )
    benchmark_parser.add_argument("--verbose", action="store_true", help="Verbose output")
    benchmark_parser.set_defaults(func=_handle_benchmark)

    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "func", None)
    if handler is None:
        parser.print_help()
        return 1
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())

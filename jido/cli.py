from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from jido.benchmark import BenchmarkRunner
from jido.core.datastore.store import write_run_outputs
from jido.core.hardware import detect
from jido.core.recommend.explain import explain_recommendation
from jido.core.recommend.policy import recommend as recommend_policy
from jido.kernels import ensure_builtin_kernels_registered
from jido.operations import AttentionOperation, ConvolutionOperation, MatMulOperation, Operation
from jido.registry import KernelWrapper, get_kernels
from jido.reports.export import export_rows, flatten_results
from jido.reports.summarize import summarize_benchmark

_EXTRA_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "torch": {
        "packages": ["torch", "transformers"],
        "description": "PyTorch + Transformers (needed for benchmarks)",
    },
    "onnx": {
        "packages": ["onnxruntime"],
        "description": "ONNX Runtime",
    },
    "vllm": {
        "packages": ["vllm"],
        "description": "vLLM backend",
    },
    "tensorrt": {
        "packages": ["tensorrt"],
        "description": "TensorRT backend (NVIDIA only)",
    },
    "llama-cpp": {
        "packages": ["llama-cpp-python"],
        "description": "llama.cpp backend (C++ build required)",
    },
    "openvino": {
        "packages": ["openvino"],
        "description": "OpenVINO backend",
    },
    "deepspeed": {
        "packages": ["deepspeed"],
        "description": "DeepSpeed backend",
    },
}

DEFAULT_CONFIG: Dict[str, Any] = {
    "benchmark": {
        "operation": "matmul",
        "hardware": "cpu",
        "iterations": 50,
        "warmup": 5,
        "kernels": "all",
        "dtype": "fp32",
    },
    "report": {
        "top": 10,
    },
    "recommend": {
        "objective": "latency",
        "top": 5,
        "pareto": True,
    },
}


def _safe_import_rich():
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
        from rich.table import Table

        return Console, Panel, Table, Progress, BarColumn, TextColumn, TimeElapsedColumn
    except Exception:
        return None, None, None, None, None, None, None


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_json_file(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _comparison_key(entry: Dict[str, Any]) -> str:
    return "|".join(
        [
            str(entry.get("operation", "")),
            str(entry.get("kernel", "")),
            str(entry.get("size", "")),
            str(entry.get("dtype", "")),
        ]
    )


def _package_available(package: str) -> bool:
    try:
        return importlib.metadata.version(package) is not None
    except importlib.metadata.PackageNotFoundError:
        return False
    except Exception:
        return False


def _parse_extras_arg(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _extra_status() -> Dict[str, Dict[str, Any]]:
    status: Dict[str, Dict[str, Any]] = {}
    for name, entry in _EXTRA_DEFINITIONS.items():
        packages = entry["packages"]
        missing = [pkg for pkg in packages if not _package_available(pkg)]
        status[name] = {
            "packages": packages,
            "missing": missing,
            "description": entry.get("description", ""),
        }
    return status


def _build_install_command(selected: List[str], editable: bool) -> str:
    extras = ",".join(selected)
    target = f".[{extras}]" if extras else "."
    if editable:
        return f"{sys.executable} -m pip install -e {target}"
    return f"{sys.executable} -m pip install {target}"


def _build_install_command_args(selected: List[str], editable: bool) -> List[str]:
    extras = ",".join(selected)
    target = f".[{extras}]" if extras else "."
    cmd = [sys.executable, "-m", "pip", "install"]
    if editable:
        cmd.append("-e")
    cmd.append(target)
    return cmd


def _render_deps_output(
    status: Dict[str, Dict[str, Any]],
    selected: List[str],
    missing_selected: List[str],
    editable: bool,
) -> None:
    Console, Panel, Table, _, _, _, _ = _safe_import_rich()
    install_cmd = _build_install_command(selected, editable)

    if Console is None:
        print("JIDO Dependencies")
        for name, entry in status.items():
            missing = entry["missing"]
            missing_note = f"missing: {', '.join(missing)}" if missing else "installed"
            print(f"- {name}: {missing_note}")
        if selected:
            print("Suggested install command:")
            print(install_cmd)
        return

    console = Console()
    console.print(Panel.fit("[bold]JIDO Dependencies[/bold]", border_style="cyan"))

    table = Table(title="Optional Dependency Groups")
    table.add_column("Extra")
    table.add_column("Packages")
    table.add_column("Status")
    table.add_column("Notes")
    for name, entry in status.items():
        packages = ", ".join(entry["packages"])
        missing = entry["missing"]
        status_label = "missing" if missing else "installed"
        notes = entry.get("description", "")
        table.add_row(name, packages, status_label, notes)
    console.print(table)

    if selected:
        hint = (
            f"Missing packages in selection: {', '.join(missing_selected)}"
            if missing_selected
            else "All selected extras appear installed."
        )
        console.print(Panel.fit(hint, title="Status", border_style="yellow"))
        console.print(
            Panel.fit(install_cmd, title="Install Command", border_style="green")
        )


def _handle_deps(args: argparse.Namespace) -> int:
    status = _extra_status()

    if args.list:
        for name in sorted(_EXTRA_DEFINITIONS):
            entry = _EXTRA_DEFINITIONS[name]
            packages = ", ".join(entry["packages"])
            description = entry.get("description", "")
            suffix = f" ({description})" if description else ""
            print(f"{name}: {packages}{suffix}")
        return 0

    if args.all:
        selected = list(_EXTRA_DEFINITIONS.keys())
    else:
        selected = _parse_extras_arg(args.extras)
        if not selected:
            if args.install:
                selected = list(_EXTRA_DEFINITIONS.keys())
            else:
                selected = [
                    name for name, entry in status.items() if entry["missing"]
                ]

    unknown = [name for name in selected if name not in _EXTRA_DEFINITIONS]
    if unknown:
        raise ValueError(f"Unknown extras: {', '.join(unknown)}")

    missing_selected: List[str] = []
    for name in selected:
        missing_selected.extend(status[name]["missing"])

    _render_deps_output(
        status=status,
        selected=selected,
        missing_selected=missing_selected,
        editable=not args.no_editable,
    )

    if args.install:
        cmd = _build_install_command_args(selected, not args.no_editable)
        subprocess.run(cmd, check=True)

    return 0


def _merge_runtime_flags(
    gpus: List[Dict[str, Any]], accelerators: List[Dict[str, Any]]
) -> Dict[str, bool]:
    runtime_flags = detect.aggregate_runtime_flags(gpus)
    for device in accelerators:
        runtime = device.get("runtime", {}) or {}
        for key, available in runtime.items():
            runtime_flags[key] = runtime_flags.get(key, False) or bool(available)
    return runtime_flags


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
        gpus = hardware.get("gpus", []) or []
        accelerators = hardware.get("accelerators", []) or []

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

        accelerator_table = Table(title="Accelerators")
        accelerator_table.add_column("Type")
        accelerator_table.add_column("Vendor")
        accelerator_table.add_column("Name")
        accelerator_table.add_column("Source")
        if accelerators:
            for item in accelerators:
                accelerator_table.add_row(
                    str(item.get("type", "unknown")),
                    str(item.get("vendor", "unknown")),
                    str(item.get("name", "unknown")),
                    str(item.get("source", "unknown")),
                )
        else:
            accelerator_table.add_row("none", "none", "none", "none")
        console.print(accelerator_table)

    if show_software:
        runtime_table = Table(title="Available Runtimes")
        runtime_table.add_column("Runtime")
        runtime_table.add_column("Available")

        runtimes = _merge_runtime_flags(
            hardware.get("gpus", []) or [],
            hardware.get("accelerators", []) or [],
        )
        for name, available in sorted(runtimes.items()):
            runtime_table.add_row(name.upper(), "yes" if available else "no")
        console.print(runtime_table)

        frameworks = env.get("frameworks", {})
        framework_table = Table(title="Installed ML Frameworks")
        framework_table.add_column("Framework")
        framework_table.add_column("Available")
        framework_table.add_column("Version")
        for framework in [
            "torch",
            "onnxruntime",
            "transformers",
            "tensorflow",
            "jax",
            "openvino",
            "libtpu",
        ]:
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
    ensure_builtin_kernels_registered()

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
    return str(dtype)


def _format_size(size: Dict[str, int]) -> str:
    return ",".join(f"{key}={value}" for key, value in size.items())


def _load_baseline(path: str) -> Dict[str, Dict[str, Any]]:
    payload = _load_json_file(path)
    baseline_map: Dict[str, Dict[str, Any]] = {}
    for entry in payload.get("results", []):
        key = _comparison_key(entry)
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
        mean_ms = _safe_float(stats.get("mean_ms")) or 0.0
        p95_ms = _safe_float(stats.get("p95_ms")) or 0.0
        p99_ms = _safe_float(stats.get("p99_ms")) or 0.0
        tflops = entry.get("tflops")
        speedup = entry.get("speedup_vs_reference")
        correct = "yes" if correctness.get("passed") else "no"
        table.add_row(
            str(entry.get("operation", "")),
            str(entry.get("kernel", "")),
            str(entry.get("size", "")),
            str(entry.get("dtype", "")),
            f"{mean_ms:.3f}",
            f"{p95_ms:.3f}",
            f"{p99_ms:.3f}",
            f"{float(tflops):.3f}" if isinstance(tflops, (int, float)) else "-",
            f"{float(speedup):.2f}x" if isinstance(speedup, (int, float)) else "-",
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
                    dtype_label = _dtype_label(dtype)
                    size_label = _format_size(size)
                    try:
                        inputs = operation.generate_inputs(size, dtype, args.hardware)
                    except Exception as exc:
                        all_results.append(
                            {
                                "operation": op_name,
                                "kernel": "-",
                                "backend": None,
                                "kernel_type": None,
                                "size": size_label,
                                "dtype": dtype_label,
                                "error": f"input generation failed: {exc}",
                            }
                        )
                        continue

                    for kernel in kernels:
                        try:
                            result = runner.run_multiple_iterations(kernel, *inputs)
                            correctness = operation.validate_correctness(
                                kernel, inputs, atol=None, rtol=None
                            )
                        except Exception as exc:
                            all_results.append(
                                {
                                    "operation": op_name,
                                    "kernel": kernel.name,
                                    "backend": kernel.backend,
                                    "kernel_type": kernel.kernel_type,
                                    "size": size_label,
                                    "dtype": dtype_label,
                                    "error": str(exc),
                                }
                            )
                            if progress is not None and task_id is not None:
                                progress.advance(task_id)
                            continue

                        stats = result.stats
                        flops = operation.flops(size)
                        mean_seconds = stats.mean_ms / 1000.0 if stats.mean_ms else 0.0
                        tflops = None
                        if flops and mean_seconds > 0:
                            tflops = (flops / mean_seconds) / 1e12

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

                        baseline_entry = baseline_map.get(_comparison_key(entry))
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
        if entry.get("stats", {}).get("mean_ms") is None:
            continue
        key = "|".join(
            [
                str(entry.get("operation", "")),
                str(entry.get("size", "")),
                str(entry.get("dtype", "")),
            ]
        )
        reference_means[key] = float(entry.get("stats", {}).get("mean_ms") or 0.0)

    for entry in all_results:
        mean_value = _safe_float(entry.get("stats", {}).get("mean_ms"))
        if mean_value is None:
            continue
        key = "|".join(
            [
                str(entry.get("operation", "")),
                str(entry.get("size", "")),
                str(entry.get("dtype", "")),
            ]
        )
        reference_mean = reference_means.get(key)
        if reference_mean and mean_value:
            entry["speedup_vs_reference"] = reference_mean / mean_value

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


def _render_compare_table(rows: List[Dict[str, Any]]) -> None:
    Console, _, Table, _, _, _, _ = _safe_import_rich()
    if Console is None:
        for row in rows:
            print(row)
        return

    console = Console()
    table = Table(title="JIDO Compare")
    table.add_column("Operation")
    table.add_column("Kernel")
    table.add_column("Size")
    table.add_column("Dtype")
    table.add_column("Baseline (ms)", justify="right")
    table.add_column("Candidate (ms)", justify="right")
    table.add_column("Speedup", justify="right")
    table.add_column("Delta %", justify="right")

    for row in rows:
        table.add_row(
            str(row["operation"]),
            str(row["kernel"]),
            str(row["size"]),
            str(row["dtype"]),
            f"{row['baseline_mean_ms']:.3f}",
            f"{row['candidate_mean_ms']:.3f}",
            f"{row['speedup']:.2f}x",
            f"{row['delta_pct']:+.2f}",
        )
    console.print(table)


def _handle_compare(args: argparse.Namespace) -> int:
    baseline = _load_json_file(args.baseline)
    candidate = _load_json_file(args.candidate)

    baseline_index: Dict[str, Dict[str, Any]] = {}
    for entry in baseline.get("results", []) or []:
        mean_ms = _safe_float(entry.get("stats", {}).get("mean_ms"))
        if mean_ms is None or mean_ms <= 0:
            continue
        baseline_index[_comparison_key(entry)] = entry

    rows: List[Dict[str, Any]] = []
    for entry in candidate.get("results", []) or []:
        mean_ms = _safe_float(entry.get("stats", {}).get("mean_ms"))
        if mean_ms is None or mean_ms <= 0:
            continue
        key = _comparison_key(entry)
        baseline_entry = baseline_index.get(key)
        if baseline_entry is None:
            continue
        baseline_mean = _safe_float(baseline_entry.get("stats", {}).get("mean_ms"))
        if baseline_mean is None or baseline_mean <= 0:
            continue

        speedup = baseline_mean / mean_ms
        delta_pct = ((mean_ms - baseline_mean) / baseline_mean) * 100.0
        row = {
            "operation": entry.get("operation"),
            "kernel": entry.get("kernel"),
            "size": entry.get("size"),
            "dtype": entry.get("dtype"),
            "baseline_mean_ms": baseline_mean,
            "candidate_mean_ms": mean_ms,
            "speedup": speedup,
            "delta_pct": delta_pct,
        }
        if args.only_regressions and mean_ms <= baseline_mean:
            continue
        rows.append(row)

    rows.sort(key=lambda item: item["speedup"], reverse=True)

    output = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "command": "compare",
            "baseline": args.baseline,
            "candidate": args.candidate,
        },
        "comparisons": rows,
        "count": len(rows),
    }

    if args.output:
        Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")

    if args.format == "json":
        print(json.dumps(output, indent=2))
    else:
        _render_compare_table(rows)

    return 0


def _render_recommend_table(entries: List[Dict[str, Any]], objective: str) -> None:
    Console, _, Table, _, _, _, _ = _safe_import_rich()
    if Console is None:
        for entry in entries:
            print(entry)
        return

    console = Console()
    table = Table(title=f"JIDO Recommendations ({objective})")
    table.add_column("Operation")
    table.add_column("Kernel")
    table.add_column("Size")
    table.add_column("Dtype")
    table.add_column("Mean (ms)", justify="right")
    table.add_column("TFLOPS", justify="right")
    table.add_column("Reason")

    for entry in entries:
        mean_ms = float(entry.get("stats", {}).get("mean_ms") or 0.0)
        tflops = entry.get("tflops")
        table.add_row(
            str(entry.get("operation", "")),
            str(entry.get("kernel", "")),
            str(entry.get("size", "")),
            str(entry.get("dtype", "")),
            f"{mean_ms:.3f}",
            f"{float(tflops):.3f}" if isinstance(tflops, (int, float)) else "-",
            explain_recommendation(entry, objective=objective),
        )
    console.print(table)


def _handle_recommend(args: argparse.Namespace) -> int:
    payload = _load_json_file(args.input)
    entries = recommend_policy(
        payload,
        objective=args.objective,
        top=args.top,
        use_pareto=not args.no_pareto,
    )

    output = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "command": "recommend",
            "objective": args.objective,
        },
        "recommendations": entries,
    }

    if args.output:
        Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")

    if args.format == "json":
        print(json.dumps(output, indent=2))
    else:
        _render_recommend_table(entries, objective=args.objective)

    return 0


def _handle_list(args: argparse.Namespace) -> int:
    ensure_builtin_kernels_registered()
    subject = args.subject

    if subject == "operations":
        payload = {"operations": sorted(_operation_map().keys())}
    elif subject == "kernels":
        kernels: Dict[str, List[str]] = {}
        for op_name in _operation_map().keys():
            names = {"reference"}
            names.update(kernel.name for kernel in get_kernels(op_name))
            kernels[op_name] = sorted(names)
        payload = {"kernels": kernels}
    elif subject == "extras":
        payload = {
            "extras": {
                name: entry["packages"] for name, entry in sorted(_EXTRA_DEFINITIONS.items())
            }
        }
    elif subject == "commands":
        payload = {
            "commands": [
                "scan",
                "deps",
                "benchmark",
                "compare",
                "recommend",
                "list",
                "report",
                "history",
                "export",
                "config",
                "profile",
            ]
        }
    else:
        raise ValueError(f"Unsupported list subject: {subject}")

    if args.format == "json":
        print(json.dumps(payload, indent=2))
        return 0

    for key, value in payload.items():
        print(f"{key}:")
        if isinstance(value, dict):
            for name, nested in value.items():
                if isinstance(nested, list):
                    print(f"  {name}: {', '.join(nested)}")
                else:
                    print(f"  {name}: {nested}")
        elif isinstance(value, list):
            for item in value:
                print(f"  {item}")
        else:
            print(value)

    return 0


def _render_report(summary: Dict[str, Any]) -> None:
    Console, _, Table, _, _, _, _ = _safe_import_rich()
    if Console is None:
        print(summary)
        return

    console = Console()
    print_table = Table(title="JIDO Report Summary")
    print_table.add_column("Metric")
    print_table.add_column("Value")
    print_table.add_row("Total results", str(summary.get("total_results")))
    print_table.add_row("Valid results", str(summary.get("valid_results")))
    print_table.add_row("Correct results", str(summary.get("correct_results")))
    console.print(print_table)

    top_table = Table(title="Top Fastest")
    top_table.add_column("Operation")
    top_table.add_column("Kernel")
    top_table.add_column("Size")
    top_table.add_column("Dtype")
    top_table.add_column("Mean (ms)", justify="right")

    for entry in summary.get("top_fastest", []):
        mean_ms = float(entry.get("stats", {}).get("mean_ms") or 0.0)
        top_table.add_row(
            str(entry.get("operation", "")),
            str(entry.get("kernel", "")),
            str(entry.get("size", "")),
            str(entry.get("dtype", "")),
            f"{mean_ms:.3f}",
        )
    console.print(top_table)


def _handle_report(args: argparse.Namespace) -> int:
    payload = _load_json_file(args.input)
    summary = summarize_benchmark(payload, top=args.top)

    output = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "command": "report",
            "input": args.input,
            "top": args.top,
        },
        "summary": summary,
    }

    if args.output:
        Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")

    if args.format == "json":
        print(json.dumps(output, indent=2))
    else:
        _render_report(summary)

    return 0


def _discover_history_entries(runs_dir: str) -> List[Dict[str, Any]]:
    root = Path(runs_dir)
    if not root.exists():
        return []

    entries: List[Dict[str, Any]] = []
    for path in root.rglob("*.json"):
        kind = "json"
        if path.name == "hardware.json":
            kind = "scan-hardware"
        elif path.name == "env.json":
            kind = "scan-env"
        else:
            try:
                payload = _load_json_file(str(path))
                command = payload.get("metadata", {}).get("command")
                if command:
                    kind = str(command)
            except Exception:
                kind = "json"

        stats = path.stat()
        entries.append(
            {
                "path": str(path),
                "kind": kind,
                "size_bytes": stats.st_size,
                "modified_ts": datetime.fromtimestamp(stats.st_mtime, timezone.utc).isoformat(),
            }
        )

    entries.sort(key=lambda item: item["modified_ts"], reverse=True)
    return entries


def _render_history(entries: List[Dict[str, Any]]) -> None:
    Console, _, Table, _, _, _, _ = _safe_import_rich()
    if Console is None:
        for entry in entries:
            print(entry)
        return

    console = Console()
    table = Table(title="JIDO History")
    table.add_column("Kind")
    table.add_column("Modified (UTC)")
    table.add_column("Size (B)", justify="right")
    table.add_column("Path")

    for entry in entries:
        table.add_row(
            str(entry.get("kind", "")),
            str(entry.get("modified_ts", "")),
            str(entry.get("size_bytes", "")),
            str(entry.get("path", "")),
        )
    console.print(table)


def _handle_history(args: argparse.Namespace) -> int:
    entries = _discover_history_entries(args.runs_dir)
    if args.kind:
        entries = [entry for entry in entries if entry.get("kind") == args.kind]
    entries = entries[: args.limit]

    output = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "command": "history",
            "runs_dir": args.runs_dir,
        },
        "entries": entries,
    }

    if args.output:
        Path(args.output).write_text(json.dumps(output, indent=2), encoding="utf-8")

    if args.format == "json":
        print(json.dumps(output, indent=2))
    else:
        _render_history(entries)

    return 0


def _handle_export(args: argparse.Namespace) -> int:
    payload = _load_json_file(args.input)

    if args.section == "summary":
        summary = summarize_benchmark(payload, top=args.top)
        rows = summary.get("operations", [])
    else:
        rows = flatten_results(payload)

    output_path = export_rows(rows, output_path=args.output, fmt=args.format)
    print(str(output_path))
    return 0


def _handle_config(args: argparse.Namespace) -> int:
    if args.write:
        path = Path(args.write)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(DEFAULT_CONFIG, indent=2), encoding="utf-8")
        print(str(path))
        return 0

    if args.example:
        print(
            "jido benchmark matmul --hardware cpu --dtype fp32 --iterations 50 --warmup 5"
        )
        return 0

    print(json.dumps(DEFAULT_CONFIG, indent=2))
    return 0


def _handle_profile(args: argparse.Namespace) -> int:
    bench_args = argparse.Namespace(
        operation=args.operation or "matmul",
        size=args.size,
        dtype=args.dtype,
        kernels=args.kernels,
        hardware=args.hardware,
        iterations=args.iterations,
        warmup=args.warmup,
        output=args.output,
        format=args.format,
        compare_against=None,
        verbose=args.verbose,
    )
    return _handle_benchmark(bench_args)


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
        help="Attempt deeper vendor/accelerator tool detection",
    )
    scan_parser.add_argument(
        "--install-hints",
        action="store_true",
        help="Print install hints prominently",
    )
    scan_parser.set_defaults(func=_handle_scan)

    deps_parser = subparsers.add_parser(
        "deps", help="Show or install optional dependencies", add_help=False,
    )
    deps_parser.add_argument("--help", action="help", help="Show this help message and exit")
    deps_parser.add_argument(
        "--list",
        action="store_true",
        help="List supported extras and packages",
    )
    deps_parser.add_argument(
        "--extras",
        help="Comma-separated extras to target (e.g. torch,onnx)",
    )
    deps_parser.add_argument(
        "--all",
        action="store_true",
        help="Target all optional extras",
    )
    deps_parser.add_argument(
        "--install",
        action="store_true",
        help="Run pip install for the selected extras",
    )
    deps_parser.add_argument(
        "--no-editable",
        action="store_true",
        help="Install without -e (non-editable)",
    )
    deps_parser.set_defaults(func=_handle_deps)

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

    compare_parser = subparsers.add_parser("compare", help="Compare two benchmark outputs")
    compare_parser.add_argument("--baseline", required=True, help="Baseline benchmark JSON")
    compare_parser.add_argument("--candidate", required=True, help="Candidate benchmark JSON")
    compare_parser.add_argument(
        "--only-regressions",
        action="store_true",
        help="Show only entries slower than baseline",
    )
    compare_parser.add_argument("--output", help="Write comparison JSON output")
    compare_parser.add_argument("--format", choices=["table", "json"], default="table")
    compare_parser.set_defaults(func=_handle_compare)

    recommend_parser = subparsers.add_parser("recommend", help="Recommend best kernels")
    recommend_parser.add_argument("--input", required=True, help="Benchmark JSON input")
    recommend_parser.add_argument(
        "--objective",
        choices=["latency", "throughput", "balanced"],
        default="latency",
        help="Optimization target",
    )
    recommend_parser.add_argument("--top", type=int, default=5, help="Max recommendations")
    recommend_parser.add_argument(
        "--no-pareto",
        action="store_true",
        help="Disable Pareto filtering before ranking",
    )
    recommend_parser.add_argument("--output", help="Write recommendation JSON output")
    recommend_parser.add_argument("--format", choices=["table", "json"], default="table")
    recommend_parser.set_defaults(func=_handle_recommend)

    list_parser = subparsers.add_parser("list", help="List commands, operations, kernels, or extras")
    list_parser.add_argument(
        "subject",
        nargs="?",
        choices=["commands", "operations", "kernels", "extras"],
        default="commands",
    )
    list_parser.add_argument("--format", choices=["table", "json"], default="table")
    list_parser.set_defaults(func=_handle_list)

    report_parser = subparsers.add_parser("report", help="Summarize benchmark results")
    report_parser.add_argument("--input", required=True, help="Benchmark JSON input")
    report_parser.add_argument("--top", type=int, default=10, help="Top N entries")
    report_parser.add_argument("--output", help="Write report JSON output")
    report_parser.add_argument("--format", choices=["table", "json"], default="table")
    report_parser.set_defaults(func=_handle_report)

    history_parser = subparsers.add_parser("history", help="List run history from runs directory")
    history_parser.add_argument("--runs-dir", default="runs", help="Runs directory")
    history_parser.add_argument("--kind", help="Filter by kind (benchmark, scan-hardware, scan-env)")
    history_parser.add_argument("--limit", type=int, default=20, help="Max entries")
    history_parser.add_argument("--output", help="Write history JSON output")
    history_parser.add_argument("--format", choices=["table", "json"], default="table")
    history_parser.set_defaults(func=_handle_history)

    export_parser = subparsers.add_parser("export", help="Export benchmark results")
    export_parser.add_argument("--input", required=True, help="Benchmark JSON input")
    export_parser.add_argument("--output", required=True, help="Output file path")
    export_parser.add_argument("--format", choices=["csv", "json"], default="csv")
    export_parser.add_argument("--section", choices=["results", "summary"], default="results")
    export_parser.add_argument("--top", type=int, default=10, help="Top N (summary export)")
    export_parser.set_defaults(func=_handle_export)

    config_parser = subparsers.add_parser("config", help="Show or write default config")
    config_parser.add_argument("--write", help="Write default config JSON to path")
    config_parser.add_argument(
        "--example",
        action="store_true",
        help="Print a sample benchmark command",
    )
    config_parser.set_defaults(func=_handle_config)

    profile_parser = subparsers.add_parser("profile", help="Quick profiling wrapper around benchmark")
    profile_parser.add_argument("operation", nargs="?", help="Operation name (default: matmul)")
    profile_parser.add_argument("--size", help="Override size")
    profile_parser.add_argument("--dtype", help="Comma-separated dtypes")
    profile_parser.add_argument("--kernels", default="all", help="Kernel names or 'all'")
    profile_parser.add_argument("--hardware", default="cpu", help="Device, e.g. cpu or cuda:0")
    profile_parser.add_argument("--iterations", type=int, default=30, help="Benchmark iterations")
    profile_parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    profile_parser.add_argument("--output", help="Write JSON output to path")
    profile_parser.add_argument(
        "--format", choices=["table", "json"], default="table", help="Output format"
    )
    profile_parser.add_argument("--verbose", action="store_true", help="Verbose output")
    profile_parser.set_defaults(func=_handle_profile)

    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "func", None)
    if handler is None:
        parser.print_help()
        return 1

    try:
        return handler(args)
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())

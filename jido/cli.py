from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, List, Tuple

from jido.core.hardware import detect


def _safe_import_rich():
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table

        return Console, Panel, Table
    except Exception:
        return None, None, None


def _render_scan_output(
    hardware: Dict[str, Any],
    env: Dict[str, Any],
    hints: List[str],
) -> None:
    Console, Panel, Table = _safe_import_rich()
    if Console is None:
        print("JIDO Scan")
        print(hardware)
        print(env)
        if hints:
            print("Install hints:")
            for hint in hints:
                print(f"- {hint}")
        return

    console = Console()
    console.print(Panel.fit("[bold]JIDO Scan[/bold]", border_style="cyan"))

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
    hardware, env, hints, _run_dir = detect.scan(
        out_dir=args.out,
        deep=args.deep,
        install_hints=args.install_hints,
    )
    if not args.json_only:
        _render_scan_output(hardware, env, hints)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jido")
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan_parser = subparsers.add_parser("scan", help="Detect hardware and runtime info")
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

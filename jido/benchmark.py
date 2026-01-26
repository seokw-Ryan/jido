from __future__ import annotations

from dataclasses import dataclass
import statistics
import time
from typing import Any, Callable, Dict, List, Optional

from jido.core.hardware import detect
from jido.registry import KernelWrapper


def _try_import_torch():
    try:
        import torch

        return torch
    except Exception:
        return None


def _try_import_pynvml():
    try:
        import pynvml

        return pynvml
    except Exception:
        return None


def _device_index(device: str) -> int:
    if ":" in device:
        _, index = device.split(":", 1)
        try:
            return int(index)
        except ValueError:
            return 0
    return 0


@dataclass(frozen=True)
class BenchmarkStats:
    iterations: int
    mean_ms: float
    median_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float
    std_ms: float


@dataclass(frozen=True)
class BenchmarkResult:
    operation: str
    kernel_name: str
    backend: Optional[str]
    kernel_type: Optional[str]
    timings_ms: List[float]
    stats: BenchmarkStats
    memory_bytes: Optional[List[int]] = None
    utilization: Optional[List[Dict[str, int]]] = None
    hardware: Optional[Dict[str, Any]] = None
    env: Optional[Dict[str, Any]] = None
    machine_id: Optional[str] = None
    scan_hints: Optional[List[str]] = None


class _Timer:
    def __init__(self, use_cuda: bool) -> None:
        self._use_cuda = use_cuda
        self._torch = _try_import_torch() if use_cuda else None

    def time_call(self, func: Callable[[], Any]) -> float:
        if self._use_cuda and self._torch is not None:
            start = self._torch.cuda.Event(enable_timing=True)
            end = self._torch.cuda.Event(enable_timing=True)
            start.record()
            func()
            end.record()
            self._torch.cuda.synchronize()
            return float(start.elapsed_time(end))
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        return (end - start) * 1000.0


class BenchmarkRunner:
    def __init__(
        self,
        device: str = "cpu",
        warmup: int = 5,
        iterations: int = 50,
        use_gpu_timer: Optional[bool] = None,
        track_memory: bool = True,
        track_utilization: bool = False,
        include_system_info: bool = False,
        deep_scan: bool = False,
        install_hints: bool = False,
    ) -> None:
        self.device = device
        self.warmup = warmup
        self.iterations = iterations
        self._torch = _try_import_torch()
        self._use_cuda = self._resolve_use_cuda(use_gpu_timer)
        self._timer = _Timer(self._use_cuda)
        self.track_memory = track_memory and self._use_cuda and self._torch is not None
        self.track_utilization = track_utilization and self._use_cuda
        self._pynvml = _try_import_pynvml() if self.track_utilization else None
        self._nvml_handle = None
        self._hardware: Optional[Dict[str, Any]] = None
        self._env: Optional[Dict[str, Any]] = None
        self._machine_id: Optional[str] = None
        self._scan_hints: Optional[List[str]] = None
        if self._pynvml is not None:
            self._pynvml.nvmlInit()
            self._nvml_handle = self._pynvml.nvmlDeviceGetHandleByIndex(
                _device_index(device)
            )
        if include_system_info:
            hardware, env, hints, machine_id = detect.scan(
                deep=deep_scan,
                install_hints=install_hints,
            )
            self._hardware = hardware
            self._env = env
            self._scan_hints = hints
            self._machine_id = machine_id

    def _resolve_use_cuda(self, use_gpu_timer: Optional[bool]) -> bool:
        if use_gpu_timer is not None:
            return use_gpu_timer
        if self._torch is None:
            return False
        if not self.device.startswith("cuda"):
            return False
        return bool(self._torch.cuda.is_available())

    def _call_kernel(self, kernel: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(kernel, KernelWrapper):
            return kernel(*args, **kwargs)
        if callable(kernel):
            return kernel(*args, **kwargs)
        raise TypeError("Kernel must be a callable or KernelWrapper")

    def _capture_utilization(self) -> Optional[Dict[str, int]]:
        if self._pynvml is None or self._nvml_handle is None:
            return None
        utilization = self._pynvml.nvmlDeviceGetUtilizationRates(
            self._nvml_handle
        )
        return {
            "gpu": int(utilization.gpu),
            "memory": int(utilization.memory),
        }

    def run_single_benchmark(self, kernel: Any, *args: Any, **kwargs: Any) -> float:
        return self._timer.time_call(lambda: self._call_kernel(kernel, *args, **kwargs))

    def run_with_warmup(self, kernel: Any, *args: Any, **kwargs: Any) -> List[float]:
        for _ in range(self.warmup):
            self._call_kernel(kernel, *args, **kwargs)
            if self._use_cuda and self._torch is not None:
                self._torch.cuda.synchronize()
        return self.run_multiple_iterations(kernel, *args, **kwargs).timings_ms

    def run_multiple_iterations(
        self, kernel: Any, *args: Any, **kwargs: Any
    ) -> BenchmarkResult:
        timings: List[float] = []
        memory: List[int] = []
        utilization: List[Dict[str, int]] = []

        for _ in range(self.iterations):
            if self.track_memory and self._torch is not None:
                self._torch.cuda.reset_peak_memory_stats()

            elapsed = self._timer.time_call(
                lambda: self._call_kernel(kernel, *args, **kwargs)
            )
            timings.append(elapsed)

            if self.track_memory and self._torch is not None:
                memory.append(int(self._torch.cuda.max_memory_allocated()))

            if self.track_utilization:
                entry = self._capture_utilization()
                if entry is not None:
                    utilization.append(entry)

        stats = _compute_stats(timings)

        if isinstance(kernel, KernelWrapper):
            operation = kernel.operation
            kernel_name = kernel.name
            backend = kernel.backend
            kernel_type = kernel.kernel_type
        else:
            operation = "unknown"
            kernel_name = getattr(kernel, "__name__", "kernel")
            backend = None
            kernel_type = None

        return BenchmarkResult(
            operation=operation,
            kernel_name=kernel_name,
            backend=backend,
            kernel_type=kernel_type,
            timings_ms=timings,
            stats=stats,
            memory_bytes=memory or None,
            utilization=utilization or None,
            hardware=self._hardware,
            env=self._env,
            machine_id=self._machine_id,
            scan_hints=self._scan_hints,
        )


def _compute_stats(timings: List[float]) -> BenchmarkStats:
    timings_sorted = sorted(timings)
    count = len(timings_sorted)
    if count == 0:
        return BenchmarkStats(
            iterations=0,
            mean_ms=0.0,
            median_ms=0.0,
            p95_ms=0.0,
            min_ms=0.0,
            max_ms=0.0,
            std_ms=0.0,
        )
    p95_index = max(int(count * 0.95) - 1, 0)
    return BenchmarkStats(
        iterations=count,
        mean_ms=float(statistics.mean(timings_sorted)),
        median_ms=float(statistics.median(timings_sorted)),
        p95_ms=float(timings_sorted[p95_index]),
        min_ms=float(min(timings_sorted)),
        max_ms=float(max(timings_sorted)),
        std_ms=float(statistics.pstdev(timings_sorted)) if count > 1 else 0.0,
    )

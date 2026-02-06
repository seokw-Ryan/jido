from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from jido.core.env.discovery import discover_env
from jido.core.hardware.accelerators import detect_accelerators
from jido.core.hardware import amd, intel, nvidia


def _try_import_torch():
    try:
        import torch

        return torch
    except Exception:
        return None


def _cuda_cores_per_sm(major: int, minor: int) -> Optional[int]:
    # Best-effort mapping for NVIDIA architectures.
    mapping = {
        (5, 0): 128,
        (5, 2): 128,
        (6, 0): 64,
        (6, 1): 128,
        (6, 2): 128,
        (7, 0): 64,
        (7, 5): 64,
        (8, 0): 64,
        (8, 6): 128,
        (8, 9): 128,
        (9, 0): 128,
    }
    return mapping.get((major, minor))


def _augment_with_torch_gpu_info(gpus: List[Dict[str, Any]]) -> None:
    torch = _try_import_torch()
    if torch is None:
        return
    if not torch.cuda.is_available():
        return

    count = torch.cuda.device_count()
    for index in range(min(count, len(gpus))):
        props = torch.cuda.get_device_properties(index)
        major = int(getattr(props, "major", 0))
        minor = int(getattr(props, "minor", 0))
        cores_per_sm = _cuda_cores_per_sm(major, minor)
        sm_count = int(getattr(props, "multi_processor_count", 0))
        clock_rate_khz = float(getattr(props, "clock_rate", 0.0))
        mem_clock_khz = float(getattr(props, "memory_clock_rate", 0.0))
        mem_bus_width = float(getattr(props, "memory_bus_width", 0.0))

        peak_fp32 = None
        if cores_per_sm and sm_count and clock_rate_khz:
            peak_fp32 = (
                cores_per_sm * sm_count * clock_rate_khz * 1000.0 * 2.0
            )

        peak_bandwidth = None
        if mem_clock_khz and mem_bus_width:
            peak_bandwidth = (
                2.0 * mem_clock_khz * 1000.0 * (mem_bus_width / 8.0)
            )

        gpus[index].update(
            {
                "index": index,
                "compute_capability": f"{major}.{minor}",
                "sm_count": sm_count or None,
                "clock_rate_khz": int(clock_rate_khz) if clock_rate_khz else None,
                "memory_clock_khz": int(mem_clock_khz) if mem_clock_khz else None,
                "memory_bus_width_bits": int(mem_bus_width) if mem_bus_width else None,
                "theoretical_peak_fp32_flops": peak_fp32,
                "theoretical_peak_mem_bytes_per_s": peak_bandwidth,
            }
        )


def _cpu_info() -> Dict[str, Any]:
    cpu = {
        "brand": "unknown",
        "physical_cores": None,
        "logical_cores": None,
        "flags": [],
    }

    try:
        import psutil  # type: ignore

        cpu["physical_cores"] = psutil.cpu_count(logical=False)
        cpu["logical_cores"] = psutil.cpu_count(logical=True)
    except Exception:
        cpu["physical_cores"] = os.cpu_count()
        cpu["logical_cores"] = os.cpu_count()

    try:
        import cpuinfo  # type: ignore

        info = cpuinfo.get_cpu_info()
        cpu["brand"] = info.get("brand_raw") or cpu["brand"]
        cpu["flags"] = info.get("flags") or []
    except Exception:
        pass

    return cpu


def _memory_total_gb() -> Optional[int]:
    try:
        import psutil  # type: ignore

        total = psutil.virtual_memory().total
        return int(round(total / (1024**3)))
    except Exception:
        pass

    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        return int(round(total / (1024**3)))
    except Exception:
        return None


def _detect_gpu_vendors_from_lspci() -> List[str]:
    try:
        result = subprocess.run(
            ["lspci"],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return []

    if result.returncode != 0:
        return []

    vendors = set()
    for line in result.stdout.lower().splitlines():
        if "vga" not in line and "3d controller" not in line:
            continue
        if "nvidia" in line:
            vendors.add("nvidia")
        if "amd" in line or "advanced micro devices" in line or "ati" in line:
            vendors.add("amd")
        if "intel" in line:
            vendors.add("intel")
    return sorted(vendors)


def _runtime_flags_for_gpu(vendor: str, env: Dict[str, Any]) -> Dict[str, bool]:
    frameworks = env.get("frameworks", {})
    vendor_tools = env.get("vendor_tools", {})

    cuda = vendor == "nvidia" and (
        vendor_tools.get("nvidia_smi", {}).get("available") is True
        or frameworks.get("torch", {}).get("available") is True
    )

    rocm = vendor == "amd" and (
        vendor_tools.get("rocm_smi", {}).get("available") is True
        or vendor_tools.get("amd_smi", {}).get("available") is True
    )

    directml = (
        frameworks.get("directml", {}).get("available") is True
        or frameworks.get("torch_directml", {}).get("available") is True
    )
    openvino = frameworks.get("openvino", {}).get("available") is True

    return {
        "cuda": cuda,
        "rocm": rocm,
        "directml": directml,
        "openvino": openvino,
    }


def aggregate_runtime_flags(gpus: List[Dict[str, Any]]) -> Dict[str, bool]:
    runtime_flags = {"cuda": False, "rocm": False, "directml": False, "openvino": False}
    for gpu in gpus:
        runtime = gpu.get("runtime", {})
        for key in runtime_flags:
            runtime_flags[key] = runtime_flags[key] or bool(runtime.get(key))
    return runtime_flags


def compute_machine_id(
    cpu: Dict[str, Any],
    memory_total_gb: Optional[int],
    os_arch: str,
    gpus: List[Dict[str, Any]],
    accelerators: Optional[List[Dict[str, Any]]] = None,
) -> str:
    gpu_names = sorted({gpu.get("name", "") for gpu in gpus if gpu.get("name")})
    accelerator_names = sorted(
        {
            device.get("name", "")
            for device in (accelerators or [])
            if device.get("name")
        }
    )
    payload = "|".join(
        [
            str(cpu.get("brand", "")),
            str(cpu.get("physical_cores", "")),
            str(cpu.get("logical_cores", "")),
            str(memory_total_gb if memory_total_gb is not None else ""),
            os_arch,
            ",".join(gpu_names),
            ",".join(accelerator_names),
        ]
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return digest[:16]


def scan(
    deep: bool = False,
    install_hints: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any], List[str], str]:
    cpu = _cpu_info()
    memory_total_gb = _memory_total_gb()

    hardware = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "arch": platform.machine(),
        },
        "python": {"version": sys.version.split()[0]},
        "cpu": cpu,
        "memory": {"total_gb": memory_total_gb},
        "gpus": [],
        "accelerators": [],
    }

    env = discover_env()

    gpus: List[Dict[str, Any]] = []
    gpus.extend(nvidia.detect_gpus(deep=deep))
    gpus.extend(amd.detect_gpus(deep=deep))
    gpus.extend(intel.detect_gpus(deep=deep))

    for gpu in gpus:
        vendor = gpu.get("vendor", "unknown")
        gpu["runtime"] = _runtime_flags_for_gpu(vendor, env)

    if any(gpu.get("vendor") == "nvidia" for gpu in gpus):
        _augment_with_torch_gpu_info(gpus)

    accelerators = detect_accelerators(env=env, deep=deep)

    hardware["gpus"] = gpus
    hardware["accelerators"] = accelerators

    machine_id = compute_machine_id(
        cpu=cpu,
        memory_total_gb=memory_total_gb,
        os_arch=hardware["os"]["arch"],
        gpus=gpus,
        accelerators=accelerators,
    )
    hardware["machine_id"] = machine_id

    hints: List[str] = []
    vendor_tools = env.get("vendor_tools", {})
    detected_vendors = {gpu.get("vendor") for gpu in gpus}
    detected_vendors.update(_detect_gpu_vendors_from_lspci())

    if "nvidia" in detected_vendors and not vendor_tools.get("nvidia_smi", {}).get("available"):
        if install_hints or deep:
            hints.append(
                "NVIDIA GPU detected but NVML/nvidia-smi not found. Install NVIDIA drivers + nvidia-smi."
            )
    if "amd" in detected_vendors and not (
        vendor_tools.get("rocm_smi", {}).get("available")
        or vendor_tools.get("amd_smi", {}).get("available")
    ):
        if install_hints or deep:
            hints.append(
                "AMD GPU detected but rocm-smi/amd-smi not found. Install ROCm or AMD GPU tools."
            )
    if "intel" in detected_vendors and not vendor_tools.get("sycl_ls", {}).get("available"):
        if install_hints or deep:
            hints.append(
                "Intel GPU detected but sycl-ls not found. Install Intel oneAPI/Level Zero tools."
            )

    accelerator_types = {device.get("type") for device in accelerators}
    if "fpga" in accelerator_types and not vendor_tools.get("xbutil", {}).get("available"):
        if install_hints or deep:
            hints.append(
                "FPGA detected but xbutil was not found. Install Xilinx/AMD management tools."
            )
    if "tpu" in accelerator_types and not (
        vendor_tools.get("edgetpu_compiler", {}).get("available")
        or env.get("frameworks", {}).get("libtpu", {}).get("available")
    ):
        if install_hints or deep:
            hints.append(
                "TPU-like device detected but Edge TPU/libtpu runtime not found. Install TPU runtime tools."
            )
    if "npu" in accelerator_types and not vendor_tools.get("npu_smi", {}).get("available"):
        if install_hints or deep:
            hints.append(
                "NPU-like device detected but npu-smi not found. Install vendor NPU runtime tools."
            )

    return hardware, env, hints, machine_id

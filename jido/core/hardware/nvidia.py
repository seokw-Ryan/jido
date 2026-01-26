from __future__ import annotations

import re
import shutil
import subprocess
from typing import Any, Dict, List, Optional


def _parse_vram_mb(value: str) -> Optional[int]:
    match = re.search(r"(\d+)\s*MiB", value)
    if match:
        return int(match.group(1))
    match = re.search(r"(\d+)", value)
    return int(match.group(1)) if match else None


def _detect_with_nvml() -> List[Dict[str, Any]]:
    try:
        import pynvml  # type: ignore
    except Exception:
        return []

    gpus: List[Dict[str, Any]] = []
    try:
        pynvml.nvmlInit()
        driver = pynvml.nvmlSystemGetDriverVersion()
        count = pynvml.nvmlDeviceGetCount()
        for index in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            name = pynvml.nvmlDeviceGetName(handle)
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpus.append(
                {
                    "vendor": "nvidia",
                    "name": name.decode() if isinstance(name, bytes) else str(name),
                    "vram_total_mb": int(memory.total / (1024 * 1024)),
                    "driver": driver.decode() if isinstance(driver, bytes) else str(driver),
                }
            )
    except Exception:
        return []
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

    return gpus


def _detect_with_nvidia_smi() -> List[Dict[str, Any]]:
    if not shutil.which("nvidia-smi"):
        return []

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return []

    if result.returncode != 0:
        return []

    gpus: List[Dict[str, Any]] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if not parts or not parts[0]:
            continue
        name = parts[0]
        vram = _parse_vram_mb(parts[1]) if len(parts) > 1 else None
        driver = parts[2] if len(parts) > 2 else None
        gpus.append(
            {
                "vendor": "nvidia",
                "name": name,
                "vram_total_mb": vram,
                "driver": driver,
            }
        )

    return gpus


def detect_gpus(deep: bool = False) -> List[Dict[str, Any]]:
    gpus = _detect_with_nvml()
    if gpus:
        return gpus

    if deep or shutil.which("nvidia-smi"):
        return _detect_with_nvidia_smi()

    return []

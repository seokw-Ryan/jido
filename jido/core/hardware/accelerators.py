from __future__ import annotations

import shutil
import subprocess
from typing import Any, Dict, Iterable, List, Sequence


def _run_command(command: Sequence[str]) -> str:
    if not command:
        return ""
    if shutil.which(command[0]) is None:
        return ""

    try:
        result = subprocess.run(
            list(command),
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return ""

    if result.returncode != 0:
        return ""
    return result.stdout or ""


def _line_vendor(line: str) -> str:
    lower = line.lower()
    if "intel" in lower:
        return "intel"
    if "xilinx" in lower or "amd" in lower:
        return "amd/xilinx"
    if "google" in lower:
        return "google"
    if "habanalabs" in lower:
        return "habana"
    if "huawei" in lower:
        return "huawei"
    return "unknown"


def _accelerator_type(line: str) -> str | None:
    lower = line.lower()
    if any(token in lower for token in ["fpga", "xilinx", "altera"]):
        return "fpga"
    if any(token in lower for token in ["tpu", "tensor processing unit", "coral"]):
        return "tpu"
    if any(token in lower for token in ["npu", "neural", "habanalabs", "gaudi", "npu-smi"]):
        return "npu"
    return None


def _detect_from_lines(lines: Iterable[str], source: str) -> List[Dict[str, Any]]:
    devices: List[Dict[str, Any]] = []
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        kind = _accelerator_type(line)
        if kind is None:
            continue
        devices.append(
            {
                "type": kind,
                "vendor": _line_vendor(line),
                "name": line,
                "source": source,
            }
        )
    return devices


def _dedupe_devices(devices: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for device in devices:
        key = (
            str(device.get("type", "")),
            str(device.get("vendor", "")),
            str(device.get("name", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(device)
    return deduped


def _runtime_flags(env: Dict[str, Any]) -> Dict[str, bool]:
    frameworks = env.get("frameworks", {})
    vendor_tools = env.get("vendor_tools", {})
    return {
        "xla": bool(
            frameworks.get("jax", {}).get("available")
            or frameworks.get("jaxlib", {}).get("available")
            or frameworks.get("tensorflow", {}).get("available")
            or frameworks.get("libtpu", {}).get("available")
        ),
        "openvino": bool(frameworks.get("openvino", {}).get("available")),
        "onednn": bool(frameworks.get("tensorflow", {}).get("available")),
        "oneapi": bool(vendor_tools.get("sycl_ls", {}).get("available")),
    }


def detect_accelerators(env: Dict[str, Any], deep: bool = False) -> List[Dict[str, Any]]:
    lspci_lines = _run_command(["lspci"]).splitlines()
    lsusb_lines = _run_command(["lsusb"]).splitlines() if deep else []

    devices: List[Dict[str, Any]] = []
    devices.extend(_detect_from_lines(lspci_lines, source="lspci"))
    devices.extend(_detect_from_lines(lsusb_lines, source="lsusb"))

    runtime = _runtime_flags(env)
    enriched: List[Dict[str, Any]] = []
    for device in _dedupe_devices(devices):
        entry = dict(device)
        entry["runtime"] = runtime
        enriched.append(entry)
    return enriched

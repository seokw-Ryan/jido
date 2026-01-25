from __future__ import annotations

import json
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


def _extract_from_json(payload: Any) -> List[Dict[str, Any]]:
    gpus: List[Dict[str, Any]] = []

    if isinstance(payload, dict):
        items = payload.values()
    elif isinstance(payload, list):
        items = payload
    else:
        items = []

    for item in items:
        if not isinstance(item, dict):
            continue
        name = (
            item.get("Product Name")
            or item.get("Card series")
            or item.get("Card Series")
            or item.get("card series")
            or item.get("Name")
        )
        vram = None
        for key in [
            "VRAM Total",
            "VRAM Total (B)",
            "VRAM Total (MB)",
            "vram_total",
            "VRAM Total (MiB)",
        ]:
            if key in item:
                vram = _parse_vram_mb(str(item[key]))
                break
        if name:
            gpus.append(
                {
                    "vendor": "amd",
                    "name": str(name),
                    "vram_total_mb": vram,
                    "driver": None,
                }
            )
    return gpus


def _detect_with_tool(command: str) -> List[Dict[str, Any]]:
    if not shutil.which(command):
        return []

    try:
        result = subprocess.run(
            [command, "--json", "--showproductname", "--showmeminfo", "vram"],
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return []

    if result.returncode != 0:
        return []

    try:
        payload = json.loads(result.stdout)
        gpus = _extract_from_json(payload)
        if gpus:
            return gpus
    except Exception:
        pass

    gpus: List[Dict[str, Any]] = []
    for line in result.stdout.splitlines():
        if "Card series" in line or "Card Series" in line:
            name = line.split(":", 1)[-1].strip()
            gpus.append(
                {
                    "vendor": "amd",
                    "name": name,
                    "vram_total_mb": None,
                    "driver": None,
                }
            )
        if "VRAM Total" in line and gpus:
            vram = _parse_vram_mb(line)
            gpus[-1]["vram_total_mb"] = vram

    return gpus


def detect_gpus(deep: bool = False) -> List[Dict[str, Any]]:
    gpus = _detect_with_tool("rocm-smi")
    if gpus:
        return gpus

    if deep or shutil.which("amd-smi"):
        return _detect_with_tool("amd-smi")

    return []

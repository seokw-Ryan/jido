from __future__ import annotations

import shutil
import subprocess
from typing import Any, Dict, List


def detect_gpus(deep: bool = False) -> List[Dict[str, Any]]:
    if not shutil.which("sycl-ls"):
        return []

    try:
        result = subprocess.run(
            ["sycl-ls"],
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
        if "GPU" not in line:
            continue
        name = line.strip()
        gpus.append(
            {
                "vendor": "intel",
                "name": name,
                "vram_total_mb": None,
                "driver": None,
            }
        )

    return gpus

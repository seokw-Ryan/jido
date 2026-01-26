from __future__ import annotations

import importlib.metadata
import shutil
from typing import Any, Dict, Optional


def _module_version(module_name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(module_name)
    except importlib.metadata.PackageNotFoundError:
        return None
    except Exception:
        return None


def _tool_info(command: str) -> Dict[str, Any]:
    path = shutil.which(command)
    return {"available": bool(path), "path": path}


def discover_env() -> Dict[str, Any]:
    frameworks = {}
    for module in ["torch", "onnxruntime", "transformers", "openvino", "directml", "torch_directml"]:
        version = _module_version(module)
        frameworks[module] = {"available": version is not None, "version": version}

    vendor_tools = {
        "nvidia_smi": _tool_info("nvidia-smi"),
        "rocm_smi": _tool_info("rocm-smi"),
        "amd_smi": _tool_info("amd-smi"),
        "sycl_ls": _tool_info("sycl-ls"),
    }

    return {
        "frameworks": frameworks,
        "vendor_tools": vendor_tools,
    }

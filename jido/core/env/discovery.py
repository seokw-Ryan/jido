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
    for module in [
        "torch",
        "onnxruntime",
        "transformers",
        "openvino",
        "directml",
        "torch_directml",
        "tensorflow",
        "jax",
        "jaxlib",
        "libtpu",
    ]:
        version = _module_version(module)
        frameworks[module] = {"available": version is not None, "version": version}

    compilers = {
        "gcc": _tool_info("gcc"),
        "clang": _tool_info("clang"),
        "nvcc": _tool_info("nvcc"),
        "hipcc": _tool_info("hipcc"),
        "icx": _tool_info("icx"),
    }

    vendor_tools = {
        "nvidia_smi": _tool_info("nvidia-smi"),
        "rocm_smi": _tool_info("rocm-smi"),
        "amd_smi": _tool_info("amd-smi"),
        "sycl_ls": _tool_info("sycl-ls"),
        "edgetpu_compiler": _tool_info("edgetpu_compiler"),
        "xbutil": _tool_info("xbutil"),
        "npu_smi": _tool_info("npu-smi"),
        "lspci": _tool_info("lspci"),
        "lsusb": _tool_info("lsusb"),
    }

    return {
        "frameworks": frameworks,
        "compilers": compilers,
        "vendor_tools": vendor_tools,
    }

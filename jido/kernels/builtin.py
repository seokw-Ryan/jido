from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

from jido.registry import register_kernel

_REGISTERED = False
_COMPILED_CACHE: Dict[Tuple[str, str, str], Callable[..., Any]] = {}


def _dtype_key(tensor: Any) -> str:
    dtype = getattr(tensor, "dtype", None)
    if dtype is None:
        return "unknown"
    return str(dtype)


def _shape_key(tensor: Any) -> str:
    shape = getattr(tensor, "shape", None)
    if shape is None:
        return "unknown"
    return "x".join(str(dim) for dim in shape)


def _torch_matmul(a: Any, b: Any) -> Any:
    import torch

    return torch.matmul(a, b)


def _torch_einsum(a: Any, b: Any) -> Any:
    import torch

    if getattr(a, "dim", lambda: 0)() == 2 and getattr(b, "dim", lambda: 0)() == 2:
        return torch.einsum("mk,kn->mn", a, b)
    if getattr(a, "dim", lambda: 0)() == 3 and getattr(b, "dim", lambda: 0)() == 3:
        return torch.einsum("bmk,bkn->bmn", a, b)
    return torch.matmul(a, b)


def _torch_mm_or_bmm(a: Any, b: Any) -> Any:
    import torch

    if getattr(a, "dim", lambda: 0)() == 2 and getattr(b, "dim", lambda: 0)() == 2:
        return torch.mm(a, b)
    if getattr(a, "dim", lambda: 0)() == 3 and getattr(b, "dim", lambda: 0)() == 3:
        return torch.bmm(a, b)
    return torch.matmul(a, b)


def _torch_compiled_matmul(a: Any, b: Any) -> Any:
    import torch

    if not hasattr(torch, "compile"):
        return torch.matmul(a, b)

    key = (
        str(getattr(a, "device", "unknown")),
        _dtype_key(a),
        f"{_shape_key(a)}|{_shape_key(b)}",
    )

    compiled = _COMPILED_CACHE.get(key)
    if compiled is None:
        fn = torch.compile(_torch_matmul)
        _COMPILED_CACHE[key] = fn
        compiled = fn
    return compiled(a, b)


def ensure_builtin_kernels_registered() -> None:
    global _REGISTERED
    if _REGISTERED:
        return

    register_kernel(
        operation="matmul",
        name="torch_matmul",
        backend="torch",
        kernel_type="eager",
    )(_torch_matmul)
    register_kernel(
        operation="matmul",
        name="torch_einsum",
        backend="torch",
        kernel_type="eager",
    )(_torch_einsum)
    register_kernel(
        operation="matmul",
        name="torch_mm_or_bmm",
        backend="torch",
        kernel_type="eager",
    )(_torch_mm_or_bmm)
    register_kernel(
        operation="matmul",
        name="torch_compiled",
        backend="torch",
        kernel_type="compiled",
    )(_torch_compiled_matmul)

    _REGISTERED = True

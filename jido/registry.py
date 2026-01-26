from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Any, Callable, Dict, Iterable, List, Optional


_KERNEL_REGISTRY: Dict[str, Dict[str, "KernelWrapper"]] = {}


@dataclass(frozen=True)
class KernelWrapper:
    operation: str
    name: str
    kernel: Any
    backend: Optional[str] = None
    kernel_type: Optional[str] = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if hasattr(self.kernel, "run") and callable(getattr(self.kernel, "run")):
            return self.kernel.run(*args, **kwargs)
        if hasattr(self.kernel, "launch") and callable(getattr(self.kernel, "launch")):
            return self.kernel.launch(*args, **kwargs)
        if callable(self.kernel):
            return self.kernel(*args, **kwargs)
        raise TypeError(f"Kernel {self.name} is not callable")


def _validate_kernel_signature(kernel: Any) -> None:
    if not callable(kernel):
        raise TypeError("Kernel must be callable")
    try:
        signature = inspect.signature(kernel)
    except (TypeError, ValueError):
        return
    params = list(signature.parameters.values())
    if not params:
        raise ValueError(
            "Kernel must accept at least one positional argument or *args"
        )
    if any(param.kind == param.VAR_POSITIONAL for param in params):
        return


def register_kernel(
    operation: str,
    name: Optional[str] = None,
    backend: Optional[str] = None,
    kernel_type: Optional[str] = None,
) -> Callable[[Any], Any]:
    def decorator(kernel: Any) -> Any:
        _validate_kernel_signature(kernel)
        kernel_name = name or getattr(kernel, "__name__", None) or str(kernel)
        registry = _KERNEL_REGISTRY.setdefault(operation, {})
        if kernel_name in registry:
            raise ValueError(
                f"Kernel already registered: {operation}/{kernel_name}"
            )
        registry[kernel_name] = KernelWrapper(
            operation=operation,
            name=kernel_name,
            kernel=kernel,
            backend=backend,
            kernel_type=kernel_type,
        )
        return kernel

    return decorator


def get_kernels(operation: str) -> List[KernelWrapper]:
    return list(_KERNEL_REGISTRY.get(operation, {}).values())


def list_kernels() -> Dict[str, Iterable[KernelWrapper]]:
    return {operation: list(kernels.values()) for operation, kernels in _KERNEL_REGISTRY.items()}

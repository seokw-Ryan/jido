from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


def _require_torch():
    try:
        import torch

        return torch
    except Exception as exc:
        raise RuntimeError("PyTorch is required for operations") from exc


@dataclass(frozen=True)
class CorrectnessResult:
    passed: bool
    max_error: float
    mean_error: float
    atol: float
    rtol: float


class Operation:
    name: str = "operation"

    def sizes(self) -> Iterable[Dict[str, int]]:
        raise NotImplementedError

    def dtypes(self) -> Sequence[Any]:
        raise NotImplementedError

    def reference_kernel(self) -> Callable[..., Any]:
        raise NotImplementedError

    def generate_inputs(
        self, size: Dict[str, int], dtype: Any, device: str
    ) -> Tuple[Any, ...]:
        raise NotImplementedError

    def flops(self, size: Dict[str, int]) -> Optional[float]:
        return None

    def validate_correctness(
        self,
        kernel: Callable[..., Any],
        inputs: Tuple[Any, ...],
        atol: Optional[float] = None,
        rtol: Optional[float] = None,
    ) -> CorrectnessResult:
        torch = _require_torch()
        reference = self.reference_kernel()
        output = kernel(*inputs)
        expected = reference(*inputs)

        if hasattr(output, "detach"):
            output_tensor = output.detach()
        else:
            output_tensor = torch.tensor(output)

        if hasattr(expected, "detach"):
            expected_tensor = expected.detach()
        else:
            expected_tensor = torch.tensor(expected)

        dtype = output_tensor.dtype
        if atol is None or rtol is None:
            atol, rtol = _default_tolerance(dtype)

        diff = (output_tensor - expected_tensor).abs()
        max_error = float(diff.max().item()) if diff.numel() else 0.0
        mean_error = float(diff.mean().item()) if diff.numel() else 0.0
        passed = torch.allclose(
            output_tensor, expected_tensor, atol=atol, rtol=rtol
        )
        return CorrectnessResult(
            passed=bool(passed),
            max_error=max_error,
            mean_error=mean_error,
            atol=float(atol),
            rtol=float(rtol),
        )


def _default_tolerance(dtype: Any) -> Tuple[float, float]:
    torch = _require_torch()
    if dtype in (torch.float16, torch.bfloat16):
        return 1e-2, 1e-2
    return 1e-4, 1e-4

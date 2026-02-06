from __future__ import annotations

from typing import Any, Dict, Iterable, Sequence, Tuple

from jido.operations.base import Operation, _require_torch


class MatMulOperation(Operation):
    name = "matmul"

    def sizes(self) -> Iterable[Dict[str, int]]:
        return [
            {"m": 1024, "n": 1024, "k": 1024, "batch": 1},
            {"m": 4096, "n": 512, "k": 1024, "batch": 1},
            {"m": 512, "n": 4096, "k": 1024, "batch": 1},
            {"m": 256, "n": 256, "k": 256, "batch": 16},
        ]

    def dtypes(self) -> Sequence[Any]:
        torch = _require_torch()
        dtypes = [torch.float32]
        if torch.cuda.is_available():
            dtypes.append(torch.float16)
        if hasattr(torch, "bfloat16"):
            dtypes.append(torch.bfloat16)
        return dtypes

    def reference_kernel(self):
        torch = _require_torch()
        return torch.matmul

    def generate_inputs(
        self, size: Dict[str, int], dtype: Any, device: str
    ) -> Tuple[Any, ...]:
        torch = _require_torch()
        m = int(size["m"])
        n = int(size["n"])
        k = int(size["k"])
        batch = int(size.get("batch", 1))

        if batch > 1:
            a = torch.randn(batch, m, k, device=device, dtype=dtype)
            b = torch.randn(batch, k, n, device=device, dtype=dtype)
        else:
            a = torch.randn(m, k, device=device, dtype=dtype)
            b = torch.randn(k, n, device=device, dtype=dtype)
        return (a, b)

    def flops(self, size: Dict[str, int]) -> float:
        m = int(size["m"])
        n = int(size["n"])
        k = int(size["k"])
        batch = int(size.get("batch", 1))
        return float(2 * m * n * k * batch)

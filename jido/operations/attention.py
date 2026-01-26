from __future__ import annotations

from typing import Any, Dict, Iterable, Sequence, Tuple

from jido.operations.base import Operation, _require_torch


class AttentionOperation(Operation):
    name = "attention"

    def sizes(self) -> Iterable[Dict[str, int]]:
        return [
            {"batch": 1, "heads": 8, "seq": 128, "head_dim": 64},
            {"batch": 2, "heads": 8, "seq": 512, "head_dim": 64},
            {"batch": 4, "heads": 16, "seq": 1024, "head_dim": 64},
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
        if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            raise RuntimeError("scaled_dot_product_attention not available")
        return torch.nn.functional.scaled_dot_product_attention

    def generate_inputs(
        self, size: Dict[str, int], dtype: Any, device: str
    ) -> Tuple[Any, ...]:
        torch = _require_torch()
        batch = int(size["batch"])
        heads = int(size["heads"])
        seq = int(size["seq"])
        head_dim = int(size["head_dim"])

        query = torch.randn(
            batch, heads, seq, head_dim, device=device, dtype=dtype
        )
        key = torch.randn(
            batch, heads, seq, head_dim, device=device, dtype=dtype
        )
        value = torch.randn(
            batch, heads, seq, head_dim, device=device, dtype=dtype
        )
        return (query, key, value)

    def flops(self, size: Dict[str, int]) -> float:
        batch = int(size["batch"])
        heads = int(size["heads"])
        seq = int(size["seq"])
        head_dim = int(size["head_dim"])
        return float(4 * batch * heads * seq * seq * head_dim)

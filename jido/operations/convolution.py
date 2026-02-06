from __future__ import annotations

from typing import Any, Dict, Iterable, Sequence, Tuple

from jido.operations.base import Operation, _require_torch


class ConvolutionOperation(Operation):
    name = "conv2d"

    def sizes(self) -> Iterable[Dict[str, int]]:
        return [
            {"batch": 1, "in_channels": 64, "out_channels": 64, "h": 56, "w": 56, "kernel": 3},
            {"batch": 4, "in_channels": 128, "out_channels": 256, "h": 28, "w": 28, "kernel": 3},
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
        return torch.nn.functional.conv2d

    def generate_inputs(
        self, size: Dict[str, int], dtype: Any, device: str
    ) -> Tuple[Any, ...]:
        torch = _require_torch()
        batch = int(size["batch"])
        in_channels = int(size["in_channels"])
        out_channels = int(size["out_channels"])
        height = int(size["h"])
        width = int(size["w"])
        kernel = int(size["kernel"])

        x = torch.randn(
            batch, in_channels, height, width, device=device, dtype=dtype
        )
        weight = torch.randn(
            out_channels, in_channels, kernel, kernel, device=device, dtype=dtype
        )
        return (x, weight)

    def flops(self, size: Dict[str, int]) -> float:
        batch = int(size["batch"])
        in_channels = int(size["in_channels"])
        out_channels = int(size["out_channels"])
        height = int(size["h"])
        width = int(size["w"])
        kernel = int(size["kernel"])
        out_h = height - kernel + 1
        out_w = width - kernel + 1
        return float(2 * batch * out_channels * out_h * out_w * in_channels * kernel * kernel)

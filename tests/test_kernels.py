from __future__ import annotations

from jido.kernels import ensure_builtin_kernels_registered
from jido.registry import get_kernels


def test_builtin_matmul_kernels_are_registered():
    ensure_builtin_kernels_registered()
    names = {kernel.name for kernel in get_kernels("matmul")}

    assert "torch_matmul" in names
    assert "torch_einsum" in names
    assert "torch_mm_or_bmm" in names
    assert "torch_compiled" in names

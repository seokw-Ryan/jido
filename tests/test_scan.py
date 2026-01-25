from __future__ import annotations

import importlib.metadata

import pytest

from jido.core.env.discovery import discover_env
from jido.core.hardware.detect import compute_machine_id


def test_machine_id_stable():
    cpu = {"brand": "Test CPU", "physical_cores": 8, "logical_cores": 16}
    gpus_a = [
        {"name": "GPU B", "vendor": "nvidia"},
        {"name": "GPU A", "vendor": "amd"},
    ]
    gpus_b = [
        {"name": "GPU A", "vendor": "amd"},
        {"name": "GPU B", "vendor": "nvidia"},
    ]

    first = compute_machine_id(cpu=cpu, memory_total_gb=32, os_arch="x86_64", gpus=gpus_a)
    second = compute_machine_id(cpu=cpu, memory_total_gb=32, os_arch="x86_64", gpus=gpus_b)

    assert first == second
    assert len(first) == 16


def test_env_discovery(monkeypatch: pytest.MonkeyPatch):
    def fake_version(module_name: str):
        if module_name == "torch":
            return "2.0.0"
        raise importlib.metadata.PackageNotFoundError

    def fake_which(command: str):
        if command == "nvidia-smi":
            return "/usr/bin/nvidia-smi"
        return None

    monkeypatch.setattr(importlib.metadata, "version", fake_version)
    monkeypatch.setattr("shutil.which", fake_which)

    env = discover_env()

    assert env["frameworks"]["torch"]["available"] is True
    assert env["frameworks"]["torch"]["version"] == "2.0.0"
    assert env["frameworks"]["onnxruntime"]["available"] is False
    assert env["vendor_tools"]["nvidia_smi"]["available"] is True

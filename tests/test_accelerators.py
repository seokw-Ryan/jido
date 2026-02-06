from __future__ import annotations

import pytest

from jido.core.hardware.accelerators import detect_accelerators


def test_detect_accelerators_from_pci_and_usb(monkeypatch: pytest.MonkeyPatch):
    def fake_run(command):
        if command == ["lspci"]:
            return (
                "03:00.0 Processing accelerators: Intel Corporation Neural Processing Unit\n"
                "04:00.0 Processing accelerators: Xilinx Device FPGA\n"
            )
        if command == ["lsusb"]:
            return "Bus 001 Device 003: ID 18d1:9302 Google Inc. Coral TPU\n"
        return ""

    monkeypatch.setattr("jido.core.hardware.accelerators._run_command", fake_run)

    env = {
        "frameworks": {
            "jax": {"available": True},
            "jaxlib": {"available": False},
            "tensorflow": {"available": False},
            "libtpu": {"available": False},
            "openvino": {"available": False},
        },
        "vendor_tools": {"sycl_ls": {"available": True}},
    }

    devices = detect_accelerators(env=env, deep=True)
    types = {device["type"] for device in devices}

    assert "npu" in types
    assert "fpga" in types
    assert "tpu" in types
    assert all(device["runtime"]["xla"] is True for device in devices)

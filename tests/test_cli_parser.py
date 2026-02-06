from __future__ import annotations

from jido.cli import build_parser


def test_scan_hardware_short_flag_parses():
    parser = build_parser()
    args = parser.parse_args(["scan", "-h"])
    assert args.command == "scan"
    assert args.hardware is True


def test_list_kernels_parses():
    parser = build_parser()
    args = parser.parse_args(["list", "kernels", "--format", "json"])
    assert args.command == "list"
    assert args.subject == "kernels"
    assert args.format == "json"

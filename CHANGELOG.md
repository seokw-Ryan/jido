# Changelog

## Unreleased

### Added
- Extended hardware scan with accelerator detection for NPU/FPGA/TPU classes.
- Built-in matmul kernel variants (`torch_matmul`, `torch_einsum`, `torch_mm_or_bmm`, `torch_compiled`).
- New CLI commands: `compare`, `recommend`, `list`, `report`, `history`, `export`, `config`, `profile`.
- Recommendation and reporting helpers with CSV/JSON export support.
- Tests for parser, kernels, accelerators, and report/recommend flows.

### Changed
- Updated README with workflow-driven usage examples.
- Updated TODOs to reflect completed CLI and benchmark milestones.

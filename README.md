# jido

jido (지도) is an early-stage ML systems toolkit focused on **hardware detection**, **backend discovery**, and **benchmark planning** for model inference. The repository is intentionally lightweight right now while the core APIs stabilize.

## Status

- Pre-alpha: structure and interfaces are being shaped.
- Not yet packaged or published.
- Many modules are placeholders; expect breaking changes.

## Installation

```bash
pip install -e .
```

### Dependencies

- `psutil` — CPU and memory detection
- `py-cpuinfo` — CPU flags and detailed info
- `rich` — formatted console output (optional, falls back to plain text)

## Usage

### `jido scan`

Scans the host machine for hardware specs and installed ML software, then outputs a formatted report to the console and saves results as JSON.

```bash
jido scan
```

#### Flags

| Flag | Description |
|------|-------------|
| `-h, --hardware` | Scan hardware only (CPU, memory, GPUs) |
| `-s, --software` | Scan software only (frameworks, runtimes, vendor tools) |
| `--out DIR` | Output directory for JSON results (default: `runs`) |
| `--json-only` | Skip console output, only save JSON files |
| `--deep` | Attempt deeper vendor tool detection using fallback methods |
| `--install-hints` | Print installation hints when dependencies are missing |

If neither `-h` nor `-s` is specified, both hardware and software are scanned.

#### Examples

```bash
# Full scan (hardware + software)
jido scan

# Hardware only
jido scan -h

# Software only
jido scan -s

# Full scan with installation hints
jido scan --install-hints

# Deep vendor tool detection
jido scan --deep

# JSON output only, custom directory
jido scan --json-only --out ./my_runs
```

#### Hardware Detection

- **CPU** — brand, physical/logical core counts, CPU feature flags
- **Memory** — total system RAM (GB)
- **GPUs** — multi-vendor detection:
  - **NVIDIA** — via NVML (`pynvml`) or `nvidia-smi` fallback. Reports name, VRAM, driver version.
  - **AMD** — via `rocm-smi` or `amd-smi` (with `--deep`). Reports card series, VRAM.
  - **Intel** — via `sycl-ls` (Intel oneAPI). Reports device name.
- **Runtime flags per GPU** — CUDA, ROCm, DirectML, OpenVINO availability
- **Machine ID** — deterministic SHA256 fingerprint from CPU, memory, architecture, and GPU names

#### Software Detection

- **ML Frameworks** — checks installed versions of `torch`, `onnxruntime`, `transformers`, `openvino`, `directml`, `torch_directml`
- **Vendor Tools** — checks PATH for `nvidia-smi`, `rocm-smi`, `amd-smi`, `sycl-ls`
- **Install Hints** — suggests missing tools when a GPU is detected but its vendor utilities are absent

#### Output

Results are always saved to `{OUT_DIR}/{MACHINE_ID}/` as:
- `hardware.json` — OS, Python, CPU, memory, GPUs, runtime flags, machine ID
- `env.json` — detected frameworks and vendor tools

Console output uses `rich` tables when available, with plain text fallback.

## Goals

- Detect host hardware and runtime capabilities.
- Discover available ML backends at runtime.
- Standardize benchmark plans and results.
- Generate recommendations and exportable configs.

## What's In The Repo

- `jido/core/hardware/` — hardware discovery (CPU, GPU, memory detection).
- `jido/core/env/` — environment and framework discovery.
- `jido/core/datastore/` — JSON result persistence.
- `jido/cli.py` — CLI entry point and argument parsing.
- `jido/backends/` — backend plugin stubs.
- `jido/sweeps/` — sweep/plan expansion stubs.
- `jido/reports/` — reporting/export stubs.
- `artifacts/` — deployment templates and example configs.

## Repository Layout (Short)

```
jido/
  jido/            # python package source
  artifacts/       # templates + example configs
  scripts/         # dev scripts (placeholders)
  tests/           # unit tests (to be filled)
```

## Roadmap

- Harden hardware detection (CPU/GPU/NPU capabilities).
- Implement environment discovery and backend registry.
- Add a minimal benchmark runner with a first backend.
- Define and persist a results schema.

## Contributing

Contributions are welcome. If you plan to add a new backend or major capability:

- Open an issue describing the scope.
- Keep modules small and composable.
- Prefer optional dependencies with graceful fallback.

## Changelog

See `CHANGELOG.md` for release notes (currently empty while pre-alpha).


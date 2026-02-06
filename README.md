# jido

jido (지도) is an ML systems toolkit for hardware/runtime discovery and kernel benchmarking.

## Installation

```bash
pip install -e .
```

## Workflow

1. Scan machine capabilities.
2. Run benchmarks.
3. Compare new runs to baselines.
4. Generate reports and recommendations.
5. Export results for external analysis.

## Commands

### `jido scan`
Detect CPU, memory, GPUs, and accelerator-class devices (NPU/TPU/FPGA), plus framework/tool availability.

```bash
jido scan
jido scan --deep --install-hints
jido scan --json-only --out runs
```

### `jido benchmark`
Run operation benchmarks (`matmul`, `attention`, `conv2d`) with configurable size, dtype, kernels, and device.

```bash
jido benchmark list
jido benchmark matmul --hardware cpu --dtype fp32
jido benchmark matmul --hardware cuda:0 --kernels reference,torch_matmul,torch_einsum
jido benchmark --output runs/bench_a.json
```

Built-in `matmul` kernels include:
- `reference`
- `torch_matmul`
- `torch_einsum`
- `torch_mm_or_bmm`
- `torch_compiled`

### `jido compare`
Compare candidate benchmark JSON against baseline.

```bash
jido compare --baseline runs/base.json --candidate runs/new.json
jido compare --baseline runs/base.json --candidate runs/new.json --only-regressions
```

### `jido recommend`
Recommend top kernels based on objective.

```bash
jido recommend --input runs/new.json --objective latency
jido recommend --input runs/new.json --objective balanced --top 10
```

### `jido report`
Summarize benchmark JSON with top fastest results and operation-level stats.

```bash
jido report --input runs/new.json
jido report --input runs/new.json --top 20 --output runs/report.json
```

### `jido history`
List stored JSON artifacts in the runs directory.

```bash
jido history
jido history --runs-dir runs --kind benchmark --limit 50
```

### `jido export`
Export results or summary data to CSV/JSON.

```bash
jido export --input runs/new.json --output runs/new.csv --format csv
jido export --input runs/new.json --output runs/summary.json --format json --section summary
```

### `jido list`
List commands, operations, kernels, or optional extras.

```bash
jido list
jido list operations
jido list kernels --format json
jido list extras
```

### `jido config`
Show defaults, write default config JSON, or print a sample command.

```bash
jido config
jido config --write .jido.defaults.json
jido config --example
```

### `jido profile`
Quick benchmark-oriented profile wrapper (defaults to `matmul`).

```bash
jido profile
jido profile attention --hardware cuda:0 --iterations 20
```

### `jido deps`
Inspect and install optional dependency groups.

```bash
jido deps
jido deps --list
jido deps --extras torch,onnx --install
```

## Outputs

- `jido scan` writes `hardware.json` and `env.json` under `runs/<machine_id>/`.
- `jido benchmark` can emit a JSON payload with run metadata and per-kernel metrics:
  - latency stats (`mean/p95/p99`)
  - FLOPS / TFLOPS
  - correctness checks
  - memory stats

## Notes

- GPU and accelerator detection uses best-effort local tooling (`nvidia-smi`, `rocm-smi`, `sycl-ls`, `lspci`, `lsusb`, etc.).
- Some benchmark kernels require optional dependencies and/or CUDA availability.

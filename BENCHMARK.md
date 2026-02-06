## Jido Benchmark MVP

Run benchmarks via:
`jido benchmark [operation] [options]`

## Commands
- `jido benchmark` run all operations
- `jido benchmark matmul` run a specific operation
- `jido benchmark attention` run attention benchmarks
- `jido benchmark conv2d` run convolution benchmarks
- `jido benchmark list` list available operations
- `jido benchmark --help` show all options

## Options and flags
- `--size` override problem size, e.g. `m=1024,n=1024,k=1024` or `batch=1,heads=8,seq=128,head_dim=64`
- `--dtype` comma-separated dtypes: `fp32`, `fp16`, `bf16`
- `--kernels` kernel names to test (`all` or comma list). `reference` runs torch reference kernel
- `--hardware` device target, e.g. `cpu`, `cuda:0`
- `--iterations` number of timed iterations (default: 50)
- `--warmup` warmup iterations (default: 5)
- `--output` JSON output file path
- `--format` output format: `table` or `json`
- `--compare-against` baseline JSON file for speedups
- `--verbose` enable extra logging (reserved for future use)

## Examples
- `jido benchmark`
- `jido benchmark matmul --size m=1024,n=1024,k=1024 --dtype fp16 --hardware cuda:0`
- `jido benchmark list`
- `jido benchmark attention --dtype fp32 --output runs/attention.json`

## Output
- Table output uses `rich` when available.
- JSON output includes `hardware`, `env`, `scan_hints`, `machine_id`, benchmark config, and results.
- Hardware and software detection uses `jido scan` internally for each run.

# jido

JIDO is an early-stage ML systems toolkit focused on **hardware detection**, **backend discovery**, and **benchmark planning** for model inference. The repository is intentionally lightweight right now while the core APIs stabilize.

## Status

- Pre-alpha: structure and interfaces are being shaped.
- Not yet packaged or published.
- Many modules are placeholders; expect breaking changes.

## Goals

- Detect host hardware and runtime capabilities.
- Discover available ML backends at runtime.
- Standardize benchmark plans and results.
- Generate recommendations and exportable configs.

## What’s In The Repo

- `jido/core/hardware/` — hardware discovery utilities (active development).
- `jido/core/env/` — environment/backends discovery stubs.
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
# JIDO

Absolutely — **JIDO (지도)** can be a “kitchen sink” ML-systems toolkit *without becoming a mess* if you design it as a **plugin-based benchmark + autotune platform**.

Below is a **full project structure** (folders/files), the **main execution flow**, and how you’ll wire in “as many frameworks as possible” cleanly.

---

## JIDO goals

**JIDO** = auto-detect hardware → discover available backends → run standardized benchmarks → store results → recommend best config → optionally generate runnable deployment configs.

Backends you can support (progressively):

- **PyTorch** (eager, `torch.compile`, Inductor, CUDA graphs)
- **Triton kernels** (custom and/or used via Inductor)
- **ONNX Runtime** (+ EPs: CPU, CUDA, TensorRT, ROCm, DirectML, OpenVINO)
- **TensorRT / TensorRT-LLM** (optional)
- **vLLM** (LLM serving + paged attention)
- **HF Transformers** (baseline)
- **Accelerate** (multi-GPU driver)
- **DeepSpeed Inference** (optional)
- **OpenVINO** (Intel)
- **llama.cpp** (GGUF path; great cross-hardware baseline)
- **FlashAttention** (where available)
- **BitsAndBytes / GPTQ / AWQ** (quantization options depending on compatibility)

Keep it modular: you don’t “hard depend” on everything—**capabilities are discovered at runtime**.

---

## Full repo structure

```
jido/
  pyproject.toml
  README.md
  LICENSE
  .gitignore
  CHANGELOG.md

  jido/                          # python package
    __init__.py
    __main__.py                  # allows: python -m jido ...
    main.py                      # main entry -> CLI
    cli.py                       # Typer/argparse commands

    config/
      defaults.yaml              # global defaults (warmup, repeats, metric policy)
      sweep_presets/
        llm.yaml                 # default LLM sweep knobs
        vision.yaml              # vision sweep knobs
        audio.yaml               # optional
      model_presets.yaml         # curated models + preferred formats
      backend_matrix.yaml        # which backends apply to which tasks

    core/
      hardware/
        detect.py                # CPU/GPU/OS detection
        nvidia.py                # NVML queries (optional)
        amd.py                   # ROCm/hip queries (optional)
        intel.py                 # OpenVINO/oneAPI hints (optional)
      env/
        discovery.py             # check which libs/backends are installed and usable
        versions.py              # capture versions (torch, cuda, driver, etc.)
      benchmark/
        runner.py                # orchestration engine
        plan.py                  # expands sweep -> concrete run plan
        protocol.py              # standard warmup/run/repeat logic
        sandbox.py               # isolates env vars / affinity / torch flags
      metrics/
        compute.py               # tokens/sec, p50/p95, mem, etc.
        profilers.py             # hooks for torch profiler / nsight markers
      datastore/
        store.py                 # sqlite or duckdb writer/reader
        schema.sql               # metrics + run metadata schema
      recommend/
        policy.py                # "maximize throughput subject to latency"
        pareto.py                # pareto frontier selection
        explain.py               # why recommendation was chosen
      logging.py                 # structured logging

    models/
      resolver.py                # parses model spec: hf:, kaggle:, local:, url:
      hf/
        download.py              # huggingface_hub integration
        export_onnx.py           # optional export pipeline
      kaggle/
        download.py              # optional v1: kaggle API integration
      local.py                   # local file paths
      formats.py                 # gguf / safetensors / onnx / engine, validators
      manifest.py                # model manifest + hashing + caching

    tasks/
      base.py                    # Task interface
      llm/
        task.py                  # LLM benchmark definition
        prompts.py               # prompt sets, synthetic + real
        tokenizer.py             # shared tokenization helpers
        correctness.py           # optional: sanity checks
      vision/
        task.py                  # vision benchmark definition (resnet/clip/etc.)
        datasets.py              # optional dataset adapters
      common/
        io_shapes.py             # standardized input shapes
        constraints.py           # constraint language p95<200, mem<VRAM, etc.

    backends/                    # plugin backends (each has: detect + run)
      base.py                    # Backend interface: detect(), run(), sweep_knobs()
      pytorch/
        backend.py               # HF Transformers baseline
        compile.py               # torch.compile modes + configs
        attention_kernels.py     # flash-attn availability, sdpa modes
      triton/
        backend.py               # custom kernels + microbench hooks
        kernels/
          matmul.py
          attention.py
      onnxrt/
        backend.py               # ORT session configs
        providers.py             # EP discovery + config
      tensorrt/
        backend.py               # build engine + run (optional)
      vllm/
        backend.py               # serve/benchmark via vLLM (optional)
      deepspeed/
        backend.py               # inference engine hooks (optional)
      llama_cpp/
        backend.py               # gguf runner (optional)
      openvino/
        backend.py               # Intel EP (optional)
      registry.py                # central plugin registry

    sweeps/
      expand.py                  # given task+backend+hardware -> run matrix
      knobs.py                   # canonical knob definitions
      filters.py                 # prune impossible configs quickly

    reports/
      summarize.py               # markdown summary
      plot.py                    # matplotlib plots
      export.py                  # export recommended config to json/yaml

    artifacts/
      templates/
        docker/
          Dockerfile.torch
          Dockerfile.ort
        k8s/
          deployment.yaml        # optional
      examples/
        configs/
          jido.example.yaml

  tests/
    test_resolver.py
    test_backend_discovery.py
    test_recommender.py
    test_plan_expansion.py

  scripts/
    dev_install.sh
    smoke_test_cpu.sh
    smoke_test_gpu.sh

  notebooks/                      # optional for exploration
    analyze_results.ipynb
```

---

## The “main file” and what it calls

**Entry point:**

- `jido/main.py` (or `jido/__main__.py`) → calls `cli.py`

**Typical command call chain (benchmark):**

1. `cli.py` parses:
   - model spec, task, constraints, sweep preset, output path
2. `core/hardware/detect.py` → produces `hardware.json`
3. `core/env/discovery.py` → produces list of usable backends
4. `models/resolver.py` → fetches model (HF/Kaggle/local/url) + validates format
5. `core/benchmark/plan.py` + `sweeps/expand.py` → expands run matrix
6. `core/benchmark/runner.py` loops runs:
   - calls `backends/registry.py` → gets backend plugin
   - `backend.run(run_config)` returns raw timings + logs
7. `core/metrics/compute.py` computes standardized metrics
8. `core/datastore/store.py` writes into SQLite/DuckDB
9. `core/recommend/policy.py` selects best config and writes `recommended.json`
10. `reports/summarize.py` writes a résumé-worthy report

---

## How many models should JIDO ship with?

Keep **defaults small but meaningful**, and allow infinite user models.

**v0 default set (LLM + Vision):**

- LLM small: `hf:Qwen/Qwen2.5-0.5B-Instruct`
- LLM medium: `hf:meta-llama/Llama-3.2-1B-Instruct` (or a similar 1–3B)
- LLM “stress”: `hf:mistralai/Mistral-7B-Instruct-v0.3` (or any 7B)
- Vision: `hf:microsoft/resnet-50` (or ONNX resnet)

This gives you: small/medium/large behavior + non-LLM.

---

## Letting users change models (HF + Kaggle + local) cleanly

### Model spec strings

- Hugging Face: `hf:<repo_id>[@revision]`
- Kaggle: `kaggle:<owner>/<dataset>:<filename>`
- Local: `local:/path/to/model.onnx` or `.safetensors` or `.gguf`
- URL: `url:https://...`

Examples:

```bash
jido benchmark --model hf:Qwen/Qwen2.5-0.5B-Instruct --task llm
jido benchmark --model local:./models/llama.gguf --backend llama_cpp
jido benchmark --model kaggle:me/llm-gguf:qwen2.5-q4.gguf --backend llama_cpp
```

### Format routing

Backend declares required formats:

- llama.cpp → **GGUF**
- ORT/TensorRT → **ONNX / engine**
- PyTorch → **safetensors / HF weights**

`models/formats.py` enforces it and can optionally trigger conversion steps (v1+).

---

## “As many frameworks as possible” without dependency hell

Use **optional extras** in `pyproject.toml`:

- `pip install jido[torch]`
- `pip install jido[onnx]`
- `pip install jido[vllm]`
- `pip install jido[tensorrt]`
- `pip install jido[full]`

`core/env/discovery.py` checks availability at runtime (import tests + small smoke checks).

---

## Minimal v0 scope (so you can actually ship)

If you want JIDO to look impressive quickly, do this first:

**v0 includes:**

- hardware detect
- backend discovery
- **PyTorch backend** (Transformers baseline + `torch.compile`)
- **ONNX Runtime backend** (CPU + CUDA if available)
- datastore (SQLite)
- recommender (Pareto + constraints)
- report generator

Then add v1/v2:

- vLLM backend
- llama.cpp backend
- TensorRT backend
- Triton custom kernels / microbench mode
- Kaggle resolver

---

## Outputs (what recruiters love)

After a run, JIDO produces:

```
runs/
  <machine_id>/
    hardware.json
    env.json
    results.sqlite
    recommended.json
    report.md
    plots/
      latency_vs_throughput.png
      pareto_frontier.png
```

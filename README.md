# jido

jido (지도) is an early-stage ML systems toolkit focused on **hardware detection**, **backend discovery**, and **benchmark planning** for model inference. The repository is intentionally lightweight right now while the core APIs stabilize.

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


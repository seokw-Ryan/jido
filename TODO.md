# TODO

Completed:
- [x] Extend hardware detection beyond CPU/GPU with accelerator probing (NPU/FPGA/TPU).
- [x] Update README with end-to-end workflow and sample commands.
- [x] Design and implement CLI-level UX with commands:
  - [x] `jido scan`
  - [x] `jido benchmark`
  - [x] `jido compare`
  - [x] `jido recommend`
  - [x] `jido list`
  - [x] `jido report`
  - [x] `jido history`
  - [x] `jido export`
  - [x] `jido config`
  - [x] `jido profile`
- [x] Finish benchmark matmul coverage with multiple kernels and cross-hardware support hooks.
- [x] Add terminal-level reporting surfaces for dtype/latency/FLOPS comparisons (`benchmark`, `report`, `compare`).

Next backlog:
- [ ] Add graph visualizations beyond tables.
- [ ] Expand workload coverage to world-model/3D/audio/DSP-focused operations.

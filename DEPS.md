# jido deps

`jido deps` helps you discover and install optional dependency groups (extras),
such as PyTorch, ONNX Runtime, or vLLM.

## Quick start

```bash
# Show missing optional extras and suggested install command
jido deps

# Install all optional extras (editable)
jido deps --install

# Install PyTorch + Transformers extras only
jido deps --extras torch --install

# List all supported extras
jido deps --list
```

## How it works

- Detects whether each optional dependency group is installed.
- Prints a pip command to install missing extras.
- Can execute the pip install directly with `--install`.

## Supported extras

| Extra | Packages | Notes |
|------|----------|-------|
| `torch` | `torch`, `transformers` | Needed for benchmarks |
| `onnx` | `onnxruntime` | ONNX Runtime |
| `vllm` | `vllm` | vLLM backend |
| `tensorrt` | `tensorrt` | NVIDIA only |
| `llama-cpp` | `llama-cpp-python` | C++ build required |
| `openvino` | `openvino` | OpenVINO backend |
| `deepspeed` | `deepspeed` | DeepSpeed backend |

## Flags

| Flag | Description |
|------|-------------|
| `--list` | List supported extras and their packages |
| `--extras torch,onnx` | Target specific extras |
| `--all` | Target all optional extras |
| `--install` | Execute `pip install` for the selected extras (defaults to all) |
| `--no-editable` | Install without `-e` (non-editable) |

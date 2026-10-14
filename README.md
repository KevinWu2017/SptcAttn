# SPTC sparse attention

This repository contains the SPTC sparse attention forward operator, its Python and Torch interfaces, and the MInference integration. Paper benchmarks, result files, plotting scripts, and legacy comparison programs are kept outside this repository.

## Source layout

| Path | Purpose |
| --- | --- |
| `components/sgl-flash-attn` | SPTC CUDA kernel, launch templates, and C++ dispatcher |
| `components/sglang/sgl-kernel` | `sgl_kernel` Torch registration, metadata conversion, and Python API |
| `integrations/minference` | MInference adapter with `backend="sptc"` |
| `third_party/cutlass` | CUTLASS/CuTe build dependency |

The source components are Git submodules. Clone with `--recurse-submodules`, or run `make submodules` after cloning.

## Build

Linux, an NVIDIA GPU with compute capability 8.x or 9.0, a matching CUDA toolkit, PyTorch, CMake, Ninja, and `scikit-build-core` are required. The installed wheel was tested on A100 (SM80). Build with an existing compatible Python environment:

```bash
make submodules
make env-check PYTHON=/path/to/python
make install-sgl-kernel PYTHON=/path/to/python SGL_KERNEL_CUDA_ARCHS=80
make install-minference PYTHON=/path/to/python
```

`SGL_KERNEL_CUDA_HOME`, `SGL_KERNEL_BUILD_JOBS`, and `CMAKE_ARGS` can be passed to `make install-sgl-kernel`. See [usage](docs/USAGE.md) for the API call and metadata contract.
Set `MINFERENCE_CUDA_ARCHS` when building the adapter for another GPU architecture (for example, `8.9` for an RTX 4090).

## Interface

```python
from sgl_kernel.sparse_flash_attn import (
    convert_vertical_slash_indexes_sparse,
    sparse_attn_func_sptc,
)

# The converter returns block_count, block_offset,
# sparse_block_count, sparse_block_offset, column_count, column_index.
output = sparse_attn_func_sptc(
    q, k, v,
    block_count, block_offset,
    sparse_block_count, sparse_block_offset,
    column_count, column_index,
    block_size_M=64, block_size_N=64, causal=True,
)
```

For MInference, set `minference_prefill_backend="sptc"` (or pass `backend="sptc"` directly to `vertical_slash_sparse_attention`). The implementation is forward only. The input metadata defines which blocks/columns are attended; a sliding-window pattern must be constructed with the desired token-level mask semantics. Known numerical limitations are listed in [usage](docs/USAGE.md).

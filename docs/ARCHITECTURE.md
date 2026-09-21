# Architecture

```text
SGLang attention backend
  -> sgl_kernel.sparse_flash_attn (Python contract)
  -> sgl_kernel common extension (Torch operator registration)
  -> flash-attn sparse C++ API and dispatcher
  -> CUDA launch template
  -> v1 baseline or v5 optimized forward kernel

vertical/slash indexes
  -> CUDA metadata conversion
  -> block and sparse-block metadata consumed by the kernel
```

The public correctness comparison is **v1 -> v5**. v1 remains only as the reference implementation; v5 is the implementation being developed. V4 must not be added to new APIs, test names, result columns, or build targets.

`components/flash-attn` and `components/sglang` are first-party fork submodules. `third_party/cutlass` is an upstream dependency submodule. `baselines/minference` is a comparison-baseline submodule, not part of the production call chain.

## Build boundary

The isolated CMake entry is `experiments/legacy/cmake_sparse_compare`. It resolves all sources relative to this repository root and writes only into `.build/cmake`. Python wheels, profile reports, and benchmark tables must be produced under ignored `results/` paths.

# Migration boundary and source map

This orchestration repository refers to source repositories from `/home/cpwu/working/SparseAttn` as Git submodules. No source tree was moved or modified.

| Old path | New path | Treatment |
| --- | --- | --- |
| `sgl-flash-attn` | `components/sgl-flash-attn` | Git submodule; first-party kernel fork |
| `sglang` | `components/sglang` | Git submodule; first-party integration fork |
| `cutlass` | `third_party/cutlass` | Git submodule; upstream dependency |
| `Kernels` | `experiments/legacy` | Git submodule; legacy demos and benchmark source |
| `MInference` | `baselines/minference` | Git submodule; optional baseline |
| `figures` | `reports/figures` | Figure-generation scripts |

Excluded from the orchestration repository: virtual environments, CMake/build directories, Python caches, wheel/dist outputs, and generated performance matrices. They are rebuildable and must not become source-of-truth artifacts.

## Invariants

1. No operation in this repository may write to its parent workspace.
2. New build output goes only to `.build/`; generated data goes only to `results/`.
3. The old workspace remains an independent rollback source until this repository passes its v1-v5 correctness and build gates.
4. Source components are managed only through the declared submodules. Do not replace a submodule with a copied source snapshot.

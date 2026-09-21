# Source provenance

Each source component is a Git submodule. The pinned commits and clone URLs live in `.gitmodules`; update them only through a reviewed top-level commit.

| Component | Submodule path | Upstream | Pinned revision |
| --- | --- | --- | --- |
| flash-attn fork | `components/flash-attn` | `KevinWu2017/sgl-flash-attn-sptc` | `bd6c87c92821a8b26367a5d027494b7703fd63c8` |
| SGLang fork | `components/sglang` | `KevinWu2017/sglang-sptc` | `3e2584aa37daa928ee2d3835dfceec108c701084` |
| CUTLASS | `third_party/cutlass` | `NVIDIA/cutlass` | `57e3cfb47a2d9e0d46eb6335c3dc411498efa198` |
| legacy Kernels | `experiments/legacy` | `KevinWu2017/SparseAttn` | `9ea6070a7976d4838c7f5de599aa2d351d1e3848` |
| MInference baseline | `baselines/minference` | `KevinWu2017/MInference` | `b41937420a582f0535534827f1bb2b2fe0e4a676` |

At conversion time, flash-attn, SGLang, legacy Kernels and MInference intentionally retain their previously uncommitted source modifications as dirty submodule worktrees. Generated profiling data and caches were deliberately excluded. Inspect each component with `git -C <path> status` and `git -C <path> diff` before deciding which changes to preserve or commit upstream.

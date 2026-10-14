# Submodules

The source forks and dependencies are pinned by `.gitmodules` and top-level gitlinks:

- `components/sgl-flash-attn`: CUDA and C++ SPTC implementation.
- `components/sglang`: `sgl_kernel` registration and Python API.
- `integrations/minference`: MInference integration.
- `third_party/cutlass`: upstream CUDA template dependency.

After changing a fork, commit and push its changes in that submodule, then update the top-level gitlink. A top-level commit alone does not store an uncommitted submodule diff. Do not reset or force-update a modified submodule while carrying local changes.

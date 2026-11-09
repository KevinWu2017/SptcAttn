# SptcAttn

SptcAttn accelerates long-context attention by mapping diagonal stripes in a sparse attention pattern to the 2:4 format used by NVIDIA Sparse Tensor Cores. This repository provides the forward operator, its PyTorch interface, and an optional MInference adapter. [SIGOPS ATC 2026 paper](https://doi.org/10.1145/3828523.3857028)

## Installation

The operator runs on Linux and supports NVIDIA GPUs with compute capability 8.x or 9.0. The A100 (SM80) is the tested GPU. The example below uses Python 3.12, CUDA Toolkit 12.8, and the CUDA 12.8 build of PyTorch 2.9.1. Install a compatible NVIDIA driver and C++ compiler, Git, Make, and Python with venv support and development headers before building. The build fetches additional dependencies from GitHub. [PyTorch wheel versions](https://pytorch.org/get-started/previous-versions/#v291)

```bash
git clone --recurse-submodules https://github.com/KevinWu2017/SptcAttn.git
cd SptcAttn

export CUDA_HOME=/usr/local/cuda-12.8  # Set this to your toolkit location.
export PATH="$CUDA_HOME/bin:$PATH"
make venv PYTHON_BOOTSTRAP=python3.12
source .venv/bin/activate
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install 'cmake>=3.31,<4' ninja 'scikit-build-core>=0.10' wheel setuptools packaging

make env-check
make install-sgl-kernel SGL_KERNEL_CUDA_ARCHS=80
```

For an existing clone, run `make submodules` to fetch the pinned revisions. On an A100, `make env-check` should report `cuda ready: True` and `sm_80`. A successful build ends with `SPTC: import OK`. Set `SGL_KERNEL_BUILD_JOBS` to limit build parallelism, or pass `CMAKE_ARGS` for additional CMake options.

### Smoke test

Run the two numerical checks from the repository root with the virtual environment active.

```bash
python tests/test_sptc_smoke.py -v
```

On a CUDA GPU, both tests should report `ok`. One compares the all-column case with PyTorch dense attention. The other checks that the sparse-pair path produces finite, nonnegative weights that sum to one for the tested query. If PyTorch cannot access CUDA, both tests report `skipped`.

## Python API

`sparse_attn_func_sptc` takes Q, K, and V on the same GPU in `(batch, sequence, heads, head_dim)` layout and the same FP16 or BF16 dtype. The head dimension is at most 128. The query and key block sizes must match and may be 32 or 64.

The following example attends to column zero and two diagonal stripes. The converter takes an int32 length per sequence and int32 index tensors shaped `(batch, heads, number_of_indices)`. Supply vertical indices in ascending order and slash offsets in descending order, as in the MInference adapter.

```python
import torch
from sgl_kernel.sparse_flash_attn import (
    convert_vertical_slash_indexes_sparse,
    sparse_attn_func_sptc,
)

length, block = 128, 64
q = torch.randn(1, length, 1, 64, device="cuda", dtype=torch.float16)
k, v = torch.randn_like(q), torch.randn_like(q)
lengths = torch.tensor([length], device="cuda", dtype=torch.int32)
vertical = torch.tensor([[[0]]], device="cuda", dtype=torch.int32)
slash = torch.tensor([[[64, 0]]], device="cuda", dtype=torch.int32)

metadata = convert_vertical_slash_indexes_sparse(
    lengths, lengths, vertical, slash, length, block, block, causal=True,
)
output = sparse_attn_func_sptc(
    q, k, v, *metadata, block_size_M=block, block_size_N=block, causal=True,
)
```

The six returned int32 tensors are `block_count`, `block_offset`, `sparse_block_count`, `sparse_block_offset`, `column_count`, and `column_index`, in that order. Use the same block sizes and causal setting for conversion and attention. The operator returns the attention output by default. Set `return_softmax_lse=True` to receive `(output, softmax_lse)`.

### MInference integration

To use SPTC through MInference, build the adapter in the same environment. The following command targets an A100.

```bash
make install-minference MINFERENCE_CUDA_ARCHS=8.0
```

Set `minference_prefill_backend="sptc"` and `minference_prefill_block_size=64` in the MInference patch configuration. Direct callers can pass `backend="sptc"` to `vertical_slash_sparse_attention`. For another supported GPU, use its architecture for both builds (for example, `SGL_KERNEL_CUDA_ARCHS=89` and `MINFERENCE_CUDA_ARCHS=8.9` for an RTX 4090).

## Architecture

| Component | Role in the forward pass |
| --- | --- |
| `integrations/minference` | Selects the vertical and slash indices and passes them to the metadata converter. |
| `components/sglang/sgl-kernel` | Converts indices to block, sparse-pair, and column metadata. The Python function `sparse_attn_func_sptc` dispatches to the registered CUDA operator `torch.ops.sgl_kernel.fwd_sparse_sptc`. |
| `components/sgl-flash-attn` | Implements `flash::mha_fwd_sparse_sptc` and dispatches to the CUDA forward kernel. The sparse-pair path uses the structured-sparsity instruction `mma.sp`. |
| `third_party/cutlass` | Supplies CUTLASS and CuTe as build dependencies. |

Run `git submodule status --recursive` to inspect the checked-out revisions. When updating a fork, push its commit before updating the submodule reference in this repository.

## License

Original code in the top-level repository is licensed under the GNU General Public License, version 2 only (`GPL-2.0-only`). Each submodule is distributed under its own license. [License text](LICENSE)

## Citation

If you use SptcAttn in your research, please cite the SIGOPS ATC 2026 paper.

```bibtex
@inproceedings{2026sptcattn,
  title     = {{SptcAttn}: Accelerating Long-Context Attention with Diagonal Stripes using Sparse Tensor Core},
  author    = {Wu, Chenpeng and Gu, Qiqi and Shi, Heng and Yao, Jianguo and Guan, Haibing},
  booktitle = {The 2026 ACM SIGOPS Annual Technical Conference},
  series    = {SIGOPS ATC 2026},
  year      = {2026},
  doi       = {10.1145/3828523.3857028}
}
```

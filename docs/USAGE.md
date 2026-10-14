# Using SPTC

Initialize the four source/dependency submodules and install a compatible PyTorch and `scikit-build-core` in your Python environment. Then build the local operator:

```bash
make submodules
make install-sgl-kernel PYTHON=/path/to/python SGL_KERNEL_CUDA_HOME=/usr/local/cuda SGL_KERNEL_CUDA_ARCHS=80
```

For the MInference adapter, install its Python package and set `minference_prefill_backend="sptc"` in the patch config (or pass `backend="sptc"` to `vertical_slash_sparse_attention` directly):

```bash
make install-minference PYTHON=/path/to/python
```

The relevant patch-config entries are `minference_prefill_backend: "sptc"` and
`minference_prefill_block_size: 64`.

The direct API is `sgl_kernel.sparse_flash_attn.sparse_attn_func_sptc`. Q, K, V use shape `(batch, sequence, heads, head_dim)` and FP16 or BF16. The metadata converter `convert_vertical_slash_indexes_sparse` accepts int32 sequence lengths and vertical/slash indices and returns six int32 tensors in this order: `block_count`, `block_offset`, `sparse_block_count`, `sparse_block_offset`, `column_count`, `column_index`. Pass those six tensors to the attention API in the same order. `block_size_M` and `block_size_N` must match and currently accept 32 or 64; head dimension is at most 128. The call can return `(output, softmax_lse)` with `return_softmax_lse=True`.

```python
from sgl_kernel.sparse_flash_attn import (
    convert_vertical_slash_indexes_sparse,
    sparse_attn_func_sptc,
)

metadata = convert_vertical_slash_indexes_sparse(
    q_seqlens, kv_seqlens, vertical_indexes, slash_indexes,
    context_size, 64, 64, causal=True,
)
output = sparse_attn_func_sptc(q, k, v, *metadata, block_size_M=64, block_size_N=64, causal=True)
```

The supplied metadata determines the sparse attention support. Converting whole sliding-window blocks does not create an exact per-token sliding-window mask. In prior validation, the 32-token block path also produced incorrect weights for some sparse pairs. Until that path is corrected and checked against a reference, prefer block size 64 and verify output numerically for each new pattern. No benchmark or paper reproduction data is included here.

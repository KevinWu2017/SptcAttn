# Architecture

```text
vertical/slash indices
  -> sgl_kernel.convert_vertical_slash_indexes_sparse
  -> block and sparse-pair metadata
  -> sgl_kernel.sparse_attn_func_sptc
  -> torch.ops.sgl_kernel.fwd_sparse_sptc
  -> flash::mha_fwd_sparse_sptc
  -> CUDA SPTC forward kernel
```

MInference's `vertical_slash_sparse_attention(..., backend="sptc")` creates the metadata and calls this Python interface. The SPTC computation uses CUDA structured sparsity instructions (`mma.sp`) in the sparse-pair path. The forked SGLang and FlashAttention repositories also retain upstream functionality required by their packages; the project-specific optimized sparse path is SPTC.

The operator is forward only. Its public inputs are Q/K/V tensors, six int32 metadata tensors, block sizes, and FlashAttention-style scale/mask options. The Python wrapper returns the output and optionally softmax LSE. `components/sgl-flash-attn` owns kernel arithmetic; `components/sglang/sgl-kernel` owns Torch registration and Python dispatch; `integrations/minference` owns the optional application adapter.

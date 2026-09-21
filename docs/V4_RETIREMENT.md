# V4 retirement plan

V4 is deliberately out of scope for the unified repository.

The current source submodules still contain v4 code only because existing sparse translation units and the C++ API co-locate v1, v4, and v5 symbols. It is not an enabled unified build target or a supported comparison path.

Removal order:

1. Add deterministic v1↔v5 metadata, output, and LSE regression tests for fp16/bf16 and supported block sizes.
2. Make the v1-v5 test executable the only correctness command and remove the v4-v5 executable from CMake.
3. Remove `sparse_attn_func_v4`, its Torch registration, API dispatcher branch, launch-template code, and v4-only kernel headers.
4. Remove v4 source references from scripts and documentation; verify that `rg 'v4|V4'` only matches migration history when appropriate.

No removal is performed in the old workspace. All deletions will occur only in this new repository after step 1 has a passing baseline.

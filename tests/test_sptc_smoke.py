"""Small numerical check for the public SPTC forward interface."""

import unittest

import torch


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class SptcSmokeTest(unittest.TestCase):
    def test_all_column_metadata_matches_dense_attention(self):
        from sgl_kernel.sparse_flash_attn import sparse_attn_func_sptc

        torch.manual_seed(0)
        length, head_dim, block_size = 128, 64, 64
        q = torch.randn(1, length, 1, head_dim, device="cuda", dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        counts = torch.zeros((1, 1, 2), device="cuda", dtype=torch.int32)
        offsets = torch.zeros((1, 1, 2, 1), device="cuda", dtype=torch.int32)
        column_counts = torch.full_like(counts, length)
        columns = torch.arange(length, device="cuda", dtype=torch.int32)
        columns = columns.view(1, 1, 1, length).expand(1, 1, 2, length).contiguous()

        actual = sparse_attn_func_sptc(
            q, k, v, counts, offsets, counts, offsets, column_counts, columns,
            block_size_M=block_size, block_size_N=block_size, causal=False,
        )
        reference = torch.nn.functional.scaled_dot_product_attention(
            q.permute(0, 2, 1, 3).float(),
            k.permute(0, 2, 1, 3).float(),
            v.permute(0, 2, 1, 3).float(),
        ).permute(0, 2, 1, 3).half()

        torch.testing.assert_close(actual, reference, atol=0.02, rtol=0.02)

    def test_sparse_pair_block64_preserves_probability_mass(self):
        from sgl_kernel.sparse_flash_attn import sparse_attn_func_sptc

        length, block_size = 128, 64
        q = torch.zeros(1, length, 1, length, device="cuda", dtype=torch.float16)
        k = torch.zeros_like(q)
        v = torch.eye(length, device="cuda", dtype=torch.float16).view(1, length, 1, length)
        counts = torch.zeros((1, 1, 2), device="cuda", dtype=torch.int32)
        offsets = torch.zeros((1, 1, 2, 1), device="cuda", dtype=torch.int32)
        sparse_counts = torch.full_like(counts, 2)
        sparse_offsets = torch.tensor([0, 64], device="cuda", dtype=torch.int32)
        sparse_offsets = sparse_offsets.view(1, 1, 1, 2).expand(1, 1, 2, 2).contiguous()

        actual = sparse_attn_func_sptc(
            q, k, v, counts, offsets, sparse_counts, sparse_offsets, counts, offsets,
            block_size_M=block_size, block_size_N=block_size, causal=False,
        )
        weights = actual[0, 64, 0].float()
        self.assertTrue(torch.isfinite(weights).all().item())
        self.assertTrue((weights >= 0).all().item())
        torch.testing.assert_close(weights.sum(), torch.tensor(1.0, device="cuda"), atol=1e-4, rtol=0)


if __name__ == "__main__":
    unittest.main()

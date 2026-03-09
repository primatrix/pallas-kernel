"""GLA kernel-level tests: Triton (FLA CUDA) vs Torch CPU reference.

Covers naive_recurrent_gla, chunk_gla, fused_recurrent_gla, fused_chunk_gla,
and the 4 chunk sub-functions.
"""
from __future__ import annotations

import unittest

import pytest
import torch
import torch.nn.functional as F

from tests.src.ops.gla import (
    naive_recurrent_gla as cpu_naive_recurrent_gla,
    chunk_gla as cpu_chunk_gla,
    fused_chunk_gla as cpu_fused_chunk_gla,
    chunk_local_cumsum as cpu_chunk_local_cumsum,
    chunk_fwd_h as cpu_chunk_fwd_h,
    chunk_gla_fwd_intra_gk as cpu_chunk_gla_fwd_intra_gk,
    chunk_gla_fwd_o_gk as cpu_chunk_gla_fwd_o_gk,
    chunk_gla_fwd as cpu_chunk_gla_fwd,
    fused_recurrent_gla as cpu_fused_recurrent_gla,
)
from tests.utils import compare_tensor

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
HAS_CUDA = torch.cuda.is_available()

triton_imports_available = False
try:
    from fla.ops.gla.naive import naive_recurrent_gla as triton_naive_recurrent_gla
    from fla.ops.gla import chunk_gla as triton_chunk_gla
    from fla.ops.gla import fused_recurrent_gla as triton_fused_recurrent_gla
    triton_imports_available = True
except ImportError:
    pass

requires_triton = pytest.mark.skipif(
    not (HAS_CUDA and triton_imports_available),
    reason="Triton / CUDA not available",
)


# ============================================================================
# TestNaiveRecurrentGLA
# ============================================================================

class TestNaiveRecurrentGLA(unittest.TestCase):
    """Triton naive_recurrent_gla vs Torch CPU."""

    @requires_triton
    def test_basic(self):
        """Basic shapes (B=2, T=32, H=4, K=32, V=64)."""
        B, T, H, K, V = 2, 32, 4, 32, 64
        q = torch.randn(B, T, H, K)
        k = torch.randn(B, T, H, K)
        v = torch.randn(B, T, H, V)
        gk = F.logsigmoid(torch.randn(B, T, H, K))

        o_triton, _ = triton_naive_recurrent_gla(q, k, v, gk)
        o_cpu, _ = cpu_naive_recurrent_gla(q, k, v, gk)
        assert compare_tensor("output", o_triton, o_cpu)

    @requires_triton
    def test_large_dims(self):
        """Larger dims (B=1, T=128, H=2, K=64, V=128)."""
        torch.manual_seed(7)
        B, T, H, K, V = 1, 128, 2, 64, 128
        q = torch.randn(B, T, H, K)
        k = torch.randn(B, T, H, K)
        v = torch.randn(B, T, H, V)
        gk = F.logsigmoid(torch.randn(B, T, H, K))

        o_triton, _ = triton_naive_recurrent_gla(q, k, v, gk)
        o_cpu, _ = cpu_naive_recurrent_gla(q, k, v, gk)
        assert compare_tensor("output", o_triton, o_cpu, atol=5e-5, rtol=5e-5)

    @requires_triton
    def test_initial_state(self):
        """With initial state + final state output."""
        torch.manual_seed(13)
        B, T, H, K, V = 2, 64, 4, 32, 64
        q = torch.randn(B, T, H, K)
        k = torch.randn(B, T, H, K)
        v = torch.randn(B, T, H, V)
        gk = F.logsigmoid(torch.randn(B, T, H, K))
        h0 = torch.randn(B, H, K, V)

        o_triton, s_triton = triton_naive_recurrent_gla(q, k, v, gk, initial_state=h0, output_final_state=True)
        o_cpu, s_cpu = cpu_naive_recurrent_gla(q, k, v, gk, initial_state=h0, output_final_state=True)
        assert compare_tensor("output", o_triton, o_cpu)
        assert compare_tensor("final_state", s_triton, s_cpu)

    @requires_triton
    def test_various_shapes(self):
        """Multiple shape combinations."""
        torch.manual_seed(99)
        for B, T, H, K, V in [(1, 16, 1, 16, 16), (4, 64, 8, 32, 64), (1, 256, 2, 128, 128)]:
            q = torch.randn(B, T, H, K)
            k = torch.randn(B, T, H, K)
            v = torch.randn(B, T, H, V)
            gk = F.logsigmoid(torch.randn(B, T, H, K))
            o_triton, _ = triton_naive_recurrent_gla(q, k, v, gk)
            o_cpu, _ = cpu_naive_recurrent_gla(q, k, v, gk)
            atol = 1e-4 if K > 64 or T > 128 else 1e-5
            assert compare_tensor(f"B={B} T={T} H={H} K={K} V={V}", o_triton, o_cpu, atol=atol, rtol=atol)

    @requires_triton
    def test_state_split(self):
        """Split sequence → process in 2 halves → state consistency."""
        torch.manual_seed(77)
        B, T, H, K, V = 1, 40, 2, 16, 32
        q = torch.randn(B, T, H, K)
        k = torch.randn(B, T, H, K)
        v = torch.randn(B, T, H, V)
        gk = F.logsigmoid(torch.randn(B, T, H, K))

        o_full_triton, s_full_triton = triton_naive_recurrent_gla(q, k, v, gk, output_final_state=True)
        o_full_cpu, s_full_cpu = cpu_naive_recurrent_gla(q, k, v, gk, output_final_state=True)

        T1 = T // 2
        _, s1_triton = triton_naive_recurrent_gla(q[:, :T1], k[:, :T1], v[:, :T1], gk[:, :T1], output_final_state=True)
        o2_triton, s2_triton = triton_naive_recurrent_gla(
            q[:, T1:], k[:, T1:], v[:, T1:], gk[:, T1:],
            initial_state=s1_triton, output_final_state=True,
        )
        _, s1_cpu = cpu_naive_recurrent_gla(q[:, :T1], k[:, :T1], v[:, :T1], gk[:, :T1], output_final_state=True)
        o2_cpu, s2_cpu = cpu_naive_recurrent_gla(
            q[:, T1:], k[:, T1:], v[:, T1:], gk[:, T1:],
            initial_state=s1_cpu, output_final_state=True,
        )

        assert compare_tensor("full output (triton vs cpu)", o_full_triton, o_full_cpu)
        assert compare_tensor("full state (triton vs cpu)", s_full_triton, s_full_cpu)
        assert compare_tensor("split-2nd output (triton vs cpu)", o2_triton, o2_cpu)
        assert compare_tensor("split-2nd state (triton vs cpu)", s2_triton, s2_cpu)
        assert compare_tensor("triton: full vs split state", s_full_triton, s2_triton)
        assert compare_tensor("cpu: full vs split state", s_full_cpu, s2_cpu)


# ============================================================================
# TestChunkGLA
# ============================================================================

class TestChunkGLA(unittest.TestCase):
    """Triton chunk_gla vs Torch CPU."""

    @requires_triton
    def test_basic(self):
        B, T, H, K, V = 2, 64, 4, 32, 64
        q = torch.randn(B, T, H, K)
        k = torch.randn(B, T, H, K)
        v = torch.randn(B, T, H, V)
        gk = F.logsigmoid(torch.randn(B, T, H, K))

        o_triton, s_triton = triton_chunk_gla(
            q.to(DEVICE), k.to(DEVICE), v.to(DEVICE), gk.to(DEVICE),
            output_final_state=True,
        )
        o_cpu, s_cpu = cpu_chunk_gla(q, k, v, gk, output_final_state=True)
        assert compare_tensor("output", o_triton.cpu(), o_cpu, atol=2e-2, rtol=2e-2)
        assert compare_tensor("final_state", s_triton.cpu(), s_cpu, atol=2e-2, rtol=2e-2)

    @requires_triton
    def test_initial_state(self):
        torch.manual_seed(13)
        B, T, H, K, V = 2, 64, 4, 32, 64
        q = torch.randn(B, T, H, K)
        k = torch.randn(B, T, H, K)
        v = torch.randn(B, T, H, V)
        gk = F.logsigmoid(torch.randn(B, T, H, K))
        h0 = torch.randn(B, H, K, V)

        o_triton, s_triton = triton_chunk_gla(
            q.to(DEVICE), k.to(DEVICE), v.to(DEVICE), gk.to(DEVICE),
            initial_state=h0.to(DEVICE), output_final_state=True,
        )
        o_cpu, s_cpu = cpu_chunk_gla(q, k, v, gk, initial_state=h0, output_final_state=True)
        assert compare_tensor("output", o_triton.cpu(), o_cpu, atol=2e-2, rtol=2e-2)
        assert compare_tensor("final_state", s_triton.cpu(), s_cpu, atol=2e-2, rtol=2e-2)

    def test_vs_naive(self):
        """chunk_gla vs naive_recurrent_gla (CPU internal)."""
        B, T, H, K, V = 2, 64, 4, 32, 64
        q = torch.randn(B, T, H, K)
        k = torch.randn(B, T, H, K)
        v = torch.randn(B, T, H, V)
        gk = F.logsigmoid(torch.randn(B, T, H, K))

        o_naive, s_naive = cpu_naive_recurrent_gla(q, k, v, gk, output_final_state=True)
        o_chunk, s_chunk = cpu_chunk_gla(q, k, v, gk, output_final_state=True)
        assert compare_tensor("output", o_naive, o_chunk, atol=5e-5, rtol=5e-5)
        assert compare_tensor("final_state", s_naive, s_chunk, atol=5e-5, rtol=5e-5)

    def test_vs_naive_init_state(self):
        """chunk vs naive with initial state."""
        torch.manual_seed(13)
        B, T, H, K, V = 2, 32, 4, 32, 64
        q = torch.randn(B, T, H, K)
        k = torch.randn(B, T, H, K)
        v = torch.randn(B, T, H, V)
        gk = F.logsigmoid(torch.randn(B, T, H, K))
        h0 = torch.randn(B, H, K, V)

        o_naive, s_naive = cpu_naive_recurrent_gla(q, k, v, gk, initial_state=h0, output_final_state=True)
        o_chunk, s_chunk = cpu_chunk_gla(q, k, v, gk, initial_state=h0, output_final_state=True)
        assert compare_tensor("output", o_naive, o_chunk, atol=5e-5, rtol=5e-5)
        assert compare_tensor("final_state", s_naive, s_chunk, atol=5e-5, rtol=5e-5)

    def test_varlen_cu_seqlens(self):
        """Variable-length sequences via cu_seqlens."""
        torch.manual_seed(7)
        H, K, V = 4, 32, 64
        T = 48
        q = torch.randn(1, T, H, K)
        k = torch.randn(1, T, H, K)
        v = torch.randn(1, T, H, V)
        gk = F.logsigmoid(torch.randn(1, T, H, K))
        cu = torch.tensor([0, 16, 32, 48], dtype=torch.long)

        o_naive, _ = cpu_naive_recurrent_gla(q, k, v, gk, cu_seqlens=cu)
        o_chunk, _ = cpu_chunk_gla(q, k, v, gk, cu_seqlens=cu)
        assert compare_tensor("output", o_naive, o_chunk, atol=5e-5, rtol=5e-5)


# ============================================================================
# TestFusedRecurrentGLA
# ============================================================================

class TestFusedRecurrentGLA(unittest.TestCase):
    """Triton fused_recurrent_gla vs Torch CPU."""

    @requires_triton
    def test_basic(self):
        B, T, H, K, V = 2, 32, 4, 32, 64
        q = torch.randn(B, T, H, K)
        k = torch.randn(B, T, H, K)
        v = torch.randn(B, T, H, V)
        gk = F.logsigmoid(torch.randn(B, T, H, K))

        o_triton, s_triton = triton_fused_recurrent_gla(
            q.to(DEVICE), k.to(DEVICE), v.to(DEVICE), gk.to(DEVICE),
            output_final_state=True,
        )
        o_cpu, s_cpu = cpu_naive_recurrent_gla(q, k, v, gk, output_final_state=True)
        assert compare_tensor("output", o_triton.cpu(), o_cpu, atol=1e-4, rtol=1e-4)
        assert compare_tensor("final_state", s_triton.cpu(), s_cpu, atol=1e-4, rtol=1e-4)

    @requires_triton
    def test_initial_state(self):
        torch.manual_seed(13)
        B, T, H, K, V = 2, 32, 4, 32, 64
        q = torch.randn(B, T, H, K)
        k = torch.randn(B, T, H, K)
        v = torch.randn(B, T, H, V)
        gk = F.logsigmoid(torch.randn(B, T, H, K))
        h0 = torch.randn(B, H, K, V)

        o_triton, s_triton = triton_fused_recurrent_gla(
            q.to(DEVICE), k.to(DEVICE), v.to(DEVICE), gk.to(DEVICE),
            initial_state=h0.to(DEVICE), output_final_state=True,
        )
        o_cpu, s_cpu = cpu_naive_recurrent_gla(q, k, v, gk, initial_state=h0, output_final_state=True)
        assert compare_tensor("output", o_triton.cpu(), o_cpu, atol=1e-4, rtol=1e-4)
        assert compare_tensor("final_state", s_triton.cpu(), s_cpu, atol=1e-4, rtol=1e-4)

    def test_vs_naive(self):
        """fused_recurrent_gla vs naive (CPU internal)."""
        B, T, H, K, V = 2, 32, 4, 16, 32
        q = torch.randn(B, T, H, K)
        k = torch.randn(B, T, H, K)
        v = torch.randn(B, T, H, V)
        gk = torch.randn(B, T, H, K) * 0.1
        h0 = torch.randn(B, H, K, V) * 0.01

        o_naive, ht_naive = cpu_naive_recurrent_gla(q, k, v, gk, initial_state=h0, output_final_state=True)
        o_fused, ht_fused = cpu_fused_recurrent_gla(q, k, v, gk=gk, initial_state=h0, output_final_state=True)
        assert compare_tensor("output", o_fused.float(), o_naive.float(), atol=1e-5, rtol=1e-5)
        assert compare_tensor("final state", ht_fused, ht_naive, atol=1e-5, rtol=1e-5)


# ============================================================================
# TestFusedChunkGLA
# ============================================================================

class TestFusedChunkGLA(unittest.TestCase):

    def test_vs_naive(self):
        """fused_chunk_gla vs naive (CPU internal)."""
        B, T, H, K, V = 2, 64, 4, 32, 64
        q = torch.randn(B, T, H, K)
        k = torch.randn(B, T, H, K)
        v = torch.randn(B, T, H, V)
        gk = F.logsigmoid(torch.randn(B, T, H, K))

        o_naive, _ = cpu_naive_recurrent_gla(q, k, v, gk)
        o_fc, _ = cpu_fused_chunk_gla(q, k, v, gk)
        assert compare_tensor("output", o_naive, o_fc, atol=5e-5, rtol=5e-5)


# ============================================================================
# TestChunkSubFunctions
# ============================================================================

class TestChunkSubFunctions(unittest.TestCase):
    """Tests for the 4 chunk sub-functions and orchestrator."""

    def test_chunk_local_cumsum(self):
        """chunk_local_cumsum vs manual reshape + cumsum."""
        B, H, K, C = 2, 4, 8, 4
        NT = 3
        T = NT * C

        g = torch.randn(B, T, H, K)
        result = cpu_chunk_local_cumsum(g, chunk_size=C)

        g_chunks = g.view(B, NT, C, H, K)
        expected = g_chunks.cumsum(dim=2).view(B, T, H, K)
        assert compare_tensor("cumsum", result, expected, atol=1e-6, rtol=1e-6)

        # Verify chunk boundaries: cumsum resets at each chunk start
        for n in range(1, NT):
            first_in_chunk = result[:, n * C]
            raw_g = g[:, n * C]
            assert compare_tensor(f"chunk {n} reset", first_in_chunk, raw_g, atol=1e-7, rtol=1e-7)

    def test_chunk_fwd_h(self):
        """Inter-chunk hidden state: final state matches naive_recurrent_gla."""
        B, T, H, K, V = 2, 32, 4, 16, 32
        C = 16

        q = torch.randn(B, T, H, K)
        k = torch.randn(B, T, H, K)
        v = torch.randn(B, T, H, V)
        gk = torch.randn(B, T, H, K) * 0.1

        _, naive_ht = cpu_naive_recurrent_gla(q, k, v, gk, output_final_state=True)

        gk_f = gk.float()
        g_cumsum = cpu_chunk_local_cumsum(gk_f, chunk_size=C)
        h_all, chunk_ht = cpu_chunk_fwd_h(
            k.float(), v.float(), g_cumsum,
            h0=None, output_final_state=True, chunk_size=C,
        )

        assert compare_tensor("final state", chunk_ht, naive_ht.float(), atol=1e-4, rtol=1e-4)
        NT = T // C
        assert h_all.shape == (B, NT, H, K, V), f"h_all shape: {h_all.shape}"

    def test_chunk_gla_fwd_intra_gk(self):
        """Intra-chunk attention matrix vs manual q*exp(g) @ (k*exp(-g))^T."""
        B, H, K, C = 1, 2, 8, 4
        NT = 2
        T = NT * C
        scale = K ** -0.5

        q = torch.randn(B, T, H, K)
        k = torch.randn(B, T, H, K)
        g = torch.randn(B, T, H, K) * 0.1

        g_cumsum = cpu_chunk_local_cumsum(g.float(), chunk_size=C)
        A = cpu_chunk_gla_fwd_intra_gk(q.float(), k.float(), g_cumsum, scale, chunk_size=C)

        # Manual computation per chunk
        q_c = q.float().view(B, NT, C, H, K)
        k_c = k.float().view(B, NT, C, H, K)
        gc = g_cumsum.view(B, NT, C, H, K)

        q_gated = q_c * gc.exp()
        k_gated = k_c * (-gc).exp()
        A_manual = torch.einsum('bnihk,bnjhk->bnhij', q_gated, k_gated)
        causal_mask = torch.tril(torch.ones(C, C, dtype=torch.bool))
        A_manual = A_manual.masked_fill(~causal_mask, 0.0)

        assert compare_tensor("A matrix", A, A_manual, atol=1e-5, rtol=1e-5)
        # Check causality: upper triangle should be zero
        upper = A[:, :, :, 0, -1]  # q_pos=0, k_pos=last → should be 0
        assert (upper.abs() < 1e-10).all().item(), "upper triangle not zero"

    def test_chunk_gla_fwd_o_gk(self):
        """Output given pre-computed A and h vs full chunk_gla."""
        B, T, H, K, V = 1, 16, 2, 8, 16
        C = 8
        scale = K ** -0.5

        q = torch.randn(B, T, H, K).float()
        k = torch.randn(B, T, H, K).float()
        v = torch.randn(B, T, H, V).float()
        g = torch.randn(B, T, H, K).float() * 0.1

        g_cumsum = cpu_chunk_local_cumsum(g, chunk_size=C)
        h_all, _ = cpu_chunk_fwd_h(k, v, g_cumsum, chunk_size=C)
        A = cpu_chunk_gla_fwd_intra_gk(q, k, g_cumsum, scale, chunk_size=C)
        o = cpu_chunk_gla_fwd_o_gk(q, v, g_cumsum, A, h_all, scale, chunk_size=C)

        o_ref, _ = cpu_chunk_gla(
            q.to(torch.float32), k.to(torch.float32),
            v.to(torch.float32), g.to(torch.float32),
            scale=scale, chunk_size=C,
        )

        assert compare_tensor("output", o, o_ref.float(), atol=1e-5, rtol=1e-5)

    def test_chunk_gla_fwd_orchestrator(self):
        """Orchestrator chunk_gla_fwd vs chunk_gla results."""
        B, T, H, K, V = 2, 50, 4, 16, 32
        C = 16
        scale = K ** -0.5

        q = torch.randn(B, T, H, K)
        k = torch.randn(B, T, H, K)
        v = torch.randn(B, T, H, V)
        gk = torch.randn(B, T, H, K) * 0.1
        h0 = torch.randn(B, H, K, V) * 0.01

        o_ref, ht_ref = cpu_chunk_gla(
            q, k, v, gk,
            scale=scale, initial_state=h0,
            output_final_state=True, chunk_size=C,
        )

        _, _, _, ht_fwd, o_fwd = cpu_chunk_gla_fwd(
            q.float(), k.float(), v.float(), gk.float(),
            g_cumsum=None, scale=scale,
            initial_state=h0,
            output_final_state=True, chunk_size=C,
        )

        assert compare_tensor("output", o_fwd, o_ref.float(), atol=1e-5, rtol=1e-5)
        assert compare_tensor("final state", ht_fwd, ht_ref, atol=1e-5, rtol=1e-5)

    def test_sub_functions_compose(self):
        """Manual composition of 4 sub-functions == chunk_gla."""
        B, T, H, K, V = 2, 48, 4, 16, 32
        C = 16
        scale = K ** -0.5

        q = torch.randn(B, T, H, K).float()
        k = torch.randn(B, T, H, K).float()
        v = torch.randn(B, T, H, V).float()
        g = torch.randn(B, T, H, K).float() * 0.1
        h0 = torch.randn(B, H, K, V).float() * 0.01

        o_ref, ht_ref = cpu_chunk_gla(q, k, v, g, scale=scale,
                                       initial_state=h0, output_final_state=True,
                                       chunk_size=C)

        # Manual composition
        g_cumsum = cpu_chunk_local_cumsum(g, C)
        h_all, ht = cpu_chunk_fwd_h(k, v, g_cumsum, h0=h0,
                                     output_final_state=True, chunk_size=C)
        A = cpu_chunk_gla_fwd_intra_gk(q, k, g_cumsum, scale, chunk_size=C)
        o = cpu_chunk_gla_fwd_o_gk(q, v, g_cumsum, A, h_all, scale, chunk_size=C)

        assert compare_tensor("composed output", o, o_ref.float(), atol=1e-5, rtol=1e-5)
        assert compare_tensor("composed final state", ht, ht_ref, atol=1e-5, rtol=1e-5)


# ============================================================================
# TestCuSeqlens
# ============================================================================

class TestCuSeqlens(unittest.TestCase):
    """cu_seqlens packed == separate batch processing."""

    def test_packed_vs_separate(self):
        torch.manual_seed(123)
        H, K, V = 2, 16, 32
        s1_len, s2_len = 10, 14
        q1 = torch.randn(1, s1_len, H, K)
        k1 = torch.randn(1, s1_len, H, K)
        v1 = torch.randn(1, s1_len, H, V)
        g1 = F.logsigmoid(torch.randn(1, s1_len, H, K))
        q2 = torch.randn(1, s2_len, H, K)
        k2 = torch.randn(1, s2_len, H, K)
        v2 = torch.randn(1, s2_len, H, V)
        g2 = F.logsigmoid(torch.randn(1, s2_len, H, K))

        o1, s1 = cpu_naive_recurrent_gla(q1, k1, v1, g1, output_final_state=True)
        o2, s2 = cpu_naive_recurrent_gla(q2, k2, v2, g2, output_final_state=True)

        q_cat = torch.cat([q1, q2], dim=1)
        k_cat = torch.cat([k1, k2], dim=1)
        v_cat = torch.cat([v1, v2], dim=1)
        g_cat = torch.cat([g1, g2], dim=1)
        cu = torch.tensor([0, s1_len, s1_len + s2_len], dtype=torch.long)
        o_cu, s_cu = cpu_naive_recurrent_gla(q_cat, k_cat, v_cat, g_cat, output_final_state=True, cu_seqlens=cu)

        assert compare_tensor("seg1 output", o1, o_cu[:, :s1_len])
        assert compare_tensor("seg2 output", o2, o_cu[:, s1_len:])
        assert compare_tensor("seg1 state", s1.squeeze(0), s_cu[0])
        assert compare_tensor("seg2 state", s2.squeeze(0), s_cu[1])

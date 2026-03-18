"""Simple GLA chunk kernel tests: Pallas kernels vs JAX reference implementations."""

from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from tops.ops.simple_gla import (
    chunk_simple_gla,
    chunk_simple_gla_fwd,
    chunk_simple_gla_bwd,
    chunk_local_cumsum_scalar,
    chunk_local_cumsum_scalar_reverse,
    simple_gla_fwd_o,
    simple_gla_fwd_o_ref,
    simple_gla_bwd_ref,
    simple_gla_bwd_dv,
    simple_gla_bwd_dqkwg,
)
from tops.ops.common.chunk_h import chunk_fwd_h_ref, chunk_bwd_dh_ref
from tests.utils import compare_tensor


# ---------------------------------------------------------------------------
# Test fixtures and helpers
# ---------------------------------------------------------------------------

CHUNK_SIZE = 16
FWD_ATOL = 5e-2
FWD_RTOL = 5e-2
BWD_ATOL = 1e-1
BWD_RTOL = 1e-1


def _make_inputs(B, T, H, K, V, seed=42, with_g=True, with_h0=False):
    """Create random inputs for Simple GLA tests."""
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 7)
    q = jax.random.normal(keys[0], (B, T, H, K), dtype=jnp.float32) * 0.1
    k = jax.random.normal(keys[1], (B, T, H, K), dtype=jnp.float32) * 0.1
    v = jax.random.normal(keys[2], (B, T, H, V), dtype=jnp.float32) * 0.1
    g = None
    if with_g:
        g_raw = jax.random.normal(keys[3], (B, T, H), dtype=jnp.float32)
        g = jax.nn.log_sigmoid(g_raw)
    h0 = None
    if with_h0:
        h0 = jax.random.normal(keys[4], (B, H, K, V), dtype=jnp.float32) * 0.01
    return q, k, v, g, h0


def _make_do(B, T, H, V, seed=99):
    key = jax.random.PRNGKey(seed)
    return jax.random.normal(key, (B, T, H, V), dtype=jnp.float32) * 0.1


# ---------------------------------------------------------------------------
# Forward tests
# ---------------------------------------------------------------------------


class TestChunkLocalCumsum:
    """Test chunk-local cumsum for scalar gates."""

    def test_basic(self):
        B, T, H, C = 2, 32, 4, 16
        key = jax.random.PRNGKey(0)
        g = jax.random.normal(key, (B, T, H), dtype=jnp.float32)

        g_cumsum = chunk_local_cumsum_scalar(g, C)
        assert g_cumsum.shape == (B, T, H)

        # Verify against manual cumsum
        g_c = g.reshape(B * (T // C), C, H)
        expected = jnp.cumsum(g_c, axis=1).reshape(B, T, H)
        assert compare_tensor("cumsum", expected, g_cumsum)

    def test_reverse(self):
        B, T, H, C = 2, 32, 4, 16
        key = jax.random.PRNGKey(1)
        dg = jax.random.normal(key, (B, T, H), dtype=jnp.float32)

        dg_rev = chunk_local_cumsum_scalar_reverse(dg, C)
        assert dg_rev.shape == (B, T, H)

        # Verify: reverse cumsum = flip(cumsum(flip(x)))
        dg_c = dg.reshape(B * (T // C), C, H)
        expected = jnp.cumsum(dg_c[:, ::-1, :], axis=1)[:, ::-1, :].reshape(B, T, H)
        assert compare_tensor("reverse_cumsum", expected, dg_rev)


class TestSimpleGLAFwdO:
    """Test fused forward output kernel vs reference."""

    @pytest.mark.parametrize("B,T,H,K,V", [
        (1, 32, 2, 128, 128),
        (2, 64, 4, 128, 128),
        (1, 16, 1, 128, 256),
    ])
    def test_fwd_o_vs_ref(self, B, T, H, K, V):
        C = CHUNK_SIZE
        q, k, v, g, _ = _make_inputs(B, T, H, K, V, with_g=True)
        scale = K**-0.5
        g_cumsum = chunk_local_cumsum_scalar(g, C)

        # Broadcast g_cumsum for h computation
        gk = jnp.broadcast_to(g_cumsum[..., None], (B, T, H, K))
        h, _ = chunk_fwd_h_ref(k, v, gk=gk, chunk_size=C)

        # Reference
        o_ref = simple_gla_fwd_o_ref(q, k, v, g_cumsum, h, scale, C)

        # Pallas kernel
        o_pl = simple_gla_fwd_o(q, k, v, g_cumsum, h, scale, C)

        assert compare_tensor("fwd_o", o_ref, o_pl, atol=FWD_ATOL, rtol=FWD_RTOL)


class TestChunkSimpleGLAFwd:
    """Test full forward orchestrator."""

    @pytest.mark.parametrize("B,T,H,K,V", [
        (1, 32, 2, 128, 128),
        (2, 64, 4, 128, 128),
    ])
    def test_fwd_with_g(self, B, T, H, K, V):
        q, k, v, g, _ = _make_inputs(B, T, H, K, V, with_g=True)
        scale = K**-0.5

        # Full orchestrator
        _, o, _ = chunk_simple_gla_fwd(
            q, k, v, g=g, g_gamma=None,
            scale=scale, initial_state=None,
            output_final_state=False, chunk_size=CHUNK_SIZE,
        )

        # Reference path
        g_cumsum = chunk_local_cumsum_scalar(g, CHUNK_SIZE)
        gk = jnp.broadcast_to(g_cumsum[..., None], q.shape)
        h, _ = chunk_fwd_h_ref(k, v, gk=gk, chunk_size=CHUNK_SIZE)
        o_ref = simple_gla_fwd_o_ref(q, k, v, g_cumsum, h, scale, CHUNK_SIZE)

        assert compare_tensor("fwd_full", o_ref, o, atol=FWD_ATOL, rtol=FWD_RTOL)

    @pytest.mark.parametrize("B,T,H,K,V", [
        (1, 32, 2, 128, 128),
    ])
    def test_fwd_with_g_gamma(self, B, T, H, K, V):
        q, k, v, _, _ = _make_inputs(B, T, H, K, V, with_g=False)
        key = jax.random.PRNGKey(7)
        g_gamma = jax.nn.log_sigmoid(jax.random.normal(key, (H,), dtype=jnp.float32))
        scale = K**-0.5

        _, o, _ = chunk_simple_gla_fwd(
            q, k, v, g=None, g_gamma=g_gamma,
            scale=scale, initial_state=None,
            output_final_state=False, chunk_size=CHUNK_SIZE,
        )
        assert o.shape == (B, T, H, V)

    @pytest.mark.parametrize("B,T,H,K,V", [
        (2, 32, 2, 128, 128),
    ])
    def test_fwd_with_initial_state(self, B, T, H, K, V):
        q, k, v, g, h0 = _make_inputs(B, T, H, K, V, with_g=True, with_h0=True)
        scale = K**-0.5

        _, o, ht = chunk_simple_gla_fwd(
            q, k, v, g=g, g_gamma=None,
            scale=scale, initial_state=h0,
            output_final_state=True, chunk_size=CHUNK_SIZE,
        )
        assert o.shape == (B, T, H, V)
        assert ht is not None
        assert ht.shape == (B, H, K, V)


class TestSimpleGLAPublicAPI:
    """Test chunk_simple_gla public API."""

    @pytest.mark.parametrize("B,T,H,K,V", [
        (1, 32, 2, 128, 128),
        (2, 64, 4, 128, 128),
    ])
    def test_basic(self, B, T, H, K, V):
        q, k, v, g, _ = _make_inputs(B, T, H, K, V, with_g=True)
        o, final_state = chunk_simple_gla(q, k, v, g=g, chunk_size=CHUNK_SIZE)
        assert o.shape == (B, T, H, V)
        assert final_state is None

    def test_no_gate(self):
        B, T, H, K, V = 1, 32, 2, 128, 128
        q, k, v, _, _ = _make_inputs(B, T, H, K, V, with_g=False)
        o, _ = chunk_simple_gla(q, k, v, g=None, chunk_size=CHUNK_SIZE)
        assert o.shape == (B, T, H, V)

    def test_output_final_state(self):
        B, T, H, K, V = 2, 32, 2, 128, 128
        q, k, v, g, _ = _make_inputs(B, T, H, K, V, with_g=True)
        o, ht = chunk_simple_gla(
            q, k, v, g=g, output_final_state=True, chunk_size=CHUNK_SIZE,
        )
        assert o.shape == (B, T, H, V)
        assert ht is not None
        assert ht.shape == (B, H, K, V)


# ---------------------------------------------------------------------------
# Backward tests
# ---------------------------------------------------------------------------


class TestSimpleGLABwdKernels:
    """Test individual backward kernels vs reference."""

    @pytest.mark.parametrize("B,T,H,K,V", [
        (1, 32, 2, 128, 128),
        (2, 64, 4, 128, 128),
    ])
    def test_bwd_dv(self, B, T, H, K, V):
        C = CHUNK_SIZE
        q, k, v, g, _ = _make_inputs(B, T, H, K, V, with_g=True)
        do = _make_do(B, T, H, V)
        scale = K**-0.5

        g_cumsum = chunk_local_cumsum_scalar(g, C)
        gk = jnp.broadcast_to(g_cumsum[..., None], q.shape)
        h, _ = chunk_fwd_h_ref(k, v, gk=gk, chunk_size=C)
        dh, _ = chunk_bwd_dh_ref(q, k, v, gk, do, scale=scale, chunk_size=C)

        # Reference
        _, _, dv_ref, _ = simple_gla_bwd_ref(
            q, k, v, g_cumsum, h, do, dh, scale, C,
        )

        # Pallas kernel
        dv_pl = simple_gla_bwd_dv(q, k, g_cumsum, do, dh, scale, C)

        assert compare_tensor("bwd_dv", dv_ref, dv_pl, atol=BWD_ATOL, rtol=BWD_RTOL)

    @pytest.mark.parametrize("B,T,H,K,V", [
        (1, 32, 2, 128, 128),
        (2, 64, 4, 128, 128),
    ])
    def test_bwd_dqkwg(self, B, T, H, K, V):
        C = CHUNK_SIZE
        q, k, v, g, _ = _make_inputs(B, T, H, K, V, with_g=True)
        do = _make_do(B, T, H, V)
        scale = K**-0.5

        g_cumsum = chunk_local_cumsum_scalar(g, C)
        gk = jnp.broadcast_to(g_cumsum[..., None], q.shape)
        h, _ = chunk_fwd_h_ref(k, v, gk=gk, chunk_size=C)
        dh, _ = chunk_bwd_dh_ref(q, k, v, gk, do, scale=scale, chunk_size=C)

        # Reference
        dq_ref, dk_ref, _, dg_ref = simple_gla_bwd_ref(
            q, k, v, g_cumsum, h, do, dh, scale, C,
        )

        # Pallas kernel
        dq_pl, dk_pl, dg_pl = simple_gla_bwd_dqkwg(
            q, k, v, g_cumsum, h, do, dh, scale, C,
        )

        assert compare_tensor("bwd_dq", dq_ref, dq_pl, atol=BWD_ATOL, rtol=BWD_RTOL)
        assert compare_tensor("bwd_dk", dk_ref, dk_pl, atol=BWD_ATOL, rtol=BWD_RTOL)
        assert compare_tensor("bwd_dg", dg_ref, dg_pl, atol=BWD_ATOL, rtol=BWD_RTOL)


class TestChunkSimpleGLABwd:
    """Test full backward orchestrator."""

    @pytest.mark.parametrize("B,T,H,K,V", [
        (1, 32, 2, 128, 128),
        (2, 64, 4, 128, 128),
    ])
    def test_bwd_full(self, B, T, H, K, V):
        C = CHUNK_SIZE
        q, k, v, g, _ = _make_inputs(B, T, H, K, V, with_g=True)
        do = _make_do(B, T, H, V)
        scale = K**-0.5

        # Forward to get g_cumsum
        g_cumsum, o, _ = chunk_simple_gla_fwd(
            q, k, v, g=g, g_gamma=None,
            scale=scale, initial_state=None,
            output_final_state=False, chunk_size=C,
        )

        # Full backward
        dq, dk, dv, dg_cumsum, dh0 = chunk_simple_gla_bwd(
            q, k, v, g_cumsum[:, :q.shape[1]], scale=scale,
            initial_state=None, do=do, dht=None, chunk_size=C,
        )

        assert dq.shape == q.shape
        assert dk.shape == k.shape
        assert dv.shape == v.shape
        assert dg_cumsum.shape == (B, q.shape[1], H)
        assert dh0 is None

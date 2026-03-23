"""simple_gla backward: Triton vs JAX kernel comparison.

Part 1: Triton fused_recurrent vs JAX naive autodiff (existing)
Part 2: Triton chunk vs JAX chunk backward (fp32 & bf16)
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
os.environ["TRITON_F32_DEFAULT"] = "ieee"
import pytest
import torch
import torch.nn.functional as F
import jax
import jax.numpy as jnp
import numpy as np

from tops.ops.simple_gla import simple_gla_naive
from tops.ops.simple_gla.chunk import chunk_simple_gla_bwd
from tests.utils import compare_tensor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HAS_CUDA = torch.cuda.is_available()

triton_imports_available = False
try:
    from fla.ops.simple_gla import fused_recurrent_simple_gla as triton_fused_recurrent
    from fla.ops.simple_gla.chunk import chunk_simple_gla_bwd as triton_chunk_bwd
    from fla.ops.utils import chunk_local_cumsum

    triton_imports_available = True
except ImportError:
    pass

requires_triton = pytest.mark.skipif(
    not (HAS_CUDA and triton_imports_available),
    reason="Triton / CUDA not available",
)

# ============================================================================
# Part 1: Triton fused_recurrent vs JAX naive autodiff
# ============================================================================

BWD_CASES = [
    # ── standard shapes ──
    dict(B=2, T=32, H=4, K=32, V=64, seed=42),
    dict(B=2, T=32, H=4, K=32, V=64, seed=13, h0=True),
    # ── single head ──
    dict(B=2, T=32, H=1, K=32, V=64, seed=10),
    # ── K != V ──
    dict(B=2, T=32, H=4, K=16, V=128, seed=20),
    dict(B=2, T=32, H=4, K=128, V=16, seed=21),
    # ── very short T ──
    dict(B=1, T=1, H=2, K=32, V=64, seed=30),
    dict(B=1, T=3, H=2, K=32, V=64, seed=31),
    # ── small dims ──
    dict(B=2, T=32, H=2, K=8, V=16, seed=70),
    dict(B=2, T=32, H=2, K=8, V=16, seed=71, h0=True),
    # ── no gate ──
    dict(B=2, T=32, H=4, K=32, V=64, seed=102, gate="none"),
    # ── g_gamma only ──
    dict(B=2, T=32, H=4, K=32, V=64, seed=150, gate="g_gamma"),
    dict(B=2, T=32, H=4, K=32, V=64, seed=151, gate="g_gamma", h0=True),
    # ── g + g_gamma ──
    dict(B=2, T=32, H=4, K=32, V=64, seed=160, gate="g+g_gamma"),
    dict(B=2, T=32, H=4, K=32, V=64, seed=161, gate="g+g_gamma", h0=True),
    # ── odd T ──
    dict(B=2, T=37, H=4, K=16, V=32, seed=40),
    # ── custom scale ──
    dict(B=2, T=32, H=4, K=32, V=64, seed=200, scale=0.1),
    dict(B=2, T=32, H=4, K=32, V=64, seed=140, scale=0.1, h0=True),
    # ── gate_logit_normalizer ──
    dict(B=2, T=32, H=4, K=32, V=64, seed=210, gate_logit_normalizer=0.1),
    dict(B=2, T=32, H=4, K=32, V=64, seed=211, gate_logit_normalizer=10),
    # ── medium T ──
    dict(B=1, T=64, H=2, K=32, V=64, seed=7),
    dict(B=1, T=64, H=2, K=32, V=64, seed=8, gate="g_gamma"),
    dict(B=1, T=64, H=2, K=32, V=64, seed=9, gate="g+g_gamma", h0=True),
]


def _case_id(c):
    parts = [f"B{c['B']}_T{c['T']}_H{c['H']}_K{c['K']}_V{c['V']}"]
    gate = c.get("gate", "g")
    if gate != "g":
        parts.append(f"gate={gate}")
    if c.get("h0"):
        parts.append("h0")
    if c.get("dht"):
        parts.append("dht")
    if c.get("scale") is not None:
        parts.append(f"scale={c['scale']}")
    gln = c.get("gate_logit_normalizer", 1)
    if gln != 1:
        parts.append(f"gln={gln}")
    return "-".join(parts)


# ── Part 1 Helpers ──


def _torch_to_jax(t: torch.Tensor) -> jnp.ndarray:
    """Convert torch tensor to JAX array on CPU (float32)."""
    cpu_device = jax.devices("cpu")[0]
    return jax.device_put(jnp.array(t.detach().cpu().float().numpy()), cpu_device)


def _run_triton_bwd(q, k, v, do, *, g=None, g_gamma=None, h0=None, scale=None):
    """Run Triton forward + backward, return (dq, dk, dv)."""
    q_t = q.clone().to(DEVICE).requires_grad_(True)
    k_t = k.clone().to(DEVICE).requires_grad_(True)
    v_t = v.clone().to(DEVICE).requires_grad_(True)

    kwargs = dict(output_final_state=True)
    if g is not None:
        kwargs["g"] = g.to(DEVICE)
    if g_gamma is not None:
        kwargs["g_gamma"] = g_gamma.to(DEVICE)
    if h0 is not None:
        kwargs["initial_state"] = h0.to(DEVICE)
    if scale is not None:
        kwargs["scale"] = scale

    o, _ = triton_fused_recurrent(q_t, k_t, v_t, **kwargs)
    o.backward(do.to(DEVICE))

    return q_t.grad.cpu(), k_t.grad.cpu(), v_t.grad.cpu()


def _run_naive_bwd(q, k, v, do, *, g=None, g_gamma=None, h0=None, scale=None):
    """Run JAX naive forward + vjp backward, return (dq, dk, dv).

    Note: FLA Triton uses g of shape [B, T, H] (scalar per head),
    while our naive uses g of shape [B, T, H, K] (per element).
    We broadcast g to [B, T, H, K] for the naive implementation.
    g_gamma [H] is also broadcast to [B, T, H, K].
    """
    q_jax = _torch_to_jax(q)
    k_jax = _torch_to_jax(k)
    v_jax = _torch_to_jax(v)
    do_jax = _torch_to_jax(do)
    B, T, H, K = q.shape

    g_jax = None
    if g is not None:
        g_expanded = g.unsqueeze(-1).expand(B, T, H, K)
        g_jax = _torch_to_jax(g_expanded)

    g_gamma_jax = None
    if g_gamma is not None:
        g_gamma_1d = _torch_to_jax(g_gamma)  # [H]
        g_gamma_jax = jnp.broadcast_to(
            g_gamma_1d.reshape(1, 1, H, 1), (B, T, H, K)
        )

    h0_jax = _torch_to_jax(h0) if h0 is not None else None

    def fwd_fn(q, k, v):
        o, _ = simple_gla_naive(
            q, k, v,
            g=g_jax, g_gamma=g_gamma_jax,
            scale=scale, initial_state=h0_jax,
        )
        return o

    _, vjp_fn = jax.vjp(fwd_fn, q_jax, k_jax, v_jax)
    dq, dk, dv = vjp_fn(do_jax)

    return dq, dk, dv


# ── Part 1 Test ──


@requires_triton
@pytest.mark.parametrize("cfg", BWD_CASES, ids=[_case_id(c) for c in BWD_CASES])
def test_triton_vs_naive_bwd(cfg):
    B, T, H, K, V = cfg["B"], cfg["T"], cfg["H"], cfg["K"], cfg["V"]
    atol = cfg.get("atol", 1e-4)
    rtol = cfg.get("rtol", 1e-4)
    gate = cfg.get("gate", "g")
    scale = cfg.get("scale", None)
    gln = cfg.get("gate_logit_normalizer", 1)

    torch.manual_seed(cfg["seed"])
    q = torch.randn(B, T, H, K)
    k = torch.randn(B, T, H, K)
    v = torch.randn(B, T, H, V)
    do = torch.randn(B, T, H, V)

    g = None
    g_gamma = None
    if gate in ("g", "g+g_gamma"):
        g = F.logsigmoid(torch.randn(B, T, H)) / gln
    if gate in ("g_gamma", "g+g_gamma"):
        g_gamma = F.logsigmoid(torch.randn(H))

    h0 = torch.randn(B, H, K, V) if cfg.get("h0") else None

    dq_tri, dk_tri, dv_tri = _run_triton_bwd(
        q, k, v, do, g=g, g_gamma=g_gamma, h0=h0, scale=scale
    )
    dq_naive, dk_naive, dv_naive = _run_naive_bwd(
        q, k, v, do, g=g, g_gamma=g_gamma, h0=h0, scale=scale
    )

    assert compare_tensor("dq", dq_tri, dq_naive, atol=atol, rtol=rtol)
    assert compare_tensor("dk", dk_tri, dk_naive, atol=atol, rtol=rtol)
    assert compare_tensor("dv", dv_tri, dv_naive, atol=atol, rtol=rtol)


# ============================================================================
# Part 2: Triton chunk backward vs JAX chunk backward
# ============================================================================

# ── Chunk helpers ──


def _next_power_of_2(n):
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _chunk_local_cumsum_jax(g, chunk_size):
    """Chunk-local cumulative sum along time axis (JAX, float32)."""
    B, T, H = g.shape
    C = chunk_size
    g_c = g.reshape(B, T // C, C, H)
    return jnp.cumsum(g_c, axis=2).reshape(B, T, H)


def _reverse_chunk_local_cumsum_jax(dg, chunk_size):
    """Reverse chunk-local cumulative sum (JAX, float32)."""
    B, T, H = dg.shape
    C = chunk_size
    dg_c = dg.reshape(B, T // C, C, H)
    return jnp.flip(jnp.cumsum(jnp.flip(dg_c, axis=2), axis=2), axis=2).reshape(B, T, H)


def _run_triton_chunk_bwd(q, k, v, do, *, g=None, g_gamma=None, h0=None,
                           dht=None, scale, chunk_size):
    """Direct call to FLA chunk_simple_gla_bwd on CUDA."""
    # Always compute in float32 to match JAX's Precision.HIGHEST behavior.
    # Triton internally casts h (fp32) to input dtype for dot products,
    # which causes O(1) errors when inputs are bf16.
    q_t, k_t, v_t, do_t = (t.float().to(DEVICE) for t in (q, k, v, do))

    # g: always cumsum in float32 for precision
    g_cs = None
    if g is not None:
        g_cs = chunk_local_cumsum(g.float().to(DEVICE), chunk_size=chunk_size)

    g_gam = g_gamma.float().to(DEVICE) if g_gamma is not None else None
    h0_t = h0.float().to(DEVICE) if h0 is not None else None
    dht_t = dht.float().to(DEVICE) if dht is not None else None

    dq, dk, dv, dg_raw, dh0 = triton_chunk_bwd(
        q=q_t, k=k_t, v=v_t,
        g=g_cs, g_gamma=g_gam,
        initial_state=h0_t,
        do=do_t, dht=dht_t, scale=scale,
        chunk_size=chunk_size,
    )

    dg = None
    if dg_raw is not None:
        dg = chunk_local_cumsum(dg_raw, chunk_size=chunk_size, reverse=True)

    return dict(
        dq=dq.cpu().float(),
        dk=dk.cpu().float(),
        dv=dv.cpu().float(),
        dg=dg.cpu().float() if dg is not None else None,
        dh0=dh0.cpu().float() if dh0 is not None else None,
    )


def _run_jax_chunk_bwd(q, k, v, do, *, g=None, g_gamma=None, h0=None,
                        dht=None, scale, chunk_size):
    """Direct call to JAX chunk_simple_gla_bwd on CPU."""
    cpu = jax.devices("cpu")[0]

    def to_jax(t):
        return jax.device_put(
            jnp.array(t.detach().cpu().float().numpy()), cpu
        )

    q_j, k_j, v_j, do_j = to_jax(q), to_jax(k), to_jax(v), to_jax(do)

    g_cs = None
    if g is not None:
        g_cs = _chunk_local_cumsum_jax(to_jax(g), chunk_size)

    g_gam = to_jax(g_gamma) if g_gamma is not None else None
    h0_j = to_jax(h0) if h0 is not None else None
    dht_j = to_jax(dht) if dht is not None else None

    dq, dk, dv, dg_raw, dh0 = chunk_simple_gla_bwd(
        q=q_j, k=k_j, v=v_j, do=do_j,
        dht=dht_j,
        g=g_cs, g_gamma=g_gam,
        h0=h0_j, scale=scale,
        chunk_size=chunk_size,
    )

    dg = None
    if dg_raw is not None:
        dg = _reverse_chunk_local_cumsum_jax(dg_raw, chunk_size)

    return dict(
        dq=np.asarray(dq, dtype=np.float32),
        dk=np.asarray(dk, dtype=np.float32),
        dv=np.asarray(dv, dtype=np.float32),
        dg=np.asarray(dg, dtype=np.float32) if dg is not None else None,
        dh0=np.asarray(dh0, dtype=np.float32) if dh0 is not None else None,
    )


def _make_inputs(cfg, dtype=torch.float32):
    """Generate random inputs for chunk backward testing."""
    B, T, H, K, V = cfg["B"], cfg["T"], cfg["H"], cfg["K"], cfg["V"]
    gate = cfg.get("gate", "g")

    torch.manual_seed(cfg["seed"])
    q = torch.randn(B, T, H, K, dtype=dtype)
    k = torch.randn(B, T, H, K, dtype=dtype)
    v = torch.randn(B, T, H, V, dtype=dtype)
    do = torch.randn(B, T, H, V, dtype=dtype)

    # g, g_gamma always float32 (cumsum precision)
    g = F.logsigmoid(torch.randn(B, T, H)) if gate in ("g", "g+g_gamma") else None
    g_gamma = F.logsigmoid(torch.randn(H)) if gate in ("g_gamma", "g+g_gamma") else None

    h0 = torch.randn(B, H, K, V, dtype=dtype) if cfg.get("h0") else None
    dht = torch.randn(B, H, K, V, dtype=dtype) if cfg.get("dht") else None

    return q, k, v, do, g, g_gamma, h0, dht


# ── Chunk test configs (K,V must be multiples of 128; T multiple of chunk_size) ──

CHUNK_BWD_CASES = [
    # ── standard ──
    dict(B=2, T=64, H=4, K=128, V=128, seed=42),
    dict(B=2, T=64, H=4, K=128, V=128, seed=43, h0=True),
    # ── longer T ──
    dict(B=1, T=128, H=2, K=128, V=128, seed=60),
    # ── K != V ──
    dict(B=2, T=64, H=4, K=256, V=128, seed=100),
    dict(B=2, T=64, H=4, K=128, V=256, seed=101),
    # ── single head ──
    dict(B=2, T=64, H=1, K=128, V=128, seed=120),
    # ── no gate ──
    dict(B=2, T=64, H=4, K=128, V=128, seed=70, gate="none"),
    # ── g_gamma only ──
    dict(B=2, T=64, H=4, K=128, V=128, seed=80, gate="g_gamma"),
    dict(B=2, T=64, H=4, K=128, V=128, seed=81, gate="g_gamma", h0=True),
    # NOTE: g+g_gamma cases are excluded. FLA Triton chunk_fwd_kernel_h has a
    # variable shadowing bug: USE_G overwrites b_g, then USE_G_GAMMA reads the
    # wrong value, producing incorrect h states. JAX kernel uses separate
    # variables and computes correctly, so results diverge.
    # ── custom scale ──
    dict(B=2, T=64, H=4, K=128, V=128, seed=110, scale=0.1),
    # ── with dht ──
    dict(B=2, T=64, H=4, K=128, V=128, seed=130, h0=True, dht=True),
]

CHUNK_BWD_BF16_CASES = [
    dict(B=2, T=64, H=4, K=128, V=128, seed=42),
    dict(B=2, T=64, H=4, K=128, V=128, seed=43, h0=True),
    dict(B=2, T=64, H=4, K=128, V=128, seed=70, gate="none"),
    dict(B=2, T=64, H=4, K=128, V=128, seed=80, gate="g_gamma"),
    # g+g_gamma excluded (FLA Triton variable shadowing bug in chunk_fwd_kernel_h)
    dict(B=2, T=64, H=4, K=128, V=128, seed=130, h0=True, dht=True),
]


def _assert_chunk_grads(tri, jax_r, *, atol, rtol, has_g, has_h0):
    """Compare Triton (gold) vs JAX chunk backward gradients."""
    assert compare_tensor("dq", tri["dq"], jax_r["dq"], atol=atol, rtol=rtol)
    assert compare_tensor("dk", tri["dk"], jax_r["dk"], atol=atol, rtol=rtol)
    assert compare_tensor("dv", tri["dv"], jax_r["dv"], atol=atol, rtol=rtol)
    if has_g and tri["dg"] is not None:
        assert compare_tensor("dg", tri["dg"], jax_r["dg"], atol=atol, rtol=rtol)
    if has_h0 and tri["dh0"] is not None and jax_r["dh0"] is not None:
        assert compare_tensor("dh0", tri["dh0"], jax_r["dh0"], atol=atol, rtol=rtol)


# ── Part 2a: fp32 ──


@requires_triton
@pytest.mark.parametrize("cfg", CHUNK_BWD_CASES, ids=[_case_id(c) for c in CHUNK_BWD_CASES])
def test_chunk_bwd_fp32(cfg):
    """Chunk backward: Triton (gold) vs JAX, float32."""
    B, T, H, K, V = cfg["B"], cfg["T"], cfg["H"], cfg["K"], cfg["V"]
    scale = cfg.get("scale", K ** -0.5)
    gate = cfg.get("gate", "g")
    chunk_size = min(64, max(16, _next_power_of_2(T)))

    q, k, v, do, g, g_gamma, h0, dht = _make_inputs(cfg, dtype=torch.float32)

    tri = _run_triton_chunk_bwd(
        q, k, v, do, g=g, g_gamma=g_gamma, h0=h0, dht=dht,
        scale=scale, chunk_size=chunk_size,
    )
    jax_r = _run_jax_chunk_bwd(
        q, k, v, do, g=g, g_gamma=g_gamma, h0=h0, dht=dht,
        scale=scale, chunk_size=chunk_size,
    )

    _assert_chunk_grads(
        tri, jax_r, atol=1e-4, rtol=1e-4,
        has_g=gate in ("g", "g+g_gamma"), has_h0=cfg.get("h0", False),
    )


# ── Part 2b: bf16 ──


@requires_triton
@pytest.mark.parametrize("cfg", CHUNK_BWD_BF16_CASES, ids=[_case_id(c) for c in CHUNK_BWD_BF16_CASES])
def test_chunk_bwd_bf16(cfg):
    """Chunk backward: Triton (gold) vs JAX, bfloat16 inputs.

    q, k, v, do, h0, dht are bf16; g and g_gamma stay float32 for cumsum
    precision.  Triton outputs bf16 (converted to fp32 for comparison);
    JAX computes internally in fp32 with Precision.HIGHEST.  Tolerance
    is set to ~1-2 ULP in bf16, with compare_tensor ULP fallback.
    """
    B, T, H, K, V = cfg["B"], cfg["T"], cfg["H"], cfg["K"], cfg["V"]
    scale = cfg.get("scale", K ** -0.5)
    gate = cfg.get("gate", "g")
    chunk_size = min(64, max(16, _next_power_of_2(T)))

    q, k, v, do, g, g_gamma, h0, dht = _make_inputs(cfg, dtype=torch.bfloat16)

    tri = _run_triton_chunk_bwd(
        q, k, v, do, g=g, g_gamma=g_gamma, h0=h0, dht=dht,
        scale=scale, chunk_size=chunk_size,
    )
    jax_r = _run_jax_chunk_bwd(
        q, k, v, do, g=g, g_gamma=g_gamma, h0=h0, dht=dht,
        scale=scale, chunk_size=chunk_size,
    )

    _assert_chunk_grads(
        tri, jax_r, atol=1e-3, rtol=1e-3,
        has_g=gate in ("g", "g+g_gamma"), has_h0=cfg.get("h0", False),
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

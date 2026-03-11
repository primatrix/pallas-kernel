# check_gla_layer.py
# Cross-framework comparison: PyTorch (torch_gla.py) vs JAX (jax_gla.py)
# Tests both the core naive_recurrent_gla kernel and the full GatedLinearAttention layer.
#
# 跨框架比对脚本：PyTorch CPU vs JAX
# 测试纯核心递归和完整 GatedLinearAttention 层。

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import torch
import torch.nn.functional as F

import jax
import jax.numpy as jnp
from flax import nnx

# Import both implementations
from src.torch.ops.gla import (
    naive_recurrent_gla as torch_naive_recurrent_gla,
    chunk_gla as torch_chunk_gla,
    fused_chunk_gla as torch_fused_chunk_gla,
)

from src.torch.layers.gla import GatedLinearAttention as TorchGLA

from src.torch.modules.convolution import ShortConvolution as TorchShortConv
from src.torch.modules.layernorm import RMSNorm as TorchRMSNorm
from src.torch.modules.fused_norm_gate import FusedRMSNormGated as TorchFNG

from src.jax.ops.gla import naive_recurrent_gla as jax_naive_recurrent_gla
from src.jax.layers.gla import GatedLinearAttention as JaxGLA
from src.jax.modules.layernorm import RMSNorm as JaxRMSNorm
from src.jax.modules.fused_norm_gate import FusedRMSNormGated as JaxFusedRMSNormGated
from src.jax.modules.convolution import ShortConvolution as JaxShortConv


# =============================================================================
# Conversion utilities
# =============================================================================


def t2j(t: torch.Tensor) -> jnp.ndarray:
    """PyTorch tensor → JAX array."""
    return jnp.array(t.detach().cpu().numpy())


def j2n(j: jnp.ndarray) -> np.ndarray:
    """JAX array → numpy."""
    return np.array(j)


def t2n(t: torch.Tensor) -> np.ndarray:
    """PyTorch tensor → numpy."""
    return t.detach().cpu().float().numpy()


# =============================================================================
# Weight transfer: PyTorch → JAX
# =============================================================================


def set_linear(jax_linear: nnx.Linear, pt_linear: torch.nn.Linear):
    """Copy weights from PyTorch Linear to JAX nnx.Linear.

    PyTorch weight: [out_features, in_features]
    JAX kernel:     [in_features, out_features]
    """
    jax_linear.kernel.value = t2j(pt_linear.weight.T)
    if pt_linear.bias is not None and jax_linear.bias is not None:
        jax_linear.bias.value = t2j(pt_linear.bias)


def set_conv(jax_conv_module, pt_conv_module):
    """Copy weights from PyTorch ShortConvolution to JAX ShortConvolution.

    PyTorch nn.Conv1d weight: [C, 1, K]  (depthwise, groups=C)
    JAX nnx.Conv kernel:     [K, 1, C]   (feature_group_count=C)
    """
    # Access the inner nnx.Conv
    jax_conv = jax_conv_module.conv
    pt_weight = pt_conv_module.weight  # [C, 1, K]
    # Permute: [C, 1, K] → [K, 1, C]
    jax_conv.kernel.value = t2j(pt_weight.permute(2, 1, 0))
    if pt_conv_module.bias is not None and jax_conv.bias is not None:
        jax_conv.bias.value = t2j(pt_conv_module.bias)


def transfer_gla_weights(jax_model: JaxGLA, pt_model: TorchGLA):
    """Transfer all weights from PyTorch GLA to JAX GLA."""
    # Linear projections
    set_linear(jax_model.q_proj, pt_model.q_proj)
    set_linear(jax_model.k_proj, pt_model.k_proj)
    set_linear(jax_model.v_proj, pt_model.v_proj)
    if pt_model.use_output_gate:
        set_linear(jax_model.g_proj, pt_model.g_proj)

    # Gate projection (Sequential of 2 Linears)
    pt_gk_layers = list(pt_model.gk_proj.children())
    jax_gk_layers = jax_model.gk_proj.layers
    set_linear(jax_gk_layers[0], pt_gk_layers[0])
    set_linear(jax_gk_layers[1], pt_gk_layers[1])

    # Output projection
    set_linear(jax_model.o_proj, pt_model.o_proj)

    # Short convolutions
    if pt_model.use_short_conv:
        set_conv(jax_model.q_conv1d, pt_model.q_conv1d)
        set_conv(jax_model.k_conv1d, pt_model.k_conv1d)
        set_conv(jax_model.v_conv1d, pt_model.v_conv1d)

    # Normalization
    if pt_model.fuse_norm_and_gate:
        if pt_model.g_norm_swish_gate.weight is not None:
            jax_model.g_norm_swish_gate.weight.value = t2j(
                pt_model.g_norm_swish_gate.weight
            )
    else:
        if pt_model.g_norm.weight is not None:
            jax_model.g_norm.weight.value = t2j(pt_model.g_norm.weight)


# =============================================================================
# Adaptive tolerance (same scheme as check_fused_recurrent_fwd.py)
# =============================================================================


def get_tolerance(K: int, V: int, T: int, is_layer: bool = False):
    """Compute adaptive atol/rtol.

    Core kernel: both use float32 scan/loop → near-exact.
    Full layer: multiple matmuls accumulate more error.
    """
    if is_layer:
        atol, rtol = 5e-4, 5e-4
        if K > 64 or V > 64:
            atol, rtol = 1e-3, 1e-3
        if T > 256:
            atol = max(atol, 2e-3)
    else:
        atol, rtol = 1e-5, 1e-5
        if K > 64 or V > 64:
            atol, rtol = 5e-5, 5e-5
        if K > 256 or V > 256:
            atol, rtol = 1e-4, 1e-4
        if T > 256:
            atol = max(atol, 5e-5)
    return atol, rtol


def compare_arrays(
    name: str, a_np: np.ndarray, b_np: np.ndarray, atol: float, rtol: float
) -> bool:
    """Compare two numpy arrays with detailed reporting.

    Returns True if match, False otherwise.
    """
    diff = np.abs(a_np - b_np)
    max_diff = diff.max()
    mean_diff = diff.mean()
    rel_diff = diff / (np.abs(a_np) + 1e-8)
    max_rel = rel_diff.max()

    print(f"\n  {name} [{list(a_np.shape)}]:")
    print(f"    Max abs diff:  {max_diff:.6e}")
    print(f"    Mean abs diff: {mean_diff:.6e}")
    print(f"    Max rel diff:  {max_rel:.6e}")

    ok = np.allclose(a_np, b_np, atol=atol, rtol=rtol)
    if not ok:
        max_idx = np.unravel_index(np.argmax(diff), diff.shape)
        print(f"    ❌ MISMATCH at {max_idx}")
        print(f"       PyTorch: {a_np[max_idx]:.8f}  JAX: {b_np[max_idx]:.8f}")
        print(f"       abs: {diff[max_idx]:.6e}  rel: {rel_diff[max_idx]:.6e}")
        print(f"       (atol={atol:.0e}, rtol={rtol:.0e})")
    return ok


# =============================================================================
# Test: Core kernel comparison
# =============================================================================


def run_test_kernel(
    B: int,
    T: int,
    H: int,
    K: int,
    V: int,
    cu_seqlens_list: list | None = None,
    use_initial_state: bool = False,
    output_final_state: bool = False,
    scale: float | None = None,
) -> bool:
    """Compare naive_recurrent_gla: PyTorch CPU vs JAX."""
    print(f"\n{'=' * 55}")
    print(f"Kernel Test: B={B}, T={T}, H={H}, K={K}, V={V}")
    print(
        f"  cu_seqlens={cu_seqlens_list}, init_state={use_initial_state}, "
        f"final_state={output_final_state}"
    )
    print(f"{'=' * 55}")

    torch.manual_seed(42)
    q_pt = torch.randn(B, T, H, K, dtype=torch.float32)
    k_pt = torch.randn(B, T, H, K, dtype=torch.float32)
    v_pt = torch.randn(B, T, H, V, dtype=torch.float32)
    gk_pt = F.logsigmoid(torch.randn(B, T, H, K, dtype=torch.float32))

    if cu_seqlens_list is not None:
        assert B == 1
        cu_seqlens_pt = torch.LongTensor(cu_seqlens_list)
        cu_seqlens_np = np.array(cu_seqlens_list, dtype=np.int64)
        N = len(cu_seqlens_list) - 1
    else:
        cu_seqlens_pt = None
        cu_seqlens_np = None
        N = B

    initial_state_pt = (
        torch.randn(N, H, K, V, dtype=torch.float32) if use_initial_state else None
    )
    s = scale if scale is not None else K**-0.5

    # --- PyTorch ---
    pt_o, pt_ht = torch_naive_recurrent_gla(
        q=q_pt,
        k=k_pt,
        v=v_pt,
        gk=gk_pt,
        scale=s,
        initial_state=initial_state_pt,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens_pt,
    )

    # --- JAX ---
    q_jax = t2j(q_pt)
    k_jax = t2j(k_pt)
    v_jax = t2j(v_pt)
    gk_jax = t2j(gk_pt)
    h0_jax = t2j(initial_state_pt) if initial_state_pt is not None else None

    jax_o, jax_ht = jax_naive_recurrent_gla(
        q=q_jax,
        k=k_jax,
        v=v_jax,
        gk=gk_jax,
        scale=s,
        initial_state=h0_jax,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens_np,
    )

    # --- Compare ---
    atol, rtol = get_tolerance(K, V, T, is_layer=False)

    pt_o_np = t2n(pt_o)
    jax_o_np = j2n(jax_o)

    ok = compare_arrays("Output o", pt_o_np, jax_o_np, atol, rtol)

    if output_final_state and pt_ht is not None and jax_ht is not None:
        ok_ht = compare_arrays("Final state ht", t2n(pt_ht), j2n(jax_ht), atol, rtol)
        ok = ok and ok_ht

    if ok:
        print("\n  ✅ MATCH")
    else:
        print("\n  ❌ MISMATCH!")
    return ok


# =============================================================================
# Test: Full layer comparison
# =============================================================================


def run_test_layer(
    B: int,
    T: int,
    hidden_size: int = 128,
    num_heads: int = 4,
    num_kv_heads: int | None = None,
    expand_k: float = 0.5,
    expand_v: float = 1.0,
    use_short_conv: bool = False,
    conv_size: int = 4,
    conv_bias: bool = False,
    use_output_gate: bool = True,
    gate_fn: str = "swish",
    fuse_norm: bool = True,
    feature_map: str | None = None,
    gate_logit_normalizer: int = 16,
    gate_low_rank_dim: int = 16,
    clamp_min: float | None = None,
    elementwise_affine: bool = True,
) -> bool:
    """Compare full GatedLinearAttention layer: PyTorch vs JAX."""
    print(f"\n{'=' * 55}")
    print(f"Layer Test: B={B}, T={T}, H_size={hidden_size}, heads={num_heads}")
    print(f"  kv_heads={num_kv_heads}, expand_k={expand_k}, expand_v={expand_v}")
    print(f"  conv={use_short_conv}, gate={use_output_gate}, fuse={fuse_norm}")
    print(
        f"  conv_bias={conv_bias}, affine={elementwise_affine}, low_rank={gate_low_rank_dim}"
    )
    print(f"{'=' * 55}")

    torch.manual_seed(42)

    # --- Create PyTorch model ---
    pt_model = TorchGLA(
        mode="chunk",
        hidden_size=hidden_size,
        expand_k=expand_k,
        expand_v=expand_v,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        feature_map=feature_map,
        use_short_conv=use_short_conv,
        conv_size=conv_size,
        conv_bias=conv_bias,
        use_output_gate=use_output_gate,
        gate_fn=gate_fn,
        elementwise_affine=elementwise_affine,
        fuse_norm=fuse_norm,
        gate_logit_normalizer=gate_logit_normalizer,
        gate_low_rank_dim=gate_low_rank_dim,
        clamp_min=clamp_min,
    )
    pt_model.eval()

    # --- Create JAX model ---
    jax_model = JaxGLA(
        mode="chunk",
        hidden_size=hidden_size,
        expand_k=expand_k,
        expand_v=expand_v,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        feature_map=feature_map,
        use_short_conv=use_short_conv,
        conv_size=conv_size,
        conv_bias=conv_bias,
        use_output_gate=use_output_gate,
        gate_fn=gate_fn,
        elementwise_affine=elementwise_affine,
        fuse_norm=fuse_norm,
        gate_logit_normalizer=gate_logit_normalizer,
        gate_low_rank_dim=gate_low_rank_dim,
        clamp_min=clamp_min,
        rngs=nnx.Rngs(0),  # dummy, weights will be overwritten
    )

    # --- Transfer weights ---
    transfer_gla_weights(jax_model, pt_model)

    # --- Generate input ---
    x_np = np.random.RandomState(42).randn(B, T, hidden_size).astype(np.float32)
    x_pt = torch.tensor(x_np)
    x_jax = jnp.array(x_np)

    # --- Forward ---
    with torch.no_grad():
        pt_out, _, _ = pt_model(x_pt)
    jax_out, _, _ = jax_model(x_jax)

    # --- Compare ---
    key_dim = int(hidden_size * expand_k)
    value_dim = int(hidden_size * expand_v)
    head_k = key_dim // num_heads
    head_v = value_dim // num_heads
    atol, rtol = get_tolerance(head_k, head_v, T, is_layer=True)

    pt_out_np = t2n(pt_out)
    jax_out_np = j2n(jax_out)

    ok = compare_arrays("Layer output", pt_out_np, jax_out_np, atol, rtol)
    if ok:
        print("\n  ✅ MATCH")
    else:
        print("\n  ❌ MISMATCH!")
    return ok


# =============================================================================
# Test: Varlen equivalence (JAX side)
# =============================================================================


def run_test_varlen_equiv(T: int, H: int, K: int, V: int) -> bool:
    """Single-seq varlen (cu_seqlens=[0,T]) == non-varlen B=1, JAX side."""
    print(f"\n{'=' * 55}")
    print(f"Varlen equiv: T={T}, H={H}, K={K}, V={V}")
    print(f"{'=' * 55}")

    np.random.seed(13)
    q = jnp.array(np.random.randn(1, T, H, K).astype(np.float32))
    k = jnp.array(np.random.randn(1, T, H, K).astype(np.float32))
    v = jnp.array(np.random.randn(1, T, H, V).astype(np.float32))
    gk = jax.nn.log_sigmoid(jnp.array(np.random.randn(1, T, H, K).astype(np.float32)))
    h0 = jnp.array(np.random.randn(1, H, K, V).astype(np.float32))
    cu = np.array([0, T], dtype=np.int64)

    o_nv, ht_nv = jax_naive_recurrent_gla(
        q, k, v, gk, initial_state=h0, output_final_state=True
    )
    o_vl, ht_vl = jax_naive_recurrent_gla(
        q, k, v, gk, initial_state=h0, output_final_state=True, cu_seqlens=cu
    )

    ok_o = np.allclose(j2n(o_nv), j2n(o_vl), atol=1e-6)
    ok_ht = np.allclose(j2n(ht_nv), j2n(ht_vl), atol=1e-6)
    ok = ok_o and ok_ht
    if ok:
        print("  ✅ single-seq varlen == non-varlen")
    else:
        d_o = np.abs(j2n(o_nv) - j2n(o_vl)).max()
        d_ht = np.abs(j2n(ht_nv) - j2n(ht_vl)).max()
        print(f"  ❌ mismatch (o diff={d_o:.2e}, ht diff={d_ht:.2e})")
    return ok


# =============================================================================
# Test: State split consistency (JAX side)
# =============================================================================


def run_test_state_split(B: int, T: int, H: int, K: int, V: int) -> bool:
    """Split sequence processing == full sequence, both frameworks."""
    print(f"\n{'=' * 55}")
    print(f"State split: B={B}, T={T}, H={H}, K={K}, V={V}")
    print(f"{'=' * 55}")

    np.random.seed(7)
    T1 = T // 2

    q = jnp.array(np.random.randn(B, T, H, K).astype(np.float32))
    k = jnp.array(np.random.randn(B, T, H, K).astype(np.float32))
    v = jnp.array(np.random.randn(B, T, H, V).astype(np.float32))
    gk = jax.nn.log_sigmoid(jnp.array(np.random.randn(B, T, H, K).astype(np.float32)))

    o_full, s_full = jax_naive_recurrent_gla(q, k, v, gk, output_final_state=True)
    o1, s1 = jax_naive_recurrent_gla(
        q[:, :T1], k[:, :T1], v[:, :T1], gk[:, :T1], output_final_state=True
    )
    o2, s2 = jax_naive_recurrent_gla(
        q[:, T1:],
        k[:, T1:],
        v[:, T1:],
        gk[:, T1:],
        initial_state=s1,
        output_final_state=True,
    )

    ok_o = np.allclose(j2n(o_full[:, :T1]), j2n(o1), atol=1e-5) and np.allclose(
        j2n(o_full[:, T1:]), j2n(o2), atol=1e-5
    )
    ok_s = np.allclose(j2n(s_full), j2n(s2), atol=1e-5)
    ok = ok_o and ok_s
    if ok:
        print("  ✅ split == full")
    else:
        print("  ❌ split != full")
    return ok


# =============================================================================
# Test: RMSNorm cross-framework
# =============================================================================


def run_test_rmsnorm() -> bool:
    """Compare RMSNorm: PyTorch vs JAX."""
    print(f"\n{'=' * 55}")
    print("RMSNorm cross-framework")
    print(f"{'=' * 55}")

    torch.manual_seed(0)
    pt_norm = TorchRMSNorm(64, elementwise_affine=True, eps=1e-5)
    jax_norm = JaxRMSNorm(64, elementwise_affine=True, eps=1e-5, rngs=nnx.Rngs(0))
    jax_norm.weight.value = t2j(pt_norm.weight)

    x_np = np.random.randn(2, 10, 64).astype(np.float32)
    x_pt = torch.tensor(x_np)
    x_jax = jnp.array(x_np)

    pt_y = pt_norm(x_pt)
    jax_y = jax_norm(x_jax)

    ok = compare_arrays("RMSNorm", t2n(pt_y), j2n(jax_y), 1e-6, 1e-6)
    if ok:
        print("\n  ✅ MATCH")
    return ok


# =============================================================================
# Test: FusedRMSNormGated cross-framework
# =============================================================================


def run_test_fused_norm_gated() -> bool:
    """Compare FusedRMSNormGated: PyTorch vs JAX."""
    print(f"\n{'=' * 55}")
    print("FusedRMSNormGated cross-framework")
    print(f"{'=' * 55}")

    torch.manual_seed(0)
    pt_fng = TorchFNG(64, elementwise_affine=True, eps=1e-5)
    jax_fng = JaxFusedRMSNormGated(
        64, elementwise_affine=True, eps=1e-5, rngs=nnx.Rngs(0)
    )
    jax_fng.weight.value = t2j(pt_fng.weight)

    x_np = np.random.randn(2, 10, 64).astype(np.float32)
    g_np = np.random.randn(2, 10, 64).astype(np.float32)

    pt_y = pt_fng(torch.tensor(x_np), torch.tensor(g_np))
    jax_y = jax_fng(jnp.array(x_np), jnp.array(g_np))

    ok = compare_arrays("FusedRMSNormGated", t2n(pt_y), j2n(jax_y), 1e-6, 1e-6)
    if ok:
        print("\n  ✅ MATCH")
    return ok


# =============================================================================
# Test: RMSNorm no affine (cross-framework)
# =============================================================================


def run_test_rmsnorm_no_affine() -> bool:
    """RMSNorm with elementwise_affine=False, cross-framework."""
    print(f"\n{'=' * 55}")
    print("RMSNorm no-affine cross-framework")
    print(f"{'=' * 55}")

    pt_norm = TorchRMSNorm(64, elementwise_affine=False, eps=1e-5)
    jax_norm = JaxRMSNorm(64, elementwise_affine=False, eps=1e-5, rngs=nnx.Rngs(0))

    x_np = np.random.RandomState(0).randn(2, 10, 64).astype(np.float32)
    pt_y = pt_norm(torch.tensor(x_np))
    jax_y = jax_norm(jnp.array(x_np))

    ok = compare_arrays("RMSNorm(no_affine)", t2n(pt_y), j2n(jax_y), 1e-6, 1e-6)
    if ok:
        print("\n  ✅ MATCH")
    return ok


# =============================================================================
# Test: FusedRMSNormGated no affine (cross-framework)
# =============================================================================


def run_test_fused_norm_gated_no_affine() -> bool:
    """FusedRMSNormGated with elementwise_affine=False, cross-framework."""
    print(f"\n{'=' * 55}")
    print("FusedRMSNormGated no-affine cross-framework")
    print(f"{'=' * 55}")

    pt_fng = TorchFNG(64, elementwise_affine=False, eps=1e-5)
    jax_fng = JaxFusedRMSNormGated(
        64, elementwise_affine=False, eps=1e-5, rngs=nnx.Rngs(0)
    )

    x_np = np.random.RandomState(0).randn(2, 10, 64).astype(np.float32)
    g_np = np.random.RandomState(1).randn(2, 10, 64).astype(np.float32)

    pt_y = pt_fng(torch.tensor(x_np), torch.tensor(g_np))
    jax_y = jax_fng(jnp.array(x_np), jnp.array(g_np))

    ok = compare_arrays(
        "FusedRMSNormGated(no_affine)", t2n(pt_y), j2n(jax_y), 1e-6, 1e-6
    )
    if ok:
        print("\n  ✅ MATCH")
    return ok


# =============================================================================
# Test: ShortConvolution cross-framework
# =============================================================================


def run_test_short_conv_xf(
    D: int = 64,
    K_size: int = 4,
    T: int = 16,
    activation: str | None = "silu",
    bias: bool = False,
) -> bool:
    """Cross-framework ShortConvolution comparison."""
    print(f"\n{'=' * 55}")
    print(f"ShortConv XF: D={D}, K={K_size}, T={T}, act={activation}, bias={bias}")
    print(f"{'=' * 55}")

    torch.manual_seed(0)
    pt_conv = TorchShortConv(
        hidden_size=D, kernel_size=K_size, bias=bias, activation=activation
    )
    jax_conv = JaxShortConv(
        hidden_size=D,
        kernel_size=K_size,
        bias=bias,
        activation=activation,
        rngs=nnx.Rngs(0),
    )
    set_conv(jax_conv, pt_conv)

    x_np = np.random.RandomState(42).randn(2, T, D).astype(np.float32)
    with torch.no_grad():
        pt_y, _ = pt_conv(torch.tensor(x_np))
    jax_y, _ = jax_conv(jnp.array(x_np))

    ok = compare_arrays("ShortConv output", t2n(pt_y), j2n(jax_y), 1e-5, 1e-5)
    if ok:
        print("\n  ✅ MATCH")
    return ok


# =============================================================================
# Test: ShortConvolution with cache prefix (JAX consistency)
# =============================================================================


def run_test_short_conv_cache() -> bool:
    """ShortConv with cache prefix — staged == full (JAX)."""
    print(f"\n{'=' * 55}")
    print("ShortConv cache prefix (JAX)")
    print(f"{'=' * 55}")

    D, K_size, B = 32, 4, 2
    T1, T2 = 8, 8
    T = T1 + T2
    conv = JaxShortConv(
        hidden_size=D, kernel_size=K_size, activation="silu", rngs=nnx.Rngs(0)
    )
    x = jnp.array(np.random.RandomState(42).randn(B, T, D).astype(np.float32))

    # Full forward
    y_full, _ = conv(x)

    # Two-stage with cache: pass last (K-1) tokens of first chunk as cache
    y1, _ = conv(x[:, :T1, :])
    cache = x[:, T1 - (K_size - 1) : T1, :]  # [B, K-1, D]
    y2, _ = conv(x[:, T1:, :], cache=cache)

    y_staged = jnp.concatenate([y1, y2], axis=1)

    ok = np.allclose(j2n(y_full), j2n(y_staged), atol=1e-5)
    if ok:
        print("  ✅ staged == full")
    else:
        diff = np.abs(j2n(y_full) - j2n(y_staged)).max()
        print(f"  ❌ max diff: {diff:.2e}")
    return ok


# =============================================================================
# Test: ShortConvolution short seq (T < kernel_size - 1)
# =============================================================================


def run_test_short_conv_short_seq() -> bool:
    """ShortConv output_final_state when T < kernel_size - 1."""
    print(f"\n{'=' * 55}")
    print("ShortConv short seq final_state (JAX)")
    print(f"{'=' * 55}")

    D, K_size, B, T = 16, 4, 1, 2  # T=2 < K-1=3
    conv = JaxShortConv(
        hidden_size=D, kernel_size=K_size, activation="silu", rngs=nnx.Rngs(0)
    )
    x = jnp.array(np.random.RandomState(0).randn(B, T, D).astype(np.float32))

    y, cache = conv(x, output_final_state=True)
    ok = True
    if y.shape != (B, T, D):
        print(f"  ❌ output shape: {y.shape} != {(B, T, D)}")
        ok = False
    if cache.shape != (B, K_size - 1, D):
        print(f"  ❌ cache shape: {cache.shape} != {(B, K_size - 1, D)}")
        ok = False
    # First element of cache should be zero (padding since T < K-1)
    if not np.allclose(j2n(cache[:, 0, :]), 0.0, atol=1e-8):
        print("  ❌ first cache element should be zero padding")
        ok = False
    # Last T elements should match input x
    if not np.allclose(j2n(cache[:, -T:, :]), j2n(x[0, :, :]), atol=1e-8):
        print("  ❌ cache tail should be input x")
        ok = False
    if ok:
        print(f"  Output: {y.shape}, Cache: {cache.shape}")
        print("  ✅ padding correctness verified")
    return ok


# =============================================================================
# Test: ShortConvolution step-by-step (JAX consistency)
# =============================================================================


def run_test_short_conv_step() -> bool:
    """ShortConv step-by-step vs full forward (JAX)."""
    print(f"\n{'=' * 55}")
    print("ShortConv step vs full (JAX)")
    print(f"{'=' * 55}")

    D, K_size, T, B = 32, 4, 12, 2
    conv = JaxShortConv(
        hidden_size=D, kernel_size=K_size, activation="silu", rngs=nnx.Rngs(0)
    )
    x = jnp.array(np.random.RandomState(0).randn(B, T, D).astype(np.float32))

    # Full forward
    y_full, _ = conv(x)

    # Step-by-step
    cache = None
    y_steps = []
    for t in range(T):
        x_t = x[:, t : t + 1, :]  # [B, 1, D]
        y_t, cache = conv.step(x_t, cache, output_final_state=True)
        y_steps.append(y_t)
    y_step = jnp.concatenate(y_steps, axis=1)

    ok = np.allclose(j2n(y_full), j2n(y_step), atol=1e-5)
    if ok:
        print("  ✅ step-by-step == full forward")
    else:
        diff = np.abs(j2n(y_full) - j2n(y_step)).max()
        print(f"  ❌ max diff: {diff:.2e}")
    return ok


# =============================================================================
# Test: ShortConvolution varlen cross-framework
# =============================================================================


def run_test_short_conv_varlen_xf() -> bool:
    """Cross-framework ShortConvolution with cu_seqlens."""
    print(f"\n{'=' * 55}")
    print("ShortConv varlen cross-framework")
    print(f"{'=' * 55}")

    D, K_size, T_total = 32, 4, 24
    cu_list = [0, 8, 16, 24]

    torch.manual_seed(0)
    pt_conv = TorchShortConv(hidden_size=D, kernel_size=K_size, activation="silu")
    jax_conv = JaxShortConv(
        hidden_size=D, kernel_size=K_size, activation="silu", rngs=nnx.Rngs(0)
    )
    set_conv(jax_conv, pt_conv)

    x_np = np.random.RandomState(42).randn(1, T_total, D).astype(np.float32)
    with torch.no_grad():
        pt_y, _ = pt_conv(torch.tensor(x_np), cu_seqlens=torch.LongTensor(cu_list))
    jax_y, _ = jax_conv(jnp.array(x_np), cu_seqlens=np.array(cu_list, dtype=np.int64))

    ok = compare_arrays("ShortConv varlen", t2n(pt_y), j2n(jax_y), 1e-5, 1e-5)
    if ok:
        print("\n  ✅ MATCH")
    return ok


# =============================================================================
# Test: Layer with cu_seqlens (cross-framework)
# =============================================================================


def run_test_layer_cu_seqlens(
    T_total: int = 48,
    hidden_size: int = 128,
    num_heads: int = 4,
    num_kv_heads: int | None = None,
    use_short_conv: bool = False,
    conv_size: int = 4,
) -> bool:
    """Full GLA layer with cu_seqlens, cross-framework."""
    print(f"\n{'=' * 55}")
    print(
        f"Layer cu_seqlens: T={T_total}, H_size={hidden_size}, "
        f"heads={num_heads}, kv_heads={num_kv_heads}, conv={use_short_conv}"
    )
    print(f"{'=' * 55}")

    torch.manual_seed(42)
    cu_list = [0, 16, 32, T_total]

    pt_model = TorchGLA(
        mode="chunk",
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        use_short_conv=use_short_conv,
        conv_size=conv_size,
        use_output_gate=True,
        gate_fn="swish",
        fuse_norm=True,
    )
    pt_model.eval()

    jax_model = JaxGLA(
        mode="chunk",
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        use_short_conv=use_short_conv,
        conv_size=conv_size,
        use_output_gate=True,
        gate_fn="swish",
        fuse_norm=True,
        rngs=nnx.Rngs(0),
    )
    transfer_gla_weights(jax_model, pt_model)

    x_np = np.random.RandomState(42).randn(1, T_total, hidden_size).astype(np.float32)

    with torch.no_grad():
        pt_out, _, _ = pt_model(
            torch.tensor(x_np), cu_seqlens=torch.LongTensor(cu_list)
        )
    jax_out, _, _ = jax_model(
        jnp.array(x_np), cu_seqlens=np.array(cu_list, dtype=np.int64)
    )

    head_k = int(hidden_size * 0.5) // num_heads
    head_v = hidden_size // num_heads
    atol, rtol = get_tolerance(head_k, head_v, T_total, is_layer=True)

    ok = compare_arrays(
        "Layer output (cu_seqlens)", t2n(pt_out), j2n(jax_out), atol, rtol
    )
    if ok:
        print("\n  ✅ MATCH")
    else:
        print("\n  ❌ MISMATCH!")
    return ok


# =============================================================================
# Test: ShortConvolution step cross-framework
# =============================================================================


def run_test_short_conv_step_xf() -> bool:
    """Cross-framework ShortConv step-by-step output comparison.

    PyTorch uses rolling [N,D,W] cache; JAX uses [B,W-1,D] concat cache.
    Outputs should match token-by-token.
    """
    print(f"\n{'=' * 55}")
    print("ShortConv step cross-framework")
    print(f"{'=' * 55}")

    D, K_size, T, B = 32, 4, 10, 2
    torch.manual_seed(0)
    pt_conv = TorchShortConv(hidden_size=D, kernel_size=K_size, activation="silu")
    jax_conv = JaxShortConv(
        hidden_size=D, kernel_size=K_size, activation="silu", rngs=nnx.Rngs(0)
    )
    set_conv(jax_conv, pt_conv)

    x_np = np.random.RandomState(42).randn(B, T, D).astype(np.float32)

    # PyTorch step-by-step
    pt_cache = None
    pt_steps = []
    for t in range(T):
        x_t = torch.tensor(x_np[:, t : t + 1, :])
        with torch.no_grad():
            y_t, pt_cache = pt_conv.step(x_t, pt_cache, output_final_state=True)
        pt_steps.append(t2n(y_t))

    # JAX step-by-step
    jax_cache = None
    jax_steps = []
    for t in range(T):
        x_t = jnp.array(x_np[:, t : t + 1, :])
        y_t, jax_cache = jax_conv.step(x_t, jax_cache, output_final_state=True)
        jax_steps.append(j2n(y_t))

    pt_out = np.concatenate(pt_steps, axis=1)
    jax_out = np.concatenate(jax_steps, axis=1)

    ok = compare_arrays("ShortConv step output", pt_out, jax_out, 1e-5, 1e-5)
    if ok:
        print("\n  ✅ MATCH")
    return ok


# =============================================================================
# Test: ShortConv varlen with short segment (seg_len < W-1) final_state
# =============================================================================


def run_test_short_conv_varlen_short_seg() -> bool:
    """ShortConv varlen with segments shorter than kernel_size-1.

    Tests the zero-padding branch in final_state computation.
    """
    print(f"\n{'=' * 55}")
    print("ShortConv varlen short-segment cache (JAX)")
    print(f"{'=' * 55}")

    D, K_size = 16, 4  # W-1 = 3
    # Segments: length 1, 2, 5 — first two are shorter than W-1=3
    cu_list = [0, 1, 3, 8]
    T_total = 8

    conv = JaxShortConv(
        hidden_size=D, kernel_size=K_size, activation="silu", rngs=nnx.Rngs(0)
    )
    x = jnp.array(np.random.RandomState(0).randn(1, T_total, D).astype(np.float32))

    y, cache = conv(
        x, output_final_state=True, cu_seqlens=np.array(cu_list, dtype=np.int64)
    )

    ok = True
    # Output shape
    if y.shape != (1, T_total, D):
        print(f"  ❌ output shape: {y.shape}")
        ok = False

    # Cache shape: [N=3, W-1=3, D]
    if cache.shape != (3, K_size - 1, D):
        print(f"  ❌ cache shape: {cache.shape} != {(3, K_size - 1, D)}")
        ok = False

    # Segment 0: length=1 < W-1=3 → 2 zero-pad rows, then 1 token
    seg0_cache = j2n(cache[0])  # [3, D]
    if not np.allclose(seg0_cache[:2], 0.0, atol=1e-8):
        print("  ❌ seg0 cache first 2 rows should be zero")
        ok = False
    if not np.allclose(seg0_cache[2], j2n(x[0, 0]), atol=1e-8):
        print("  ❌ seg0 cache last row should be x[0]")
        ok = False

    # Segment 1: length=2 < W-1=3 → 1 zero-pad row, then 2 tokens
    seg1_cache = j2n(cache[1])  # [3, D]
    if not np.allclose(seg1_cache[0], 0.0, atol=1e-8):
        print("  ❌ seg1 cache first row should be zero")
        ok = False
    if not np.allclose(seg1_cache[1:], j2n(x[0, 1:3]), atol=1e-8):
        print("  ❌ seg1 cache rows 1-2 should match x[1:3]")
        ok = False

    # Segment 2: length=5 >= W-1=3 → no padding, last 3 tokens
    seg2_cache = j2n(cache[2])  # [3, D]
    if not np.allclose(seg2_cache, j2n(x[0, 5:8]), atol=1e-8):
        print("  ❌ seg2 cache should be last 3 tokens of segment")
        ok = False

    if ok:
        print("  ✅ short segment padding verified")
    return ok


# =============================================================================
# Test: ShortConv final_state cross-framework
# =============================================================================


def run_test_short_conv_final_state_xf() -> bool:
    """Cross-framework ShortConv output_final_state comparison.

    PyTorch stores final_state as [B, D, W] (conv1d format).
    JAX stores final_state as [B, W-1, D].
    The underlying *tokens* stored should be equivalent.
    """
    print(f"\n{'=' * 55}")
    print("ShortConv final_state cross-framework")
    print(f"{'=' * 55}")

    D, K_size, T, B = 32, 4, 12, 2
    torch.manual_seed(0)
    pt_conv = TorchShortConv(hidden_size=D, kernel_size=K_size, activation="silu")
    jax_conv = JaxShortConv(
        hidden_size=D, kernel_size=K_size, activation="silu", rngs=nnx.Rngs(0)
    )
    set_conv(jax_conv, pt_conv)

    x_np = np.random.RandomState(42).randn(B, T, D).astype(np.float32)

    # PyTorch forward with final_state
    with torch.no_grad():
        pt_y, pt_cache = pt_conv(torch.tensor(x_np), output_final_state=True)
    # JAX forward with final_state
    jax_y, jax_cache = jax_conv(jnp.array(x_np), output_final_state=True)

    ok = True

    # Output should match
    ok_y = compare_arrays("ShortConv output", t2n(pt_y), j2n(jax_y), 1e-5, 1e-5)
    ok = ok and ok_y

    # PyTorch cache: [B, D, W] → extract last W-1 tokens as [B, W-1, D]
    # pt_cache is the raw input tokens in conv1d layout [B, D, W]
    # The last W-1 columns correspond to the last W-1 input tokens
    # To compare: pt_cache[:, :, 1:] transposed → [B, W-1, D]
    pt_cache_tokens = t2n(pt_cache[:, :, 1:]).transpose(0, 2, 1)  # [B, W-1, D]
    jax_cache_tokens = j2n(jax_cache)  # [B, W-1, D]

    ok_cache = compare_arrays(
        "ShortConv final_state (tokens)", pt_cache_tokens, jax_cache_tokens, 1e-6, 1e-6
    )
    ok = ok and ok_cache

    if ok:
        print("\n  ✅ MATCH")
    return ok


# =============================================================================
# Test: ShortConv activation='swish' alias
# =============================================================================


def run_test_short_conv_swish_alias() -> bool:
    """Verify activation='swish' is same as 'silu'."""
    print(f"\n{'=' * 55}")
    print("ShortConv swish alias == silu")
    print(f"{'=' * 55}")

    D, K_size, T, B = 32, 4, 10, 2
    torch.manual_seed(0)
    pt_silu = TorchShortConv(hidden_size=D, kernel_size=K_size, activation="silu")
    pt_swish = TorchShortConv(hidden_size=D, kernel_size=K_size, activation="swish")
    # Copy weights
    pt_swish.weight.data.copy_(pt_silu.weight.data)

    jax_silu = JaxShortConv(
        hidden_size=D, kernel_size=K_size, activation="silu", rngs=nnx.Rngs(0)
    )
    jax_swish = JaxShortConv(
        hidden_size=D, kernel_size=K_size, activation="swish", rngs=nnx.Rngs(0)
    )
    set_conv(jax_silu, pt_silu)
    set_conv(jax_swish, pt_silu)

    x_np = np.random.RandomState(42).randn(B, T, D).astype(np.float32)

    with torch.no_grad():
        pt_y_silu, _ = pt_silu(torch.tensor(x_np))
        pt_y_swish, _ = pt_swish(torch.tensor(x_np))
    jax_y_silu, _ = jax_silu(jnp.array(x_np))
    jax_y_swish, _ = jax_swish(jnp.array(x_np))

    ok_pt = np.allclose(t2n(pt_y_silu), t2n(pt_y_swish), atol=1e-8)
    ok_jax = np.allclose(j2n(jax_y_silu), j2n(jax_y_swish), atol=1e-8)
    ok = ok_pt and ok_jax
    if ok:
        print("  ✅ swish === silu (both frameworks)")
    else:
        print(f"  ❌ mismatch: PT={ok_pt}, JAX={ok_jax}")
    return ok


# =============================================================================
# Test: PyTorch attention_mask pipeline
# =============================================================================


def run_test_attention_mask_pipeline() -> bool:
    """PyTorch-only: attention_mask should give same output as cu_seqlens.

    Tests: get_unpad_data, index_first_axis, pad_input pipeline.
    """
    from src.torch.layers.utils import (
        get_unpad_data,
        index_first_axis,
        pad_input,
    )

    print(f"\n{'=' * 55}")
    print("PyTorch attention_mask pipeline")
    print(f"{'=' * 55}")

    torch.manual_seed(42)
    B, T, hidden_size = 3, 16, 128
    num_heads = 4

    # Create model
    pt_model = TorchGLA(
        mode="chunk",
        hidden_size=hidden_size,
        num_heads=num_heads,
        use_short_conv=False,
        use_output_gate=True,
        gate_fn="swish",
        fuse_norm=True,
    )
    pt_model.eval()

    # Create input with variable-length padding
    # Seq lengths: 12, 16, 8
    x_np = np.random.RandomState(42).randn(B, T, hidden_size).astype(np.float32)
    x_pt = torch.tensor(x_np)

    # Create attention mask: 1=valid, 0=padding
    mask = torch.zeros(B, T, dtype=torch.bool)
    mask[0, :12] = True
    mask[1, :16] = True
    mask[2, :8] = True

    # Method 1: attention_mask
    with torch.no_grad():
        out_mask, _, _ = pt_model(x_pt, attention_mask=mask)

    # Method 2: manual unpad → cu_seqlens
    indices, cu_seqlens, max_seqlen = get_unpad_data(mask)
    x_packed = index_first_axis(x_pt.reshape(B * T, hidden_size), indices).unsqueeze(
        0
    )  # [1, total_valid, hidden_size]

    with torch.no_grad():
        out_cu, _, _ = pt_model(x_packed, cu_seqlens=cu_seqlens)

    # Unpack cu_seqlens result to compare
    out_cu_padded = pad_input(out_cu.squeeze(0), indices, B, T)

    # Compare — the valid positions should match
    ok = True
    for b in range(B):
        seq_len = mask[b].sum().item()
        diff = (out_mask[b, :seq_len] - out_cu_padded[b, :seq_len]).abs().max().item()
        if diff > 1e-5:
            print(f"  ❌ batch {b}: max diff = {diff:.2e}")
            ok = False

    if ok:
        print("  ✅ attention_mask == cu_seqlens pipeline")
    return ok


# =============================================================================
# Test: PyTorch ShortConv forward auto-step routing
# =============================================================================


def run_test_pt_short_conv_auto_step() -> bool:
    """PyTorch ShortConv.forward auto-routes to step() when B*T == N.

    When input has B=N, T=1 tokens per sequence, forward() should
    auto-detect and route to step().
    """
    print(f"\n{'=' * 55}")
    print("PyTorch ShortConv auto-step routing")
    print(f"{'=' * 55}")

    D, K_size = 32, 4
    B = 2  # N = B since no cu_seqlens

    torch.manual_seed(0)
    pt_conv = TorchShortConv(hidden_size=D, kernel_size=K_size, activation="silu")

    x_np = np.random.RandomState(42).randn(B, 1, D).astype(np.float32)
    x_pt = torch.tensor(x_np)

    # B * T = 2 * 1 = 2, N = B = 2 → should auto-route to step()
    with torch.no_grad():
        y_forward, cache_fwd = pt_conv(x_pt, output_final_state=True)

    # Compare with explicit step() call
    with torch.no_grad():
        y_step, cache_step = pt_conv.step(x_pt, cache=None, output_final_state=True)

    ok = True
    diff_y = (y_forward - y_step).abs().max().item()
    if diff_y > 1e-8:
        print(f"  ❌ output diff: {diff_y:.2e}")
        ok = False

    diff_c = (cache_fwd - cache_step).abs().max().item()
    if diff_c > 1e-8:
        print(f"  ❌ cache diff: {diff_c:.2e}")
        ok = False

    if ok:
        print(f"  forward() auto-routed to step(): output diff={diff_y:.2e}")
        print("  ✅ auto-step routing verified")
    return ok


# =============================================================================
# Test: Layer gate_fn='gelu' and 'tanh'
# =============================================================================
# (Uses run_test_layer, registered below in main)


# =============================================================================
# Test: Layer feature_map variants (tanh, gelu)
# =============================================================================
# (Uses run_test_layer, registered below in main)


# =============================================================================
# Test: Layer use_output_gate=False + fuse_norm=True
# =============================================================================
# (Uses run_test_layer, registered below in main)


# =============================================================================
# Test: ShortConv step edge-case branches
# =============================================================================


def run_test_short_conv_step_no_cache_no_output() -> bool:
    """JAX ShortConv.step with cache=None, output_final_state=False.

    Tests the else branch: y = conv(x), new_cache = None.
    """
    print(f"\n{'=' * 55}")
    print("ShortConv step: no cache, no output_final_state")
    print(f"{'=' * 55}")

    D, K_size, B = 32, 4, 2
    jax_conv = JaxShortConv(
        hidden_size=D, kernel_size=K_size, activation="silu", rngs=nnx.Rngs(0)
    )

    x_np = np.random.RandomState(0).randn(B, 1, D).astype(np.float32)
    x = jnp.array(x_np)

    y, cache = jax_conv.step(x, cache=None, output_final_state=False)

    ok = True
    if y.shape != (B, 1, D):
        print(f"  ❌ output shape: {y.shape}")
        ok = False
    if cache is not None:
        print(f"  ❌ cache should be None, got {type(cache)}")
        ok = False
    if not jnp.all(jnp.isfinite(y)):
        print("  ❌ output has non-finite values")
        ok = False

    if ok:
        print("  ✅ step(cache=None, output_final_state=False) correct")
    return ok


def run_test_short_conv_step_discard_cache() -> bool:
    """JAX ShortConv.step with cache!=None, output_final_state=False.

    Tests the branch: cache is used for computation, but new_cache = None.
    """
    print(f"\n{'=' * 55}")
    print("ShortConv step: has cache, output_final_state=False")
    print(f"{'=' * 55}")

    D, K_size, B = 32, 4, 2
    jax_conv = JaxShortConv(
        hidden_size=D, kernel_size=K_size, activation="silu", rngs=nnx.Rngs(0)
    )

    x_np = np.random.RandomState(0).randn(B, 1, D).astype(np.float32)
    x = jnp.array(x_np)

    # First step to get cache
    _, cache0 = jax_conv.step(x, cache=None, output_final_state=True)
    assert cache0 is not None

    # Second step with cache but output_final_state=False
    y, new_cache = jax_conv.step(x, cache=cache0, output_final_state=False)

    ok = True
    if y.shape != (B, 1, D):
        print(f"  ❌ output shape: {y.shape}")
        ok = False
    if new_cache is not None:
        print(f"  ❌ new_cache should be None, got shape {new_cache.shape}")
        ok = False

    # Compare: computation should still be correct (same as with output_final_state=True)
    y2, _ = jax_conv.step(x, cache=cache0, output_final_state=True)
    diff = np.abs(j2n(y) - j2n(y2)).max()
    if diff > 1e-8:
        print(f"  ❌ output differs from output_final_state=True: {diff:.2e}")
        ok = False

    if ok:
        print("  Output matches, new_cache correctly None")
        print("  ✅ step(cache!=None, output_final_state=False) correct")
    return ok


# =============================================================================
# Test: PyTorch ShortConv step no-cache else branch
# =============================================================================


def run_test_pt_short_conv_step_no_cache() -> bool:
    """PyTorch ShortConv.step with no cache, no output_final_state.

    Tests the else branch: y = x_step * w[:, -1].
    """
    print(f"\n{'=' * 55}")
    print("PT ShortConv step: no cache, no output")
    print(f"{'=' * 55}")

    D, K_size, B = 32, 4, 2
    torch.manual_seed(0)
    pt_conv = TorchShortConv(hidden_size=D, kernel_size=K_size, activation="silu")

    x_np = np.random.RandomState(0).randn(B, 1, D).astype(np.float32)
    x_pt = torch.tensor(x_np)

    with torch.no_grad():
        y, cache = pt_conv.step(x_pt, cache=None, output_final_state=False)

    ok = True
    if y.shape != (B, 1, D):
        print(f"  ❌ output shape: {y.shape}")
        ok = False
    if cache is not None:
        print("  ❌ cache should be None")
        ok = False
    if not torch.all(torch.isfinite(y)):
        print("  ❌ non-finite output")
        ok = False

    if ok:
        print("  ✅ PT step no-cache else branch verified")
    return ok


# =============================================================================
# Test: ShortConv step with bias (cross-framework)
# =============================================================================


def run_test_short_conv_step_bias_xf() -> bool:
    """Cross-framework ShortConv step with bias=True.

    Tests bias addition in step computation for both frameworks.
    """
    print(f"\n{'=' * 55}")
    print("ShortConv step with bias (XF)")
    print(f"{'=' * 55}")

    D, K_size, T, B = 32, 4, 8, 2
    torch.manual_seed(0)
    pt_conv = TorchShortConv(
        hidden_size=D, kernel_size=K_size, bias=True, activation="silu"
    )
    jax_conv = JaxShortConv(
        hidden_size=D,
        kernel_size=K_size,
        bias=True,
        activation="silu",
        rngs=nnx.Rngs(0),
    )
    set_conv(jax_conv, pt_conv)

    x_np = np.random.RandomState(42).randn(B, T, D).astype(np.float32)

    # PyTorch step-by-step
    pt_cache = None
    pt_steps = []
    for t in range(T):
        with torch.no_grad():
            y_t, pt_cache = pt_conv.step(
                torch.tensor(x_np[:, t : t + 1, :]), pt_cache, output_final_state=True
            )
        pt_steps.append(t2n(y_t))

    # JAX step-by-step
    jax_cache = None
    jax_steps = []
    for t in range(T):
        y_t, jax_cache = jax_conv.step(
            jnp.array(x_np[:, t : t + 1, :]), jax_cache, output_final_state=True
        )
        jax_steps.append(j2n(y_t))

    pt_out = np.concatenate(pt_steps, axis=1)
    jax_out = np.concatenate(jax_steps, axis=1)

    ok = compare_arrays("ShortConv step+bias", pt_out, jax_out, 1e-5, 1e-5)
    if ok:
        print("\n  ✅ MATCH")
    return ok


# =============================================================================
# Test: PyTorch ShortConv step with cu_seqlens
# =============================================================================


def run_test_pt_short_conv_step_cu_seqlens() -> bool:
    """PyTorch ShortConv.step with cu_seqlens (squeeze(0) path)."""
    print(f"\n{'=' * 55}")
    print("PT ShortConv step with cu_seqlens")
    print(f"{'=' * 55}")

    D, K_size, N = 32, 4, 3
    cu_list = torch.LongTensor([0, 1, 2, 3])

    torch.manual_seed(0)
    pt_conv = TorchShortConv(hidden_size=D, kernel_size=K_size, activation="silu")

    # Input: [1, N, D] where each of N sequences has 1 token
    x_np = np.random.RandomState(0).randn(1, N, D).astype(np.float32)
    x_pt = torch.tensor(x_np)

    with torch.no_grad():
        y, cache = pt_conv.step(
            x_pt, cache=None, output_final_state=True, cu_seqlens=cu_list
        )

    ok = True
    if y.shape != (1, N, D):
        print(f"  ❌ output shape: {y.shape}")
        ok = False
    if cache is None:
        print("  ❌ cache should not be None")
        ok = False
    elif cache.shape != (N, D, K_size):
        print(f"  ❌ cache shape: {cache.shape} != {(N, D, K_size)}")
        ok = False
    if not torch.all(torch.isfinite(y)):
        print("  ❌ non-finite output")
        ok = False

    if ok:
        print(f"  Output: {y.shape}, Cache: {cache.shape}")
        print("  ✅ PT step with cu_seqlens (squeeze(0)) verified")
    return ok


# =============================================================================
# Test: PyTorch ShortConv short-seq final_state (T < W)
# =============================================================================


def run_test_pt_short_conv_short_seq_final_state() -> bool:
    """PyTorch ShortConv output_final_state when T < W (left-pad branch)."""
    print(f"\n{'=' * 55}")
    print("PT ShortConv short-seq final_state (T < W)")
    print(f"{'=' * 55}")

    D, K_size, B, T = 16, 4, 2, 2  # T=2 < W=4

    torch.manual_seed(0)
    pt_conv = TorchShortConv(hidden_size=D, kernel_size=K_size, activation="silu")

    x_np = np.random.RandomState(0).randn(B, T, D).astype(np.float32)
    x_pt = torch.tensor(x_np)

    with torch.no_grad():
        y, cache = pt_conv(x_pt, output_final_state=True)

    ok = True
    if y.shape != (B, T, D):
        print(f"  ❌ output shape: {y.shape}")
        ok = False
    # cache should be [B, D, W] with zero-padding on the left
    if cache is None:
        print("  ❌ cache is None")
        ok = False
    elif cache.shape != (B, D, K_size):
        print(f"  ❌ cache shape: {cache.shape} != {(B, D, K_size)}")
        ok = False
    else:
        cache_np = t2n(cache)
        # First (W - T) = 2 columns should be zero (left padding)
        pad_cols = K_size - T
        if not np.allclose(cache_np[:, :, :pad_cols], 0.0, atol=1e-8):
            print("  ❌ left padding should be zero")
            ok = False
        # Last T columns should match input x transposed
        x_t_np = x_np.transpose(0, 2, 1)  # [B, D, T]
        if not np.allclose(cache_np[:, :, pad_cols:], x_t_np, atol=1e-8):
            print("  ❌ tail should match input")
            ok = False

    if ok:
        print(f"  Cache: {cache.shape}, padding verified")
        print("  ✅ PT short-seq final_state correct")
    return ok


# =============================================================================
# Test: PyTorch ShortConv varlen final_state (cross-framework token equiv)
# =============================================================================


def run_test_short_conv_varlen_final_state_xf() -> bool:
    """Cross-framework ShortConv varlen final_state token equivalence.

    Tests the PyTorch cu_seqlens output_final_state path including
    the seg_len < W left-padding branch.
    PT format: [N, D, W], JAX format: [N, W-1, D].
    """
    print(f"\n{'=' * 55}")
    print("ShortConv varlen final_state XF")
    print(f"{'=' * 55}")

    D, K_size = 16, 4  # W = 4, W-1 = 3
    cu_list = [0, 2, 5, 12]  # segments: len 2 (< W), 3, 7
    T_total = 12

    torch.manual_seed(0)
    pt_conv = TorchShortConv(hidden_size=D, kernel_size=K_size, activation="silu")
    jax_conv = JaxShortConv(
        hidden_size=D, kernel_size=K_size, activation="silu", rngs=nnx.Rngs(0)
    )
    set_conv(jax_conv, pt_conv)

    x_np = np.random.RandomState(42).randn(1, T_total, D).astype(np.float32)

    with torch.no_grad():
        pt_y, pt_cache = pt_conv(
            torch.tensor(x_np),
            output_final_state=True,
            cu_seqlens=torch.LongTensor(cu_list),
        )
    jax_y, jax_cache = jax_conv(
        jnp.array(x_np),
        output_final_state=True,
        cu_seqlens=np.array(cu_list, dtype=np.int64),
    )

    ok = True

    # Output must match
    ok_y = compare_arrays("ShortConv varlen output", t2n(pt_y), j2n(jax_y), 1e-5, 1e-5)
    ok = ok and ok_y

    # Cache token equivalence:
    # PT cache: [N, D, W] — last W tokens per segment (in conv layout)
    # JAX cache: [N, W-1, D] — last W-1 tokens per segment
    # The last W-1 columns of PT cache transposed should match JAX cache
    N = len(cu_list) - 1
    for i in range(N):
        pt_seg = t2n(pt_cache[i])  # [D, W]
        # Last W-1 columns → transpose → [W-1, D]
        pt_tokens = pt_seg[:, 1:].T  # [W-1, D]
        jax_tokens = j2n(jax_cache[i])  # [W-1, D]
        if not np.allclose(pt_tokens, jax_tokens, atol=1e-6):
            diff = np.abs(pt_tokens - jax_tokens).max()
            print(f"  ❌ segment {i} cache tokens differ: {diff:.2e}")
            ok = False

    if ok:
        print("\n  ✅ MATCH (output + cache tokens)")
    return ok


# =============================================================================
# Test: PyTorch ShortConv forward with pre-allocated cache (cache.copy_)
# =============================================================================


def run_test_pt_short_conv_cache_copy() -> bool:
    """PyTorch ShortConv.forward with pre-allocated cache tensor.

    Tests the cache.copy_(final_state) branch at the end of forward().
    """
    print(f"\n{'=' * 55}")
    print("PT ShortConv forward with pre-allocated cache")
    print(f"{'=' * 55}")

    D, K_size, B, T = 32, 4, 2, 10

    torch.manual_seed(0)
    pt_conv = TorchShortConv(hidden_size=D, kernel_size=K_size, activation="silu")

    x_np = np.random.RandomState(0).randn(B, T, D).astype(np.float32)
    x_pt = torch.tensor(x_np)

    # Pre-allocate cache tensor
    pre_cache = torch.zeros(B, D, K_size)

    with torch.no_grad():
        # Without pre-allocated cache
        y1, cache1 = pt_conv(x_pt, output_final_state=True)
        # With pre-allocated cache
        y2, cache2 = pt_conv(x_pt, cache=pre_cache, output_final_state=True)

    ok = True
    # Output should be same
    diff_y = (y1 - y2).abs().max().item()
    if diff_y > 1e-8:
        print(f"  ❌ output diff: {diff_y:.2e}")
        ok = False

    # cache2 should be the same object as pre_cache
    if cache2 is not pre_cache:
        print("  ❌ cache2 should be same object as pre_cache (in-place copy)")
        ok = False

    # Values should match
    diff_c = (cache1 - cache2).abs().max().item()
    if diff_c > 1e-8:
        print(f"  ❌ cache diff: {diff_c:.2e}")
        ok = False

    if ok:
        print(f"  Output diff: {diff_y:.2e}, cache diff: {diff_c:.2e}")
        print("  ✅ cache.copy_ branch verified")
    return ok


# =============================================================================
# Test: chunk_gla kernel vs naive_recurrent_gla (数学等价性)
# =============================================================================


def run_test_chunk_vs_naive(
    B: int = 2,
    T: int = 64,
    H: int = 4,
    K: int = 32,
    V: int = 64,
    chunk_size: int = 16,
    use_initial_state: bool = False,
    output_final_state: bool = False,
    cu_seqlens_list: list | None = None,
) -> bool:
    """Compare chunk_gla output against naive_recurrent_gla.

    Both should produce identical results since chunk_gla is a
    mathematically equivalent reformulation of the recurrence.
    """
    label = (
        f"chunk_gla vs naive: B={B}, T={T}, H={H}, K={K}, V={V}, "
        f"C={chunk_size}, init={use_initial_state}, "
        f"final={output_final_state}, varlen={cu_seqlens_list is not None}"
    )
    print(f"\n{'=' * 55}")
    print(label)
    print(f"{'=' * 55}")

    torch.manual_seed(42)
    q = torch.randn(B, T, H, K, dtype=torch.float32)
    k = torch.randn(B, T, H, K, dtype=torch.float32)
    v = torch.randn(B, T, H, V, dtype=torch.float32)
    gk = F.logsigmoid(torch.randn(B, T, H, K, dtype=torch.float32))

    cu_seqlens = None
    if cu_seqlens_list is not None:
        assert B == 1
        cu_seqlens = torch.LongTensor(cu_seqlens_list)
        N = len(cu_seqlens_list) - 1
    else:
        N = B

    initial_state = torch.randn(N, H, K, V) if use_initial_state else None

    # --- naive recurrent ---
    o_naive, ht_naive = torch_naive_recurrent_gla(
        q=q,
        k=k,
        v=v,
        gk=gk,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )

    # --- chunk ---
    o_chunk, ht_chunk = torch_chunk_gla(
        q=q,
        k=k,
        v=v,
        g=gk,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )

    atol, rtol = get_tolerance(K, V, T, is_layer=False)
    ok = compare_arrays("Output o", t2n(o_naive), t2n(o_chunk), atol, rtol)

    if output_final_state and ht_naive is not None and ht_chunk is not None:
        ok_ht = compare_arrays("Final state", t2n(ht_naive), t2n(ht_chunk), atol, rtol)
        ok = ok and ok_ht

    if ok:
        print("\n  ✅ MATCH")
    else:
        print("\n  ❌ MISMATCH!")
    return ok


def run_test_fused_chunk_vs_naive(
    B: int = 2,
    T: int = 64,
    H: int = 4,
    K: int = 32,
    V: int = 64,
    use_initial_state: bool = False,
    output_final_state: bool = False,
    cu_seqlens_list: list | None = None,
) -> bool:
    """Compare fused_chunk_gla output against naive_recurrent_gla.

    fused_chunk_gla is a thin wrapper around chunk_gla, so this
    verifies the wrapper passes parameters correctly.
    """
    label = (
        f"fused_chunk_gla vs naive: B={B}, T={T}, H={H}, K={K}, V={V}, "
        f"init={use_initial_state}, final={output_final_state}, "
        f"varlen={cu_seqlens_list is not None}"
    )
    print(f"\n{'=' * 55}")
    print(label)
    print(f"{'=' * 55}")

    torch.manual_seed(42)
    q = torch.randn(B, T, H, K, dtype=torch.float32)
    k = torch.randn(B, T, H, K, dtype=torch.float32)
    v = torch.randn(B, T, H, V, dtype=torch.float32)
    gk = F.logsigmoid(torch.randn(B, T, H, K, dtype=torch.float32))

    cu_seqlens = None
    if cu_seqlens_list is not None:
        assert B == 1
        cu_seqlens = torch.LongTensor(cu_seqlens_list)
        N = len(cu_seqlens_list) - 1
    else:
        N = B

    initial_state = torch.randn(N, H, K, V) if use_initial_state else None

    # --- naive ---
    o_naive, ht_naive = torch_naive_recurrent_gla(
        q=q,
        k=k,
        v=v,
        gk=gk,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )

    # --- fused_chunk ---
    o_fc, ht_fc = torch_fused_chunk_gla(
        q=q,
        k=k,
        v=v,
        g=gk,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )

    atol, rtol = get_tolerance(K, V, T, is_layer=False)
    ok = compare_arrays("Output o", t2n(o_naive), t2n(o_fc), atol, rtol)

    if output_final_state and ht_naive is not None and ht_fc is not None:
        ok_ht = compare_arrays("Final state", t2n(ht_naive), t2n(ht_fc), atol, rtol)
        ok = ok and ok_ht

    if ok:
        print("\n  ✅ MATCH")
    else:
        print("\n  ❌ MISMATCH!")
    return ok


def run_test_chunk_layer_mode() -> bool:
    """Test full GLA layer with mode='chunk' uses chunk_gla kernel
    and produces same output as mode='fused_recurrent' (naive kernel).

    Since all kernels are mathematically equivalent, the layer output
    should match regardless of mode, verifying the kernel integration.
    """
    print(f"\n{'=' * 55}")
    print("Layer: chunk mode vs fused_recurrent mode")
    print(f"{'=' * 55}")

    torch.manual_seed(42)
    hidden_size, num_heads = 128, 4
    B, T = 2, 96  # T > 64 to bypass auto fused_recurrent

    x_np = np.random.RandomState(42).randn(B, T, hidden_size).astype(np.float32)

    results = {}
    for mode in ["chunk", "fused_recurrent", "fused_chunk"]:
        torch.manual_seed(42)
        model = TorchGLA(
            mode=mode,
            hidden_size=hidden_size,
            num_heads=num_heads,
            use_short_conv=False,
            use_output_gate=True,
            gate_fn="swish",
            fuse_norm=True,
        )
        model.eval()
        with torch.no_grad():
            out, _, _ = model(torch.tensor(x_np))
        results[mode] = t2n(out)

    ok = True
    diff_cr = np.abs(results["chunk"] - results["fused_recurrent"]).max()
    diff_fc = np.abs(results["chunk"] - results["fused_chunk"]).max()
    print(f"  chunk vs fused_recurrent: {diff_cr:.2e}")
    print(f"  chunk vs fused_chunk:     {diff_fc:.2e}")

    atol = 5e-5  # kernel-level tolerance
    if diff_cr > atol:
        print(f"  ❌ chunk != fused_recurrent (diff={diff_cr:.2e})")
        ok = False
    if diff_fc > atol:
        print(f"  ❌ chunk != fused_chunk (diff={diff_fc:.2e})")
        ok = False

    if ok:
        print("  ✅ All three modes produce matching output")
    return ok


# =============================================================================
# Test: RMSNorm / FusedRMSNormGated bfloat16 dtype preservation
# =============================================================================


def run_test_norm_bfloat16() -> bool:
    """RMSNorm + FusedRMSNormGated with bfloat16 input — dtype preserved XF.

    Both JAX and PyTorch norm implementations cast to float32 internally
    then cast back to input dtype. All previous tests used float32
    (making the cast a no-op). This tests the actual dtype conversion path.
    """
    print(f"\n{'=' * 55}")
    print("Norm bfloat16 dtype preservation (XF)")
    print(f"{'=' * 55}")

    torch.manual_seed(0)
    D = 64

    # --- RMSNorm ---
    pt_rms = TorchRMSNorm(D, elementwise_affine=True, eps=1e-5)
    jax_rms = JaxRMSNorm(D, elementwise_affine=True, eps=1e-5, rngs=nnx.Rngs(0))
    jax_rms.weight.value = t2j(pt_rms.weight)

    x_np = np.random.RandomState(0).randn(2, 10, D).astype(np.float32)
    x_pt_bf = torch.tensor(x_np).to(torch.bfloat16)
    x_jax_bf = jnp.array(x_np).astype(jnp.bfloat16)

    pt_y_rms = pt_rms(x_pt_bf)
    jax_y_rms = jax_rms(x_jax_bf)

    assert pt_y_rms.dtype == torch.bfloat16, f"PT dtype {pt_y_rms.dtype}"
    assert jax_y_rms.dtype == jnp.bfloat16, f"JAX dtype {jax_y_rms.dtype}"

    ok_rms = compare_arrays(
        "RMSNorm bf16",
        pt_y_rms.float().detach().numpy(),
        np.array(jax_y_rms.astype(jnp.float32)),
        atol=5e-3,
        rtol=5e-3,
    )

    # --- FusedRMSNormGated ---
    pt_fng = TorchFNG(D, elementwise_affine=True, eps=1e-5)
    jax_fng = JaxFusedRMSNormGated(
        D, elementwise_affine=True, eps=1e-5, rngs=nnx.Rngs(0)
    )
    jax_fng.weight.value = t2j(pt_fng.weight)

    g_np = np.random.RandomState(1).randn(2, 10, D).astype(np.float32)

    pt_y_fng = pt_fng(
        torch.tensor(x_np).to(torch.bfloat16),
        torch.tensor(g_np).to(torch.bfloat16),
    )
    jax_y_fng = jax_fng(
        jnp.array(x_np).astype(jnp.bfloat16),
        jnp.array(g_np).astype(jnp.bfloat16),
    )

    assert pt_y_fng.dtype == torch.bfloat16, f"PT dtype {pt_y_fng.dtype}"
    assert jax_y_fng.dtype == jnp.bfloat16, f"JAX dtype {jax_y_fng.dtype}"

    ok_fng = compare_arrays(
        "FusedRMSNormGated bf16",
        pt_y_fng.float().detach().numpy(),
        np.array(jax_y_fng.astype(jnp.float32)),
        atol=5e-3,
        rtol=5e-3,
    )

    ok = ok_rms and ok_fng
    if ok:
        print("\n  ✅ MATCH (bfloat16 dtype preserved)")
    return ok


# =============================================================================
# Test: Kernel varlen + output_final_state + no initial_state
# =============================================================================


def run_test_kernel_varlen_final_no_init() -> bool:
    """Varlen kernel with output_final_state=True but no initial_state.

    Previous tests either used (varlen + no final_state) or
    (varlen + final_state + initial_state). This tests the gap:
    initial_state=None starts each segment from zeros and collects
    final states.
    """
    print(f"\n{'=' * 55}")
    print("Kernel: varlen + final_state, no init_state")
    print(f"{'=' * 55}")

    return run_test_kernel(
        B=1,
        T=32,
        H=4,
        K=32,
        V=64,
        cu_seqlens_list=[0, 12, 32],
        use_initial_state=False,
        output_final_state=True,
    )


# =============================================================================
# Test: Layer mode='fused_chunk' and mode='fused_recurrent'
# =============================================================================


def run_test_layer_mode_variants() -> bool:
    """Test layer with explicit mode='fused_chunk' and 'fused_recurrent'.

    chunk_gla and fused_chunk_gla use a chunked algorithm that is
    mathematically equivalent to naive_recurrent_gla, but floating-point
    operation ordering differs. Small numerical differences are expected.
    """
    print(f"\n{'=' * 55}")
    print("Layer mode variants (chunk/fused_recurrent/fused_chunk)")
    print(f"{'=' * 55}")

    torch.manual_seed(42)
    hidden_size, num_heads = 128, 4
    B, T = 2, 96  # T > 64 to use the mode parameter directly

    x_np = np.random.RandomState(42).randn(B, T, hidden_size).astype(np.float32)

    results = {}
    for mode in ["chunk", "fused_recurrent", "fused_chunk"]:
        torch.manual_seed(42)
        model = TorchGLA(
            mode=mode,
            hidden_size=hidden_size,
            num_heads=num_heads,
            use_short_conv=False,
            use_output_gate=True,
            gate_fn="swish",
            fuse_norm=True,
        )
        model.eval()
        with torch.no_grad():
            out, _, _ = model(torch.tensor(x_np))
        results[mode] = t2n(out)

    ok = True
    diff_fr = np.abs(results["chunk"] - results["fused_recurrent"]).max()
    diff_fc = np.abs(results["chunk"] - results["fused_chunk"]).max()
    print(f"  chunk vs fused_recurrent: {diff_fr:.2e}")
    print(f"  chunk vs fused_chunk:     {diff_fc:.2e}")

    # chunk_gla uses different FP ordering → allow small diffs (layer-level tolerance)
    tol = 5e-5
    if diff_fr > tol:
        print(f"  ❌ chunk != fused_recurrent (diff={diff_fr:.2e} > tol={tol:.0e})")
        ok = False
    if diff_fc > tol:
        print(f"  ❌ chunk != fused_chunk (diff={diff_fc:.2e} > tol={tol:.0e})")
        ok = False

    if ok:
        print("  ✅ All three modes produce matching output")
    return ok


# =============================================================================
# Main: run all tests
# =============================================================================


def main():
    print("=" * 70)
    print("check_gla_layer.py — PyTorch vs JAX cross-framework comparison")
    print(f"JAX backend: {jax.default_backend()}")
    print("=" * 70)

    test_cases = []

    # =============================================
    # Category 1: Core kernel — basic configs
    # =============================================
    test_cases.extend(
        [
            (
                "Kernel: basic small",
                lambda: run_test_kernel(B=1, T=16, H=2, K=16, V=32),
            ),
            ("Kernel: B=2, T=32", lambda: run_test_kernel(B=2, T=32, H=4, K=32, V=64)),
            ("Kernel: B=4, T=64", lambda: run_test_kernel(B=4, T=64, H=4, K=32, V=32)),
        ]
    )

    # =============================================
    # Category 2: Core kernel — various K/V dims
    # =============================================
    test_cases.extend(
        [
            ("Kernel: K=64, V=64", lambda: run_test_kernel(B=2, T=32, H=4, K=64, V=64)),
            (
                "Kernel: K=128, V=128",
                lambda: run_test_kernel(B=1, T=32, H=2, K=128, V=128),
            ),
            (
                "Kernel: K=16, V=256",
                lambda: run_test_kernel(B=1, T=16, H=2, K=16, V=256),
            ),
        ]
    )

    # =============================================
    # Category 3: Core kernel — long sequences
    # =============================================
    test_cases.extend(
        [
            ("Kernel: T=128", lambda: run_test_kernel(B=1, T=128, H=4, K=32, V=64)),
            ("Kernel: T=256", lambda: run_test_kernel(B=1, T=256, H=2, K=32, V=32)),
            ("Kernel: T=512", lambda: run_test_kernel(B=1, T=512, H=2, K=16, V=32)),
        ]
    )

    # =============================================
    # Category 4: Core kernel — with initial_state
    # =============================================
    test_cases.extend(
        [
            (
                "Kernel: init_state",
                lambda: run_test_kernel(
                    B=2, T=32, H=4, K=32, V=64, use_initial_state=True
                ),
            ),
            (
                "Kernel: init+final state",
                lambda: run_test_kernel(
                    B=2,
                    T=32,
                    H=4,
                    K=32,
                    V=64,
                    use_initial_state=True,
                    output_final_state=True,
                ),
            ),
        ]
    )

    # =============================================
    # Category 5: Core kernel — cu_seqlens
    # =============================================
    test_cases.extend(
        [
            (
                "Kernel: varlen 2 seqs",
                lambda: run_test_kernel(
                    B=1, T=32, H=4, K=32, V=64, cu_seqlens_list=[0, 16, 32]
                ),
            ),
            (
                "Kernel: varlen 3 seqs",
                lambda: run_test_kernel(
                    B=1, T=48, H=4, K=32, V=64, cu_seqlens_list=[0, 10, 30, 48]
                ),
            ),
            (
                "Kernel: varlen + init_state",
                lambda: run_test_kernel(
                    B=1,
                    T=48,
                    H=4,
                    K=32,
                    V=64,
                    cu_seqlens_list=[0, 16, 32, 48],
                    use_initial_state=True,
                    output_final_state=True,
                ),
            ),
        ]
    )

    # =============================================
    # Category 6: Varlen equivalence (JAX side)
    # =============================================
    test_cases.extend(
        [
            (
                "Varlen equiv: T=32",
                lambda: run_test_varlen_equiv(T=32, H=4, K=32, V=64),
            ),
            (
                "Varlen equiv: T=128",
                lambda: run_test_varlen_equiv(T=128, H=2, K=64, V=64),
            ),
        ]
    )

    # =============================================
    # Category 7: State split consistency
    # =============================================
    test_cases.extend(
        [
            (
                "State split: B=2, T=64",
                lambda: run_test_state_split(B=2, T=64, H=4, K=32, V=64),
            ),
        ]
    )

    # =============================================
    # Category 8: Module-level cross-framework
    # =============================================
    test_cases.extend(
        [
            ("RMSNorm cross-framework", run_test_rmsnorm),
            ("FusedRMSNormGated cross-framework", run_test_fused_norm_gated),
        ]
    )

    # =============================================
    # Category 9: Full layer — basic
    # =============================================
    test_cases.extend(
        [
            (
                "Layer: basic (no conv, fuse_norm)",
                lambda: run_test_layer(B=2, T=32, hidden_size=128, num_heads=4),
            ),
            (
                "Layer: basic (no conv, no fuse)",
                lambda: run_test_layer(
                    B=2, T=32, hidden_size=128, num_heads=4, fuse_norm=False
                ),
            ),
            (
                "Layer: T=64",
                lambda: run_test_layer(B=1, T=64, hidden_size=128, num_heads=4),
            ),
        ]
    )

    # =============================================
    # Category 10: Full layer — with short conv
    # =============================================
    test_cases.extend(
        [
            (
                "Layer: with conv",
                lambda: run_test_layer(
                    B=2,
                    T=32,
                    hidden_size=128,
                    num_heads=4,
                    use_short_conv=True,
                    conv_size=4,
                ),
            ),
            (
                "Layer: conv + T=64",
                lambda: run_test_layer(
                    B=1, T=64, hidden_size=128, num_heads=4, use_short_conv=True
                ),
            ),
        ]
    )

    # =============================================
    # Category 11: Full layer — MQA
    # =============================================
    test_cases.extend(
        [
            (
                "Layer: MQA (kv_heads=2)",
                lambda: run_test_layer(
                    B=2, T=32, hidden_size=128, num_heads=4, num_kv_heads=2
                ),
            ),
            (
                "Layer: MQA + conv",
                lambda: run_test_layer(
                    B=2,
                    T=32,
                    hidden_size=128,
                    num_heads=4,
                    num_kv_heads=2,
                    use_short_conv=True,
                ),
            ),
        ]
    )

    # =============================================
    # Category 12: Full layer — expand ratios
    # =============================================
    test_cases.extend(
        [
            (
                "Layer: expand_k=1.0",
                lambda: run_test_layer(
                    B=2, T=32, hidden_size=128, num_heads=4, expand_k=1.0
                ),
            ),
            (
                "Layer: expand_v=2.0",
                lambda: run_test_layer(
                    B=2, T=32, hidden_size=128, num_heads=4, expand_v=2.0
                ),
            ),
        ]
    )

    # =============================================
    # Category 13: Full layer — no output gate
    # =============================================
    test_cases.extend(
        [
            (
                "Layer: no output gate",
                lambda: run_test_layer(
                    B=2,
                    T=32,
                    hidden_size=128,
                    num_heads=4,
                    use_output_gate=False,
                    fuse_norm=False,
                ),
            ),
        ]
    )

    # =============================================
    # Category 14: Full layer — gate options
    # =============================================
    test_cases.extend(
        [
            (
                "Layer: clamp_min=-0.5",
                lambda: run_test_layer(
                    B=2, T=32, hidden_size=128, num_heads=4, clamp_min=-0.5
                ),
            ),
            (
                "Layer: normalizer=8",
                lambda: run_test_layer(
                    B=2, T=32, hidden_size=128, num_heads=4, gate_logit_normalizer=8
                ),
            ),
        ]
    )

    # =============================================
    # Category 15: Full layer — edge cases
    # =============================================
    test_cases.extend(
        [
            (
                "Layer: B=1, T=1",
                lambda: run_test_layer(B=1, T=1, hidden_size=64, num_heads=2),
            ),
            (
                "Layer: B=1, T=4",
                lambda: run_test_layer(B=1, T=4, hidden_size=64, num_heads=2),
            ),
        ]
    )

    # =============================================
    # Category 16: Norm — elementwise_affine=False
    # =============================================
    test_cases.extend(
        [
            ("RMSNorm no-affine", run_test_rmsnorm_no_affine),
            ("FusedRMSNormGated no-affine", run_test_fused_norm_gated_no_affine),
        ]
    )

    # =============================================
    # Category 17: ShortConv — cross-framework
    # =============================================
    test_cases.extend(
        [
            (
                "ShortConv XF: basic silu",
                lambda: run_test_short_conv_xf(D=64, K_size=4, T=16),
            ),
            (
                "ShortConv XF: no activation",
                lambda: run_test_short_conv_xf(D=32, K_size=4, T=16, activation=None),
            ),
            (
                "ShortConv XF: with bias",
                lambda: run_test_short_conv_xf(D=32, K_size=4, T=16, bias=True),
            ),
            (
                "ShortConv XF: bias + no act",
                lambda: run_test_short_conv_xf(
                    D=32, K_size=3, T=8, activation=None, bias=True
                ),
            ),
            ("ShortConv XF: varlen", run_test_short_conv_varlen_xf),
        ]
    )

    # =============================================
    # Category 18: ShortConv — JAX consistency
    # =============================================
    test_cases.extend(
        [
            ("ShortConv: cache prefix", run_test_short_conv_cache),
            ("ShortConv: short seq final_state", run_test_short_conv_short_seq),
            ("ShortConv: step vs full", run_test_short_conv_step),
        ]
    )

    # =============================================
    # Category 19: Kernel — custom scale & final_state
    # =============================================
    test_cases.extend(
        [
            (
                "Kernel: custom scale=0.1",
                lambda: run_test_kernel(B=2, T=32, H=4, K=32, V=64, scale=0.1),
            ),
            (
                "Kernel: final_state only",
                lambda: run_test_kernel(
                    B=2, T=32, H=4, K=32, V=64, output_final_state=True
                ),
            ),
        ]
    )

    # =============================================
    # Category 20: Layer — feature_map
    # =============================================
    test_cases.extend(
        [
            (
                "Layer: feature_map=relu",
                lambda: run_test_layer(
                    B=2, T=32, hidden_size=128, num_heads=4, feature_map="relu"
                ),
            ),
        ]
    )

    # =============================================
    # Category 21: Layer — gate_fn variants
    # =============================================
    test_cases.extend(
        [
            (
                "Layer: gate_fn=sigmoid",
                lambda: run_test_layer(
                    B=2,
                    T=32,
                    hidden_size=128,
                    num_heads=4,
                    gate_fn="sigmoid",
                    fuse_norm=False,
                ),
            ),
            (
                "Layer: gate_fn=relu",
                lambda: run_test_layer(
                    B=2,
                    T=32,
                    hidden_size=128,
                    num_heads=4,
                    gate_fn="relu",
                    fuse_norm=False,
                ),
            ),
        ]
    )

    # =============================================
    # Category 22: Layer — T > 64 (mode switch)
    # =============================================
    test_cases.extend(
        [
            (
                "Layer: T=128 (chunk mode)",
                lambda: run_test_layer(B=1, T=128, hidden_size=128, num_heads=4),
            ),
            (
                "Layer: T=96 + conv",
                lambda: run_test_layer(
                    B=1, T=96, hidden_size=128, num_heads=4, use_short_conv=True
                ),
            ),
        ]
    )

    # =============================================
    # Category 23: Layer — cu_seqlens
    # =============================================
    test_cases.extend(
        [
            (
                "Layer: cu_seqlens (no conv)",
                lambda: run_test_layer_cu_seqlens(use_short_conv=False),
            ),
            (
                "Layer: cu_seqlens + conv",
                lambda: run_test_layer_cu_seqlens(use_short_conv=True),
            ),
        ]
    )

    # =============================================
    # Category 24: ShortConv step cross-framework
    # =============================================
    test_cases.extend(
        [
            ("ShortConv: step XF", run_test_short_conv_step_xf),
        ]
    )

    # =============================================
    # Category 25: Varlen conv short-segment cache
    # =============================================
    test_cases.extend(
        [
            ("ShortConv: varlen short seg cache", run_test_short_conv_varlen_short_seg),
        ]
    )

    # =============================================
    # Category 26: Kernel edge — H=1, seg_len=1
    # =============================================
    test_cases.extend(
        [
            (
                "Kernel: H=1 single head",
                lambda: run_test_kernel(B=2, T=16, H=1, K=32, V=64),
            ),
            (
                "Kernel: varlen seg_len=1",
                lambda: run_test_kernel(
                    B=1,
                    T=5,
                    H=2,
                    K=16,
                    V=32,
                    cu_seqlens_list=[0, 1, 3, 5],
                    use_initial_state=True,
                    output_final_state=True,
                ),
            ),
        ]
    )

    # =============================================
    # Category 27: Layer — conv_bias, affine, low_rank
    # =============================================
    test_cases.extend(
        [
            (
                "Layer: conv_bias=True",
                lambda: run_test_layer(
                    B=2,
                    T=32,
                    hidden_size=128,
                    num_heads=4,
                    use_short_conv=True,
                    conv_size=4,
                    conv_bias=True,
                ),
            ),
            (
                "Layer: norm no-affine",
                lambda: run_test_layer(
                    B=2, T=32, hidden_size=128, num_heads=4, elementwise_affine=False
                ),
            ),
            (
                "Layer: gate_low_rank_dim=8",
                lambda: run_test_layer(
                    B=2, T=32, hidden_size=128, num_heads=4, gate_low_rank_dim=8
                ),
            ),
        ]
    )

    # =============================================
    # Category 28: ShortConv final_state XF + swish alias
    # =============================================
    test_cases.extend(
        [
            ("ShortConv: final_state XF", run_test_short_conv_final_state_xf),
            ("ShortConv: swish alias", run_test_short_conv_swish_alias),
        ]
    )

    # =============================================
    # Category 29: PyTorch attention_mask pipeline
    # =============================================
    test_cases.extend(
        [
            ("PT: attention_mask pipeline", run_test_attention_mask_pipeline),
        ]
    )

    # =============================================
    # Category 30: PyTorch ShortConv auto-step
    # =============================================
    test_cases.extend(
        [
            ("PT: ShortConv auto-step", run_test_pt_short_conv_auto_step),
        ]
    )

    # =============================================
    # Category 31: ACT2FN completeness (gelu, tanh)
    # =============================================
    test_cases.extend(
        [
            (
                "Layer: gate_fn=gelu",
                lambda: run_test_layer(
                    B=2,
                    T=32,
                    hidden_size=128,
                    num_heads=4,
                    gate_fn="gelu",
                    fuse_norm=False,
                ),
            ),
            (
                "Layer: gate_fn=tanh",
                lambda: run_test_layer(
                    B=2,
                    T=32,
                    hidden_size=128,
                    num_heads=4,
                    gate_fn="tanh",
                    fuse_norm=False,
                ),
            ),
        ]
    )

    # =============================================
    # Category 32: feature_map variants
    # =============================================
    test_cases.extend(
        [
            (
                "Layer: feature_map=tanh",
                lambda: run_test_layer(
                    B=2, T=32, hidden_size=128, num_heads=4, feature_map="tanh"
                ),
            ),
            (
                "Layer: feature_map=gelu",
                lambda: run_test_layer(
                    B=2, T=32, hidden_size=128, num_heads=4, feature_map="gelu"
                ),
            ),
        ]
    )

    # =============================================
    # Category 33: Layer no-gate + fuse_norm=True
    # =============================================
    test_cases.extend(
        [
            (
                "Layer: no gate + fuse_norm=True",
                lambda: run_test_layer(
                    B=2,
                    T=32,
                    hidden_size=128,
                    num_heads=4,
                    use_output_gate=False,
                    fuse_norm=True,
                ),
            ),
        ]
    )

    # =============================================
    # Category 34: ShortConv step edge cases
    # =============================================
    test_cases.extend(
        [
            (
                "ShortConv: step no-cache no-output",
                run_test_short_conv_step_no_cache_no_output,
            ),
            ("ShortConv: step discard cache", run_test_short_conv_step_discard_cache),
            ("PT: step no-cache else branch", run_test_pt_short_conv_step_no_cache),
        ]
    )

    # =============================================
    # Category 35: ShortConv step with bias (XF)
    # =============================================
    test_cases.extend(
        [
            ("ShortConv: step+bias XF", run_test_short_conv_step_bias_xf),
        ]
    )

    # =============================================
    # Category 36: PT ShortConv step cu_seqlens
    # =============================================
    test_cases.extend(
        [
            ("PT: step cu_seqlens", run_test_pt_short_conv_step_cu_seqlens),
        ]
    )

    # =============================================
    # Category 37: PT ShortConv final_state edge
    # =============================================
    test_cases.extend(
        [
            ("PT: short-seq final_state", run_test_pt_short_conv_short_seq_final_state),
            (
                "ShortConv: varlen final_state XF",
                run_test_short_conv_varlen_final_state_xf,
            ),
            ("PT: cache.copy_ branch", run_test_pt_short_conv_cache_copy),
        ]
    )

    # =============================================
    # Category 38: Layer mode variants
    # =============================================
    test_cases.extend(
        [
            ("PT: mode variants", run_test_layer_mode_variants),
        ]
    )

    # =============================================
    # Category 39: chunk_gla vs naive — basic
    # =============================================
    test_cases.extend(
        [
            (
                "chunk_gla vs naive: basic",
                lambda: run_test_chunk_vs_naive(B=2, T=64, H=4, K=32, V=64),
            ),
            (
                "chunk_gla vs naive: C=8",
                lambda: run_test_chunk_vs_naive(
                    B=2, T=64, H=4, K=32, V=64, chunk_size=8
                ),
            ),
            (
                "chunk_gla vs naive: C=32",
                lambda: run_test_chunk_vs_naive(
                    B=2, T=64, H=4, K=32, V=64, chunk_size=32
                ),
            ),
            (
                "chunk_gla vs naive: C=64 (1 chunk)",
                lambda: run_test_chunk_vs_naive(
                    B=2, T=64, H=4, K=32, V=64, chunk_size=64
                ),
            ),
        ]
    )

    # =============================================
    # Category 40: chunk_gla — non-aligned T
    # =============================================
    test_cases.extend(
        [
            (
                "chunk_gla vs naive: T=50 (unaligned)",
                lambda: run_test_chunk_vs_naive(
                    B=2, T=50, H=4, K=32, V=64, chunk_size=16
                ),
            ),
            (
                "chunk_gla vs naive: T=17 C=8",
                lambda: run_test_chunk_vs_naive(
                    B=1, T=17, H=2, K=16, V=32, chunk_size=8
                ),
            ),
            (
                "chunk_gla vs naive: T=1 (single token)",
                lambda: run_test_chunk_vs_naive(
                    B=2, T=1, H=4, K=32, V=64, chunk_size=16
                ),
            ),
        ]
    )

    # =============================================
    # Category 41: chunk_gla — state management
    # =============================================
    test_cases.extend(
        [
            (
                "chunk_gla vs naive: init+final state",
                lambda: run_test_chunk_vs_naive(
                    B=2,
                    T=64,
                    H=4,
                    K=32,
                    V=64,
                    use_initial_state=True,
                    output_final_state=True,
                ),
            ),
            (
                "chunk_gla vs naive: init only",
                lambda: run_test_chunk_vs_naive(
                    B=2, T=32, H=4, K=32, V=64, use_initial_state=True
                ),
            ),
            (
                "chunk_gla vs naive: final only",
                lambda: run_test_chunk_vs_naive(
                    B=2, T=32, H=4, K=32, V=64, output_final_state=True
                ),
            ),
        ]
    )

    # =============================================
    # Category 42: chunk_gla — varlen (cu_seqlens)
    # =============================================
    test_cases.extend(
        [
            (
                "chunk_gla vs naive: varlen 2 seg",
                lambda: run_test_chunk_vs_naive(
                    B=1, T=32, H=4, K=32, V=64, cu_seqlens_list=[0, 16, 32]
                ),
            ),
            (
                "chunk_gla vs naive: varlen 3 seg",
                lambda: run_test_chunk_vs_naive(
                    B=1, T=48, H=4, K=32, V=64, cu_seqlens_list=[0, 10, 30, 48]
                ),
            ),
            (
                "chunk_gla vs naive: varlen + init+final",
                lambda: run_test_chunk_vs_naive(
                    B=1,
                    T=48,
                    H=4,
                    K=32,
                    V=64,
                    cu_seqlens_list=[0, 16, 32, 48],
                    use_initial_state=True,
                    output_final_state=True,
                ),
            ),
        ]
    )

    # =============================================
    # Category 43: chunk_gla — larger dims
    # =============================================
    test_cases.extend(
        [
            (
                "chunk_gla vs naive: K=64 V=128",
                lambda: run_test_chunk_vs_naive(B=1, T=64, H=2, K=64, V=128),
            ),
            (
                "chunk_gla vs naive: T=128 C=16",
                lambda: run_test_chunk_vs_naive(
                    B=1, T=128, H=4, K=32, V=64, chunk_size=16
                ),
            ),
        ]
    )

    # =============================================
    # Category 44: fused_chunk_gla vs naive
    # =============================================
    test_cases.extend(
        [
            (
                "fused_chunk vs naive: basic",
                lambda: run_test_fused_chunk_vs_naive(B=2, T=64, H=4, K=32, V=64),
            ),
            (
                "fused_chunk vs naive: init+final",
                lambda: run_test_fused_chunk_vs_naive(
                    B=2,
                    T=64,
                    H=4,
                    K=32,
                    V=64,
                    use_initial_state=True,
                    output_final_state=True,
                ),
            ),
            (
                "fused_chunk vs naive: varlen",
                lambda: run_test_fused_chunk_vs_naive(
                    B=1, T=32, H=4, K=32, V=64, cu_seqlens_list=[0, 12, 32]
                ),
            ),
        ]
    )

    # =============================================
    # Category 45: Layer mode chunk/fused_chunk integration
    # =============================================
    test_cases.extend(
        [
            ("Layer: chunk vs fused_recurrent mode", run_test_chunk_layer_mode),
        ]
    )

    # =============================================
    # Category 46: Norm bfloat16 dtype preservation
    # =============================================
    test_cases.extend(
        [
            ("Norm: bfloat16 dtype XF", run_test_norm_bfloat16),
        ]
    )

    # =============================================
    # Category 47: Kernel varlen final_state w/o init
    # =============================================
    test_cases.extend(
        [
            ("Kernel: varlen final no init", run_test_kernel_varlen_final_no_init),
        ]
    )

    # =============================================
    # Category 48: feature_map='sigmoid'
    # =============================================
    test_cases.extend(
        [
            (
                "Layer: feature_map=sigmoid",
                lambda: run_test_layer(
                    B=2, T=32, hidden_size=128, num_heads=4, feature_map="sigmoid"
                ),
            ),
        ]
    )

    # =============================================
    # Category 49: MQA + cu_seqlens
    # =============================================
    test_cases.extend(
        [
            (
                "Layer: MQA + cu_seqlens",
                lambda: run_test_layer_cu_seqlens(use_short_conv=False, num_kv_heads=2),
            ),
        ]
    )

    # =============================================
    # Category 50: expand_k + expand_v both non-default
    # =============================================
    test_cases.extend(
        [
            (
                "Layer: expand_k=1.0 + expand_v=2.0",
                lambda: run_test_layer(
                    B=2, T=32, hidden_size=128, num_heads=4, expand_k=1.0, expand_v=2.0
                ),
            ),
        ]
    )

    # =============================================
    # Category 51: conv + feature_map combo
    # =============================================
    test_cases.extend(
        [
            (
                "Layer: conv + feature_map=relu",
                lambda: run_test_layer(
                    B=2,
                    T=32,
                    hidden_size=128,
                    num_heads=4,
                    use_short_conv=True,
                    feature_map="relu",
                ),
            ),
        ]
    )

    # =============================================
    # Category 52: MQA + no output_gate
    # =============================================
    test_cases.extend(
        [
            (
                "Layer: MQA + no gate",
                lambda: run_test_layer(
                    B=2,
                    T=32,
                    hidden_size=128,
                    num_heads=4,
                    num_kv_heads=2,
                    use_output_gate=False,
                    fuse_norm=False,
                ),
            ),
        ]
    )

    # =============================================
    # Category 53: T=65 boundary (mode switch)
    # =============================================
    test_cases.extend(
        [
            (
                "Layer: T=65 mode boundary",
                lambda: run_test_layer(B=1, T=65, hidden_size=128, num_heads=4),
            ),
        ]
    )

    # Run all tests
    all_passed = True
    passed_count = 0
    total_count = len(test_cases)

    for i, (name, test_fn) in enumerate(test_cases):
        print(f"\n{'#' * 70}")
        print(f"Test {i + 1}/{total_count}: {name}")
        print(f"{'#' * 70}")
        try:
            ok = test_fn()
            if ok:
                passed_count += 1
            else:
                all_passed = False
        except Exception:
            print("  ❌ Exception:")
            traceback.print_exc()
            all_passed = False

    print(f"\n{'=' * 70}")
    print(f"Summary: {passed_count}/{total_count} passed")
    print(f"{'=' * 70}")
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
    else:
        print(f"❌ {total_count - passed_count} test(s) FAILED")

    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

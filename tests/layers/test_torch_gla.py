# =============================================================================
# Triton GLA 全面测试: Triton (FLA CUDA) vs Torch_CPU (local pure PyTorch)
#
# Triton 版: flash-linear-attention (FLA) 库, 基于 Triton CUDA kernel
# Torch_CPU 版: tests/src/ 纯 PyTorch CPU 实现 (代码布局和函数结构与 Triton 版一致)
#
# 测试策略:
#   Part A - Kernel 级 Triton 函数测试:
#     naive_recurrent_gla / chunk_gla / fused_recurrent_gla
#   Part B - Module 级 Triton 函数测试:
#     RMSNorm / FusedRMSNormGated / ShortConvolution
#   Part C - Layer 级 Triton 函数测试:
#     GatedLinearAttention (各种配置)
#   Part D - 架构一致性:
#     state_dict keys / 参数量 / 属性 / 权重迁移
#   Part E - Torch_CPU 独立验证:
#     内部等价性 / 独立正确性 / 梯度 / 数值稳定性
# =============================================================================
from __future__ import annotations

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn.functional as F
from einops import rearrange

# --- Triton implementations (FLA library, CUDA) ---
from fla.ops.gla.naive import naive_recurrent_gla as triton_naive_recurrent_gla
from fla.ops.gla import chunk_gla as triton_chunk_gla
from fla.ops.gla import fused_recurrent_gla as triton_fused_recurrent_gla
from fla.layers.gla import GatedLinearAttention as TritonGLA
from fla.modules import RMSNorm as TritonRMSNorm
from fla.modules import FusedRMSNormGated as TritonFNG
from fla.modules import ShortConvolution as TritonShortConv

# --- Torch_CPU implementations (local, pure PyTorch) ---
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
from tests.src.layers.gla import GatedLinearAttention as CpuGLA
from tests.src.layers.utils import get_unpad_data, index_first_axis, pad_input
from tests.src.modules.layernorm import RMSNorm as CpuRMSNorm
from tests.src.modules.fused_norm_gate import FusedRMSNormGated as CpuFNG
from tests.src.modules.convolution import ShortConvolution as CpuShortConv
from tests.utils import compare_tensor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# Helpers
# =============================================================================


def to_gpu(sd: dict) -> dict:
    return {k: v.to(DEVICE) for k, v in sd.items()}


def to_cpu(sd: dict) -> dict:
    return {k: v.cpu() for k, v in sd.items()}


# =============================================================================
# Part A: Triton Kernel Tests (vs Torch_CPU reference)
# =============================================================================

# --- A1: naive_recurrent_gla ---


def test_triton_naive_basic() -> bool:
    """Triton naive vs Torch_CPU naive: basic shapes."""
    print("\n[Triton Kernel] naive: basic (B=2, T=32, H=4, K=32, V=64)")
    torch.manual_seed(42)
    B, T, H, K, V = 2, 32, 4, 32, 64
    q = torch.randn(B, T, H, K)
    k = torch.randn(B, T, H, K)
    v = torch.randn(B, T, H, V)
    gk = F.logsigmoid(torch.randn(B, T, H, K))

    o_triton, _ = triton_naive_recurrent_gla(q, k, v, gk)
    o_cpu, _ = cpu_naive_recurrent_gla(q, k, v, gk)
    return compare_tensor("output", o_triton, o_cpu)


def test_triton_naive_large() -> bool:
    """Triton naive vs Torch_CPU naive: larger dims."""
    print("\n[Triton Kernel] naive: large (B=1, T=128, H=2, K=64, V=128)")
    torch.manual_seed(7)
    B, T, H, K, V = 1, 128, 2, 64, 128
    q = torch.randn(B, T, H, K)
    k = torch.randn(B, T, H, K)
    v = torch.randn(B, T, H, V)
    gk = F.logsigmoid(torch.randn(B, T, H, K))

    o_triton, _ = triton_naive_recurrent_gla(q, k, v, gk)
    o_cpu, _ = cpu_naive_recurrent_gla(q, k, v, gk)
    return compare_tensor("output", o_triton, o_cpu, atol=5e-5, rtol=5e-5)


def test_triton_naive_initial_state() -> bool:
    """Triton naive vs Torch_CPU naive: with initial state."""
    print("\n[Triton Kernel] naive: initial + final state (B=2, T=64, H=4, K=32, V=64)")
    torch.manual_seed(13)
    B, T, H, K, V = 2, 64, 4, 32, 64
    q = torch.randn(B, T, H, K)
    k = torch.randn(B, T, H, K)
    v = torch.randn(B, T, H, V)
    gk = F.logsigmoid(torch.randn(B, T, H, K))
    h0 = torch.randn(B, H, K, V)

    o_triton, s_triton = triton_naive_recurrent_gla(
        q, k, v, gk, initial_state=h0, output_final_state=True
    )
    o_cpu, s_cpu = cpu_naive_recurrent_gla(
        q, k, v, gk, initial_state=h0, output_final_state=True
    )
    ok = compare_tensor("output", o_triton, o_cpu)
    ok &= compare_tensor("final_state", s_triton, s_cpu)
    return ok


def test_triton_naive_various_shapes() -> bool:
    """Triton naive vs Torch_CPU naive: multiple shape combos."""
    print("\n[Triton Kernel] naive: various shapes")
    torch.manual_seed(99)
    ok = True
    for B, T, H, K, V in [
        (1, 16, 1, 16, 16),
        (4, 64, 8, 32, 64),
        (1, 256, 2, 128, 128),
    ]:
        q = torch.randn(B, T, H, K)
        k = torch.randn(B, T, H, K)
        v = torch.randn(B, T, H, V)
        gk = F.logsigmoid(torch.randn(B, T, H, K))
        o_triton, _ = triton_naive_recurrent_gla(q, k, v, gk)
        o_cpu, _ = cpu_naive_recurrent_gla(q, k, v, gk)
        atol = 1e-4 if K > 64 or T > 128 else 1e-5
        ok &= compare_tensor(
            f"B={B} T={T} H={H} K={K} V={V}", o_triton, o_cpu, atol=atol, rtol=atol
        )
    return ok


def test_triton_naive_state_split() -> bool:
    """Triton naive vs Torch_CPU naive: split sequence consistency."""
    print("\n[Triton Kernel] naive: state split (process in 2 halves)")
    torch.manual_seed(77)
    B, T, H, K, V = 1, 40, 2, 16, 32
    q = torch.randn(B, T, H, K)
    k = torch.randn(B, T, H, K)
    v = torch.randn(B, T, H, V)
    gk = F.logsigmoid(torch.randn(B, T, H, K))

    o_full_triton, s_full_triton = triton_naive_recurrent_gla(
        q, k, v, gk, output_final_state=True
    )
    o_full_cpu, s_full_cpu = cpu_naive_recurrent_gla(
        q, k, v, gk, output_final_state=True
    )

    T1 = T // 2
    _, s1_triton = triton_naive_recurrent_gla(
        q[:, :T1], k[:, :T1], v[:, :T1], gk[:, :T1], output_final_state=True
    )
    o2_triton, s2_triton = triton_naive_recurrent_gla(
        q[:, T1:],
        k[:, T1:],
        v[:, T1:],
        gk[:, T1:],
        initial_state=s1_triton,
        output_final_state=True,
    )
    _, s1_cpu = cpu_naive_recurrent_gla(
        q[:, :T1], k[:, :T1], v[:, :T1], gk[:, :T1], output_final_state=True
    )
    o2_cpu, s2_cpu = cpu_naive_recurrent_gla(
        q[:, T1:],
        k[:, T1:],
        v[:, T1:],
        gk[:, T1:],
        initial_state=s1_cpu,
        output_final_state=True,
    )

    ok = compare_tensor("full output (triton vs cpu)", o_full_triton, o_full_cpu)
    ok &= compare_tensor("full state (triton vs cpu)", s_full_triton, s_full_cpu)
    ok &= compare_tensor("split-2nd output (triton vs cpu)", o2_triton, o2_cpu)
    ok &= compare_tensor("split-2nd state (triton vs cpu)", s2_triton, s2_cpu)
    ok &= compare_tensor("triton: full vs split state", s_full_triton, s2_triton)
    ok &= compare_tensor("cpu: full vs split state", s_full_cpu, s2_cpu)
    return ok


# --- A2: chunk_gla (CUDA) ---


def test_triton_chunk_basic() -> bool:
    """Triton chunk_gla (CUDA) vs Torch_CPU chunk_gla."""
    print("\n[Triton Kernel] chunk_gla (GPU) vs Torch_CPU (CPU)")
    torch.manual_seed(42)
    B, T, H, K, V = 2, 64, 4, 32, 64
    q = torch.randn(B, T, H, K)
    k = torch.randn(B, T, H, K)
    v = torch.randn(B, T, H, V)
    gk = F.logsigmoid(torch.randn(B, T, H, K))

    o_triton, s_triton = triton_chunk_gla(
        q.to(DEVICE), k.to(DEVICE), v.to(DEVICE), gk.to(DEVICE), output_final_state=True
    )
    o_cpu, s_cpu = cpu_chunk_gla(q, k, v, gk, output_final_state=True)
    ok = compare_tensor("output", o_triton.cpu(), o_cpu, atol=2e-2, rtol=2e-2)
    ok &= compare_tensor("final_state", s_triton.cpu(), s_cpu, atol=2e-2, rtol=2e-2)
    return ok


def test_triton_chunk_init_state() -> bool:
    """Triton chunk_gla (CUDA) vs Torch_CPU with initial state."""
    print("\n[Triton Kernel] chunk_gla vs Torch_CPU: initial state")
    torch.manual_seed(13)
    B, T, H, K, V = 2, 64, 4, 32, 64
    q = torch.randn(B, T, H, K)
    k = torch.randn(B, T, H, K)
    v = torch.randn(B, T, H, V)
    gk = F.logsigmoid(torch.randn(B, T, H, K))
    h0 = torch.randn(B, H, K, V)

    o_triton, s_triton = triton_chunk_gla(
        q.to(DEVICE),
        k.to(DEVICE),
        v.to(DEVICE),
        gk.to(DEVICE),
        initial_state=h0.to(DEVICE),
        output_final_state=True,
    )
    o_cpu, s_cpu = cpu_chunk_gla(q, k, v, gk, initial_state=h0, output_final_state=True)
    ok = compare_tensor("output", o_triton.cpu(), o_cpu, atol=2e-2, rtol=2e-2)
    ok &= compare_tensor("final_state", s_triton.cpu(), s_cpu, atol=2e-2, rtol=2e-2)
    return ok


# --- A3: fused_recurrent_gla (CUDA) ---


def test_triton_fused_recurrent_basic() -> bool:
    """Triton fused_recurrent_gla (CUDA) vs Torch_CPU naive."""
    print("\n[Triton Kernel] fused_recurrent_gla (GPU) vs Torch_CPU naive (CPU)")
    torch.manual_seed(42)
    B, T, H, K, V = 2, 32, 4, 32, 64
    q = torch.randn(B, T, H, K)
    k = torch.randn(B, T, H, K)
    v = torch.randn(B, T, H, V)
    gk = F.logsigmoid(torch.randn(B, T, H, K))

    o_triton, s_triton = triton_fused_recurrent_gla(
        q.to(DEVICE), k.to(DEVICE), v.to(DEVICE), gk.to(DEVICE), output_final_state=True
    )
    o_cpu, s_cpu = cpu_naive_recurrent_gla(q, k, v, gk, output_final_state=True)
    ok = compare_tensor("output", o_triton.cpu(), o_cpu, atol=1e-4, rtol=1e-4)
    ok &= compare_tensor("final_state", s_triton.cpu(), s_cpu, atol=1e-4, rtol=1e-4)
    return ok


def test_triton_fused_recurrent_init_state() -> bool:
    """Triton fused_recurrent_gla (CUDA) vs Torch_CPU with initial state."""
    print("\n[Triton Kernel] fused_recurrent_gla vs Torch_CPU: initial state")
    torch.manual_seed(13)
    B, T, H, K, V = 2, 32, 4, 32, 64
    q = torch.randn(B, T, H, K)
    k = torch.randn(B, T, H, K)
    v = torch.randn(B, T, H, V)
    gk = F.logsigmoid(torch.randn(B, T, H, K))
    h0 = torch.randn(B, H, K, V)

    o_triton, s_triton = triton_fused_recurrent_gla(
        q.to(DEVICE),
        k.to(DEVICE),
        v.to(DEVICE),
        gk.to(DEVICE),
        initial_state=h0.to(DEVICE),
        output_final_state=True,
    )
    o_cpu, s_cpu = cpu_naive_recurrent_gla(
        q, k, v, gk, initial_state=h0, output_final_state=True
    )
    ok = compare_tensor("output", o_triton.cpu(), o_cpu, atol=1e-4, rtol=1e-4)
    ok &= compare_tensor("final_state", s_triton.cpu(), s_cpu, atol=1e-4, rtol=1e-4)
    return ok


# =============================================================================
# Part B: Triton Module Tests (vs Torch_CPU reference)
# =============================================================================


def test_triton_rmsnorm() -> bool:
    """Triton RMSNorm (CUDA) vs Torch_CPU RMSNorm."""
    print("\n[Triton Module] RMSNorm vs Torch_CPU")
    torch.manual_seed(0)
    dim = 64
    triton_norm = TritonRMSNorm(dim, elementwise_affine=True, eps=1e-5).to(DEVICE)
    cpu_norm = CpuRMSNorm(dim, elementwise_affine=True, eps=1e-5)
    cpu_norm.load_state_dict(to_cpu(triton_norm.state_dict()))
    triton_norm.eval()
    cpu_norm.eval()

    x = torch.randn(2, 10, dim)
    with torch.no_grad():
        y_triton = triton_norm(x.to(DEVICE)).cpu()
        y_cpu = cpu_norm(x)
    return compare_tensor("RMSNorm", y_triton, y_cpu, atol=1e-5, rtol=1e-5)


def test_triton_fused_norm_gated() -> bool:
    """Triton FusedRMSNormGated (CUDA) vs Torch_CPU."""
    print("\n[Triton Module] FusedRMSNormGated vs Torch_CPU")
    torch.manual_seed(0)
    dim = 64
    triton_fn = TritonFNG(dim, elementwise_affine=True, eps=1e-5).to(DEVICE)
    cpu_fn = CpuFNG(dim, elementwise_affine=True, eps=1e-5)
    cpu_fn.load_state_dict(to_cpu(triton_fn.state_dict()))
    triton_fn.eval()
    cpu_fn.eval()

    x = torch.randn(2, 10, dim)
    g = torch.randn(2, 10, dim)
    with torch.no_grad():
        y_triton = triton_fn(x.to(DEVICE), g.to(DEVICE)).cpu()
        y_cpu = cpu_fn(x, g)
    return compare_tensor("FusedRMSNormGated", y_triton, y_cpu, atol=1e-5, rtol=1e-5)


def test_triton_short_conv() -> bool:
    """Triton ShortConvolution (CUDA) vs Torch_CPU."""
    print("\n[Triton Module] ShortConvolution vs Torch_CPU")
    torch.manual_seed(42)
    triton_conv = TritonShortConv(
        hidden_size=32, kernel_size=4, bias=True, activation="silu"
    ).to(DEVICE)
    cpu_conv = CpuShortConv(hidden_size=32, kernel_size=4, bias=True, activation="silu")
    cpu_conv.load_state_dict(to_cpu(triton_conv.state_dict()))
    triton_conv.eval()
    cpu_conv.eval()

    x = torch.randn(2, 16, 32)
    with torch.no_grad():
        y_triton, _ = triton_conv(x.to(DEVICE))
        y_cpu, _ = cpu_conv(x)
    return compare_tensor(
        "ShortConv output", y_triton.cpu(), y_cpu, atol=1e-5, rtol=1e-5
    )


def test_triton_short_conv_cache() -> bool:
    """Triton ShortConvolution (CUDA) vs Torch_CPU: cache output."""
    print("\n[Triton Module] ShortConvolution vs Torch_CPU: cache")
    torch.manual_seed(42)
    triton_conv = TritonShortConv(
        hidden_size=16, kernel_size=4, bias=True, activation="silu"
    ).to(DEVICE)
    cpu_conv = CpuShortConv(hidden_size=16, kernel_size=4, bias=True, activation="silu")
    cpu_conv.load_state_dict(to_cpu(triton_conv.state_dict()))
    triton_conv.eval()
    cpu_conv.eval()

    x = torch.randn(1, 8, 16)
    with torch.no_grad():
        y_triton, cache_triton = triton_conv(x.to(DEVICE), output_final_state=True)
        y_cpu, cache_cpu = cpu_conv(x, output_final_state=True)
    ok = compare_tensor("output", y_triton.cpu(), y_cpu, atol=1e-5, rtol=1e-5)
    ok &= compare_tensor("cache", cache_triton.cpu(), cache_cpu, atol=1e-5, rtol=1e-5)
    return ok


# =============================================================================
# Part C: Triton Layer Tests (vs Torch_CPU reference)
# =============================================================================


def test_triton_layer_basic() -> bool:
    """Triton GLA layer (CUDA) vs Torch_CPU: basic forward."""
    print("\n[Triton Layer] vs Torch_CPU: basic (B=2, T=32, D=128, H=2)")
    torch.manual_seed(42)
    cfg = dict(
        hidden_size=128,
        num_heads=2,
        use_short_conv=False,
        use_output_gate=True,
        fuse_norm=True,
        layer_idx=0,
    )
    triton_m = TritonGLA(**cfg).to(DEVICE)
    cpu_m = CpuGLA(**cfg)
    cpu_m.load_state_dict(to_cpu(triton_m.state_dict()))
    triton_m.eval()
    cpu_m.eval()

    x = torch.randn(2, 32, 128)
    with torch.no_grad():
        o_triton = triton_m(x.to(DEVICE))[0].cpu()
        o_cpu = cpu_m(x)[0]
    return compare_tensor("output", o_triton, o_cpu, atol=1e-4, rtol=1e-4)


def test_triton_layer_conv() -> bool:
    """Triton GLA layer (CUDA) vs Torch_CPU: with ShortConvolution."""
    print("\n[Triton Layer] vs Torch_CPU: with ShortConvolution")
    torch.manual_seed(42)
    cfg = dict(
        hidden_size=128,
        num_heads=2,
        use_short_conv=True,
        conv_size=4,
        use_output_gate=True,
        fuse_norm=True,
        layer_idx=0,
    )
    triton_m = TritonGLA(**cfg).to(DEVICE)
    cpu_m = CpuGLA(**cfg)
    cpu_m.load_state_dict(to_cpu(triton_m.state_dict()))
    triton_m.eval()
    cpu_m.eval()

    x = torch.randn(2, 32, 128)
    with torch.no_grad():
        o_triton = triton_m(x.to(DEVICE))[0].cpu()
        o_cpu = cpu_m(x)[0]
    return compare_tensor("output", o_triton, o_cpu, atol=1e-4, rtol=1e-4)


def test_triton_layer_mqa() -> bool:
    """Triton GLA layer (CUDA) vs Torch_CPU: MQA config."""
    print("\n[Triton Layer] vs Torch_CPU: MQA (H=8, KV=2)")
    torch.manual_seed(42)
    cfg = dict(
        hidden_size=256,
        num_heads=8,
        num_kv_heads=2,
        use_output_gate=True,
        fuse_norm=True,
        layer_idx=0,
    )
    triton_m = TritonGLA(**cfg).to(DEVICE)
    cpu_m = CpuGLA(**cfg)
    cpu_m.load_state_dict(to_cpu(triton_m.state_dict()))
    triton_m.eval()
    cpu_m.eval()

    x = torch.randn(2, 64, 256)
    with torch.no_grad():
        o_triton = triton_m(x.to(DEVICE))[0].cpu()
        o_cpu = cpu_m(x)[0]
    return compare_tensor("output", o_triton, o_cpu, atol=1e-4, rtol=1e-4)


def test_triton_layer_no_gate() -> bool:
    """Triton GLA layer (CUDA) vs Torch_CPU: no output gate."""
    print("\n[Triton Layer] vs Torch_CPU: no output gate")
    torch.manual_seed(42)
    cfg = dict(
        hidden_size=128,
        num_heads=2,
        use_output_gate=False,
        fuse_norm=False,
        layer_idx=0,
    )
    triton_m = TritonGLA(**cfg).to(DEVICE)
    cpu_m = CpuGLA(**cfg)
    cpu_m.load_state_dict(to_cpu(triton_m.state_dict()))
    triton_m.eval()
    cpu_m.eval()

    x = torch.randn(2, 32, 128)
    with torch.no_grad():
        o_triton = triton_m(x.to(DEVICE))[0].cpu()
        o_cpu = cpu_m(x)[0]
    return compare_tensor("output", o_triton, o_cpu, atol=1e-4, rtol=1e-4)


def test_triton_layer_expand() -> bool:
    """Triton GLA layer (CUDA) vs Torch_CPU: non-default expand_k/expand_v."""
    print("\n[Triton Layer] vs Torch_CPU: expand_k/expand_v combos")
    ok = True
    for ek, ev in [(1.0, 1.0), (0.25, 2.0)]:
        torch.manual_seed(42)
        cfg = dict(
            hidden_size=128,
            num_heads=4,
            expand_k=ek,
            expand_v=ev,
            use_output_gate=True,
            fuse_norm=True,
            layer_idx=0,
        )
        triton_m = TritonGLA(**cfg).to(DEVICE)
        cpu_m = CpuGLA(**cfg)
        cpu_m.load_state_dict(to_cpu(triton_m.state_dict()))
        triton_m.eval()
        cpu_m.eval()

        x = torch.randn(1, 32, 128)
        with torch.no_grad():
            o_triton = triton_m(x.to(DEVICE))[0].cpu()
            o_cpu = cpu_m(x)[0]
        ok &= compare_tensor(f"ek={ek} ev={ev}", o_triton, o_cpu, atol=1e-4, rtol=1e-4)
    return ok


def test_triton_layer_mask() -> bool:
    """Triton GLA layer (CUDA) vs Torch_CPU: attention_mask."""
    print("\n[Triton Layer] vs Torch_CPU: attention_mask")
    torch.manual_seed(42)
    cfg = dict(
        hidden_size=128, num_heads=2, use_output_gate=True, fuse_norm=True, layer_idx=0
    )
    triton_m = TritonGLA(**cfg).to(DEVICE)
    cpu_m = CpuGLA(**cfg)
    cpu_m.load_state_dict(to_cpu(triton_m.state_dict()))
    triton_m.eval()
    cpu_m.eval()

    B, T, D = 2, 32, 128
    x = torch.randn(B, T, D)
    mask = torch.ones(B, T, dtype=torch.long)
    mask[0, -8:] = 0
    mask[1, -4:] = 0

    with torch.no_grad():
        o_triton = triton_m(x.to(DEVICE), attention_mask=mask.to(DEVICE))[0].cpu()
        o_cpu = cpu_m(x, attention_mask=mask)[0]
    return compare_tensor("output", o_triton, o_cpu, atol=1e-4, rtol=1e-4)


def test_triton_layer_long_seq() -> bool:
    """Triton GLA layer (CUDA) vs Torch_CPU: longer sequence (T=256)."""
    print("\n[Triton Layer] vs Torch_CPU: long seq (T=256)")
    torch.manual_seed(7)
    cfg = dict(
        hidden_size=128, num_heads=2, use_output_gate=True, fuse_norm=True, layer_idx=0
    )
    triton_m = TritonGLA(**cfg).to(DEVICE)
    cpu_m = CpuGLA(**cfg)
    cpu_m.load_state_dict(to_cpu(triton_m.state_dict()))
    triton_m.eval()
    cpu_m.eval()

    x = torch.randn(1, 256, 128)
    with torch.no_grad():
        o_triton = triton_m(x.to(DEVICE))[0].cpu()
        o_cpu = cpu_m(x)[0]
    return compare_tensor("output", o_triton, o_cpu, atol=5e-4, rtol=5e-4)


def test_triton_layer_clamp() -> bool:
    """Triton GLA layer (CUDA) vs Torch_CPU: clamp_min."""
    print("\n[Triton Layer] vs Torch_CPU: clamp_min=-1.0")
    torch.manual_seed(42)
    cfg = dict(
        hidden_size=128,
        num_heads=2,
        clamp_min=-1.0,
        use_output_gate=True,
        fuse_norm=True,
        layer_idx=0,
    )
    triton_m = TritonGLA(**cfg).to(DEVICE)
    cpu_m = CpuGLA(**cfg)
    cpu_m.load_state_dict(to_cpu(triton_m.state_dict()))
    triton_m.eval()
    cpu_m.eval()

    x = torch.randn(1, 32, 128)
    with torch.no_grad():
        o_triton = triton_m(x.to(DEVICE))[0].cpu()
        o_cpu = cpu_m(x)[0]
    return compare_tensor("output", o_triton, o_cpu, atol=1e-4, rtol=1e-4)


# =============================================================================
# Part D: Architecture Parity (Triton vs Torch_CPU)
# =============================================================================


def test_arch_state_dict_keys() -> bool:
    """Triton vs Torch_CPU: state_dict keys match across configs."""
    print("\n[Arch] state_dict keys match across configs")
    configs = [
        dict(
            hidden_size=128,
            num_heads=2,
            use_short_conv=False,
            use_output_gate=True,
            fuse_norm=True,
            layer_idx=0,
        ),
        dict(
            hidden_size=128,
            num_heads=2,
            use_short_conv=True,
            conv_size=4,
            use_output_gate=True,
            fuse_norm=True,
            layer_idx=0,
        ),
        dict(
            hidden_size=128,
            num_heads=2,
            use_short_conv=False,
            use_output_gate=False,
            fuse_norm=False,
            layer_idx=0,
        ),
        dict(
            hidden_size=256,
            num_heads=8,
            num_kv_heads=2,
            use_output_gate=True,
            fuse_norm=True,
            layer_idx=0,
        ),
        dict(
            hidden_size=128,
            num_heads=4,
            expand_k=1.0,
            expand_v=2.0,
            use_output_gate=True,
            fuse_norm=True,
            layer_idx=0,
        ),
    ]
    ok = True
    for i, cfg in enumerate(configs):
        triton_m = TritonGLA(**cfg)
        cpu_m = CpuGLA(**cfg)
        triton_keys = sorted(triton_m.state_dict().keys())
        cpu_keys = sorted(cpu_m.state_dict().keys())
        match = triton_keys == cpu_keys
        ok &= match
        label = ", ".join(f"{k}={v}" for k, v in cfg.items() if k != "layer_idx")
        print(f"  {'[PASS]' if match else '[FAIL]'} cfg{i}({label})")
        if not match:
            print(f"      Triton only: {set(triton_keys) - set(cpu_keys)}")
            print(f"      CPU only: {set(cpu_keys) - set(triton_keys)}")
    return ok


def test_arch_weight_transfer() -> bool:
    """Triton <-> Torch_CPU: bidirectional weight transfer."""
    print("\n[Arch] bidirectional weight transfer")
    torch.manual_seed(100)
    cfg = dict(
        hidden_size=256,
        num_heads=4,
        use_short_conv=True,
        conv_size=4,
        use_output_gate=True,
        fuse_norm=True,
        layer_idx=0,
    )
    triton_m = TritonGLA(**cfg)
    cpu_m = CpuGLA(**cfg)

    cpu_m.load_state_dict(triton_m.state_dict())
    for k in triton_m.state_dict():
        assert torch.equal(triton_m.state_dict()[k], cpu_m.state_dict()[k])
    print("  [PASS] Triton -> CPU: all params identical")

    cpu_m2 = CpuGLA(**cfg)
    triton_m.load_state_dict(cpu_m2.state_dict())
    for k in cpu_m2.state_dict():
        assert torch.equal(triton_m.state_dict()[k], cpu_m2.state_dict()[k])
    print("  [PASS] CPU -> Triton: all params identical")
    return True


def test_arch_attributes() -> bool:
    """Triton vs Torch_CPU: key attributes/config match."""
    print("\n[Arch] attribute parity")
    cfg = dict(
        hidden_size=256,
        num_heads=8,
        num_kv_heads=2,
        expand_k=0.5,
        expand_v=1.0,
        gate_logit_normalizer=16,
        clamp_min=-1.0,
        use_output_gate=True,
        fuse_norm=True,
        layer_idx=0,
    )
    triton_m = TritonGLA(**cfg)
    cpu_m = CpuGLA(**cfg)
    attrs = [
        "hidden_size",
        "num_heads",
        "num_kv_heads",
        "num_kv_groups",
        "key_dim",
        "value_dim",
        "head_k_dim",
        "head_v_dim",
        "key_dim_per_group",
        "value_dim_per_group",
        "expand_k",
        "expand_v",
        "gate_logit_normalizer",
        "clamp_min",
        "use_output_gate",
        "use_short_conv",
        "fuse_norm_and_gate",
    ]
    ok = True
    for attr in attrs:
        triton_val = getattr(triton_m, attr, "MISSING")
        cpu_val = getattr(cpu_m, attr, "MISSING")
        match = triton_val == cpu_val
        ok &= match
        if not match:
            print(f"  [FAIL] {attr}: Triton={triton_val} vs CPU={cpu_val}")
    if ok:
        print(f"  [PASS] all {len(attrs)} attributes match")
    return ok


def test_arch_param_count() -> bool:
    """Triton vs Torch_CPU: parameter count match."""
    print("\n[Arch] parameter count match")
    configs = [
        dict(hidden_size=256, num_heads=4, layer_idx=0),
        dict(
            hidden_size=256, num_heads=4, use_short_conv=True, conv_size=4, layer_idx=0
        ),
        dict(hidden_size=256, num_heads=8, num_kv_heads=2, layer_idx=0),
        dict(hidden_size=512, num_heads=8, expand_k=1.0, expand_v=2.0, layer_idx=0),
    ]
    ok = True
    for i, cfg in enumerate(configs):
        triton_m = TritonGLA(**cfg)
        cpu_m = CpuGLA(**cfg)
        triton_n = sum(p.numel() for p in triton_m.parameters())
        cpu_n = sum(p.numel() for p in cpu_m.parameters())
        match = triton_n == cpu_n
        ok &= match
        print(
            f"  {'[PASS]' if match else '[FAIL]'} cfg{i}: Triton={triton_n:,} CPU={cpu_n:,} params"
        )
    return ok


# =============================================================================
# Part E: Torch_CPU Internal Tests
# =============================================================================

# --- E1: Internal equivalence (chunk vs naive, fused_chunk vs naive) ---


def test_cpu_chunk_vs_naive() -> bool:
    """Torch_CPU chunk_gla vs naive_recurrent_gla equivalence."""
    print("\n[CPU] chunk vs naive: basic (B=2, T=64, H=4, K=32, V=64)")
    torch.manual_seed(42)
    B, T, H, K, V = 2, 64, 4, 32, 64
    q = torch.randn(B, T, H, K)
    k = torch.randn(B, T, H, K)
    v = torch.randn(B, T, H, V)
    gk = F.logsigmoid(torch.randn(B, T, H, K))

    o_naive, s_naive = cpu_naive_recurrent_gla(q, k, v, gk, output_final_state=True)
    o_chunk, s_chunk = cpu_chunk_gla(q, k, v, gk, output_final_state=True)
    ok = compare_tensor("output", o_naive, o_chunk, atol=5e-5, rtol=5e-5)
    ok &= compare_tensor("final_state", s_naive, s_chunk, atol=5e-5, rtol=5e-5)
    return ok


def test_cpu_chunk_vs_naive_init_state() -> bool:
    """Torch_CPU chunk vs naive with initial state."""
    print("\n[CPU] chunk vs naive: initial+final state")
    torch.manual_seed(13)
    B, T, H, K, V = 2, 32, 4, 32, 64
    q = torch.randn(B, T, H, K)
    k = torch.randn(B, T, H, K)
    v = torch.randn(B, T, H, V)
    gk = F.logsigmoid(torch.randn(B, T, H, K))
    h0 = torch.randn(B, H, K, V)

    o_naive, s_naive = cpu_naive_recurrent_gla(
        q, k, v, gk, initial_state=h0, output_final_state=True
    )
    o_chunk, s_chunk = cpu_chunk_gla(
        q, k, v, gk, initial_state=h0, output_final_state=True
    )
    ok = compare_tensor("output", o_naive, o_chunk, atol=5e-5, rtol=5e-5)
    ok &= compare_tensor("final_state", s_naive, s_chunk, atol=5e-5, rtol=5e-5)
    return ok


def test_cpu_chunk_vs_naive_varlen() -> bool:
    """Torch_CPU chunk vs naive with cu_seqlens."""
    print("\n[CPU] chunk vs naive: varlen cu_seqlens")
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
    return compare_tensor("output", o_naive, o_chunk, atol=5e-5, rtol=5e-5)


def test_cpu_fused_chunk_vs_naive() -> bool:
    """Torch_CPU fused_chunk_gla vs naive equivalence."""
    print("\n[CPU] fused_chunk vs naive")
    torch.manual_seed(42)
    B, T, H, K, V = 2, 64, 4, 32, 64
    q = torch.randn(B, T, H, K)
    k = torch.randn(B, T, H, K)
    v = torch.randn(B, T, H, V)
    gk = F.logsigmoid(torch.randn(B, T, H, K))

    o_naive, _ = cpu_naive_recurrent_gla(q, k, v, gk)
    o_fc, _ = cpu_fused_chunk_gla(q, k, v, gk)
    return compare_tensor("output", o_naive, o_fc, atol=5e-5, rtol=5e-5)


# --- E2: cu_seqlens consistency ---


def test_cpu_cu_seqlens_vs_separate() -> bool:
    """cu_seqlens packed == separate batch processing."""
    print("\n[CPU] cu_seqlens packed vs separate batches")
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
    o_cu, s_cu = cpu_naive_recurrent_gla(
        q_cat, k_cat, v_cat, g_cat, output_final_state=True, cu_seqlens=cu
    )

    ok = compare_tensor("seg1 output", o1, o_cu[:, :s1_len])
    ok &= compare_tensor("seg2 output", o2, o_cu[:, s1_len:])
    ok &= compare_tensor("seg1 state", s1.squeeze(0), s_cu[0])
    ok &= compare_tensor("seg2 state", s2.squeeze(0), s_cu[1])
    return ok


# --- E3: Module standalone ---


def test_cpu_rmsnorm_standalone() -> bool:
    """Torch_CPU RMSNorm standalone correctness."""
    print("\n[CPU] RMSNorm standalone correctness")
    torch.manual_seed(0)
    dim = 64
    norm = CpuRMSNorm(dim, elementwise_affine=True, eps=1e-5)
    x = torch.randn(2, 10, dim)
    y = norm(x)
    x_f = x.float()
    rms = torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + 1e-5)
    y_exp = (x_f * rms * norm.weight.float()).to(x.dtype)
    ok = compare_tensor("RMSNorm vs manual", y, y_exp, atol=1e-6, rtol=1e-6)
    norm2 = CpuRMSNorm(dim, elementwise_affine=False)
    y2 = norm2(x)
    assert y2.shape == x.shape, f"Shape mismatch: {y2.shape}"
    print("  [PASS] no-affine shape OK")
    return ok


def test_cpu_fused_norm_gated_standalone() -> bool:
    """Torch_CPU FusedRMSNormGated standalone correctness."""
    print("\n[CPU] FusedRMSNormGated standalone correctness")
    torch.manual_seed(0)
    dim = 64
    fnorm = CpuFNG(dim, elementwise_affine=True, eps=1e-5)
    x = torch.randn(2, 10, dim)
    g = torch.randn(2, 10, dim)
    y = fnorm(x, g)
    x_f, g_f = x.float(), g.float()
    rms = torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + 1e-5)
    y_exp = (x_f * rms * fnorm.weight.float() * F.silu(g_f)).to(x.dtype)
    return compare_tensor("FNG vs manual", y, y_exp, atol=1e-6, rtol=1e-6)


def test_cpu_short_conv_causal() -> bool:
    """Torch_CPU ShortConvolution: causality property."""
    print("\n[CPU] ShortConv causality verification")
    conv = CpuShortConv(hidden_size=8, kernel_size=3, bias=True, activation="silu")
    x1 = torch.randn(1, 16, 8)
    x2 = x1.clone()
    x2[:, 10:, :] = torch.randn(1, 6, 8)
    y1, _ = conv(x1)
    y2, _ = conv(x2)
    ok = torch.allclose(y1[:, :10], y2[:, :10], atol=1e-6)
    differs = not torch.allclose(y1[:, 10:], y2[:, 10:], atol=1e-6)
    if ok and differs:
        print("  [PASS] causal: future changes don't affect past outputs")
    else:
        print("  [FAIL] causality violated")
    return ok and differs


def test_cpu_short_conv_cu_seqlens() -> bool:
    """Torch_CPU ShortConvolution: cu_seqlens packed == separate."""
    print("\n[CPU] ShortConv cu_seqlens vs separate")
    conv = CpuShortConv(hidden_size=8, kernel_size=3, bias=False, activation="silu")
    x_s1 = torch.randn(1, 6, 8)
    x_s2 = torch.randn(1, 10, 8)
    y_s1, _ = conv(x_s1)
    y_s2, _ = conv(x_s2)
    x_packed = torch.cat([x_s1, x_s2], dim=1)
    cu = torch.tensor([0, 6, 16], dtype=torch.long)
    y_packed, _ = conv(x_packed, cu_seqlens=cu)
    ok = compare_tensor("seg1", y_s1, y_packed[:, :6])
    ok &= compare_tensor("seg2", y_s2, y_packed[:, 6:])
    return ok


def test_cpu_short_conv_step() -> bool:
    """Torch_CPU ShortConvolution: step (single-token decode with cache)."""
    print("\n[CPU] ShortConv step decode with cache")
    conv = CpuShortConv(hidden_size=16, kernel_size=4, bias=True, activation="silu")
    x_pre = torch.randn(1, 8, 16)
    y_pre, cache = conv(x_pre, output_final_state=True)
    assert cache is not None and cache.shape == (1, 16, 4), (
        f"Cache shape: {cache.shape}"
    )
    x_dec = torch.randn(1, 1, 16)
    y_dec, cache_dec = conv.step(x_dec, cache, output_final_state=True)
    assert y_dec.shape == (1, 1, 16)
    assert cache_dec.shape == (1, 16, 4)
    print(f"  [PASS] prefill cache: {cache.shape}, decode: {y_dec.shape}")
    return True


# --- E4: Layer standalone ---


def test_cpu_layer_basic() -> bool:
    """Torch_CPU GLA layer: basic forward."""
    print("\n[CPU Layer] basic forward (B=2, T=32, H=4, D=256)")
    torch.manual_seed(42)
    model = CpuGLA(
        mode="chunk",
        hidden_size=256,
        num_heads=4,
        use_output_gate=True,
        fuse_norm=True,
        layer_idx=0,
    )
    model.eval()
    x = torch.randn(2, 32, 256)
    with torch.no_grad():
        o, _, _ = model(x)
    assert o.shape == x.shape
    print(f"  [PASS] output shape: {o.shape}")
    return True


def test_cpu_layer_feature_map() -> bool:
    """Torch_CPU GLA layer: feature_map='relu'."""
    print("\n[CPU Layer] feature_map=relu")
    model = CpuGLA(
        hidden_size=128,
        num_heads=2,
        feature_map="relu",
        use_output_gate=True,
        fuse_norm=True,
        layer_idx=0,
    )
    model.eval()
    x = torch.randn(1, 16, 128)
    with torch.no_grad():
        o, _, _ = model(x)
    assert o.shape == x.shape
    print(f"  [PASS] output shape: {o.shape}")
    return True


def test_cpu_layer_normalizer() -> bool:
    """Different gate_logit_normalizer values produce different outputs."""
    print("\n[CPU Layer] gate_logit_normalizer effect")
    torch.manual_seed(200)
    m16 = CpuGLA(
        hidden_size=128,
        num_heads=2,
        gate_logit_normalizer=16,
        use_output_gate=True,
        fuse_norm=True,
        layer_idx=0,
    )
    m1 = CpuGLA(
        hidden_size=128,
        num_heads=2,
        gate_logit_normalizer=1,
        use_output_gate=True,
        fuse_norm=True,
        layer_idx=0,
    )
    m1.load_state_dict(m16.state_dict())
    x = torch.randn(1, 16, 128)
    with torch.no_grad():
        o16, _, _ = m16(x)
        o1, _, _ = m1(x)
    ok = not torch.allclose(o16, o1, atol=1e-3)
    print(
        f"  {'[PASS]' if ok else '[FAIL]'} different normalizers -> different outputs"
    )
    return ok


def test_cpu_layer_seq1() -> bool:
    """Torch_CPU GLA layer: single token (T=1)."""
    print("\n[CPU Layer] T=1 single token")
    model = CpuGLA(
        mode="chunk",
        hidden_size=256,
        num_heads=4,
        use_output_gate=True,
        fuse_norm=True,
        layer_idx=0,
    )
    model.eval()
    x = torch.randn(2, 1, 256)
    with torch.no_grad():
        o, _, _ = model(x)
    assert o.shape == x.shape
    print(f"  [PASS] output shape: {o.shape}")
    return True


# --- E5: Gradient tests ---


def test_cpu_grad_basic() -> bool:
    """Gradient backward pass."""
    print("\n[CPU Grad] basic backward")
    model = CpuGLA(
        mode="chunk",
        hidden_size=256,
        num_heads=4,
        use_output_gate=True,
        fuse_norm=True,
        layer_idx=0,
    )
    x = torch.randn(2, 32, 256, requires_grad=True)
    o, _, _ = model(x)
    o.sum().backward()
    assert x.grad is not None and x.grad.shape == x.shape
    n_grads = sum(
        1 for p in model.parameters() if p.requires_grad and p.grad is not None
    )
    n_params = sum(1 for p in model.parameters() if p.requires_grad)
    ok = n_grads == n_params
    print(
        f"  {'[PASS]' if ok else '[FAIL]'} {n_grads}/{n_params} params have gradients"
    )
    return ok


def test_cpu_grad_conv() -> bool:
    """Gradient backward with ShortConvolution."""
    print("\n[CPU Grad] backward with ShortConvolution")
    model = CpuGLA(
        mode="chunk",
        hidden_size=256,
        num_heads=4,
        use_short_conv=True,
        conv_size=4,
        use_output_gate=True,
        fuse_norm=True,
        layer_idx=0,
    )
    x = torch.randn(2, 32, 256, requires_grad=True)
    o, _, _ = model(x)
    o.sum().backward()
    ok = x.grad is not None
    print(f"  {'[PASS]' if ok else '[FAIL]'} input grad computed")
    return ok


def test_cpu_grad_mqa() -> bool:
    """Gradient backward with MQA."""
    print("\n[CPU Grad] backward with MQA")
    model = CpuGLA(
        mode="chunk",
        hidden_size=256,
        num_heads=8,
        num_kv_heads=2,
        use_output_gate=True,
        fuse_norm=True,
        layer_idx=0,
    )
    x = torch.randn(2, 32, 256, requires_grad=True)
    o, _, _ = model(x)
    o.sum().backward()
    ok = x.grad is not None
    print(f"  {'[PASS]' if ok else '[FAIL]'} input grad computed")
    return ok


# --- E6: Numerical stability & determinism ---


def test_cpu_numerical_stability() -> bool:
    """No inf/nan with large/small inputs."""
    print("\n[CPU Stability] large/small inputs")
    model = CpuGLA(
        hidden_size=128, num_heads=2, use_output_gate=True, fuse_norm=True, layer_idx=0
    )
    model.eval()
    ok = True
    for scale, label in [(10.0, "large"), (0.001, "small")]:
        x = torch.randn(1, 16, 128) * scale
        with torch.no_grad():
            o, _, _ = model(x)
        finite = torch.isfinite(o).all().item()
        ok &= finite
        print(f"  {'[PASS]' if finite else '[FAIL]'} {label} inputs: finite={finite}")
    return ok


def test_cpu_determinism() -> bool:
    """Same seed -> same output."""
    print("\n[CPU Determinism] reproducibility")
    torch.manual_seed(42)
    m1 = CpuGLA(
        hidden_size=128, num_heads=2, use_output_gate=True, fuse_norm=True, layer_idx=0
    )
    x1 = torch.randn(1, 16, 128)
    with torch.no_grad():
        o1, _, _ = m1(x1)
    torch.manual_seed(42)
    m2 = CpuGLA(
        hidden_size=128, num_heads=2, use_output_gate=True, fuse_norm=True, layer_idx=0
    )
    x2 = torch.randn(1, 16, 128)
    with torch.no_grad():
        o2, _, _ = m2(x2)
    ok = torch.allclose(o1, o2, atol=1e-7)
    print(f"  {'[PASS]' if ok else '[FAIL]'} deterministic outputs")
    return ok


# --- E7: Utilities ---


def test_cpu_unpad_pad_roundtrip() -> bool:
    """get_unpad_data / index_first_axis / pad_input roundtrip."""
    print("\n[CPU Util] unpad/pad roundtrip")
    B, T, D = 3, 20, 32
    mask = torch.ones(B, T, dtype=torch.long)
    mask[0, 15:] = 0
    mask[1, 18:] = 0
    mask[2, 10:] = 0
    indices, cu, max_len = get_unpad_data(mask)
    x = torch.randn(B, T, D)
    x_flat = rearrange(x, "b s d -> (b s) d")
    x_packed = index_first_axis(x_flat, indices)
    x_restored = pad_input(x_packed, indices, B, T)

    ok = True
    for b in range(B):
        vl = mask[b].sum().item()
        ok &= torch.allclose(x[b, :vl], x_restored[b, :vl], atol=1e-7)
    ok &= (x_restored[0, 15:] == 0).all().item()
    ok &= (x_restored[2, 10:] == 0).all().item()
    print(
        f"  {'[PASS]' if ok else '[FAIL]'} packed {indices.shape[0]} tokens, restored correctly"
    )
    return ok


# =============================================================================
# Part F: Sub-function Tests (chunk decomposition)
# =============================================================================

# --- F1: chunk_local_cumsum ---


def test_cpu_chunk_local_cumsum() -> bool:
    """Verify chunk-local cumsum against manual computation."""
    print("\n[CPU Sub] chunk_local_cumsum correctness")
    torch.manual_seed(42)
    B, H, K, C = 2, 4, 8, 4
    NT = 3
    T = NT * C

    g = torch.randn(B, T, H, K)

    result = cpu_chunk_local_cumsum(g, chunk_size=C)

    # Manual: reshape, cumsum within each chunk, reshape back
    g_chunks = g.view(B, NT, C, H, K)
    expected = g_chunks.cumsum(dim=2).view(B, T, H, K)

    ok = compare_tensor("cumsum", result, expected, atol=1e-6, rtol=1e-6)
    # Verify chunk boundaries: cumsum resets at each chunk start
    for n in range(1, NT):
        first_in_chunk = result[:, n * C]
        raw_g = g[:, n * C]
        ok &= compare_tensor(
            f"chunk {n} reset", first_in_chunk, raw_g, atol=1e-7, rtol=1e-7
        )
    return ok


# --- F2: chunk_fwd_h ---


def test_cpu_chunk_fwd_h() -> bool:
    """Verify inter-chunk hidden state: final state matches naive_recurrent_gla."""
    print("\n[CPU Sub] chunk_fwd_h final state vs naive_recurrent_gla")
    torch.manual_seed(42)
    B, T, H, K, V = 2, 32, 4, 16, 32
    C = 16

    q = torch.randn(B, T, H, K)
    k = torch.randn(B, T, H, K)
    v = torch.randn(B, T, H, V)
    gk = torch.randn(B, T, H, K) * 0.1

    # Naive final state
    _, naive_ht = cpu_naive_recurrent_gla(q, k, v, gk, output_final_state=True)

    # chunk_fwd_h needs chunk-local cumsum of gates
    gk_f = gk.float()
    g_cumsum = cpu_chunk_local_cumsum(gk_f, chunk_size=C)
    h_all, chunk_ht = cpu_chunk_fwd_h(
        k.float(),
        v.float(),
        g_cumsum,
        h0=None,
        output_final_state=True,
        chunk_size=C,
    )

    ok = compare_tensor("final state", chunk_ht, naive_ht.float(), atol=1e-4, rtol=1e-4)
    # h_all shape check
    NT = T // C
    assert h_all.shape == (B, NT, H, K, V), f"h_all shape: {h_all.shape}"
    print(f"  [PASS] h_all shape: {h_all.shape}")
    return ok


# --- F3: chunk_gla_fwd_intra_gk ---


def test_cpu_chunk_gla_fwd_intra_gk() -> bool:
    """Verify intra-chunk attention matrix via manual q*exp(g) @ (k*exp(-g))^T."""
    print("\n[CPU Sub] chunk_gla_fwd_intra_gk correctness")
    torch.manual_seed(42)
    B, H, K, C = 1, 2, 8, 4
    NT = 2
    T = NT * C
    scale = K**-0.5

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
    A_manual = torch.einsum("bnihk,bnjhk->bnhij", q_gated, k_gated)
    causal_mask = torch.tril(torch.ones(C, C, dtype=torch.bool))
    A_manual = A_manual.masked_fill(~causal_mask, 0.0)

    ok = compare_tensor("A matrix", A, A_manual, atol=1e-5, rtol=1e-5)
    # Check causality: upper triangle should be zero
    upper = A[:, :, :, 0, -1]  # q_pos=0, k_pos=last → should be 0
    ok &= (upper.abs() < 1e-10).all().item()
    print(f"  {'[PASS]' if ok else '[FAIL]'} upper triangle is zero (causal)")
    return ok


# --- F4: chunk_gla_fwd_o_gk ---


def test_cpu_chunk_gla_fwd_o_gk() -> bool:
    """Verify output given pre-computed A and h."""
    print("\n[CPU Sub] chunk_gla_fwd_o_gk correctness")
    torch.manual_seed(42)
    B, T, H, K, V = 1, 16, 2, 8, 16
    C = 8
    scale = K**-0.5

    q = torch.randn(B, T, H, K).float()
    k = torch.randn(B, T, H, K).float()
    v = torch.randn(B, T, H, V).float()
    g = torch.randn(B, T, H, K).float() * 0.1

    g_cumsum = cpu_chunk_local_cumsum(g, chunk_size=C)
    h_all, _ = cpu_chunk_fwd_h(k, v, g_cumsum, chunk_size=C)
    A = cpu_chunk_gla_fwd_intra_gk(q, k, g_cumsum, scale, chunk_size=C)
    o = cpu_chunk_gla_fwd_o_gk(q, v, g_cumsum, A, h_all, scale, chunk_size=C)

    # compare_tensor with full chunk_gla
    o_ref, _ = cpu_chunk_gla(
        q.to(torch.float32),
        k.to(torch.float32),
        v.to(torch.float32),
        g.to(torch.float32),
        scale=scale,
        chunk_size=C,
    )

    ok = compare_tensor("output", o, o_ref.float(), atol=1e-5, rtol=1e-5)
    return ok


# --- F5: chunk_gla_fwd orchestrator ---


def test_cpu_chunk_gla_fwd() -> bool:
    """Orchestrator chunk_gla_fwd vs chunk_gla results."""
    print("\n[CPU Sub] chunk_gla_fwd orchestrator vs chunk_gla")
    torch.manual_seed(42)
    B, T, H, K, V = 2, 50, 4, 16, 32
    C = 16
    scale = K**-0.5

    q = torch.randn(B, T, H, K)
    k = torch.randn(B, T, H, K)
    v = torch.randn(B, T, H, V)
    gk = torch.randn(B, T, H, K) * 0.1
    h0 = torch.randn(B, H, K, V) * 0.01

    o_ref, ht_ref = cpu_chunk_gla(
        q,
        k,
        v,
        gk,
        scale=scale,
        initial_state=h0,
        output_final_state=True,
        chunk_size=C,
    )

    _, _, _, ht_fwd, o_fwd = cpu_chunk_gla_fwd(
        q.float(),
        k.float(),
        v.float(),
        gk.float(),
        g_cumsum=None,
        scale=scale,
        initial_state=h0,
        output_final_state=True,
        chunk_size=C,
    )

    ok = compare_tensor("output", o_fwd, o_ref.float(), atol=1e-5, rtol=1e-5)
    ok &= compare_tensor("final state", ht_fwd, ht_ref, atol=1e-5, rtol=1e-5)
    return ok


# --- F6: fused_recurrent_gla ---


def test_cpu_fused_recurrent_gla() -> bool:
    """New fused_recurrent_gla vs naive_recurrent_gla."""
    print("\n[CPU Sub] fused_recurrent_gla vs naive_recurrent_gla")
    torch.manual_seed(42)
    B, T, H, K, V = 2, 32, 4, 16, 32

    q = torch.randn(B, T, H, K)
    k = torch.randn(B, T, H, K)
    v = torch.randn(B, T, H, V)
    gk = torch.randn(B, T, H, K) * 0.1
    h0 = torch.randn(B, H, K, V) * 0.01

    o_naive, ht_naive = cpu_naive_recurrent_gla(
        q,
        k,
        v,
        gk,
        initial_state=h0,
        output_final_state=True,
    )

    o_fused, ht_fused = cpu_fused_recurrent_gla(
        q,
        k,
        v,
        gk=gk,
        initial_state=h0,
        output_final_state=True,
    )

    ok = compare_tensor(
        "output", o_fused.float(), o_naive.float(), atol=1e-5, rtol=1e-5
    )
    ok &= compare_tensor("final state", ht_fused, ht_naive, atol=1e-5, rtol=1e-5)
    return ok


# --- F7: Sub-functions compose ---


def test_cpu_sub_functions_compose() -> bool:
    """Manual composition of 4 sub-functions == chunk_gla."""
    print("\n[CPU Sub] manual sub-function composition vs chunk_gla")
    torch.manual_seed(42)
    B, T, H, K, V = 2, 48, 4, 16, 32
    C = 16
    scale = K**-0.5

    q = torch.randn(B, T, H, K).float()
    k = torch.randn(B, T, H, K).float()
    v = torch.randn(B, T, H, V).float()
    g = torch.randn(B, T, H, K).float() * 0.1
    h0 = torch.randn(B, H, K, V).float() * 0.01

    # Reference
    o_ref, ht_ref = cpu_chunk_gla(
        q, k, v, g, scale=scale, initial_state=h0, output_final_state=True, chunk_size=C
    )

    # Manual composition (T=48 is already a multiple of C=16, no padding needed)
    g_cumsum = cpu_chunk_local_cumsum(g, C)
    h_all, ht = cpu_chunk_fwd_h(
        k, v, g_cumsum, h0=h0, output_final_state=True, chunk_size=C
    )
    A = cpu_chunk_gla_fwd_intra_gk(q, k, g_cumsum, scale, chunk_size=C)
    o = cpu_chunk_gla_fwd_o_gk(q, v, g_cumsum, A, h_all, scale, chunk_size=C)

    ok = compare_tensor("composed output", o, o_ref.float(), atol=1e-5, rtol=1e-5)
    ok &= compare_tensor("composed final state", ht, ht_ref, atol=1e-5, rtol=1e-5)
    return ok


# =============================================================================
# Main
# =============================================================================


def main() -> bool:
    print("=" * 70)
    print(f"Triton GLA Tests: Triton (CUDA={DEVICE}) vs Torch_CPU")
    print("=" * 70)

    test_cases = [
        # Part A: Triton Kernel Tests
        ("Triton naive: basic", test_triton_naive_basic),
        ("Triton naive: large dims", test_triton_naive_large),
        ("Triton naive: initial state", test_triton_naive_initial_state),
        ("Triton naive: various shapes", test_triton_naive_various_shapes),
        ("Triton naive: state split", test_triton_naive_state_split),
        ("Triton chunk_gla: basic", test_triton_chunk_basic),
        ("Triton chunk_gla: init state", test_triton_chunk_init_state),
        ("Triton fused_recurrent: basic", test_triton_fused_recurrent_basic),
        ("Triton fused_recurrent: init state", test_triton_fused_recurrent_init_state),
        # Part B: Triton Module Tests
        ("Triton RMSNorm", test_triton_rmsnorm),
        ("Triton FusedRMSNormGated", test_triton_fused_norm_gated),
        ("Triton ShortConv", test_triton_short_conv),
        ("Triton ShortConv: cache", test_triton_short_conv_cache),
        # Part C: Triton Layer Tests
        ("Triton layer: basic", test_triton_layer_basic),
        ("Triton layer: conv", test_triton_layer_conv),
        ("Triton layer: MQA", test_triton_layer_mqa),
        ("Triton layer: no gate", test_triton_layer_no_gate),
        ("Triton layer: expand", test_triton_layer_expand),
        ("Triton layer: mask", test_triton_layer_mask),
        ("Triton layer: long seq", test_triton_layer_long_seq),
        ("Triton layer: clamp", test_triton_layer_clamp),
        # Part D: Architecture Parity
        ("Arch: state_dict keys", test_arch_state_dict_keys),
        ("Arch: weight transfer", test_arch_weight_transfer),
        ("Arch: attributes", test_arch_attributes),
        ("Arch: param count", test_arch_param_count),
        # Part E: Torch_CPU Internal Tests
        ("CPU chunk vs naive", test_cpu_chunk_vs_naive),
        ("CPU chunk vs naive: init state", test_cpu_chunk_vs_naive_init_state),
        ("CPU chunk vs naive: varlen", test_cpu_chunk_vs_naive_varlen),
        ("CPU fused_chunk vs naive", test_cpu_fused_chunk_vs_naive),
        ("CPU cu_seqlens vs separate", test_cpu_cu_seqlens_vs_separate),
        ("CPU RMSNorm standalone", test_cpu_rmsnorm_standalone),
        ("CPU FNG standalone", test_cpu_fused_norm_gated_standalone),
        ("CPU ShortConv causal", test_cpu_short_conv_causal),
        ("CPU ShortConv cu_seqlens", test_cpu_short_conv_cu_seqlens),
        ("CPU ShortConv step", test_cpu_short_conv_step),
        ("CPU layer basic", test_cpu_layer_basic),
        ("CPU layer feature_map", test_cpu_layer_feature_map),
        ("CPU layer normalizer", test_cpu_layer_normalizer),
        ("CPU layer T=1", test_cpu_layer_seq1),
        ("CPU grad basic", test_cpu_grad_basic),
        ("CPU grad conv", test_cpu_grad_conv),
        ("CPU grad MQA", test_cpu_grad_mqa),
        ("CPU numerical stability", test_cpu_numerical_stability),
        ("CPU determinism", test_cpu_determinism),
        ("CPU unpad/pad roundtrip", test_cpu_unpad_pad_roundtrip),
        # Part F: Sub-function Tests
        ("CPU chunk_local_cumsum", test_cpu_chunk_local_cumsum),
        ("CPU chunk_fwd_h", test_cpu_chunk_fwd_h),
        ("CPU chunk_gla_fwd_intra_gk", test_cpu_chunk_gla_fwd_intra_gk),
        ("CPU chunk_gla_fwd_o_gk", test_cpu_chunk_gla_fwd_o_gk),
        ("CPU chunk_gla_fwd orchestrator", test_cpu_chunk_gla_fwd),
        ("CPU fused_recurrent_gla", test_cpu_fused_recurrent_gla),
        ("CPU sub-functions compose", test_cpu_sub_functions_compose),
    ]

    passed = 0
    total = len(test_cases)

    for i, (name, fn) in enumerate(test_cases):
        print(f"\n{'#' * 70}")
        print(f"Test {i + 1}/{total}: {name}")
        print(f"{'#' * 70}")
        try:
            if fn():
                passed += 1
            else:
                print("  >>> FAILED")
        except Exception:
            print("  [FAIL] Exception:")
            traceback.print_exc()

    print(f"\n{'=' * 70}")
    print(f"Summary: {passed}/{total} passed")
    print(f"{'=' * 70}")
    if passed == total:
        print("ALL TESTS PASSED!")
    else:
        print(f"{total - passed} test(s) FAILED")
    return passed == total


if __name__ == "__main__":
    exit(0 if main() else 1)

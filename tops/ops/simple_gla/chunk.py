import jax
import jax.numpy as jnp

from tops.ops.common.chunk_h import chunk_fwd_h_kernel as chunk_fwd_h
from tops.ops.common.chunk_h import chunk_fwd_h_ref
from tops.ops.common.chunk_h import chunk_bwd_dh_kernel as chunk_bwd_dh
from tops.ops.common.chunk_o import chunk_fwd_o, chunk_bwd_dv, chunk_bwd_dqkwg

def chunk_simple_gla_fwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    g: jax.Array | None = None,
    g_gamma: jax.Array | None = None,
    scale: float | None = None,
    h0: jax.Array | None = None,
    use_ht: bool = False,
    cu_seqlens: jax.Array | None = None,
    chunk_size: int = 64,
) -> tuple[jax.Array, jax.Array | None]:
  B, T, H, K, V = *q.shape, v.shape[-1]
  N = B if cu_seqlens is None else cu_seqlens.shape[0] - 1

  if scale is None:
    scale = K ** -0.5

  assert (B, T, H, K) == k.shape
  assert (B, T, H, V) == v.shape
  assert (g is None) or ((B, T, H) == g.shape)
  assert (g_gamma is None) or ((H,) == g_gamma.shape)
  assert (h0 is None) or ((B, H, K, V) == h0.shape)
  assert (cu_seqlens is None) or ((B + 1,) == cu_seqlens.shape)
  assert (cu_seqlens is None) or (cu_seqlens % chunk_size == 0).all()
  assert T % chunk_size == 0
  assert (K % 128 == 0) and (V % 128 == 0)

  h, ht = chunk_fwd_h_ref(
      k=k,
      v=v,
      g=g,
      g_gamma=g_gamma,
      gk=None,
      gv=None,
      h0=h0,
      output_final_state=use_ht,
      states_in_fp32=False,
      cu_seqlens=cu_seqlens,
      chunk_size=chunk_size,
  )
  o = chunk_fwd_o(
      q=q,
      k=k,
      v=v,
      g=g,
      g_gamma=g_gamma,
      h=h,
      scale=scale,
      cu_seqlens_cpu=cu_seqlens,
      chunk_size=chunk_size,
  )

  assert (B, T, H, V) == o.shape
  assert (ht is None) or ((N, H, K, V) == ht.shape)
  return o, ht


def chunk_simple_gla_bwd(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    do: jax.Array,
    *,
    dht: jax.Array | None = None,
    g: jax.Array | None = None,
    g_gamma: jax.Array | None = None,
    h0: jax.Array | None = None,
    scale: float | None = None,
    cu_seqlens: jax.Array | None = None,
    chunk_size: int = 64,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
  B, T, H, K, V = *q.shape, v.shape[-1]
  if scale is None:
    scale = K ** -0.5
  N = B if cu_seqlens is None else cu_seqlens.shape[0] - 1

  assert (B, T, H, K) == k.shape
  assert (B, T, H, V) == v.shape
  assert (B, T, H, V) == do.shape
  assert (dht is None) or ((N, H, K, V) == dht.shape)
  assert (g is None) or ((B, T, H) == g.shape)
  assert (g_gamma is None) or ((H,) == g_gamma.shape)
  assert (h0 is None) or ((B, H, K, V) == h0.shape)
  assert (cu_seqlens is None) or ((B + 1,) == cu_seqlens.shape)
  assert T % chunk_size == 0
  assert (cu_seqlens is None) or (cu_seqlens % chunk_size == 0).all()
  assert (K % 128 == 0) and (V % 128 == 0)

  h, _ = chunk_fwd_h(
      k=k,
      v=v,
      g=g,
      g_gamma=g_gamma,
      gk=None,
      gv=None,
      h0=h0,
      output_final_state=False,
      states_in_fp32=True,
      cu_seqlens=cu_seqlens,
      chunk_size=chunk_size,
  )

  dh, dh0 = chunk_bwd_dh(
      q=q,
      k=k,
      v=v,
      g=g,
      g_gamma=g_gamma,
      gk=None,
      gv=None,
      do=do,
      h0=h0,
      dht=dht,
      scale=scale,
      states_in_fp32=True,
      cu_seqlens=cu_seqlens,
      chunk_size=chunk_size,
  )

  dq, dk, _, dg = chunk_bwd_dqkwg(
      q=q,
      k=k,
      v=v,
      h=h,
      do=do,
      dh=dh,
      g=g,
      g_gamma=g_gamma,
      scale=scale,
      cu_seqlens_cpu=cu_seqlens,
      chunk_size=chunk_size,
  )

  dv = chunk_bwd_dv(
      q=q,
      k=k,
      do=do,
      dh=dh,
      g=g,
      g_gamma=g_gamma,
      scale=scale,
      cu_seqlens_cpu=cu_seqlens,
      chunk_size=chunk_size,
  )
  return dq, dk, dv, dg, dh0

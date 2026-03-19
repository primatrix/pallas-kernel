import jax
import jax.numpy as jnp
from tops.ops.utils import exp

def chunk_fwd_o(
    q: jax.Array,       # [B, T, H, K]
    k: jax.Array,       # [B, T, H, K]
    v: jax.Array,       # [B, T, H, V]
    h: jax.Array,       # [NT_total, H, K, V]
    *,
    g: jax.Array | None = None,        # [B, T, H] chunk-local cumsum of scalar gate
    g_gamma: jax.Array | None = None,  # [H] per-head fixed decay rate
    scale: float | None = None,
    cu_seqlens_cpu: jax.Array | None = None,
    chunk_size: int = 64,
) -> jax.Array:
    """Chunk forward output computation (pure JAX reference).

    O_c = scale * ( Q_c @ H_c * exp(g_c) + causal(Q_c K_c^T * exp(g_row - g_col)) V_c )

    Note: when cu_seqlens is provided, each sequence length must be
    a multiple of chunk_size so that chunk boundaries align with
    sequence boundaries. Under this constraint the varlen case
    reduces to the standard path — just reshape h from
    [NT_total, H, K, V] to [B, NT, H, K, V].
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    C = chunk_size
    NT = T // C

    if scale is None:
      scale = K ** -0.5

    assert scale is not None
    assert T % C == 0, f"Sequence length T={T} must be divisible by chunk_size={C}"
    assert (cu_seqlens_cpu is None) or (cu_seqlens_cpu % chunk_size == 0).all(), "All sequence lengths must be divisible by chunk_size"

    h = h.reshape(B, NT, H, K, V)

    # Reshape into chunks and transpose for batched matmul
    # [B, NT, C, H, D] -> [B, NT, H, C, D]
    q_c = q.reshape(B, NT, C, H, K).transpose(0, 1, 3, 2, 4)
    k_c = k.reshape(B, NT, C, H, K).transpose(0, 1, 3, 2, 4)
    v_c = v.reshape(B, NT, C, H, V).transpose(0, 1, 3, 2, 4)

    # Inter-chunk: Q_c @ H_c -> [B, NT, H, C, V]
    o_inter = jnp.zeros((B, NT, H, C, V), dtype=jnp.float32)
    A = jnp.zeros((B, NT, H, C, C), dtype=jnp.float32)
    o_inter += jnp.matmul(q_c, h, precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)

    # Intra-chunk: Q_c @ K_c^T -> [B, NT, H, C, C]
    A += jnp.matmul(q_c, jnp.swapaxes(k_c, -2, -1), precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)

    # Apply scalar gate g
    if g is not None:
        g_c = g.reshape(B, NT, C, H).transpose(0, 1, 3, 2)  # [B, NT, H, C]
        o_inter = o_inter * exp(g_c)[..., None]
        A = A * exp(g_c[..., :, None] - g_c[..., None, :])

    # Apply per-head fixed decay g_gamma
    if g_gamma is not None:
        g_gamma_f32 = g_gamma.astype(jnp.float32)
        ramp = g_gamma_f32[:, None] * (jnp.arange(C) + 1)[None, :]  # [H, C] float32
        o_inter = o_inter * exp(ramp)[None, None, :, :, None]
        A = A * exp(ramp[..., :, None] - ramp[..., None, :])[None, None]

    # Causal mask (lower triangular: i >= j)
    causal_mask = jnp.tril(jnp.ones((C, C), dtype=jnp.bool_))
    A = jnp.where(causal_mask, A, 0.0)

    # Intra: A @ V_c -> [B, NT, H, C, V]
    o_intra = jnp.matmul(A.astype(v_c.dtype), v_c, precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32)

    # Combine
    o = (o_inter + o_intra) * scale

    # [B, NT, H, C, V] -> [B, NT, C, H, V] -> [B, T, H, V]
    o = o.transpose(0, 1, 3, 2, 4).reshape(B, T, H, V)
    return o.astype(v.dtype)  # Cast back to input dtype


def chunk_bwd_dv(
    q: jax.Array,       # [B, T, H, K]
    k: jax.Array,       # [B, T, H, K]
    do: jax.Array,      # [B, T, H, V]
    dh: jax.Array,      # [NT_total, H, K, V]
    *,
    g: jax.Array | None = None,        # [B, T, H] chunk-local cumsum of scalar gate
    g_gamma: jax.Array | None = None,  # [H] per-head fixed decay rate
    scale: float | None = None,
    cu_seqlens_cpu: jax.Array | None = None,
    chunk_size: int = 64,
) -> jax.Array:
    """Backward: gradient of v (pure JAX reference).

    dv = dv_inter + dv_intra
      dv_inter[j] = sum_k  k[j,k] * dh[k,v] * exp(-g[j] + g_last)   (inter-chunk)
      dv_intra[j] = sum_{i>=j} A[j,i] * do[i]                        (intra-chunk)

    where A[j,i] = sum_k k[j,k]*q[i,k] * exp(g[i]-g[j]) * scale,  masked j<=i

    Note: when cu_seqlens_cpu is provided, each sequence length must be
    a multiple of chunk_size.
    """
    B, T, H, K = q.shape
    V = do.shape[-1]
    C = chunk_size
    NT = T // C

    if scale is None:
        scale = K ** -0.5

    assert T % C == 0
    assert (cu_seqlens_cpu is None) or (cu_seqlens_cpu % C == 0).all()

    dh = dh.reshape(B, NT, H, K, V)

    # Reshape into chunks: [B, NT, C, H, D] -> [B, NT, H, C, D]
    q_c = q.reshape(B, NT, C, H, K).transpose(0, 1, 3, 2, 4)   # [B, NT, H, C, K]
    k_c = k.reshape(B, NT, C, H, K).transpose(0, 1, 3, 2, 4)   # [B, NT, H, C, K]
    do_c = do.reshape(B, NT, C, H, V).transpose(0, 1, 3, 2, 4)  # [B, NT, H, C, V]

    # Inter-chunk: K_c @ dH -> [B, NT, H, C, V]
    dv_inter = jnp.matmul(
        k_c, dh,
        precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32,
    )

    # Intra-chunk attention: K_c @ Q_c^T -> [B, NT, H, C, C]
    # A[j, i] = k[j] @ q[i]^T  (note: row=key, col=query)
    A = jnp.matmul(
        k_c, jnp.swapaxes(q_c, -2, -1),
        precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32,
    )

    # Gate: g is chunk-local cumsum [B, T, H]
    if g is not None:
        g_c = g.reshape(B, NT, C, H).transpose(0, 1, 3, 2)  # [B, NT, H, C]
        g_last = g_c[..., -1:]  # [B, NT, H, 1]
        # Inter-chunk gate: exp(-g_j + g_last)
        dv_inter = dv_inter * exp(-g_c + g_last)[..., None]
        # Intra-chunk gate: A[j,i] *= exp(g_i - g_j)
        # row=j (key), col=i (query), so: g_c[..., None, :] - g_c[..., :, None]
        A = A * exp(g_c[..., None, :] - g_c[..., :, None])

    # Gate: g_gamma — per-head fixed decay [H]
    if g_gamma is not None:
        ramp = g_gamma[:, None] * (jnp.arange(C) + 1)[None, :]  # [H, C]
        ramp_last = ramp[..., -1:]  # [H, 1]
        dv_inter = dv_inter * exp(-ramp + ramp_last)[None, None, :, :, None]
        A = A * exp(ramp[..., None, :] - ramp[..., :, None])[None, None]

    # Upper-triangular mask: keep j <= i  (row=j, col=i)
    upper_mask = jnp.triu(jnp.ones((C, C), dtype=jnp.bool_))
    A = jnp.where(upper_mask, A, 0.0) * scale

    # Intra-chunk: A @ do -> [B, NT, H, C, V]
    dv_intra = jnp.matmul(
        A.astype(do_c.dtype), do_c,
        precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32,
    )

    dv = dv_inter + dv_intra

    # [B, NT, H, C, V] -> [B, NT, C, H, V] -> [B, T, H, V]
    dv = dv.transpose(0, 1, 3, 2, 4).reshape(B, T, H, V)
    return dv


def chunk_bwd_dqkwg(
    q: jax.Array,       # [B, T, H, K]
    k: jax.Array,       # [B, T, H, K]
    v: jax.Array,       # [B, T, H, V]
    h: jax.Array,       # [NT_total, H, K, V]
    do: jax.Array,      # [B, T, H, V]
    dh: jax.Array,      # [NT_total, H, K, V]
    *,
    g: jax.Array | None = None,        # [B, T, H] chunk-local cumsum of scalar gate
    g_gamma: jax.Array | None = None,  # [H] per-head fixed decay rate
    scale: float | None = None,
    cu_seqlens_cpu: jax.Array | None = None,
    chunk_size: int = 64,
) -> tuple[jax.Array, jax.Array, None, jax.Array | None]:
    """Backward: gradients of q, k, and g (pure JAX reference).

    dq = (do @ h^T + causal(ds) @ k) * scale      (no gate)
    dk = v @ dh^T + causal(ds)^T @ q * scale       (no gate)

    where ds = do @ v^T, masked by causal (i >= j).

    Returns (dq, dk, None, dg).  Third element is a placeholder (dw for delta rule).

    Note: when g is provided, the returned dg is the "raw" per-position gradient
    (NOT reverse-cumsummed). The caller must apply revcumsum if needed.
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    C = chunk_size
    NT = T // C

    if scale is None:
        scale = K ** -0.5

    assert scale is not None
    assert T % C == 0
    assert (cu_seqlens_cpu is None) or (cu_seqlens_cpu % C == 0).all()

    h = h.reshape(B, NT, H, K, V)
    dh = dh.reshape(B, NT, H, K, V)

    # Reshape into chunks: [B, NT, C, H, D] -> [B, NT, H, C, D]
    q_c = q.reshape(B, NT, C, H, K).transpose(0, 1, 3, 2, 4)
    k_c = k.reshape(B, NT, C, H, K).transpose(0, 1, 3, 2, 4)
    v_c = v.reshape(B, NT, C, H, V).transpose(0, 1, 3, 2, 4)
    do_c = do.reshape(B, NT, C, H, V).transpose(0, 1, 3, 2, 4)

    # Inter-chunk: dq_inter = do @ h^T -> [B, NT, H, C, K]
    dq_inter = jnp.matmul(
        do_c, jnp.swapaxes(h, -2, -1),
        precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32,
    )

    # Inter-chunk: dk_inter = v @ dh^T -> [B, NT, H, C, K]
    dk_inter = jnp.matmul(
        v_c, jnp.swapaxes(dh, -2, -1),
        precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32,
    )

    # Intra-chunk: ds = do @ v^T -> [B, NT, H, C, C]
    ds = jnp.matmul(
        do_c, jnp.swapaxes(v_c, -2, -1),
        precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32,
    )

    # Causal mask (lower triangular: i >= j)
    causal_mask = jnp.tril(jnp.ones((C, C), dtype=jnp.bool_))

    dg = None

    if g is not None:
        g_c = g.reshape(B, NT, C, H).transpose(0, 1, 3, 2)  # [B, NT, H, C]
        g_last = g_c[..., -1:]  # [B, NT, H, 1]

        # dg_last = sum(h * dh) * exp(g_last) over K,V dims
        dg_last = jnp.sum(h * dh, axis=(-2, -1))  # [B, NT, H]
        dg_last = dg_last * exp(g_last[..., 0])    # [B, NT, H]

        # dq_inter with gate
        dq_inter = dq_inter * exp(g_c)[..., None] * scale
        # dg from dq: sum(dq_inter * q, axis=-1)
        dg_dq = jnp.sum(dq_inter * q_c, axis=-1)  # [B, NT, H, C]

        # dk_inter with gate
        dk_inter = dk_inter * exp(-g_c + g_last)[..., None]
        # dg from dk: -sum(k * dk_inter, axis=-1)
        dg_dk = -jnp.sum(k_c * dk_inter, axis=-1)  # [B, NT, H, C]
        # dg_last += sum(dk_inter * k) over positions and K
        dg_last = dg_last + jnp.sum(dk_inter * k_c, axis=(-2, -1))  # [B, NT, H]

        # ds with gate
        ds = jnp.where(causal_mask, ds * exp(g_c[..., :, None] - g_c[..., None, :]), 0.0) * scale
        # ds2 = ds * (q @ k^T)
        qk = jnp.matmul(
            q_c, jnp.swapaxes(k_c, -2, -1),
            precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32,
        )
        ds2 = ds * qk
        dg_ds = jnp.sum(ds2, axis=-1) - jnp.sum(ds2, axis=-2)  # [B, NT, H, C]

        # Intra-chunk: dq += ds @ k, dk += ds^T @ q
        ds_cast = ds.astype(k_c.dtype)
        dq_intra = jnp.matmul(
            ds_cast, k_c,
            precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32,
        )
        dk_intra = jnp.matmul(
            jnp.swapaxes(ds_cast, -2, -1), q_c,
            precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32,
        )

        dq = dq_inter + dq_intra
        dk = dk_inter + dk_intra

        # Combine dg: add dg_last to the last position
        dg_c = dg_dq + dg_dk + dg_ds  # [B, NT, H, C]
        dg_c = dg_c.at[..., -1].add(dg_last)

        # [B, NT, H, C] -> [B, NT, C, H] -> [B, T, H]
        dg = dg_c.transpose(0, 1, 3, 2).reshape(B, T, H)

    elif g_gamma is not None:
        g_gamma_f32 = g_gamma.astype(jnp.float32)
        ramp = g_gamma_f32[:, None] * (jnp.arange(C) + 1)[None, :]  # [H, C]
        ramp_last = ramp[:, -1:]  # [H, 1]

        # dq_inter with gamma
        dq_inter = dq_inter * exp(ramp)[None, None, :, :, None] * scale
        # dk_inter with gamma
        dk_inter = dk_inter * exp(-ramp + ramp_last)[None, None, :, :, None]

        # ds with gamma
        ds = jnp.where(
            causal_mask,
            ds * exp(ramp[..., :, None] - ramp[..., None, :])[None, None],
            0.0,
        ) * scale

        # Intra-chunk
        ds_cast = ds.astype(k_c.dtype)
        dq_intra = jnp.matmul(
            ds_cast, k_c,
            precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32,
        )
        dk_intra = jnp.matmul(
            jnp.swapaxes(ds_cast, -2, -1), q_c,
            precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32,
        )

        dq = dq_inter + dq_intra
        dk = dk_inter + dk_intra

    else:
        # No gate
        ds = jnp.where(causal_mask, ds, 0.0)
        ds_cast = ds.astype(k_c.dtype)

        dq_intra = jnp.matmul(
            ds_cast, k_c,
            precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32,
        )
        dk_intra = jnp.matmul(
            jnp.swapaxes(ds_cast, -2, -1), q_c,
            precision=jax.lax.Precision.HIGHEST, preferred_element_type=jnp.float32,
        ) * scale

        dq = (dq_inter + dq_intra) * scale
        dk = dk_inter + dk_intra

    # [B, NT, H, C, K] -> [B, NT, C, H, K] -> [B, T, H, K]
    dq = dq.transpose(0, 1, 3, 2, 4).reshape(B, T, H, K)
    dk = dk.transpose(0, 1, 3, 2, 4).reshape(B, T, H, K)

    return dq.astype(q.dtype), dk.astype(k.dtype), None, dg

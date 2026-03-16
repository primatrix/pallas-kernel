from .chunk import chunk_gla_fwd


def fused_chunk_gla(
    q,
    k,
    v,
    g,
    scale=None,
    initial_state=None,
    output_final_state=False,
    cu_seqlens=None,
    chunk_size=16,
):
    """Fused chunk GLA — delegates to chunk_gla_fwd.

    In FLA this function is deprecated. On CPU there is no difference
    between fused and non-fused chunk implementations.
    """
    dtype = q.dtype
    q, k, v, g = (x.float() for x in (q, k, v, g))
    B, T, H, K = q.shape

    if scale is None:
        scale = K**-0.5

    _, _, _, ht, o = chunk_gla_fwd(
        q, k, v, g,
        g_cumsum=None,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )

    final_state = ht if output_final_state else None
    return o.to(dtype), final_state

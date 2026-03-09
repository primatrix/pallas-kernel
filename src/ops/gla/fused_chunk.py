from .chunk import chunk_gla


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
    """Fused chunk GLA — delegates to chunk_gla.

    In FLA this function is deprecated. On JAX there is no difference
    between fused and non-fused chunk implementations.
    """
    return chunk_gla(
        q=q, k=k, v=v, g=g,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )

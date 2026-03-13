import jax
B, T, H, K = 4, 1024, 4, 128
chunk_size = 64
N = 1
cu = None
if cu is not None:
    N = len(cu) - 1
else:
    N = B
key = jax.random.PRNGKey(1)
k = jax.random.normal(key, (B, T, H, K))
v = jax.random.normal(key, (B, T, H, K))
gk = jax.random.normal(key, (B, T, H, K))
h0 = jax.random.normal(key, (N, H, K, K))

from src.ops.common.chunk_h import chunk_fwd_h_kernel, chunk_fwd_h_kernel_with_same_seq

def _run_pallas(
    k,
    v,
    gk=None,
    h0=None,
    chunk_size=64,
    *,
    cu_seqlens=None,
):
    h, ht = chunk_fwd_h_kernel_with_same_seq(
        k,
        v,
        gk=gk,
        h0=h0,
        chunk_size=chunk_size,
        output_final_state=True,
    )
    return h, ht


pallas_h, pallas_ht = _run_pallas(
    k,
    v,
    gk=gk,
    h0=h0,
    chunk_size=chunk_size,
    cu_seqlens=cu,
)
jax.block_until_ready(pallas_h)
jax.block_until_ready(pallas_ht)
jax.profiler.start_trace("/home/gcpuser/profile")
for i in range(3):
    pallas_h, pallas_ht = _run_pallas(
        k,
        v,
        gk=gk,
        h0=h0,
        chunk_size=chunk_size,
        cu_seqlens=cu,
    )
    jax.block_until_ready(pallas_h)
    jax.block_until_ready(pallas_ht)
jax.profiler.stop_trace()
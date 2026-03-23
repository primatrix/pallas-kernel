import os
DUMP_ROOT = "compiler_dump/"
HLO_DUMP_PATH = os.path.join(DUMP_ROOT, "hlo")
LLO_DUMP_PATH = os.path.join(DUMP_ROOT, "llo")

os.makedirs(HLO_DUMP_PATH, exist_ok=True)
os.makedirs(LLO_DUMP_PATH, exist_ok=True)

if False:
    os.environ["XLA_FLAGS"] = (
        f"--xla_dump_hlo_as_text "
        f"--xla_dump_to={HLO_DUMP_PATH} "
        f"--xla_dump_hlo_pass_re=.* "
    )

    os.environ["LIBTPU_INIT_ARGS"] = (
        f"--xla_jf_dump_to={LLO_DUMP_PATH} "
        f"--xla_jf_dump_hlo_text=true "
        f"--xla_jf_dump_llo_text=true "
        f"--xla_jf_dump_llo_html=false "
        f"--xla_jf_dump_llo_static_gaps=true "
        f"--xla_jf_emit_annotations=true "
        f"--xla_jf_debug_level=2"
    )

import jax
B, T, H, K = 4, 1024, 4, 128
chunk_size = 512
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
import torch
import torch.nn.functional as F
from einops import rearrange, repeat


def prepare_lens_from_mask(mask: torch.BoolTensor) -> torch.LongTensor:
    """Calculate the lengths of each sequence from the attention mask."""
    return mask.sum(dim=-1, dtype=torch.int32)


def prepare_cu_seqlens_from_mask(
    mask: torch.BoolTensor,
    dtype: torch.dtype = torch.int32,
) -> torch.LongTensor:
    """Calculate the cumulative sequence lengths (cu_seqlens) from the attention mask."""
    lens = prepare_lens_from_mask(mask)
    return F.pad(lens.cumsum(dim=0, dtype=dtype), (1, 0))


def index_put_first_axis(
    x: torch.Tensor,
    indices: torch.Tensor,
    first_axis_dim: int,
) -> torch.Tensor:
    """Put compact tensor back to specified positions in flattened first dimension.

    x: [num_selected, ...], indices: [num_selected]
    returns: [first_axis_dim, ...]
    """
    assert indices.ndim == 1
    assert x.ndim >= 2
    y = torch.zeros(first_axis_dim, *x.shape[1:], device=x.device, dtype=x.dtype)
    y[indices] = x
    return y


def get_unpad_data(
    attention_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Get indexing data required for unpadding.

    Args:
        attention_mask: [batch_size, seq_len], 1 for valid, 0 for padding.

    Returns:
        indices: Indices of valid tokens in the flattened sequence.
        cu_seqlens: Cumulative sequence lengths [batch_size + 1].
        max_seqlen_in_batch: Maximum sequence length in the batch.
    """
    lens = prepare_lens_from_mask(attention_mask)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = lens.max().item()
    cu_seqlens = prepare_cu_seqlens_from_mask(attention_mask)
    return indices, cu_seqlens, max_seqlen_in_batch


def index_first_axis(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Select elements from flattened first dimension using specified indices.

    x: [total_tokens, ...], indices: [num_selected]
    returns: [num_selected, ...]
    """
    assert x.ndim >= 2
    other_shape = x.shape[1:]
    second_dim = other_shape.numel()
    return torch.gather(
        rearrange(x, "b ... -> b (...)"),
        0,
        repeat(indices, "z -> z d", d=second_dim),
    ).reshape(-1, *other_shape)


def pad_input(
    hidden_states: torch.Tensor,
    indices: torch.LongTensor,
    batch_size: int,
    seq_len: int,
) -> torch.Tensor:
    """Pad compact tensor back into dense tensor of [batch_size, seq_len, ...]."""
    output = index_put_first_axis(hidden_states, indices, batch_size * seq_len)
    return rearrange(output, "(b s) ... -> b s ...", b=batch_size)

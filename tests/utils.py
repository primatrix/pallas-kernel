import torch
import jax
import numpy as np


def compare_tensor(
    name: str,
    gold: np.ndarray | jax.Array | torch.Tensor,
    tensor: np.ndarray | jax.Array | torch.Tensor,
    atol=1e-5,
    rtol=1e-5,
) -> bool:
    if gold is None and tensor is None:
        print(f"[{name}] Both are None. MATCH.")
        return False
    if gold is None or tensor is None:
        print(f"[{name}] One is None! MISMATCH.")
        return False

    if isinstance(gold, torch.Tensor):
        gold = gold.detach().to(torch.float32).cpu().numpy()
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().to(torch.float32).cpu().numpy()
    if isinstance(gold, jax.Array):
        gold = np.array(gold, dtype=np.float32)
    if isinstance(tensor, jax.Array):
        tensor = np.array(tensor, dtype=np.float32)

    if gold.shape != tensor.shape:
        print(
            f"[{name}] Shape mismatch: Left {gold.shape} vs Right {tensor.shape}. FAIL."
        )
        if gold.squeeze().shape == tensor.squeeze().shape:
            print(
                f"  Attempting comparison with squeezed shapes: {gold.squeeze().shape}"
            )
            gold = gold.squeeze()
            tensor = tensor.squeeze()
        else:
            return False

    diff = np.abs(gold - tensor)
    max_diff = np.max(diff)
    max_val = np.max(np.abs(tensor))
    max_rel_diff = np.max(diff / (np.abs(tensor) + 1e-12))

    is_close = np.allclose(gold, tensor, atol=atol, rtol=rtol, equal_nan=True)
    status = "PASS" if is_close else "FAIL"

    print(f"[{name}] {status}")
    print(f"  Max Value        : {max_val:.6e}")
    print(f"  Max Abs Diff     : {max_diff:.6e}")
    print(f"  Max Rel Diff     : {max_rel_diff:.6e}")

    if not is_close:
        tolerance = atol + rtol * np.abs(tensor)
        error_ratio = diff / (tolerance + 1e-12)
        idx = np.unravel_index(np.argmax(error_ratio), error_ratio.shape)
        print(f"  Max Mismatch details at index {idx}:")
        print(f"    Left (Triton)  = {gold[idx]}")
        print(f"    Right (Pallas) = {tensor[idx]}")
        print(f"    Diff           = {diff[idx]}")
        print(
            f"    Tolerance      = {tolerance[idx]} (atol={atol} + rtol={rtol}*|Right|)"
        )
        print(f"    Ratio          = {error_ratio[idx]}")

    return is_close

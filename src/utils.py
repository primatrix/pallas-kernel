import jax
import jax.numpy as jnp
from functools import singledispatch


def next_power_of_2(n: int):
    """Return the smallest power of 2 greater than or equal to n"""
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    n += 1
    return n


@singledispatch
def cdiv(x: int, y: int):
    return (x + y - 1) // y


@cdiv.register
def cdiv(x: jax.Array, y: int):
    return (x + y - 1) // y


def align_up(x: int, align: int):
    return cdiv(x, align) * align


@singledispatch
def pad_to_multiple(x, multiple: int, axis: int, val):
    raise NotImplementedError(f"pad_to_multiple is not implemented for type {type(x)}")


@pad_to_multiple.register
def pad_to_multiple(x: jax.Array, multiple: int | list, axis: int | list, val):
    if isinstance(multiple, int):
        multiple = [multiple]
    if isinstance(axis, int):
        axis = [axis]

    assert len(multiple) == len(axis), (
        f"Length of multiple {len(multiple)} must match length of axis {len(axis)}"
    )

    shape = list(x.shape)
    pad_width = [(0, 0)] * len(shape)
    for idx in range(0, len(axis)):
        ax = axis[idx]
        mu = multiple[idx]
        length = shape[ax]
        remainder = length % mu
        if remainder == 0:
            continue
        pad_len = mu - remainder
        pad_width[ax] = (0, pad_len)
    return jnp.pad(x, pad_width, constant_values=val)

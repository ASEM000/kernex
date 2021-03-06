"""
[credits] Mahmoud Asem@CVBML KAIST May 2022
"""

from __future__ import annotations

import functools

import jax
from jax import numpy as jnp
from jax import vmap


def ZIP(*args):
    """assert all args have the same number of elements before zipping"""
    n = len(args[0])

    assert all(
        len(x) == n for x in args[1:]
    ), f"zip arguments dont have the same length. Args length = {tuple(len(arg) for arg in args)}"

    return zip(*args)


def _offset_to_padding(input_argument, kernel_size):
    """convert offset argument to negative border values"""

    padding = [[]] * len(kernel_size)

    # offset = 1 ==> padding= 0 for kernel_size =3
    same = lambda wi: ((wi - 1) // 2, wi // 2)

    for i, item in enumerate(input_argument):

        zero_offset_padding = same(kernel_size[i])

        if isinstance(item, tuple):
            assert item >= (0, 0, ), f"offset must be non negative value.Found offset={item}"  # fmt: skip
            padding[i] = tuple(ai - bi for ai, bi in ZIP(zero_offset_padding, item))

        elif isinstance(item, int):
            assert item >= 0, f"offset must be non negative value.Found offset={item}"
            padding[i] = tuple(ai - bi for ai, bi in ZIP(zero_offset_padding, (item, item)))  # fmt: skip

    # [TODO] throw an error if offset value is larger than kernel_size
    return tuple(padding)


@functools.partial(jax.profiler.annotate_function, name="roll_view")
def roll_view(array: jnp.ndarray) -> jnp.ndarray:
    """Roll view along all axes

    Example:
    >>> x = jnp.arange(1,26).reshape(5,5)
    >>> print(roll_view(x))
        [[13 14 15 11 12]
        [18 19 20 16 17]
        [23 24 25 21 22]
        [ 3  4  5  1  2]
        [ 8  9 10  6  7]]
    """
    shape = jnp.array(array.shape)
    axes = tuple(range(len(shape)))  # list all axes
    shift = tuple(
        -(si // 2) if si % 2 == 1 else -((si - 1) // 2) for si in array.shape
    )  # right padding>left padding
    return jnp.roll(array, shift=shift, axis=axes)


def ix_(*args):
    """modified version of jnp.ix_"""
    n = len(args)
    output = []
    for i, a in enumerate(args):
        shape = [1] * n
        shape[i] = a.shape[0]
        output.append(jax.lax.broadcast_in_dim(a, shape, (i,)))
    return tuple(output)


@functools.partial(jax.profiler.annotate_function, name="general_arange")
def general_arange(di: int, ki: int, si: int, x0: int, xf: int) -> jnp.ndarray:
    """Calculate the windows indices for a given dimension.

    Args:
        di (int): shape of the dimension
        ki (int): kernel size
        si (int): stride
        x0 (int): left padding
        xf (int): rght padding

    Returns:
        jnp.ndarray: array of windows indices

    Example:
        >>> di = 5
        >>> ki = 3
        >>> si = 1
        >>> x0 = 0
        >>> xf = 0
        >>> print(general_arange(di, ki, si, x0, xf))
            [[0 1 2]
            [1 2 3]
            [2 3 4]]
    """
    start, end = -x0 + ((ki - 1) // 2), di + xf - (ki // 2)
    size = end - start
    lhs = jax.lax.broadcasted_iota(dtype=jnp.int32, shape=(size, ki), dimension=0) + (start)  # fmt: skip
    rhs = jax.lax.broadcasted_iota(dtype=jnp.int32, shape=(ki, size), dimension=0).T - ((ki - 1) // 2)  # fmt: skip
    res = lhs + rhs

    # res[::si] is slightly slower.
    return (res) if si == 1 else (res)[::si]


@functools.partial(jax.profiler.annotate_function, name="general_product")
def general_product(*args):
    """Equivalent to tuple(zip(*itertools.product(*args)))` for arrays

    Example:
    >>> general_product(
    ... jnp.array([[1,2],[3,4]]),
    ... jnp.array([[5,6],[7,8]]))
    (
        DeviceArray([[[1, 2],[1, 2]],[[3, 4],[3, 4]]], dtype=int32),
        DeviceArray([[[5, 6],[7, 8]],[[5, 6],[7, 8]]], dtype=int32)
    )


    >>> tuple(zip(*(itertools.product([[1,2],[3,4]],[[5,6],[7,8]]))))
    (
        ([1, 2], [1, 2], [3, 4], [3, 4]),
        ([5, 6], [7, 8], [5, 6], [7, 8])
    )

    """

    def nvmap(n):
        in_axes = [None] * len(args)
        in_axes[-n] = 0
        return (
            vmap(lambda *x: x, in_axes=in_axes)
            if n == 1
            else vmap(nvmap(n - 1), in_axes=in_axes)
        )

    return nvmap(len(args))(*args)


def _index_from_view(
    view: tuple[jnp.ndarray, ...], kernel_size: tuple[int, ...]
) -> tuple[int, ...]:
    """Get the index of array from the view

    Args:
        view (tuple[jnp.ndarray,...]): patch indices for each dimension
        kernel_size (tuple[int,...]): kernel size for each dimension

    Returns:
        tuple[int, ...]: index as a tuple of int for each dimension
    """

    return tuple(
        view[i][wi // 2] if wi % 2 == 1 else view[i][(wi - 1) // 2]
        for i, wi in enumerate(kernel_size)
    )


def _compare_key(x: tuple[jnp.ndarray, ...], y: tuple[jnp.ndarray, ...]) -> bool:
    """check if index as array x is in the range of index as array y for all dimensions

    Args:
        x (jnp.ndarray): lhs index
        y (jnp.ndarray): rhs index

    Returns:
        bool: if x in range(y) or x == y
    """

    def _compare_key_item(xi: jnp.ndarray, yi: jnp.ndarray) -> bool:
        """check if index as array xi is in the range of index as array yi for single dimension

        Args:
            xi (jnp.ndarray): lhs index
            yi (jnp.ndarray): rhs index

        Returns:
            bool: if xi in range(yi) or xi == yi
        """
        # index style = (start,end,step)
        if yi.size == 3:
            return (yi[0] <= xi) * (xi < yi[1]) * (xi % yi[2] == 0)

        # index style = (start,end )
        elif yi.size == 2:
            return (yi[0] <= xi) * (xi < yi[1])

        # index style = (index)
        elif yi.size == 1:
            return jnp.where(yi == jnp.inf, True, xi == yi)

    return jnp.all(jnp.array([_compare_key_item(xi, yi) for (xi, yi) in zip(x, y)]))


def _key_search(key: tuple[jnp.ndarray, ...], keys: tuple[jnp.ndarray]) -> int:
    """returns the index of the key in the keys array if key is within the key range or equal to it.

    Args:
        key (tuple[jnp.ndarray,...]):
            a tuple of jnp.arrays for each dimension ( {dim0},...,{dimN}) with size({dim}) == 1

        keys (tuple[jnp.ndarray):
            a tuple of jnp.arrays for range of each dimension
            ( {dim0},...,{dimN}) with size({dim})with size ({dim}) in [1,2,3]

    Returns:
        int: index of the key in the keys array
    """

    length = len(keys)

    # [<(0,0),(0,1)> , <(0,0)>] ->
    # ( ({0},{0}), ({0},{1}) ) , ( ({0,0}) )
    # key = [jnp.array([ki]) for ki in key]

    def in_key_group(key, key_group):

        comparisons = tuple(
            [
                _compare_key(
                    [jnp.array([ki]) for ki in key],  # tuples of array
                    [jnp.array(ki) for ki in rhs_key],
                )
            ]
            for rhs_key in key_group
        )

        # check if key match any key in the key_groupx
        return jnp.any(jnp.array(comparisons))

    def recurse(idx, keys):
        # under jit , no short circuit is possible .
        return jnp.where(
            in_key_group(key, keys[0]),
            idx,
            idx + 1 if idx == length - 1 else recurse(idx + 1, keys[1:]),
        )

    return recurse(0, keys[::-1]) if len(keys) > 0 else 0

# Copyright 2023 Kernex authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import functools as ft
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.util import safe_zip

transform_func_map = {
    "vmap": jax.vmap,
    "lmap": lambda func, **k: (lambda xs: jax.lax.map(func, xs, **k)),
    "pmap": jax.pmap,
    "scan": jax.lax.scan,
}


def _calculate_pad_width(border: tuple[tuple[int, int], ...]):
    """Calcuate the positive padding from border value

    Returns:
        padding value passed to `pad_width` in `jnp.pad`
    """
    # this function is cached because it is called multiple times
    # and it is expensive to calculate
    # if the border is negative, the padding is 0
    # if the border is positive, the padding is the border value
    return tuple([0, max(0, pi[0]) + max(0, pi[1])] for pi in border)


def _get_index_from_view(
    view: jax.Array,
    kernel_size: tuple[tuple[int, int], ...],
) -> tuple[int, ...]:
    """Get the index of array from the view

    Args:
        view (tuple[jax.Array,...]): patch indices for each dimension

    Returns:
        tuple[int, ...]: index as a tuple of int for each dimension
    """
    return tuple(
        view[i][wi // 2] if wi % 2 == 1 else view[i][(wi - 1) // 2]
        for i, wi in enumerate(kernel_size)
    )


def _generate_views(
    shape: tuple[int, ...],
    kernel_size: tuple[int, ...],
    strides: tuple[int, ...],
    border: tuple[tuple[int, int], ...],
) -> tuple[jax.Array, ...]:
    """Generate absolute sampling matrix"""
    # this function is cached because it is called multiple times
    # and it is expensive to calculate
    # the view is the indices of the array that is used to calculate
    # the output value
    dim_range = tuple(
        general_arange(di, ki, si, x0, xf)
        for (di, ki, si, (x0, xf)) in zip(shape, kernel_size, strides, border)
    )
    matrix = general_product(*dim_range)
    return tuple(map(lambda xi, wi: xi.reshape(-1, wi), matrix, kernel_size))


def _calculate_output_shape(
    shape: tuple[int, ...],
    kernel_size: tuple[int, ...],
    strides: tuple[int, ...],
    border: tuple[tuple[int, int], ...],
) -> tuple[int, ...]:
    """Calculate the output shape of the kernel operation from
    the input shape, kernel size, stride and border.

    Returns:
        tuple[int, ...]: resulting shape of the kernel operation
    """
    # this function is cached because it is called multiple times
    # and it is expensive to calculate
    # the output shape is the shape of the array after the kernel operation
    # is applied to the input array
    return tuple(
        (xi + (li + ri) - ki) // si + 1
        for xi, ki, si, (li, ri) in safe_zip(shape, kernel_size, strides, border)
    )


@ft.lru_cache(maxsize=128)
def _offset_to_padding(input_argument, kernel_size: tuple[int, ...]):
    """convert offset argument to negative border values"""
    # for example for a kernel_size = (3,3) and offset = (1,1)
    # the padding will be (-1,-1) for each dimension
    padding = [[]] * len(kernel_size)

    # offset = 1 ==> padding= 0 for kernel_size =3
    same = lambda wi: ((wi - 1) // 2, wi // 2)

    for i, item in enumerate(input_argument):

        zero_offset_padding = same(kernel_size[i])

        if isinstance(item, tuple):
            if item < (0, 0):
                raise ValueError(f"Negative value found, offset={item}")
            padding[i] = tuple(
                ai - bi for ai, bi in safe_zip(zero_offset_padding, item)
            )

        elif isinstance(item, int):
            if item < 0:
                raise ValueError(f"Negative value found, offset={item}")
            padding[i] = tuple(
                ai - bi for ai, bi in safe_zip(zero_offset_padding, (item, item))
            )

    # [TODO] throw an error if offset value is larger than kernel_size
    return tuple(padding)


def roll(a: jax.Array, shift: Sequence[int], axis: Sequence[int]) -> jax.Array:
    # https://github.com/jakevdp/jax/blob/5497bb03b6c353853b2ef6536e954898214736c8/jax/_src/numpy/lax_numpy.py
    for ax, s in zip(*np.broadcast_arrays(axis, shift)):
        if a.shape[ax] == 0:
            continue
        i = (-s) % a.shape[ax]
        a = lax.concatenate(
            [
                lax.slice_in_dim(a, i, a.shape[ax], axis=ax),
                lax.slice_in_dim(a, 0, i, axis=ax),
            ],
            dimension=ax,
        )
    return a


@ft.partial(jax.profiler.annotate_function, name="roll_view")
def roll_view(array: jax.Array) -> jax.Array:
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
    # this function is used to roll the view along all axes
    shape = jnp.array(array.shape)
    axes = tuple(range(len(shape)))  # list all axes
    # right padding>left padding
    shift = tuple(-(si // 2) if si % 2 == 1 else -((si - 1) // 2) for si in array.shape)
    return roll(array, shift=shift, axis=axes)


def ix_(*args):
    """modified version of jnp.ix_"""
    n = len(args)
    output = []
    for i, a in enumerate(args):
        shape = [1] * n
        shape[i] = a.shape[0]
        output.append(jax.lax.broadcast_in_dim(a, shape, (i,)))
    return tuple(output)


def _get_set_indices(shape, strides, offset):
    # the indices of the array that are set by the kernel operation
    # this is used to set the values of the array after the kernel operation
    # is applied
    return tuple(
        jnp.arange(x0, di - xf, si)
        for di, si, (x0, xf) in safe_zip(shape, strides, offset)
    )


@ft.partial(jax.profiler.annotate_function, name="general_arange")
def general_arange(di: int, ki: int, si: int, x0: int, xf: int) -> jax.Array:
    """Calculate the windows indices for a given dimension.

    Args:
        di (int): shape of the dimension
        ki (int): kernel size
        si (int): stride
        x0 (int): left padding
        xf (int): rght padding

    Returns:
        jax.Array: array of windows indices

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
    # this function is used to calculate the windows indices for a given dimension
    start, end = -x0 + ((ki - 1) // 2), di + xf - (ki // 2)
    size = end - start

    res = (
        lax.broadcasted_iota(dtype=jnp.int32, shape=(size, ki), dimension=0)
        + lax.broadcasted_iota(dtype=jnp.int32, shape=(ki, size), dimension=0).T
        + (start)
        - ((ki - 1) // 2)
    )

    # res[::si] is slightly slower.
    return (res) if si == 1 else (res)[::si]


@ft.lru_cache(maxsize=128)
def recursive_vmap(*, ndim: int):
    def nvmap(n):
        in_axes = [None] * ndim
        in_axes[-n] = 0
        return (
            jax.vmap(lambda *x: x, in_axes=in_axes)
            if n == 1
            else jax.vmap(nvmap(n - 1), in_axes=in_axes)
        )

    return nvmap(ndim)


@ft.partial(jax.profiler.annotate_function, name="general_product")
def general_product(*args: jax.Array):
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
    return recursive_vmap(ndim=len(args))(*args)


def _index_from_view(
    view: tuple[jax.Array, ...],
    kernel_size: tuple[int, ...],
) -> tuple[int, ...]:
    """Get the index of array from the view

    Args:
        view (tuple[jax.Array,...]): patch indices for each dimension
        kernel_size (tuple[int,...]): kernel size for each dimension

    Returns:
        tuple[int, ...]: index as a tuple of int for each dimension
    """
    return tuple(
        view[i][wi // 2] if wi % 2 == 1 else view[i][(wi - 1) // 2]
        for i, wi in enumerate(kernel_size)
    )


def _compare_key(x: tuple[jax.Array, ...], y: tuple[jax.Array, ...]) -> bool:
    """check if index as array x is in the range of index as array y for all dimensions

    Args:
        x (jax.Array): lhs index
        y (jax.Array): rhs index

    Returns:
        bool: if x in range(y) or x == y
    """

    def _compare_key_item(xi: jax.Array, yi: jax.Array) -> bool:
        """check if index as array xi is in the range of index as array yi for single dimension

        Args:
            xi (jax.Array): lhs index
            yi (jax.Array): rhs index

        Returns:
            bool: if xi in range(yi) or xi == yi
        """
        # index style = (start,end,step)
        if yi.size == 3:
            return (yi[0] <= xi) * (xi < yi[1]) * (xi % yi[2] == 0)

        # index style = (start,end )
        if yi.size == 2:
            return (yi[0] <= xi) * (xi < yi[1])

        # index style = (index)
        if yi.size == 1:
            return jnp.where(yi == jnp.inf, True, xi == yi)

    return jnp.all(jnp.array([_compare_key_item(xi, yi) for (xi, yi) in zip(x, y)]))


def _key_search(key: tuple[jax.Array, ...], keys: tuple[jax.Array]) -> int:
    """returns the index of the key in the keys array if key is within the key range or equal to it.

    Args:
        key (tuple[jax.Array,...]):
            a tuple of jnp.arrays for each dimension ( {dim0},...,{dimN}) with size({dim}) == 1

        keys (tuple[jax.Array):
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

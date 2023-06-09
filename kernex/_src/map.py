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
from typing import Any, Callable, Literal

import jax
import jax.numpy as jnp
from jax import lax

from kernex._src.utils import (
    _calculate_output_shape,
    _calculate_pad_width,
    _generate_views,
    _get_index_from_view,
    _get_set_indices,
    _key_search,
    _offset_to_padding,
    ix_,
    roll_view,
    transform_func_map,
)

MapKind = Literal["vmap", "lmap", "pmap"]


@ft.lru_cache(maxsize=None)
def _transform_map_func(func: Callable, relative: bool) -> Callable:
    # transform a function that takes array to
    # a function that takes a view and an array
    def relative_wrapper(*a, **k):
        def map_func(view: jax.Array, array: jax.Array):
            return func(roll_view(array[ix_(*view)]), *a, **k)

        return map_func

    def absolute_wrapper(*a, **k):
        def map_func(view: jax.Array, array: jax.Array):
            return func(array[ix_(*view)], *a, **k)

        return map_func

    return relative_wrapper if relative else absolute_wrapper


def kernel_map(
    func_map: dict,
    shape: tuple[int, ...],
    kernel_size: tuple[int, ...],
    strides: tuple[int, ...],
    border: tuple[tuple[int, int], ...],
    relative: bool = False,
    map_kind: MapKind = "vmap",
    map_kwargs: dict[str, Any] | None = None,
) -> Callable:

    map_kwargs = map_kwargs or {}
    map_tranform = transform_func_map[map_kind]
    pad_width = _calculate_pad_width(border)
    args = (shape, kernel_size, strides, border)
    views = _generate_views(*args)
    # reshape the result to the output shape
    # for example if the input shape is (3, 3) and the kernel shape is (2, 2)
    # and the stride is 1 , and the padding is 0, the output shape is (2, 2)
    output_shape = _calculate_output_shape(*args)

    slices = tuple(func_map.values())

    def single_call_wrapper(array: jax.Array, *a, **k):
        padded_array = jnp.pad(array, pad_width)

        # convert the function to a callable that takes a view and an array
        # and returns the result of the function applied to the view
        func0 = next(iter(func_map))
        reduced_func = _transform_map_func(func0, relative)(*a, **k)

        # apply the function to each view using vmap
        # the result is a 1D array of the same length as the number of views
        def map_func(view):
            return reduced_func(view, padded_array)

        result = map_tranform(map_func, **map_kwargs)(views)
        return result.reshape(*output_shape, *result.shape[1:])

    def multi_call_wrapper(array: jax.Array, *a, **k):
        padded_array = jnp.pad(array, pad_width)
        # convert the functions to a callable that takes a view and an array
        # and returns the result of the function applied to the view
        # the result is a 1D array of the same length as the number of views
        reduced_funcs = tuple(
            _transform_map_func(func, relative)(*a, **k)
            for func in tuple(func_map.keys())[::-1]
        )

        # apply the functions to each view using vmap
        # the result is a 1D array of the same length as the number of views
        # here, lax.switch is used to apply the functions in order
        # the first function is applied to the first view, the second function
        # is applied to the second view, and so on
        def map_func(view):
            index_ = _get_index_from_view(view, kernel_size)
            func_index = _key_search(key=tuple(index_), keys=slices)
            return lax.switch(func_index, reduced_funcs, view, padded_array)

        result = map_tranform(map_func, **map_kwargs)(views)
        func_shape = result.shape[1:]
        return result.reshape(*output_shape, *func_shape)

    return single_call_wrapper if len(func_map) == 1 else multi_call_wrapper


def offset_kernel_map(
    func_map: dict,
    shape: tuple[int, ...],
    kernel_size: tuple[int, ...],
    strides: tuple[int, ...],
    offset: tuple[tuple[int, int], ...],
    relative: bool = False,
    map_kind: MapKind = "vmap",
    map_kwargs: dict[str, Any] = None,
):

    func = kernel_map(
        func_map=func_map,
        shape=shape,
        kernel_size=kernel_size,
        strides=strides,
        border=_offset_to_padding(tuple(offset), kernel_size),
        relative=relative,
        map_kind=map_kind,
        map_kwargs=map_kwargs,
    )
    set_indices = _get_set_indices(shape, strides, offset)

    def call(array: jax.Array, *a, **k):
        result = func(array, *a, **k)
        if result.shape > array.shape:
            raise ValueError(f"Output must be scalar. Foud {result.shape=}")
        return array.at[ix_(*set_indices)].set(result)

    return call

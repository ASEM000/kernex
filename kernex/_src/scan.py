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

ScanKind = Literal["scan"]


@ft.lru_cache(maxsize=None)
def _transform_scan_func(
    func: Callable,
    kernel_size: tuple[int, ...],
    relative: bool,
) -> Callable:
    def relative_wrapper(*a, **k):
        def scan_func(view: jax.Array, array: jax.Array):
            index = _get_index_from_view(view, kernel_size)
            return array.at[index].set(func(roll_view(array[ix_(*view)]), *a, **k))

        return scan_func

    def absolute_wrapper(*a, **k):
        def scan_func(view: jax.Array, array: jax.Array):
            index = _get_index_from_view(view, kernel_size)
            return array.at[index].set(func(array[ix_(*view)], *a, **k))

        return scan_func

    return relative_wrapper if relative else absolute_wrapper


def kernel_scan(
    func_map: dict,
    shape: tuple[int, ...],
    kernel_size: tuple[int, ...],
    strides: tuple[int, ...],
    border: tuple[tuple[int, int], ...],
    relative: bool = False,
    scan_kind: ScanKind = "scan",  # dummy to make signature consistent with kernel_map
    scan_kwargs: dict[str, Any] | None = None,
):

    scan_kwargs = scan_kwargs or {}
    scan_transform = transform_func_map[scan_kind]
    pad_width = _calculate_pad_width(border)
    args = (shape, kernel_size, strides, border)
    views = _generate_views(*args)
    output_shape = _calculate_output_shape(*args)
    slices = tuple(func_map.values())

    def single_call_wrapper(array: jax.Array, *a, **k):
        padded_array = jnp.pad(array, pad_width)
        func0 = next(iter(func_map))
        reduced_func = _transform_scan_func(func0, kernel_size, relative)(*a, **k)

        def scan_body(padded_array: jax.Array, view: jax.Array):
            result = reduced_func(view, padded_array).reshape(padded_array.shape)
            index = _get_index_from_view(view, kernel_size)
            return result, result[index]

        _, result = scan_transform(scan_body, padded_array, views, **scan_kwargs)
        return result.reshape(output_shape)

    def multi_call_wrapper(array: jax.Array, *a, **k):
        padded_array = jnp.pad(array, pad_width)

        reduced_funcs = tuple(
            _transform_scan_func(func, kernel_size, relative)(*a, **k)
            for func in tuple(func_map.keys())[::-1]
        )

        def scan_body(padded_array, view):
            index = _get_index_from_view(view, kernel_size)
            func_index = _key_search(key=tuple(index), keys=slices)
            result = lax.switch(func_index, reduced_funcs, view, padded_array)
            result = result.reshape(padded_array.shape)
            return result, result[index]

        _, result = scan_transform(scan_body, padded_array, views, **scan_kwargs)
        return result.reshape(output_shape)

    return single_call_wrapper if len(func_map) == 1 else multi_call_wrapper


def offset_kernel_scan(
    func_map: dict,
    shape: tuple[int, ...],
    kernel_size: tuple[int, ...],
    strides: tuple[int, ...],
    offset: tuple[tuple[int, int], ...],
    relative: bool = False,
    scan_kind: ScanKind = "scan",
    scan_kwargs: dict[str, Any] | None = None,
):

    func = kernel_scan(
        func_map=func_map,
        shape=shape,
        kernel_size=kernel_size,
        strides=strides,
        border=_offset_to_padding(tuple(offset), kernel_size),
        relative=relative,
        scan_kind=scan_kind,
        scan_kwargs=scan_kwargs,
    )
    set_indices = _get_set_indices(shape, strides, offset)

    def call(array, *a, **k):
        result = func(array, *a, **k)
        if result.shape > array.shape:
            raise ValueError("scan operation output must be scalar.")
        return array.at[ix_(*set_indices)].set(result)

    return call

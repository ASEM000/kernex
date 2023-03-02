# [credits] Mahmoud Asem@CVBML KAIST May 2022

from __future__ import annotations

import functools as ft

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
)


@ft.lru_cache(maxsize=None)
def _transform_scan_func(func, kernel_size, relative: bool):
    def relative_wrapper(*a, **k):
        def scan_func(view, array):
            index = _get_index_from_view(view, kernel_size)
            return array.at[index].set(func(roll_view(array[ix_(*view)]), *a, **k))

        return scan_func

    def absolute_wrapper(*a, **k):
        def scan_func(view, array):
            index = _get_index_from_view(view, kernel_size)
            return array.at[index].set(func(array[ix_(*view)], *a, **k))

        return scan_func

    return relative_wrapper if relative else absolute_wrapper


def kernel_scan(
    func_index_map: dict,
    shape: tuple[int, ...],
    kernel_size: tuple[int, ...],
    strides: tuple[int, ...],
    border: tuple[tuple[int, int], ...],
    relative: bool = False,
):
    def single_call_wrapper(array, *a, **k):
        padded_array = jnp.pad(array, _calculate_pad_width(border))
        func0 = next(iter(func_index_map))
        reduced_func = _transform_scan_func(func0, kernel_size, relative)(*a, **k)

        def scan_body(padded_array, view):
            result = reduced_func(view, padded_array).reshape(padded_array.shape)
            index = _get_index_from_view(view, kernel_size)
            return result, result[index]

        args = (shape, kernel_size, strides, border)
        views = _generate_views(*args)
        output_shape = _calculate_output_shape(*args)
        return lax.scan(scan_body, padded_array, views)[1].reshape(output_shape)

    def multi_call_wrapper(array, *a, **k):
        padded_array = jnp.pad(array, _calculate_pad_width(border))

        reduced_funcs = tuple(
            _transform_scan_func(func, kernel_size, relative)(*a, **k)
            for func in tuple(func_index_map.keys())[::-1]
        )

        slices = tuple(func_index_map.values())

        def scan_body(padded_array, view):
            index = _get_index_from_view(view, kernel_size)
            func_index = _key_search(key=tuple(index), keys=slices)
            result = lax.switch(func_index, reduced_funcs, view, padded_array)
            result = result.reshape(padded_array.shape)
            return result, result[index]

        args = (shape, kernel_size, strides, border)
        views = _generate_views(*args)
        output_shape = _calculate_output_shape(*args)
        return lax.scan(scan_body, padded_array, views)[1].reshape(output_shape)

    return single_call_wrapper if len(func_index_map) == 1 else multi_call_wrapper


def offset_kernel_scan(
    func_dict: dict,
    shape: tuple[int, ...],
    kernel_size: tuple[int, ...],
    strides: tuple[int, ...],
    offset: tuple[tuple[int, int], ...],
    relative: bool = False,
):
    offset = tuple(offset)
    padding = _offset_to_padding(offset, kernel_size)
    func = kernel_scan(func_dict, shape, kernel_size, strides, padding, relative)
    set_indices = _get_set_indices(shape, strides, offset)

    def call(array, *a, **k):
        result = func(array, *a, **k)
        if result.shape > array.shape:
            msg = "scan operation output must be scalar."
            raise ValueError(msg)
        assert result.shape <= array.shape, msg
        return array.at[ix_(*set_indices)].set(result)

    return call

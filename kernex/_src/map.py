# [credits] Mahmoud Asem@CVBML KAIST May 2022

from __future__ import annotations

import functools as ft

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
)


@ft.lru_cache(maxsize=None)
def _transform_map_func(func, relative: bool):
    def relative_wrapper(*a, **k):
        def map_func(view, array):
            return func(roll_view(array[ix_(*view)]), *a, **k)

        return map_func

    def absolute_wrapper(*a, **k):
        def map_func(view, array):
            return func(array[ix_(*view)], *a, **k)

        return map_func

    return relative_wrapper if relative else absolute_wrapper


def kernel_map(
    func_index_map: dict,
    shape: tuple[int, ...],
    kernel_size: tuple[int, ...],
    strides: tuple[int, ...],
    border: tuple[tuple[int, int], ...],
    relative: bool = False,
):
    def single_call_wrapper(array, *a, **k):
        padded_array = jnp.pad(array, _calculate_pad_width(border))

        # convert the function to a callable that takes a view and an array
        # and returns the result of the function applied to the view
        func0 = next(iter(func_index_map))
        reduced_func = _transform_map_func(func0, relative)(*a, **k)

        # apply the function to each view using vmap
        # the result is a 1D array of the same length as the number of views
        args = (shape, kernel_size, strides, border)
        views = _generate_views(*args)

        def map_func(view):
            return reduced_func(view, padded_array)

        result = jax.vmap(map_func)(views)

        # reshape the result to the output shape
        # for example if the input shape is (3, 3) and the kernel shape is (2, 2)
        # and the stride is 1 , and the padding is 0, the output shape is (2, 2)
        output_shape = _calculate_output_shape(*args)

        return result.reshape(*output_shape, *result.shape[1:])

    def multi_call_wrapper(array, *a, **k):

        padded_array = jnp.pad(array, _calculate_pad_width(border))
        # convert the functions to a callable that takes a view and an array
        # and returns the result of the function applied to the view
        # the result is a 1D array of the same length as the number of views
        reduced_funcs = tuple(
            _transform_map_func(func, relative)(*a, **k)
            for func in tuple(func_index_map.keys())[::-1]
        )

        # apply the functions to each view using vmap
        # the result is a 1D array of the same length as the number of views
        # here, lax.switch is used to apply the functions in order
        # the first function is applied to the first view, the second function
        # is applied to the second view, and so on

        args = (shape, kernel_size, strides, border)
        views = _generate_views(*args)
        slices = tuple(func_index_map.values())

        def map_func(view):
            index_ = _get_index_from_view(view, kernel_size)
            func_index = _key_search(key=tuple(index_), keys=slices)
            return lax.switch(func_index, reduced_funcs, view, padded_array)

        result = jax.vmap(map_func)(views)

        func_shape = result.shape[1:]
        output_shape = _calculate_output_shape(*args)

        return result.reshape(*output_shape, *func_shape)

    return single_call_wrapper if len(func_index_map) == 1 else multi_call_wrapper


def offset_kernel_map(
    func_dict: dict,
    shape: tuple[int, ...],
    kernel_size: tuple[int, ...],
    strides: tuple[int, ...],
    offset: tuple[tuple[int, int], ...],
    relative: bool = False,
):
    offset = tuple(offset)
    padding = _offset_to_padding(offset, kernel_size)
    func = kernel_map(func_dict, shape, kernel_size, strides, padding, relative)
    set_indices = _get_set_indices(shape, strides, offset)

    def call(array, *a, **k):
        result = func(array, *a, **k)
        if result.shape > array.shape:
            msg = f"kernel operation output must be scalar. Foud {result.shape}"
            raise ValueError(msg)
        return array.at[ix_(*set_indices)].set(result)

    return call

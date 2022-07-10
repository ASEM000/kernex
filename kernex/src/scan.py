from __future__ import annotations

import functools
from typing import Callable

from jax import lax
from jax import numpy as jnp

from kernex.src.base import kernelOperation
from kernex.src.utils import ZIP, ix_, offset_to_padding, roll_view
from kernex.treeclass.decorator import treeclass


@treeclass
class baseKernelScan(kernelOperation):

    def __post_init__(self):
        self.__call__ = (self.__single_call__
                         if len(self.funcs) == 1 else self.__multi_call__)

    def reduce_scan_func(self, func, *args, **kwargs) -> Callable:

        if self.relative:
            def reduce_scan_func_callable(view, array):
                return array.at[self.index_from_view(view)].set(
                    func(roll_view(array[ix_(*view)]), *args, **kwargs))

        else:
            def reduce_scan_func_callable(view, array):
                return array.at[self.index_from_view(view)].set(
                    func((array[ix_(*view)]), *args, **kwargs))

        return reduce_scan_func_callable

    def __single_call__(self, array, *args, **kwargs):

        padded_array = jnp.pad(array, self.pad_width)
        reduced_func = self.reduce_scan_func(self.funcs[0], *args, **kwargs)

        def scan_body(padded_array, view):
            result = reduced_func(view,
                                  padded_array).reshape(padded_array.shape)
            return result, result[self.index_from_view(view)]

        return lax.scan(scan_body, padded_array,
                        self.views)[1].reshape(self.output_shape)

    def __multi_call__(self, array, *args, **kwargs):

        padded_array = jnp.pad(array, self.pad_width)

        reduced_funcs = tuple(
            self.reduce_scan_func(func, *args, **kwargs)
            for func in self.funcs[::-1])

        def scan_body(padded_array, view):
            result = lax.switch(self.func_index_from_view(view), reduced_funcs,
                                view, padded_array).reshape(padded_array.shape)

            return result, result[self.index_from_view(view)]

        return lax.scan(scan_body, padded_array,
                        self.views)[1].reshape(self.output_shape)


@treeclass
class kernelScan(baseKernelScan):

    def __init__(self, func_dict, shape, kernel_size, strides, padding,
                 relative):

        super().__init__(func_dict, shape, kernel_size, strides, padding,
                         relative)

    def __call__(self, array, *args, **kwargs):
        return self.__call__(array, *args, **kwargs)


@treeclass
class offsetKernelScan(kernelScan):

    def __init__(self, func_dict, shape, kernel_size, strides, offset,
                 relative):

        self.offset = offset

        super().__init__(
            func_dict,
            shape,
            kernel_size,
            strides,
            offset_to_padding(offset, kernel_size),
            relative,
        )

    @functools.cached_property
    def __set_indices__(self):
        return tuple(
            jnp.arange(x0, di - xf, si) for di, ki, si, (x0, xf) in ZIP(
                self.shape, self.kernel_size, self.strides, self.offset))

    def __call__(self, array, *args, **kwargs):
        result = self.__call__(array, *args, **kwargs)
        assert result.shape <= array.shape, "scan operation output must be scalar."
        return array.at[ix_(*self.__set_indices__)].set(result)

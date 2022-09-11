from __future__ import annotations

from typing import Callable

import pytreeclass as pytc
from jax import lax
from jax import numpy as jnp
from jax import vmap

from kernex._src.base import kernelOperation
from kernex._src.utils import ZIP, _offset_to_padding, cached_property, ix_, roll_view


@pytc.treeclass
class baseKernelMap(kernelOperation):
    def __post_init__(self):

        self.__call__ = (
            # if there is only one function, use the single call method
            # this is faster than the multi call method
            # this is because the multi call method uses lax.switch
            self.__single_call__
            if len(self.funcs) == 1
            else self.__multi_call__
        )

    def reduce_map_func(self, func, *args, **kwargs) -> Callable:
        if self.relative:
            # if the function is relative, the function is applied to the view
            return lambda view, array: func(
                roll_view(array[ix_(*view)]), *args, **kwargs
            )

        else:
            return lambda view, array: func(array[ix_(*view)], *args, **kwargs)

    def __single_call__(self, array: jnp.ndarray, *args, **kwargs):
        padded_array = jnp.pad(array, self.pad_width)

        # convert the function to a callable that takes a view and an array
        # and returns the result of the function applied to the view
        reduced_func = self.reduce_map_func(self.funcs[0], *args, **kwargs)

        # apply the function to each view using vmap
        # the result is a 1D array of the same length as the number of views
        result = vmap(lambda view: reduced_func(view, padded_array))(self.views)

        # reshape the result to the output shape
        # for example if the input shape is (3, 3) and the kernel shape is (2, 2)
        # and the stride is 1 , and the padding is 0, the output shape is (2, 2)
        return result.reshape(*self.output_shape, *result.shape[1:])

    def __multi_call__(self, array, *args, **kwargs):

        padded_array = jnp.pad(array, self.pad_width)
        # convert the functions to a callable that takes a view and an array
        # and returns the result of the function applied to the view
        # the result is a 1D array of the same length as the number of views
        reduced_funcs = tuple(
            self.reduce_map_func(func, *args, **kwargs) for func in self.funcs[::-1]
        )

        # apply the functions to each view using vmap
        # the result is a 1D array of the same length as the number of views
        # here, lax.switch is used to apply the functions in order
        # the first function is applied to the first view, the second function
        # is applied to the second view, and so on
        result = vmap(
            lambda view: lax.switch(
                self.func_index_from_view(view), reduced_funcs, view, padded_array
            )
        )(self.views)

        func_shape = result.shape[1:]
        return result.reshape(*self.output_shape, *func_shape)


@pytc.treeclass
class kernelMap(baseKernelMap):
    """A class for applying a function to a kernel map of an array"""

    def __init__(self, func_dict, shape, kernel_size, strides, padding, relative):
        super().__init__(func_dict, shape, kernel_size, strides, padding, relative)

    def __call__(self, array, *args, **kwargs):
        return self.__call__(array, *args, **kwargs)


@pytc.treeclass
class offsetKernelMap(kernelMap):
    """A class for applying a function to a kernel map of an array"""

    def __init__(self, func_dict, shape, kernel_size, strides, offset, relative):
        # the offset is converted to padding and the padding is used to pad the array
        # the padding is then used to calculate the views

        self.offset = offset

        super().__init__(
            func_dict,
            shape,
            kernel_size,
            strides,
            _offset_to_padding(offset, kernel_size),
            relative,
        )

    @cached_property
    def set_indices(self):
        # the indices of the array that are set by the kernel operation
        # this is used to set the values of the array after the kernel operation
        # is applied
        return tuple(
            jnp.arange(x0, di - xf, si)
            for di, ki, si, (x0, xf) in ZIP(
                self.shape, self.kernel_size, self.strides, self.offset
            )
        )

    def __call__(self, array, *args, **kwargs):
        # apply the kernel operation
        # the result is a 1D array of the same length as the number of views
        # the result is reshaped to the output shape
        result = self.__call__(array, *args, **kwargs)
        assert (
            result.shape <= array.shape
        ), f"kernel operation output must be scalar. Foud {result.shape}"

        return array.at[ix_(*self.set_indices)].set(result)

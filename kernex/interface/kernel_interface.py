"""
[credits] Mahmoud Asem@CVBML KAIST May 2022
This script defines the decorators and class object
"""

from __future__ import annotations

import functools
from typing import Callable

from jax import numpy as jnp
from pytreeclass import static_field, treeclass

from kernex.interface.named_axis import named_axis_wrapper
from kernex.interface.resolve_utils import (
    normalize_slices,
    resolve_kernel_size,
    resolve_offset_argument,
    resolve_padding_argument,
    resolve_strides,
)
from kernex.src.map import kernelMap, offsetKernelMap
from kernex.src.scan import kernelScan, offsetKernelScan


@treeclass(op=False)
class kernelInterface:

    kernel_size: tuple[int, ...] | int = static_field()
    strides: tuple[int, ...] | int = static_field(default=1)
    border: tuple[int, ...] | tuple[tuple[int, int], ...] | int | str = static_field(default=0, repr=False)  # fmt: skip
    relative: bool = static_field(default=False)
    inplace: bool = static_field(default=False)
    use_offset: bool = static_field(default=False)
    named_axis: dict[int, str] | None = static_field(default=None)
    container: dict[Callable, slice | int] = static_field(default_factory=dict)

    def __post_init__(self):
        """resolve the border values and the kernel operation"""

        if self.use_offset:
            self.border = resolve_offset_argument(self.border, self.kernel_size)
            self.kernel_op = offsetKernelScan if self.inplace else offsetKernelMap

        else:
            self.border = resolve_padding_argument(self.border, self.kernel_size)
            self.kernel_op = kernelScan if self.inplace else kernelMap

    def __setitem__(self, index, func):

        assert isinstance(
            func, Callable
        ), f"Input must be of type Callable. Found {type(func)}"

        # append slice/index to func key list
        self.container[func] = [*self.container.get(func, []), index]

    def __mesh_call__(self, array, *args, **kwargs):
        # TODO : run once resolve_kernel_size/resolve_strides

        self.shape = array.shape
        self.kernel_size = resolve_kernel_size(self.kernel_size, self.shape)
        self.strides = resolve_strides(self.strides, self.shape)
        self.container = normalize_slices(self.container, self.shape)
        self.resolved_container = {}

        for (func, index) in self.container.items():

            if func is not None and self.named_axis is not None:
                resolved_func = named_axis_wrapper(self.kernel_size, self.named_axis)(
                    func
                )
                self.resolved_container[resolved_func] = index

            else:
                self.resolved_container[func] = index

        return self.kernel_op(
            self.resolved_container,
            self.shape,
            self.kernel_size,
            self.strides,
            self.border,
            self.relative,
        )(array, *args, **kwargs)

    def __decorator_call__(self, func):
        def call(array, *args, **kwargs):

            # TODO : run once resolve_kernel_size/resolve_strides
            self.shape = array.shape
            self.kernel_size = resolve_kernel_size(self.kernel_size, self.shape)
            self.strides = resolve_strides(self.strides, self.shape)

            self.resolved_container = {
                named_axis_wrapper(self.kernel_size, self.named_axis)(func)
                if self.named_axis is not None
                else func: ()
            }

            return self.kernel_op(
                self.resolved_container,
                self.shape,
                self.kernel_size,
                self.strides,
                self.border,
                self.relative,
            )(array, *args, **kwargs)

        return call

    def __call__(self, *args, **kwargs):

        if len(args) == 1 and callable(args[0]) and len(kwargs) == 0:
            return functools.wraps(args[0])(self.__decorator_call__(args[0]))

        elif len(args) > 0 and isinstance(args[0], jnp.ndarray):
            return self.__mesh_call__(*args, **kwargs)

        else:
            raise ValueError(
                (
                    "Expected `jnp.ndarray` or `Callable` for the first argument.",
                    f" Found {tuple(*args,**kwargs)}",
                )
            )


@treeclass(op=False)
class sscan(kernelInterface):
    def __init__(
        self, kernel_size=1, strides=1, offset=0, relative=False, named_axis=None
    ):

        super().__init__(
            kernel_size=kernel_size,
            strides=strides,
            border=offset,
            relative=relative,
            inplace=True,
            use_offset=True,
            named_axis=named_axis,
        )


@treeclass(op=False)
class smap(kernelInterface):
    def __init__(
        self, kernel_size=1, strides=1, offset=0, relative=False, named_axis=None
    ):

        super().__init__(
            kernel_size=kernel_size,
            strides=strides,
            border=offset,
            relative=relative,
            inplace=False,
            use_offset=True,
            named_axis=named_axis,
        )


@treeclass(op=False)
class kscan(kernelInterface):
    def __init__(
        self, kernel_size=1, strides=1, padding=0, relative=False, named_axis=None
    ):

        super().__init__(
            kernel_size=kernel_size,
            strides=strides,
            border=padding,
            relative=relative,
            inplace=True,
            use_offset=False,
            named_axis=named_axis,
        )


@treeclass(op=False)
class kmap(kernelInterface):
    def __init__(
        self, kernel_size=1, strides=1, padding=0, relative=False, named_axis=None
    ):

        super().__init__(
            kernel_size=kernel_size,
            strides=strides,
            border=padding,
            relative=relative,
            inplace=False,
            use_offset=False,
            named_axis=named_axis,
        )

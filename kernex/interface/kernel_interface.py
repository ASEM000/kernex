"""
[credits] Mahmoud Asem@CVBML KAIST May 2022
This script defines the decorators and class object
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Callable

from jax import numpy as jnp

from kernex.interface.named_axis import named_axis_wrapper
from kernex.interface.resolve_utils import (
    _normalize_slices,
    _resolve_kernel_size,
    _resolve_offset_argument,
    _resolve_padding_argument,
    _resolve_strides,
)
from kernex.src.map import kernelMap, offsetKernelMap
from kernex.src.scan import kernelScan, offsetKernelScan


@dataclass
class kernelInterface:

    kernel_size: tuple[int, ...] | int = field()
    strides: tuple[int, ...] | int = field(default=1)
    border: tuple[int, ...] | tuple[tuple[int, int], ...] | int | str = field(default=0, repr=False)  # fmt: skip
    relative: bool = field(default=False)
    inplace: bool = field(default=False)
    use_offset: bool = field(default=False)
    named_axis: dict[int, str] | None = field(default=None)
    container: dict[Callable, slice | int] = field(default_factory=dict)

    def __post_init__(self):
        """resolve the border values and the kernel operation"""
        self.border = (
            _resolve_offset_argument(self.border, self.kernel_size)
            if self.use_offset
            else _resolve_padding_argument(self.border, self.kernel_size)
        )

    def __setitem__(self, index, func):

        assert isinstance(
            func, Callable
        ), f"Input must be of type Callable. Found {type(func)}"

        # append slice/index to func key list
        self.container[func] = [*self.container.get(func, []), index]

    def _wrap_mesh(self, array, *args, **kwargs):
        self.shape = array.shape
        self.kernel_size = _resolve_kernel_size(self.kernel_size, self.shape)
        self.strides = _resolve_strides(self.strides, self.shape)
        self.container = _normalize_slices(self.container, self.shape)
        self.resolved_container = {}

        for (func, index) in self.container.items():

            if func is not None and self.named_axis is not None:
                self.resolved_container[
                    named_axis_wrapper(self.kernel_size, self.named_axis)(func)
                ] = index

            else:
                self.resolved_container[func] = index

        kernel_op = (
            (offsetKernelScan if self.inplace else offsetKernelMap)
            if self.use_offset
            else (kernelScan if self.inplace else kernelMap)
        )

        return kernel_op(
            self.resolved_container,
            self.shape,
            self.kernel_size,
            self.strides,
            self.border,
            self.relative,
        )(array, *args, **kwargs)

    def _wrap_decorator(self, func):
        def call(array, *args, **kwargs):
            object.__setattr__(self, "shape", array.shape)  # fmt: skip
            object.__setattr__(self, "kernel_size", _resolve_kernel_size(self.kernel_size, self.shape))  # fmt: skip
            object.__setattr__(self, "strides", _resolve_strides(self.strides, self.shape))  # fmt: skip

            object.__setattr__(
                self,
                "resolved_container",
                {
                    named_axis_wrapper(self.kernel_size, self.named_axis)(func)
                    if self.named_axis is not None
                    else func: ()
                },
            )

            kernel_op = (
                (offsetKernelScan if self.inplace else offsetKernelMap)
                if self.use_offset
                else (kernelScan if self.inplace else kernelMap)
            )

            return kernel_op(
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
            return functools.wraps(args[0])(self._wrap_decorator(args[0]))

        elif len(args) > 0 and isinstance(args[0], jnp.ndarray):
            return self._wrap_mesh(*args, **kwargs)

        else:
            raise ValueError(
                (
                    f"Expected `jnp.ndarray` or `Callable` for the first argument. Found {tuple(*args,**kwargs)}"
                )
            )


@dataclass
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


@dataclass
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


@dataclass
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


@dataclass
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

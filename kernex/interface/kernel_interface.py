"""
[credits] Mahmoud Asem@CVBML KAIST May 2022
This script defines the decorators and class object
"""

from __future__ import annotations

import dataclasses as dc
import functools
from typing import Callable

import jax.numpy as jnp

from kernex._src.map import kernel_map, offset_kernel_map
from kernex._src.scan import kernel_scan, offset_kernel_scan
from kernex.interface.named_axis import named_axis_wrapper
from kernex.interface.resolve_utils import (
    _normalize_slices,
    _resolve_kernel_size,
    _resolve_offset_argument,
    _resolve_padding_argument,
    _resolve_strides,
)


@dc.dataclass
class kernelInterface:

    kernel_size: tuple[int, ...] | int
    strides: tuple[int, ...] | int = 1
    border: tuple[int, ...] | tuple[tuple[int, int], ...] | int | str = dc.field(default=0, repr=False)  # fmt: skip
    relative: bool = False
    inplace: bool = False
    use_offset: bool = False
    named_axis: dict[int, str] | None = None
    container: dict[Callable, slice | int] = dc.field(default_factory=dict)

    def __post_init__(self):
        """resolve the border values and the kernel operation"""
        self.border = (
            _resolve_offset_argument(self.border, self.kernel_size)
            if self.use_offset
            else _resolve_padding_argument(self.border, self.kernel_size)
        )

    def __setitem__(self, index, func):

        msg = f"Input must be of type Callable. Found {type(func)}"
        assert isinstance(func, Callable), msg

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
            (offset_kernel_scan if self.inplace else offset_kernel_map)
            if self.use_offset
            else (kernel_scan if self.inplace else kernel_map)
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
            self.shape = array.shape
            self.kernel_size = _resolve_kernel_size(self.kernel_size, self.shape)
            self.strides = _resolve_strides(self.strides, self.shape)

            self.resolved_container = {
                named_axis_wrapper(self.kernel_size, self.named_axis)(func)
                if self.named_axis is not None
                else func: ()
            }

            kernel_op = (
                (offset_kernel_scan if self.inplace else offset_kernel_map)
                if self.use_offset
                else (kernel_scan if self.inplace else kernel_map)
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

        if len(args) > 0 and isinstance(args[0], jnp.ndarray):
            return self._wrap_mesh(*args, **kwargs)

        msg = f"Expected `jnp.ndarray` or `Callable` for the first argument. Found {tuple(*args,**kwargs)}"
        raise ValueError(msg)


@dc.dataclass
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


@dc.dataclass
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


@dc.dataclass
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


@dc.dataclass
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

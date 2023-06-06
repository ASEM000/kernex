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
from typing import Callable, Literal, Union

import jax

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

BorderType = Union[
    tuple[int, ...],
    tuple[tuple[int, int], ...],
    int,
    Literal["valid", "same", "SAME", "VALID"],
]

StridesType = Union[tuple[int, ...], int]
OffsetType = Union[tuple[int, ...], int]
KernelSizeType = tuple[int, ...]


class KernelInterface:
    def __init__(
        self,
        kernel_size: KernelSizeType,
        strides: StridesType = 1,
        border: BorderType = 0,
        relative: bool = False,
        inplace: bool = False,
        use_offset: bool = False,
        named_axis: dict[int, str] | None = None,
        container: dict[Callable, slice | int] | None = None,
    ):
        self.kernel_size = kernel_size
        self.strides = strides
        self.relative = relative
        self.inplace = inplace
        self.use_offset = use_offset
        self.named_axis = named_axis
        self.container = container or dict()
        self.border = (
            _resolve_offset_argument(border, self.kernel_size)
            if self.use_offset
            else _resolve_padding_argument(border, self.kernel_size)
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(kernel_size={self.kernel_size}, "
            f"strides={self.strides}, "
            f"{'offset' if self.use_offset else 'padding'}={self.border}, "
            f"relative={self.relative}, "
            f"named_axis={self.named_axis})"
        )

    def __setitem__(self, index, func):
        if not isinstance(func, Callable):
            raise TypeError(f"Input not callable. Found {type(func)}")

        # append slice/index to func key list
        self.container[func] = [*self.container.get(func, []), index]

    def _wrap_mesh(self, array: jax.Array, *a, **k):
        self.shape = array.shape
        self.kernel_size = _resolve_kernel_size(self.kernel_size, self.shape)
        self.strides = _resolve_strides(self.strides, self.shape)
        self.container = _normalize_slices(self.container, self.shape)
        self.resolved_container = dict()

        for (func, index) in self.container.items():

            if func is not None and self.named_axis is not None:
                named_func = named_axis_wrapper(
                    kernel_size=self.kernel_size,
                    named_axis=self.named_axis,
                )(func)

                self.resolved_container[named_func] = index

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
        )(array, *a, **k)

    def _wrap_decorator(self, func):
        def call(array, *args, **kwargs):
            self.shape = array.shape
            self.kernel_size = _resolve_kernel_size(self.kernel_size, self.shape)
            self.strides = _resolve_strides(self.strides, self.shape)

            self.resolved_container = {
                named_axis_wrapper(self.kernel_size, self.named_axis)(func)
                if self.named_axis is not None
                # single function
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
            return ft.wraps(args[0])(self._wrap_decorator(args[0]))

        if len(args) > 0 and isinstance(args[0], jax.Array):
            return self._wrap_mesh(*args, **kwargs)

        raise ValueError(
            f"Expected `jax.Array` or `Callable` for the first argument."
            f" Found {tuple(*args,**kwargs)}"
        )


class sscan(KernelInterface):
    def __init__(
        self,
        kernel_size: KernelSizeType = 1,
        strides: StridesType = 1,
        offset: OffsetType = 0,
        relative: bool = False,
        named_axis: dict[int, str] | None = None,
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


class smap(KernelInterface):
    def __init__(
        self,
        kernel_size: KernelSizeType = 1,
        strides: StridesType = 1,
        offset: OffsetType = 0,
        relative: bool = False,
        named_axis: dict[int, str] = None,
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


class kscan(KernelInterface):
    def __init__(
        self,
        kernel_size: KernelSizeType = 1,
        strides: StridesType = 1,
        padding: BorderType = 0,
        relative: bool = False,
        named_axis: dict[int, str] = None,
    ):
        """Apply a function to a sliding window of the input array sequentially.

        Args:
            kernel_size: size of the kernel must be a tuple of integers.
            strides: strides of the kernel, int or tuple of ints.
            padding: padding of the kernel, int or tuple of ints.
            relative: if True, the kernel is relative to the current index.
            named_axis: optional dictionary of named axis to be used in the
                kernel function instead of the default integer indexing.

        Returns:
            A function that takes an array as input and returns the result of
            the kernel function applied to the array.

        Examples:
            >>> import kernex as kex
            >>> import jax.numpy as jnp
            >>> @kex.kscan(kernel_size=(3,))
            ... def sum_all(x):
            ...     return jnp.sum(x)
            >>> x = jnp.array([1,2,3,4,5])
            >>> print(sum_all(x))
            [ 6 13 22]

        Note:
        The previous example is equivalent to the following:
        v1 := [1,2,3] -> sum(v1) = 6
        v2 := [6,3,4] -> sum(v2) = 13
        v3 := [13,4,5] -> sum(v3) = 22
        """
        super().__init__(
            kernel_size=kernel_size,
            strides=strides,
            border=padding,
            relative=relative,
            inplace=True,
            use_offset=False,
            named_axis=named_axis,
        )


class kmap(KernelInterface):
    def __init__(
        self,
        kernel_size: KernelSizeType = 1,
        strides: StridesType = 1,
        padding: BorderType = 0,
        relative: bool = False,
        named_axis: dict[int, str] = None,
    ):
        """Apply a function to a sliding window of the input array in parallel.

        Args:
            kernel_size: size of the kernel must be a tuple of integers.
            strides: strides of the kernel, int or tuple of ints.
            padding: padding of the kernel, int or tuple of ints.
            relative: if True, the kernel is relative to the current index.
            named_axis: optional dictionary of named axis to be used instead
                of integers inside the kernel function.

        Returns:
            A function that takes an array as input and applies the kernel

        Example:
            >>> import kernex as kex
            >>> import jax.numpy as jnp
            >>> @kex.kmap(kernel_size=(3,))
            ... def sum_all(x):
            ...     return jnp.sum(x)
            >>> x = jnp.array([1,2,3,4,5])
            >>> print(sum_all(x))
            [ 6  9 12]

        Note:
            The previous example is equivalent to the following:
            v1 := [1,2,3] -> sum(v1) = 6
            v2 := [2,3,4] -> sum(v2) = 9
            v3 := [3,4,5] -> sum(v3) = 12
        """
        super().__init__(
            kernel_size=kernel_size,
            strides=strides,
            border=padding,
            relative=relative,
            inplace=False,
            use_offset=False,
            named_axis=named_axis,
        )

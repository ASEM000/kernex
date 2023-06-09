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
from typing import Any, Callable, Literal, Tuple, Union

import jax
import jax.tree_util as jtu

from kernex._src.map import MapKind, kernel_map, offset_kernel_map
from kernex._src.scan import ScanKind, kernel_scan, offset_kernel_scan
from kernex.interface.named_axis import named_axis_wrapper
from kernex.interface.resolve_utils import (
    _normalize_slices,
    _resolve_kernel_size,
    _resolve_offset_argument,
    _resolve_padding_argument,
    _resolve_strides,
)

BorderType = Union[
    int,
    Tuple[int, ...],
    Tuple[Tuple[int, int], ...],
    Literal["valid", "same", "SAME", "VALID"],
]

StridesType = Union[Tuple[int, ...], int]
OffsetType = Union[Tuple[int, ...], int]
KernelSizeType = Tuple[int, ...]


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
        transform_kind: MapKind | ScanKind | None = None,
        transform_kwargs: dict[str, Any] | None = None,
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
        self.transform_kind = transform_kind
        self.transform_kwargs = transform_kwargs

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

        kernel_op = jtu.Partial(
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
            self.transform_kind,
            self.transform_kwargs,
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

            kernel_op = jtu.Partial(
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
                self.transform_kind,
                self.transform_kwargs,
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
        scan_kind: ScanKind = "scan",
        scan_kwargs: dict[str, Any] | None = None,
    ):
        """Applies a *scalar* function to a sliding window using `jax.lax.scan`.

        Args:
            kernel_size: size of the kernel must be a tuple of integers.
            strides: strides of the kernel, int or tuple of ints.
            offset: where to start the function application. for example, for 1d
                array input `offset=((1,1),)` the function application will start
                from the second row and first column of the input array and end
                at the n-1 row of the input array.
            relative: if True, the kernel is relative to the current index.
                for example, an array = [1, 2, 3, 4, 5] with kernel_size = 3 and
                relative = True f= lambda x: x[0] will be [1,2,3]=>[2],
                [2,3,4]=>[3], [3,4,5]=>[4]. if relative = False, the same array will
                be [1,2,3]=>[1], [2,3,4]=>[2], [3,4,5]=>[3].
            named_axis: optional dictionary of named axis to be used in the
                kernel function instead of the default integer indexing.
                for example: {0: 'i', 1: 'j'}, then
                this notation can be used in the kernel function e.g.:
                `f = lambda x: x['i+1','j+1']` is equivalent to lambda x: x[1,1]
            scan_kind: the kind of scan to be used. available options are:
                jax.lax.scan under `scan`.
            scan_kwargs: optional kwargs to be passed to the scan function.
                for example, `scan_kwargs={'reverse': True}` will reverse the
                application of the function.

        Example:
            >>> import jax
            >>> import kernex as kex
            >>> @kex.sscan(
            ...    # view shape
            ...    kernel_size=(3,),
            ...    # indexing is relative: x[0,0] points to
            ...    # array view center and not top left corner
            ...    relative=True,
            ...    # start the application of the function
            ...    # from the third column till the end (i.e. no offset at the end)
            ...    offset=((2,0),),
            ... )
            ... def F(x):
            ...    return x[-1] + x[0] + x[1]
            >>> x = jax.numpy.arange(1,6)
            >>> print(x)
            [1 2 3 4 5]
            >>> print(F(x))
            [ 1  2  9 18 23]

        Note:
            - This transformation is always applying within the bounds of the
                of the array. and shape of the output is the same as the input.
            - The previous example can be decomposed into the following steps:

            # view shape       # process
            v1:=[0,1,2]        skip as offset is 2
            v2:=[1,2,3]        skip as offset is 2
            v3:=[2,3,4]        F(v3) = 2+3+4 = 9
            v4:=[9,4,5]        F(v4) = 9+4+5 = 18
            v5:=[18,5,0]       F(v5) = 18+5+0 = 23

        """
        super().__init__(
            kernel_size=kernel_size,
            strides=strides,
            border=offset,
            relative=relative,
            inplace=True,
            use_offset=True,
            named_axis=named_axis,
            transform_kind="scan",
            transform_kwargs=scan_kwargs,
        )


class smap(KernelInterface):
    def __init__(
        self,
        kernel_size: KernelSizeType = 1,
        strides: StridesType = 1,
        offset: OffsetType = 0,
        relative: bool = False,
        named_axis: dict[int, str] = None,
        map_kind: MapKind = "vmap",
        map_kwargs: dict[str, Any] | None = None,
    ):
        """Applies a *scalar* function to a sliding window using `jax.vmap`.

        Args:
            kernel_size: size of the kernel must be a tuple of integers.
            strides: strides of the kernel, int or tuple of ints.
            offset: where to start the function application. for example, for 1d
                array input `offset=((1,1),)` the function application will start
                from the second row and first column of the input array and end
                at the n-1 row of the input array.
            relative: if True, the kernel is relative to the current index.
                for example, an array = [1, 2, 3, 4, 5] with kernel_size = 3 and
                relative = True f= lambda x: x[0] will be [1,2,3]=>[2],
                [2,3,4]=>[3], [3,4,5]=>[4]. if relative = False, the same array will
                be [1,2,3]=>[1], [2,3,4]=>[2], [3,4,5]=>[3].
            named_axis: optional dictionary of named axis to be used in the
                kernel function instead of the default integer indexing.
                for example: {0: 'i', 1: 'j'}, then
                this notation can be used in the kernel function e.g.:
                `f = lambda x: x['i+1','j+1']` is equivalent to lambda x: x[1,1]
            map_kind: the kind of map to be used. available options are:
                jax.vmap under `vmap`, jax.lax.map under `map`, and jax.lax.pmap
                under `pmap`.
            map_kwargs: optional kwargs to be passed to the map function.
                for example, `map_kwargs={'axis_name': 'i'}` will apply the
                function along the axis named `i` for `pmap`.

        Example:
            >>> import jax
            >>> import kernex as kex
            >>> @kex.smap(
            ...    # view shape
            ...    kernel_size=(2,),
            ...    # indexing is relative: x[0,0] points to
            ...    # array view center and not top left corner
            ...    relative=True,
            ...    # start the application of the function
            ...    # from the third column till the end (i.e. no offset at the end)
            ...    offset=((2,0),),
            ... )
            ... def F(x):
            ...    return x[0] + x[1]
            >>> x = jax.numpy.arange(1,6)
            >>> print(x)
            [1 2 3 4 5]
            >>> print(F(x))
            [1 2 7 9 5]

        Note:
            - Unlike `kmap` this transformation expects a scalar function output.
                additionally, it is always applying within the bounds of the
                input array, i.e. shape of the output is the same as the input.
            - The previous example can be decomposed into the following steps:
            # view shape       # process
            v1:=[1,2]          skip as offset is 2
            v2:=[2,3]          skip as offset is 2
            v3:=[3,4]          apply F(v3) = 3+4 = 7
            v4:=[4,5]          apply F(v4) = 4+5 = 9
            v5:=[5,0]          apply F(v5) = 5+0 = 5
            ==================================================
            output:            [1,2,7,9,5]
        """

        super().__init__(
            kernel_size=kernel_size,
            strides=strides,
            border=offset,
            relative=relative,
            inplace=False,
            use_offset=True,
            named_axis=named_axis,
            transform_kind=map_kind,
            transform_kwargs=map_kwargs,
        )


class kscan(KernelInterface):
    def __init__(
        self,
        kernel_size: KernelSizeType = 1,
        strides: StridesType = 1,
        padding: BorderType = 0,
        relative: bool = False,
        named_axis: dict[int, str] = None,
        scan_kind: ScanKind = "scan",
        scan_kwargs: dict[str, Any] | None = None,
    ):
        """Apply a function to a sliding window of the input array sequentially.

        Args:
            kernel_size: size of the kernel must be a tuple of integers.
            strides: strides of the kernel, int or tuple of ints.
            padding: padding of the kernel, int or tuple of ints.
            relative: if True, the kernel is relative to the current index.
            named_axis: optional dictionary of named axis to be used in the
                kernel function instead of the default integer indexing.
                for example: {0: 'i', 1: 'j'}, then
                this notation can be used in the kernel function e.g.:
                `f = lambda x: x['i+1','j+1']` is equivalent to lambda x: x[1,1]
            scan_kind: the kind of scan to be used. available options are:
                jax.lax.scan under `scan`.
            scan_kwargs: optional kwargs to be passed to the scan function.
                for example, `scan_kwargs={'reverse': True}` will reverse the
                application of the function.

        Returns:
            A function that takes an array as input and returns the result of
            the kernel function applied to the array.

        Example:
            >>> import kernex as kex
            >>> import jax.numpy as jnp
            >>> @kex.kscan(kernel_size=(3,))
            ... def sum_all(x):
            ...     return jnp.sum(x)
            >>> x = jnp.array([1,2,3,4,5])
            >>> print(sum_all(x))
            [ 6 13 22]

        Note:
            - The previous example is equivalent to the following:
            v1 := [1,2,3] -> sum(v1) = 6
            v2 := [6,3,4] -> sum(v2) = 13
            v3 := [13,4,5] -> sum(v3) = 22
            Where vi is the ith view of the input array.
        """
        super().__init__(
            kernel_size=kernel_size,
            strides=strides,
            border=padding,
            relative=relative,
            inplace=True,
            use_offset=False,
            named_axis=named_axis,
            transform_kind=scan_kind,
            transform_kwargs=scan_kwargs,
        )


class kmap(KernelInterface):
    def __init__(
        self,
        kernel_size: KernelSizeType = 1,
        strides: StridesType = 1,
        padding: BorderType = 0,
        relative: bool = False,
        named_axis: dict[int, str] = None,
        map_kind: MapKind = "vmap",
        map_kwargs: dict = None,
    ):
        """Apply a function to a sliding window of the input array in parallel.

        Args:
            kernel_size: size of the kernel must be a tuple of integers.
            strides: strides of the kernel, int or tuple of ints.
            padding: padding of the kernel, int or tuple of ints.
            relative: if True, the kernel is relative to the current index.
            named_axis: optional dictionary of named axis to be used in the
                kernel function instead of the default integer indexing.
                for example: {0: 'i', 1: 'j'}, then
                this notation can be used in the kernel function e.g.:
                `f = lambda x: x['i+1','j+1']` is equivalent to lambda x: x[1,1]
            map_kind: the kind of map to be used. available options are:
                jax.vmap under `vmap`, jax.lax.map under `map`, and jax.lax.pmap
                under `pmap`.
            map_kwargs: optional kwargs to be passed to the map function.
                for example, `map_kwargs={'axis_name': 'i'}` will apply the
                function along the axis named `i` for `pmap`.

        Returns:
            A function that takes an array as input and applies the kernel

        Example:
            >>> # as a function decorator
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
            Where vi is the ith view of the input array.
        """
        super().__init__(
            kernel_size=kernel_size,
            strides=strides,
            border=padding,
            relative=relative,
            inplace=False,
            use_offset=False,
            named_axis=named_axis,
            transform_kind=map_kind,
            transform_kwargs=map_kwargs,
        )

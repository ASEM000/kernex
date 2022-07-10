"""
[credits] Mahmoud Asem@CVBML KAIST May 2022
This script defines the decorators and class object
"""

from __future__ import annotations

import functools
from typing import Callable

from jax import numpy as jnp

from kernex.interface.named_axis import named_axis_wrapper
from kernex.interface.resolve_utils import (
    resolve_index,
    resolve_offset_argument,
    resolve_padding_argument,
)
from kernex.src.map import kernelMap, offsetKernelMap
from kernex.src.scan import kernelScan, offsetKernelScan
from kernex.src.utils import ZIP
from kernex.treeclass.decorator import static_field, treeclass


@treeclass
class kernexClass(dict):

    kernel_size: tuple[int, ...] | int = static_field()
    strides: tuple[int, ...] | int = static_field(default=1)
    border: tuple[int, ...] | tuple[tuple[int, int],
                                    ...] | int | str = static_field(default=0,
                                                                    repr=False)
    relative: bool = static_field(default=False)
    inplace: bool = static_field(default=False)
    use_offset: bool = static_field(default=False)
    named_axis: dict[int, str] | None = static_field(default=None)

    def __post_init__(self):

        self.border = (resolve_offset_argument if self.use_offset else
                       resolve_padding_argument)(self.border, self.kernel_size)

        if self.inplace:
            self.kernel_op = offsetKernelScan if self.use_offset else kernelScan

        else:
            self.kernel_op = offsetKernelMap if self.use_offset else kernelMap

        self.__resolved__ = False

    def __post_resolutions__(self):

        if self.__resolved__:
            return

        self.__resolved__ = True

        if isinstance(self.kernel_size, tuple):
            assert  all(isinstance(wi, int) for wi in self.kernel_size), \
                ("kernel_size  input must be a tuple of int.\n",
                f"Found {tuple(type(wi) for wi in self.kernel_size  )}")

            assert  len(self.kernel_size) == len(self.shape), \
                ("kernel_size  dimension must be equal to array dimension.",
                f"Found len({self.kernel_size }) != len{(self.shape)}")

            assert  all(ai <= si for (ai, si) in zip(self.kernel_size, self.shape)),\
                ("kernel_size  shape must be less than array shape.\n",
                f"Found kernel_size  = {self.kernel_size } array shape = {self.shape} ")

            # convert kernel_size  = -1 to shape dimension
            self.kernel_size = tuple(
                si if wi == -1 else wi
                for si, wi in ZIP(self.shape, self.kernel_size))

        elif isinstance(self.kernel_size, int):
            self.kernel_size = (self.kernel_size, ) * len(self.shape)

        else:
            raise ValueError(
                f"kernel_size  must be instance of int or tuple. Found {type(self.kernel_size )}"
            )

        if isinstance(self.strides, tuple):
            assert  all(isinstance(wi, int) for wi in self.strides),\
                f"strides  input must be a tuple of int. Found {tuple(type(wi) for wi in self.strides  )}"

            assert  len(self.strides) == len(self.shape),\
                ("strides  dimension must be equal to array dimension." ,
                f"Found len({self.strides }) != len{(self.shape)}")

            assert  all(ai <= si for (ai, si) in zip(self.strides, self.shape)),\
                ("strides  shpae must be less than array shape.\n" ,
                f"Found strides  = {self.strides } array shape = {self.shape}")

        elif isinstance(self.strides, int):
            self.strides = (self.strides, ) * len(self.shape)

        else:
            raise ValueError(
                f"strides  must be instance of int or tuple. Found {type(self.strides )}"
            )

        # normalize slices
        for func, slices in self.items():
            slices = [resolve_index(index, self.shape) for index in slices]
            super().__setitem__(func, slices)

    def __setitem__(self, index, func):

        assert  isinstance(func, Callable),\
            f"Input must be of type Callable. Found {type(func)}"

        if func in self:
            index = self[func] + [index]
            super().__setitem__(func, index)

        else:
            super().__setitem__(func, [index])

    def __mesh_call__(self, array, *args, **kwargs):

        self.shape = array.shape
        self.__post_resolutions__()

        self.func_dict = {}

        for (func, index) in self.items():

            if func is not None and self.named_axis is not None:
                transformed_function = named_axis_wrapper(
                    self.kernel_size, self.named_axis)(func)
                self.func_dict[transformed_function] = index
            
            else:
                self.func_dict[func] = index

        return self.kernel_op(self.func_dict, self.shape, self.kernel_size,
                              self.strides, self.border,
                              self.relative)(array, *args, **kwargs)

    def __decorator_call__(self, func):

        def call(array, *args, **kwargs):
            self.shape = array.shape
            self.__post_resolutions__()

            self.func_dict = {
                named_axis_wrapper(self.kernel_size, self.named_axis)(func) if self.named_axis is not None else func:
                ()
            }

            return self.kernel_op(self.func_dict, self.shape, self.kernel_size,
                                  self.strides, self.border,
                                  self.relative)(array, *args, **kwargs)

        return call

    def __call__(self, *args, **kwargs):

        if len(args) == 1 and callable(args[0]) and len(kwargs) == 0:
            return functools.wraps(args[0])(self.__decorator_call__(args[0]))

        elif len(args) > 0 and isinstance(args[0], jnp.ndarray):
            return self.__mesh_call__(*args, **kwargs)

        else:
            raise ValueError(
                f"expected jnp.ndarray or Callable for the first argument. Found {tuple(*args,**kwargs)}"
            )


@treeclass
class sscan(kernexClass):

    def __init__(self,
                 kernel_size=1,
                 strides=1,
                 offset=0,
                 relative=False,
                 named_axis=None):

        super().__init__(kernel_size=kernel_size,
                         strides=strides,
                         border=offset,
                         relative=relative,
                         inplace=True,
                         use_offset=True,
                         named_axis=named_axis)


@treeclass
class smap(kernexClass):

    def __init__(self,
                 kernel_size=1,
                 strides=1,
                 offset=0,
                 relative=False,
                 named_axis=None):

        super().__init__(kernel_size=kernel_size,
                         strides=strides,
                         border=offset,
                         relative=relative,
                         inplace=False,
                         use_offset=True,
                         named_axis=named_axis)


@treeclass
class kscan(kernexClass):

    def __init__(self,
                 kernel_size=1,
                 strides=1,
                 padding=0,
                 relative=False,
                 named_axis=None):

        super().__init__(kernel_size=kernel_size,
                         strides=strides,
                         border=padding,
                         relative=relative,
                         inplace=True,
                         use_offset=False,
                         named_axis=named_axis)


@treeclass
class kmap(kernexClass):

    def __init__(self,
                 kernel_size=1,
                 strides=1,
                 padding=0,
                 relative=False,
                 named_axis=None):

        super().__init__(kernel_size=kernel_size,
                         strides=strides,
                         border=padding,
                         relative=relative,
                         inplace=False,
                         use_offset=False,
                         named_axis=named_axis)

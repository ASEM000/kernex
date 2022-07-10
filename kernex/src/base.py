from __future__ import annotations

import functools
from itertools import product
from typing import Any, Callable

import jax
from jax import numpy as jnp

from kernex.src.utils import (
    ZIP,
    general_arange,
    general_product,
    index_from_view,
    key_search,
)
from kernex.treeclass.decorator import static_field, treeclass


@treeclass
class kernelOperation:
    """base class all kernel operations"""

    func_dict: dict[Callable[[Any], jnp.ndarray:tuple[int, ...],
                             ...]] = static_field()
    shape: tuple[int, ...] = static_field()
    kernel_size: tuple[int, ...] = static_field()
    strides: tuple[int, ...] = static_field()
    border: tuple[tuple[int, int], ...] = static_field()
    relative: bool = static_field()

    # @functools.cached_property
    @property
    def pad_width(self):
        return tuple([0, max(0, pi[0]) + max(0, pi[1])] for pi in self.border)

    # @functools.cached_property
    @property
    def output_shape(self):
        return tuple(
            (xi + (li + ri) - ki) // si + 1 for xi, ki, si, (li, ri) in ZIP(
                self.shape, self.kernel_size, self.strides, self.border))

    # @functools.cached_property
    @property
    @functools.partial(jax.profiler.annotate_function, name="views")
    def views(self) -> tuple[jnp.ndarray, ...]:
        """Generate absolute sampling matrix"""
        dim_range = tuple(
            general_arange(di, ki, si, x0, xf) for (di, ki, si, (x0, xf)) in
            zip(self.shape, self.kernel_size, self.strides, self.border))

        matrix = general_product(*dim_range)
        return tuple(
            map(lambda xi, wi: xi.reshape(-1, wi), matrix, self.kernel_size))

    # @functools.cached_property
    @property
    def indices(self):
        return tuple(product(*[range(d) for d in self.shape]))

    def func_index_from_view(self, view: tuple[jnp.ndarray, ...]):

        return key_search(key=tuple(index_from_view(view, self.kernel_size)),
                          keys=self.slices)

    # @functools.cached_property
    @property
    def funcs(self):
        return tuple(self.func_dict.keys())

    # @functools.cached_property
    @property
    def slices(self):
        return tuple(self.func_dict.values())

    def index_from_view(self, view):
        return tuple(view[i][wi // 2] if wi % 2 == 1 else view[i][(wi - 1) //2]
                     for i, wi in enumerate(self.kernel_size))

    # @staticmethod
    # def patch_from_view(view,array):
    #     return array[ix_(*view)]

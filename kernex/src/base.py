from __future__ import annotations

from itertools import product
from typing import Any, Callable

import pytreeclass as pytc
from jax import numpy as jnp
from pytreeclass.src.decorator_util import cached_property

from kernex.src.utils import ZIP, _key_search, general_arange, general_product


@pytc.treeclass
class kernelOperation:
    """base class for all kernel operations"""

    func_index_map: dict[Callable[[Any], jnp.ndarray] : tuple[int, ...]] = pytc.static_field(repr=False)  # fmt: skip
    shape: tuple[int, ...] = pytc.static_field()
    kernel_size: tuple[int, ...] = pytc.static_field()
    strides: tuple[int, ...] = pytc.static_field()
    border: tuple[tuple[int, int], ...] = pytc.static_field()
    relative: bool = pytc.static_field()

    @cached_property
    def pad_width(self):
        """Calcuate the positive padding from border value

        Returns:
            padding value passed to `pad_width` in `jnp.pad`
        """
        return tuple([0, max(0, pi[0]) + max(0, pi[1])] for pi in self.border)

    @cached_property
    def output_shape(self) -> tuple[int, ...]:
        """Calculate the output shape of the kernel operation from
        the input shape, kernel size, stride and border.

        Returns:
            tuple[int, ...]: resulting shape of the kernel operation
        """
        return tuple(
            (xi + (li + ri) - ki) // si + 1
            for xi, ki, si, (li, ri) in ZIP(
                self.shape, self.kernel_size, self.strides, self.border
            )
        )

    @cached_property
    def views(self) -> tuple[jnp.ndarray, ...]:
        """Generate absolute sampling matrix"""
        dim_range = tuple(
            general_arange(di, ki, si, x0, xf)
            for (di, ki, si, (x0, xf)) in zip(
                self.shape, self.kernel_size, self.strides, self.border
            )
        )

        matrix = general_product(*dim_range)
        return tuple(map(lambda xi, wi: xi.reshape(-1, wi), matrix, self.kernel_size))

    @cached_property
    def indices(self):
        return tuple(product(*[range(d) for d in self.shape]))

    def func_index_from_view(self, view: tuple[jnp.ndarray, ...]) -> int:
        """find the index of the function from the view

        Args:
            view (tuple[jnp.ndarray, ...]): patch indices for each dimension

        Returns:
            int: function index in the `func_index_map.values()`
        """
        return _key_search(key=tuple(self.index_from_view(view)), keys=self.slices)

    @property
    def funcs(self) -> tuple[Callable[[Any], jnp.ndarray]]:
        return tuple(self.func_index_map.keys())

    @property
    def slices(self):
        return tuple(self.func_index_map.values())

    def index_from_view(self, view: tuple[jnp.ndarray, ...]) -> tuple[int, ...]:
        """Get the index of array from the view

        Args:
            view (tuple[jnp.ndarray,...]): patch indices for each dimension

        Returns:
            tuple[int, ...]: index as a tuple of int for each dimension
        """
        return tuple(
            view[i][wi // 2] if wi % 2 == 1 else view[i][(wi - 1) // 2]
            for i, wi in enumerate(self.kernel_size)
        )

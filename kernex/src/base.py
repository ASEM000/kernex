from __future__ import annotations

from itertools import product
from typing import Any, Callable

from jax import numpy as jnp
from pytreeclass import static_field, treeclass
from pytreeclass.src.decorator_util import cached_property

from kernex.src.utils import (
    ZIP,
    general_arange,
    general_product,
    index_from_view,
    key_search,
)


@treeclass(op=False)
class kernelOperation:
    """base class for all kernel operations"""

    func_dict: dict[Callable[[Any], jnp.ndarray:tuple[int, ...], ...]] = static_field()  # fmt: skip
    shape: tuple[int, ...] = static_field()
    kernel_size: tuple[int, ...] = static_field()
    strides: tuple[int, ...] = static_field()
    border: tuple[tuple[int, int], ...] = static_field()
    relative: bool = static_field()

    @cached_property
    def pad_width(self):
        """Calcuate the positive padding from border value

        Returns:
            padding value passed to `pad_width` in `jnp.pad`
        
        Example :
        """
        return tuple([0, max(0, pi[0]) + max(0, pi[1])] for pi in self.border)

    @cached_property
    def output_shape(self):
        """Calculate the resultant output shape given 

        Returns:
            _type_: _description_
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

    def func_index_from_view(self, view: tuple[jnp.ndarray, ...]):
        return key_search(
            key=tuple(index_from_view(view, self.kernel_size)), keys=self.slices
        )

    @property
    def funcs(self):
        return tuple(self.func_dict.keys())

    @property
    def slices(self):
        return tuple(self.func_dict.values())

    def index_from_view(self, view):
        return tuple(
            view[i][wi // 2] if wi % 2 == 1 else view[i][(wi - 1) // 2]
            for i, wi in enumerate(self.kernel_size)
        )

    # @staticmethod
    # def patch_from_view(view,array):
    #     return array[ix_(*view)]

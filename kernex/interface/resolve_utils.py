from __future__ import annotations

from typing import Sequence

import jax.numpy as jnp

# ---------------------- resolve_arguments ------------------------ #


def resolve_padding_argument(
        input_argument: tuple[int | tuple[int, int] | str, ...] | int | str,
        kernel_size: tuple[int, ...]):
    """Helper function to generate padding"""
    same = lambda wi: ((wi - 1) // 2, wi // 2)  # trunk-ignore(flake8/E731)

    if isinstance(input_argument, tuple):

        assert  len(input_argument) == len( kernel_size), \
                f"kernel_size dimension != padding dimension.Found length(kernel_size)={len(kernel_size)} length(padding)={len(input_argument)}"

        padding = [[]] * len(kernel_size)

        for i, item in enumerate(input_argument):

            if isinstance(item, int):
                padding[i] = (item, item)

            elif isinstance(item, tuple):
                padding[i] = item

            elif isinstance(item, str):

                if item in ["same", "SAME"]:
                    padding[i] = same(kernel_size[i])

                elif item in ["valid", "VALID"]:
                    padding[i] = (0, 0)

                else:
                    raise ValueError(
                        f'string argument must be in ["same","SAME","VALID","valid"].Found {item}'
                    )

    elif isinstance(input_argument, str):

        if input_argument in ["same", "SAME", "Same"]:
            return tuple(same(wi) for wi in kernel_size)

        elif input_argument in ["valid", "VALID", "Valid"]:
            return ((0, 0), ) * len(kernel_size)

        else:
            raise ValueError(
                f'string argument must be in ["same","SAME","VALID","valid"].Found {input_argument}'
            )

    elif isinstance(input_argument, int):
        padding = ((input_argument, input_argument), ) * len(kernel_size)

    return tuple(padding)


def resolve_dict_argument(input_dict: dict[str, int], dim: int, default):
    """
    --- Explanation
        return a tuple that map values of dict into tuple

    --- Examples
    >>> resolve_dict_argument({0:1,1:2,2:3},dim=3,default=0) -> (1,2,3)
    >>> resolve_dict_argument({1:2,0:1,2:3},dim=3,default=0) -> (1,2,3)
    >>> resolve_dict_argument({0:1},dim=3,default=0) -> (1,0,0)
    >>> resolve_dict_argument({0:(1,0)},dim=3,default=0) -> ((1,0),(0,0),(0,0)) # tuple is inferred

    """
    # assign mutable list
    temp = [default] * dim

    for dim, val in input_dict.items():
        temp[dim] = val

    return tuple(temp)


def resolve_offset_argument(input_argument, kernel_size):

    if isinstance(input_argument, int):
        return [(input_argument, input_argument)] * len(kernel_size)

    elif isinstance(input_argument, Sequence):

        offset = [[]] * len(kernel_size)

        for i, item in enumerate(input_argument):
            offset[i] = (item, item) if isinstance(item, int) else item

        return (offset)


def resolve_index(index, shape):
    """
    --- Explanation
        handles int and slice input arguments

    --- Examples
    >>> resolve_index((3,1,slice(None,None,None)))
        [3, 1, (0, inf, 1)]

    """

    index = [index] if not isinstance(index, tuple) else index
    resolved_index = [[]] * len(index)

    for i, (item, in_dim) in enumerate(zip(index, shape)):

        if isinstance(item, slice):
            start, end, step = item.start, item.stop, item.step

            start = start or 0
            start += in_dim if start < 0 else 0

            end = end or in_dim
            end += in_dim if end < 0 else 0

            step = step or 1

            resolved_index[i] = (start, end, step)

        # reduce [index] -> index
        elif isinstance(item, int):
            item += in_dim if item < 0 else 0
            resolved_index[i] = item  #(item,item+1,1)

        else:
            raise ValueError("type is not understood")

    return (resolved_index)

"""
[credits] Mahmoud Asem@CVBML KAIST May 2022
"""

from __future__ import annotations

import copy
import functools
from itertools import product
from typing import Callable

import jax.numpy as jnp


class sortedDict(dict):
    """a class that sort a key before setting or getting an item"""

    # this dict is used to store the kernel values
    # the key is a tuple of the axis names
    # the value is the kernel values
    # for example if the kernel is 3x3 and the axis names are ['x', 'y']
    # the key will be ('x', 'y') and the value will be the kernel values
    def __getitem__(self, key: tuple[str, ...]):
        key = (key,) if isinstance(key, str) else tuple(sorted(key))
        return super().__getitem__(key)

    def __setitem__(self, key: tuple[str, ...], val: jnp.ndarray):
        key = (key,) if isinstance(key, str) else tuple(sorted(key))
        super().__setitem__(key, val)


def generate_named_axis(
    kernel_size: tuple[int, ...], named_axis: dict[int, str], relative: bool = True
) -> dict[str, tuple[int, ...]]:
    """Return a dict that maps named axis to their integer value indices

    Args:
        kernel_size (tuple[int, ...]): kernel_size
        named_axis (dict[int, str]): axis argnum and its corresponding name
        relative (bool, optional): relative indexing boolean. Defaults to True.

    Raises:
        ValueError: Not int/str in named_axis

    Returns:
        dict[str, tuple[int, ...]]:
            [1] fully named axes (len(keys)==len(kernel_size ))
            return sortedDict object , where keys order is insignificant ( A['a','b']==A['b','a'])

            [2] partially named axes (len(keys)<len(kernel_size s))
            return dictionary object where key order matters.

    Examples:
        ### fully named axes case (inordered keys) :
        >>> generate_named_axis(named_axis={0:'b',1:'a'} , kernel_size =(2,3),relative=True)
            { ('b', 'a')    : (0, 0),
            ('b', 'a+1')  : (0, 1),
            ('b', 'a-1')  : (0, -1),
            ('b+1', 'a')  : (1, 0),
            ('b+1', 'a+1'): (1, 1),
            ('b+1', 'a-1'): (1, -1)}

        #### partially named axes case (ordered keys) (-1,'a') != ('a',-1):
        >>> generate_named_axis(named_axis={0:'b'} , kernel_size =(2,3),relative=True)
            {('b', -1)    : (0, -1),
            ('b', 0)    : (0, 0),
            ('b', 1)    : (0, 1),
            ('b+1', -1) : (1, -1),
            ('b+1', 0)  :  (1, 0),
            ('b+1', 1)  : (1, 1)}

        >>> generate_named_axis(named_axis={0:'b'} , kernel_size =(2,3),relative=False)
            { ('b', 0)    : (0, 0),
            ('b', 1)    : (0, 1),
            ('b', 2)    : (0, 2),
            ('b+1', 0)  : (1, 0),
            ('b+1', 1)  : (1, 1),
            ('b+1', 2)  : (1, 2),
            ('b+2', 0)  : (2, 0),
            ('b+2', 1)  : (2, 1),
            ('b+2', 2)  : (2, 2)}
    """
    # helper function to return range of sliding kernel_size  for a given dimension
    def range_func(wi):
        if relative:
            return tuple(range(-((wi - 1) // 2), (wi) // 2 + 1))

        else:
            return tuple(range(wi))

    # default case is numeric dimension maps to itself
    default_named_axis = {dim: dim for dim in range(len(kernel_size))}

    # replace the default keys
    default_named_axis.update(named_axis)

    # helepr function to return +d,-d,d if d>0,d=0,d<-0 respectively

    def operator_func(idx):
        return f"+{idx}" if idx > 0 else (f"{idx}" if idx < 0 else "")

    # partial named axes if not all dimensions are named
    partial_naming = not (len(kernel_size) == len(named_axis))

    keys = [[]] * len(kernel_size)
    vals = [[]] * len(kernel_size)

    # iterate over key,val in named_axis dict
    for dim, val in default_named_axis.items():
        if isinstance(val, str):
            # get keys for each dimension : [ ['i-m' ,..,'i+m'] , ['j-n',...,'j+n'] , .. ]
            # single charater case for each dimensoon
            # example {0:'i'}
            # index is incremented and decremented
            keys[dim] = [
                f"{val}{operator_func(idx)}" for idx in range_func(kernel_size[dim])
            ]
            vals[dim] = range_func(kernel_size[dim])

        # if named_axis = {0:('Q','C')} with kernel_size  = (2,)
        # then at index = 1 , it will work properly
        # at index =2 , x[Q] will return C , x[C] will return 0

        elif isinstance(val, int):
            # unnamed axis case
            partial_naming = True
            keys[dim] = range_func(kernel_size[dim])
            vals[dim] = range_func(kernel_size[dim])

        else:
            raise ValueError("Wrong format for named_axis.")

    keys = product(*keys)

    # reserve order if the named_axis are partially passed , otherwise its not order
    return_dict = dict() if partial_naming else sortedDict()

    # multiply [-m,...,m ] x [-n,...n] = [(-m,n) (-m,n-1) , .. ] to get the mesh integer indices
    vals = product(*vals)

    for k, v in zip(keys, vals):
        return_dict[k] = v

    # print(return_dict,type(return_dict))
    return return_dict


def named_axis_wrapper(kernel_size, named_axis):

    named_axis_dict = generate_named_axis(kernel_size, named_axis)
    x = copy.copy(named_axis_dict)

    def call(func: Callable):
        @functools.wraps(func)
        def inner(X: jnp.ndarray, *args, **kwargs):
            # switch the input of the function to operate on dictionary
            for k, idx in named_axis_dict.items():
                # assign the literal character keys to array numeric values
                x[k] = X[idx]
            return func(x, *args, **kwargs)

        return inner

    return call

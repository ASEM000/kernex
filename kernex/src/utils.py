"""
[credits] Mahmoud Asem@CVBML KAIST May 2022
"""

from __future__ import annotations

import functools

import jax
from jax import numpy as jnp
from jax import vmap


def ZIP(*args):
    """assert all args have the same number of elements before zipping"""
    n = len(args[0])

    assert all(len(x) == n for x in args[1:]), \
    f"zip arguments dont have the same length. Args length = {tuple(len(arg) for arg in args)}"

    return zip(*args)


def offset_to_padding(input_argument, kernel_size):
    """convert offset argument to negative border values"""

    padding = [[]] * len(kernel_size)

    # offset = 1 ==> padding= 0 for kernel_size =3
    # trunk-ignore(flake8/E731)
    same = lambda wi: ((wi - 1) // 2, wi // 2)

    for i, item in enumerate(input_argument):

        zero_offset_padding = same(kernel_size[i])

        if isinstance(item, tuple):
            assert  item >= (0,0,), \
                    f"offset must be non negative value.Found offset={item}"

            padding[i] = tuple(ai - bi
                               for ai, bi in ZIP(zero_offset_padding, item))

        elif isinstance(item, int):
            assert item >= 0, f"offset must be non negative value.Found offset={item}"
            padding[i] = tuple(
                ai - bi for ai, bi in ZIP(zero_offset_padding, (item, item)))

    # [TODO] throw an error if offset value is larger than kernel_size
    return tuple(padding)


# @functools.partial(jax.jit,static_argnums=(1,))
# def kron_broadcast(array:jnp.ndarray,kernel_size:tuple[int,...]):
#   '''
#       --- Explanation
#           broadcast input array by kernel_size shape on last axes

#       --- Example
#       >>> kron_broadcast(mat(3,3) , (5,5)).shape
#       ... (3,3,5,5)

#   '''
#   array   = jnp.expand_dims(array,axis=np.arange(-1,-len(kernel_size)-1,-1))
#   return jnp.kron(array,jnp.ones(kernel_size))


@jax.jit
@functools.partial(jax.profiler.annotate_function, name="roll_view")
def roll_view(array: jnp.ndarray) -> jnp.ndarray:
    """
        *** Explanation
            Given an n-dimesional array , rearrange the elements of 
            array to set the center element to top left of the array

        *** Args
            array : jnp array

        *** Output
            jnp array

        *** Examples

        >>> jnp.arange(1,6)     , roll_view(jnp.arange(1,6))
            (1   2   3   4   5) , (3   4   5   1   2)

        >>> mat = lambda n,m : jnp.arange(1,n*m+1).reshape(n,m)
        >>> mat(3,3)  , roll_view(mat(3,3))
            1   2   3     5   6   4
            4   5   6     8   9   7
            7   8   9     2   3   1

    """
    shape = jnp.array(array.shape)
    axes = tuple(range(len(shape)))  # list all axes
    shift = tuple(-(si // 2) if si % 2 == 1 else -((si - 1) // 2)
                  for si in array.shape)  # right padding>left padding
    return jnp.roll(array, shift=shift, axis=axes)

def ix_(*args):
    
    """modified version of jnp.ix_"""
    n = len(args)
    output = []
    for i, a in enumerate(args):
        shape = [1] * n
        shape[i] = a.shape[0]
        output.append(jax.lax.broadcast_in_dim(a, shape, (i, )))
    return tuple(output)


@functools.partial(jax.profiler.annotate_function, name="general_arange")
def general_arange(di: int, ki: int, si: int, x0: int, xf: int):
    start, end = -x0 + ((ki - 1) // 2), di + xf - (ki // 2)
    size = end - start

    lhs = jax.lax.broadcasted_iota(
        dtype=jnp.int32, shape=(size, ki), dimension=0) + (start)
    rhs = jax.lax.broadcasted_iota(
        dtype=jnp.int32, shape=(ki, size), dimension=0).T - ((ki - 1) // 2)
    res = lhs + rhs

    return (res) if si == 1 else (res)[::si]


@functools.partial(jax.profiler.annotate_function, name="general_product")
def general_product(*args):
    """`itertools.product` for arrays"""

    def nvmap(n):
        in_axes = [None] * len(args)
        in_axes[-n] = 0
        return (vmap(lambda *x: x, in_axes=in_axes)
                if n == 1 else vmap(nvmap(n - 1), in_axes=in_axes))

    return nvmap(len(args))(*args)


def index_from_view(view, kernel_size):
    """get array index from a given `view` and `kernel_size`"""
    return tuple(view[i][wi // 2] if wi % 2 == 1 else view[i][(wi - 1) // 2]
                 for i, wi in enumerate(kernel_size))


def compare_key(x: tuple[jnp.ndarray, ...], y: tuple[jnp.ndarray, ...]):

    def compare_key_item(xi: jnp.ndarray, yi: jnp.ndarray) -> jnp.ndarray:

        # index style = (start,end,step)
        if yi.size == 3:
            return (yi[0] <= xi) * (xi < yi[1]) * (xi % yi[2] == 0)

        # index style = (start,end )
        elif yi.size == 2:
            return (yi[0] <= xi) * (xi < yi[1])
        # index style = (index)
        elif yi.size == 1:
            return jnp.where(yi == jnp.inf, True, xi == yi)

    return jnp.all(
        jnp.array([compare_key_item(xi, yi) for (xi, yi) in zip(x, y)]))


def key_search(key, keys):
    """
    === Explanation

    
    === Args
        lhs_key : 
        a tuple of jnp.arrays for each dimension ( {dim0},...,{dimN}) with size({dim}) == 1 
 
        rhs_key : 
        a tuple of jnp.arrays for range of each dimension ( {dim0},...,{dimN}) with size({dim})
        with size ({dim}) in [1,2,3]

    === Example
        >>> compare_key((jnp.array([1]),),(jnp.array([0,2,1]))) # 1 in [0,2)
        True
    """

    length = len(keys)

    # [<(0,0),(0,1)> , <(0,0)>] ->
    # ( ({0},{0}), ({0},{1}) ) , ( ({0,0}) )
    # key = [jnp.array([ki]) for ki in key]

    def in_key_group(key, key_group):

        comparisons = tuple([
            compare_key(
                [jnp.array([ki]) for ki in key],  # tuples of array
                [jnp.array(ki) for ki in rhs_key])
        ] for rhs_key in key_group)

        # check if key match any key in the key_groupx
        return jnp.any(jnp.array(comparisons))

    def recurse(idx, keys):
        # under jit , no short circuit is possible .
        return jnp.where(
            in_key_group(key, keys[0]),
            idx,
            idx + 1 if idx == length - 1 else recurse(idx + 1, keys[1:]),
        )

    return recurse(0, keys[::-1]) if len(keys) > 0 else 0

from functools import reduce

import jax.numpy as jnp
from numpy.testing import assert_array_equal

from kernex._src.map import kernel_map, offset_kernel_map
from kernex._src.scan import kernel_scan, offset_kernel_scan


def mat(*args):
    """# helper function to construct nd arrays"""
    return jnp.arange(1, reduce(lambda x, y: x * y, args) + 1).reshape(
        *args
    )  # ignore-trunk


def test_kernel_map():

    array = mat(5)
    in_dim = (5,)
    kernel_size = (2,)
    strides = (1,)

    relative = False

    f0 = {lambda x: x[0] * 10: ()}

    padding = ((0, 1),)
    result = kernel_map(f0, in_dim, kernel_size, strides, padding, relative)(array)
    pred = jnp.array([10, 20, 30, 40, 50])
    assert_array_equal(result, pred)

    padding = ((-1, 1),)
    result = kernel_map(f0, in_dim, kernel_size, strides, padding, relative)(array)
    pred = jnp.array([20, 30, 40, 50])
    assert_array_equal(result, pred)

    padding = ((0, 0),)
    result = kernel_map(f0, in_dim, kernel_size, strides, padding, relative)(array)
    pred = jnp.array([10, 20, 30, 40])
    assert_array_equal(result, pred)

    padding = ((-1, 0),)
    result = kernel_map(f0, in_dim, kernel_size, strides, padding, relative)(array)
    pred = jnp.array([20, 30, 40])
    assert_array_equal(result, pred)

    padding = ((-2, 0),)
    result = kernel_map(f0, in_dim, kernel_size, strides, padding, relative)(array)
    pred = jnp.array([30, 40])
    assert_array_equal(result, pred)

    padding = ((-1, -1),)
    result = kernel_map(f0, in_dim, kernel_size, strides, padding, relative)(array)
    pred = jnp.array([20, 30])
    assert_array_equal(result, pred)

    padding = ((-2, -1),)
    result = kernel_map(f0, in_dim, kernel_size, strides, padding, relative)(array)
    pred = jnp.array([30])
    assert_array_equal(result, pred)

    f0 = {lambda x: x[0] + x[1]: ([[]]), lambda x: 90: [((0, 1, 1),)]}

    array = mat(5)
    in_dim = (5,)
    kernel_size = (3,)
    strides = (1,)
    relative = True

    padding = ((1, 1),)
    result = kernel_map(f0, in_dim, kernel_size, strides, padding, relative)(array)
    pred = jnp.array([90, 5, 7, 9, 5])
    assert_array_equal(result, pred)

    padding = ((0, 1),)
    result = kernel_map(f0, in_dim, kernel_size, strides, padding, relative)(array)
    pred = jnp.array([5, 7, 9, 5])
    assert_array_equal(result, pred)

    padding = ((1, 0),)
    result = kernel_map(f0, in_dim, kernel_size, strides, padding, relative)(array)
    pred = jnp.array([90, 5, 7, 9])
    assert_array_equal(result, pred)

    padding = ((0, 0),)
    result = kernel_map(f0, in_dim, kernel_size, strides, padding, relative)(array)
    pred = jnp.array([5, 7, 9])
    assert_array_equal(result, pred)

    padding = ((-1, -1),)
    result = kernel_map(f0, in_dim, kernel_size, strides, padding, relative)(array)
    pred = jnp.array([7])
    assert_array_equal(result, pred)

    padding = ((0, -1),)
    result = kernel_map(f0, in_dim, kernel_size, strides, padding, relative)(array)
    pred = jnp.array([5, 7])
    assert_array_equal(result, pred)

    padding = ((-1, 0),)
    result = kernel_map(f0, in_dim, kernel_size, strides, padding, relative)(array)
    pred = jnp.array([7, 9])
    assert_array_equal(result, pred)

    f0 = {
        lambda x: x[0] + x[1]: ([[]]),
        lambda x: 90: [((0, 1, 1),)],
        lambda x: 60: [((4, 5, 1),)],
    }

    array = mat(5)
    in_dim = (5,)
    kernel_size = (3,)
    strides = (1,)
    relative = True

    padding = ((1, 1),)
    result = kernel_map(f0, in_dim, kernel_size, strides, padding, relative)(array)
    pred = jnp.array([90, 5, 7, 9, 60])
    assert_array_equal(result, pred)

    f0 = {
        lambda x: x[-1] + x[0]: ([[]]),
        lambda x: 90: [((0, 1, 1),)],
        lambda x: 60: [((4, 5, 1),)],
    }

    array = mat(5)
    in_dim = (5,)
    kernel_size = (3,)
    strides = (1,)
    relative = True

    padding = ((1, 1),)
    result = kernel_map(f0, in_dim, kernel_size, strides, padding, relative)(array)
    pred = jnp.array([90, 3, 5, 7, 60])
    assert_array_equal(result, pred)

    f0 = {lambda x: x[0] * 10: ()}
    array = mat(5)
    in_dim = (5,)
    kernel_size = (2,)
    strides = (1,)
    padding = ((0, 1),)
    relative = True

    result = kernel_map(f0, in_dim, kernel_size, strides, padding, relative)(array)
    pred = jnp.array([10, 20, 30, 40, 50])
    assert_array_equal(result, pred)

    padding = ((-1, 1),)
    result = kernel_map(f0, in_dim, kernel_size, strides, padding, relative)(array)
    pred = jnp.array([20, 30, 40, 50])
    assert_array_equal(result, pred)

    padding = ((0, 0),)
    result = kernel_map(f0, in_dim, kernel_size, strides, padding, relative)(array)
    pred = jnp.array([10, 20, 30, 40])
    assert_array_equal(result, pred)

    padding = ((-1, 0),)
    result = kernel_map(f0, in_dim, kernel_size, strides, padding, relative)(array)
    pred = jnp.array([20, 30, 40])
    assert_array_equal(result, pred)

    padding = ((-2, 0),)
    result = kernel_map(f0, in_dim, kernel_size, strides, padding, relative)(array)
    pred = jnp.array([30, 40])
    assert_array_equal(result, pred)

    padding = ((-1, -1),)
    result = kernel_map(f0, in_dim, kernel_size, strides, padding, relative)(array)
    pred = jnp.array([20, 30])
    assert_array_equal(result, pred)

    padding = ((-2, -1),)
    result = kernel_map(f0, in_dim, kernel_size, strides, padding, relative)(array)
    pred = jnp.array([30])
    assert_array_equal(result, pred)

    f0 = {lambda x: x[0] + x[1] + x[-1]: ()}

    array = mat(5)
    in_dim = (5,)
    kernel_size = (3,)
    strides = (1,)
    relative = True

    padding = ((1, 1),)
    result = kernel_map(f0, in_dim, kernel_size, strides, padding, relative)(array)
    pred = jnp.array([3, 6, 9, 12, 9])
    assert_array_equal(result, pred)

    padding = ((0, 1),)
    result = kernel_map(f0, in_dim, kernel_size, strides, padding, relative)(array)
    pred = jnp.array([6, 9, 12, 9])
    assert_array_equal(result, pred)

    padding = ((1, 0),)
    result = kernel_map(f0, in_dim, kernel_size, strides, padding, relative)(array)
    pred = jnp.array([3, 6, 9, 12])
    assert_array_equal(result, pred)

    padding = ((0, 0),)
    result = kernel_map(f0, in_dim, kernel_size, strides, padding, relative)(array)
    pred = jnp.array([6, 9, 12])
    assert_array_equal(result, pred)

    padding = ((-1, 0),)
    result = kernel_map(f0, in_dim, kernel_size, strides, padding, relative)(array)
    pred = jnp.array([9, 12])
    assert_array_equal(result, pred)

    padding = ((0, -1),)
    result = kernel_map(f0, in_dim, kernel_size, strides, padding, relative)(array)
    pred = jnp.array([6, 9])
    assert_array_equal(result, pred)

    padding = ((-1, -1),)
    result = kernel_map(f0, in_dim, kernel_size, strides, padding, relative)(array)
    pred = jnp.array([9])
    assert_array_equal(result, pred)

    f0 = {lambda x: x[0, 0] * 10: ()}
    array = mat(3, 3)
    in_dim = (3, 3)
    kernel_size = (2, 2)
    strides = (1, 1)
    relative = True

    padding = ((0, 1), (0, 1))
    result = kernel_map(f0, in_dim, kernel_size, strides, padding, relative)(array)
    pred = mat(3, 3) * 10
    assert_array_equal(result, pred)

    padding = ((0, 1), (-1, 1))
    result = kernel_map(f0, in_dim, kernel_size, strides, padding, relative)(array)
    pred = jnp.array([[20, 30], [50, 60], [80, 90]])
    assert_array_equal(result, pred)

    padding = ((-1, 1), (0, 1))
    result = kernel_map(f0, in_dim, kernel_size, strides, padding, relative)(array)
    pred = jnp.array([[40, 50, 60], [70, 80, 90]])
    assert_array_equal(result, pred)

    padding = ((-1, 0), (0, 1))
    result = kernel_map(f0, in_dim, kernel_size, strides, padding, relative)(array)
    pred = jnp.array([[40, 50, 60]])
    assert_array_equal(result, pred)

    padding = ((0, 1), (-1, 0))
    result = kernel_map(f0, in_dim, kernel_size, strides, padding, relative)(array)
    pred = jnp.array([[20], [50], [80]])
    assert_array_equal(result, pred)

    padding = ((-1, 0), (-1, 0))
    result = kernel_map(f0, in_dim, kernel_size, strides, padding, relative)(array)
    pred = jnp.array([[50]])
    assert_array_equal(result, pred)

    f0 = {lambda x: x[0, 0, 0] * 10: ([[]])}
    array = mat(2, 3, 3)
    in_dim = (2, 3, 3)
    kernel_size = (1, 2, 2)
    strides = (1, 1, 1)
    relative = True

    padding = ((0, 0), (0, 1), (0, 1))
    result = kernel_map(f0, in_dim, kernel_size, strides, padding, relative)(array)
    pred = mat(2, 3, 3) * 10
    assert_array_equal(result, pred)

    padding = ((0, -1), (0, 1), (0, 1))
    result = kernel_map(f0, in_dim, kernel_size, strides, padding, relative)(array)
    pred = jnp.array([[[10, 20, 30], [40, 50, 60], [70, 80, 90]]])
    assert_array_equal(result, pred)

    padding = ((-1, 0), (0, 1), (0, 1))
    result = kernel_map(f0, in_dim, kernel_size, strides, padding, relative)(array)
    pred = jnp.array([[[100, 110, 120], [130, 140, 150], [160, 170, 180]]])
    assert_array_equal(result, pred)

    padding = ((0, 0), (-1, 1), (0, 1))
    result = kernel_map(f0, in_dim, kernel_size, strides, padding, relative)(array)
    pred = jnp.array([[[40, 50, 60], [70, 80, 90]], [[130, 140, 150], [160, 170, 180]]])
    assert_array_equal(result, pred)

    f0 = {lambda x: x[0] * -10: ([[]]), lambda x: x[0] * 10: [((0, 1, 1),)]}

    array = mat(5)
    in_dim = (5,)
    kernel_size = (2,)
    strides = (1,)
    padding = ((0, 1),)
    relative = True

    result = kernel_map(f0, in_dim, kernel_size, strides, padding, relative)(array)
    pred = jnp.array([10, -20, -30, -40, -50])
    assert_array_equal(result, pred)

    padding = ((-1, 1),)
    result = kernel_map(f0, in_dim, kernel_size, strides, padding, relative)(array)
    pred = jnp.array([-20, -30, -40, -50])
    assert_array_equal(result, pred)

    padding = ((0, 0),)
    result = kernel_map(f0, in_dim, kernel_size, strides, padding, relative)(array)
    pred = jnp.array([10, -20, -30, -40])
    assert_array_equal(result, pred)

    padding = ((-1, 0),)
    result = kernel_map(f0, in_dim, kernel_size, strides, padding, relative)(array)
    pred = jnp.array([-20, -30, -40])
    assert_array_equal(result, pred)

    padding = ((-2, 0),)
    result = kernel_map(f0, in_dim, kernel_size, strides, padding, relative)(array)
    pred = jnp.array([-30, -40])
    assert_array_equal(result, pred)

    padding = ((-1, -1),)
    result = kernel_map(f0, in_dim, kernel_size, strides, padding, relative)(array)
    pred = jnp.array([-20, -30])
    assert_array_equal(result, pred)

    padding = ((-2, -1),)
    result = kernel_map(f0, in_dim, kernel_size, strides, padding, relative)(array)
    pred = jnp.array([-30])
    assert_array_equal(result, pred)

    f0 = {lambda x: x[0, 0] * 10: ([[]])}
    array = mat(3, 3)
    in_dim = (3, 3)
    kernel_size = (2, 2)
    strides = (1, 1)
    relative = True

    padding = ((0, 1), (0, 1))
    result = kernel_map(f0, in_dim, kernel_size, strides, padding, relative)(array)
    pred = mat(3, 3) * 10
    assert_array_equal(result, pred)

    padding = ((0, 1), (-1, 1))
    result = kernel_map(f0, in_dim, kernel_size, strides, padding, relative)(array)
    pred = jnp.array([[20, 30], [50, 60], [80, 90]])
    assert_array_equal(result, pred)

    padding = ((-1, 1), (0, 1))
    result = kernel_map(f0, in_dim, kernel_size, strides, padding, relative)(array)
    pred = jnp.array([[40, 50, 60], [70, 80, 90]])
    assert_array_equal(result, pred)

    padding = ((-1, 0), (0, 1))
    result = kernel_map(f0, in_dim, kernel_size, strides, padding, relative)(array)
    pred = jnp.array([[40, 50, 60]])
    assert_array_equal(result, pred)

    padding = ((0, 1), (-1, 0))
    result = kernel_map(f0, in_dim, kernel_size, strides, padding, relative)(array)
    pred = jnp.array([[20], [50], [80]])
    assert_array_equal(result, pred)

    padding = ((-1, 0), (-1, 0))
    result = kernel_map(f0, in_dim, kernel_size, strides, padding, relative)(array)
    pred = jnp.array([[50]])
    assert_array_equal(result, pred)

    # f0 = { lambda x: -x[0,0]*10 :  ([[]]) ,
    #        lambda x: x[0,0]*10  : [ ((0,jnp.inf,1),(0,jnp.inf,1)) ]  }

    # array = mat(3,3)
    # in_dim = (3,3)
    # kernel_size = (2,2)
    # strides = (1,1)
    # relative = True

    # padding = ( (0,1),(-1,1) )
    # result = kernel_map(f0,in_dim,kernel_size,strides,padding,relative)(array)
    # pred = jnp.array([[20,30],[-50,-60],[-80,-90]])
    # assert_array_equal(result,pred)

    # padding = ( (-1,1) , (0,1) )
    # result = kernel_map(f0,in_dim,kernel_size,strides,padding,relative)(array)
    # pred = jnp.array([[-40,-50,-60],[-70,-80,-90]])
    # assert_array_equal(result,pred)

    # padding = ( (-1,0),(0,1) )
    # result = kernel_map(f0,in_dim,kernel_size,strides,padding,relative)(array)
    # pred = jnp.array([[-40,-50,-60]])
    # assert_array_equal(result,pred)

    # padding = ( (0,1),(-1,0) )
    # result = kernel_map(f0,in_dim,kernel_size,strides,padding,relative)(array)
    # pred = jnp.array([[20],[-50],[-80]])
    # assert_array_equal(result,pred)

    # padding = ( (-1,0) , (-1,0) )
    # result = kernel_map(f0,in_dim,kernel_size,strides,padding,relative)(array)
    # pred = jnp.array([[-50]])
    # assert_array_equal(result,pred)

    f0 = {lambda x: x[0, 0, 0] * 10: ([[]])}
    array = mat(2, 3, 3)
    in_dim = (2, 3, 3)
    kernel_size = (1, 2, 2)
    strides = (1, 1, 1)

    padding = ((0, 0), (0, 1), (0, 1))
    result = kernel_map(f0, in_dim, kernel_size, strides, padding, relative)(array)
    pred = mat(2, 3, 3) * 10
    assert_array_equal(result, pred)

    padding = ((0, -1), (0, 1), (0, 1))
    result = kernel_map(f0, in_dim, kernel_size, strides, padding, relative)(array)
    pred = jnp.array([[[10, 20, 30], [40, 50, 60], [70, 80, 90]]])
    assert_array_equal(result, pred)

    padding = ((-1, 0), (0, 1), (0, 1))
    result = kernel_map(f0, in_dim, kernel_size, strides, padding, relative)(array)
    pred = jnp.array([[[100, 110, 120], [130, 140, 150], [160, 170, 180]]])
    assert_array_equal(result, pred)

    padding = ((0, 0), (-1, 1), (0, 1))
    result = kernel_map(f0, in_dim, kernel_size, strides, padding, relative)(array)
    pred = jnp.array([[[40, 50, 60], [70, 80, 90]], [[130, 140, 150], [160, 170, 180]]])
    assert_array_equal(result, pred)

    f0 = {
        lambda x: x[0, 0, 0]: ([[]]),  # equivalent of [...]
        lambda x: -1: [
            ((0, 1, 1), (0, jnp.inf, 1), (0, jnp.inf, 1))
        ],  # equivalent of [0,...] = -1
    }

    padding = ((0, 0), (0, 1), (0, 1))
    result = kernel_map(f0, in_dim, kernel_size, strides, padding, relative)(array)
    pred = jnp.array(
        [
            [[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]],
            [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
        ]
    )

    assert_array_equal(result, pred)

    f0 = {
        lambda x: x[0, 0, 0]: ([[]]),  # equivalent of [...]
        lambda x: -1: [
            ((0, 1, 1), (0, 1, 1), (0, jnp.inf, 1))
        ],  # equivalent of [0,0,:] = -1
    }

    padding = ((0, 0), (0, 1), (0, 1))
    result = kernel_map(f0, in_dim, kernel_size, strides, padding, relative)(array)
    pred = jnp.array(
        [
            [[-1, -1, -1], [4, 5, 6], [7, 8, 9]],
            [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
        ]
    )

    assert_array_equal(result, pred)

    f0 = {
        lambda x: x[0, 0, 0]: ([[]]),  # equivalent of [...]
        lambda x: -1: [((0, 1, 1), (0, 1, 1), (0, 1, 1))],  # equivalent of [0,0,0] = -1
    }

    padding = ((0, 0), (0, 1), (0, 1))
    result = kernel_map(f0, in_dim, kernel_size, strides, padding, relative)(array)
    pred = jnp.array(
        [[[-1, 2, 3], [4, 5, 6], [7, 8, 9]], [[10, 11, 12], [13, 14, 15], [16, 17, 18]]]
    )

    assert_array_equal(result, pred)


def test_offset_kernel_map():

    f0 = {lambda x: x[0] * 10: ([[]])}
    array = mat(5)
    in_dim = (5,)
    kernel_size = (2,)
    strides = (1,)
    relative = True

    offset = ((0, 0),)
    result = offset_kernel_map(f0, in_dim, kernel_size, strides, offset, relative)(
        array
    )
    pred = jnp.array([10, 20, 30, 40, 50])
    assert_array_equal(result, pred)

    offset = ((1, 0),)
    result = offset_kernel_map(f0, in_dim, kernel_size, strides, offset, relative)(
        array
    )
    pred = jnp.array([1, 20, 30, 40, 50])
    assert_array_equal(result, pred)

    offset = ((0, 1),)
    result = offset_kernel_map(f0, in_dim, kernel_size, strides, offset, relative)(
        array
    )
    pred = jnp.array([10, 20, 30, 40, 5])
    assert_array_equal(result, pred)

    offset = ((1, 1),)
    result = offset_kernel_map(f0, in_dim, kernel_size, strides, offset, relative)(
        array
    )
    pred = jnp.array([1, 20, 30, 40, 5])
    assert_array_equal(result, pred)

    offset = ((2, 1),)
    result = offset_kernel_map(f0, in_dim, kernel_size, strides, offset, relative)(
        array
    )
    pred = jnp.array([1, 2, 30, 40, 5])
    assert_array_equal(result, pred)

    offset = ((1, 2),)
    result = offset_kernel_map(f0, in_dim, kernel_size, strides, offset, relative)(
        array
    )
    pred = jnp.array([1, 20, 30, 4, 5])
    assert_array_equal(result, pred)

    offset = ((2, 2),)
    result = offset_kernel_map(f0, in_dim, kernel_size, strides, offset, relative)(
        array
    )
    pred = jnp.array([1, 2, 30, 4, 5])
    assert_array_equal(result, pred)

    f0 = {lambda x: x[0] + x[1] + x[-1]: ([[]])}

    array = mat(5)
    in_dim = (5,)
    kernel_size = (3,)
    strides = (1,)
    relative = True

    offset = ((0, 0),)
    result = offset_kernel_map(f0, in_dim, kernel_size, strides, offset, relative)(
        array
    )
    pred = jnp.array([3, 6, 9, 12, 9])
    assert_array_equal(result, pred)

    offset = (
        (
            1,
            0,
        ),
    )
    result = offset_kernel_map(f0, in_dim, kernel_size, strides, offset, relative)(
        array
    )
    pred = jnp.array([1, 6, 9, 12, 9])
    assert_array_equal(result, pred)

    offset = ((0, 1),)
    result = offset_kernel_map(f0, in_dim, kernel_size, strides, offset, relative)(
        array
    )
    pred = jnp.array([3, 6, 9, 12, 5])
    assert_array_equal(result, pred)

    offset = ((1, 1),)
    result = offset_kernel_map(f0, in_dim, kernel_size, strides, offset, relative)(
        array
    )
    pred = jnp.array([1, 6, 9, 12, 5])
    assert_array_equal(result, pred)

    offset = ((2, 1),)
    result = offset_kernel_map(f0, in_dim, kernel_size, strides, offset, relative)(
        array
    )
    pred = jnp.array([1, 2, 9, 12, 5])
    assert_array_equal(result, pred)

    offset = ((1, 2),)
    result = offset_kernel_map(f0, in_dim, kernel_size, strides, offset, relative)(
        array
    )
    pred = jnp.array([1, 6, 9, 4, 5])
    assert_array_equal(result, pred)

    offset = ((2, 2),)
    result = offset_kernel_map(f0, in_dim, kernel_size, strides, offset, relative)(
        array
    )
    pred = jnp.array([1, 2, 9, 4, 5])
    assert_array_equal(result, pred)

    f0 = {lambda x: x[0, 0] * 10: ([[]])}
    array = mat(3, 3)
    in_dim = (3, 3)
    kernel_size = (2, 2)
    strides = (1, 1)
    relative = True

    offset = ((0, 0), (0, 0))
    result = offset_kernel_map(f0, in_dim, kernel_size, strides, offset, relative)(
        array
    )
    pred = mat(3, 3) * 10
    assert_array_equal(result, pred)

    offset = ((0, 0), (1, 0))
    result = offset_kernel_map(f0, in_dim, kernel_size, strides, offset, relative)(
        array
    )
    pred = jnp.array([[1, 20, 30], [4, 50, 60], [7, 80, 90]])
    assert_array_equal(result, pred)

    offset = ((1, 0), (0, 0))
    result = offset_kernel_map(f0, in_dim, kernel_size, strides, offset, relative)(
        array
    )
    pred = jnp.array([[1, 2, 3], [40, 50, 60], [70, 80, 90]])
    assert_array_equal(result, pred)

    offset = ((1, 1), (0, 0))
    result = offset_kernel_map(f0, in_dim, kernel_size, strides, offset, relative)(
        array
    )
    pred = jnp.array([[1, 2, 3], [40, 50, 60], [7, 8, 9]])
    assert_array_equal(result, pred)

    offset = ((0, 0), (1, 1))
    result = offset_kernel_map(f0, in_dim, kernel_size, strides, offset, relative)(
        array
    )
    pred = jnp.array([[1, 20, 3], [4, 50, 6], [7, 80, 9]])
    assert_array_equal(result, pred)

    offset = ((1, 1), (1, 1))
    result = offset_kernel_map(f0, in_dim, kernel_size, strides, offset, relative)(
        array
    )
    pred = jnp.array([[1, 2, 3], [4, 50, 6], [7, 8, 9]])
    assert_array_equal(result, pred)

    f0 = {lambda x: x[0, 0, 0] * 10: ([[]])}
    array = mat(2, 3, 3)
    in_dim = (2, 3, 3)
    kernel_size = (1, 2, 2)
    strides = (1, 1, 1)
    relative = True

    offset = ((0, 0), (0, 0), (0, 0))
    result = offset_kernel_map(f0, in_dim, kernel_size, strides, offset, relative)(
        array
    )
    pred = mat(2, 3, 3) * 10
    assert_array_equal(result, pred)

    offset = ((0, 1), (0, 0), (0, 0))
    result = offset_kernel_map(f0, in_dim, kernel_size, strides, offset, relative)(
        array
    )
    pred = jnp.array(
        [
            [[10, 20, 30], [40, 50, 60], [70, 80, 90]],
            [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
        ]
    )
    assert_array_equal(result, pred)

    offset = ((1, 0), (0, 0), (0, 0))
    result = offset_kernel_map(f0, in_dim, kernel_size, strides, offset, relative)(
        array
    )
    pred = jnp.array(
        [
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[100, 110, 120], [130, 140, 150], [160, 170, 180]],
        ]
    )
    assert_array_equal(result, pred)

    offset = ((0, 0), (1, 0), (0, 0))
    result = offset_kernel_map(f0, in_dim, kernel_size, strides, offset, relative)(
        array
    )
    pred = jnp.array(
        [
            [[1, 2, 3], [40, 50, 60], [70, 80, 90]],
            [[10, 11, 12], [130, 140, 150], [160, 170, 180]],
        ]
    )
    assert_array_equal(result, pred)


def test_kernel_scan():

    f0 = {lambda x: x[0] + x[1] + x[-1]: ([[]])}
    array = mat(5)
    in_dim = (5,)
    kernel_size = (3,)
    strides = (1,)
    relative = True

    padding = ((1, 1),)
    result = kernel_scan(f0, in_dim, kernel_size, strides, padding, relative)(array)
    pred = jnp.array([3, 8, 15, 24, 29])
    assert_array_equal(result, pred)

    padding = ((0, 1),)
    result = kernel_scan(f0, in_dim, kernel_size, strides, padding, relative)(array)
    pred = jnp.array([6, 13, 22, 27])
    assert_array_equal(result, pred)

    padding = ((1, 0),)
    result = kernel_scan(f0, in_dim, kernel_size, strides, padding, relative)(array)
    pred = jnp.array([3, 8, 15, 24])
    assert_array_equal(result, pred)

    padding = ((0, 0),)
    result = kernel_scan(f0, in_dim, kernel_size, strides, padding, relative)(array)
    pred = jnp.array([6, 13, 22])
    assert_array_equal(result, pred)

    padding = ((-1, 0),)
    result = kernel_scan(f0, in_dim, kernel_size, strides, padding, relative)(array)
    pred = jnp.array([9, 18])
    assert_array_equal(result, pred)

    padding = ((0, -1),)
    result = kernel_scan(f0, in_dim, kernel_size, strides, padding, relative)(array)
    pred = jnp.array([6, 13])
    assert_array_equal(result, pred)

    padding = ((-1, -1),)
    result = kernel_scan(f0, in_dim, kernel_size, strides, padding, relative)(array)
    pred = jnp.array([9])
    assert_array_equal(result, pred)

    f0 = {lambda x: jnp.sum(x): ([[]])}
    array = mat(5)
    in_dim = (5,)
    kernel_size = (3,)
    strides = (1,)
    relative = True

    padding = ((0, 0),)
    result = kernel_scan(f0, in_dim, kernel_size, strides, padding, relative)(array)
    pred = jnp.array([6, 13, 22])
    assert_array_equal(result, pred)

    f0 = {lambda x: x[-1] + x[0]: ([[]]), lambda x: 100: [((0, 1, 1),)]}

    array = mat(5)
    in_dim = (5,)
    kernel_size = (3,)
    strides = (1,)
    relative = True
    padding = ((1, 1),)
    result = kernel_scan(f0, in_dim, kernel_size, strides, padding, relative)(array)
    pred = jnp.array([100, 102, 105, 109, 114])
    assert_array_equal(result, pred)


def test_offset_kernel_scan():

    f0 = {lambda x: x[0] + x[1] + x[-1]: ([[]])}
    array = mat(5)
    in_dim = (5,)
    kernel_size = (3,)
    strides = (1,)
    relative = True

    offset = ((0, 0),)
    result = offset_kernel_scan(f0, in_dim, kernel_size, strides, offset, relative)(
        array
    )
    pred = jnp.array([3, 8, 15, 24, 29])
    assert_array_equal(result, pred)

    offset = ((1, 0),)
    result = offset_kernel_scan(f0, in_dim, kernel_size, strides, offset, relative)(
        array
    )
    pred = jnp.array([1, 6, 13, 22, 27])
    assert_array_equal(result, pred)

    offset = ((0, 1),)
    result = offset_kernel_scan(f0, in_dim, kernel_size, strides, offset, relative)(
        array
    )
    pred = jnp.array([3, 8, 15, 24, 5])
    assert_array_equal(result, pred)

    offset = ((1, 1),)
    result = offset_kernel_scan(f0, in_dim, kernel_size, strides, offset, relative)(
        array
    )
    pred = jnp.array([1, 6, 13, 22, 5])
    assert_array_equal(result, pred)

    offset = ((2, 1),)
    result = offset_kernel_scan(f0, in_dim, kernel_size, strides, offset, relative)(
        array
    )
    pred = jnp.array([1, 2, 9, 18, 5])
    assert_array_equal(result, pred)

    offset = ((1, 2),)
    result = offset_kernel_scan(f0, in_dim, kernel_size, strides, offset, relative)(
        array
    )
    pred = jnp.array([1, 6, 13, 4, 5])
    assert_array_equal(result, pred)

    offset = ((2, 2),)
    result = offset_kernel_scan(f0, in_dim, kernel_size, strides, offset, relative)(
        array
    )
    pred = jnp.array([1, 2, 9, 4, 5])
    assert_array_equal(result, pred)

    f0 = {lambda x: jnp.sum(x): ([[]])}
    array = mat(5)
    in_dim = (5,)
    kernel_size = (3,)
    strides = (1,)
    relative = True

    offset = ((1, 1),)
    result = offset_kernel_scan(f0, in_dim, kernel_size, strides, offset, relative)(
        array
    )
    pred = jnp.array([1, 6, 13, 22, 5])
    assert_array_equal(result, pred)

from dataclasses import field

import jax
import jax.numpy as jnp
import numpy as np

from kernex import treeclass


@treeclass
class Test:
    a: float
    b: float
    c: float
    name: str = field(metadata={'static': True})


def test_ops():

    @treeclass
    class Test:
        a: float
        b: float
        c: float
        name: str = field(metadata={'static': True})

    A = Test(10, 20, 30, 'A')
    # binary operations

    assert (A + A) == Test(20, 40, 60, 'A')
    assert (A - A) == Test(0, 0, 0, 'A')
    assert ((A['a'] + A) | A) == Test(20, 20, 30, 'A')
    assert A.reduce_mean() == jnp.array(60)
    assert abs(A) == A

    @treeclass
    class Test:
        a: int
        b: int
        name: str = field(metadata={'static': True})

    A = Test(-10, 20, 'A')

    # magic ops
    assert abs(A) == Test(10, 20, 'A')
    assert A + A == Test(-20, 40, 'A')
    assert (A == A)
    assert A // 2 == Test(-5, 10, 'A')
    assert A / 2 == Test(-5., 10., 'A')
    assert (A > A) == Test(False, False, 'A')
    assert (A >= A) == Test(True, True, 'A')
    assert (A <= A) == Test(True, True, 'A')
    assert -A == Test(10, -20, 'A')
    assert A * A == Test(100, 400, 'A')
    assert A**A == Test((-10)**(-10), 20**20, 'A')
    assert A - A == Test(0, 0, 'A')

    # numpy ops
    A = Test(a=jnp.array([-10, -10]), b=1, name='A')

    assert A.reduce_amax() == jnp.array(-9)
    assert A.reduce_amin() == jnp.array(-9)
    assert A.reduce_sum() == jnp.array(-19)
    assert A.reduce_prod() == jnp.array(100)
    assert A.reduce_mean() == jnp.array(-9)


def test_indexing():
    A = Test(10, 20, 30, 'A')
    assert A['a'] == Test(10, None, None, 'A')
    assert A[A >= 20] == Test(None, 20, 30, 'A')


def test_asdict():
    A = Test(10, 20, 30, 'A')
    assert A.asdict() == {'a': 10, 'b': 20, 'c': 30, 'name': 'A'}


def test_register_op():
    A = Test(10, 20, 30, 'A')
    A.register_op(func=jnp.prod,
                  name='product',
                  reduce_op=lambda x, y: x * y,
                  init_val=1)

    assert A.reduce_product() == 6000
    assert (A * A).reduce_mean() == 1400

from functools import reduce

import jax.numpy as jnp
from numpy.testing import assert_array_equal

from kernex._src.utils import roll_view

# helper function to construct nd arrays
mat = lambda *args: jnp.arange(1, reduce(lambda x, y: x * y, args) + 1).reshape(*args)


def test_roll_view():
    assert_array_equal(roll_view(mat(0)), jnp.array([]))

    assert_array_equal(roll_view(mat(1)), jnp.array(1))

    assert_array_equal(roll_view(mat(1, 2)), jnp.array([[1, 2]]))
    assert_array_equal(roll_view(mat(1, 3)), jnp.array([[2, 3, 1]]))

    assert_array_equal(roll_view(mat(3, 1, 1)), jnp.array([[[2]], [[3]], [[1]]]))

    assert_array_equal(roll_view(mat(2, 1)), jnp.array([[1], [2]]))
    assert_array_equal(roll_view(mat(3, 1)), jnp.array([[2], [3], [1]]))

    assert_array_equal(roll_view(mat(2, 3)), jnp.array([[2, 3, 1], [5, 6, 4]]))
    assert_array_equal(
        roll_view(mat(3, 3)), jnp.array([[5, 6, 4], [8, 9, 7], [2, 3, 1]])
    )


# def test_key_switch():

#   key = (3,)
#   keys = [ [(2,),(3,)] , [(4,)] , [(5,)] ]
#   in_dim = (10,)

#   pred = key_switch(format_item(key),format_keys(keys),format_item(in_dim))
#   assert pred == 0


#   key = (3,3)
#   keys = [ [(2,5),(3,3)] , [(4,4)] , [(5,9)] ]
#   in_dim = (10,)

#   pred = key_switch(format_item(key),format_keys(keys),format_item(in_dim))
#   assert pred == 0


#   key = (4,3)
#   keys = [ [(2,2),(3,1)] , [(4,3)] , [(5,8)] ]
#   in_dim = (10,)

#   pred = key_switch(format_item(key),format_keys(keys),format_item(in_dim))
#   assert pred == 1

#   key = (5,8)
#   keys = [ [(2,2),(3,1)] , [(4,3)] , [(5,8)] ]
#   in_dim = (10,)

#   pred = key_switch(format_item(key),format_keys(keys),format_item(in_dim))
#   assert pred == 2

#   key = (50,0)
#   keys = [  [ ( (0, jnp.inf, 1), 0) ] ]
#   in_dim = (100,)

#   pred = key_switch(format_item(key),format_keys(keys),format_item(in_dim))
#   assert pred == 0


# def test_key_switch():

#   key = (3,)
#   keys = [ [((1,2,1),),((3,4,1),)] , [((4,5,1),)] , [((5,6,1),)] ]
#   in_dim = (10,)

#   pred = key_switch((key),(keys),(in_dim))
#   assert pred == 0


#   key = (3,3)
#   keys = [ [((2,3,1),(5,6,1)),((3,4,1),(3,4,1))] , [((4,5,1),(4,5,1))] , [((5,6,1),(9,10,1))] ]
#   in_dim = (10,10)

#   pred = key_switch((key),(keys),(in_dim))
#   assert pred == 0


#   key = (4,3)
#   keys = [ [((2,3,1),(2,3,1)),((3,4,1),(1,2,1))] , [((4,5,1),(3,4,1))] , [((5,6,1),(8,9,10))] ]
#   in_dim = (10,10)

#   pred = key_switch((key),(keys),(in_dim))
#   assert pred == 1

#   key = (5,8)
#   keys = [ [((2,3,1),(2,3,1)),((3,4,1),(1,2,1))] , [((4,5,1),(3,4,1))] , [((5,6,1),(8,9,1))] ]
#   in_dim = (10,10)

#   pred = key_switch((key),(keys),(in_dim))
#   assert pred == 2

#   key = (50,0)
#   keys = [  [ ( (0, jnp.inf, 1), (0,1,1)) ] ]
#   in_dim = (100,100)

#   pred = key_switch((key),(keys),(in_dim))
#   assert pred == 0

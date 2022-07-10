import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax import tree_map
from jax.tree_util import tree_reduce


def is_treeclass(model):
    return hasattr(model, 'tree_fields')


def is_treeclass_leaf(model):

    if is_treeclass(model):
        fields = model.__dataclass_fields__.values()

        return (is_treeclass(model) and not any(
            [is_treeclass(model.__dict__[field.name]) for field in fields]))
    else:
        return False


def sequential_model_shape_eval(model, array):
    leaves = jax.tree_util.tree_leaves(model, is_treeclass_leaf)
    shape = [jax.eval_shape(lambda x: x, array)]
    for leave in leaves:
        shape += [jax.eval_shape(leave, shape[-1])]
    return shape


def node_class_name(node):
    return node.__class__.__name__


def node_size(node):
    '''get size of `trainable` and `non-trainable` parameters '''

    # store trainable in real , nontrainable in imag
    if isinstance(node, (jnp.ndarray, np.ndarray)):

        if jnp.issubdtype(node, jnp.inexact):
            return complex(int(node.nbytes), 0)

        else:
            return complex(0, int(node.nbytes))

    elif isinstance(node, (float, complex)):
        return complex(sys.getsizeof(node), 0)

    else:
        return complex(0, sys.getsizeof(node))


def node_count(node):
    '''count number of `trainable` and `non-trainable` parameters '''
    if isinstance(node, (jnp.ndarray, np.ndarray)):

        if jnp.issubdtype(node, jnp.inexact):
            return complex(int(jnp.array(node.shape).prod()), 0)

        else:
            return complex(0, int(jnp.array(node.shape).prod()))

    elif isinstance(node, (float, complex)):
        return complex(1, 0)

    elif isinstance(node, int):
        return complex(0, 1)

    else:
        return complex(0, 0)


def node_format(node):
    '''format shape and dtype of jnp.array'''

    if isinstance(node, (jnp.ndarray, jax.ShapeDtypeStruct)):
        replace_tuple = (("int", "i"), ("float", "f"), ("complex", "c"),
                         ("(", "["), (")", "]"), (" ", ""))

        formatted_string = f"{node.dtype}{jnp.shape(node)!r}"

        # trunk-ignore
        for lhs, rhs in replace_tuple:
            formatted_string = formatted_string.replace(lhs, rhs)
        return formatted_string

    else:
        return f"{node!r}"


def leaves_param_count(leaves):
    """returns param count for each leave"""
    return [
        tree_reduce(lambda acc, x: acc + node_count(x), leave, complex(0, 0))
        for leave in leaves
    ]


def leaves_param_size(leaves):
    """returns param count for each leave"""
    return [
        tree_reduce(lambda acc, x: acc + node_size(x), leave, 0)
        for leave in leaves
    ]


def leaves_param_format(leaves):
    return tree_map(lambda x: node_format(x), [leave for leave in leaves])


def leaves_param_count_and_size(leaves):
    """returns param count and param size for each leave"""

    def reduce_func(acc, x):
        cur_param_count = node_count(x)
        cur_param_size = node_size(x)
        prev_param_count, prev_param_size = acc

        return (cur_param_count + prev_param_count,
                cur_param_size + prev_param_size)

    return [
        tree_reduce(reduce_func, leave, (complex(0, 0), 0)) for leave in leaves
    ]

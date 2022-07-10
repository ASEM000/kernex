from __future__ import annotations

from dataclasses import dataclass, field

import jax

from .tree_base import treeBase, treeOpBase


def static_field(**kwargs):
    """ignore from pytree computations"""
    return field(**{**kwargs, **{"metadata": {"static": True}}})


def treeclass(cls):
    """Class decorator for `dataclass` to make it compaitable with JAX pytree"""

    user_defined_init = "__init__" in cls.__dict__

    dCls = dataclass(unsafe_hash=True, init=not user_defined_init,
                     repr=False)(cls)

    newCls = type(cls.__name__, (dCls, treeBase, treeOpBase), {})

    return jax.tree_util.register_pytree_node_class(newCls)

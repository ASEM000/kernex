from __future__ import annotations

import copy
import functools
import operator as op

import jax
from jax import numpy as jnp
from jax import tree_map
from jax.tree_util import tree_reduce

from .utils import is_treeclass_leaf
from .viz import tree_indent


def append_math_op(func):
    """binary and unary magic operations"""

    @functools.wraps(func)
    def call(self, rhs=None):

        if rhs is None:  # unary operation
            return tree_map(lambda x: func(x), self)

        elif isinstance(rhs, (int, float, complex, bool)):  # binary operation
            return (tree_map(lambda x: func(x, rhs), self)
                    if rhs is not None else self)

        elif isinstance(rhs, type(self)):  # class instance
            return tree_map(lambda x, y: func(x, y)
                            if y is not None else x, self, rhs)

        else:
            raise NotImplementedError(f"Found type(rhs) = {type(rhs)}")

    return call


def append_numpy_op(func):
    """array operations"""

    @functools.wraps(func)
    def call(self, *args, **kwargs):
        return tree_map(lambda node: func(node, *args, **kwargs), self)

    return call


def append_reduced_numpy_op(func, reduce_op, init_val):
    """reduced array operations"""

    @functools.wraps(func)
    def call(self, *args, **kwargs):
        return tree_reduce(
            lambda acc, cur: reduce_op(acc, func(cur, *args, **kwargs)), self,
            init_val)

    return call


class treeBase:

    @property
    def tree_fields(self):
        static, dynamic = dict(), dict()

        for field in self.__dataclass_fields__.values():
            value = self.__dict__[field.name]

            if ("static"
                    in field.metadata) and field.metadata["static"] is True:
                static[field.name] = value

            else:
                dynamic[field.name] = value

        return (dynamic, static)

    def tree_flatten(self):
        dynamic, static = self.tree_fields
        return (dynamic.values(), (dynamic.keys(), static))

    @classmethod
    def tree_unflatten(cls, aux, children):
        dynamic_vals, dynamic_keys = children, aux[0]

        static_keys, static_vals = aux[1].keys(), aux[1].values()
        attrs = dict(
            zip((*dynamic_keys, *static_keys), (*dynamic_vals, *static_vals)))

        newCls = cls.__new__(cls)
        for k, v in attrs.items():
            object.__setattr__(newCls, k, v)
        return newCls

    # @functools.cached_property
    @property
    def treeclass_leaves(self):
        return jax.tree_util.tree_leaves(self, is_treeclass_leaf)

    def __hash__(self):
        return hash(tuple(jax.tree_flatten(self)[0]))

    def __repr__(self):
        return tree_indent(self)

    def __str__(self):
        params_dict = {**self.tree_fields[0], **self.tree_fields[1]}
        return (f"{type(self).__name__}(" +
                ",".join([f"{k}={v}" for k, v in params_dict.items()]) + ")")

    def asdict(self):
        return {**self.tree_fields[0], **self.tree_fields[1]}

    def register_op(self, func, *, name, reduce_op=None, init_val=None):
        """register a math operation"""

        def element_call(*args, **kwargs):
            return jax.tree_map(lambda node: func(node, *args, **kwargs), self)

        setattr(self, name, element_call)

        if (reduce_op is not None) and (init_val is not None):

            def reduced_call(*args, **kwargs):
                return tree_reduce(
                    lambda acc, cur: reduce_op(acc, func(cur, *args, **kwargs)
                                               ), self, init_val)

            setattr(self, f"reduce_{name}", reduced_call)


class treeOpBase:

    __abs__ = append_math_op(op.abs)
    __add__ = append_math_op(op.add)
    __radd__ = append_math_op(op.add)
    __eq__ = append_math_op(op.eq)
    __floordiv__ = append_math_op(op.floordiv)
    __ge__ = append_math_op(op.ge)
    __gt__ = append_math_op(op.gt)
    __inv__ = append_math_op(op.inv)
    __invert__ = append_math_op(op.invert)
    __le__ = append_math_op(op.le)
    __lshift__ = append_math_op(op.lshift)
    __lt__ = append_math_op(op.lt)
    __matmul__ = append_math_op(op.matmul)
    __mod__ = append_math_op(op.mod)
    __mul__ = append_math_op(op.mul)
    __rmul__ = append_math_op(op.mul)
    __ne__ = append_math_op(op.ne)
    __neg__ = append_math_op(op.neg)
    __not__ = append_math_op(op.not_)
    __pos__ = append_math_op(op.pos)
    __pow__ = append_math_op(op.pow)
    __rshift__ = append_math_op(op.rshift)
    __sub__ = append_math_op(op.sub)
    __rsub__ = append_math_op(op.sub)
    __truediv__ = append_math_op(op.truediv)
    __xor__ = append_math_op(op.xor)

    imag = property(append_numpy_op(jnp.imag))
    real = property(append_numpy_op(jnp.real))
    conj = property(append_numpy_op(jnp.conj))

    abs = append_numpy_op(jnp.abs)
    amax = append_numpy_op(jnp.amax)
    amin = append_numpy_op(jnp.amin)
    arccos = append_numpy_op(jnp.arccos)
    arcsin = append_numpy_op(jnp.arcsin)
    sum = append_numpy_op(jnp.sum)
    prod = append_numpy_op(jnp.prod)
    mean = append_numpy_op(jnp.mean)

    reduce_abs = append_reduced_numpy_op(jnp.abs, op.add, 0)
    reduce_amax = append_reduced_numpy_op(jnp.amax, op.add, 0)
    reduce_amin = append_reduced_numpy_op(jnp.amin, op.add, 0)
    reduce_arccos = append_reduced_numpy_op(jnp.arccos, op.add, 0)
    reduce_arcsin = append_reduced_numpy_op(jnp.arcsin, op.add, 0)
    reduce_sum = append_reduced_numpy_op(jnp.sum, op.add, 0)
    reduce_prod = append_reduced_numpy_op(jnp.prod, op.mul, 1)
    reduce_mean = append_reduced_numpy_op(jnp.mean, op.add, 0)

    def __or__(self, rhs):
        return tree_map(lambda x, y: x or y,
                        self,
                        rhs,
                        is_leaf=lambda x: x is None)

    def __getitem__(self, *args):
        """return a new class with non chosen fields = None"""

        # select by param name
        if all(isinstance(arg, str) for arg in args):
            newCls = copy.copy(self)

            for field in newCls.__dataclass_fields__.values():
                if field.name not in (args) and "static" not in field.metadata:
                    newCls.__dict__[field.name] = None
            return newCls

        # select by class boolean
        elif len(args) == 1 and isinstance(args[0], type(self)):
            newCls = copy.copy(self)

            for field in newCls.__dataclass_fields__.values():
                argv = args[0].__dict__[field.name]

                is_static = (("static" not in field.metadata)
                             or ("static" in field.metadata
                                 and field.metadata["static"] is False))

                if (argv is False) and is_static:
                    newCls.__dict__[field.name] = None
            return newCls

        else:
            raise NotImplementedError(
                f"indexing with type{tuple(type(arg) for arg in args)} is not implemented."
            )

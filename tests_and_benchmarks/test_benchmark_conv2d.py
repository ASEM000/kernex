import itertools
import time
from functools import reduce
from statistics import mean, stdev

import jax
import jax.numpy as jnp
import numpy as np

import kernex as kex

mat = lambda *args: jnp.arange(1, reduce(lambda x, y: x * y, args) + 1).reshape(*args)


def test_and_time_conv2d():

    print()
    print("backend name = ", jax.devices())

    iters = 50

    dims = list(
        sorted(itertools.product([4, 8, 16, 32, 64], [16, 32, 64, 128, 256, 512, 1024]))
    )

    for C, H in dims:

        print()

        @jax.jit
        def jax_conv2d(x, w):
            return jax.lax.conv_general_dilated(
                lhs=x,
                rhs=w,
                window_strides=(1, 1),
                padding="SAME",
                dimension_numbers=("NCHW", "OIHW", "NCHW"),
            )[0]

        @jax.jit
        @kex.kmap(
            kernel_size=(C, 3, 3), padding=("valid", "same", "same"), relative=False
        )
        def kex_conv2d(x, w):
            return jnp.sum(x * w)

        times = []
        x = jax.random.normal(jax.random.PRNGKey(0), (C, H, H))
        xx = x[None]
        w = jax.random.normal(jax.random.PRNGKey(0), (C, 3, 3))
        ww = w[None]
        np.testing.assert_allclose(kex_conv2d(x, w), jax_conv2d(xx, ww), atol=1e-3)

        # warm up
        kex_conv2d(x, w)
        jax_conv2d(xx, ww)

        for _ in range(iters):
            t1 = time.time()
            kex_conv2d(x, w).block_until_ready()
            t2 = time.time()
            times += [(t2 - t1)]

        print(f"[benchmarking]:\tconv2d @ dim={(C,H,H)}")
        print("=" * 50)

        print(
            f"[kex] average : {mean(times)*1e6:.0f} us\t stddev : {stdev(times)*1e6:.3f}us"
        )

        times = []
        for _ in range(iters):
            t1 = time.time()
            jax_conv2d(xx, ww).block_until_ready()
            t2 = time.time()
            times += [t2 - t1]

        print(
            f"[jax] average : {mean(times)*1e6:.0f} us\t stddev : {stdev(times)*1e6:.3f}us"
        )

        print("=" * 50)

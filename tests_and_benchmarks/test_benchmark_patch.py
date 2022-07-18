import time
from functools import reduce
from statistics import mean, stdev

import jax
import jax.numpy as jnp
import numpy as np

import kernex as kex

mat = lambda *args: jnp.arange(1, reduce(lambda x, y: x * y, args) + 1).reshape(*args)


def test_and_time_patch():

    print()
    print("backend name = ", jax.devices())

    iters = 1_000

    dims = [10, 100, 1_000, 10_000, 100_000, 1_000_000]

    for dim in dims:


        print()

        @jax.jit
        @kex.kmap(kernel_size=(3,), padding="same", relative=False)
        def get_patches(x):
            return x

        @jax.jit
        def jax_get_patches(x):
            return jax.lax.conv_general_dilated_patches(x, (3,), (1,), padding="same")

        x = mat(dim)
        times = []
        get_patches(x)

        for _ in range(iters):
            t1 = time.time()
            get_patches(x).block_until_ready()  # warm up
            t2 = time.time()
            times += [(t2 - t1)]

        print(f"[benchmarking]:\tget_patch @ dim={dim:,}")
        print("=" * 50)

        print(
            f"[kex] average : {mean(times)*1e6:.0f} us\t stddev : {stdev(times)*10e6:.3f}us"
        )
        x = mat(1, 1, dim)
        jax_get_patches(x)
        times = []
        for _ in range(iters):
            t1 = time.time()
            jax_get_patches(x).block_until_ready()  # warm up
            t2 = time.time()
            times += [t2 - t1]

        print(
            f"[jax] average : {mean(times)*1e6:.0f} us\t stddev : {stdev(times)*10e6:.3f}us"
        )

        print("=" * 50)

        np.testing.assert_array_equal(
            get_patches(mat(dim)), jax_get_patches(mat(1, 1, dim)).reshape(-1, dim).T
        )

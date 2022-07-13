import jax
import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose

import kernex as kex

# # helper function to construct nd arrays
# mat   = lambda *args : jnp.arange(1,reduce(lambda x,y:x*y,args)+1).reshape(*args)


def test_Diffusion2D():
    @jax.jit
    @kex.sscan(
        kernel_size=(3, 3, 3),
        offset=((1, 0), (1, 1), (1, 1)),
        named_axis={0: "n", 1: "i", 2: "j"},
        relative=True,
    )
    def DIFFUSION2D(T, nu, dt, dx, dy):
        return (
            T["n-1", "i", "j"]
            + (nu * dt)
            / (dy**2)
            * (T["n-1", "i+1", "j"] - 2 * T["n-1", "i", "j"] + T["n-1", "i-1", "j"])
            + (nu * dt)
            / (dx**2)
            * (T["n-1", "i", "j+1"] - 2 * T["n-1", "i", "j"] + T["n-1", "i", "j-1"])
        )

    # variable declarations
    nx = 10
    ny = 10
    nt = 5
    nu = 1.0
    dx = 2 / (nx - 1)
    dy = 2 / (ny - 1)
    sigma = 0.25
    dt = sigma * dx * dy / nu

    # x = np.linspace(0, 2, nx)
    # y = np.linspace(0, 2, ny)

    u = np.ones((ny, nx))  # create a 1xn vector of 1's
    # un = np.ones((ny, nx))
    # Assign initial conditions
    # set hat function I.C. : u(.5<=x<=1 && .5<=y<=1 ) is 2
    u[int(0.5 / dy) : int(1 / dy + 1), int(0.5 / dx) : int(1 / dx + 1)] = 2

    U = jnp.ones([nt + 3, ny, nx])
    U = U.at[0].set(u)

    def diffuse(nt):
        u[int(0.5 / dy) : int(1 / dy + 1), int(0.5 / dx) : int(1 / dx + 1)] = 2

        for n in range(nt + 1):
            un = u.copy()
            u[1:-1, 1:-1] = (
                un[1:-1, 1:-1]
                + nu
                * dt
                / dx**2
                * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2])
                + nu
                * dt
                / dy**2
                * (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])
            )
            u[0, :] = 1
            u[-1, :] = 1
            u[:, 0] = 1
            u[:, -1] = 1

    diffuse(nt)
    # func_diffusion2D = partial(DIFFUSION2D, nu=nu, dt=dt, dx=dx, dy=dy)

    U = DIFFUSION2D(U, nu, dt, dx, dy)

    # U = func_diffusion2D(T=U)

    assert_allclose(U[-2], u)


# https://lorenabarba.com/blog/cfd-python-12-steps-to-navier-stokes/
# http://www.thevisualroom.com/02_barba_projects/linear_convection.html

# --------------------------- FDM solution --------------------------- #


def test_linear_convection():
    def convection(nt, nx, tmax, xmax, c):
        """
        Returns the velocity field and distance for 1D linear convection
        """
        # Increments
        dt = tmax / (nt - 1)
        dx = xmax / (nx - 1)

        # Initialise data structures
        u = np.zeros((nx, nt))
        x = np.zeros(nx)

        # Boundary conditions
        u[0, :] = u[nx - 1, :] = 1

        # Initial conditions
        for i in range(1, nx - 1):
            if i > (nx - 1) / 4 and i < (nx - 1) / 2:
                u[i, 0] = 2
            else:
                u[i, 0] = 1

        # Loop
        for n in range(0, nt - 1):
            for i in range(1, nx - 1):
                u[i, n + 1] = u[i, n] - c * (dt / dx) * (u[i, n] - u[i - 1, n])

        # X Loop
        for i in range(0, nx):
            x[i] = i * dx

        return u, x

    fdm_solution, _ = convection(nt=151, nx=51, tmax=0.5, xmax=2.0, c=0.5)

    # --------------------------- kex solution --------------------------- #
    tmax, xmax = 0.5, 2.0
    nt, nx = 151, 51
    dt, dx = tmax / (nt - 1), xmax / (nx - 1)
    u = np.ones([nx, nt])
    c = 0.5
    # Boundary conditions
    u[0, :] = u[nx - 1, :] = 1

    # Initial conditions
    for i in range(1, nx - 1):
        if i > (nx - 1) / 4 and i < (nx - 1) / 2:
            u[i, 0] = 2
        else:
            u[i, 0] = 1

    u = u.T

    F = kex.sscan(kernel_size=(3, 3), named_axis={0: "n", 1: "i"}, relative=True)

    F[:] = lambda u: u["i", "n"]
    F[1:, 1:-1] = lambda u: u["i", "n-1"] - (c * dt / dx) * (
        u["i", "n-1"] - u["i-1", "n-1"]
    )

    kex_solution = F(jnp.array(u))

    assert_allclose(kex_solution, fdm_solution.T, atol=1e-3)

    F = kex.kscan(
        kernel_size=(3, 3),
        padding=((1, 1), (1, 1)),
        named_axis={0: "n", 1: "i"},  # n for time axis , i for spatial axis
        relative=True,
    )

    F[:, 0] = F[:, -1] = lambda u: 1  # assign 1 for left and right boundary for all t

    F[0:1, int((nx - 1) / 4) + 1 : int((nx - 1) / 2)] = lambda u: 2

    F[:, : int((nx - 1) / 4) + 1] = F[:, int((nx - 1) / 2) :] = lambda u: 1

    F[1:, 1:-1] = lambda u: u["i", "n-1"] - (c * dt / dx) * (
        u["i", "n-1"] - u["i-1", "n-1"]
    )
    # kex_solution = F(jnp.array(u))

    # F = jax.jit(F.__call__)
    # F
    # plt.figure(figsize=(20,7))
    # print(F.items())
    # for line in kex_solution[::20]:
    #   plt.plot(jnp.linspace(0,xmax,nx),line)
    kex_solution = F(jnp.array(u))

    assert_allclose(kex_solution, fdm_solution.T, atol=1e-3)


def test_mesh():

    Mesh = kex.kmap(kernel_size=(1,), relative=True, padding="same")

    Mesh[0] = lambda x: x[0] * 10
    Mesh[1] = lambda x: x[0] * -10
    Mesh[2:] = lambda x: x[0] * 100
    array = jnp.arange(1, 11)

    np.testing.assert_allclose(
        Mesh(array), jnp.array([10, -20, 300, 400, 500, 600, 700, 800, 900, 1000])
    )

    Mesh = kex.kmap(kernel_size=(3, 3), relative=True, padding="same")

    Mesh[0, 0] = lambda x: x[0, 0] * 10
    Mesh[1, 1:] = lambda x: x[0, 0] * -10
    Mesh[2:] = lambda x: x[0, 0] * 100
    array = jnp.arange(1, 26).reshape(5, 5)
    np.testing.assert_allclose(
        Mesh(array),
        jnp.array(
            [
                [10, 20, 30, 40, 50],
                [60, -70, -80, -90, -100],
                [1100, 1200, 1300, 1400, 1500],
                [1600, 1700, 1800, 1900, 2000],
                [2100, 2200, 2300, 2400, 2500],
            ]
        ),
    )

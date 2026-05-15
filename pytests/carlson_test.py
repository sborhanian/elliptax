import time

import jax
import numpy as np
import pytest
from jax import numpy as jnp
from jax import random as jr

from scipy.special import elliprf, elliprc, elliprd, elliprj, elliprg
from elliptax import rf, rc, rd, rj, rg

jax.config.update("jax_enable_x64", True)


def test_rf():
    xyz = jnp.array(
        [
            [1.0, 2.0, 0.0],
            #            [1.j, -1.j, 0.0],
            #            [1.j - 1.0, 1.j, 0.0],
            [2.0, 3.0, 4.0],
            #            [1j, -1j, 2.0],
            #            [1j - 1.0, 1j, 1 - 1j],
        ]
    )

    expected = jnp.array(
        [
            1.3110287771461,
            #            1.8540746773014,
            #            0.79612586584234 - 1.2138566698365j,
            0.58408284167715,
            #            1.0441445654064,
            #            0.93912050218619 - 0.53296252018635j,
        ]
    )

    assert rf(*xyz.T) == pytest.approx(expected.T)


def test_rc():
    xy = jnp.array(
        [
            [0.0, 0.25],
            [2.25, 2.0],
            #            [0.0, 1j],
            #            [-1j, 1j],
            [0.25, -2.0],
            #            [1j, -1.0],
        ]
    )

    expected = jnp.array(
        [
            jnp.pi,
            jnp.log(2.0),
            #            (1.0 - 1j) * 1.1107207345396,
            #            1.2260849569072 - 0.34471136988768j,
            jnp.log(2.0) / 3.0,
            #            0.77778596920447 + 0.19832484993429j,
        ]
    )

    assert rc(*xy.T) == pytest.approx(expected)


def test_rj():

    xyzp = jnp.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0, 5.0],
            #            [2, 3, 4, -1 + 1j],
            #            [1j, -1j, 0, 2],
            #            [-1 + 1j, -1 - 1j, 1, 2],
            #            [1j, -1j, 0, 1 - 1j],
            #            [-1 + 1j, -1 - 1j, 1, -3 + 1j],
            #            [-1 + 1j, -2 - 1j, -1j, -1 + 1j],
        ]
    )

    expected = jnp.array(
        [
            0.7768862377858200,
            0.1429757966715700,
            #            0.13613945827771 - 0.38207561624427j,
            #            1.6490011662711000,
            #            0.9414835884122000,
            #            1.8260115229009 + 1.2290661908643j,
            #            -0.61127970812028 - 1.068403839000700j,
            #            1.8249027393704 - 1.2218475784827j,
        ]
    )

    assert rj(*jnp.array(xyzp).T) == pytest.approx(expected)


def test_rd():

    xyz = jnp.array(
        [
            [0.0, 2.0, 1.0],
            [2.0, 3.0, 4.0],
            #            [1j, -1j, 2],
            #            [0, 1j, -1j],
            #            [0, 1j - 1, 1j],
            #            [-2 - 1j, -1j, -1 + 1j],
        ]
    )

    expected = jnp.array(
        [
            1.7972103521034,
            0.1651052729426100,
            #            0.6593385415422000,
            #            1.2708196271910 + 2.7811120159521j,
            #            -1.8577235439239 - 0.96193450888839j,
            #            1.8249027393704 - 1.2218475784827j,
        ]
    )

    assert rd(*jnp.array(xyz).T) == pytest.approx(expected)


def test_rg():

    xyz = jnp.array(
        [
            [0.0, 16.0, 16.0],
            [2.0, 3.0, 4.0],
            #            [0, 1j, -1j],
            #            [1j - 1, 1j, 0],
            #            [-1j, 1j - 1, 1j],
            [0.0, 0.0796, 4.0],
        ]
    )

    expected = jnp.array(
        [
            3.1415926535898,
            1.7255030280692,
            #            0.4236065423969900,
            #            0.44660591677018 + 0.70768352357515j,
            #            0.36023392184473 + 0.40348623401722j,
            1.0284758090288,
        ]
    )

    assert rg(*jnp.array(xyz).T) == pytest.approx(expected)


def compute_args(key, n):


    return x, y, z, p, lam, mu, a, b


def carlson_consistency_template(f_lhs, key_seed=23, N=10):

    key = jr.key(key_seed)

    x, y, z, p, lam = jr.uniform(key, (5, N), minval=0.0, maxval=10.0)
    mu = x * y / lam
    a = p**2 * (lam + mu + x + y)
    b = p * (p + lam) * (p + mu)

    lhs = f_lhs(x, y, z, p, lam, mu, a, b)

    assert lhs == pytest.approx(jnp.zeros_like(lhs))


def test_carlson_consistency_eq49():

    carlson_consistency_template(
        lambda x, y, z, p, lam, mu, a, b: rf(x + lam, y + lam, lam)
        + rf(x + mu, y + mu, mu)
        - rf(x, y, 0)
    )


# def test_carlson_consistency_eq50():
#
#    carlson_consistency_template(
#        lambda x, y, p, lam, mu: rc(lam, x + lam) + rc(mu, x + mu) - rc(0.0, x)
#    )


def test_carlson_consistency_eq51():

    carlson_consistency_template(
        lambda x, y, z, p, lam, mu, a, b: rj(x + lam, y + lam, lam, p + lam)
        + rj(x + mu, y + mu, mu, p + mu)
        - rj(x, y, 0, p)
        + 3 * rc(a, b)
    )


def test_carlson_consistency_eq53():

    carlson_consistency_template(
        lambda x, y, z, p, lam, mu, a, b: rd(lam, x + lam, y + lam)
        + rd(mu, x + mu, y + mu)
        - rd(0.0, x, y)
        + 3 / y / jnp.sqrt(x + y + lam + mu)
    )

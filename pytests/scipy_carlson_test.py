import jax
import pytest
from jax import numpy as jnp
from jax import random as jr

import scipy.special as sp
from elliptax.special import (
    ellipe,
    ellipeinc,
    ellipk,
    ellipkinc,
    elliprc,
    elliprd,
    elliprf,
    elliprg,
    elliprj,
)

jax.config.update("jax_enable_x64", True)


"""
Test consistency of elliptic integrals against scipy.special implementations.
"""


#
# Carlson symmetric elliptic integrals
#


def carlson_consistency_template(f_lhs, key_seed=0, N=100_000, minval=0.0, maxval=10.0):

    k = jr.key(key_seed)
    x, y, z, p = jr.uniform(k, (4, N), minval=0.0, maxval=100.0)
    lhs = f_lhs(x, y, z, p)
    assert lhs == pytest.approx(jnp.zeros_like(lhs))


def test_scipy_consistency_elliprf():

    carlson_consistency_template(
        lambda x, y, z, p: elliprf(x, y, z) - sp.elliprf(x, y, z),
        key_seed=42,
    )


def test_scipy_consistency_elliprc():

    carlson_consistency_template(
        lambda x, y, z, p: elliprc(x, y) - sp.elliprc(x, y),
        key_seed=43,
    )


def test_scipy_consistency_elliprj():

    carlson_consistency_template(
        lambda x, y, z, p: elliprj(x, y, z, p) - sp.elliprj(x, y, z, p),
        key_seed=44,
    )


def test_scipy_consistency_elliprd():

    carlson_consistency_template(
        lambda x, y, z, p: elliprd(x, y, z) - sp.elliprd(x, y, z),
        key_seed=45,
    )


def test_scipy_consistency_elliprg():

    carlson_consistency_template(
        lambda x, y, z, p: elliprg(x, y, z) - sp.elliprg(x, y, z),
        key_seed=46,
    )


#
# Legendre elliptic integrals (the pi functions are not implemented in scipy, so we skip those)
#


def legendre_consistency_template(f_lhs, key_seed=0, N=100_000, minval=0.0, maxval=1.0):

    k = jr.key(key_seed)
    k1, k2 = jr.split(k)
    m, n = jr.uniform(k1, (2, N), minval=0.0, maxval=1.0)
    phi = jr.uniform(k2, (1, N), minval=0.0, maxval=jnp.pi / 2)
    lhs = f_lhs(phi, m, n)
    assert lhs == pytest.approx(jnp.zeros_like(lhs), abs=1e-11)


def test_scipy_consistency_ellipk():

    legendre_consistency_template(
        lambda phi, m, n: ellipk(m) - sp.ellipk(m),
        key_seed=47,
    )


def test_scipy_consistency_ellipkinc():

    legendre_consistency_template(
        lambda phi, m, n: ellipkinc(phi, m) - sp.ellipkinc(phi, m),
        key_seed=48,
    )


def test_scipy_consistency_ellipe():

    legendre_consistency_template(
        lambda phi, m, n: ellipe(m) - sp.ellipe(m),
        key_seed=49,
    )


def test_scipy_consistency_ellipeinc():

    legendre_consistency_template(
        lambda phi, m, n: ellipeinc(phi, m) - sp.ellipeinc(phi, m),
        key_seed=50,
    )

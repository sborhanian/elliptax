# math taken from Whittaker and Watson 1990
# Jacobi theta functions in Chapter 21, p. 486
# Jacobi elliptic functions in Chapter 22, p. 517

import jax
from jax import numpy as jnp

from elliptax.carlson import *
from elliptax.legendre import ellipk, ellipk_complement, ellipfinc

@jax.jit 
@jnp.vectorize
def theta_1(z, q, n=100):
    
    def body(n):
        return jnp.sign((-1)**n) * q**((n + 0.5)**2) * jnp.sin((2 * n + 1) * z)

    return 2. * jnp.sum(body(jnp.arange(n)))

@jax.jit 
@jnp.vectorize
def theta_2(z, q, n=100):
    
    def body(n):
        return q**((n + 0.5)**2) * jnp.cos((2 * n + 1) * z)

    return 2. * jnp.sum(body(jnp.arange(n)))

@jax.jit 
@jnp.vectorize
def theta_3(z, q, n=100):
    
    def body(n):
        return q**(n**2) * jnp.cos(2 * n * z)

    return 1. + 2. * jnp.sum(body(jnp.arange(n)))

@jax.jit 
@jnp.vectorize
def theta_4(z, q, n=100):
    
    def body(n):
        return jnp.sign((-1)**n) * q**(n**2) * jnp.cos(2 * n * z)

    return 1. + 2. * jnp.sum(body(jnp.arange(n)))

@jax.jit 
@jnp.vectorize
def nome_q(k):
    return jnp.exp(-jnp.pi * ellipk_complement(k) / ellipk(k))

@jax.jit 
@jnp.vectorize
def modulus_k(q, n=100):
    return theta_2(0, q, n=n)**2 / theta_3(0, q, n=n)**2

@jax.jit 
@jnp.vectorize
def ellipj(z, k, n=100):
    q = nome_q(k)

    th2 = theta_2(0, q, n=n)
    th3 = theta_3(0, q, n=n)
    th4 = theta_4(0, q, n=n)

    zeta = z / th3**2

    th4zeta = theta_4(zeta, q, n=n)

    sn = th3 * theta_1(zeta, q, n=n) / th2 / th4zeta
    cn = th4 * theta_2(zeta, q, n=n) / th2 / th4zeta
    dn = th4 * theta_3(zeta, q, n=n) / th3 / th4zeta

    phi1 = jnp.arcsin(sn)
    phi2 = jnp.arccos(cn)
    z1 = ellipfinc(phi1, k)
    z2 = ellipfinc(phi2, k)

    return sn, cn, dn, q, phi1, phi2, z1, z2

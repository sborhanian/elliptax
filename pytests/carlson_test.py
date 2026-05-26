"""
Test suite for Carlson symmetric elliptic integrals (R_F, R_C, R_D, R_J, R_G).

Numerical reference values are taken from Section 3 of Carlson (1994),
arXiv:math/9409227. Only real-valued test cases are included; the complex
cases from the same section are not supported by the current implementation.

Consistency tests verify the duplication / addition theorems from Carlson
(1994), §§ 3–4. Each identity must hold for random positive inputs.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import scipy.special as ssp

from elliptax import rf, rc, rd, rj, rg

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Reference-value tests (Carlson 1994, Section 3)
# ---------------------------------------------------------------------------

def test_rf():
    """R_F against the real-argument examples in Section 3 of Carlson (1994)."""
    # Columns: x, y, z
    xyz = jnp.array([
        [1.0, 2.0, 0.0],
        [2.0, 3.0, 4.0],
    ])
    expected = jnp.array([
        1.3110287771461,
        0.58408284167715,
    ])
    assert rf(*xyz.T) == pytest.approx(expected)


def test_rc():
    """R_C against the real-argument examples in Section 3 of Carlson (1994)."""
    # Columns: x, y
    xy = jnp.array([
        [0.0,  0.25],
        [2.25, 2.0],
        [0.25, -2.0],
    ])
    expected = jnp.array([
        jnp.pi,
        jnp.log(2.0),
        jnp.log(2.0) / 3.0,
    ])
    assert rc(*xy.T) == pytest.approx(expected)


def test_rj():
    """R_J against the real-argument examples in Section 3 of Carlson (1994)."""
    # Columns: x, y, z, p
    xyzp = jnp.array([
        [0.0, 1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0, 5.0],
    ])
    expected = jnp.array([
        0.77688623778582,
        0.14297579667157,
    ])
    assert rj(*xyzp.T) == pytest.approx(expected)


def test_rd():
    """R_D against the real-argument examples in Section 3 of Carlson (1994)."""
    # Columns: x, y, z
    xyz = jnp.array([
        [0.0, 2.0, 1.0],
        [2.0, 3.0, 4.0],
    ])
    expected = jnp.array([
        1.7972103521034,
        0.16510527294261,
    ])
    assert rd(*xyz.T) == pytest.approx(expected)


def test_rg():
    """R_G against the real-argument examples in Section 3 of Carlson (1994)."""
    # Columns: x, y, z
    xyz = jnp.array([
        [0.0,  16.0,   16.0],
        [2.0,   3.0,    4.0],
        [0.0,   0.0796, 4.0],
    ])
    expected = jnp.array([
        3.1415926535898,
        1.7255030280692,
        1.0284758090288,
    ])
    assert rg(*xyz.T) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Duplication / addition theorem tests (Carlson 1994, Section 3)
#
# Each identity takes the form  LHS = 0  after rearrangement, verified over
# N random draws from [0, 10]. The helper generates shared auxiliary
# variables (λ, μ, a, b) used by several of the identities.
# ---------------------------------------------------------------------------

def _consistency_check(f_residual, key_seed=23, N=100_000):
    """
    Draw N random argument sets and assert that f_residual returns ~0.

    Shared random variables
    -----------------------
    x, y, z, p  drawn uniformly from (0, 10]
    lam = lambda, drawn uniformly from (0, 10]
    mu  = x*y / lambda             (from the duplication relation)
    a   = p**2 * (lambda + mu + x + y)
    b   = p * (p + lambda) * (p + mu)

    f_residual must return a (raw_residual, ref) tuple. The helper normalises
    by max(|ref|, 1e-9) before asserting — the same pattern as
    _mpmath_ellipj_check — so the check is relative rather than absolute.
    The 1e-9 floor keeps the check relative down to very small ref values
    while staying safely above float64 machine epsilon (1e-12 * 1e-9 = 1e-21
    would demand sub-machine-epsilon accuracy, but Carlson integrals with
    positive inputs never produce refs that small). This handles identities
    whose terms grow near the boundary of the input domain (e.g. rd(0, x, y)
    as y -> 0 in eq. 53) without needing to clamp the sampling range.
    """
    key = jr.key(key_seed)
    x, y, z, p, lam = jr.uniform(key, (5, N), minval=0.0, maxval=10.0)
    mu = x * y / lam
    a  = p**2 * (lam + mu + x + y)
    b  = p * (p + lam) * (p + mu)

    raw, ref = f_residual(x, y, z, p, lam, mu, a, b)
    residual  = raw / jnp.maximum(jnp.abs(ref), 1e-9)
    assert residual == pytest.approx(jnp.zeros_like(residual))


def test_carlson_consistency_eq49():
    """
    R_F duplication theorem (Carlson 1994, eq. 49):

        R_F(x+λ, y+λ, λ) + R_F(x+μ, y+μ, μ) = R_F(x, y, 0)
    """
    _consistency_check(
        lambda x, y, z, p, lam, mu, a, b: (
            rf(x + lam, y + lam, lam) + rf(x + mu, y + mu, mu) - rf(x, y, 0),
            rf(x, y, 0),
        )
    )


def test_carlson_consistency_eq51():
    """
    R_J duplication theorem (Carlson 1994, eq. 51):

        R_J(x+λ, y+λ, λ, p+λ) + R_J(x+μ, y+μ, μ, p+μ)
            = R_J(x, y, 0, p) - 3·R_C(a, b)
    """
    _consistency_check(
        lambda x, y, z, p, lam, mu, a, b: (
            rj(x + lam, y + lam, lam, p + lam)
                + rj(x + mu, y + mu, mu, p + mu)
                - rj(x, y, 0, p)
                + 3 * rc(a, b),
            rj(x, y, 0, p),
        )
    )


def test_carlson_consistency_eq53():
    """
    R_D duplication theorem (Carlson 1994, eq. 53):

        R_D(λ, x+λ, y+λ) + R_D(μ, x+μ, y+μ)
            = R_D(0, x, y) - 3 / (y · √(x+y+λ+μ))
    """
    _consistency_check(
        lambda x, y, z, p, lam, mu, a, b: (
            rd(lam, x + lam, y + lam)
                + rd(mu, x + mu, y + mu)
                - rd(0.0, x, y)
                + 3 / y / jnp.sqrt(x + y + lam + mu),
            rd(0.0, x, y),
        )
    )


# ---------------------------------------------------------------------------
# scipy.special comparison tests
#
# Verify that our results match scipy.special over random positive inputs.
# scipy uses the same Carlson conventions and real-only support, so
# agreement to float64 precision is expected.
#
# Each test is expressed as a residual  ours(args) - scipy(args) = 0,
# mirroring the structure of the consistency tests above.
# ---------------------------------------------------------------------------

def _scipy_consistency_check(f_residual, key_seed, N=100_000, rel=1e-12):
    """Draw N random argument sets and assert that f_residual returns ~0.

    Arguments (x, y, z, p) are drawn uniformly from (0.1, 5]. The lower
    bound avoids singular behaviour at 0 that can amplify differences between
    implementations near degenerate arguments.
    """
    key = jr.key(key_seed)
    x, y, z, p = (np.asarray(a) for a in jr.uniform(key, (4, N), minval=0.1, maxval=5.0))
    residual = f_residual(x, y, z, p)
    assert residual == pytest.approx(np.zeros_like(residual), rel=rel)


def test_rf_vs_scipy():
    """R_F(x, y, z) - scipy.elliprf(x, y, z) = 0 over N random inputs."""
    _scipy_consistency_check(
        lambda x, y, z, p: np.asarray(rf(x, y, z)) - ssp.elliprf(x, y, z),
        key_seed=0,
    )


def test_rc_vs_scipy():
    """R_C(x, y) - scipy.elliprc(x, y) = 0 over N random inputs."""
    _scipy_consistency_check(
        lambda x, y, z, p: np.asarray(rc(x, y)) - ssp.elliprc(x, y),
        key_seed=1,
    )


def test_rd_vs_scipy():
    """R_D(x, y, z) - scipy.elliprd(x, y, z) = 0 over N random inputs."""
    _scipy_consistency_check(
        lambda x, y, z, p: np.asarray(rd(x, y, z)) - ssp.elliprd(x, y, z),
        key_seed=2,
    )


def test_rj_vs_scipy():
    """R_J(x, y, z, p) - scipy.elliprj(x, y, z, p) = 0 over N random inputs."""
    _scipy_consistency_check(
        lambda x, y, z, p: np.asarray(rj(x, y, z, p)) - ssp.elliprj(x, y, z, p),
        key_seed=3,
    )


def test_rg_vs_scipy():
    """R_G(x, y, z) - scipy.elliprg(x, y, z) = 0 over N random inputs."""
    _scipy_consistency_check(
        lambda x, y, z, p: np.asarray(rg(x, y, z)) - ssp.elliprg(x, y, z),
        key_seed=4,
    )

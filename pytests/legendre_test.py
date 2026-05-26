"""
Test suite for Legendre-form elliptic integrals (elliptax.legendre).

All functions use the modulus convention: the argument is k in (0, 1), where
m = k^2 is the parameter used by scipy and mpmath.

K, E, F_inc, E_inc : compared against scipy.special (scipy uses parameter m,
    so we pass k**2).
K' : compared against scipy.special.ellipk(1 - k**2), since K'(k) = K(kc)
    with kc = sqrt(1-k^2), i.e. K(kc^2) in parameter convention = ellipk(1-k^2).
Pi (complete and incomplete) : compared against mpmath (no scipy equivalent).
    mpmath also uses parameter m, so we pass k**2.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
import scipy.special as ssp

from elliptax.legendre import (
    ellipk, ellipe, ellippi,
    ellipfinc, ellipeinc, ellippiinc,
)

jax.config.update("jax_enable_x64", True)

try:
    import mpmath as mp
except ImportError:
    mp = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _args(key_seed, N):
    """Draw N random (phi, k, n) triples.

    phi in (0.1, pi/2 - 0.01): Carlson formula uses c = 1/sin(phi)^2, so phi
        must not be 0 or pi; lower bound keeps sin(phi) >= 0.1.
    k   in (0.01, 0.99)      : avoids circular (k=0) and hyperbolic (k=1)
        degenerate endpoints where K -> pi/2 or K -> inf.
    n   in (-5.0, 0.99)      : n=1 is a pole of Pi(n,k); the Carlson argument
        p = 1-n is positive throughout this range.

    Each test unpacks only the variables it needs and ignores the rest.
    """
    k1, k2, k3 = jr.split(jr.key(key_seed), 3)
    phi = np.asarray(jr.uniform(k1, (N,), minval=0.1,  maxval=np.pi / 2 - 0.01))
    k   = np.asarray(jr.uniform(k2, (N,), minval=0.01, maxval=0.99))
    n   = np.asarray(jr.uniform(k3, (N,), minval=-5.0, maxval=0.99))
    return phi, k, n


def _scipy_check(ours, ref, N, rel=1e-12):
    """ours - ref = 0 to relative tolerance rel."""
    residual = np.asarray(ours) - np.asarray(ref)
    assert residual == pytest.approx(np.zeros(N), rel=rel)


def _mpmath_check(ours, ref):
    """(ours - ref) / max(|ref|, 1.0) = 0 to default tolerance."""
    ours = np.asarray(ours, dtype=float)
    ref  = np.asarray(ref,  dtype=float)
    denom    = np.maximum(np.abs(ref), 1.0)
    residual = np.abs(ours - ref) / denom
    assert residual == pytest.approx(np.zeros(len(ref)))


# ---------------------------------------------------------------------------
# K(k) — complete elliptic integral of the first kind
# ---------------------------------------------------------------------------

def test_ellipk_vs_scipy():
    """K(k) - scipy.ellipk(k^2) = 0 over N random k."""
    _, k, _ = _args(0, 100_000)
    _scipy_check(ellipk(k), ssp.ellipk(k ** 2), N=100_000)


# ---------------------------------------------------------------------------
# E(k) — complete elliptic integral of the second kind
# ---------------------------------------------------------------------------

def test_ellipe_vs_scipy():
    """E(k) - scipy.ellipe(k^2) = 0 over N random k."""
    _, k, _ = _args(1, 100_000)
    _scipy_check(ellipe(k), ssp.ellipe(k ** 2), N=100_000)


# ---------------------------------------------------------------------------
# Pi(n, k) — complete elliptic integral of the third kind
# ---------------------------------------------------------------------------

@pytest.mark.skipif(mp is None, reason="mpmath not installed")
def test_ellippi_vs_mpmath():
    """Pi(n, k) - mpmath.ellippi(n, k^2) = 0 over N random (n, k)."""
    _, k, n = _args(3, 1_000)
    ours = np.asarray(ellippi(n, k))
    with mp.workdps(30):
        ref = np.array([float(mp.ellippi(ni, ki ** 2))
                        for ni, ki in zip(n, k)])
    _mpmath_check(ours, ref)


# ---------------------------------------------------------------------------
# F(phi, k) — incomplete elliptic integral of the first kind
# ---------------------------------------------------------------------------

def test_ellipfinc_vs_scipy():
    """F(phi, k) - scipy.ellipkinc(phi, k^2) = 0 over N random (phi, k)."""
    phi, k, _ = _args(4, 100_000)
    _scipy_check(ellipfinc(phi, k), ssp.ellipkinc(phi, k ** 2), N=100_000)


# ---------------------------------------------------------------------------
# E(phi, k) — incomplete elliptic integral of the second kind
# ---------------------------------------------------------------------------

def test_ellipeinc_vs_scipy():
    """E(phi, k) - scipy.ellipeinc(phi, k^2) = 0 over N random (phi, k)."""
    phi, k, _ = _args(5, 100_000)
    _scipy_check(ellipeinc(phi, k), ssp.ellipeinc(phi, k ** 2), N=100_000)


# ---------------------------------------------------------------------------
# Pi(phi, n, k) — incomplete elliptic integral of the third kind
# ---------------------------------------------------------------------------

@pytest.mark.skipif(mp is None, reason="mpmath not installed")
def test_ellippiinc_vs_mpmath():
    """Pi(phi, n, k) - mpmath.ellippi(n, phi, k^2) = 0 over N random (phi, k, n)."""
    phi, k, n = _args(6, 1_000)
    ours = np.asarray(ellippiinc(phi, k, n))
    with mp.workdps(30):
        ref = np.array([float(mp.ellippi(ni, phii, ki ** 2))
                        for phii, ki, ni in zip(phi, k, n)])
    _mpmath_check(ours, ref)

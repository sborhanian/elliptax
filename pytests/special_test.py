"""
Test suite for scipy-style elliptic integrals (elliptax.special).

All functions use the parameter convention: the argument is m in (0, 1), where
m = k^2 and k is the modulus.  This matches the scipy and mpmath conventions
directly, so no squaring is needed when calling the reference libraries.

ellipk, ellipe, ellipkinc, ellipeinc : compared against scipy.special (same
    parameter m convention, so arguments are passed as-is).
ellippi (complete) : compared against mpmath.ellippi(n, m) (2-arg form).
ellippiinc (incomplete) : compared against mpmath.ellippi(n, phi, m) (3-arg).
    Note: special.ellippiinc has signature (phi, m, n) — the n comes last.
"""

import jax
import jax.random as jr
import numpy as np
import pytest
import scipy.special as ssp

from elliptax.special import (
    ellipk, ellipe, ellippi,
    ellipkinc, ellipeinc, ellippiinc,
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
    """Draw N random (phi, m, n) triples.

    phi in (0.1, pi/2 - 0.01): Carlson formula uses c = 1/sin(phi)^2, so phi
        must not be 0 or pi; lower bound keeps sin(phi) >= 0.1.
    m   in (0.01, 0.99)      : avoids circular (m=0) and hyperbolic (m=1)
        degenerate endpoints where K -> pi/2 or K -> inf.
    n   in (-5.0, 0.99)      : n=1 is a pole of Pi(n,m); the Carlson argument
        p = 1-n is positive throughout this range.

    Each test unpacks only the variables it needs and ignores the rest.
    """
    k1, k2, k3 = jr.split(jr.key(key_seed), 3)
    phi = np.asarray(jr.uniform(k1, (N,), minval=0.1,  maxval=np.pi / 2 - 0.01))
    m   = np.asarray(jr.uniform(k2, (N,), minval=0.01, maxval=0.99))
    n   = np.asarray(jr.uniform(k3, (N,), minval=-5.0, maxval=0.99))
    return phi, m, n


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
# K(m) — complete elliptic integral of the first kind
# ---------------------------------------------------------------------------

def test_ellipk_vs_scipy():
    """K(m) - scipy.ellipk(m) = 0 over N random m."""
    _, m, _ = _args(0, 100_000)
    _scipy_check(ellipk(m), ssp.ellipk(m), N=100_000)


# ---------------------------------------------------------------------------
# E(m) — complete elliptic integral of the second kind
# ---------------------------------------------------------------------------

def test_ellipe_vs_scipy():
    """E(m) - scipy.ellipe(m) = 0 over N random m."""
    _, m, _ = _args(1, 100_000)
    _scipy_check(ellipe(m), ssp.ellipe(m), N=100_000)


# ---------------------------------------------------------------------------
# Pi(n, m) — complete elliptic integral of the third kind
# ---------------------------------------------------------------------------

@pytest.mark.skipif(mp is None, reason="mpmath not installed")
def test_ellippi_vs_mpmath():
    """Pi(n, m) - mpmath.ellippi(n, m) = 0 over N random (n, m)."""
    _, m, n = _args(3, 1_000)
    ours = np.asarray(ellippi(n, m))
    with mp.workdps(30):
        ref = np.array([float(mp.ellippi(ni, mi))
                        for ni, mi in zip(n, m)])
    _mpmath_check(ours, ref)


# ---------------------------------------------------------------------------
# F(phi, m) — incomplete elliptic integral of the first kind
# ---------------------------------------------------------------------------

def test_ellipkinc_vs_scipy():
    """F(phi, m) - scipy.ellipkinc(phi, m) = 0 over N random (phi, m)."""
    phi, m, _ = _args(4, 100_000)
    _scipy_check(ellipkinc(phi, m), ssp.ellipkinc(phi, m), N=100_000)


# ---------------------------------------------------------------------------
# E(phi, m) — incomplete elliptic integral of the second kind
# ---------------------------------------------------------------------------

def test_ellipeinc_vs_scipy():
    """E(phi, m) - scipy.ellipeinc(phi, m) = 0 over N random (phi, m)."""
    phi, m, _ = _args(5, 100_000)
    _scipy_check(ellipeinc(phi, m), ssp.ellipeinc(phi, m), N=100_000)


# ---------------------------------------------------------------------------
# Pi(phi, m, n) — incomplete elliptic integral of the third kind
# ---------------------------------------------------------------------------

@pytest.mark.skipif(mp is None, reason="mpmath not installed")
def test_ellippiinc_vs_mpmath():
    """Pi(phi, m, n) - mpmath.ellippi(n, phi, m) = 0 over N random (phi, m, n).

    Note: special.ellippiinc has signature (phi, m, n); mpmath uses (n, phi, m).
    """
    phi, m, n = _args(6, 1_000)
    ours = np.asarray(ellippiinc(phi, m, n))
    with mp.workdps(30):
        ref = np.array([float(mp.ellippi(ni, phii, mi))
                        for phii, mi, ni in zip(phi, m, n)])
    _mpmath_check(ours, ref)

"""
Test suite for Jacobi theta functions (theta_1–theta_4) and Jacobi elliptic
functions (sn, cn, dn) against external references.

Theta function tests use mpmath.jtheta as ground truth (skipped if mpmath is
not installed). Elliptic function tests compare against both mpmath.ellipfun
and scipy.special.ellipj.

Conventions: q = exp(i*pi*tau), w = exp(iz), period pi in z (DLMF / mpmath).
Random (z, tau) are drawn with Re(tau) in (-0.5, 0.5), Im(tau) in (0.2, 3.0)
and z in (-pi, pi) + i*(-1, 1), keeping inputs near the fundamental domain.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

try:
    import mpmath as mp
except ImportError:
    mp = None

if mp is None:
    pytest.skip(
        "mpmath is not installed — install it with 'pip install mpmath'.",
        allow_module_level=True,
    )

import scipy.special as ssp

from elliptax import jacobi_theta, jacobi_ellip

jax.config.update("jax_enable_x64", True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _theta_args(key_seed, N):
    """Draw N random (z, tau) pairs.

    z = z_re + i*z_im
      z_re in (-pi, pi)   — full real period of the theta functions; wider
                            would just repeat values by periodicity
      z_im in (-1, 1)     — W = exp(|Im z|) <= e ~= 2.72; larger |Im z| increases
                            the required series length, but values beyond +-1
                            give no new qualitative coverage and slow convergence

    tau = tau_re + i*tau_im
      tau_re in (-0.5, 0.5) — exact width of the fundamental domain; outside
                              this strip the algorithm still works but convergence
                              is slower (tau is not reduced before evaluation)
      tau_im in  (0.2, 3.0) — the only hard constraint is Im(tau) > 0; the lower
                              bound 0.2 gives |q| <= exp(-pi*0.2) ~= 0.53, which
                              converges in a reasonable number of series terms.
                              Upper bound: no constraint — large Im(tau) just
                              makes |q| -> 0 faster; 3.0 is generous.
    """
    key = jr.key(key_seed)
    k1, k2, k3, k4 = jr.split(key, 4)
    z_re   = np.asarray(jr.uniform(k1, (N,), minval=-np.pi, maxval= np.pi))
    z_im   = np.asarray(jr.uniform(k2, (N,), minval=-1.0,   maxval= 1.0))
    tau_re = np.asarray(jr.uniform(k3, (N,), minval=-0.5,   maxval= 0.5))
    tau_im = np.asarray(jr.uniform(k4, (N,), minval= 0.2,   maxval= 3.0))
    return z_re + 1j * z_im, tau_re + 1j * tau_im


def _ellipj_args(key_seed, N):
    """Draw N random (u, m) pairs.

    u in (0, 4)        — covers multiple quarter-periods for any m; since u is
                         real, zeta = pi*u/(2K) is also real (W = 1 in the theta
                         series), so convergence is unaffected by how large u is
    m in (0.01, 0.99)  — small margin from the degenerate endpoints: m = 0
                         (circular limit, K' -> inf, tau -> i*inf) and m = 1
                         (hyperbolic limit, K -> inf, tau -> 0+) are not supported
    """
    key = jr.key(key_seed)
    k1, k2 = jr.split(key)
    u = np.asarray(jr.uniform(k1, (N,), minval=0.0,  maxval=4.0))
    m = np.asarray(jr.uniform(k2, (N,), minval=0.01, maxval=0.99))
    return u, m


def _mpmath_theta_check(n, key_seed, N=10_000):
    """Draw N random (z, tau) and assert theta_n - mpmath = 0 to ATOL.

    n is 1-indexed (1-4).
    """
    zs, taus = _theta_args(key_seed, N)
    ours = np.asarray(jacobi_theta(zs, taus)[n - 1])
    ref  = np.array([complex(mp.jtheta(n, z, mp.exp(1j * mp.pi * tau)))
                     for z, tau in zip(zs, taus)])
    denom    = np.maximum(np.abs(ref), 1.0)
    residual = np.abs(ours - ref) / denom
    assert residual == pytest.approx(np.zeros(N))


def _mpmath_ellipj_check(fn_idx, key_seed, N=10_000):
    """Draw N random (u, m) and assert ellipj[fn_idx] - mpmath = 0 to ATOL.

    fn_idx: 0 = sn, 1 = cn, 2 = dn.
    """
    fns = ("sn", "cn", "dn")
    u, m = _ellipj_args(key_seed, N)
    ours = np.asarray(jacobi_ellip(u, m)[fn_idx])
    with mp.workdps(30):
        ref = np.array([complex(mp.ellipfun(fns[fn_idx], ui, mi)).real
                        for ui, mi in zip(u, m)])
    # Normalise by max(|ref|, 1.0): relative check where |ref| >= 1, absolute
    # check below that floor. Unlike the Carlson identity checks (where ref is
    # always a moderate positive integral), ref here is the function value
    # itself, which can be near zero (cn at its zeros, sn near z=0, etc.).
    # Near those zeros float64 can only deliver absolute accuracy, not
    # relative, so a floor of 1.0 is intentional — 1e-9 would demand relative
    # precision that no float64 implementation can satisfy there.
    denom    = np.maximum(np.abs(ref), 1.0)
    residual = np.abs(ours - ref) / denom
    assert residual == pytest.approx(np.zeros(N))


def _scipy_consistency_check(f_residual, key_seed, N=100_000, rel=1e-12):
    """Draw N random (u, m) pairs and assert f_residual returns ~0."""
    u, m = _ellipj_args(key_seed, N)
    residual = f_residual(u, m)
    assert residual == pytest.approx(np.zeros_like(residual), rel=rel)


# ---------------------------------------------------------------------------
# Theta functions vs mpmath
# ---------------------------------------------------------------------------

def test_theta1_vs_mpmath():
    """theta_1(z, tau) - mpmath.jtheta(1, z, q) = 0 over N random (z, tau)."""
    _mpmath_theta_check(1, key_seed=0)


def test_theta2_vs_mpmath():
    """theta_2(z, tau) - mpmath.jtheta(2, z, q) = 0 over N random (z, tau)."""
    _mpmath_theta_check(2, key_seed=1)


def test_theta3_vs_mpmath():
    """theta_3(z, tau) - mpmath.jtheta(3, z, q) = 0 over N random (z, tau)."""
    _mpmath_theta_check(3, key_seed=2)


def test_theta4_vs_mpmath():
    """theta_4(z, tau) - mpmath.jtheta(4, z, q) = 0 over N random (z, tau)."""
    _mpmath_theta_check(4, key_seed=3)


# ---------------------------------------------------------------------------
# Jacobi elliptic functions vs mpmath
# ---------------------------------------------------------------------------

def test_sn_vs_mpmath():
    """sn(u|m) − mpmath.ellipfun('sn', u, m) = 0 over N random (u, m)."""
    _mpmath_ellipj_check(0, key_seed=4)


def test_cn_vs_mpmath():
    """cn(u|m) − mpmath.ellipfun('cn', u, m) = 0 over N random (u, m)."""
    _mpmath_ellipj_check(1, key_seed=5)


def test_dn_vs_mpmath():
    """dn(u|m) − mpmath.ellipfun('dn', u, m) = 0 over N random (u, m)."""
    _mpmath_ellipj_check(2, key_seed=6)


# ---------------------------------------------------------------------------
# Jacobi elliptic functions vs scipy.special
# ---------------------------------------------------------------------------

def test_sn_vs_scipy():
    """sn(u|m) − scipy.ellipj(u, m)[0] = 0 over N random (u, m)."""
    _scipy_consistency_check(
        lambda u, m: np.asarray(jacobi_ellip(u, m)[0]) - ssp.ellipj(u, m)[0],
        key_seed=7,
    )


def test_cn_vs_scipy():
    """cn(u|m) − scipy.ellipj(u, m)[1] = 0 over N random (u, m)."""
    _scipy_consistency_check(
        lambda u, m: np.asarray(jacobi_ellip(u, m)[1]) - ssp.ellipj(u, m)[1],
        key_seed=8,
    )


def test_dn_vs_scipy():
    """dn(u|m) − scipy.ellipj(u, m)[2] = 0 over N random (u, m)."""
    _scipy_consistency_check(
        lambda u, m: np.asarray(jacobi_ellip(u, m)[2]) - ssp.ellipj(u, m)[2],
        key_seed=9,
    )

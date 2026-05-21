"""
Test suite for jacobi_theta_v3.

Verifies the JAX Algorithm 1 implementation against mpmath.jtheta as ground
truth. Covers scalar inputs, 1D arrays, 2D broadcasting, theta constants,
and derivatives via jax.grad.

Run with:
    pytest test_jacobi_theta.py
    pytest test_jacobi_theta.py -v       # verbose
    pytest test_jacobi_theta.py -k grad  # run only grad-related tests
"""

import jax
jax.config.update("jax_enable_x64", True)  # mpmath uses float64 precision
import jax.numpy as jnp
import numpy as np
import pytest

try:
    import mpmath as mp
except ImportError:
    mp = None

if mp is None:
    pytest.skip(
        "mpmath is not installed — install it with 'pip install mpmath' to run this test suite.",
        allow_module_level=True,
    )

from elliptax.jacobi import jacobi_theta, ellipj

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Tolerance for "machine precision" comparisons. Truncation tolerance is
# 1e-18 internally, but float64 epsilon + accumulated rounding gives ~1e-14
# in the worst case for moderately deep series.
ATOL = 1e-13


def mp_truth(z, tau):
    """High-precision (θ_1, θ_2, θ_3, θ_4) from mpmath."""
    q = mp.exp(1j * mp.pi * tau)
    return tuple(complex(mp.jtheta(n, z, q)) for n in (1, 2, 3, 4))


def max_rel_err(ours, truth):
    """Max relative error across the four theta functions."""
    err = 0.0
    for o, t in zip(ours, truth):
        o_arr = np.asarray(o)
        t_arr = np.asarray(t)
        denom = np.maximum(np.abs(t_arr), 1.0)
        err = max(err, float(np.max(np.abs(o_arr - t_arr) / denom)))
    return err


# ---------------------------------------------------------------------------
# Scalar input tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("z, tau", [
    (0.3,         1.0j),
    (0.2,         0.5 + 1.0j),
    (0.25 + 0.1j, 0.7 + 0.8j),
    (0.0,         1.0j),                  # theta constants
    (-0.6,        1.2j),                  # negative real z
    (0.0,         np.exp(1j * np.pi / 3)),  # τ on |τ|=1 boundary
    (0.15,        0.1 + 0.5j),            # τ near real axis (|q| ≈ 0.21)
    (0.4,         0.5 + 1.3j),            # Re(τ) = 0.5 (fund. domain boundary)
])
def test_scalar(z, tau):
    """Scalar (z, τ) inputs match mpmath to machine precision."""
    ours = jacobi_theta(z, tau)
    truth = mp_truth(z, tau)
    assert max_rel_err(ours, truth) < ATOL


def test_scalar_returns_scalar_shape():
    """Scalar input produces 0-d output."""
    ours = jacobi_theta(0.3, 1.0j)
    for theta in ours:
        assert theta.shape == ()


# ---------------------------------------------------------------------------
# Array input / broadcasting tests
# ---------------------------------------------------------------------------

def test_1d_z_scalar_tau():
    """1D array of z, scalar τ — output has z's shape."""
    zs = jnp.linspace(-0.5, 0.5, 7)
    tau = 1.2j
    ours = jacobi_theta(zs, tau)
    truth = tuple(
        np.array([mp_truth(float(z), tau)[n] for z in np.asarray(zs).real])
        for n in range(4)
    )
    for theta in ours:
        assert theta.shape == (7,)
    assert max_rel_err(ours, truth) < ATOL


def test_scalar_z_1d_tau():
    """Scalar z, 1D array of τ — output has τ's shape."""
    z = 0.25
    taus = jnp.array([0.5j, 1.0j, 0.3 + 0.8j, -0.4 + 1.1j])
    ours = jacobi_theta(z, taus)
    truth = tuple(
        np.array([mp_truth(z, complex(t))[n] for t in np.asarray(taus)])
        for n in range(4)
    )
    for theta in ours:
        assert theta.shape == (4,)
    assert max_rel_err(ours, truth) < ATOL


def test_broadcasting_2d():
    """Broadcasting (3,1) × (1,4) → (3,4)."""
    zs = jnp.array([[0.1], [0.3], [0.5]])
    taus = jnp.array([[0.6j, 1.0j, 1.5j, 2.0j]])
    ours = jacobi_theta(zs, taus)
    for theta in ours:
        assert theta.shape == (3, 4)
    truth = tuple(
        np.array([[mp_truth(float(zs[i, 0].real), complex(taus[0, j]))[n]
                   for j in range(4)] for i in range(3)])
        for n in range(4)
    )
    assert max_rel_err(ours, truth) < ATOL


def test_2d_complex_grid():
    """Typical use case: evaluate on a 2D grid of complex z at fixed τ."""
    xs = jnp.linspace(-0.4, 0.4, 5)
    ys = jnp.linspace(0.1, 0.3, 3)
    X, Y = jnp.meshgrid(xs, ys)
    Z = X + 1j * Y
    ours = jacobi_theta(Z, 1.0j)
    for theta in ours:
        assert theta.shape == (3, 5)
    truth = tuple(
        np.array([[mp_truth(complex(Z[i, j]), 1.0j)[n] for j in range(5)]
                  for i in range(3)])
        for n in range(4)
    )
    assert max_rel_err(ours, truth) < ATOL


def test_same_shape_arrays():
    """Same-shape z and τ arrays — element-wise pairing, no broadcasting."""
    zs = jnp.array([0.1, 0.2, 0.3, 0.4])
    taus = jnp.array([1.0j, 1.1j, 1.2j, 1.3j])
    ours = jacobi_theta(zs, taus)
    for theta in ours:
        assert theta.shape == (4,)
    truth = tuple(
        np.array([mp_truth(float(zs[i].real), complex(taus[i]))[n]
                  for i in range(4)])
        for n in range(4)
    )
    assert max_rel_err(ours, truth) < ATOL


def test_python_scalar_inputs():
    """Plain Python float/complex scalars work (not just JAX arrays)."""
    ours = jacobi_theta(0.3, 1.0j)
    truth = mp_truth(0.3, 1.0j)
    assert max_rel_err(ours, truth) < ATOL


def test_numpy_array_inputs():
    """NumPy arrays work, not just JAX arrays."""
    zs = np.linspace(-0.5, 0.5, 5)
    ours = jacobi_theta(zs, 1.0j)
    truth = tuple(
        np.array([mp_truth(float(z), 1.0j)[n] for z in zs])
        for n in range(4)
    )
    assert max_rel_err(ours, truth) < ATOL


# ---------------------------------------------------------------------------
# Identity tests (don't need mpmath)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tau", [1.0j, 0.5 + 1.2j, 2.0j, 0.3 + 0.9j])
def test_jacobi_derivative_identity(tau):
    """ϑ_1'(0, τ) = ϑ_2(0, τ) · ϑ_3(0, τ) · ϑ_4(0, τ).  (period-π / iz convention)"""
    # Derivative of -i·θ_1 w.r.t. real z, then multiply by i to recover θ_1'
    def neg_i_theta1_real(zr):
        return (-1j * jacobi_theta(zr + 0.0j, tau)[0]).real

    def neg_i_theta1_imag(zr):
        return (-1j * jacobi_theta(zr + 0.0j, tau)[0]).imag

    d_real = float(jax.grad(neg_i_theta1_real)(0.0))
    d_imag = float(jax.grad(neg_i_theta1_imag)(0.0))
    d_theta1 = 1j * (d_real + 1j * d_imag)

    _, t2, t3, t4 = jacobi_theta(0.0, tau)
    rhs = complex(t2 * t3 * t4)

    rel_err = abs(d_theta1 - rhs) / abs(rhs)
    assert rel_err < ATOL


@pytest.mark.parametrize("tau", [1.0j, 2.0j, 0.5 + 1.2j])
def test_theta1_at_zero_vanishes(tau):
    """ϑ_1(0, τ) = 0 identically."""
    th1, _, _, _ = jacobi_theta(0.0, tau)
    assert abs(complex(th1)) < 1e-14


@pytest.mark.parametrize("tau", [1.0j, 2.0j])
def test_quartic_identity(tau):
    """ϑ_3(0, τ)^4 = ϑ_2(0, τ)^4 + ϑ_4(0, τ)^4."""
    _, t2, t3, t4 = jacobi_theta(0.0, tau)
    lhs = complex(t3) ** 4
    rhs = complex(t2) ** 4 + complex(t4) ** 4
    rel_err = abs(lhs - rhs) / abs(lhs)
    assert rel_err < ATOL


# ---------------------------------------------------------------------------
# Differentiability tests
# ---------------------------------------------------------------------------

def test_grad_through_jit_vectorize():
    """jax.grad composes correctly through jit + vectorize."""
    grad_th3 = jax.grad(lambda z: jacobi_theta(z + 0.0j, 1.0j)[2].real)
    g = float(grad_th3(0.3))

    # mpmath reference: d θ_3 / dz at z=0.3, τ=i, via central difference.
    # Stick to float64-comparable precision; mpmath dps=50 is overkill but cheap.
    with mp.workdps(50):
        q = mp.exp(1j * mp.pi * 1.0j)
        h = mp.mpf('1e-10')
        d_truth = float(
            ((mp.jtheta(3, 0.3 + h, q)
              - mp.jtheta(3, 0.3 - h, q)).real / (2 * h))
        )
    rel_err = abs(g - d_truth) / abs(d_truth)
    assert rel_err < 1e-10  # central diff limits truth precision


def test_jit_recompiles_on_n_change():
    """Calling with different worst-case |q| picks a different N (sanity).

    This isn't a correctness check, just a smoke test that varying inputs
    don't crash and produce reasonable values.
    """
    # Easy case: small q, few terms needed
    res_easy = jacobi_theta(0.3, 3.0j)
    # Hard case: q closer to 1, more terms
    res_hard = jacobi_theta(0.3, 0.3j)
    # Both should produce finite values
    for r in res_easy + res_hard:
        assert jnp.isfinite(r).all()


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_rejects_real_tau():
    """Im(τ) ≤ 0 raises ValueError."""
    with pytest.raises(ValueError, match=r"Im\(τ\)"):
        jacobi_theta(0.3, 1.0 + 0.0j)


def test_rejects_negative_imag_tau():
    """Im(τ) < 0 raises ValueError."""
    with pytest.raises(ValueError, match=r"Im\(τ\)"):
        jacobi_theta(0.3, -1.0j)


def test_rejects_mixed_array_with_bad_tau():
    """Even one bad τ in an array trips the check."""
    taus = jnp.array([1.0j, 2.0j, -0.5j])  # third one invalid
    with pytest.raises(ValueError, match=r"Im\(τ\)"):
        jacobi_theta(0.3, taus)


# ---------------------------------------------------------------------------
# Jacobi elliptic function tests
# ---------------------------------------------------------------------------

try:
    import scipy.special as ssp
    _have_scipy = True
except ImportError:
    _have_scipy = False

scipy_only = pytest.mark.skipif(not _have_scipy, reason="scipy not installed")

# ellipj composes two jacobi_theta calls + two Carlson RF evaluations, so
# accumulated rounding is naturally a bit higher than for the raw theta tests.
ELLIPJ_ATOL = 1e-12


def _mp_ellipj(u, m):
    """High-precision (sn, cn, dn) from mpmath."""
    with mp.workdps(30):
        sn = complex(mp.ellipfun("sn", u, m))
        cn = complex(mp.ellipfun("cn", u, m))
        dn = complex(mp.ellipfun("dn", u, m))
    return sn.real, cn.real, dn.real


_ELLIPJ_PARAMS = [
    (0.3,  0.5),
    (1.0,  0.5),
    (0.5,  0.9),
    (0.2,  0.1),
    (1.5,  0.7),
    (0.0,  0.5),   # u = 0: sn=0, cn=1, dn=1
    (0.7,  0.01),  # nearly circular
    (0.4,  0.99),  # nearly hyperbolic
]


@pytest.mark.parametrize("u, m", _ELLIPJ_PARAMS)
def test_ellipj_vs_mpmath_scalar(u, m):
    """Scalar ellipj matches mpmath to near machine precision."""
    sn, cn, dn = ellipj(u, m)
    sn_ref, cn_ref, dn_ref = _mp_ellipj(u, m)
    for val, ref, name in [(sn, sn_ref, "sn"), (cn, cn_ref, "cn"), (dn, dn_ref, "dn")]:
        denom = max(abs(ref), 1e-14)
        rel = abs(float(val) - ref) / denom
        assert rel < ELLIPJ_ATOL, f"{name}({u},{m}): got {float(val):.16g}, expected {ref:.16g}, rel_err={rel:.2e}"


@pytest.mark.parametrize("u, m", _ELLIPJ_PARAMS)
@scipy_only
def test_ellipj_vs_scipy_scalar(u, m):
    """Scalar ellipj matches scipy.special.ellipj."""
    sn, cn, dn = ellipj(u, m)
    sn_ref, cn_ref, dn_ref, _ = ssp.ellipj(u, m)
    for val, ref, name in [(sn, sn_ref, "sn"), (cn, cn_ref, "cn"), (dn, dn_ref, "dn")]:
        denom = max(abs(ref), 1e-14)
        rel = abs(float(val) - ref) / denom
        assert rel < ELLIPJ_ATOL, f"{name}({u},{m}) vs scipy: rel_err={rel:.2e}"


@pytest.mark.parametrize("m", [0.1, 0.3, 0.5, 0.7, 0.9])
def test_ellipj_pythagorean_sn_cn(m):
    """sn(u|m)² + cn(u|m)² = 1 for several u values."""
    us = jnp.linspace(0.1, 1.5, 10)
    sn, cn, _ = ellipj(us, m)
    err = jnp.max(jnp.abs(sn**2 + cn**2 - 1.0))
    assert float(err) < ATOL, f"sn²+cn²≠1 for m={m}, max_err={float(err):.2e}"


@pytest.mark.parametrize("m", [0.1, 0.3, 0.5, 0.7, 0.9])
def test_ellipj_pythagorean_sn_dn(m):
    """m·sn(u|m)² + dn(u|m)² = 1 for several u values."""
    us = jnp.linspace(0.1, 1.5, 10)
    sn, _, dn = ellipj(us, m)
    err = jnp.max(jnp.abs(m * sn**2 + dn**2 - 1.0))
    assert float(err) < ATOL, f"m·sn²+dn²≠1 for m={m}, max_err={float(err):.2e}"


def test_ellipj_at_u_zero():
    """sn(0|m)=0, cn(0|m)=1, dn(0|m)=1 for any m."""
    ms = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])
    sn, cn, dn = ellipj(0.0, ms)
    assert jnp.max(jnp.abs(sn)) < 1e-14
    assert jnp.max(jnp.abs(cn - 1.0)) < ATOL
    assert jnp.max(jnp.abs(dn - 1.0)) < ATOL


@scipy_only
def test_ellipj_1d_array_vs_scipy():
    """1D array of u at fixed m matches scipy element-wise."""
    us = jnp.linspace(0.0, 1.4, 12)
    m = 0.6
    sn, cn, dn = ellipj(us, m)
    for i, u in enumerate(np.asarray(us)):
        sn_ref, cn_ref, dn_ref, _ = ssp.ellipj(float(u), m)
        for val, ref, name in [(sn[i], sn_ref, "sn"), (cn[i], cn_ref, "cn"), (dn[i], dn_ref, "dn")]:
            denom = max(abs(ref), 1e-14)
            rel = abs(float(val) - ref) / denom
            assert rel < ELLIPJ_ATOL, f"{name}[{i}]: rel_err={rel:.2e}"


@scipy_only
def test_ellipj_broadcasting():
    """Broadcasting (4,1) u × (1,3) m produces (4,3) output matching scipy."""
    us = jnp.array([[0.2], [0.5], [0.8], [1.1]])
    ms = jnp.array([[0.3, 0.6, 0.9]])
    sn, cn, dn = ellipj(us, ms)
    assert sn.shape == (4, 3)
    for i in range(4):
        for j in range(3):
            sn_ref, cn_ref, dn_ref, _ = ssp.ellipj(float(us[i, 0]), float(ms[0, j]))
            for val, ref, name in [(sn[i, j], sn_ref, "sn"), (cn[i, j], cn_ref, "cn"), (dn[i, j], dn_ref, "dn")]:
                denom = max(abs(ref), 1e-14)
                rel = abs(float(val) - ref) / denom
                assert rel < ELLIPJ_ATOL, f"{name}[{i},{j}]: rel_err={rel:.2e}"

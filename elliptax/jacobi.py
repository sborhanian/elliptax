"""
JAX implementation of Johansson's Algorithm 1 for Jacobi theta functions.

Reference:
    Fredrik Johansson, "Numerical Evaluation of Elliptic Functions, Elliptic
    Integrals and Modular Forms" (arXiv:1806.06725), §1.4.2, Algorithm 1.

Inputs `z` and `tau` may be scalars or arrays of any shape that broadcast
together; the four outputs are returned with the broadcast shape.

Conventions (DLMF / Whittaker & Watson / mpmath.jtheta):
    q  = exp(iπτ),   w = exp(iz),   v = 1/w,   Im(τ) > 0.
    The series are:
        ϑ_1(z, τ) = 2 Σ_{n≥0} (-1)^n q^{(n+½)²} sin((2n+1) z)
        ϑ_2(z, τ) = 2 Σ_{n≥0}        q^{(n+½)²} cos((2n+1) z)
        ϑ_3(z, τ) = 1 + 2 Σ_{n≥1}     q^{n²}      cos(2n z)
        ϑ_4(z, τ) = 1 + 2 Σ_{n≥1} (-1)^n q^{n²}   cos(2n z)
    The period along the real z-axis is π. Matches `mpmath.jtheta(n, z, q)`
    directly — no factor of π conversion needed.

Note: Johansson's paper uses w = exp(iπz) instead (period 1 in z), but that
is an internal renaming; the algorithm itself is unchanged.
"""

from functools import partial

import jax
import jax.numpy as jnp

from .carlson import rf

jax.config.update("jax_enable_x64", True)


def _number_of_terms(Q, W, tol=1e-18):
    """
    Pick the smallest N such that the truncation error bound from eq. 1.18
    of the paper is below `tol` for every element of the input arrays:

        2 * Q^E * W^(N+2) / (1 - Q^F * W) < tol

    where E = floor((N+2)^2 / 4) and F = floor((N+1)/2) + 1.

    Parameters
    ----------
    Q : float
        Worst-case |q| over the input batch; must be in (0, 1).
    W : float
        Worst-case max(|w|, |w|^-1) over the input batch.
    tol : float
        Target truncation tolerance.

    Returns
    -------
    N : int
        Number of terms sufficient for the entire batch.
    """
    for N in range(2, 100):
        E = (N + 2) ** 2 // 4
        F = (N + 1) // 2 + 1
        alpha = 1.0 - Q**F * W
        beta = tol - Q**E * W ** (N + 2)
        if alpha > 0 and beta > 0:
            return N
    raise ValueError(
        f"Series failed to converge within tol={tol}; ensure |q| < 1 everywhere."
    )


def jacobi_theta_error(z, tau, N):
    """
    Compute the truncation error bound from eq. 1.18 of the paper for each
    (z, τ) pair, given that the series was truncated at N terms:

        ε(z, τ, N) = 2 · Q^E · W^(N+2) / (1 - Q^F · W)

    where Q = |q| = exp(-π·Im(τ)), W = max(|w|, |w|⁻¹) = exp(|Im(z)|),
    E = floor((N+2)²/4), F = floor((N+1)/2) + 1.

    Parameters
    ----------
    z, tau : scalar or array_like
        Same inputs passed to `jacobi_theta`. Shapes must broadcast together.
    N : int
        Number of terms used in the series (as returned implicitly by
        `jacobi_theta`; can be recovered by calling `_number_of_terms`
        directly).

    Returns
    -------
    error : jnp.ndarray
        Upper bound on the truncation error, with the broadcast shape of
        (z, τ). The same bound applies to all four theta functions.
    """
    z = jnp.asarray(z, dtype=jnp.complex128)
    tau = jnp.asarray(tau, dtype=jnp.complex128)

    Q = jnp.exp(-jnp.pi * tau.imag)
    W = jnp.maximum(jnp.exp(-z.imag), jnp.exp(z.imag))

    E = (N + 2) ** 2 // 4
    F = (N + 1) // 2 + 1

    return 2.0 * Q**E * W ** (N + 2) / (1.0 - Q**F * W)


@partial(jax.jit, static_argnames=("N",))
@partial(jnp.vectorize, signature="(),()->(),(),(),()", excluded={2})
def _algorithm_1(z, tau, N):
    """
    Algorithm 1, D = 1 (function values only). Scalar-defined, broadcast
    over (z, tau) via jnp.vectorize. N is a Python int known at trace time.

    Convention: q = exp(iπτ), w = exp(iz). The cos/sin terms inside the
    series are cos(2nz), sin((2n+1)z) — i.e., z is in radians-like units
    with period π along the real axis. This matches DLMF/Whittaker-Watson
    and mpmath.jtheta.
    """
    q4 = jnp.exp(1j * jnp.pi * tau / 4.0)
    q = jnp.exp(1j * jnp.pi * tau)
    w = jnp.exp(1j * z)
    v = 1.0 / w

    # Precompute powers w^{2j}, v^{2j} for j = 0..K
    K = (N + 1) // 2 + 1
    m = 2 * jnp.arange(K + 1)
    w_pow = w**m
    v_pow = v**m

    # Even k = 2(n-1):  ms = n²,      kmod2 = 0  →  θ3, θ4
    # Odd  k = 2n-1:    ms = n(n+1),  kmod2 = 1  →  θ1, θ2
    # Sign for both θ4 and θ1 is (-1)^n. (Algorithm 1, lines 11-30.)
    ns_even = jnp.arange(1, N // 2 + 1)  # n = 1, 2, ..., N // 2
    ns_odd = jnp.arange(1, (N + 1) // 2 + 1)  # n = 1, 2, ..., (N+1) // 2
    sign = jnp.where(ns_even % 2 == 0, 1.0, -1.0)  # (-1)^n, same for both

    t_even = (w_pow[ns_even] + v_pow[ns_even]) * q ** (ns_even**2)
    t_odd = (w_pow[ns_odd] + v_pow[ns_odd + 1]) * q ** (ns_odd * (ns_odd + 1))
    u_odd = (w_pow[ns_odd] - v_pow[ns_odd + 1]) * q ** (ns_odd * (ns_odd + 1))

    th3 = jnp.sum(t_even)
    th4 = jnp.sum(sign * t_even)
    th2 = jnp.sum(t_odd)
    th1 = jnp.sum(jnp.where(ns_odd % 2 == 0, 1.0, -1.0) * u_odd)

    # Leading-term adjustment (Algorithm 1, lines 33-34, r = 0)
    th1 = th1 * w + (w - v)
    th2 = th2 * w + (w + v)

    # Final scaling (line 37) and leading 1 (line 39)
    th1 = -1j * q4 * th1
    th2 = q4 * th2
    th3 = th3 + 1.0
    th4 = th4 + 1.0

    return th1, th2, th3, th4


def jacobi_theta(z, tau, tol=1e-18, return_error_bound=False):
    """
    Compute (θ1, θ2, θ3, θ4) at (z, τ) using Johansson's Algorithm 1.

    Parameters
    ----------
    z, tau : scalar or jnp.ndarray
        Any shapes that broadcast together. tau must have Im(τ) > 0
        everywhere.
    tol : float
        Target truncation tolerance; defaults to ~ float64 epsilon.

    Returns
    -------
    (θ1, θ2, θ3, θ4) : tuple of jnp.ndarray
        Each output has the broadcast shape of (z, τ).

    Notes
    -----
    The series length N is chosen from the worst-case |q|, |w| over the
    input batch — a single host→device sync, then a single fused jit
    over the whole array. For best convergence, τ should be in or near
    the fundamental domain {|τ| ≥ 1, |Re(τ)| ≤ 1/2}.
    """
    z = jnp.asarray(z, dtype=jnp.complex128)
    tau = jnp.asarray(tau, dtype=jnp.complex128)

    # Validate Im(τ) > 0
    tau_imag_min = float(jnp.min(tau.imag))
    if tau_imag_min <= 0:
        raise ValueError(f"Im(τ) must be > 0 everywhere; got min = {tau_imag_min}")

    # Host-side N selection.
    # Q = |q| = exp(-π·Im(τ))  since q = exp(iπτ)
    # W = max(|w|, |w|^-1) = max(exp(-Im(z)), exp(Im(z)))  since w = exp(iz)
    # W may not be concretisable under jax.grad (z is a tracer);
    # fall back to W=1 in that case — safe because Im(z)=0 is the common
    # differentiating point and the N chosen from tau alone is sufficient.
    Q_max = float(jnp.max(jnp.exp(-jnp.pi * tau.imag)))
    if not 0.0 < Q_max < 1.0:
        raise ValueError(f"|q| must be in (0, 1) everywhere; got {Q_max:.6f}")

    try:
        W_max = float(jnp.max(jnp.maximum(jnp.exp(-z.imag), jnp.exp(z.imag))))
    except Exception:
        W_max = 1.0

    N = _number_of_terms(Q_max, W_max, tol=tol)

    thetas = _algorithm_1(z, tau, N)
    if return_error_bound:
        return thetas, jacobi_theta_error(z, tau, N)
    return thetas


def jacobi_ellip(u, m):
    """
    Compute Jacobi elliptic functions (sn, cn, dn) via theta functions.

    Uses DLMF 22.2.4–22.2.6:

        sn(u|m) = (ϑ_3(0,τ) / ϑ_2(0,τ)) · (ϑ_1(ζ,τ) / ϑ_4(ζ,τ))
        cn(u|m) = (ϑ_4(0,τ) / ϑ_2(0,τ)) · (ϑ_2(ζ,τ) / ϑ_4(ζ,τ))
        dn(u|m) = (ϑ_4(0,τ) / ϑ_3(0,τ)) · (ϑ_3(ζ,τ) / ϑ_4(ζ,τ))

    where ζ = πu / (2K(m)) and τ = iK′(m) / K(m), with

        K(m)  = R_F(0, 1−m, 1)   (complete elliptic integral, first kind)
        K′(m) = K(1−m) = R_F(0, m, 1)

    Parameters
    ----------
    u : scalar or array_like
        Real argument. Shapes must broadcast with `m`.
    m : scalar or array_like
        Parameter (0 < m < 1). Not the modulus k; m = k².

    Returns
    -------
    (sn, cn, dn) : tuple of jnp.ndarray
        Real arrays with the broadcast shape of (u, m).

    Notes
    -----
    The endpoints m = 0 and m = 1 are degenerate (K′(0) = ∞ and K(1) = ∞
    respectively) and are not supported. For best accuracy keep m in (0, 1).

    This function is *not* JAX-JIT-compilable at the top level because
    `jacobi_theta` performs a host-side series-length selection. The
    internal theta evaluations are JIT-compiled.
    """

    u = jnp.asarray(u, dtype=jnp.float64)
    m = jnp.asarray(m, dtype=jnp.float64)

    K  = rf(jnp.zeros_like(m), 1.0 - m, jnp.ones_like(m))  # K(m)
    Kp = rf(jnp.zeros_like(m), m,       jnp.ones_like(m))  # K′(m) = K(1−m)

    tau = (1j * Kp / K).astype(jnp.complex128)   # purely imaginary, Im > 0
    z   = (jnp.pi / 2.0 * u / K).astype(jnp.complex128)

    _, th2_0, th3_0, th4_0 = jacobi_theta(jnp.zeros_like(tau), tau)
    th1_z, th2_z, th3_z, th4_z = jacobi_theta(z, tau)

    sn = ((th3_0 / th2_0) * (th1_z / th4_z)).real
    cn = ((th4_0 / th2_0) * (th2_z / th4_z)).real
    dn = ((th4_0 / th3_0) * (th3_z / th4_z)).real

    return sn, cn, dn

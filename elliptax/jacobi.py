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

import math
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .carlson import rf

jax.config.update("jax_enable_x64", True)

# m needs to be clamped away from the degenerate endpoints 0 and 1 where rf(0, 0, 1) → ∞.
# _M_EPS tolerance chosen so that |q| = exp(-π Im(τ)) ≲ 0.77 even at the clamped value,
# keeping the theta series convergent with a manageable number of terms.
_M_EPS = 1e-15

# Static series length used when N is not given and the inputs are tracers
# (inside jit/vmap/grad), where select_series_length cannot concretise tau.
# Extra terms beyond the minimal N underflow to exact zeros, so a conservative
# fixed N loses no accuracy. After argument reduction Im(τ) ≥ √3/2 and
# |Im z| ≤ π·Im(τ)/2, giving a truncation bound 2·exp(−π·Im(τ)·(E−(N+2)/2)),
# which N = 10 pushes to ~4e-36; τ that escape the 4-round reduction (deep
# continued-fraction rationals) are still covered down to Im(τ) ≥ 0.45.
# A conditional retry at larger N was considered instead and rejected: the
# bound above makes the retry unreachable for every reduced τ, and inside
# vmap a lax.cond fallback degrades to select, always evaluating both tiers.
_N_STATIC = 10

# Rounds of modular (shift + invert) reduction applied to tau. Each round
# peels one continued-fraction term of Re(tau), so 4 rounds reach the
# fundamental domain for all tau except those clustered near rationals with
# continued-fraction depth > 4 — purely imaginary tau (the jacobi_ellip case)
# needs a single round.
_REDUCE_ROUNDS = 4


def _ellip_to_theta_args(u, m):
    """Convert jacobi_ellip arguments (u, m) to the (z, tau) used by jacobi_theta, clamping m away from 0 and 1."""
    m_safe = jnp.clip(m, _M_EPS, 1.0 - _M_EPS)
    K = rf(jnp.zeros_like(m_safe), 1.0 - m_safe, jnp.ones_like(m_safe))
    Kp = rf(jnp.zeros_like(m_safe), m_safe, jnp.ones_like(m_safe))
    tau = (1j * Kp / K).astype(jnp.complex128)
    z = (jnp.pi / 2.0 * u / K).astype(jnp.complex128)
    return z, tau


def _compose(perm, lcoef, sign, step_perm, step_l, step_sign):
    """
    Fold one transformation step into the accumulated relation

        θ_i(z₀, τ₀) = sign[i] · exp(lcoef[i]) · θ_{perm[i]}(z_cur, τ_cur),

    given the step's own relation θ_j(cur) = step_sign[j] · exp(step_l[j]) ·
    θ_{step_perm[j]}(next). All arrays have a trailing axis of length 4.
    """
    lcoef = lcoef + jnp.take_along_axis(step_l, perm, axis=-1)
    sign = sign * jnp.take_along_axis(step_sign, perm, axis=-1)
    perm = jnp.take_along_axis(step_perm, perm, axis=-1)
    return perm, lcoef, sign


def _reduce_theta_args(z, tau):
    """
    Reduce (z, τ) toward the fundamental domain using exact theta identities,
    so the series converges rapidly even as |q| → 1.

    Applies ``_REDUCE_ROUNDS`` rounds of the modular generators
    (DLMF 20.7.28 and 20.7.30–33)

        T:  τ → τ − n,   n = round(Re τ)      θ1,θ2 gain phase e^{inπ/4};
                                              θ3 ↔ θ4 for odd n
        S:  τ → −1/τ,  z → z·(−1/τ)  (where |τ| < 1)
                                              θ_j = c·θ_{σ(j)},  σ = (2 4),
                                              c = (−iτ)^{−1/2} e^{iτ'z²/π}
                                              (extra −i for θ1)

    followed by one quasi-periodicity shift z → z − aπ − bπτ (DLMF 20.2.6–9)
    to bring |Im z| down after inversions.

    The accumulated prefactor is tracked as sign · exp(lcoef) in log space:
    its magnitude can dwarf the float64 range mid-computation while the final
    values (or ratios of thetas) remain representable. The real part of lcoef
    is identical for all four thetas — only phases differ — so theta *ratios*
    pick up unit-modulus factors only.

    Returns
    -------
    (z_r, tau_r, perm, lcoef, sign) such that

        θ_i(z, τ) = sign[..., i] · exp(lcoef[..., i]) · θ_{perm[..., i]}(z_r, τ_r)

    with z_r, tau_r of the broadcast shape of (z, τ) and perm/lcoef/sign
    carrying an extra trailing axis of length 4 (theta index, 0-based).
    """
    z, tau = jnp.broadcast_arrays(
        jnp.asarray(z, dtype=jnp.complex128), jnp.asarray(tau, dtype=jnp.complex128)
    )
    shape = z.shape
    perm = jnp.broadcast_to(jnp.arange(4), shape + (4,))
    lcoef = jnp.zeros(shape + (4,), dtype=jnp.complex128)
    sign = jnp.ones(shape + (4,))

    ones = jnp.ones(shape)
    zeros = jnp.zeros(shape, dtype=jnp.complex128)
    id_perm = perm
    no_sign = jnp.ones(shape + (4,))

    def t_step(tau, perm, lcoef, sign):
        n = jnp.round(tau.real)
        tau = tau - n
        # e^{inπ/4} for θ1, θ2; reduce n mod 8 to keep the phase argument small
        ph = 1j * jnp.pi * jnp.mod(n, 8.0) / 4.0
        step_l = jnp.stack([ph, ph, zeros, zeros], axis=-1)
        odd = jnp.mod(n, 2.0) != 0.0
        i2 = jnp.where(odd, 3, 2)
        i3 = jnp.where(odd, 2, 3)
        step_perm = jnp.stack(
            [jnp.zeros_like(i2), jnp.ones_like(i2), i2, i3], axis=-1
        )
        perm, lcoef, sign = _compose(perm, lcoef, sign, step_perm, step_l, no_sign)
        return tau, perm, lcoef, sign

    s_perm = jnp.broadcast_to(jnp.array([0, 3, 2, 1]), shape + (4,))

    for _ in range(_REDUCE_ROUNDS):
        tau, perm, lcoef, sign = t_step(tau, perm, lcoef, sign)

        # S-inversion where |τ| < 1 (τ sanitised on the untaken branch so the
        # unused values stay finite — avoids NaN gradients through jnp.where).
        do = jnp.abs(tau) < 1.0
        tau_s = jnp.where(do, tau, 1j)
        tau_i = -1.0 / tau_s
        lc = -0.5 * jnp.log(-1j * tau_s) + 1j * tau_i * z * z / jnp.pi
        step_l = jnp.stack([lc - 1j * jnp.pi / 2.0, lc, lc, lc], axis=-1)
        p2, l2, s2 = _compose(perm, lcoef, sign, s_perm, step_l, no_sign)
        mask = do[..., None]
        perm = jnp.where(mask, p2, perm)
        lcoef = jnp.where(mask, l2, lcoef)
        sign = jnp.where(mask, s2, sign)
        z = jnp.where(do, z * tau_i, z)
        tau = jnp.where(do, tau_i, tau)

    # Final shift in case the last inversion moved Re τ out of [-1/2, 1/2].
    tau, perm, lcoef, sign = t_step(tau, perm, lcoef, sign)

    # Quasi-periodicity: z → z − aπ − bπτ with
    #   θ_j(z + aπ + bπτ) = ε_j q^{−b²} e^{−2ibz} θ_j(z),
    #   ε = ((−1)^{a+b}, (−1)^a, 1, (−1)^b).
    b = jnp.round(z.imag / (jnp.pi * tau.imag))
    a = jnp.round((z - b * jnp.pi * tau).real / jnp.pi)
    z = z - a * jnp.pi - b * jnp.pi * tau
    lsh = -b * b * (1j * jnp.pi * tau) - 2j * b * z
    parity = lambda x: jnp.where(jnp.mod(x, 2.0) == 0.0, 1.0, -1.0)
    step_l = jnp.stack([lsh, lsh, lsh, lsh], axis=-1)
    step_sign = jnp.stack([parity(a + b), parity(a), ones, parity(b)], axis=-1)
    perm, lcoef, sign = _compose(perm, lcoef, sign, id_perm, step_l, step_sign)

    return z, tau, perm, lcoef, sign


def select_series_length(u=None, m=None, z=None, tau=None, tol=1e-18):
    """
    Compute the theta series length N required to meet a truncation tolerance for
    the given inputs.

    This is the host-side sync that ``jacobi_theta`` performs automatically
    when ``N`` is not provided. Call it explicitly to control *when* the sync
    happens — for example, once during setup on CPU — then pass the returned
    ``N`` to ``jacobi_theta`` or ``jacobi_ellip`` to keep those calls
    sync-free and JIT-compilable.

    Supply either ``(u, m)`` for use with ``jacobi_ellip``, or ``(z, tau)``
    for use with ``jacobi_theta``.

    Parameters
    ----------
    u : scalar or array_like, optional
        Argument passed to ``jacobi_ellip``. Must be provided with ``m``.
    m : scalar or array_like, optional
        Parameter (0 < m < 1) passed to ``jacobi_ellip``. Must be provided
        with ``u``.
    z : scalar or array_like, optional
        Argument passed to ``jacobi_theta``. Must be provided with ``tau``.
    tau : scalar or array_like, optional
        Nome τ; Im(τ) > 0 everywhere. Must be provided with ``z``.
    tol : float
        Target truncation tolerance.

    Returns
    -------
    N : int
        Theta series length sufficient for the entire batch at ``tol``.
        Computed for the internally *reduced* arguments — the same modular
        reduction ``jacobi_theta`` applies before summing the series — so
        it stays small even as |q| → 1.

    Examples
    --------
    >>> N = select_series_length(u=u, m=m)
    >>> jit_ellip = jax.jit(lambda u, m: jacobi_ellip(u, m, N=N))

    >>> N = select_series_length(z=z, tau=tau)
    >>> jit_jacobi_theta = jax.jit(lambda z, tau: jacobi_theta(z, tau, N=N))
    """
    if u is not None and m is not None:
        u = jnp.asarray(u, dtype=jnp.float64)
        m = jnp.asarray(m, dtype=jnp.float64)
        z, tau = _ellip_to_theta_args(u, m)
    elif z is not None and tau is not None:
        z = jnp.asarray(z, dtype=jnp.complex128)
        tau = jnp.asarray(tau, dtype=jnp.complex128)
    else:
        raise ValueError("Provide either (u, m) or (z, tau).")

    # N is chosen for the *reduced* arguments — the same reduction
    # jacobi_theta applies before evaluating the series.
    z, tau, _, _, _ = _reduce_theta_args(z, tau)
    return _select_N(z, tau, tol)


def _select_N(z, tau, tol):
    """Series length for already-reduced (z, tau); concretises the inputs."""
    tau_imag_min = float(jnp.min(tau.imag))
    if tau_imag_min <= 0:
        raise ValueError(f"Im(τ) must be > 0 everywhere; got min = {tau_imag_min}")

    # Work with logs — after argument reduction W = exp(|Im z|) can exceed
    # the float64 range even though the convergence test itself is benign.
    # The bound is checked per element: large |Im z| and large |q| never
    # coincide after reduction, so pairing the batch-wide maxima of W and |q|
    # (which may come from different elements) would grossly overestimate N.
    log_Q = -jnp.pi * np.asarray(tau.imag, dtype=np.float64)  # log |q|, < 0

    # W may not be concretisable under jax.grad (z is a tracer);
    # fall back to W=1 — safe because Im(z)=0 is the common differentiating
    # point and N chosen from tau alone is sufficient there.
    try:
        log_W = np.abs(np.asarray(z.imag, dtype=np.float64))
    except Exception:
        log_W = np.zeros_like(log_Q)

    log_tol = math.log(tol)
    for N in range(2, 100):
        E = (N + 2) ** 2 // 4
        F = (N + 1) // 2 + 1
        ok = (F * log_Q + log_W < 0) & (E * log_Q + (N + 2) * log_W < log_tol)
        if np.all(ok):
            return N
    raise ValueError(
        f"Series failed to converge within tol={tol}; ensure |q| < 1 everywhere."
    )


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
    w = jnp.exp(1j * z)
    v = 1.0 / w

    # Even k = 2(n-1):  ms = n²,      kmod2 = 0  →  θ3, θ4
    # Odd  k = 2n-1:    ms = n(n+1),  kmod2 = 1  →  θ1, θ2
    # Sign for both θ4 and θ1 is (-1)^n. (Algorithm 1, lines 11-30.)
    ns_even = jnp.arange(1, N // 2 + 1)  # n = 1, 2, ..., N // 2
    ns_odd = jnp.arange(1, (N + 1) // 2 + 1)  # n = 1, 2, ..., (N+1) // 2
    sign = jnp.where(ns_even % 2 == 0, 1.0, -1.0)  # (-1)^n, same for both

    # Each term w^{2n} q^{ms} = exp(2inz + iπτ·ms) is formed as a single
    # exponential: computing the factors separately lets w^{2n} overflow to
    # inf while q^{ms} underflows to 0, producing NaN instead of the correct
    # (vanishing) product. The combined exponent decays whenever the series
    # converges, so terms past the required length underflow to exact zeros.
    lq = 1j * jnp.pi * tau  # log q
    lw = 2j * z  # log w²
    t_even = jnp.exp(lq * ns_even**2 + lw * ns_even) + jnp.exp(
        lq * ns_even**2 - lw * ns_even
    )
    p_odd = jnp.exp(lq * ns_odd * (ns_odd + 1) + lw * ns_odd)
    m_odd = jnp.exp(lq * ns_odd * (ns_odd + 1) - lw * (ns_odd + 1))
    t_odd = p_odd + m_odd
    u_odd = p_odd - m_odd

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
        Number of terms used in the series (as returned by
        ``select_series_length``).

    Returns
    -------
    error : jnp.ndarray
        Upper bound on the *relative* truncation error, with the broadcast
        shape of (z, τ). The same bound applies to all four theta functions.

    Notes
    -----
    The bound is evaluated for the reduced arguments actually fed to the
    series (see ``_reduce_theta_args``); the reduction prefactor scales the
    truncation error and the function value alike, so the relative bound is
    unchanged by it.
    """
    z_r, tau_r, _, _, _ = _reduce_theta_args(z, tau)
    return _series_error_bound(z_r, tau_r, N)


def _series_error_bound(z, tau, N):
    """Truncation bound (eq. 1.18) for arguments fed directly to the series."""
    Q = jnp.exp(-jnp.pi * tau.imag)
    W = jnp.maximum(jnp.exp(-z.imag), jnp.exp(z.imag))

    E = (N + 2) ** 2 // 4
    F = (N + 1) // 2 + 1

    return 2.0 * Q**E * W ** (N + 2) / (1.0 - Q**F * W)


def jacobi_theta(z, tau, N=None, tol=1e-18, return_error_bound=False):
    """
    Compute (θ1, θ2, θ3, θ4) at (z, τ) using Johansson's Algorithm 1.

    Parameters
    ----------
    z, tau : scalar or jnp.ndarray
        Any shapes that broadcast together. tau must have Im(τ) > 0
        everywhere.
    N : int, optional
        Fix the number of series terms, skipping the host-device sync.
        Obtain a suitable value from ``select_series_length(z, tau, tol)``.
    tol : float
        Target truncation tolerance; defaults to ~ float64 epsilon.
        Ignored if N is provided.
    return_error_bound : bool
        If True, also return the truncation error bound array.

    Returns
    -------
    (θ1, θ2, θ3, θ4) : tuple of jnp.ndarray
        Each output has the broadcast shape of (z, τ).

    Notes
    -----
    The arguments are first reduced toward the fundamental domain
    {|τ| ≥ 1, |Re τ| ≤ 1/2} using exact modular and quasi-periodicity
    identities (see ``_reduce_theta_args``), so accuracy is maintained as
    |q| → 1 (small Im τ): only a handful of series terms are ever needed.
    Precision degrades gradually with the size of the reduction prefactor's
    exponent, roughly ε·|z|²/Im(τ) — e.g. ~1e-11 relative error at
    Im τ = 0.001 with |z| ~ π. τ clustered near rationals of continued-
    fraction depth > 4 may not reduce fully; ``jacobi_theta_error`` gives a
    runtime bound if in doubt.

    When N is None and the inputs are concrete, ``select_series_length``
    is called internally, which causes a host-device sync. Under jit, vmap
    or grad (where the inputs are tracers and cannot be inspected) a
    conservative static length ``N = 10`` is used instead — after reduction
    this covers the full domain with a large margin; terms beyond the
    minimal length underflow to exact zeros. Pass N explicitly to avoid
    the sync in eager mode or to override the static fallback.

    Examples
    --------
    >>> N = select_series_length(z=z, tau=tau)
    >>> jit_jacobi_theta = jax.jit(lambda z, tau: jacobi_theta(z, tau, N=N))
    """
    thp, lcoef, sign, z_r, tau_r, N = _theta_parts(z, tau, N, tol)
    out = sign * jnp.exp(lcoef) * thp
    thetas = (out[..., 0], out[..., 1], out[..., 2], out[..., 3])
    if return_error_bound:
        return thetas, _series_error_bound(z_r, tau_r, N)
    return thetas


def _theta_parts(z, tau, N, tol):
    """
    Reduce (z, τ), resolve N, and evaluate the series.

    Returns (thp, lcoef, sign, z_r, tau_r, N) where the original thetas are
    θ_i(z, τ) = sign[..., i] · exp(lcoef[..., i]) · thp[..., i]. Keeping the
    log-prefactor unexpanded lets callers form theta *ratios* without
    overflow: Re(lcoef) is identical across the four thetas, so it cancels
    exactly in lcoef[..., i] − lcoef[..., j].
    """
    z = jnp.asarray(z, dtype=jnp.complex128)
    tau = jnp.asarray(tau, dtype=jnp.complex128)

    z_r, tau_r, perm, lcoef, sign = _reduce_theta_args(z, tau)

    if N is None:
        try:
            N = _select_N(z_r, tau_r, tol)
        except jax.errors.ConcretizationTypeError:
            # Inside jit/vmap/grad the inputs are tracers, so the adaptive
            # selection cannot run; fall back to the conservative static
            # length. Check jacobi_theta_error (or return_error_bound=True)
            # if in doubt.
            N = _N_STATIC

    th = jnp.stack(_algorithm_1(z_r, tau_r, N), axis=-1)
    thp = jnp.take_along_axis(th, perm, axis=-1)
    return thp, lcoef, sign, z_r, tau_r, N


def jacobi_ellip(u, m, N=None):
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
    N : int, optional
        Fix the number of theta-series terms. When provided, no host-device
        sync is performed. See ``jacobi_theta`` for details.

    Returns
    -------
    (sn, cn, dn) : tuple of jnp.ndarray
        Real arrays with the broadcast shape of (u, m).

    Notes
    -----
    The endpoints m = 0 and m = 1 are handled exactly via their limiting
    trigonometric / hyperbolic expressions:

    * m = 0: sn = sin u,  cn = cos u,  dn = 1
    * m = 1: sn = tanh u, cn = sech u, dn = sech u

    For m strictly in (0, 1) the theta-function representation is used.

    Without an explicit N, this function performs a host-device sync to
    select the series length automatically when the inputs are concrete;
    under jit/vmap/grad it falls back to a conservative static length
    (see ``jacobi_theta``). Pass ``N`` explicitly to avoid the sync and
    control the series length end-to-end.

    Examples
    --------
    >>> N = select_series_length(u=u, m=m)
    >>> jit_ellip = jax.jit(lambda u, m: jacobi_ellip(u, m, N=N))
    """

    u = jnp.asarray(u, dtype=jnp.float64)
    m = jnp.asarray(m, dtype=jnp.float64)

    z, tau = _ellip_to_theta_args(u, m)

    # Ratio form: θ_i/θ_j = (sign_i·sign_j) · exp(lcoef_i − lcoef_j) · thp_i/thp_j.
    # Re(lcoef) cancels exactly in the difference, so the ratios stay finite
    # even where the individual theta values would overflow float64
    # (e.g. m → 1 with large u).
    p0, l0, s0, *_ = _theta_parts(jnp.zeros_like(tau), tau, N, 1e-18)
    pz, lz, sz, *_ = _theta_parts(z, tau, N, 1e-18)

    def ratio(p, l, s, i, j):
        return (s[..., i] * s[..., j]) * jnp.exp(l[..., i] - l[..., j]) * (
            p[..., i] / p[..., j]
        )

    sn_t = (ratio(p0, l0, s0, 2, 1) * ratio(pz, lz, sz, 0, 3)).real
    cn_t = (ratio(p0, l0, s0, 3, 1) * ratio(pz, lz, sz, 1, 3)).real
    dn_t = (ratio(p0, l0, s0, 3, 2) * ratio(pz, lz, sz, 2, 3)).real

    # Exact limiting expressions at the degenerate endpoints.
    sn = jnp.where(m == 0.0, jnp.sin(u), jnp.where(m == 1.0, jnp.tanh(u), sn_t))
    cn = jnp.where(m == 0.0, jnp.cos(u), jnp.where(m == 1.0, 1.0 / jnp.cosh(u), cn_t))
    dn = jnp.where(
        m == 0.0, jnp.ones_like(u), jnp.where(m == 1.0, 1.0 / jnp.cosh(u), dn_t)
    )

    return sn, cn, dn

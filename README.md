# elliptax

**elliptax** is a [JAX](https://github.com/google/jax) library of elliptic integrals and related special functions, supporting `float64` precision, GPU/TPU execution, and — where applicable — forward and reverse mode automatic differentiation.

## Installation

Clone the repository and install from source:

```bash
git clone https://github.com/your-username/elliptax.git
cd elliptax
pip install -e .
```

This will also install all dependencies, including JAX. `float64` support must be enabled before use:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

## Implemented functions

### Bulirsch integrals (`elliptax.bulirsch`)

Forward and reverse mode autodiff compatible

| Function | Description |
|---|---|
| `el1(x, kc)` | $el1(x, k_c)$ - Incomplete elliptic integral of the first kind |
| `el2(x, kc, a, b)` | $el2(x, k_c, a, b)$ - Incomplete elliptic integral of the second kind |
| `el3(x, kc, p)` | $el3(x, k_c, p)$ - Incomplete elliptic integral of the third kind |
| `cel(kc, p, a, b)` | $cel(k_c, p, a, b)$ - Generalized complete elliptic integral |

### Carlson symmetric integrals (`elliptax.carlson`)

Forward mode autodiff only.

| Function | Description |
|---|---|
| `rf(x, y, z)` | $R_F(x, y, z)$ — symmetric integral of the first kind |
| `rc(x, y)` | $R_C(x, y)$ |
| `rj(x, y, z, p)` | $R_J(x, y, z, p)$ — symmetric integral of the third kind |
| `rd(x, y, z)` | $R_D(x, y, z)$ — degenerate symmetric integral |
| `rg(x, y, z)` | $R_G(x, y, z)$ — symmetric integral of the second kind |

A scipy-compatible interface is also available under `elliptax.special` (see below).

### Legendre-form integrals — modulus convention (`elliptax.legendre`)

Forward mode autodiff only.  

All functions take the **modulus** $k \in (0, 1)$, where $m = k^2$ is the parameter used by scipy and mpmath.

| Function | Description |
|---|---|
| `ellipk(k)` | $K(k)$ — complete integral of the first kind |
| `ellipe(k)` | $E(k)$ — complete integral of the second kind |
| `ellippi(n, k)` | $\Pi(n, k)$ — complete integral of the third kind |
| `ellipfinc(phi, k)` | $F(\phi, k)$ — incomplete integral of the first kind |
| `ellipeinc(phi, k)` | $E(\phi, k)$ — incomplete integral of the second kind |
| `ellippiinc(phi, k, n)` | $\Pi(\phi, k, n)$ — incomplete integral of the third kind |

A scipy-compatible interface is also available under `elliptax.special` (see below).

### Jacobi functions (`elliptax.jacobi`)

| Function | Description |
|---|---|
| `jacobi_theta(z, tau)` | $(\vartheta_1, \vartheta_2, \vartheta_3, \vartheta_4)$ — all four Jacobi theta functions at complex $(z, \tau)$ |
| `jacobi_ellip(u, m)` | $({\rm sn}, {\rm cn}, {\rm dn})$ — Jacobi elliptic functions |

The series length for `jacobi_theta` is selected automatically to meet a target tolerance (default $\approx 10^{-18}$). An error bound can be returned alongside the values with `return_error_bound=True`.

### Interface for scipy.special compatibility (`elliptax.special`)

This module wraps the functions of interest above to match the `scipy.special` interface, which uses the parameter $m = k^2$ instead of the modulus $k$ for the Legendre-form functions.

| Function | Description |
|---|---|
| `ellipj(u, m)` | $({\rm sn}, {\rm cn}, {\rm dn}, {\rm ph})$ — Jacobi elliptic functions and amplitude |
| `ellipk(m)` | $K(m)$ — complete integral of the first kind |
| `ellipe(m)` | $E(m)$ — complete integral of the second kind |
| `ellippi(n, m)` | $\Pi(n, m)$ — complete integral of the third kind |
| `ellipkinc(phi, m)` | $F(\phi, m)$ — incomplete integral of the first kind |
| `ellipeinc(phi, m)` | $E(\phi, m)$ — incomplete integral of the second kind |
| `ellippiinc(phi, m, n)` | $\Pi(\phi, m, n)$ — incomplete integral of the third kind |
| `elliprf(x, y, z)` | $R_F(x, y, z)$ — symmetric integral of the first kind |
| `elliprc(x, y)` | $R_C(x, y)$ |
| `elliprd(x, y, z)` | $R_D(x, y, z)$ — degenerate symmetric integral |
| `elliprj(x, y, z, p)` | $R_J(x, y, z, p)$ — symmetric integral of the third kind |
| `elliprg(x, y, z)` | $R_G(x, y, z)$ — symmetric integral of the second kind |

## Quick example

```python
import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from elliptax import ellipk, ellipe, jacobi_ellip

k = jnp.linspace(0.1, 0.9, 9)
print(ellipk(k))   # K(k) for modulus k
print(ellipe(k))   # E(k) for modulus k

sn, cn, dn = jacobi_ellip(1.0, 0.5)
```

## Testing

The test suite lives in `pytests/` and uses [pytest](https://docs.pytest.org/). Each module is tested independently against [scipy.special](https://docs.scipy.org/doc/scipy/reference/special.html) (for most functions) and [mpmath](https://mpmath.org/) (for the third-kind and theta functions, where scipy has no equivalent).

```bash
pip install pytest mpmath
pytest pytests/ -v
```

| Test file | What it covers | Reference |
|---|---|---|
| `carlson_test.py` | $R_F, R_C, R_J, R_D, R_G$ — reference values and duplication identities | Carlson (1994) §3; scipy |
| `legendre_test.py` | Legendre-form integrals, modulus convention | scipy, mpmath |
| `special_test.py` | Legendre-form integrals, parameter convention | scipy, mpmath |
| `jacobi_test.py` | Theta functions $\vartheta_{1\text{–}4}$ and Jacobi elliptic functions | scipy, mpmath |

All tests compare against external references to at least float64 relative precision ($\lesssim 10^{-12}$). The mpmath evaluations use 30-digit working precision as ground truth.

## References

[1] Bulirsch, R. (1969). [Numerical calculation of elliptic integrals and elliptic functions III](https://doi.org/10.1007/BF02165405). *Numerische Mathematik*, 13, 305–315.

[2] Carlson, B. C. (1994). [Numerical computation of real or complex elliptic integrals](https://doi.org/10.1007/BF02198293). *Numerical Algorithms*, 10, 13–26.

[3] NIST Digital Library of Mathematical Functions. [§19.25 — Relations to Other Functions](https://dlmf.nist.gov/19.25#i).

[4] Johansson, F. (2018). [Numerical evaluation of elliptic functions, elliptic integrals and modular forms](https://arxiv.org/abs/1806.06725). *arXiv:1806.06725*.

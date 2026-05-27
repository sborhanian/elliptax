# Contributing to elliptax

Thank you for your interest in contributing. This document covers how to set up
the development environment, run the test suite, and submit changes.

## Authors

elliptax was created and is maintained by
 - Tyler Gordon 
 - Ssohrab Borhanian

## Development setup

Clone the repository and install in editable mode:

```bash
git clone https://github.com/your-username/elliptax.git
cd elliptax
pip install -e .
```

Install the optional test dependencies:

```bash
pip install pytest mpmath
```

## Running the tests

```bash
pytest pytests/ -v
```

All tests must pass before a change is submitted. The test suite compares
against scipy and mpmath at float64 precision; a new function should come with
a corresponding test in the appropriate file under `pytests/`.

| Test file | Module under test |
|---|---|
| `carlson_test.py` | `elliptax.carlson` |
| `legendre_test.py` | `elliptax.legendre` |
| `special_test.py` | `elliptax.special` |
| `jacobi_test.py` | `elliptax.jacobi` |

## Code conventions

- **Precision**: all functions must work in `float64`. Enable it at the top of
  every module with `jax.config.update("jax_enable_x64", True)`.
- **Decorators**: scalar functions should be decorated with `@jax.jit` and
  `@jnp.vectorize` so they broadcast over arrays automatically.
- **Autodiff**: if a function uses a non-differentiable primitive (e.g.
  `jax.lax.while_loop`), provide a `@jax.custom_jvp` rule.
- **References**: every new function should cite the algorithm or identity it
  implements (DLMF section, paper DOI, or arXiv ID) in its docstring.
- **No complex support** (unless explicitly stated): the current implementations
  are real-valued only, except for `jacobi_theta` which accepts complex inputs
  by design.

## Submitting changes

1. Fork the repository and create a branch for your change.
2. Add or update tests as appropriate.
3. Open a pull request with a clear description of what was changed and why.

## Reporting issues

Please open a GitHub issue describing the problem, the inputs that trigger it,
and the expected versus actual output.

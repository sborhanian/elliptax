# Changelog

All notable changes to elliptax are documented here.

## [0.1.2] — 2026-05-29

### Added
- `select_series_length`: public function that performs the host-device sync needed to choose the theta-series length N. Calling it once ahead of time and passing the result as `N` to `jacobi_theta` or `jacobi_ellip` makes those calls sync-free and JIT-compilable, which is the recommended pattern on GPU/TPU.
- `jacobi_theta` and `jacobi_ellip` now accept an optional `N` argument to skip the automatic series-length selection entirely.
- README section "GPU/TPU usage" under Jacobi functions documenting the `select_series_length` workflow with examples.

### Changed
- Internal helper `_ellip_to_theta_args(u, m)` extracted to centralise the $(u, m) \to (z, \tau)$ conversion used by both `jacobi_ellip` and `select_series_length`.

---

## [0.1.1] — 2026-05-27

### Changed
- Renamed `jacobi.ellipj` to `jacobi.jacobi_ellip` for clarity.
- Expanded and restructured `README.md`: installation, per-module function tables, testing section, full references.
- Unified test argument-sampling helpers into a single `_args(key_seed, N)` function per test file.
- Removed unused `ellipk_complement` from `elliptax.legendre`, `elliptax.special`, and `elliptax.__init__`.

### Added
- Test suite for `elliptax.legendre` (`pytests/legendre_test.py`) — compared against scipy and mpmath.
- Test suite for `elliptax.special` (`pytests/special_test.py`) — compared against scipy and mpmath.
- `CHANGELOG.md` (this file).
- `CONTRIBUTING.md` with authors (Tyler Gordon, Ssohrab Borhanian), development setup, code conventions, and contribution guidelines.

---

## [0.1.0] — 2026-05-21

### Added
- `elliptax.jacobi`: Jacobi theta functions (`jacobi_theta`, `jacobi_theta_error`) via Johansson's Algorithm 1 (arXiv:1806.06725). Supports complex inputs, adaptive series-length selection, and error-bound estimation.
- `elliptax.jacobi`: Jacobi elliptic functions (`jacobi_ellip`) via the theta-function representation (DLMF 22.2.4–22.2.6).
- `elliptax.special`: scipy-compatible interface wrapping Legendre-form integrals (parameter $m = k^2$ convention), Carlson integrals, and Jacobi elliptic functions.
- Test suites for Carlson integrals (`pytests/carlson_test.py`) and Jacobi functions (`pytests/jacobi_test.py`), with comparisons against scipy and mpmath.

### Changed
- Cleaned up module imports in `elliptax/__init__.py`.

---

## [0.0.3] — 2026-04-17

### Added
- `elliptax.carlson`: Carlson symmetric integral $R_G$ (`rg`).
- scipy-compatible alias `elliprg` in `elliptax.special`.
- Expanded demo notebook.

### Changed
- Packaged the project as `elliptax` with `pyproject.toml`.

---

## [0.0.2] — 2024-07-24

### Added
- `elliptax.legendre`: Legendre-form integrals ($K$, $E$, $\Pi$, $F$, $E_{\rm inc}$, $\Pi_{\rm inc}$) reimplemented via Carlson integrals using DLMF §19.25 relations.

### Fixed
- Bug in `el3` for edge cases where $k_c > 1$.
- Bug in Legendre $E$.
- Special-case handling in `el3` near singular parameter values.
- Added `maxiter` guard to Bulirsch iteration loops.

---

## [0.0.1] — 2024-07-12

### Added
- `elliptax.bulirsch`: Bulirsch incomplete elliptic integrals `el1`, `el2`, `el3`, and general complete integral `cel`, with custom JVPs for forward mode autodiff.
- `elliptax.carlson`: Carlson symmetric integrals $R_F$, $R_C$, $R_J$, $R_D$ via duplication-algorithm iteration.
- GPL license.

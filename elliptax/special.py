import jax
import jax.numpy as jnp

import elliptax.carlson as carlson
import elliptax.jacobi as jacobi

@jax.jit
@jnp.vectorize
def ellipk(m):
    r"""JAX implementation of the complete elliptic integral of the first kind 

    .. math::

        \[K\left(m\right)=\int_{0}^{\pi/2}\frac{\,\mathrm{d}\theta}{\sqrt{1-m}{%
\sin}^{2}\theta}}]

     Args:
       m: arraylike, real valued.

     Returns:
       The value of the complete elliptic integral of the first kind, :math:`K(m)`

     Notes:
       ``ellipk`` does not support complex-valued inputs.
       ``ellipk`` requires `jax.config.update("jax_enable_x64", True)`
    """

    kc2 = 1 - m
    return carlson.rf(0.0, kc2, 1.0)

@jax.jit
@jnp.vectorize
def ellipe(m):
    r"""JAX implementation of the complete elliptic integral of the second kind 

    .. math::

        \[E\left(m\right)=\int_{0}^{\pi/2}\sqrt{1-m{\sin}^{2}\theta}\,\mathrm{d}%
\theta\]

     Args:
       m: arraylike, real valued.

     Returns:
       The value of the complete elliptic integral of the second kind, :math:`E(m)`

     Notes:
       ``ellipe`` does not support complex-valued inputs.
       ``ellipe`` requires `jax.config.update("jax_enable_x64", True)`
    """

    kc2 = 1 - m
    return kc2 * (carlson.rd(0.0, kc2, 1.0) + carlson.rd(0.0, 1.0, kc2)) / 3.0

@jax.jit
@jnp.vectorize
def ellippi(n, m):
    r"""JAX implementation of the complete elliptic integral of the third kind 

    .. math::

        \[\Pi\left(n, m\right)=\int_{0}^{\pi/2}\frac{\,\mathrm{d}\theta}{%
\sqrt{1-m}{\sin}^{2}\theta}(1+n{\sin}^{2}\theta)}]

     Args:
       n: arraylike, real valued.
       m: arraylike, real valued.

     Returns:
       The value of the complete elliptic integral of the third kind, :math:`\Pi(n, m)`

     Notes:
       ``ellippi`` does not support complex-valued inputs.
       ``ellippi`` requires `jax.config.update("jax_enable_x64", True)`
    """

    kc2 = 1 - m
    return n * carlson.rj(0.0, kc2, 1.0, 1 - n) / 3.0 + ellipk(m)

@jax.jit
@jnp.vectorize
def ellipkinc(phi, m):
    r"""JAX implementation of the incomplete elliptic integral of the first kind 

    .. math::

        \[F\left(\phi,m\right)=\int_{0}^{\phi}\frac{\,\mathrm{d}\theta}{\sqrt{1-m{%
\sin}^{2}\theta}}]

     Args:
       phi: arraylike, real valued.
       m: arraylike, real valued.

     Returns:
       The value of the complete elliptic integral of the first kind, :math:`F(\phi, m)`

     Notes:
       ``ellipfinc`` does not support complex-valued inputs.
       ``ellipfinc`` requires `jax.config.update("jax_enable_x64", True)`
    """

    c = 1.0 / jnp.sin(phi)**2
    return carlson.rf(c - 1, c - m, c)
    
@jax.jit
@jnp.vectorize
def ellipeinc(phi, m):
    r"""JAX implementation of the incomplete elliptic integral of the second kind 

    .. math::

        \[E\left(\phi,m\right)=\int_{0}^{\phi}\sqrt{1-m{\sin}^{2}\theta}\,\mathrm{d}%
\theta\\]

     Args:
       phi: arraylike, real valued.
       m: arraylike, real valued.

     Returns:
       The value of the complete elliptic integral of the second kind, :math:`E(\phi, m)`

     Notes:
       ``ellipeinc`` does not support complex-valued inputs.
       ``ellipeinc`` requires `jax.config.update("jax_enable_x64", True)`
    """

    c = 1.0 / jnp.sin(phi)**2
    return carlson.rf(c - 1, c - m, c) - m * carlson.rd(c - 1, c - m, c) / 3.0

@jax.jit
@jnp.vectorize
def ellippiinc(phi, m, n):
    r"""JAX implementation of the incomplete elliptic integral of the third kind 

    .. math::

        \[E\left(\phi,m\right)=\int_{0}^{\phi}\sqrt{1-m{\sin}^{2}\theta}\,\mathrm{d}%
\theta\\]

     Args:
       phi: arraylike, real valued.
       m: arraylike, real valued.
       n: arraylike, real valued.

     Returns:
       The value of the complete elliptic integral of the third kind, :math:`\Pi(\phi, m)'

     Notes:
       ``ellippiinc`` does not support complex-valued inputs.
       ``ellippiinc`` requires `jax.config.update("jax_enable_x64", True)`
    """

    c = 1.0 / jnp.sin(phi)**2
    return n * carlson.rj(c - 1, c - m, c, c - n) / 3.0 + ellipkinc(phi, m)


@jax.jit
@jnp.vectorize
def ellipk_complement(k):

    r"""JAX implementation of the complemenary complete elliptic integral of the first kind 

     Args:
       k: arraylike, real valued, with abs(k) <= 1.

     Returns:
       The value of the complemenary complete elliptic integral of the first kind, :math:`K'(k)`

     Notes:
       ``ellipk_complement`` does not support complex-valued inputs.
       ``ellipk_complement`` requires `jax.config.update("jax_enable_x64", True)`
    """

    return ellipk(jnp.sqrt(1 - jnp.abs(k)))

@jax.jit
@jnp.vectorize
def elliprf(x, y, z):
    r"""JAX implementation of the Carlson symmetric elliptic integral of the first kind 

    .. math::

        \[R_F\left(x, y, z\right)=\frac{1}{2}\int_{0}^{\infty}\left(t+x\right)^{-\frac{1}{2}}\left(t+y\right)^{-\frac{1}{2}}\left(t+z\right)^{-\frac{1}{2}}\,\mathrm{d}t\]

     Args:
       x: arraylike, real valued.
       y: arraylike, real valued.
       z: arraylike, real valued.

     Returns:
       The value of the Carlson symmetric elliptic integral of the first kind, :math:`R_F(x, y, z)`

     Notes:
       ``elliprf`` does not support complex-valued inputs.
       ``elliprf`` requires `jax.config.update("jax_enable_x64", True)`
    """

    return carlson.rf(x, y, z)

@jax.jit
@jnp.vectorize
def elliprc(x, y):
    r"""JAX implementation of the Carlson symmetric elliptic integral of the second kind 

    .. math::

        \[R_C\left(x, y\right)=\frac{1}{2}\int_{0}^{\infty}\left(t+x\right)^{-\frac{1}{2}}\left(t+y\right)^{-1}\,\mathrm{d}t\]

     Args:
       x: arraylike, real valued.
       y: arraylike, real valued.

     Returns:
       The value of the Carlson symmetric elliptic integral of the second kind, :math:`R_C(x, y)`

     Notes:
       ``elliprc`` does not support complex-valued inputs.
       ``elliprc`` requires `jax.config.update("jax_enable_x64", True)`
    """

    return carlson.rc(x, y)

@jax.jit
@jnp.vectorize
def elliprd(x, y, z):
    r"""JAX implementation of the Carlson symmetric elliptic integral of the third kind 

    .. math::

        \[R_D\left(x, y, z\right)=\frac{3}{2}\int_{0}^{\infty}\left(t+x\right)^{-\frac{1}{2}}\left(t+y\right)^{-\frac{1}{2}}\left(t+z\right)^{-\frac{3}{2}}\,\mathrm{d}t\]

     Args:
       x: arraylike, real valued.
       y: arraylike, real valued.
       z: arraylike, real valued.

     Returns:
       The value of the Carlson symmetric elliptic integral of the third kind, :math:`R_D(x, y, z)`

     Notes:
       ``elliprd`` does not support complex-valued inputs.
       ``elliprd`` requires `jax.config.update("jax_enable_x64", True)`
    """

    return carlson.rd(x, y, z)

@jax.jit
@jnp.vectorize
def elliprj(x, y, z, p):
    r"""JAX implementation of the Carlson symmetric elliptic integral of the fourth kind 

    .. math::

        \[R_J\left(x, y, z, p\right)=\frac{3}{2}\int_{0}^{\infty}\left(t+x\right)^{-\frac{1}{2}}\left(t+y\right)^{-\frac{1}{2}}\left(t+z\right)^{-\frac{1}{2}}\left(t+p\right)^{-1}\,\mathrm{d}t\]

     Args:
       x: arraylike, real valued.
       y: arraylike, real valued.
       z: arraylike, real valued.
       p: arraylike, real valued.

     Returns:
       The value of the Carlson symmetric elliptic integral of the fourth kind, :math:`R_J(x, y, z, p)`

     Notes:
       ``elliprj`` does not support complex-valued inputs.
       ``elliprj`` requires `jax.config.update("jax_enable_x64", True)`
    """

    return carlson.rj(x, y, z, p)

@jax.jit
@jnp.vectorize
def elliprg(x, y, z):
    r"""JAX implementation of the Carlson symmetric elliptic integral of the fifth kind 

    .. math::

        \[R_G\left(x, y, z\right)=\frac{1}{4}\int_{0}^{\infty}\left(t+x\right)^{-\frac{1}{2}}\left(t+y\right)^{-\frac{1}{2}}\left(t+z\right)^{-\frac{1}{2}}\left(\sqrt{t+x}+\sqrt{t+y}+\sqrt{t+z}\right)^{2}\,\mathrm{d}t\]

     Args:
       x: arraylike, real valued.
       y: arraylike, real valued.
       z: arraylike, real valued.

     Returns:
       The value of the Carlson symmetric elliptic integral of the fifth kind, :math:`R_G(x, y, z)`

     Notes:
       ``elliprg`` does not support complex-valued inputs.
       ``elliprg`` requires `jax.config.update("jax_enable_x64", True)`
    """

    return carlson.rg(x, y, z)

=================
Why this Package?
=================

So, why would one opt for this package?

It is **easy** to *install* (using pip) and intuitive to *use*.

**SHEAP** features:

- A fully JAX-based engine for spectral modeling and optimization, enabling GPU acceleration and efficient parallel processing.
- Emission line fitting with physically motivated constraints (tied parameters, width limits, blended features).
- Modular spectral region handling using YAML-based definitions with the `RegionBuilder` and `RegionFitting` modules.
- Support for complex multi-component models: broad and narrow lines, Fe~II emission, Balmer continuum, and coronal lines.
- Native propagation of parameter uncertainties in all calculations using the `auto_uncertainties` package, fully compatible with JAX’s autodiff.
- Flexible constraint systems for additive and multiplicative parameter dependencies during minimization.
- Seamless integration with scientific libraries: `astropy`, `scipy`, `lmfit`, and `optax`.

Well-tested against modern Python versions (3.12 and 3.13),
and validated on both *Linux* (Ubuntu) and *Darwin* (macOS) platforms.

Tests trigger automatically on **CI**.
The package's releases follow **Semantic Versioning**, ensuring backward compatibility and transparency across versions.

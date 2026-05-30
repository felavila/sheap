Optimization Strategy
=====================

The spectral fitting in :code:`sheap` uses the Adam optimizer by default but can also use other optimizers available in the `Optax <https://optax.readthedocs.io/en/latest/index.html>`_ library. These optimizers provide efficient, adaptive gradient updates within JAX and are well-suited to the non-linear, high-dimensional parameter spaces of AGN spectral models.


The total loss minimized during optimization combines several terms:

- a residual loss based on log-cosh,
- an optional parameter penalty term,
- a curvature-matching term on the second derivative of the flux,
- and a smoothness term on the residuals.

The primary component is the log-cosh residual, which behaves quadratically for small residuals and linearly for large ones, making it robust to outliers while preserving sensitivity to high-S/N features:

.. math::
   \mathcal{L}_{\text{residual}} =
   \left\langle \log\!\cosh\!\left( \frac{f_{\mathrm{model}} - f_{\mathrm{obs}}}{\sigma} \right) \right\rangle
   \;+\;
   \alpha \cdot \max_{i}\, \log\!\cosh\!\left( \frac{f_{\mathrm{model}} - f_{\mathrm{obs}}}{\sigma} \right)

where :math:`\alpha` is a small weight that emphasizes the worst residual pixel.

In addition, the model can include a curvature term to match the second derivatives of the predicted and observed fluxes:

.. math::
   \mathcal{L}_{\text{curvature}} =
   \gamma \cdot \left\langle \big(f''_{\mathrm{model}} - f''_{\mathrm{obs}}\big)^2 \right\rangle

and a smoothness constraint on the residual vector:

.. math::
   \mathcal{L}_{\text{smoothness}} =
   \delta \cdot \left\langle \big(\nabla (f_{\mathrm{model}} - f_{\mathrm{obs}})\big)^2 \right\rangle

Here, :math:`\gamma` and :math:`\delta` are hyperparameters that control the contribution of curvature and smoothness regularization, respectively.

The total loss minimized is then

.. math::
   \mathcal{L}_{\text{total}} =
   \mathcal{L}_{\text{residual}}
   \;+\; \mathcal{L}_{\text{curvature}}
   \;+\; \mathcal{L}_{\text{smoothness}}
   \;+\; \lambda\,\mathcal{P}(\theta),

where :math:`\mathcal{P}(\theta)` is an optional penalty function on the model parameters and :math:`\lambda` is its associated weight.

.. note::
   Angle brackets :math:`\langle \cdot \rangle` denote averaging over spectral pixels; :math:`\max_i` denotes the maximum over pixels.

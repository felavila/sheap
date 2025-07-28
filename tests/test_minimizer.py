import pytest
import jax.numpy as jnp
import numpy as np
from sheap.Minimizer.Minimizer import Minimizer


def linear_model(x, params):
    return params[0] * x + params[1]


def test_minimizer_adam_basic_fit():
    # simple y = 2x + 1 model
    x = jnp.linspace(0, 10, 20)
    true_params = jnp.array([2.0, 1.0])
    y = linear_model(x, true_params)
    y_err = jnp.ones_like(y) * 0.1
    init_params = jnp.array([0.5, 0.5])

    x = x[None, :]
    y = y[None, :]
    y_err = y_err[None, :]
    init_params = init_params[None, :]

    constraints = jnp.array([[-10, 10], [-10, 10]])

    minimizer = Minimizer(
        func=linear_model,
        learning_rate=0.05,
        num_steps=300,
        method="adam",
    )

    fitted, loss = minimizer(init_params, y, x, y_err, constraints)
    assert jnp.isclose(fitted[0, 0], 2.0, atol=0.2)
    assert jnp.isclose(fitted[0, 1], 1.0, atol=0.2)


def test_minimizer_lbfgs_fit():
    # linear model y = -1.5x + 0.2
    x = jnp.linspace(-5, 5, 50)
    true_params = jnp.array([-1.5, 0.2])
    y = linear_model(x, true_params)
    y_err = jnp.ones_like(y) * 0.2
    init_params = jnp.array([1.0, 0.0])

    x = x[None, :]
    y = y[None, :]
    y_err = y_err[None, :]
    init_params = init_params[None, :]

    constraints = jnp.array([[-5, 5], [-5, 5]])

    minimizer = Minimizer(
        func=linear_model,
        learning_rate=0.1,
        num_steps=100,
        method="lbfgs",
        lbfgs_options={"maxiter": 30, "tolerance_grad": 1e-6}
    )

    fitted, loss = minimizer(init_params, y, x, y_err, constraints)
    assert jnp.isclose(fitted[0, 0], -1.5, atol=0.1)
    assert jnp.isclose(fitted[0, 1], 0.2, atol=0.1)

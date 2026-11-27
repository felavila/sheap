"""Global NumPy/SciPy variable projection for SHEAP, with fast robust options.

Uses each profile's ``numpy_model`` and ``numpy_design_builder`` attached
by SHEAP profile factories. The first release handles independently free,
fixed, and repeated global linear indices. Other expression-style ties
require ``tie_resolver``. Tied linear parameters can depend only on nonlinear
parameters (not on profiled amplitudes), otherwise they need an affine column
map, not this prototype.

Fits weighted chi-square using warm BVLS by default. If a BVLS SVD
fails, optionally retries the entire affected spectrum with bounded log-cosh
using direct L-BFGS-B. A TRF/LSMR quadratic fallback is also available.
"""

from time import perf_counter
import numpy as np
from scipy.optimize import minimize, lsq_linear, OptimizeResult
from scipy.linalg import lstsq
from sheap.Minimizer.Utils.Nutils import _logcosh,_irls_weights,_solve_bvls_warm


def _fit_sheap_varpro_core(params_class,profile_functions,profile_params_index_list,p0, x, y, yerr, *, fit_mask=None,
    nonlinear_indices=None, scales=None,fd_step_u=1e-3, tie_resolver=None,maxiter=100,
    maxfun=300, linear_tol=1e-8, linear_solver="warm_bvls", warm_maxiter=5,
    warm_kkt_rtol=1e-10, loss="chi2", logcosh_solver="lbfgsb", logcosh_maxiter=500, logcosh_gtol=1e-7, logcosh_ftol=1e-12,
    irls_maxiter=20, irls_rtol=1e-8, logcosh_error_floor=1e-8, check_profiles=True, verbose=False, _svd_recovery="trf",):
    """Fit all SHEAP profiles together with bounded linear VarPro + L-BFGS-B.

    Parameters
    ----------
    p0: (n_parameters,) PHYSICAL parameter values. Defaults to
        sheapspectral.result.initial_params (the original SHEAP start).
    nonlinear_indices: optional subset of global *nonlinear* parameters to
        optimize; other nonlinear parameters are held fixed at p0.
        Accepts global indices or exact strings from params_class.names.
    scales: optional positive array of length n_free_nonlinear, or dict
        mapping global indices / exact parameter names to scales.
    fd_step_u: finite-difference step in *scaled* nonlinear coordinates.
    tie_resolver: optional function of one full physical vector returning a
        new full physical vector with SHEAP expression ties applied. Required
        when any ``params_class.summary[i]['tie']`` is set. It must not make
        dependent values depend on the profiled linear coefficients.
    check_profiles: check NumPy model = offset + design @ local amps at p0.
    linear_solver: "warm_bvls" (default) or "bvls" (reference). The warm
        version starts with a full BVLS solve, then attempts reduced
        active-set solves, and falls back to BVLS when KKT checks fail.

    loss: "chi2" (default, fastest) or "logcosh" (SHEAP objective).
    logcosh_solver: "lbfgsb" (direct bounded optimization, no SVD) or "irls".
    logcosh_maxiter, logcosh_gtol, logcosh_ftol: inner L-BFGS-B settings.
    The linear log-cosh problem is convex; inner convergence should be checked.
    irls_maxiter: maximum bounded IRLS updates per nonlinear objective evaluation.
    irls_rtol: relative convergence tolerance of the inner log-cosh loss.
    logcosh_error_floor: clip positive uncertainties from below for log-cosh,
        matching the usual SHEAP normalized-residual convention.

    Returns both chi2 and logcosh_loss, regardless of the selected objective.
    When loss="logcosh", fit.fun is the *mean log-cosh*, not chi2.
    The model must be linear in parameters marked ``linear_param_indices``.
    """
    params_class = params_class
    names = list(params_class.names)
    specs = list(params_class.summary)
    npar = len(names)

    p0 = np.asarray(p0, dtype=np.float64).squeeze()
    if p0.shape != (npar,):
        raise ValueError(f"Expected one physical vector ({npar},), got {p0.shape}")
    if not np.all(np.isfinite(p0)):
        raise ValueError("p0 contains nonfinite values")

    # Keep x, y, yerr in their input form. NumPy float64 conversion occurs
    # only for matrix algebra inside the linear solver / profile functions.
    err = yerr
    if not (x.ndim == y.ndim == err.ndim == 1):
        raise ValueError("Expected 1D x, y, yerr; select a single spectrum first")
    if not (x.shape == y.shape == err.shape):
        raise ValueError("x, y, yerr need the same 1D shape")
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(err) & (err > 0)
    if fit_mask is not None:
        mask = np.asarray(fit_mask, dtype=bool).ravel()
        if mask.shape != x.shape:
            raise ValueError("fit_mask shape mismatch")
        valid &= mask
    if not valid.any():
        raise ValueError("No usable fitting pixels")
    # Build/fit only on the unmasked wavelengths; evaluate the full model
    # once at the end for plotting and spectral products.
    x_fit = x[valid]
    # SciPy works on masked double-precision arrays; the input x/y/yerr
    # objects themselves are left untouched.
    err_v = np.array(err[valid], dtype=np.float64, copy=True)
    y_v = np.array(y[valid], dtype=np.float64, copy=True)

    def has_tie(spec):
        val = spec.get("tie", None)
        return val is not None and not (isinstance(val, (str, bool)) and (val == "" or val is False))

    tied_indices = np.asarray([i for i, s in enumerate(specs) if has_tie(s)], dtype=int)
    if len(tied_indices) and tie_resolver is None:
        example = ", ".join(f"{names[i]} ({specs[i]['tie']})" for i in tied_indices[:5])
        raise ValueError(
            "SHEAP contains expression-tied parameters. Pass a tie_resolver "
            "that reproduces the SHEAP physical-space ties, or first test "
            f"an untied configuration. Examples: {example}"
        )

    def apply_ties(p):
        if tie_resolver is None:
            return p
        p = np.asarray(tie_resolver(p.copy()), dtype=np.float64).ravel()
        if p.shape != (npar,) or not np.all(np.isfinite(p)):
            raise ValueError("tie_resolver must return one finite full physical parameter vector")
        return p

    profiles = list(profile_functions)
    idx_list = list(profile_params_index_list)
    if len(profiles) != len(idx_list):
        raise ValueError("Profile and parameter-index list lengths differ")

    # In each profile, the design columns must be in exactly the order of
    # linear_param_indices. Keep a global map; indices may occur in >1 profile.
    profile_data = []
    linear_ids = set()
    nonlinear_ids = set()
    for ip, (profile, index) in enumerate(zip(profiles, idx_list)):
        index = np.asarray(index, dtype=int).ravel()
        if np.any(index < 0) or np.any(index >= npar):
            raise ValueError(f"Profile {ip} has invalid global indices")
        if not callable(getattr(profile, "numpy_model", None)) or not callable(
            getattr(profile, "numpy_design_builder", None)
        ):
            raise ValueError(f"Profile {ip} ({getattr(profile, 'profile_name', 'unknown')}) "
                             "is missing numpy_model or numpy_design_builder")
        linear_local = np.asarray(getattr(profile, "linear_param_indices", []), dtype=int).ravel()
        if len(np.unique(linear_local)) != len(linear_local):
            raise ValueError(f"Profile {ip} has duplicate linear_param_indices")
        if np.any(linear_local < 0) or np.any(linear_local >= len(index)):
            raise ValueError(f"Profile {ip} has invalid linear_param_indices")
        islin = np.zeros(len(index), dtype=bool)
        islin[linear_local] = True
        linear_ids.update(int(i) for i in index[islin])
        nonlinear_ids.update(int(i) for i in index[~islin])
        profile_data.append((profile, index, linear_local))

    ambiguous = linear_ids & nonlinear_ids
    if ambiguous:
        raise ValueError("These global parameters are linear in one profile and "
                         f"nonlinear in another: {[names[i] for i in sorted(ambiguous)]}")
    # Handle fixed linear parameters as offset. Tied linear amplitudes are
    # also treated as offset, and must not depend on independent linear weights.
    lin_free = np.array(sorted(
        i for i in linear_ids
        if not specs[i].get("fixed", False)
        and not has_tie(specs[i])
        and float(specs[i]["min"]) < float(specs[i]["max"])
    ), dtype=int)
    lin_lookup = {int(k): j for j, k in enumerate(lin_free)}
    lo = np.array([float(specs[i]["min"]) for i in lin_free], dtype=float)
    hi = np.array([float(specs[i]["max"]) for i in lin_free], dtype=float)
    if np.any(lo >= hi):
        raise ValueError("Invalid linear bounds")

    nlin_ids = sorted(i for i in nonlinear_ids
                      if not specs[i].get("fixed", False)
                      and not has_tie(specs[i])
                      and float(specs[i]["min"]) < float(specs[i]["max"]))
    if nonlinear_indices is not None:
        requested = []
        for arg in nonlinear_indices:
            if isinstance(arg, str):
                if arg not in names:
                    raise KeyError(f"Nonlinear parameter {arg!r} not found")
                requested.append(names.index(arg))
            else:
                requested.append(int(arg))
        if len(set(requested)) != len(requested):
            raise ValueError("Repeated nonlinear_indices")
        bad = set(requested) - set(nlin_ids)
        if bad:
            raise ValueError(f"Selected parameters are not free nonlinear parameters: "
                             f"{[(i, names[i]) for i in sorted(bad)]}")
        outer = np.array(requested, dtype=int)
    else:
        outer = np.array(nlin_ids, dtype=int)

    # Initialize fixed equal-bound parameters at their actual bound value.
    p0 = p0.copy()
    for i in range(npar):
        a, b = float(specs[i]["min"]), float(specs[i]["max"])
        if np.isfinite(a) and a == b and not has_tie(specs[i]):
            p0[i] = a
    p0 = apply_ties(p0)

    theta0 = p0[outer].copy()
    outer_lo = np.array([float(specs[i]["min"]) for i in outer], dtype=float)
    outer_hi = np.array([float(specs[i]["max"]) for i in outer], dtype=float)
    if np.any(theta0 < outer_lo - 1e-8) or np.any(theta0 > outer_hi + 1e-8):
        raise ValueError("Starting nonlinear parameters lie outside their physical bounds")
    theta0 = np.clip(theta0, outer_lo, outer_hi)

    def default_scale(i):
        name = str(names[i]).lower()
        if "vshift" in name or "velocity" in name or name.endswith("_kms"):
            return 200.0
        if "logfwhm" in name or "logamp" in name or "alpha" in name:
            return 0.2
        if "x_break" in name or "x_cut" in name:
            return 500.0
        if "raw" in name or "eta" in name or "beta" in name:
            return 1.0
        a, b = float(specs[i]["min"]), float(specs[i]["max"])
        if np.isfinite(a) and np.isfinite(b):
            return max((b - a) / 10.0, 1e-3)
        return max(abs(p0[i]) * 0.1, 1.0)

    if scales is None:
        scale = np.array([default_scale(i) for i in outer], dtype=float)
    elif isinstance(scales, dict):
        scale = np.array([scales.get(i, scales.get(names[i], default_scale(i)))
                          for i in outer], dtype=float)
    else:
        scale = np.asarray(scales, dtype=float).ravel()
    if scale.shape != (len(outer),) or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError(f"scales must contain {len(outer)} positive finite entries")
    if not (np.isfinite(fd_step_u) and fd_step_u > 0):
        raise ValueError("fd_step_u must be positive and finite")

    u_bounds = [(None if not np.isfinite(a) else (a - t)/s,
                 None if not np.isfinite(b) else (b - t)/s)
                for a, b, t, s in zip(outer_lo, outer_hi, theta0, scale)]

    # For a partial finite-difference derivative, only profiles which use
    # this parameter need to be evaluated. Ties can alter any profile, so
    # conservatively evaluate all if a tie_resolver is provided.
    affected = {}
    for i in outer:
        affected[int(i)] = [k for k, (_, idx, _) in enumerate(profile_data)
                            if i in idx]
    if tie_resolver is not None:
        affected = {int(i): list(range(len(profiles))) for i in outer}

    def vector_1d(value, label, grid):
        value = np.asarray(value, dtype=np.float64).ravel()
        if value.shape != grid.shape:
            raise ValueError(f"{label} returned shape {value.shape} not {grid.shape}")
        return value

    def model_sum(p, selected=None, grid=None):
        if selected is None:
            selected = range(len(profile_data))
        if grid is None:
            grid = x_fit
        out = np.zeros(grid.shape, dtype=np.float64)
        for k in selected:
            profile, idx, _ = profile_data[k]
            out += vector_1d(profile.numpy_model(grid, p[idx]), f"profile {k} model", grid)
        return out

    def build_matrix(p, grid=None):
        if grid is None:
            grid = x_fit
        b_total = np.zeros(grid.shape, dtype=np.float64)
        A_all = np.zeros((len(grid), len(lin_free)), dtype=np.float64)
        for k, (profile, idx, linear_local) in enumerate(profile_data):
            b, A = profile.numpy_design_builder(grid, p[idx])
            b = vector_1d(b, f"profile {k} offset", grid)
            A = np.asarray(A, dtype=np.float64)
            expected = (len(grid), len(linear_local))
            if A.shape != expected:
                raise ValueError(f"profile {k} {getattr(profile, 'profile_name', '')} "
                                 f"design shape {A.shape}; expected {expected}")
            b_total += b
            for j, loc in enumerate(linear_local):
                gid = int(idx[loc])
                dest = lin_lookup.get(gid)
                if dest is None:
                    # Fixed/tied linear parameter: current physical value
                    # goes into the offset, not into the free linear solve.
                    b_total += A[:, j] * p[gid]
                else:
                    A_all[:, dest] += A[:, j]
        if not np.all(np.isfinite(b_total)) or not np.all(np.isfinite(A_all)):
            raise FloatingPointError("Global NumPy offset/design contains nonfinite fitting pixels")
        return b_total, A_all

    if check_profiles:
        # Validate each local profile before using its design columns.
        for k, (profile, idx, linear_local) in enumerate(profile_data):
            b, A = profile.numpy_design_builder(x_fit, p0[idx])
            b = vector_1d(b, f"profile {k} offset", x_fit)
            A = np.asarray(A, dtype=np.float64)
            actual = vector_1d(profile.numpy_model(x_fit, p0[idx]), f"profile {k} model", x_fit)
            predicted = b + A @ p0[idx[linear_local]]
            # x can contain pixels where the profile has been truncated;
            # compare on fitting pixels, with a tolerance accommodating
            # the finite precision of intermediate template arrays.
            difference = np.max(np.abs(actual - predicted))
            denom = max(np.max(np.abs(actual)), 1.0)
            if not np.isfinite(difference) or difference / denom > 1e-7:
                raise ValueError(
                    f"Profile {k} ({getattr(profile, 'profile_name', '')}) "
                    "does not reconstruct from its NumPy design builder. "
                    f"max abs difference={difference:.3e}, normalized={difference/denom:.3e}"
                )

    if linear_solver not in ("warm_bvls", "bvls"):
        raise ValueError("linear_solver must be warm_bvls or bvls")
    if loss not in ("logcosh", "chi2"):
        raise ValueError('loss must be "logcosh" or "chi2"')
    if logcosh_solver not in ("lbfgsb", "irls"):
        raise ValueError('logcosh_solver must be "lbfgsb" or "irls"')
    if not isinstance(logcosh_maxiter, (int, np.integer)) or logcosh_maxiter < 1:
        raise ValueError("logcosh_maxiter must be a positive integer")
    if not (np.isfinite(logcosh_gtol) and logcosh_gtol > 0
            and np.isfinite(logcosh_ftol) and logcosh_ftol > 0):
        raise ValueError("logcosh_gtol and logcosh_ftol must be positive")
    if not isinstance(irls_maxiter, (int, np.integer)) or irls_maxiter < 1:
        raise ValueError("irls_maxiter must be a positive integer")
    if not np.isfinite(irls_rtol) or irls_rtol <= 0:
        raise ValueError("irls_rtol must be positive and finite")
    if not np.isfinite(logcosh_error_floor) or logcosh_error_floor <= 0:
        raise ValueError("logcosh_error_floor must be positive and finite")

    err_objective = (np.maximum(err_v, logcosh_error_floor)
                     if loss == "logcosh" else err_v)

    stats = {"basis_s": 0.0, "linear_s": 0.0, "gradient_s": 0.0,
             "evaluations": 0, "gradient_evaluations": 0,
             "warm_solves": 0, "fallback_solves": 0,
             "cold_bvls_solves": 0, "fallback_lsmr_solves": 0, "active_iterations": 0,
             "irls_linear_solves": 0, "irls_total_steps": 0,
             "irls_unconverged_evaluations": 0, "irls_backtracked": 0,
             "logcosh_lbfgsb_calls": 0, "logcosh_lbfgsb_iterations": 0,
             "logcosh_unconverged_evaluations": 0}
    previous_linear_weights = None

    def bounded_solve(weighted_matrix, weighted_rhs, initial):
        """Reuse the existing bounded BVLS/robust fallback, and track solves."""
        if linear_solver == "warm_bvls":
            w, mode, n_active, _ = _solve_bvls_warm(
                weighted_matrix, weighted_rhs, lo, hi, initial,
                tol=linear_tol, max_active_iterations=warm_maxiter,
                kkt_rtol=warm_kkt_rtol,
                svd_recovery=_svd_recovery,
            )
            stats["active_iterations"] += n_active
            if mode == "warm":
                stats["warm_solves"] += 1
            elif mode == "fallback_bvls":
                stats["fallback_solves"] += 1
            elif mode == "cold_bvls":
                stats["cold_bvls_solves"] += 1
            elif mode == "fallback_trf_lsmr":
                stats["fallback_lsmr_solves"] += 1
            return w

        try:
            sol = lsq_linear(
                weighted_matrix, weighted_rhs, bounds=(lo, hi),
                method="bvls", lsq_solver="exact", tol=linear_tol,
            )
            if not sol.success:
                raise RuntimeError(f"BVLS failed: {sol.message}")
            stats["cold_bvls_solves"] += 1
            return sol.x
        except np.linalg.LinAlgError:
            if _svd_recovery == "raise":
                raise
            # Retain the robust numerical recovery from our previous module.
            w, mode, n_active, _ = _solve_bvls_warm(
                weighted_matrix, weighted_rhs, lo, hi, None,
                tol=linear_tol, max_active_iterations=warm_maxiter,
                kkt_rtol=warm_kkt_rtol,
                svd_recovery=_svd_recovery,
            )
            stats["active_iterations"] += n_active
            stats["fallback_lsmr_solves"] += (mode == "fallback_trf_lsmr")
            stats["cold_bvls_solves"] += (mode == "cold_bvls")
            return w

    def project_linear_gradient(gradient, weights):
        """Projected gradient for box-constrained linear parameters."""
        out = np.array(gradient, dtype=np.float64, copy=True)
        bound_atol = 1e-11
        at_lo = np.isfinite(lo) & (weights <= lo + bound_atol)
        at_hi = np.isfinite(hi) & (weights >= hi - bound_atol)
        out[at_lo & (out > 0)] = 0.0
        out[at_hi & (out < 0)] = 0.0
        return out

    def solve_logcosh_lbfgsb(Aw, rhs, starting_weights):
        """Optimize bounded log-cosh directly using matrix-vector products.

        No sequence of expensive bounded least-squares/SVD factorizations.
        Scaled variables z_j=||Aw[:,j]||/sqrt(n) * w_j improve conditioning;
        the original coefficient bounds are preserved under this transform.
        """
        nobs = Aw.shape[0]
        if not np.all(np.isfinite(Aw)) or not np.all(np.isfinite(rhs)):
            raise FloatingPointError("Nonfinite weighted design or RHS")

        col_norm = np.linalg.norm(Aw, axis=0) / np.sqrt(nobs)
        col_scale = np.where(np.isfinite(col_norm) & (col_norm > 1e-14),
                             col_norm, 1.0)
        As = Aw / col_scale[None, :]
        bounds_scaled = list(zip(lo * col_scale, hi * col_scale))

        if starting_weights is None:
            # SHEAP's provided linear initialization can be especially
            # useful when p0 contains already-fitted spectral parameters.
            w0 = np.clip(p0[lin_free], lo, hi)
        else:
            w0 = np.clip(np.asarray(starting_weights, dtype=np.float64), lo, hi)
        z0 = w0 * col_scale

        def objective_gradient(z):
            residual = As @ z - rhs
            value = float(np.mean(_logcosh(residual)))
            gradient = (As.T @ np.tanh(residual)) / nobs
            return value, gradient

        sol = minimize(
            objective_gradient, z0, jac=True, method="L-BFGS-B",
            bounds=bounds_scaled,
            options={"maxiter": int(logcosh_maxiter),
                     "gtol": float(logcosh_gtol),
                     "ftol": float(logcosh_ftol), "maxls": 30},
        )
        if not np.all(np.isfinite(sol.x)) or not np.isfinite(sol.fun):
            raise FloatingPointError("Direct log-cosh solver returned NaN/Inf")
        w = np.clip(sol.x / col_scale, lo, hi)

        # True projected gradient in the scaled coordinates, a useful
        # diagnostic especially if optimizer stops due to small ftol.
        _, g = objective_gradient(sol.x)
        pg = g.copy()
        zlo = lo * col_scale
        zhi = hi * col_scale
        pg[(sol.x <= zlo + 1e-10) & (pg > 0)] = 0
        pg[(sol.x >= zhi - 1e-10) & (pg < 0)] = 0
        kkt = float(np.max(np.abs(pg)))
        converged = bool(kkt <= max(10 * logcosh_gtol, 1e-6))
        stats["logcosh_lbfgsb_calls"] += 1
        stats["logcosh_lbfgsb_iterations"] += int(sol.nit)
        if not converged:
            stats["logcosh_unconverged_evaluations"] += 1
        return w, int(sol.nit), converged, kkt

    def solve_logcosh_irls(Aw, rhs, starting_weights):
        """Constrained IRLS for mean(log(cosh(Aw @ a - rhs))).

        This is a robust convex linear subproblem at fixed nonlinear params.
        The matrix and all linear coefficient bounds are unchanged.
        """
        if starting_weights is None:
            # Start from the bounded quadratic solution, once per spectrum.
            w = bounded_solve(Aw, rhs, None)
        else:
            w = np.clip(
                np.asarray(starting_weights, dtype=np.float64).copy(), lo, hi
            )

        r = Aw @ w - rhs
        old_loss = float(np.mean(_logcosh(r)))
        converged = False
        steps = 0
        for iteration in range(1, irls_maxiter + 1):
            # tanh(r)/r makes quadratic surrogate gradient match log-cosh.
            sqrt_weights = np.sqrt(_irls_weights(r))
            A_irls = np.asfortranarray(Aw * sqrt_weights[:, None])
            rhs_irls = rhs * sqrt_weights
            candidate = bounded_solve(A_irls, rhs_irls, w)
            stats["irls_linear_solves"] += 1
            steps = iteration

            r_new = Aw @ candidate - rhs
            new_loss = float(np.mean(_logcosh(r_new)))
            if not np.isfinite(new_loss):
                raise FloatingPointError("Nonfinite log-cosh linear objective")

            # Exact IRLS is descending. Safeguard approximate warm solves
            # against a spurious increase with a feasible segment search.
            if new_loss > old_loss + 1e-13 * max(1.0, abs(old_loss)):
                direction = candidate - w
                alpha = 0.5
                accepted = False
                for _ in range(24):
                    trial = w + alpha * direction
                    r_trial = Aw @ trial - rhs
                    trial_loss = float(np.mean(_logcosh(r_trial)))
                    if trial_loss <= old_loss:
                        candidate, r_new, new_loss = trial, r_trial, trial_loss
                        stats["irls_backtracked"] += 1
                        accepted = True
                        break
                    alpha *= 0.5
                if not accepted:
                    break

            improvement = old_loss - new_loss
            w, r, old_loss = candidate, r_new, new_loss
            # Check both objective improvement and the true, bounded KKT
            # gradient. The latter guards against premature IRLS stopping.
            g = (Aw.T @ np.tanh(r)) / r.size
            g_proj = project_linear_gradient(g, w)
            # Rescale gradients by their weighted design-column magnitude.
            col_scale = np.linalg.norm(Aw, axis=0) / np.sqrt(r.size)
            relative_kkt = float(np.max(
                np.abs(g_proj) / np.maximum(col_scale, 1e-12)
            ))
            if (improvement >= -1e-12
                    and improvement <= irls_rtol * max(abs(old_loss), 1e-6)
                    and relative_kkt < max(1e-6, np.sqrt(irls_rtol))):
                converged = True
                break

        if not converged:
            stats["irls_unconverged_evaluations"] += 1
        stats["irls_total_steps"] += steps
        return w, steps, converged

    def evaluate(u, need_gradient=True, with_details=False):
        nonlocal previous_linear_weights
        p_theta = p0.copy()
        p_theta[outer] = theta0 + scale * np.asarray(u, dtype=np.float64)
        p_theta = apply_ties(p_theta)

        t0 = perf_counter()
        b, A = build_matrix(p_theta)
        stats["basis_s"] += perf_counter() - t0

        t0 = perf_counter()
        if len(lin_free):
            # Use the same normalized residuals in the linear objective and
            # nonlinear envelope gradient. For log-cosh, clip tiny errors.
            Aw = np.asfortranarray(
                np.asarray(A / err_objective[:, None], dtype=np.float64)
            )
            rhs = np.asarray((y_v - b) / err_objective, dtype=np.float64)
            if loss == "logcosh":
                if logcosh_solver == "lbfgsb":
                    w, last_irls_steps, last_irls_converged, last_inner_kkt = (
                        solve_logcosh_lbfgsb(Aw, rhs, previous_linear_weights)
                    )
                else:
                    w, last_irls_steps, last_irls_converged = solve_logcosh_irls(
                        Aw, rhs, previous_linear_weights
                    )
                    last_inner_kkt = np.nan
            else:
                w = bounded_solve(Aw, rhs, previous_linear_weights)
                last_irls_steps = 0
                last_irls_converged = True
                last_inner_kkt = np.nan
            previous_linear_weights = w.copy()
        else:
            w = np.empty(0, dtype=float)
            last_irls_steps = 0
            last_irls_converged = True
            last_inner_kkt = np.nan
        p_best = p_theta.copy()
        p_best[lin_free] = w
        p_best = apply_ties(p_best)
        if len(tied_indices) and not np.allclose(
            p_best[tied_indices], p_theta[tied_indices], rtol=1e-9, atol=1e-10
        ):
            raise ValueError(
                "A tied parameter depends on a profiled linear amplitude. "
                "This requires converting that tie into an affine design-matrix "
                "column relation; it cannot be used as a fixed offset."
            )

        # Use a consistent model from the linear design, not the slower
        # full forward evaluation, for the objective.
        model = b + A @ w
        res = np.asarray((model - y_v) / err_objective, dtype=np.float64)
        chi2_res = np.asarray((model - y_v) / err_v, dtype=np.float64)
        chi2 = float(chi2_res @ chi2_res)
        logcosh_loss = float(np.mean(_logcosh(res)))
        objective = logcosh_loss if loss == "logcosh" else chi2
        stats["linear_s"] += perf_counter() - t0
        stats["evaluations"] += 1

        if not need_gradient:
            if with_details:
                return objective, p_best, model, b, A, w, last_irls_steps, last_irls_converged, last_inner_kkt
            return objective

        t0 = perf_counter()
        grad = np.zeros(len(outer), dtype=np.float64)
        for j, gidx in enumerate(outer):
            val = p_best[gidx]
            h0 = fd_step_u * scale[j]
            hplus = min(h0, max(outer_hi[j] - val, 0.0))
            hminus = min(h0, max(val - outer_lo[j], 0.0))
            selected = affected[int(gidx)]

            if hplus > 0 and hminus > 0:
                h = min(hplus, hminus)
                pp = p_best.copy(); pp[gidx] += h
                pm = p_best.copy(); pm[gidx] -= h
                fp = model_sum(apply_ties(pp), selected=selected)
                fm = model_sum(apply_ties(pm), selected=selected)
                deriv = (fp - fm) / (2.0*h)
            elif hplus > 0 or hminus > 0:
                f0 = model_sum(p_best, selected=selected)
                if hplus > 0:
                    pp = p_best.copy(); pp[gidx] += hplus
                    fp = model_sum(apply_ties(pp), selected=selected)
                    deriv = (fp - f0) / hplus
                else:
                    pm = p_best.copy(); pm[gidx] -= hminus
                    fm = model_sum(apply_ties(pm), selected=selected)
                    deriv = (f0 - fm) / hminus
            else:
                # Finite-difference step cannot be represented at the bound.
                # The projected-gradient optimizer can still proceed.
                deriv = np.zeros(x_fit.shape, dtype=np.float64)
            if loss == "logcosh":
                grad[j] = (np.tanh(res) @ (deriv / err_objective)) * scale[j] / res.size
            else:
                grad[j] = 2.0 * (res @ (deriv / err_v)) * scale[j]
        stats["gradient_s"] += perf_counter() - t0
        stats["gradient_evaluations"] += 1
        return objective, grad

    tic = perf_counter()
    if len(outer):
        fit = minimize(evaluate, np.zeros(len(outer), dtype=float),
                       method="L-BFGS-B", jac=True, bounds=u_bounds,
                       options={"maxiter": maxiter, "maxfun": maxfun,
                                "ftol": 1e-9, "maxls": 25})
        best_u = fit.x
    else:
        fit = OptimizeResult(x=np.empty(0, dtype=float), success=True,
                             message="No free nonlinear parameters",
                             fun=np.nan, nfev=0, nit=0)
        best_u = fit.x
    objective, p_best, model_fit, b_fit, A_fit, w, final_irls_steps, final_irls_converged, final_inner_kkt = evaluate(
        best_u, need_gradient=False, with_details=True
    )
    fit.fun = objective
    chi2 = float(np.sum(((model_fit - y_v) / err_v) ** 2))
    logcosh_loss = float(np.mean(_logcosh((model_fit - y_v) / err_objective)))
    # Full-grid models are evaluated ONLY once, for plots and science output.
    model = model_sum(p_best, grid=x)
    b, _ = build_matrix(p_best, grid=x)
    # A final consistency check at the best fit, with all NumPy forward
    # components evaluated together. This catches nonlinearity or bad ties.
    diff = np.max(np.abs(model[valid] - model_fit))
    denom = max(np.max(np.abs(model_fit)), 1.0)
    if not np.isfinite(diff) or diff / denom > 1e-7:
        raise RuntimeError("Full forward spectrum is inconsistent with the "
                           f"VarPro matrix reconstruction: {diff/denom:.3e}")
    time_s = perf_counter() - tic
    if verbose:
        print(f"Global VarPro: {len(profiles)} profiles, "
              f"{len(lin_free)} linear, {len(outer)} nonlinear; "
              f"loss={objective:.6g} ({loss}), chi2={chi2:.6g}, "
              f"nfev={fit.nfev}, time={time_s:.3f}s")
    return {
        "fit": fit, "chi2": chi2, "time_s": time_s,
        "loss": float(objective), "loss_name": loss,
        "logcosh_loss": logcosh_loss,
        "irls_final_steps": final_irls_steps if loss == "logcosh" and logcosh_solver == "irls" else 0,
        "irls_final_converged": (final_irls_converged if loss == "logcosh" and logcosh_solver == "irls" else None),
        "inner_final_converged": final_irls_converged,
        "inner_final_iterations": final_irls_steps,
        "inner_final_projected_gradient": final_inner_kkt,
        "logcosh_solver": logcosh_solver if loss == "logcosh" else None,
        "p_best": p_best, "model": model,
        "offset": b, "linear_weights": w,
        "linear_indices": lin_free, "nonlinear_indices": outer,
        "linear_names": [names[i] for i in lin_free],
        "nonlinear_names": [names[i] for i in outer],
        "n_at_lower": int(np.count_nonzero(np.isclose(w, lo, rtol=1e-7, atol=1e-8))),
        "n_at_upper": int(np.count_nonzero(np.isclose(w, hi, rtol=1e-7, atol=1e-8))),
        "n_pixels": int(valid.sum()),
        "n_profiles": len(profiles),
        "linear_solver": linear_solver,
        **stats,
    }


def fit_sheap_varpro_gradient(
    params_class,profile_functions,profile_params_index_list,p0, x, y, yerr, *,
    fit_mask=None, nonlinear_indices=None, scales=None,
    fd_step_u=1e-3, tie_resolver=None,
    maxiter=100, maxfun=300, linear_tol=1e-8,
    linear_solver="warm_bvls", warm_maxiter=5, warm_kkt_rtol=1e-10,
    loss="chi2", logcosh_solver="lbfgsb", logcosh_maxiter=500,
    logcosh_gtol=1e-7, logcosh_ftol=1e-12,
    irls_maxiter=20, irls_rtol=1e-8,
    logcosh_error_floor=1e-8,
    check_profiles=True, verbose=False,
    svd_fallback="logcosh",
):
    """Fast VarPro with per-spectrum fallback on a BVLS SVD failure.

    svd_fallback:
       'logcosh' (default): restart THIS spectrum from p0 with bounded
                 log-cosh optimization (direct L-BFGS-B, no SVD required).
       'trf':     preserve quadratic chi2, recover with bounded TRF/LSMR.
       'raise':   propagate the original LinAlgError.

    Critically, switching to log-cosh restarts the entire spectrum fitting
    under the new objective; it does not mix losses inside an optimization.
    The `fit.fun` value reports the objective actually optimized. `chi2`
    and `logcosh_loss` are always returned for comparisons across spectra.
    """
    if svd_fallback not in ("logcosh", "trf", "raise"):
        raise ValueError("svd_fallback must be 'logcosh', 'trf' or 'raise'")

    kwargs = dict(
        fit_mask=fit_mask, nonlinear_indices=nonlinear_indices,
        scales=scales, fd_step_u=fd_step_u, tie_resolver=tie_resolver,
        maxiter=maxiter, maxfun=maxfun, linear_tol=linear_tol,
        linear_solver=linear_solver, warm_maxiter=warm_maxiter,
        warm_kkt_rtol=warm_kkt_rtol, loss=loss,
        logcosh_solver=logcosh_solver, logcosh_maxiter=logcosh_maxiter,
        logcosh_gtol=logcosh_gtol, logcosh_ftol=logcosh_ftol,
        irls_maxiter=irls_maxiter, irls_rtol=irls_rtol,
        logcosh_error_floor=logcosh_error_floor,
        check_profiles=check_profiles, verbose=verbose,
    )
    tic_all = perf_counter()
    # The ordinary path is unchanged, except SVD errors can escape the
    # old BVLS internal handler when svd_fallback == 'logcosh'.
    kwargs['_svd_recovery'] = ('trf' if svd_fallback == 'trf' or loss == 'logcosh' else 'raise')

    try:
        result = _fit_sheap_varpro_core(params_class,profile_functions,profile_params_index_list,p0, x, y, yerr, **kwargs)
        result['svd_fallback_triggered'] = False
        result['svd_fallback_error'] = None
        result['svd_fallback_from_loss'] = None
        result['time_s'] = perf_counter() - tic_all
        return result
    except np.linalg.LinAlgError as exc:
        # Don't hide unrelated linear algebra problems.
        if not (
            svd_fallback == 'logcosh' and loss == 'chi2'
            and 'SVD did not converge' in str(exc)
        ):
            raise
        if verbose:
            print('SVD failed in quadratic BVLS; restarting this spectrum '
                  'with bounded log-cosh optimization.')
        error_message = f'{type(exc).__name__}: {exc}'

    # A fresh optimizer instance is necessary: L-BFGS-B must see a single,
    # consistent objective/gradient throughout the run. We reuse p0 and all
    # user-selected bounds/scales/masks. Direct log-cosh avoids BVLS entirely.
    kwargs.update(loss='logcosh', logcosh_solver='lbfgsb', _svd_recovery='trf',)

    result = _fit_sheap_varpro_core(profile_functions,profile_params_index_list,p0, x, y, yerr, **kwargs)
    result['svd_fallback_triggered'] = True
    result['svd_fallback_error'] = error_message
    result['svd_fallback_from_loss'] = 'chi2'
    result['time_s'] = perf_counter() - tic_all
    return result

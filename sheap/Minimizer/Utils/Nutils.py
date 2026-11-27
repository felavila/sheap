import numpy as np 
from scipy.optimize import minimize, lsq_linear, OptimizeResult
from scipy.linalg import lstsq


def _logcosh(r):
    """Numerically stable log(cosh(r)), even for very large residuals."""
    return np.logaddexp(r, -r) - np.log(2.0)


def _irls_weights(r):
    """IRLS weights tanh(r)/r for a log-cosh M-estimator."""
    r = np.asarray(r, dtype=np.float64)
    weights = np.empty_like(r)
    tiny = np.abs(r) < 1e-4
    rt = r[tiny] ** 2
    weights[tiny] = 1.0 - rt / 3.0 + 2.0 * rt ** 2 / 15.0
    weights[~tiny] = np.tanh(r[~tiny]) / r[~tiny]
    # Keep a tiny, positive diagonal for the bounded least-squares solver.
    return np.maximum(weights, 1e-12)

def _solve_bvls_warm(A, rhs, lower, upper, previous, *, tol=1e-8,
                     max_active_iterations=15, kkt_rtol=1e-10,
                     svd_recovery="trf"):
    """Warm-started bounded least-squares, with BVLS and TRF/LSMR fallbacks.

    The previous solution supplies the active set. This routine solves an
    equality-constrained least-squares system on the interior coefficients,
    checks the bound constraints and all KKT conditions, and releases bound
    coefficients when necessary. If it cannot certify optimality quickly,
    it falls back to scipy.optimize.lsq_linear(method='bvls').

    Returns (weights, mode, iterations, kkt_violation).

    Notes: This optimizes 1/2*||A w-rhs||^2, which has the same minimizer as
    VarPro's ||A w-rhs||^2. 'tol' is passed unchanged to SciPy's BVLS.
    """
    n = len(lower)
    if n == 0:
        return np.empty(0, dtype=np.float64), 'none', 0, 0.0

    def fallback(reason, niter=0):
        # BVLS is fastest for our problem, but NumPy's internal SVD may fail
        # for a poorly conditioned free-template subset. Use a bounded
        # iterative alternative *only when BVLS raises a numerical error*.
        if not np.all(np.isfinite(A)) or not np.all(np.isfinite(rhs)):
            raise FloatingPointError(
                "The weighted design matrix or RHS contains NaN/Inf. "
                "Inspect the spectrum, mask, error values, and model basis.")

        mode = 'cold_bvls' if previous is None else 'fallback_bvls'
        try:
            sol = lsq_linear(A, rhs, bounds=(lower, upper), method='bvls', lsq_solver='exact', tol=tol,)
            
            if sol.success and np.all(np.isfinite(sol.x)):
                return sol.x, mode, niter, float(sol.optimality)
            reason_bvls = f'BVLS returned status={sol.status}: {sol.message}'
        except np.linalg.LinAlgError as exc:
            # Let the outer dispatcher restart this spectrum with log-cosh.
            if svd_recovery == "raise":
                raise
            reason_bvls = f'{type(exc).__name__}: {exc}'
        except (ValueError, FloatingPointError) as exc:
            reason_bvls = f'{type(exc).__name__}: {exc}'

        # Column scaling helps when host SSP and emission-line basis columns
        # have very different magnitudes. For z_j = scale_j*w_j,
        # (A/scale) @ z = A @ w; bounds on z are correspondingly rescaled.
        col_scale = np.linalg.norm(A, axis=0)
        good = np.isfinite(col_scale) & (col_scale > 0)
        col_scale = np.where(good, col_scale, 1.0)
        A_scaled = A / col_scale[None, :]
        lo_scaled = lower * col_scale
        hi_scaled = upper * col_scale
        if (not np.all(np.isfinite(A_scaled))
                or np.any(np.isnan(lo_scaled)) or np.any(np.isnan(hi_scaled))):
            raise FloatingPointError(
                f'BVLS failed ({reason_bvls}); scaled problem invalid. '
                'Check error scaling and basis magnitudes.'
            )

        # TRF+LSMR does not rely on the SVD used by the BVLS fallback.
        # It still enforces the ORIGINAL box bounds after rescaling.
        try:
            robust = lsq_linear(
                A_scaled, rhs, bounds=(lo_scaled, hi_scaled),
                method='trf', lsq_solver='lsmr', tol=tol,
                lsmr_tol=1e-10, lsmr_maxiter=1000,
                max_iter=500,
            )
        except Exception as exc:
            raise RuntimeError(
                f'BVLS failed ({reason_bvls}); TRF/LSMR also failed: {exc}'
            ) from exc

        w = robust.x / col_scale
        if (not robust.success or not np.all(np.isfinite(w))
                or np.any(w < lower - 1e-8)
                or np.any(w > upper + 1e-8)):
            raise RuntimeError(
                f'BVLS failed ({reason_bvls}); bounded TRF/LSMR did not '
                f'converge: {robust.message}; optimality={robust.optimality:g}'
            )
        return np.clip(w, lower, upper), 'fallback_trf_lsmr', niter, float(robust.optimality)

    if previous is None or max_active_iterations <= 0:
        return fallback('no previous active set')

    w = np.clip(np.array(previous, dtype=np.float64, copy=True), lower, upper)
    if w.shape != (n,) or not np.all(np.isfinite(w)):
        return fallback('invalid warm solution')

    # Distinguish genuinely interior coefficients from those at a bound.
    # Use a very small tolerance, appropriate for double precision.
    bound_atol = 1e-12
    at_lo = np.isfinite(lower) & (w <= lower + bound_atol)
    at_hi = np.isfinite(upper) & (w >= upper - bound_atol)
    w[at_lo] = lower[at_lo]
    w[at_hi] = upper[at_hi]
    free = ~(at_lo | at_hi)

    # Gradient tolerance scales with the current weighted design matrix.
    # This is a screening check, not a substitute for BVLS on failure.
    rhs_gradient = A.T @ rhs
    kkt_tol = max(1e-10, kkt_rtol * max(1.0, float(np.max(np.abs(rhs_gradient)))))

    for iteration in range(1, max_active_iterations + 1):
        free_idx = np.flatnonzero(free)
        if len(free_idx):
            # Subtract the fixed-at-bound part. Compute from the current
            # feasible solution without a new large fixed-column matrix.
            target = rhs - A @ w + A[:, free_idx] @ w[free_idx]
            try:
                candidate, *_ = lstsq(
                    A[:, free_idx], target,
                    cond=None, lapack_driver='gelsy',
                    overwrite_a=False, overwrite_b=False,
                    check_finite=False,
                )
            except np.linalg.LinAlgError:
                if svd_recovery == "raise":
                    raise
                return fallback('reduced least-squares failure', iteration)
            except ValueError:
                return fallback('reduced least-squares failure', iteration)
            if not np.all(np.isfinite(candidate)):
                return fallback('nonfinite reduced solution', iteration)

            old = w[free_idx]
            direction = candidate - old
            alpha = 1.0
            hit = None
            # If the unconstrained reduced solution leaves the feasible
            # box, move only as far as the FIRST encountered bound.
            for j, global_j in enumerate(free_idx):
                if direction[j] < 0 and candidate[j] < lower[global_j]:
                    step = (lower[global_j] - old[j]) / direction[j]
                    if step < alpha:
                        alpha = max(0.0, step)
                        hit = (global_j, 'lo')
                elif direction[j] > 0 and candidate[j] > upper[global_j]:
                    step = (upper[global_j] - old[j]) / direction[j]
                    if step < alpha:
                        alpha = max(0.0, step)
                        hit = (global_j, 'hi')
            w[free_idx] = old + alpha * direction
            w = np.clip(w, lower, upper)
            if hit is not None and alpha < 1.0 - 1e-13:
                j, side = hit
                w[j] = lower[j] if side == 'lo' else upper[j]
                free[j] = False
                continue

        residual = A @ w - rhs
        gradient = A.T @ residual
        if not np.all(np.isfinite(gradient)):
            return fallback('nonfinite KKT gradient', iteration)

        at_lo = np.isfinite(lower) & (w <= lower + bound_atol) & ~free
        at_hi = np.isfinite(upper) & (w >= upper - bound_atol) & ~free
        # For 1/2||A w-rhs||^2: bound gradients must point outwards.
        violation = np.zeros(n, dtype=float)
        violation[at_lo] = np.maximum(-gradient[at_lo], 0.0)
        violation[at_hi] = np.maximum(gradient[at_hi], 0.0)
        violation[free] = np.abs(gradient[free])
        max_violation = float(np.max(violation))
        if max_violation <= kkt_tol:
            return w, 'warm', iteration, max_violation

        # If an interior gradient is bad, the reduced least-squares fit is
        # numerically suspect (rank-deficiency/ill-conditioning). Use BVLS.
        if np.any(violation[free] > kkt_tol):
            return fallback('interior gradient violation', iteration)

        j = int(np.argmax(violation))
        if free[j] or violation[j] <= kkt_tol:
            return fallback('cannot release violated bound', iteration)
        free[j] = True

    return fallback('active-set iteration budget exceeded', max_active_iterations)

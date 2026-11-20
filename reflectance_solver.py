import numpy as np
import scipy.optimize as opt

# Module implementing reflectance spectrum reconstruction from pixel values
# N refers to number of light source illuminants

def LaplacianMat(wl_range):
    """
    Make the Laplacian matrix used to discretely approximate the integral of slope squared.
    Uses the convention D = L / dx, so that the smoothness term is (1/2) rho^T D rho. 

    input:
        wl_range: 1d-array of shape (n,) of wavelength values.
    output:
        D: 2d-array of shape (n, n).
    """

    wl_range = np.asarray(wl_range)
    n = wl_range.shape[0]
    wl_init = wl_range[0]
    wl_final = wl_range[-1]
    dx = (wl_final - wl_init) / (n - 1)

    # squared difference matrix L (unscaled)
    L = np.zeros((n, n))
    L[np.arange(n), np.arange(n)] = 2
    L[np.arange(n - 1), np.arange(1, n)] = -1
    L[np.arange(1, n), np.arange(n - 1)] = -1
    L[0, 0] = L[-1, -1] = 1

    # correct scaling for integral of slope^2: D = L / dx
    D = L / dx

    return D


def mat(weight, constraint, wl_range):
    """
    Calculate the regularized matrix A.
    input:
        weight: float, hyperparameter in loss function.
        constraint: 2d-array of shape (3N, n).
        wl_range: 1d-array of shape (n,) of wavelength values.
    return:
        A: 2d-array of shape (n, n), quadratic coefficient matrix in loss function.
    """

    D = LaplacianMat(wl_range)
    C = constraint.T @ constraint                  # (n, 3N)(3N, n) -> (n, n)
    A = C + weight * D                   

    return A


def vec(pixel, constraint):
    """
    Calculate the vector coefficient in the linear term.
    input:
        pixel: 1d-array of shape (3N,), actual pixel values.
        constraint: 2d-array of shape (3N, n).
    return:
        B: 1d-array of shape (n,), vector coefficient of linear term in loss function.
    """

    pixel = np.asarray(pixel).ravel()          # (3N,)
    B = constraint.T @ pixel                   # (n, 3N)(3N,) -> (n,)

    return B


def reflectance(weight, pixel, constraint, wl_range):
    """
    Find reflectance that minimizes the regularized loss function.
    No bounds imposed here.
    input:
        weight: float.
        pixel: 1d-array of shape (3N,).
        constraint: 2d-array of shape (3N, n).
        wl_range: 1d-array of shape (n,).
    return:
        ref: 1d-array of shape (n,).
    """

    A = mat(weight, constraint, wl_range)
    B = vec(pixel, constraint)
    ref = np.linalg.solve(A, B)                # (n,)

    return ref


def in_range(ref, less_than_one=True):
    """
    Check if all components of a reflectance vector are in a desired range.
    input:
        ref: 1d-array of shape (n,) of reflectance values.
        less_than_one: bool
            True  -> check 0 <= ref <= 1 (no fluorescence)
            False -> only check ref >= 0
    return:
        bool: True if all components are in range, else False.
    """

    ref = np.asarray(ref)

    if less_than_one:
        return np.all((ref >= 0.0) & (ref <= 1.0))
    
    else:
        return np.all(ref >= 0.0)


def sum_der_squared(ref, wl_range):
    """
    Smoothness objective term: sum of (slope)^2.
    input:
        ref: 1d-array of shape (n,).
        wl_range: 1d-array of shape (n,).
    return:
        float, the sum of slope squared.
    """

    ref = np.asarray(ref).ravel()
    D = LaplacianMat(wl_range)
    smoothness = 0.5 * (ref @ D @ ref)       # factor of 1/2 is conventional

    return float(smoothness)


def sum_der_squared_Jac(ref, wl_range):
    """
    Jacobian of the sum of slope squared objective term.
    input:
        ref: 1d-array of shape (n,).
        wl_range: 1d-array of shape (n,).
    return:
        1d-array of shape (n,), the Jacobian.
    """

    ref = np.asarray(ref).ravel()
    D = LaplacianMat(wl_range)
    Jac = D @ ref   

    return Jac


def pixel_error(ref, actual_pixels, constraint_matrix):
    """
    Sum of squared pixel error term.
    input:
        ref: 1d-array of shape (n,).
        actual_pixels: 1d-array of shape (3N,), actual pixel values.
        constraint_matrix: 2d-array of shape (3N, n).
    return:
        float, sum of squared error.
    """

    ref = np.asarray(ref).ravel()
    actual_pixels = np.asarray(actual_pixels).ravel()      # (3N,)
    theor_pixels = constraint_matrix @ ref                 # (3N, n)(n,) -> (3N,)
    diff_pixels = theor_pixels - actual_pixels             # (3N,)
    y = 0.5 * (diff_pixels @ diff_pixels)                  # factor of 1/2 is conventional

    return float(y)


def pixel_error_Jac(ref, actual_pixels, constraint_matrix):
    """
    Jacobian of pixel squared error term.
    input:
        ref: 1d-array of shape (n,).
        actual_pixels: 1d-array of shape (3N,).
        constraint_matrix: 2d-array of shape (3N, n).
    return:
        1d-array of shape (n,), the Jacobian.
    """

    ref = np.asarray(ref).ravel()
    actual_pixels = np.asarray(actual_pixels).ravel()          # (3N,)
    theor_pixels = constraint_matrix @ ref                     # (3N,)
    diff_pixels = theor_pixels - actual_pixels                 # (3N,)
    Jac = constraint_matrix.T @ diff_pixels                    # (n,)

    return Jac


def obj_func(ref, actual_pixels, constraint_matrix, weight, wl_range):
    """
    Objective function: pixel_error + weight * smoothness.
    input:
        ref: 1d-array of shape (n,).
        actual_pixels: 1d-array of shape (3N,).
        constraint_matrix: 2d-array of shape (3N, n).
        weight: float.
        wl_range: 1d-array of shape (n,).
    return:
        float, value of the objective function.
    """

    pix_err = pixel_error(ref, actual_pixels, constraint_matrix)
    smoothness = sum_der_squared(ref, wl_range)
    loss = float(pix_err + weight * smoothness)

    return loss


def obj_func_Jac(ref, actual_pixels, constraint_matrix, weight, wl_range):
    """
    Jacobian of the objective function.
    input:
        ref: 1d-array of shape (n,).
        actual_pixels: 1d-array of shape (3N,).
        constraint_matrix: 2d-array of shape (3N, n).
        weight: float.
        wl_range: 1d-array of shape (n,).
    return:
        1d-array of shape (n,), the Jacobian.
    """

    Jac_pix_err = pixel_error_Jac(ref, actual_pixels, constraint_matrix)
    Jac_smoothness = sum_der_squared_Jac(ref, wl_range)
    Jac_loss = Jac_pix_err + weight * Jac_smoothness

    return Jac_loss


def _solve_bounded(weight, pixel, constraint, wl_range, less_than_one=True):
    """
    Bounded minimization of the loss function, used by solve() and solve_batch() when the closed-form
    solution is out of range: start at 0.5, L-BFGS-B, bounds [0, 1] (or [0, inf)).
    """
    n = wl_range.shape[0]
    x0 = 0.5 * np.ones(n)
    lower = np.zeros(n)
    upper = np.ones(n) if less_than_one else np.full(n, np.inf)
    bound = opt.Bounds(lb=lower, ub=upper, keep_feasible=False)

    sol = opt.minimize(
        obj_func,
        x0,
        args=(np.asarray(pixel).ravel(), constraint, weight, wl_range),
        method='L-BFGS-B',
        jac=obj_func_Jac,
        bounds=bound
    )

    return sol.x


def solve(weight, pixel, constraint, wl_range, less_than_one=True):
    """
    Find reflectance that minimizes loss function.
    First use analytic solution; if it's out of bounds, use SciPy with box constraints.
    input:
        weight: float.
        pixel: 1d-array of shape (3N,).
        constraint: 2d-array of shape (3N, n).
        wl_range: 1d-array of shape (n,).
        less_than_one: bool, if True enforce 0 <= ref <= 1, else ref >= 0.
    return:
        ref: 1d-array of shape (n,).
    """

    pixel = np.asarray(pixel).ravel()
    ref = reflectance(weight, pixel, constraint, wl_range)

    if in_range(ref, less_than_one):
        return ref

    return _solve_bounded(weight, pixel, constraint, wl_range, less_than_one)


def recon_pixel_errors(ref_recon, pixel, constraint):
    """
    calculate pixel error between reconstructed reflectance spectra and actual pixel values.
    M is the number of spectra to compare.
    input:
        ref_recon: nd-array of shape (M, n), reconstructed reflectance spectra.
        pixel: nd-array of shape (M, 3N), actual pixel values.
        constraint: 2d-array of shape (3N, n).
    return:
        pixel_error: (M, 3N), pixel errors for all M spectra.
        RMS_errors: (M,), root mean square errors for all M spectra.
    """

    pixel_errors = np.zeros((pixel.shape))     # (M, 3N)
    theor_pixels = constraint @ ref_recon.T    # (3N, n)(n, M) -> (3N, M)
    theor_pixels = theor_pixels.T              # (M, 3N)

    pixel_errors = theor_pixels - pixel         # (M, 3N)
    RMS_errors = np.sqrt(np.mean(pixel_errors ** 2, axis=1))  # (M,)

    return pixel_errors, RMS_errors


def reflectance_RMS_error(ref_recon, ref_true):
    """
    calculate root mean square error between reconstructed and true reflectance spectra.
    only possible when the true reflectance is known (e.g., in simulations).
    M is the number of spectra to compare.
    input:
        ref_recon: nd-array of shape (M, n), reconstructed reflectance spectra.
        ref_true: nd-array of shape (M, n), true reflectance spectra.
    return:
        rms_error: (M,), root mean square errors for all M spectra.
    """

    diff = ref_recon - ref_true
    mse = np.mean(diff ** 2, axis=1)
    rms_error = np.sqrt(mse)

    return rms_error


# Optional methods for choosing an "optimal" weight hyperparameter

# Method 1: Let w = tan(theta), minimize || d_ref / d_theta ||^2
def der_ref_w(weight, pixel, constraint, wl_range):
    """
    calculate derivative of the reflectance w.r.t. weight.
    input:
        weight: float.
        pixel: 1d-array of shape (3N,).
        constraint: 2d-array of shape (3N, n).
        wl_range: 1d-array of shape (n,).
    return:
        d_ref: 1d-array of shape (n,), derivative of reflectance w.r.t. weight.
    note:
        d_ref = -A^-1 D ref is the exact derivative of the closed-form (unbounded) solution, Eq. (6).
        If that solution lies outside [0, 1], solve() returns the bounded L-BFGS-B solution instead, and
        the formula is then only an approximation. In the paper this criterion is used only for the
        ColorChecker patch of Figure 3a, whose closed-form solutions stay within [0, 1].
    """

    ref = solve(weight, pixel, constraint, wl_range)
    D = LaplacianMat(wl_range)
    A = mat(weight, constraint, wl_range)
    Ainv = np.linalg.inv(A)
    d_ref = -Ainv @ (D @ ref)   # (n, n)(n,) -> (n,)

    return d_ref


def der_ref_theta2(weight, pixel, constraint, wl_range):
    """
    objective function for optimizing weight using theta = arctan(w) as hyperparameter.
    input:
        weight: float.
        pixel: 1d-array of shape (3N,).
        constraint: 2d-array of shape (3N, n).
        wl_range: 1d-array of shape (n,).
    return:
        float, objective value for weight optimization.
    """

    d_ref = der_ref_w(weight, pixel, constraint, wl_range)
    d2_ref = np.sum(d_ref ** 2)
    d2_ref_theta = d2_ref * (1 + weight ** 2) ** 2

    return float(d2_ref_theta)


def optimal_weight(pixel, constraint, wl_range):
    """
    calculate optimal weight using arctan(w) as hyperparameter.
    input:
        pixel: 1d-array of shape (3N,), actual pixel values.
        constraint: 2d-array of shape (3N, n), product of illuminant spectrum and sensor response.
        wl_range: 1d-array of shape (n,).
    return:
        w_opt: float, optimal weight.
    """

    pixel = np.asarray(pixel).ravel()
    wl_range = np.asarray(wl_range)
    n = wl_range.shape[0]
    wl_init = wl_range[0]
    wl_final = wl_range[-1]
    dx = (wl_final - wl_init) / (n - 1)

    w0 = dx**2                 # found empirically to be a reasonable starting point
    bracket = (w0, 2.0 * w0)   

    res = opt.minimize_scalar(
        der_ref_theta2,
        bracket=bracket,
        args=(pixel, constraint, wl_range),
        method='brent'
    )

    if res.success:
        w_opt = res.x

    else:
        w_opt = w0   # default to the initial guess if optimization fails
 
    return w_opt


# Method 2: Minimize RMS error when true reflectance is known (e.g., in simulations)
def optimal_weight_rms_auto(pixel, constraint, wl_range, ref_true,log10_bounds=(-8.0, 4.0), n_grid=25):
    """
    find an (approximate) globally optimal weight each time without pre-knowing a good bracket.

    1. coarse log10(w) scan over [log10_bounds[0], log10_bounds[1]]
    2. pick best grid point (lowest RMS error)
    3. bracket locally around that point
    4. run a local 1D minimization inside that bracket

    input:
        pixel: 1d-array of shape (3N,), actual pixel values.
        constraint: 2d-array of shape (3N, n), product of illuminant spectrum and sensor response.
        wl_range: 1d-array of shape (n,).
        ref_true: 1d-array of shape (n,), true reflectance spectrum.
        log10_bounds: (float, float), optional, global search range for log10(weight).
        n_grid: int, optional, number of coarse grid points in the global search.
    return:
        w_opt: float, estimated optimal weight.
    """

    pixel = np.asarray(pixel).ravel()
    ref_true = np.asarray(ref_true).ravel()
    wl_range = np.asarray(wl_range)

    def rms_given_w(w):
        try:
            ref_recon = solve(w, pixel, constraint, wl_range, less_than_one=True)
            err = reflectance_RMS_error(ref_recon, ref_true)
        except Exception:
            return np.inf
        if not np.isfinite(err):
            return np.inf
        return err

    # 1. coarse global search in log10(w) 
    log10_grid = np.linspace(log10_bounds[0], log10_bounds[1], n_grid)
    w_grid = 10.0**log10_grid
    errs = np.empty_like(w_grid)

    for i, w in enumerate(w_grid):
        errs[i] = rms_given_w(w)

    # filter out infinities (where solve blew up)
    finite_mask = np.isfinite(errs)
    if not np.any(finite_mask):
        # If everything failed, return something small
        return 1e-3

    w_grid_f = w_grid[finite_mask]
    errs_f = errs[finite_mask]
    log10_grid_f = np.log10(w_grid_f)

    # index of best coarse point
    i_best = np.argmin(errs_f)
    log10_w_best = log10_grid_f[i_best]

    # 2. define a local bracket around the best point in log space
    # take neighbors in the coarse grid if they exist, otherwise expand a bit
    if i_best == 0:
        left = log10_w_best - (log10_bounds[1] - log10_bounds[0]) / n_grid
    else:
        left = log10_grid_f[i_best - 1]

    if i_best == len(log10_grid_f) - 1:
        right = log10_w_best + (log10_bounds[1] - log10_bounds[0]) / n_grid
    else:
        right = log10_grid_f[i_best + 1]

    # ensure left < right
    left, right = min(left, right), max(left, right)

    # optionally clamp to the global bounds
    left = max(left, log10_bounds[0])
    right = min(right, log10_bounds[1])

    # 3. local refinement around best coarse point
    def objective_log10(log10_w):
        w = 10.0**log10_w
        return rms_given_w(w)

    # 4. run local 1D minimization inside bracket
    res = opt.minimize_scalar(
        objective_log10,
        bounds=(left, right),
        method='bounded'
    )

    if res.success and np.isfinite(res.fun):
        return float(10.0**res.x)
    else:
        # fall back to best coarse grid point
        return float(10.0**log10_w_best)


# ---------------------------------------------------------------------------
# Vectorized solver for many pixels at once
#
# solve_batch() returns the same result as calling solve() on every pixel, but
# computes the closed-form solution (Equation (6) of the paper) for all pixels
# with one batched linear solve. Only pixels whose closed-form solution falls
# outside the bounds go through the bounded L-BFGS-B minimization (_solve_bounded),
# exactly as in solve(); these can optionally be spread over several processes.
# ---------------------------------------------------------------------------

def _solve_bounded_task(args):
    """Picklable wrapper for multiprocessing."""
    return _solve_bounded(*args)


def solve_batch(weight, pixels, constraint, wl_range, less_than_one=True, n_jobs=1, executor=None):
    """
    Reconstruct many reflectance spectra at once; equivalent to solve() per pixel.
    input:
        weight: float.
        pixels: 2d-array of shape (M, 3N), one row of pixel values per spectrum.
        constraint: 2d-array of shape (3N, n) shared by all pixels, or
                    3d-array of shape (M, 3N, n) with one constraint matrix per pixel
                    (e.g. when the illumination varies across the image).
        wl_range: 1d-array of shape (n,).
        less_than_one: bool, if True enforce 0 <= ref <= 1, else ref >= 0.
        n_jobs: int, number of processes for the bounded fallback (1 = no multiprocessing).
                With n_jobs > 1 on Windows, call this from inside `if __name__ == "__main__":`.
        executor: optional concurrent.futures.ProcessPoolExecutor to reuse across calls
                  (overrides n_jobs); avoids restarting worker processes on every call.
    return:
        refs: 2d-array of shape (M, n).
    """

    pixels = np.atleast_2d(np.asarray(pixels, dtype=float))
    constraint = np.asarray(constraint, dtype=float)
    wl_range = np.asarray(wl_range)
    D = LaplacianMat(wl_range)

    # closed-form solution for every pixel, Equation (6)
    if constraint.ndim == 2:
        A = constraint.T @ constraint + weight * D                    # (n, n)
        B = pixels @ constraint                                       # (M, n)
        refs = np.linalg.solve(A, B.T).T                              # (M, n)
    else:
        Ct = constraint.transpose(0, 2, 1)                                         # (M, n, 3N)
        A = Ct @ constraint + weight * D[None]                                     # (M, n, n)
        B = (Ct @ pixels[..., None])[..., 0]                                       # (M, n)
        refs = np.linalg.solve(A, B[..., None])[..., 0]                            # (M, n)

    # bounded minimization where the closed-form solution is out of range
    if less_than_one:
        ok = np.all((refs >= 0.0) & (refs <= 1.0), axis=1)
    else:
        ok = np.all(refs >= 0.0, axis=1)
    bad = np.flatnonzero(~ok)

    if bad.size:
        def C_of(i):
            return constraint if constraint.ndim == 2 else constraint[i]

        tasks = [(weight, pixels[i], C_of(i), wl_range, less_than_one) for i in bad]
        if executor is not None and bad.size > 8:
            n_workers = getattr(executor, "_max_workers", 4)
            results = list(executor.map(_solve_bounded_task, tasks,
                                        chunksize=max(1, bad.size // (8 * n_workers))))
        elif n_jobs > 1 and bad.size > 4 * n_jobs:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=n_jobs) as ex:
                results = list(ex.map(_solve_bounded_task, tasks,
                                      chunksize=max(1, bad.size // (8 * n_jobs))))
        else:
            results = [_solve_bounded_task(t) for t in tasks]
        refs[bad] = np.asarray(results)

    return refs
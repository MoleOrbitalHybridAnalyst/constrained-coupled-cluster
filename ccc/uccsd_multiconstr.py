import numpy
from scipy import optimize
from scipy.sparse import linalg as sparse_linalg

from pyscf.cc import uccsd, uccsd_lambda
from pyscf.lib import logger

from .uccsd_constr import _checked_denominators
from .uccsd_linconstr import _same_spin_t2_key


BLOCKS = ("t1a", "t1b", "t2aa", "t2ab", "t2bb")
MULTIPLIER_SOLVERS = ("linear", "nonlinear")
CONSTRAINT_TIMERS = (
    "value", "jacobian", "jvp", "vjp", "row_norms", "gram_diagonal",
)


def configure_constraint_timing(cc):
    """Zero the constraint timing counters.

    A constraint callback can dominate a constrained CC cycle -- the spin engines
    rebuild a determinant wavefunction on every call, and the nonlinear
    multiplier solver invokes them repeatedly within one amplitude update -- so
    their cost is tracked separately instead of disappearing into the amplitude
    equations.  Shared by the constrained UCCSD and UCCSDT solvers.
    """
    cc.constraint_time = dict.fromkeys(CONSTRAINT_TIMERS, 0.0)
    cc.constraint_ncall = dict.fromkeys(CONSTRAINT_TIMERS, 0)
    cc._constraint_cycle_time = dict.fromkeys(CONSTRAINT_TIMERS, 0.0)
    cc._constraint_cycle_ncall = dict.fromkeys(CONSTRAINT_TIMERS, 0)


def timed_constraint(cc, name, function, *args):
    """Run one constraint callback, accumulating its wall time and call count."""
    start = logger.perf_counter()
    result = function(*args)
    elapsed = logger.perf_counter() - start
    cc.constraint_time[name] += elapsed
    cc.constraint_ncall[name] += 1
    cc._constraint_cycle_time[name] += elapsed
    cc._constraint_cycle_ncall[name] += 1
    return result


def log_constraint_timing(cc, cycle):
    """Report one cycle's constraint cost, then reset the per-cycle counters."""
    logger.info(
        cc,
        "cycle = %d  C(t) = %d calls %.3f s  derivatives = %d calls %.3f s",
        cycle + 1,
        cc._constraint_cycle_ncall["value"],
        cc._constraint_cycle_time["value"],
        sum(cc._constraint_cycle_ncall[name]
            for name in CONSTRAINT_TIMERS if name != "value"),
        sum(cc._constraint_cycle_time[name]
            for name in CONSTRAINT_TIMERS if name != "value"),
    )
    cc._constraint_cycle_time = dict.fromkeys(CONSTRAINT_TIMERS, 0.0)
    cc._constraint_cycle_ncall = dict.fromkeys(CONSTRAINT_TIMERS, 0)


def dump_constraint_timing(cc):
    """Report the accumulated constraint cost.

    Called from ``_finalize`` rather than ``dump_flags``, which runs before the
    iterations and so could only ever report zeros.
    """
    logger.note(
        cc,
        "Constraint cost: C(t) %d calls %.3f s, derivatives %d calls %.3f s, "
        "total %.3f s",
        cc.constraint_ncall["value"], cc.constraint_time["value"],
        sum(cc.constraint_ncall[name]
            for name in CONSTRAINT_TIMERS if name != "value"),
        sum(cc.constraint_time[name]
            for name in CONSTRAINT_TIMERS if name != "value"),
        sum(cc.constraint_time.values()),
    )


def _constraint_diis_error(cc, constraint, row_norms):
    """Normalize constraint residuals without amplifying zero-row noise."""
    constraint = numpy.asarray(constraint)
    row_norms = numpy.asarray(row_norms)
    if constraint.ndim != 1 or row_norms.shape != constraint.shape:
        raise ValueError("constraint values and Jacobian row norms must match")
    if numpy.iscomplexobj(constraint) or numpy.iscomplexobj(row_norms):
        raise ValueError("constraint DIIS data must be real")
    constraint = numpy.asarray(constraint, dtype=float)
    row_norms = numpy.asarray(row_norms, dtype=float)
    if not numpy.all(numpy.isfinite(constraint)):
        raise ValueError("constraint DIIS values must be finite")
    if not numpy.all(numpy.isfinite(row_norms)) or numpy.any(row_norms < 0):
        raise ValueError(
            "constraint Jacobian row norms must be finite and nonnegative"
        )

    scale = max(1.0, float(numpy.max(row_norms, initial=0.0)))
    threshold = numpy.sqrt(numpy.finfo(float).eps) * scale
    inactive = row_norms <= threshold
    violated = inactive & (numpy.abs(constraint) > cc.constraint_tol)
    if numpy.any(violated):
        raise RuntimeError(
            "a numerically inactive constraint row is violated (max |C| = "
            "%.3e for ||J_r|| <= %.3e); no reliable amplitude update can "
            "satisfy it"
            % (float(numpy.max(numpy.abs(constraint[violated]))), threshold)
        )

    scaled = numpy.zeros_like(constraint)
    active = ~inactive
    scaled[active] = (
        cc.constraint_diis_weight * constraint[active] / row_norms[active]
    )
    if numpy.any(inactive):
        logger.debug1(
            cc,
            "Constraint DIIS ignored %d numerically inactive rows "
            "(||J_r|| <= %.3e)",
            int(numpy.count_nonzero(inactive)), threshold,
        )
    return scaled


def configure_multiplier_solver(cc, solver, max_cycle, ftol):
    solver = str(solver).lower()
    if solver not in MULTIPLIER_SOLVERS:
        raise ValueError(
            "constraint_multiplier_solver must be 'linear' or 'nonlinear'"
        )
    if isinstance(max_cycle, (bool, numpy.bool_)):
        raise ValueError("constraint_multiplier_max_cycle must be a positive integer")
    integer_cycle = int(max_cycle)
    if integer_cycle <= 0 or integer_cycle != max_cycle:
        raise ValueError("constraint_multiplier_max_cycle must be a positive integer")
    ftol = float(ftol)
    if not numpy.isfinite(ftol) or ftol <= 0:
        raise ValueError("constraint_multiplier_ftol must be positive and finite")

    cc.constraint_multiplier_solver = solver
    cc.constraint_multiplier_max_cycle = integer_cycle
    cc.constraint_multiplier_ftol = ftol
    cc.constraint_multiplier_success = None
    cc.constraint_multiplier_message = None
    cc.constraint_multiplier_nfev = None
    cc.constraint_multiplier_objective = None


DIIS_SCHEMES = ("legacy", "tangent")


def configure_diis_scheme(cc, scheme):
    """Validate and set `cc.constraint_diis_scheme`.

    "legacy" (default) is today's behavior: the DIIS error vector is the raw
    amplitude update concatenated with a row-norm-scaled constraint
    deviation.  "tangent" instead solves an extra, cheap least-squares
    projection of the raw residual onto the constraint Jacobian's row space
    each cycle (reusing the same Gram/response objects already built for the
    ordinary multiplier solve) and feeds DIIS the resulting tangent residual
    in place of the constraint segment -- no concatenation, so the DIIS error
    vector lives entirely in T-space at the same scale as what it
    extrapolates.  This never changes the amplitude update itself (the
    ordinary Newton correction is computed identically in both schemes); it
    only changes what run_diis treats as its error vector.

    "tangent" is not defined together with the nonlinear multiplier
    refinement, which minimizes the exact constraint norm directly -- a
    legacy-only concept with no meaning for the tangent-residual side
    channel.
    """
    scheme = str(scheme).lower()
    if scheme not in DIIS_SCHEMES:
        raise ValueError("constraint_diis_scheme must be 'legacy' or 'tangent'")
    if (scheme == "tangent"
            and getattr(cc, "constraint_multiplier_solver", "linear") == "nonlinear"):
        raise ValueError(
            "constraint_diis_scheme='tangent' is not yet defined together "
            "with constraint_multiplier_solver='nonlinear'"
        )
    cc.constraint_diis_scheme = scheme
    cc._constraint_tangent_residual = None


def tangent_least_squares_dense(gram, rhs_ls):
    """Solve ``gram eta_ls = rhs_ls`` for the tangent-residual DIIS scheme.

    A plain least-squares solve, not the trust-region solve in
    `damped_multipliers`: eta_ls is a projection used only to build the DIIS
    error vector, never applied as a step to the amplitudes, so this must not
    touch any of `damped_multipliers`' diagnostic side effects
    (`constraint_gram_rank`, `constraint_damping`, ...), which describe the
    real multiplier correction for the cycle.  `numpy.linalg.lstsq` handles a
    rank-deficient `gram` the same way `damped_multipliers` does (minimum-norm
    solution) without needing its own eigendecomposition.
    """
    eta_ls, _, _, _ = numpy.linalg.lstsq(gram, rhs_ls, rcond=None)
    return eta_ls


def tangent_least_squares_matrix_free(cc, gram_matvec, diagonal, rhs_ls, initial=None):
    """Matrix-free counterpart of `tangent_least_squares_dense`.

    Reuses the same Gram operator and Jacobi-style preconditioner as
    `matrix_free_multipliers`'s plain (untrusted-region) solve, but performs
    no trust-region damping and leaves no diagnostic side effects on `cc` --
    see `tangent_least_squares_dense` for why.
    """
    rhs_ls = numpy.asarray(rhs_ls, dtype=float)
    nconstraint = rhs_ls.size
    diagonal = numpy.asarray(diagonal, dtype=float)
    largest = float(numpy.max(diagonal, initial=0.0))
    threshold = max(largest, 1.0) * numpy.finfo(float).eps
    inverse = numpy.ones_like(diagonal)
    nonzero = diagonal > threshold
    inverse[nonzero] = 1.0 / diagonal[nonzero]
    preconditioner = sparse_linalg.LinearOperator(
        (nconstraint, nconstraint), matvec=lambda vector: inverse * vector,
        dtype=float,
    )
    operator = sparse_linalg.LinearOperator(
        (nconstraint, nconstraint), matvec=gram_matvec, dtype=float,
    )
    if initial is not None:
        initial = numpy.asarray(initial, dtype=float)
        if initial.shape != rhs_ls.shape or not numpy.all(numpy.isfinite(initial)):
            initial = None
    eta_ls, info = sparse_linalg.cg(
        operator, rhs_ls, x0=initial, M=preconditioner,
        rtol=cc.constraint_krylov_tol, atol=0.0,
        maxiter=cc.constraint_krylov_max_cycle,
    )
    if info != 0 or not numpy.all(numpy.isfinite(eta_ls)):
        logger.debug1(
            cc, "Tangent-residual least-squares solve did not fully converge "
            "(info = %s); using the partial iterate", info,
        )
    return eta_ls


def nonlinear_multipliers(cc, initial, response, base, evaluate):
    """Minimize the exact constraint norm in the multiplier coordinates.

    ``evaluate(values)`` returns ``(C, J)`` at selected amplitude coordinates
    ``values``.  ``response`` maps multipliers to those same coordinates.  The
    expensive unconstrained CC update and the response are fixed throughout
    this small optimization.
    """
    initial = numpy.asarray(initial, dtype=float)
    response = numpy.asarray(response, dtype=float)
    base = numpy.asarray(base, dtype=float)
    ncoordinate, nconstraint = response.shape
    if initial.shape != (nconstraint,):
        raise ValueError("initial multiplier vector has an invalid shape")
    if base.shape != (ncoordinate,):
        raise ValueError("base constraint-coordinate vector has an invalid shape")

    limit = cc.constraint_max_step
    feasibility_tolerance = max(1.0, limit) * 1e-10
    cached_multiplier = None
    cached_result = None
    best_multiplier = None
    best_objective = numpy.inf
    evaluations = 0

    def feasible(multiplier):
        if limit <= 0:
            return True
        correction = response @ multiplier
        return float(numpy.max(numpy.abs(correction))) <= (
            limit + feasibility_tolerance
        )

    def objective(multiplier):
        nonlocal cached_multiplier, cached_result
        nonlocal best_multiplier, best_objective, evaluations
        multiplier = numpy.asarray(multiplier, dtype=float)
        if (cached_multiplier is not None
                and numpy.array_equal(multiplier, cached_multiplier)):
            return cached_result

        values = base + response @ multiplier
        constraint, jacobian = evaluate(values)
        constraint = numpy.asarray(constraint, dtype=float)
        jacobian = numpy.asarray(jacobian, dtype=float)
        if constraint.shape != (nconstraint,):
            raise ValueError("trial constraint vector has an invalid shape")
        if jacobian.shape != (nconstraint, ncoordinate):
            raise ValueError("trial constraint Jacobian has an invalid shape")
        if (not numpy.all(numpy.isfinite(constraint))
                or not numpy.all(numpy.isfinite(jacobian))):
            raise ValueError("trial constraint evaluation is not finite")

        value = 0.5 * float(numpy.dot(constraint, constraint))
        gradient = response.T @ jacobian.T @ constraint
        evaluations += 1
        if feasible(multiplier) and value < best_objective:
            best_multiplier = multiplier.copy()
            best_objective = value
        cached_multiplier = multiplier.copy()
        cached_result = value, gradient
        return cached_result

    # Keep both the unconstrained candidate and the existing linear solution
    # available as guaranteed finite fallbacks.
    objective(numpy.zeros(nconstraint))
    objective(initial)

    constraints = ()
    if limit > 0:
        constraints = (optimize.LinearConstraint(response, -limit, limit),)
    result = optimize.minimize(
        objective,
        initial.copy(),
        jac=True,
        method="SLSQP",
        constraints=constraints,
        options={
            "maxiter": cc.constraint_multiplier_max_cycle,
            "ftol": cc.constraint_multiplier_ftol,
            "disp": False,
        },
    )
    objective(result.x)
    if best_multiplier is None:
        raise RuntimeError("Nonlinear multiplier solve found no finite feasible step")

    retained_linear = numpy.array_equal(best_multiplier, initial)
    message = str(result.message)
    if retained_linear:
        message += "; retained linear multiplier"
    cc.constraint_multiplier_success = bool(result.success)
    cc.constraint_multiplier_message = message
    cc.constraint_multiplier_nfev = evaluations
    cc.constraint_multiplier_objective = float(best_objective)
    if not result.success:
        logger.warn(
            cc,
            "Nonlinear multiplier optimization did not converge (%s); "
            "using the best finite feasible candidate",
            result.message,
        )
    logger.debug1(
        cc,
        "Nonlinear multiplier solve: objective = %.6e  evaluations = %d  %s",
        best_objective,
        evaluations,
        message,
    )
    return best_multiplier, response @ best_multiplier


def damped_multipliers(cc, gram, response, rhs, nconstraint):
    """Solve ``(gram + mu I) lambda = rhs`` inside a trust region.

    ``gram`` is ``J D^-1 J^T``, so its conditioning is the square of the
    constraint Jacobian's.  When the residual has a component along a nearly
    flat direction of the Jacobian, the exact Newton step is enormous and the
    linearization that produced it no longer describes the constraint.  A
    Levenberg-Marquardt shift redirects the step toward directions the
    amplitudes actually control rather than merely shortening it, so the
    residual is removed over several iterations instead of one.

    With ``cc.constraint_max_step = 0`` this is the plain Newton step and the
    multiplier matrix must be nonsingular.  Otherwise ``mu`` is the smallest
    shift keeping ``max|correction|`` within the trust radius.

    Shared by the constrained UCCSD and UCCSDT solvers; both build the same
    multiplier system and differ only in how the correction is scattered back
    into the amplitudes.
    """
    limit = cc.constraint_max_step
    if limit <= 0:
        rank = int(numpy.linalg.matrix_rank(gram))
        if rank != nconstraint:
            raise RuntimeError(
                "Constraint multiplier matrix is singular (rank %d of %d); "
                "set constraint_max_step to damp the step instead"
                % (rank, nconstraint)
            )
        multipliers = numpy.linalg.solve(gram, rhs)
        cc.constraint_damping = 0.0
        return multipliers, response @ multipliers

    # gram is symmetric positive semidefinite; clamp the rounding-level
    # negative eigenvalues that a nearly singular Gram matrix produces.
    eigenvalues, vectors = numpy.linalg.eigh(gram)
    eigenvalues = numpy.maximum(eigenvalues, 0.0)
    projected = vectors.T @ rhs

    # Damping tolerates a singular multiplier matrix, so this reports rather
    # than raises.  The rank is read off the eigenvalues already computed, using
    # numpy's matrix_rank convention, and is only announced when it changes so a
    # long run is not flooded.
    largest = float(eigenvalues[-1])
    threshold = largest * nconstraint * numpy.finfo(float).eps
    gram_rank = int(numpy.count_nonzero(eigenvalues > threshold))
    cc.constraint_gram_rank = gram_rank
    cc.constraint_gram_condition = (
        largest / eigenvalues[0] if eigenvalues[0] > 0 else numpy.inf
    )
    if gram_rank != cc._reported_gram_rank:
        cc._reported_gram_rank = gram_rank
        if gram_rank != nconstraint:
            logger.warn(
                cc,
                "Constraint multiplier matrix is rank %d of %d (cond = %.3e); "
                "trust-region damping is absorbing the deficient directions, "
                "so those constraint combinations converge only as far as the "
                "damping allows",
                gram_rank, nconstraint, cc.constraint_gram_condition,
            )
        else:
            logger.info(
                cc, "Constraint multiplier matrix is full rank (cond = %.3e)",
                cc.constraint_gram_condition,
            )

    def solve(shift):
        multipliers = vectors @ (projected / (eigenvalues + shift))
        return multipliers, response @ multipliers

    multipliers, correction = solve(0.0)
    excess = float(numpy.max(numpy.abs(correction)))
    if numpy.isfinite(excess) and excess <= limit:
        cc.constraint_damping = 0.0
        return multipliers, correction

    # Bracket by doubling, then bisect on log(mu).  Each trial costs one small
    # solve and one response matvec, negligible next to an amplitude iteration.
    high = max(largest, 1.0) * numpy.finfo(float).eps
    for _ in range(200):
        trial, trial_correction = solve(high)
        if float(numpy.max(numpy.abs(trial_correction))) <= limit:
            break
        high *= 2.0
    else:
        raise RuntimeError(
            "Constraint trust region could not be satisfied; max|dT| stayed "
            "above %.3g" % limit
        )

    low = 0.0
    for _ in range(60):
        mid = numpy.sqrt(high * low) if low > 0 else high * 0.5
        candidate, candidate_correction = solve(mid)
        if float(numpy.max(numpy.abs(candidate_correction))) <= limit:
            high, trial, trial_correction = mid, candidate, candidate_correction
        else:
            low = mid
        if high <= low * (1.0 + 1e-3):
            break

    cc.constraint_damping = float(high)
    logger.debug1(
        cc, "Constraint trust region active: mu = %.6e, max|dT| = %.6e",
        high, float(numpy.max(numpy.abs(trial_correction))),
    )
    return trial, trial_correction


def matrix_free_multipliers(cc, gram_matvec, response, rhs, diagonal, initial=None):
    """Solve the multiplier system using only Gram and response products.

    `initial`, if given, warm-starts the first (shift=0) Krylov solve from
    the previous outer CC cycle's converged multipliers -- CG converges to
    the same solution regardless of x0, so this can only reduce iteration
    count, never change the answer or the failure mode.  It is a performance
    hint only: any shape/finiteness mismatch is silently ignored rather than
    raised, since a bad hint must never be able to turn a working cold-start
    run into a crash.
    """
    rhs = numpy.asarray(rhs, dtype=float)
    diagonal = numpy.asarray(diagonal, dtype=float)
    nconstraint = rhs.size
    if diagonal.shape != (nconstraint,):
        raise ValueError("matrix-free Gram diagonal has an invalid shape")
    if not numpy.all(numpy.isfinite(diagonal)):
        raise ValueError("matrix-free Gram diagonal must be finite")
    if initial is not None:
        initial = numpy.asarray(initial, dtype=float)
        if initial.shape != rhs.shape or not numpy.all(numpy.isfinite(initial)):
            initial = None
    # A rank-deficient row (a collinear reference zeroes some spin rows
    # identically) has an exact-zero diagonal that the closed-form tensor
    # contractions deliver as rounding noise of either sign; only a negative
    # entry beyond that noise means the metric is genuinely indefinite.
    scale = max(1.0, float(numpy.max(diagonal, initial=0.0)))
    if numpy.any(diagonal < -1e3 * numpy.finfo(float).eps * scale):
        raise ValueError("matrix-free Gram diagonal must be nonnegative")
    diagonal = numpy.maximum(diagonal, 0.0)

    largest = float(numpy.max(diagonal, initial=0.0))

    def solve(shift, guess=None):
        operator = sparse_linalg.LinearOperator(
            (nconstraint, nconstraint),
            matvec=lambda vector: gram_matvec(vector) + shift * vector,
            dtype=float,
        )
        preconditioner_diagonal = diagonal + shift
        threshold = max(largest + shift, 1.0) * numpy.finfo(float).eps
        inverse = numpy.ones_like(preconditioner_diagonal)
        nonzero = preconditioner_diagonal > threshold
        inverse[nonzero] = 1.0 / preconditioner_diagonal[nonzero]
        preconditioner = sparse_linalg.LinearOperator(
            (nconstraint, nconstraint), matvec=lambda vector: inverse * vector,
            dtype=float,
        )
        iterations = 0

        def count(_):
            nonlocal iterations
            iterations += 1

        multipliers, info = sparse_linalg.cg(
            operator, rhs, x0=guess, M=preconditioner,
            rtol=cc.constraint_krylov_tol, atol=0.0,
            maxiter=cc.constraint_krylov_max_cycle, callback=count,
        )
        residual = float(numpy.linalg.norm(
            gram_matvec(multipliers) + shift * multipliers - rhs
        ))
        target = cc.constraint_krylov_tol * max(1.0, numpy.linalg.norm(rhs))
        success = info == 0 and numpy.isfinite(residual) and residual <= 10 * target
        correction = response(multipliers)
        return multipliers, correction, success, info, iterations, residual

    limit = cc.constraint_max_step
    zero = solve(0.0, guess=initial)
    multipliers, correction, success, info, iterations, residual = zero
    if success and (limit <= 0 or float(numpy.max(numpy.abs(correction))) <= limit):
        cc.constraint_damping = 0.0
        cc.constraint_krylov_info = info
        cc.constraint_krylov_iterations = iterations
        cc.constraint_krylov_residual = residual
        return multipliers, correction
    if limit <= 0:
        raise RuntimeError(
            "Matrix-free constraint multiplier solve did not converge "
            "(info = %s, residual = %.3e); set constraint_max_step to damp "
            "rank-deficient or ill-conditioned constraints" % (info, residual)
        )

    high = max(largest, 1.0) * numpy.finfo(float).eps
    trial = None
    guess = multipliers if numpy.all(numpy.isfinite(multipliers)) else None
    for _ in range(200):
        candidate = solve(high, guess=guess)
        guess = candidate[0]
        if (candidate[2]
                and float(numpy.max(numpy.abs(candidate[1]))) <= limit):
            trial = candidate
            break
        high *= 2.0
    if trial is None:
        raise RuntimeError(
            "Constraint trust region could not be satisfied; max|dT| stayed "
            "above %.3g" % limit
        )

    low = 0.0
    for _ in range(60):
        mid = numpy.sqrt(high * low) if low > 0 else high * 0.5
        candidate = solve(mid, guess=trial[0])
        if (candidate[2]
                and float(numpy.max(numpy.abs(candidate[1]))) <= limit):
            high, trial = mid, candidate
        else:
            low = mid
        if high <= low * (1.0 + 1e-3):
            break

    (multipliers, correction, _, info, iterations, residual) = trial
    cc.constraint_damping = float(high)
    cc.constraint_krylov_info = info
    cc.constraint_krylov_iterations = iterations
    cc.constraint_krylov_residual = residual
    logger.debug1(
        cc, "Matrix-free trust region: mu = %.6e, max|dT| = %.6e",
        high, float(numpy.max(numpy.abs(correction))),
    )
    return multipliers, correction


def nonlinear_multipliers_matrix_free(
        cc, initial, response, response_transpose, base, evaluate):
    """Gauss-Newton refinement of the exact constraint norm using JVP/VJP."""
    initial = numpy.asarray(initial, dtype=float)
    base = numpy.asarray(base, dtype=float)
    nconstraint = initial.size
    limit = cc.constraint_max_step
    tolerance = max(1.0, limit) * 1e-10
    evaluations = 0

    def assess(multiplier):
        nonlocal evaluations
        correction = response(multiplier)
        if (limit > 0
                and float(numpy.max(numpy.abs(correction))) > limit + tolerance):
            return None
        constraint, jvp, vjp = evaluate(base + correction)
        constraint = numpy.asarray(constraint, dtype=float)
        if constraint.shape != (nconstraint,) or not numpy.all(numpy.isfinite(constraint)):
            raise ValueError("trial constraint vector has an invalid value")
        evaluations += 1
        return (0.5 * float(numpy.dot(constraint, constraint)),
                constraint, jvp, vjp, correction)

    current = initial.copy()
    state = assess(current)
    zero_state = assess(numpy.zeros(nconstraint))
    if state is None:
        state = zero_state
        current = numpy.zeros(nconstraint)
    best, best_state = current.copy(), state
    if zero_state is not None and zero_state[0] < best_state[0]:
        best, best_state = numpy.zeros(nconstraint), zero_state

    success = False
    message = "maximum cycles reached"
    for _ in range(cc.constraint_multiplier_max_cycle):
        objective, constraint, trial_jvp, trial_vjp, _ = state
        if objective <= cc.constraint_multiplier_ftol:
            success = True
            message = "constraint objective converged"
            break

        operator = sparse_linalg.LinearOperator(
            (nconstraint, nconstraint),
            matvec=lambda vector: trial_jvp(response(vector)),
            rmatvec=lambda weights: response_transpose(trial_vjp(weights)),
            dtype=float,
        )
        step = sparse_linalg.lsmr(
            operator, -constraint,
            atol=cc.constraint_multiplier_ftol,
            btol=cc.constraint_multiplier_ftol,
            maxiter=cc.constraint_krylov_max_cycle,
        )[0]
        if not numpy.all(numpy.isfinite(step)):
            message = "non-finite Gauss-Newton step"
            break

        accepted = False
        alpha = 1.0
        for _ in range(60):
            candidate = current + alpha * step
            candidate_state = assess(candidate)
            if candidate_state is not None and candidate_state[0] < objective:
                current, state = candidate, candidate_state
                accepted = True
                if state[0] < best_state[0]:
                    best, best_state = current.copy(), state
                break
            alpha *= 0.5
        if not accepted:
            message = "line search found no improving feasible step"
            break
    else:
        success = False

    cc.constraint_multiplier_success = success
    cc.constraint_multiplier_message = message
    cc.constraint_multiplier_nfev = evaluations
    cc.constraint_multiplier_objective = float(best_state[0])
    if not success:
        logger.warn(
            cc, "Nonlinear multiplier optimization did not converge (%s); "
            "using the best finite feasible candidate", message,
        )
    return best, response(best)


class MultiConstrUCCSD(uccsd.UCCSD):
    """UCCSD with multiple constraints in a compact amplitude subspace.

    ``constraint(t1, t2)`` returns a one-dimensional residual vector.  The
    Jacobian callback returns derivatives with respect to the full-amplitude
    coordinates listed in ``constraint_coordinates``.  A coordinate is a
    ``(block, index)`` pair, for example ``("t2ab", (i, j, a, b))``.
    """

    def __init__(self, mf, constraint, constraint_jacobian,
                 constraint_coordinates, constraint_tol=1e-10,
                 constraint_diis_weight=1.0, constraint_max_step=0.0,
                 frozen=None, mo_coeff=None, mo_occ=None,
                 constraint_multiplier_solver="linear",
                 constraint_multiplier_max_cycle=20,
                 constraint_multiplier_ftol=1e-12,
                 constraint_hessian_vector=None,
                 constraint_jvp=None, constraint_vjp=None,
                 constraint_jacobian_row_norms=None,
                 constraint_weighted_gram_diagonal=None,
                 constraint_krylov_tol=1e-10,
                 constraint_krylov_max_cycle=None,
                 conv_check=True,
                 constraint_diis_scheme="legacy"):
        uccsd.UCCSD.__init__(
            self, mf, frozen=frozen, mo_coeff=mo_coeff, mo_occ=mo_occ
        )
        if not callable(constraint):
            raise TypeError("constraint must be callable")
        if not callable(constraint_jacobian):
            raise TypeError("constraint_jacobian must be callable")
        if constraint_tol <= 0:
            raise ValueError("constraint_tol must be positive")
        if constraint_diis_weight <= 0:
            raise ValueError("constraint_diis_weight must be positive")

        self.constraint = constraint
        self.constraint_jacobian = constraint_jacobian
        owner = getattr(constraint, "__self__", None)
        if owner is not None and owner is getattr(constraint_jacobian, "__self__", None):
            constraint_hessian_vector = (
                constraint_hessian_vector
                or getattr(owner, "hessian_vector", None)
            )
            constraint_jvp = constraint_jvp or getattr(owner, "jvp", None)
            constraint_vjp = constraint_vjp or getattr(owner, "vjp", None)
            constraint_jacobian_row_norms = (
                constraint_jacobian_row_norms
                or getattr(owner, "jacobian_row_norms", None)
            )
            constraint_weighted_gram_diagonal = (
                constraint_weighted_gram_diagonal
                or getattr(owner, "weighted_gram_diagonal", None)
            )
        if (constraint_hessian_vector is not None
                and not callable(constraint_hessian_vector)):
            raise TypeError("constraint_hessian_vector must be callable")
        products = (
            constraint_jvp, constraint_vjp,
            constraint_jacobian_row_norms,
            constraint_weighted_gram_diagonal,
        )
        if any(product is not None for product in products):
            if not all(callable(product) for product in products):
                raise TypeError(
                    "constraint JVP, VJP, row norms, and Gram diagonal must "
                    "all be callable when matrix-free constraints are enabled"
                )
            self.constraint_jvp = constraint_jvp
            self.constraint_vjp = constraint_vjp
            self.constraint_jacobian_row_norms = constraint_jacobian_row_norms
            self.constraint_weighted_gram_diagonal = (
                constraint_weighted_gram_diagonal
            )
            self._matrix_free_constraints = True
        else:
            self.constraint_jvp = None
            self.constraint_vjp = None
            self.constraint_jacobian_row_norms = None
            self.constraint_weighted_gram_diagonal = None
            self._matrix_free_constraints = False
        if getattr(constraint_coordinates, "_ccc_compact_coordinates", False):
            self.constraint_coordinates = constraint_coordinates
        else:
            self.constraint_coordinates = tuple(
                (str(block), tuple(index))
                for block, index in constraint_coordinates
            )
        if not self.constraint_coordinates:
            raise ValueError("constraint_coordinates must not be empty")

        self.constraint_tol = float(constraint_tol)
        self.constraint_diis_weight = float(constraint_diis_weight)
        # Trust radius on the max-norm of the multiplier correction to T.  Zero,
        # the default, disables damping and takes the exact Newton step; set it
        # only when the multiplier system is ill-conditioned enough that the
        # Newton step leaves the region where the linearization is meaningful.
        self.constraint_max_step = float(constraint_max_step)
        self.constraint_krylov_tol = float(constraint_krylov_tol)
        if (not numpy.isfinite(self.constraint_krylov_tol)
                or self.constraint_krylov_tol <= 0):
            raise ValueError("constraint_krylov_tol must be positive and finite")
        if constraint_krylov_max_cycle is None:
            # The Krylov system has one unknown per constraint, not one per
            # involved amplitude.  Its size is learned on the first callback.
            self.constraint_krylov_max_cycle = None
        else:
            self.constraint_krylov_max_cycle = int(
                constraint_krylov_max_cycle
            )
            if (self.constraint_krylov_max_cycle <= 0
                    or self.constraint_krylov_max_cycle
                    != constraint_krylov_max_cycle):
                raise ValueError(
                    "constraint_krylov_max_cycle must be a positive integer"
                )
        configure_multiplier_solver(
            self,
            constraint_multiplier_solver,
            constraint_multiplier_max_cycle,
            constraint_multiplier_ftol,
        )
        configure_diis_scheme(self, constraint_diis_scheme)
        configure_constraint_timing(self)
        self._constraint_cycle = 0
        self.newton_polish = False
        self.newton_options = {}
        self.newton_report = None
        self.conv_check = bool(conv_check)
        self.constraint_hessian_vector = constraint_hessian_vector
        self.constraint_kappa = None
        self.constraint_lambda_orthogonality = None
        self.constraint_value = None
        self.constraint_multipliers = None
        self.constraint_rank = None
        self.constraint_damping = 0.0
        self.constraint_gram_rank = None
        self.constraint_gram_condition = None
        self.constraint_krylov_info = None
        self.constraint_krylov_iterations = None
        self.constraint_krylov_residual = None
        self._reported_gram_rank = None
        self.constraint_linearized_residual = None
        self.constraint_diis_error = None
        self._constraint_old_vector = None
        self._coordinate_shapes = None
        self._coordinate_flat = None
        self._coordinate_sign = None
        self._constraint_size = None
        self._keys = set(self.__dict__).union({
            "constraint", "constraint_jacobian", "constraint_coordinates",
            "constraint_tol", "constraint_diis_weight", "constraint_value",
            "constraint_multipliers", "constraint_rank",
            "constraint_max_step", "constraint_damping",
            "constraint_gram_rank", "constraint_gram_condition",
            "constraint_linearized_residual", "constraint_diis_error",
            "constraint_multiplier_solver", "constraint_multiplier_max_cycle",
            "constraint_multiplier_ftol", "constraint_multiplier_success",
            "constraint_multiplier_message", "constraint_multiplier_nfev",
            "constraint_multiplier_objective",
            "constraint_time", "constraint_ncall",
            "constraint_hessian_vector", "constraint_kappa",
            "constraint_lambda_orthogonality",
            "constraint_jvp", "constraint_vjp",
            "constraint_jacobian_row_norms",
            "constraint_weighted_gram_diagonal",
            "constraint_krylov_tol", "constraint_krylov_max_cycle",
            "constraint_krylov_info", "constraint_krylov_iterations",
            "constraint_krylov_residual",
        })

    @classmethod
    def from_constraint_engine(cls, mf, engine, **kwargs):
        """Construct from a sparse constraint engine without dense Jacobians."""
        kwargs.setdefault("constraint_jvp", engine.jvp)
        kwargs.setdefault("constraint_vjp", engine.vjp)
        kwargs.setdefault(
            "constraint_jacobian_row_norms", engine.jacobian_row_norms
        )
        kwargs.setdefault(
            "constraint_weighted_gram_diagonal",
            engine.weighted_gram_diagonal,
        )
        kwargs.setdefault("constraint_hessian_vector", engine.hessian_vector)
        return cls(
            mf, engine.constraint, engine.jacobian, engine.coordinates, **kwargs
        )

    def dump_flags(self, verbose=None):
        uccsd.UCCSD.dump_flags(self, verbose=verbose)
        log = logger.new_logger(self, verbose)
        log.info("Vector constraint tolerance = %.3g", self.constraint_tol)
        log.info("Constraint DIIS weight = %.3g", self.constraint_diis_weight)
        log.info("Constraint amplitude coordinates = %d",
                 len(self.constraint_coordinates))
        log.info("Constraint trust radius max|dT| = %s",
                 "%.3g" % self.constraint_max_step
                 if self.constraint_max_step > 0 else "disabled")
        log.info("Constraint multiplier solver = %s",
                 self.constraint_multiplier_solver)
        log.info("Constraint DIIS scheme = %s", self.constraint_diis_scheme)
        log.info("Constraint derivatives = %s",
                 "matrix-free" if self._matrix_free_constraints else "dense")
        log.info(
            "Constraint Krylov max cycles = %s",
            (self.constraint_krylov_max_cycle
             if self.constraint_krylov_max_cycle is not None
             else "automatic: max(100, 2*nconstraint)"),
        )
        if self.constraint_multiplier_solver == "nonlinear":
            log.info("Nonlinear multiplier max cycles = %d",
                     self.constraint_multiplier_max_cycle)
            log.info("Nonlinear multiplier ftol = %.3g",
                     self.constraint_multiplier_ftol)
        return self

    @staticmethod
    def _vector_sizes(t1):
        nocca, nvira = t1[0].shape
        noccb, nvirb = t1[1].shape
        sizea = (nocca * nvira + nocca * (nocca - 1) // 2
                 * nvira * (nvira - 1) // 2)
        sizeb = (noccb * nvirb + noccb * (noccb - 1) // 2
                 * nvirb * (nvirb - 1) // 2)
        return sizea, sizeb

    def _coordinate_index(self, block, index, t1):
        nocca, nvira = t1[0].shape
        noccb, nvirb = t1[1].shape
        sizea, sizeb = self._vector_sizes(t1)
        if block == "t1a":
            if len(index) != 2:
                raise ValueError("t1a index must have length 2")
            return numpy.ravel_multi_index(index, (nocca, nvira)), 1.0
        if block == "t1b":
            if len(index) != 2:
                raise ValueError("t1b index must have length 2")
            flat = numpy.ravel_multi_index(index, (noccb, nvirb))
            return sizea + flat, 1.0
        if block == "t2aa":
            if len(index) != 4:
                raise ValueError("t2aa index must have length 4")
            p, q, sign = _same_spin_t2_key(
                *index, nocca, nvira, block
            )
            nvir_pairs = nvira * (nvira - 1) // 2
            return nocca * nvira + p * nvir_pairs + q, sign
        if block == "t2bb":
            if len(index) != 4:
                raise ValueError("t2bb index must have length 4")
            p, q, sign = _same_spin_t2_key(
                *index, noccb, nvirb, block
            )
            nvir_pairs = nvirb * (nvirb - 1) // 2
            flat = noccb * nvirb + p * nvir_pairs + q
            return sizea + flat, sign
        if block == "t2ab":
            if len(index) != 4:
                raise ValueError("t2ab index must have length 4")
            flat = numpy.ravel_multi_index(
                index, (nocca, noccb, nvira, nvirb)
            )
            return sizea + sizeb + flat, 1.0
        raise ValueError("Unknown amplitude block %r" % block)

    def _ensure_coordinates(self, t1, t2):
        shapes = tuple(block.shape for block in t1 + t2)
        if self._coordinate_shapes == shapes:
            return
        compact = getattr(
            self.constraint_coordinates, "packed_indices_uccsd", None
        )
        if compact is not None:
            flat, signs = compact(t1)
            self._coordinate_shapes = shapes
            self._coordinate_flat = numpy.asarray(flat, dtype=numpy.int64)
            self._coordinate_sign = numpy.asarray(signs, dtype=float)
            return
        flat = []
        signs = []
        for block, index in self.constraint_coordinates:
            if block not in BLOCKS:
                raise ValueError("Unknown amplitude block %r" % block)
            coordinate, sign = self._coordinate_index(block, index, t1)
            flat.append(coordinate)
            signs.append(sign)
        if len(set(flat)) != len(flat):
            raise ValueError("constraint_coordinates contain duplicate amplitudes")
        self._coordinate_shapes = shapes
        self._coordinate_flat = numpy.asarray(flat, dtype=numpy.int64)
        self._coordinate_sign = numpy.asarray(signs)

    @staticmethod
    def _coordinate_blocks(t1, t2):
        return dict(zip(BLOCKS, tuple(t1) + tuple(t2)))

    def _set_coordinate_values(self, t1, t2, values):
        compact = getattr(self.constraint_coordinates, "set_values", None)
        if compact is not None:
            compact(t1, t2, values)
            return
        blocks = self._coordinate_blocks(t1, t2)
        for (name, index), value in zip(self.constraint_coordinates, values):
            block = blocks[name]
            if name not in ("t2aa", "t2bb"):
                block[index] = value
                continue
            i, j, a, b = index
            block[i, j, a, b] = value
            block[j, i, a, b] = -value
            block[i, j, b, a] = -value
            block[j, i, b, a] = value

    def _constraint_value(self, t1, t2):
        value = numpy.asarray(
            timed_constraint(self, "value", self.constraint, t1, t2)
        )
        if value.ndim != 1 or value.size == 0:
            raise ValueError(
                "constraint must return a nonempty one-dimensional array, got %s"
                % (value.shape,)
            )
        if numpy.iscomplexobj(value) and numpy.any(numpy.abs(value.imag) > 1e-12):
            raise ValueError("constraint must be real")
        value = numpy.asarray(value.real, dtype=float)
        if not numpy.all(numpy.isfinite(value)):
            raise ValueError("constraint returned non-finite values")
        if self._constraint_size is None:
            self._constraint_size = value.size
            if self.constraint_krylov_max_cycle is None:
                self.constraint_krylov_max_cycle = max(
                    100, 2 * self._constraint_size
                )
        elif value.size != self._constraint_size:
            raise ValueError("constraint size changed between calls")
        return value

    def _constraint_jacobian(self, t1, t2, require_full_rank=True):
        self._ensure_coordinates(t1, t2)
        value = self._constraint_value(t1, t2)
        jacobian = numpy.asarray(
            timed_constraint(self, "jacobian", self.constraint_jacobian, t1, t2)
        )
        expected = (value.size, len(self.constraint_coordinates))
        if jacobian.shape != expected:
            raise ValueError(
                "constraint_jacobian has shape %s; expected %s"
                % (jacobian.shape, expected)
            )
        if (numpy.iscomplexobj(jacobian)
                and numpy.any(numpy.abs(jacobian.imag) > 1e-12)):
            raise ValueError("constraint_jacobian must be real")
        jacobian = numpy.asarray(jacobian.real, dtype=float)
        if not numpy.all(numpy.isfinite(jacobian)):
            raise ValueError("constraint_jacobian contains non-finite values")

        packed = jacobian * self._coordinate_sign
        rank = int(numpy.linalg.matrix_rank(packed))
        if require_full_rank and rank != value.size:
            raise RuntimeError(
                "Constraint Jacobian is rank deficient (rank %d of %d)"
                % (rank, value.size)
            )
        if require_full_rank:
            self.constraint_rank = rank
        return value, packed

    def _constraint_jvp(self, t1, t2, vector):
        """JVP in PySCF's packed independent-amplitude convention."""
        self._ensure_coordinates(t1, t2)
        vector = numpy.asarray(vector, dtype=float)
        if vector.shape != (len(self.constraint_coordinates),):
            raise ValueError("constraint JVP vector has an invalid shape")
        if self._matrix_free_constraints:
            result = timed_constraint(
                self, "jvp", self.constraint_jvp,
                t1, t2, self._coordinate_sign * vector,
            )
        else:
            _, jacobian = self._constraint_jacobian(
                t1, t2, require_full_rank=False
            )
            result = jacobian @ vector
        result = numpy.asarray(result, dtype=float)
        if result.shape != (self._constraint_size,):
            raise ValueError("constraint JVP returned an invalid shape")
        return result

    def _constraint_vjp(self, t1, t2, weights):
        """VJP in PySCF's packed independent-amplitude convention."""
        self._ensure_coordinates(t1, t2)
        weights = numpy.asarray(weights, dtype=float)
        if weights.shape != (self._constraint_size,):
            raise ValueError("constraint VJP weights have an invalid shape")
        if self._matrix_free_constraints:
            result = timed_constraint(
                self, "vjp", self.constraint_vjp, t1, t2, weights
            )
            result = self._coordinate_sign * numpy.asarray(result, dtype=float)
        else:
            _, jacobian = self._constraint_jacobian(
                t1, t2, require_full_rank=False
            )
            result = jacobian.T @ weights
        if result.shape != (len(self.constraint_coordinates),):
            raise ValueError("constraint VJP returned an invalid shape")
        return result

    def _constraint_row_norms(self, t1, t2):
        if self._matrix_free_constraints:
            norms = timed_constraint(
                self, "row_norms", self.constraint_jacobian_row_norms, t1, t2
            )
            norms = numpy.asarray(norms, dtype=float)
        else:
            _, jacobian = self._constraint_jacobian(
                t1, t2, require_full_rank=False
            )
            norms = numpy.linalg.norm(jacobian, axis=1)
        if norms.shape != (self._constraint_size,):
            raise ValueError("constraint row norms have an invalid shape")
        return norms

    def _constraint_gram_diagonal(self, t1, t2, metric):
        if self._matrix_free_constraints:
            diagonal = timed_constraint(
                self, "gram_diagonal",
                self.constraint_weighted_gram_diagonal,
                t1, t2, metric,
            )
            diagonal = numpy.asarray(diagonal, dtype=float)
        else:
            _, jacobian = self._constraint_jacobian(
                t1, t2, require_full_rank=False
            )
            diagonal = numpy.einsum("ri,i,ri->r", jacobian, metric, jacobian)
        if diagonal.shape != (self._constraint_size,):
            raise ValueError("constraint Gram diagonal has an invalid shape")
        return diagonal

    def constraint_residual(self, t1=None, t2=None):
        if t1 is None:
            t1 = self.t1
        if t2 is None:
            t2 = self.t2
        return self._constraint_value(t1, t2)

    def polish_newton(self, eris=None, options=None):
        """Opt-in outer solve of amplitude and constraint equations together."""
        from .constraint_newton import polish
        return polish(self, eris, self.newton_options if options is None else options)

    def ccsd(self, t1=None, t2=None, eris=None, mbpt2=False):
        configure_constraint_timing(self)
        self._constraint_cycle = 0
        self.constraint_multipliers = None
        return uccsd.UCCSD.ccsd(self, t1=t1, t2=t2, eris=eris, mbpt2=mbpt2)

    def update_amps(self, t1, t2, eris):
        t1base, t2base = uccsd.update_amps(self, t1, t2, eris)
        old = self.amplitudes_to_vector(t1, t2)
        base = self.amplitudes_to_vector(t1base, t2base)
        constraint = self._constraint_value(t1, t2)
        self._ensure_coordinates(t1, t2)
        denominators = _checked_denominators(self, t1, t2, eris)
        selected = self._coordinate_flat
        metric = -1.0 / denominators[selected]
        if self._matrix_free_constraints:
            response = lambda weights: metric * self._constraint_vjp(
                t1, t2, weights
            )
            response_transpose = lambda vector: self._constraint_jvp(
                t1, t2, metric * vector
            )
            gram_matvec = lambda weights: self._constraint_jvp(
                t1, t2, response(weights)
            )
            rhs = -(constraint + self._constraint_jvp(
                t1, t2, base[selected] - old[selected]
            ))
            diagonal = self._constraint_gram_diagonal(t1, t2, metric)
            multipliers, correction = matrix_free_multipliers(
                self, gram_matvec, response, rhs, diagonal,
                initial=self.constraint_multipliers,
            )
            self.constraint_rank = None
            self.constraint_gram_rank = None
            self.constraint_gram_condition = None
        else:
            constraint, jacobian = self._constraint_jacobian(t1, t2)
            response_matrix = jacobian.T * metric[:, None]
            gram = jacobian @ response_matrix
            rhs = -(constraint + jacobian @ (
                base[selected] - old[selected]
            ))
            multipliers, correction = damped_multipliers(
                self, gram, response_matrix, rhs, constraint.size
            )
        if self.constraint_diis_scheme == "tangent":
            # eta_LS reconstructs the multiplier that would make the raw
            # amplitude residual alone (no -C forcing term) consistent with
            # J^T eta -- so it needs the SAME J . residual_selected term
            # already built for `rhs` above, not response_transpose/
            # response_matrix.T, which would apply the metric a second time
            # (residual_selected = base-old already has one factor of the
            # metric baked in) and keep eta_LS from converging to the true
            # multiplier even when the raw residual is fully explainable.
            #
            # r_par is residual_selected + response(eta_LS), not minus: the
            # real update is new = base + response(multipliers), so at a
            # fixed point residual_selected = base-old = -response(eta*) --
            # the sign that makes r_par vanish there is a PLUS.
            residual_selected = base[selected] - old[selected]
            if self._matrix_free_constraints:
                rhs_ls = -self._constraint_jvp(t1, t2, residual_selected)
                eta_ls = tangent_least_squares_matrix_free(
                    self, gram_matvec, diagonal, rhs_ls, initial=multipliers
                )
                self._constraint_tangent_residual = (
                    residual_selected + response(eta_ls)
                )
            else:
                rhs_ls = -(jacobian @ residual_selected)
                eta_ls = tangent_least_squares_dense(gram, rhs_ls)
                self._constraint_tangent_residual = (
                    residual_selected + response_matrix @ eta_ls
                )
        if self.constraint_multiplier_solver == "nonlinear":
            if self._matrix_free_constraints:
                def evaluate_products(values):
                    values = numpy.asarray(values, dtype=float).copy()
                    self._set_coordinate_values(
                        t1base, t2base, self._coordinate_sign * values
                    )
                    trial_constraint = self._constraint_value(t1base, t2base)

                    def trial_jvp(vector):
                        self._set_coordinate_values(
                            t1base, t2base, self._coordinate_sign * values
                        )
                        return self._constraint_jvp(t1base, t2base, vector)

                    def trial_vjp(weights):
                        self._set_coordinate_values(
                            t1base, t2base, self._coordinate_sign * values
                        )
                        return self._constraint_vjp(t1base, t2base, weights)

                    return (
                        trial_constraint,
                        trial_jvp,
                        trial_vjp,
                    )

                multipliers, correction = nonlinear_multipliers_matrix_free(
                    self, multipliers, response, response_transpose,
                    base[selected], evaluate_products,
                )
            else:
                def evaluate(values):
                    public_values = self._coordinate_sign * values
                    self._set_coordinate_values(
                        t1base, t2base, public_values
                    )
                    return self._constraint_jacobian(
                        t1base, t2base, require_full_rank=False
                    )

                multipliers, correction = nonlinear_multipliers(
                    self, multipliers, response_matrix, base[selected], evaluate
                )
        new = base.copy()
        new[selected] += correction
        if self._matrix_free_constraints:
            linearized = constraint + self._constraint_jvp(
                t1, t2, new[selected] - old[selected]
            )
        else:
            linearized = constraint + jacobian @ (
                new[selected] - old[selected]
            )

        self.constraint_multipliers = multipliers
        self.constraint_linearized_residual = linearized
        self._constraint_old_vector = old.copy()
        t1new, t2new = self.vector_to_amplitudes(new)
        self.constraint_value = self._constraint_value(t1new, t2new)
        log_constraint_timing(self, self._constraint_cycle)
        self._constraint_cycle += 1
        return t1new, t2new

    def run_diis(self, t1, t2, istep, normt, de, adiis):
        self.constraint_value = self._constraint_value(t1, t2)
        if (adiis and istep >= self.diis_start_cycle
                and abs(de) < self.diis_start_energy_diff):
            vector = self.amplitudes_to_vector(t1, t2)
            if self._constraint_old_vector is None:
                amplitude_error = numpy.zeros_like(vector)
            else:
                amplitude_error = vector - self._constraint_old_vector
            if self.constraint_diis_scheme == "tangent":
                error = amplitude_error.copy()
                if self._constraint_tangent_residual is not None:
                    error[self._coordinate_flat] = self._constraint_tangent_residual
                self.constraint_diis_error = None
            else:
                row_norms = self._constraint_row_norms(t1, t2)
                scaled = _constraint_diis_error(
                    self, self.constraint_value, row_norms
                )
                error = numpy.append(amplitude_error, scaled)
                self.constraint_diis_error = scaled
            vector = adiis.update(vector, xerr=error)
            t1, t2 = self.vector_to_amplitudes(vector)
            self.constraint_value = self._constraint_value(t1, t2)
            logger.debug1(self, "Vector-constrained DIIS for step %d", istep)
        return t1, t2

    def _finalize(self):
        if self.t1 is not None and self.t2 is not None:
            # `converged` is about to fold in the constraint tolerance (and
            # possibly the post-loop resync pass below); keep the amplitude-
            # equation verdict separate, since that's what a caller uses to
            # decide whether to continue with the nonlinear multiplier
            # solver.  Captured here, before either of those can touch it.
            self.amplitudes_converged = self.converged
            if (self.conv_check and self.converged
                    and float(numpy.max(numpy.abs(
                        self._constraint_value(self.t1, self.t2)))) > self.constraint_tol):
                # One extra Newton/eta correction pass, no DIIS after it --
                # the main loop's convergence test uses the pre-DIIS residual
                # norm, so it can exit leaving the true (post-DIIS) t1/t2 with
                # an unresolved constraint violation that nothing else
                # corrects.  Bounded to exactly one pass (never a retry loop):
                # if it doesn't resolve the constraint, or moves the energy/
                # amplitudes far more than a small cleanup step should, flag
                # not converged rather than silently accepting the result.
                eris = self.ao2mo(self.mo_coeff)
                e_before = self.e_corr
                old_vec = self.amplitudes_to_vector(self.t1, self.t2)
                t1new, t2new = self.update_amps(self.t1, self.t2, eris)
                normt_extra = float(numpy.linalg.norm(
                    self.amplitudes_to_vector(t1new, t2new) - old_vec))
                self.t1, self.t2 = t1new, t2new
                self.e_corr = self.energy(self.t1, self.t2, eris)
                maximum_after = float(numpy.max(numpy.abs(
                    self._constraint_value(self.t1, self.t2))))
                de_extra = abs(self.e_corr - e_before)
                if (maximum_after > self.constraint_tol
                        or de_extra > 10 * self.conv_tol
                        or normt_extra > 10 * self.conv_tol_normt):
                    logger.warn(
                        self,
                        "Post-loop constraint resync did not settle cleanly "
                        "(max|C(t)| = %.6e, dE = %.6e, norm(d t1,t2) = %.6e) "
                        "-- flagging not converged",
                        maximum_after, de_extra, normt_extra,
                    )
                    self.converged = False
            self.constraint_value = self._constraint_value(self.t1, self.t2)
            maximum = float(numpy.max(numpy.abs(self.constraint_value)))
            if maximum > self.constraint_tol:
                if self.converged:
                    logger.warn(
                        self,
                        "CC equations converged but max|C(t)| = %.6e exceeds %.6e",
                        maximum,
                        self.constraint_tol,
                    )
                self.converged = False
            if self._matrix_free_constraints:
                logger.note(
                    self,
                    "Constraint max|C(t)| = %.12g  Krylov residual = %s",
                    maximum,
                    ("%.3e" % self.constraint_krylov_residual
                     if self.constraint_krylov_residual is not None else "n/a"),
                )
            else:
                logger.note(
                    self, "Constraint max|C(t)| = %.12g  rank = %d",
                    maximum, self.constraint_rank,
                )
        dump_constraint_timing(self)
        if self.newton_polish:
            self.polish_newton()
        return uccsd.UCCSD._finalize(self)

    def check_constraint_jacobian(self, t1=None, t2=None, step=1e-6,
                                  ndir=3, seed=1, tolerance=1e-6):
        if t1 is None or t2 is None:
            if self.t1 is not None and self.t2 is not None:
                t1, t2 = self.t1, self.t2
            else:
                t1, t2 = self.get_init_guess()
        self._ensure_coordinates(t1, t2)
        vector = self.amplitudes_to_vector(t1, t2)
        self._constraint_value(t1, t2)
        rng = numpy.random.default_rng(seed)
        maximum = 0.0
        for _ in range(ndir):
            direction = rng.standard_normal(len(self._coordinate_flat))
            direction /= numpy.linalg.norm(direction)
            plus = vector.copy()
            minus = vector.copy()
            plus[self._coordinate_flat] += step * direction
            minus[self._coordinate_flat] -= step * direction
            t1p, t2p = self.vector_to_amplitudes(plus)
            t1m, t2m = self.vector_to_amplitudes(minus)
            finite_difference = (
                self._constraint_value(t1p, t2p)
                - self._constraint_value(t1m, t2m)
            ) / (2.0 * step)
            analytic = self._constraint_jvp(t1, t2, direction)
            scale = numpy.maximum(1.0, numpy.maximum(
                numpy.abs(finite_difference), numpy.abs(analytic)
            ))
            maximum = max(
                maximum,
                float(numpy.max(numpy.abs(finite_difference - analytic) / scale)),
            )
        if maximum > tolerance:
            raise ValueError(
                "Constraint Jacobian finite-difference error %.6e exceeds %.6e"
                % (maximum, tolerance)
            )
        return maximum

    def solve_lambda(self, t1=None, t2=None, l1=None, l2=None, eris=None):
        """Solve the true lambda equations of the constrained Lagrangian.

        L = E + sum_mu lambda_mu (R_mu - zeta.dC/dt_mu) + kappa.C, with the
        converged constraint multipliers zeta promoted to variables.
        Stationarity in zeta forces lambda orthogonal to every constraint
        gradient, and stationarity in t gives the ordinary lambda equations
        with two modifications: the zeta-weighted constraint Hessian, and
        the true-Jacobian correction (the ordinary equations assume the
        amplitude residual vanishes, but here R = G^T zeta, and dR/dt then
        gains commutator terms of which a Lambda2 . R1 contraction survives
        in CCSD).  kappa never enters the density matrices (the constraints
        carry no integral dependence), so it is projected out rather than
        solved for; its value is recovered afterwards as a diagnostic.

        Requires the constraint's Hessian-vector product; construct the
        solver with constraint_hessian_vector= (for the CISD engine,
        engine.hessian_vector).  The resulting l1, l2 make the inherited
        make_rdm1/make_rdm2 exact response densities: dE/dh_pq = gamma_pq.
        """
        if self.constraint_hessian_vector is None:
            raise NotImplementedError(
                "the constraint has not implemented the Hessian-vector "
                "product; construct the solver with "
                "constraint_hessian_vector= to solve the lambda equations"
            )
        if self.constraint_multipliers is None:
            raise RuntimeError(
                "solve the amplitude equations first: the lambda equations "
                "need the converged constraint multipliers"
            )
        if t1 is None:
            t1 = self.t1
        if t2 is None:
            t2 = self.t2
        if eris is None:
            eris = self.ao2mo(self.mo_coeff)

        self._ensure_coordinates(t1, t2)
        selected, signs = self._coordinate_flat, self._coordinate_sign
        self._constraint_value(t1, t2)
        gradients = gram = None
        if self._matrix_free_constraints:
            gram_diagonal = self._constraint_gram_diagonal(
                t1, t2, numpy.ones(len(selected))
            )

            def gram_matvec(weights):
                return self._constraint_jvp(
                    t1, t2, self._constraint_vjp(t1, t2, weights)
                )

            gram_operator = sparse_linalg.LinearOperator(
                (self._constraint_size, self._constraint_size),
                matvec=gram_matvec, dtype=float,
            )
            threshold = max(
                float(numpy.max(gram_diagonal, initial=0.0)), 1.0
            ) * numpy.finfo(float).eps
            inverse_diagonal = numpy.ones_like(gram_diagonal)
            nonzero = gram_diagonal > threshold
            inverse_diagonal[nonzero] = 1.0 / gram_diagonal[nonzero]
            gram_preconditioner = sparse_linalg.LinearOperator(
                gram_operator.shape,
                matvec=lambda values: inverse_diagonal * values,
                dtype=float,
            )

            def solve_gram(rhs):
                solution, info = sparse_linalg.cg(
                    gram_operator, rhs, M=gram_preconditioner,
                    rtol=self.constraint_krylov_tol, atol=0.0,
                    maxiter=self.constraint_krylov_max_cycle,
                )
                if info != 0:
                    raise RuntimeError(
                        "matrix-free constraint projection failed "
                        "(CG info = %s)" % info
                    )
                return solution
        else:
            _, gradients = self._constraint_jacobian(t1, t2)
            gram = gradients @ gradients.T

            def solve_gram(rhs):
                return numpy.linalg.solve(gram, rhs)
        multipliers = numpy.asarray(self.constraint_multipliers, dtype=float)
        denominators = _checked_denominators(self, t1, t2, eris)
        # singles block of the converged amplitude residual R = G^T zeta;
        # it feeds the true-Jacobian correction inside the residual
        t1base, t2base = uccsd.update_amps(self, t1, t2, eris)
        rvec = denominators * (
            self.amplitudes_to_vector(t1base, t2base)
            - self.amplitudes_to_vector(t1, t2)
        )
        residual1 = self.vector_to_amplitudes(rvec)[0]
        imds = uccsd_lambda.make_intermediates(self, t1, t2, eris)

        def project(values):
            # remove the span of the constraint gradients:
            # values - G^T (G G^T)^-1 G values
            if self._matrix_free_constraints:
                return values - self._constraint_vjp(
                    t1, t2, solve_gram(self._constraint_jvp(t1, t2, values))
                )
            return values - gradients.T @ solve_gram(gradients @ values)

        def residual(lamvec):
            """sigma + lambda.(A - zeta.d2C) at the given lambda.

            Three pieces: pyscf's lambda residual (its update is lambda +
            rho/D with the exact textbook rho, verified against a Wick
            derivation); the true-Jacobian correction -(Lambda2 . R1) on
            the Lambda1 rows (pyscf's equations assume R = 0, but here
            R = G^T zeta); and the constraint Hessian term.
            """
            l1x, l2x = self.vector_to_amplitudes(lamvec)
            l1n, l2n = uccsd_lambda.update_lambda(
                self, t1, t2, l1x, l2x, eris, imds
            )
            rho = denominators * (
                self.amplitudes_to_vector(l1n, l2n) - lamvec
            )
            c1a = (numpy.einsum("ijab,jb->ia", l2x[0], residual1[0])
                   + numpy.einsum("iJaB,JB->ia", l2x[1], residual1[1]))
            c1b = (numpy.einsum("IJAB,JB->IA", l2x[2], residual1[1])
                   + numpy.einsum("jIbA,jb->IA", l2x[1], residual1[0]))
            rho -= self.amplitudes_to_vector(
                (c1a, c1b), tuple(numpy.zeros_like(b) for b in l2x)
            )
            rho[selected] -= signs * self.constraint_hessian_vector(
                t1, t2, multipliers, signs * lamvec[selected]
            )
            return rho

        # solve P (M^T lambda + sigma) = 0 with lambda = P lambda by Krylov;
        # kappa (the multiplier of C) is exactly what the projector removes,
        # and the residual is affine so its linear action is a difference
        nvector = denominators.size
        rho0 = residual(numpy.zeros(nvector))

        def matvec(x):
            x = x.copy()
            x[selected] = project(x[selected])
            out = residual(x) - rho0
            out[selected] = project(out[selected])
            return out

        rhs = -rho0.copy()
        rhs[selected] = project(rhs[selected])
        guess = (self.amplitudes_to_vector(t1, t2) if l1 is None
                 else self.amplitudes_to_vector(l1, l2))
        guess[selected] = project(guess[selected])
        # dtype must be declared: without it scipy probes matvec with an int8
        # vector to infer one, and that integer array reaches pyscf's
        # _contract_s4vvvv_t2, which asserts float64.  The pytblis einsum
        # backend happens to promote it and hide this; the numpy path does not.
        lambda_operator = sparse_linalg.LinearOperator(
            (nvector, nvector), matvec=matvec, dtype=float,
        )
        lambda_preconditioner = sparse_linalg.LinearOperator(
            (nvector, nvector), matvec=lambda x: x / denominators, dtype=float,
        )
        lam, info = sparse_linalg.lgmres(
            lambda_operator, rhs, x0=guess, M=lambda_preconditioner,
            rtol=1e-8, atol=0.0, maxiter=self.max_cycle
        )
        lam[selected] = project(lam[selected])
        self.converged_lambda = info == 0
        if not self.converged_lambda:
            logger.warn(self, "constrained lambda Krylov solve did not "
                        "converge (lgmres info = %s)", info)
        self.l1, self.l2 = self.vector_to_amplitudes(lam)

        rho = residual(lam)
        if self._matrix_free_constraints:
            self.constraint_kappa = -solve_gram(
                self._constraint_jvp(t1, t2, rho[selected])
            )
            orthogonality = self._constraint_jvp(t1, t2, lam[selected])
        else:
            self.constraint_kappa = -solve_gram(gradients @ rho[selected])
            orthogonality = gradients @ lam[selected]
        self.constraint_lambda_orthogonality = float(numpy.max(
            numpy.abs(orthogonality), initial=0.0
        ))
        logger.info(self, "lambda orthogonality max|G lambda| = %.3e",
                    self.constraint_lambda_orthogonality)
        logger.info(self, "eliminated constraint multiplier max|kappa| = %.3e",
                    float(numpy.max(numpy.abs(self.constraint_kappa),
                                    initial=0.0)))
        if self.constraint_lambda_orthogonality > self.conv_tol_normt:
            logger.warn(self, "lambda is not orthogonal to the constraint "
                        "gradients (max|G lambda| = %.3e)",
                        self.constraint_lambda_orthogonality)
        return self.l1, self.l2

"""Opt-in, regularized outer Newton polishing of constrained CCSD.

This solves [R(t)-G(t).T eta, C(t)] jointly. It does not replace the
existing multiplier-only Newton/Gauss-Newton correction inside a CC update.
The matrix-free backend forms no coupled Jacobian, normal matrix, or
constraint Gram matrix; the dense backend is a small reference solver.
"""
from dataclasses import dataclass, fields
from typing import Optional
import numpy as np
from scipy.sparse.linalg import LinearOperator, cg, lsmr
from scipy.linalg import lstsq
from pyscf.cc import uccsd
from pyscf.lib import logger
from .uccsd_response import UCCSDResponse
from .uccsd_constr import _checked_denominators


@dataclass(frozen=True)
class NewtonOptions:
    backend: str = "matrix_free"
    dense_cutoff: float = 1e-13
    residual_tol: float = 1e-9
    constraint_tol: Optional[float] = None
    max_cycle: int = 50
    krylov_max_cycle: int = 200
    max_trials: int = 8
    max_step: float = .1
    damping_initial: float = 1e-8
    damping_min: float = 1e-14
    damping_max: float = 1e2
    krylov_tol: float = 1e-12

    @classmethod
    def parse(cls, options, cc):
        if options is None:
            values = {}
        elif isinstance(options, cls):
            values = {f.name: getattr(options, f.name) for f in fields(cls)}
        else:
            values = dict(options)
        if values.get('constraint_tol') is None:
            values['constraint_tol'] = cc.constraint_tol
        obj = cls(**values)
        for name in ('max_cycle', 'krylov_max_cycle', 'max_trials'):
            value = getattr(obj, name)
            if isinstance(value, bool) or int(value) != value or value <= 0:
                raise ValueError(name + ' must be a positive integer')
        if obj.backend not in ("matrix_free", "dense"):
            raise ValueError("backend must be matrix_free or dense")
        for field in fields(obj):
            if field.name == "backend":
                continue
            value = getattr(obj, field.name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(field.name + ' must be finite and positive')
        if not obj.damping_min <= obj.damping_initial <= obj.damping_max:
            raise ValueError('require damping_min <= damping_initial <= damping_max')
        return obj


@dataclass
class NewtonReport:
    success: bool
    reason: str
    iterations: int
    history: list
    residual_evaluations: int
    linear_iterations: int
    max_amplitude_residual: float
    max_constraint: float
    energy: float


class ConstraintActions:
    """Original constraints in the full packed independent-amplitude space."""
    def __init__(self, cc, t1, t2):
        self.cc, self.t1, self.t2 = cc, t1, t2
        cc._ensure_coordinates(t1, t2)
        self.selected = cc._coordinate_flat
        self.signs = cc._coordinate_sign
        self.n = cc.amplitudes_to_vector(t1, t2).size
        self.value = cc._constraint_value(t1, t2)
        if self.n > 1024 and not cc._matrix_free_constraints:
            raise NotImplementedError('large-system Newton requires constraint JVP/VJP and row-norm callbacks')
        if cc.constraint_hessian_vector is None:
            raise NotImplementedError('outer Newton requires the weighted constraint Hessian-vector product')

    def jvp(self, vector):
        if not self.value.size:
            return np.zeros(0)
        return self.cc._constraint_jvp(self.t1, self.t2, vector[self.selected])

    def vjp(self, weights):
        result = np.zeros(self.n)
        if not self.value.size:
            return result
        result[self.selected] = self.cc._constraint_vjp(self.t1, self.t2, weights)
        return result

    def hvp(self, weights, vector):
        result = np.zeros(self.n)
        if not self.value.size:
            return result
        result[self.selected] = self.signs * self.cc.constraint_hessian_vector(
            self.t1, self.t2, weights, self.signs * vector[self.selected])
        return result

    def row_norms(self):
        if not self.value.size:
            return np.zeros(0)
        return self.cc._constraint_row_norms(self.t1, self.t2)


def coupled_operator(response, constraints, eta):
    n, m = response.x.size, constraints.value.size

    def matvec(v):
        a, e = v[:n], v[n:]
        return np.concatenate((response.jvp(a) - constraints.hvp(eta, a)
                               - constraints.vjp(e), constraints.jvp(a)))

    def rmatvec(v):
        a, e = v[:n], v[n:]
        return np.concatenate((response.vjp(a) - constraints.hvp(eta, a)
                               + constraints.vjp(e), -constraints.jvp(a)))

    return LinearOperator((n+m, n+m), matvec=matvec, rmatvec=rmatvec, dtype=float)


def scaled_operator(operator, rows, columns):
    return LinearOperator(operator.shape,
                          matvec=lambda v: rows * operator.matvec(columns*v),
                          rmatvec=lambda v: columns * operator.rmatvec(rows*v),
                          dtype=float)


def operator_scale(operator):
    # Deterministic, fixed-storage power iteration (not a dense norm estimate).
    v = np.random.default_rng(971).normal(size=operator.shape[1])
    v /= np.linalg.norm(v)
    for _ in range(5):
        w = operator.rmatvec(operator.matvec(v))
        norm = np.linalg.norm(w)
        if not np.isfinite(norm):
            raise FloatingPointError('nonfinite Newton operator')
        if norm == 0:
            return 1e-12
        v = w/norm
    return max(float(np.linalg.norm(operator.matvec(v))), 1e-12)


def _residual(cc, x, eta, eris):
    t1, t2 = cc.vector_to_amplitudes(x)
    u1, u2 = uccsd.update_amps(cc, t1, t2, eris)
    r = _checked_denominators(cc, t1, t2, eris) * (
        cc.amplitudes_to_vector(u1, u2) - x)
    constraints = ConstraintActions(cc, t1, t2)
    return np.concatenate((r-constraints.vjp(eta), constraints.value))


def fit_multipliers(response, constraints, tolerance, max_cycle):
    """Least-squares fit of G.T eta to R without forming G G.T."""
    m = constraints.value.size
    if not m:
        return np.zeros(0)
    operator = LinearOperator(
        (m, m),
        matvec=lambda value: constraints.jvp(constraints.vjp(value)),
        dtype=float,
    )
    diagonal = constraints.cc._constraint_gram_diagonal(
        constraints.t1, constraints.t2,
        np.ones(len(constraints.selected)),
    )
    floor = max(float(np.max(diagonal, initial=0)), 1.0) * np.finfo(float).eps
    preconditioner = LinearOperator(
        (m, m), matvec=lambda value: value / np.maximum(diagonal, floor),
        dtype=float,
    )
    eta, info = cg(
        operator, constraints.jvp(response.residual), M=preconditioner,
        rtol=max(tolerance, 1e-12), atol=0.0, maxiter=max_cycle,
    )
    if info == 0 and np.all(np.isfinite(eta)):
        return eta
    # LSMR is a robust fallback for a singular or incomplete CG fit.
    gt = LinearOperator((response.x.size, m), matvec=constraints.vjp,
                        rmatvec=constraints.jvp, dtype=float)
    return lsmr(gt, response.residual, atol=tolerance, btol=tolerance,
                maxiter=max_cycle)[0]


def polish(cc, eris=None, options=None):
    opt = NewtonOptions.parse(options, cc)
    if cc.t1 is None or cc.t2 is None:
        raise RuntimeError('run the parent amplitude iteration before Newton polishing')
    if eris is None:
        eris = cc.ao2mo(cc.mo_coeff)
    if np.iscomplexobj(eris.focka) or np.iscomplexobj(eris.fockb):
        raise NotImplementedError('Newton CCSD currently supports real Hamiltonians')
    x = cc.amplitudes_to_vector(cc.t1, cc.t2).copy()
    if not np.all(np.isfinite(x)):
        raise ValueError('Newton initial amplitudes must be finite')
    original = x.copy()
    n = x.size
    constraints = ConstraintActions(cc, cc.t1, cc.t2)
    m = constraints.value.size
    if opt.backend == "dense" and n+m > 1024:
        raise ValueError("dense Newton is a small-system reference backend (at most 1024 unknowns)")
    eta = cc.constraint_multipliers
    if m == 0:
        eta = np.zeros(0)
    elif eta is None or np.shape(eta) != (m,) or not np.all(np.isfinite(eta)):
        response = UCCSDResponse(cc, cc.t1, cc.t2, eris, transpose=False)
        if opt.backend == "dense":
            gt = LinearOperator((n, m), matvec=constraints.vjp,
                                rmatvec=constraints.jvp, dtype=float)
            matrix = np.column_stack([gt.matvec(e) for e in np.eye(m)])
            eta = lstsq(matrix, response.residual, cond=opt.dense_cutoff)[0]
        else:
            eta = fit_multipliers(
                response, constraints, opt.krylov_tol,
                opt.krylov_max_cycle,
            )
    else:
        eta = np.asarray(eta, dtype=float).copy()
    history = []
    evaluations, total_linear = 0, 0
    relative_damping = opt.damping_initial
    success, reason = False, 'iteration_limit'
    last_energy = None

    for iteration in range(opt.max_cycle+1):
        f = _residual(cc, x, eta, eris)
        evaluations += 1
        if not np.all(np.isfinite(f)):
            reason = 'nonfinite_residual'
            break
        t1, t2 = cc.vector_to_amplitudes(x)
        energy = float(cc.energy(t1, t2, eris) + cc._scf.e_tot)
        max_r, max_c = float(np.max(abs(f[:n]), initial=0)), float(np.max(abs(f[n:]), initial=0))
        row = dict(iteration=iteration, energy=energy,
                   energy_change=None if last_energy is None else energy-last_energy,
                   max_amplitude_residual=max_r, max_constraint=max_c, trials=[])
        history.append(row)
        logger.info(cc, 'Outer Newton %d E = %.14g max|R-G^T eta| = %.3e max|C| = %.3e',
                    iteration, energy, max_r, max_c)
        last_energy = energy
        if max_r < opt.residual_tol and max_c < opt.constraint_tol:
            success, reason = True, 'joint_residual_converged'
            break
        if iteration == opt.max_cycle:
            break
        response = UCCSDResponse(cc, t1, t2, eris)
        try:
            constraints = ConstraintActions(cc, t1, t2)
            operator = coupled_operator(response, constraints, eta)
            if opt.backend == "dense":
                # Reference implementation of the validated diagnostic solve.
                # The size guard above prevents accidental production-scale use.
                matrix = np.column_stack([operator.matvec(e) for e in np.eye(n+m)])
                step, _, rank, singular = lstsq(matrix, -f, cond=opt.dense_cutoff)
                step *= min(1., opt.max_step/max(float(np.max(abs(step[:n]))), 1e-30))
                accepted = False
                merit = .5*float(f@f)
                for trial in range(opt.max_trials):
                    predicted = merit-.5*float((f+matrix@step)@(f+matrix@step))
                    candidate = _residual(cc, x+step[:n], eta+step[n:], eris)
                    evaluations += 1
                    ratio = ((merit-.5*float(candidate@candidate))/predicted
                             if predicted > 0 and np.all(np.isfinite(candidate)) else -np.inf)
                    row['trials'].append(dict(rank=int(rank), cutoff=opt.dense_cutoff,
                                              reduction_ratio=float(ratio),
                                              smallest_singular=float(singular[-1])))
                    if ratio > .1:
                        x += step[:n]
                        eta += step[n:]
                        accepted = True
                        break
                    step *= .5
                if not accepted:
                    reason = 'no_acceptable_step'
                    break
                continue
            gaps = abs(response.denominators)
            floor = 1e-8*max(1., float(max(gaps)))
            rows = np.concatenate((1/np.maximum(gaps, floor),
                                   1/np.maximum(constraints.row_norms(), 1e-12)))
            columns = np.concatenate((np.ones(n), np.full(m, max(1., float(np.median(gaps))))))
            scaled = scaled_operator(operator, rows, columns)
            scale = operator_scale(scaled)
            b = rows*f
            merit = .5*float(b@b)
            accepted = False
            for trial in range(opt.max_trials):
                mu = relative_damping*scale
                solved = lsmr(scaled, -b, damp=mu, atol=opt.krylov_tol,
                              btol=opt.krylov_tol, maxiter=opt.krylov_max_cycle)
                krylov_step, status, inner_iterations = solved[:3]
                total_linear += inner_iterations
                step = columns*krylov_step
                step *= min(1., opt.max_step/max(float(np.max(abs(step[:n]))), 1e-30))
                prediction = b + scaled.matvec(step/columns)
                predicted_reduction = merit-.5*float(prediction@prediction)
                ratio = -np.inf
                if predicted_reduction > 0 and np.all(np.isfinite(step)):
                    candidate = _residual(cc, x+step[:n], eta+step[n:], eris)
                    evaluations += 1
                    if np.all(np.isfinite(candidate)):
                        candidate_merit = .5*float((rows*candidate)@(rows*candidate))
                        ratio = (merit-candidate_merit)/predicted_reduction
                trial_info = dict(damping=mu, relative_damping=relative_damping,
                                  lsmr_status=int(status), linear_iterations=int(inner_iterations),
                                  condition_estimate=float(solved[6]), reduction_ratio=float(ratio),
                                  max_amplitude_step=float(np.max(abs(step[:n]))))
                row['trials'].append(trial_info)
                logger.debug(cc, 'Outer Newton linear solve: damping %.3e iterations %d status %d ratio %.3e', mu, inner_iterations, status, ratio)
                if ratio > .1:
                    x += step[:n]
                    eta += step[n:]
                    accepted = True
                    relative_damping *= 1/3 if ratio > .75 else 3 if ratio < .25 else 1
                    relative_damping = float(np.clip(relative_damping, opt.damping_min, opt.damping_max))
                    break
                relative_damping = min(opt.damping_max, relative_damping*10)
            if not accepted:
                reason = 'no_acceptable_step'
                break
        finally:
            response.close()

    # Commit exactly the last accepted iterate; never a rejected trial/cache.
    cc.t1, cc.t2 = cc.vector_to_amplitudes(x)
    cc.e_hf = cc._scf.e_tot
    cc.e_corr = cc.energy(cc.t1, cc.t2, eris)
    cc.constraint_multipliers = eta.copy()
    cc.constraint_value = ConstraintActions(cc, cc.t1, cc.t2).value.copy()
    cc.converged = success
    if not np.array_equal(original, x):
        cc.l1 = cc.l2 = None
        cc.converged_lambda = False
        cc.constraint_kappa = None
        cc.constraint_lambda_orthogonality = None
    final = _residual(cc, x, eta, eris)
    evaluations += 1
    report = NewtonReport(success, reason, len(history)-1, history, evaluations,
                          int(total_linear), float(np.max(abs(final[:n]), initial=0)),
                          float(np.max(abs(final[n:]), initial=0)), float(cc.e_tot))
    cc.newton_report = report
    logger.note(cc, 'Outer Newton: %s; max|R-G^T eta| %.3e, max|C| %.3e',
                reason, report.max_amplitude_residual, report.max_constraint)
    return report

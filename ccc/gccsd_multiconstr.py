"""GCCSD with CISD-closed spin constraints on a general (spinor) reference.

The solver is the same Newton-on-the-amplitude-update construction as
:class:`~ccc.uccsd_multiconstr.MultiConstrUCCSD`, and it imports that
module's multiplier machinery unchanged -- those helpers only ever touch generic
``cc.constraint_*`` attributes.  What it does not need is the coordinate
book-keeping: dense spin operators involve *every* amplitude, so the constraint
coordinates are the whole packed vector, the packed-index map is the identity,
and there are no per-block offsets or antisymmetry signs to carry around.

One structural fact shapes the multiplier solve.  For a complete spinor basis --
which is what a GHF in a given AO basis always gives -- the rows are linearly
dependent: some fixed combination of them vanishes identically, ``w . C(t) = 0``
for all amplitudes.  For any useful size (both the occupied and virtual sets
holding at least three spinors) the deficiency is exactly one; the small
degenerate corners carry a few more.  It is a cross-operator relation -- neither
the Sz rows nor the S+ rows alone are deficient -- it does not depend on Ms or on
the amplitudes, and it has nothing to do with particle number: adding any
multiple of N to Sz, or rescaling S+, leaves it in place.  Truncating the spinor
set removes it entirely.

What matters here is that the dependency is *exact* rather than merely constant,
so the multiplier system is singular but consistent: its minimum-norm solution is
an exact Newton step and no Levenberg damping is needed to get one.
``_minimum_norm_multipliers`` therefore solves through an eigendecomposition and
reports how much of the right-hand side falls outside the range, which is the
number that would grow if the dependency were ever inexact.

Lambda equations and response densities are not implemented: the constrained
Lagrangian gains a J^T lambda term whose derivation is specific to the amplitude
equations, exactly as for the constrained UCCSD solver.
"""
import numpy

from pyscf.cc import gccsd
from pyscf.lib import logger

from .uccsd_multiconstr import (
    _constraint_diis_error,
    configure_constraint_timing,
    configure_diis_scheme,
    damped_multipliers,
    dump_constraint_timing,
    log_constraint_timing,
    matrix_free_multipliers,
    tangent_least_squares_dense,
    tangent_least_squares_matrix_free,
    timed_constraint,
)


def _amplitude_denominators(cc, t1, eris):
    nocc, nvir = t1.shape
    occupied = numpy.asarray(eris.mo_energy[:nocc])
    virtual = numpy.asarray(eris.mo_energy[nocc:]) + cc.level_shift
    eia = occupied[:, None] - virtual[None, :]
    oi, oj = numpy.tril_indices(nocc, k=-1)
    va, vb = numpy.tril_indices(nvir, k=-1)
    occupied_pairs = occupied[oi] + occupied[oj]
    virtual_pairs = virtual[va] + virtual[vb]
    vector = numpy.empty(
        nocc * nvir + oi.size * va.size,
        dtype=numpy.result_type(eia, occupied_pairs, virtual_pairs),
    )
    vector[:nocc * nvir] = eia.ravel()
    vector[nocc * nvir:] = (
        occupied_pairs[:, None] - virtual_pairs[None, :]
    ).ravel()
    return vector


def _checked_denominators(cc, t1, t2, eris):
    denominators = _amplitude_denominators(cc, t1, eris)
    scale = max(1.0, float(numpy.max(numpy.abs(denominators))))
    threshold = numpy.finfo(denominators.dtype).eps * scale * 10.0
    if numpy.any(numpy.abs(denominators) <= threshold):
        raise RuntimeError(
            "Near-zero CC amplitude denominator (min |D| = %.6e)"
            % float(numpy.min(numpy.abs(denominators)))
        )
    return denominators


def _minimum_norm_multipliers(cc, gram, response, rhs, nconstraint):
    """Minimum-norm solution of ``gram lambda = rhs`` for a consistent system.

    The Gram matrix is singular by the exact row dependency described in the
    module docstring, so a plain solve would fail and Levenberg damping would
    perturb every direction to work around one.  Discarding the null space
    instead is exact, provided the right-hand side really has no component
    there; that component is measured and reported rather than assumed.
    """
    eigenvalues, vectors = numpy.linalg.eigh(gram)
    eigenvalues = numpy.maximum(eigenvalues, 0.0)
    largest = float(eigenvalues[-1])
    threshold = largest * nconstraint * numpy.finfo(float).eps
    keep = eigenvalues > threshold
    rank = int(numpy.count_nonzero(keep))

    projected = vectors.T @ rhs
    scale = max(float(numpy.linalg.norm(rhs)), 1.0)
    cc.constraint_consistency_error = float(
        numpy.linalg.norm(projected[~keep]) / scale
    )
    cc.constraint_gram_rank = rank
    cc.constraint_gram_condition = (
        largest / float(eigenvalues[keep][0]) if rank else numpy.inf
    )
    if rank != cc._reported_gram_rank:
        cc._reported_gram_rank = rank
        logger.info(
            cc,
            "Constraint multiplier matrix is rank %d of %d (cond = %.3e); "
            "the right-hand side has %.3e of its norm outside the range",
            rank, nconstraint, cc.constraint_gram_condition,
            cc.constraint_consistency_error,
        )
    if cc.constraint_consistency_error > cc.constraint_consistency_tol:
        raise RuntimeError(
            "the constraint multiplier system is inconsistent: %.3e of the "
            "right-hand side lies outside the range of a rank-%d Gram matrix "
            "(tolerance %.3e).  The rows are not merely redundant, so no "
            "amplitude update can satisfy them all"
            % (cc.constraint_consistency_error, rank,
               cc.constraint_consistency_tol)
        )

    inverse = numpy.zeros_like(eigenvalues)
    inverse[keep] = 1.0 / eigenvalues[keep]
    multipliers = vectors @ (inverse * projected)
    cc.constraint_damping = 0.0
    return multipliers, response @ multipliers


class MultiConstrGCCSD(gccsd.GCCSD):
    """GCCSD constrained by a CISD-closed spin engine.

    Construct with :meth:`from_constraint_engine` and a
    :class:`~ccc.gcisd_spin_constraint.GCISDSpinConstraint`.  The engine
    supplies ``constraint``, ``jvp``, ``vjp``, ``jacobian``,
    ``jacobian_row_norms``, ``weighted_gram_diagonal`` and ``coordinates``; the
    coordinates are the full packed amplitude vector.
    """

    def __init__(self, mf, constraint, constraint_jacobian,
                 constraint_coordinates, constraint_tol=1e-10,
                 constraint_diis_weight=1.0, constraint_max_step=0.0,
                 frozen=None, mo_coeff=None, mo_occ=None,
                 constraint_jvp=None, constraint_vjp=None,
                 constraint_jacobian_row_norms=None,
                 constraint_weighted_gram_diagonal=None,
                 constraint_hessian_vector=None,
                 constraint_krylov_tol=1e-10,
                 constraint_krylov_max_cycle=None,
                 constraint_dense_limit=512 * 1024 ** 2,
                 constraint_consistency_tol=1e-6,
                 conv_check=True, constraint_diis_scheme="legacy"):
        gccsd.GCCSD.__init__(self, mf, frozen=frozen, mo_coeff=mo_coeff,
                             mo_occ=mo_occ)
        if not callable(constraint):
            raise TypeError("constraint must be callable")
        if not callable(constraint_jacobian):
            raise TypeError("constraint_jacobian must be callable")
        if constraint_tol <= 0:
            raise ValueError("constraint_tol must be positive")
        if constraint_diis_weight <= 0:
            raise ValueError("constraint_diis_weight must be positive")
        products = (constraint_jvp, constraint_vjp,
                    constraint_jacobian_row_norms,
                    constraint_weighted_gram_diagonal)
        if not all(callable(product) for product in products):
            raise TypeError(
                "the GCCSD constraints are matrix-free throughout: JVP, VJP, "
                "row norms and Gram diagonal must all be callable"
            )

        self.constraint = constraint
        self.constraint_jacobian = constraint_jacobian
        self.constraint_jvp = constraint_jvp
        self.constraint_vjp = constraint_vjp
        self.constraint_jacobian_row_norms = constraint_jacobian_row_norms
        self.constraint_weighted_gram_diagonal = (
            constraint_weighted_gram_diagonal
        )
        self.constraint_hessian_vector = constraint_hessian_vector
        self.constraint_coordinates = constraint_coordinates
        if not len(self.constraint_coordinates):
            raise ValueError("constraint_coordinates must not be empty")

        self.constraint_tol = float(constraint_tol)
        self.constraint_diis_weight = float(constraint_diis_weight)
        self.constraint_max_step = float(constraint_max_step)
        self.constraint_krylov_tol = float(constraint_krylov_tol)
        if (not numpy.isfinite(self.constraint_krylov_tol)
                or self.constraint_krylov_tol <= 0):
            raise ValueError("constraint_krylov_tol must be positive and finite")
        self.constraint_krylov_max_cycle = (
            None if constraint_krylov_max_cycle is None
            else int(constraint_krylov_max_cycle)
        )
        # A dense Jacobian costs nconstraint * nvec doubles but gives the exact
        # multiplier solve and the rank diagnostic; above this many bytes the
        # solve goes matrix-free instead.
        self.constraint_dense_limit = float(constraint_dense_limit)
        self.constraint_consistency_tol = float(constraint_consistency_tol)

        configure_constraint_timing(self)
        self._constraint_cycle = 0
        self.conv_check = bool(conv_check)
        configure_diis_scheme(self, constraint_diis_scheme)
        self._constraint_size = None
        self._dense_gram = None
        self._constraint_old_vector = None
        self._constraint_metric_cache = None
        self._constraint_metric_eris = None
        self._constraint_metric_level_shift = None
        self._constraint_metric_mo_energy = None
        self.constraint_value = None
        self.constraint_multipliers = None
        self.constraint_rank = None
        self.constraint_damping = 0.0
        self.constraint_gram_rank = None
        self.constraint_gram_condition = None
        self.constraint_consistency_error = None
        self.constraint_krylov_info = None
        self.constraint_krylov_iterations = None
        self.constraint_krylov_residual = None
        self.constraint_linearized_residual = None
        self.constraint_diis_error = None
        self._reported_gram_rank = None
        self._keys = set(self.__dict__).union({
            "constraint", "constraint_jacobian", "constraint_coordinates",
            "constraint_jvp", "constraint_vjp",
            "constraint_jacobian_row_norms",
            "constraint_weighted_gram_diagonal",
            "constraint_hessian_vector",
            "constraint_tol", "constraint_diis_weight", "constraint_max_step",
            "constraint_krylov_tol", "constraint_krylov_max_cycle",
            "constraint_dense_limit", "constraint_consistency_tol",
            "constraint_value", "constraint_multipliers", "constraint_rank",
            "constraint_damping", "constraint_gram_rank",
            "constraint_gram_condition", "constraint_consistency_error",
            "constraint_krylov_info", "constraint_krylov_iterations",
            "constraint_krylov_residual",
            "constraint_linearized_residual", "constraint_diis_error",
            "constraint_time", "constraint_ncall",
        })

    @classmethod
    def from_constraint_engine(cls, mf, engine, **kwargs):
        kwargs.setdefault("constraint_jvp", engine.jvp)
        kwargs.setdefault("constraint_vjp", engine.vjp)
        kwargs.setdefault("constraint_jacobian_row_norms",
                          engine.jacobian_row_norms)
        kwargs.setdefault("constraint_weighted_gram_diagonal",
                          engine.weighted_gram_diagonal)
        kwargs.setdefault("constraint_hessian_vector", engine.hessian_vector)
        return cls(mf, engine.constraint, engine.jacobian, engine.coordinates,
                   **kwargs)

    def dump_flags(self, verbose=None):
        gccsd.GCCSD.dump_flags(self, verbose=verbose)
        log = logger.new_logger(self, verbose)
        log.info("Vector constraint tolerance = %.3g", self.constraint_tol)
        log.info("Constraint DIIS weight = %.3g", self.constraint_diis_weight)
        log.info("Constraint amplitude coordinates = %d",
                 len(self.constraint_coordinates))
        log.info("Constraint trust radius max|dT| = %s",
                 "%.3g" % self.constraint_max_step
                 if self.constraint_max_step > 0 else "disabled")
        log.info("Constraint dense-Jacobian limit = %.3g GiB",
                 self.constraint_dense_limit / 1024 ** 3)
        log.info("Constraint DIIS scheme = %s", self.constraint_diis_scheme)
        return self

    # ------------------------------------------------- constraint callbacks

    def _constraint_value(self, t1, t2):
        value = numpy.asarray(
            timed_constraint(self, "value", self.constraint, t1, t2)
        )
        if value.ndim != 1 or value.size == 0:
            raise ValueError(
                "constraint must return a nonempty one-dimensional array, got %s"
                % (value.shape,)
            )
        if numpy.iscomplexobj(value):
            raise ValueError("constraint must be real")
        value = numpy.asarray(value, dtype=float)
        if not numpy.all(numpy.isfinite(value)):
            raise ValueError("constraint returned non-finite values")
        if self._constraint_size is None:
            self._constraint_size = value.size
            nvec = len(self.constraint_coordinates)
            self._dense_gram = (
                value.size * nvec * 8 <= self.constraint_dense_limit
            )
            if self.constraint_krylov_max_cycle is None:
                self.constraint_krylov_max_cycle = max(100, 2 * value.size)
            logger.info(
                self,
                "Constraint: %d rows, %d amplitude coordinates, "
                "multiplier solve = %s",
                value.size, nvec,
                "dense Gram" if self._dense_gram else "matrix-free CG",
            )
        elif value.size != self._constraint_size:
            raise ValueError("constraint size changed between calls")
        return value

    def _constraint_jacobian(self, t1, t2):
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
        if not numpy.all(numpy.isfinite(jacobian)):
            raise ValueError("constraint_jacobian contains non-finite values")
        return value, numpy.asarray(jacobian, dtype=float)

    def _constraint_jvp(self, t1, t2, vector):
        result = numpy.asarray(timed_constraint(
            self, "jvp", self.constraint_jvp, t1, t2, vector
        ), dtype=float)
        if result.shape != (self._constraint_size,):
            raise ValueError("constraint JVP returned an invalid shape")
        return result

    def _constraint_vjp(self, t1, t2, weights):
        weights = numpy.asarray(weights, dtype=float)
        if weights.shape != (self._constraint_size,):
            raise ValueError("constraint VJP weights have an invalid shape")
        result = numpy.asarray(timed_constraint(
            self, "vjp", self.constraint_vjp, t1, t2, weights
        ), dtype=float)
        if result.shape != (len(self.constraint_coordinates),):
            raise ValueError("constraint VJP returned an invalid shape")
        return result

    def _constraint_row_norms(self, t1, t2):
        norms = numpy.asarray(timed_constraint(
            self, "row_norms", self.constraint_jacobian_row_norms, t1, t2
        ), dtype=float)
        if norms.shape != (self._constraint_size,):
            raise ValueError("constraint row norms have an invalid shape")
        return norms

    def _constraint_gram_diagonal(self, t1, t2, metric):
        diagonal = numpy.asarray(timed_constraint(
            self, "gram_diagonal", self.constraint_weighted_gram_diagonal,
            t1, t2, metric,
        ), dtype=float)
        if diagonal.shape != (self._constraint_size,):
            raise ValueError("constraint Gram diagonal has an invalid shape")
        return diagonal

    def _get_constraint_metric(self, t1, t2, eris):
        mo_energy = numpy.asarray(eris.mo_energy)
        reusable = (
            self._constraint_metric_cache is not None
            and self._constraint_metric_eris is eris
            and self._constraint_metric_level_shift == self.level_shift
            and self._constraint_metric_mo_energy.shape == mo_energy.shape
            and numpy.array_equal(self._constraint_metric_mo_energy, mo_energy)
        )
        if not reusable:
            denominators = _checked_denominators(self, t1, t2, eris)
            self._constraint_metric_cache = -1.0 / denominators
            self._constraint_metric_eris = eris
            self._constraint_metric_level_shift = self.level_shift
            self._constraint_metric_mo_energy = numpy.array(
                mo_energy, copy=True
            )
        return self._constraint_metric_cache

    def constraint_residual(self, t1=None, t2=None):
        if t1 is None:
            t1 = self.t1
        if t2 is None:
            t2 = self.t2
        return self._constraint_value(t1, t2)

    # ------------------------------------------------------------ iterations

    def ccsd(self, t1=None, t2=None, eris=None, mbpt2=False):
        if mbpt2:
            raise NotImplementedError(
                "the MBPT2 shortcut skips the amplitude equations, so the "
                "constraints would never be imposed"
            )
        configure_constraint_timing(self)
        self._constraint_cycle = 0
        self._constraint_metric_cache = None
        self._constraint_metric_eris = None
        self._constraint_metric_level_shift = None
        self._constraint_metric_mo_energy = None
        self.constraint_multipliers = None
        return gccsd.GCCSD.ccsd(self, t1=t1, t2=t2, eris=eris)

    def update_amps(self, t1, t2, eris):
        t1base, t2base = gccsd.update_amps(self, t1, t2, eris)
        old = self.amplitudes_to_vector(t1, t2)
        base = self.amplitudes_to_vector(t1base, t2base)
        metric = self._get_constraint_metric(t1, t2, eris)

        # The dense-or-matrix-free choice is made inside the first constraint
        # evaluation; force it before branching on it, or the first cycle
        # always runs matrix-free whatever the decision comes out to be.
        if self._dense_gram is None:
            self._constraint_value(t1, t2)
        # ``rhs`` is the predicted post-step violation.  When it is negligible
        # against the tolerance -- a reference that is already a spin
        # eigenstate keeps it at rounding noise for every cycle -- the solve
        # would invert that noise through the near-null directions of the
        # rank-deficient Gram and inject garbage corrections, so it is skipped
        # rather than solved.
        rhs_floor = 1e-2 * self.constraint_tol

        if self._dense_gram:
            constraint, jacobian = self._constraint_jacobian(t1, t2)
            rhs = -(constraint + jacobian @ (base - old))
            used_zero_correction = (
                float(numpy.max(numpy.abs(rhs), initial=0.0)) <= rhs_floor
            )
            if used_zero_correction:
                multipliers = numpy.zeros(constraint.size)
                correction = numpy.zeros_like(base)
            elif self.constraint_max_step > 0:
                response = jacobian.T * metric[:, None]
                gram = jacobian @ response
                multipliers, correction = damped_multipliers(
                    self, gram, response, rhs, constraint.size
                )
            else:
                response = jacobian.T * metric[:, None]
                gram = jacobian @ response
                multipliers, correction = _minimum_norm_multipliers(
                    self, gram, response, rhs, constraint.size
                )
            # the metric is positive, so the Gram and the Jacobian share a rank
            # and the eigendecomposition already done is enough -- no extra SVD
            self.constraint_rank = self.constraint_gram_rank
        else:
            constraint = self._constraint_value(t1, t2)

            def response(weights):
                return metric * self._constraint_vjp(t1, t2, weights)

            def gram_matvec(weights):
                return self._constraint_jvp(t1, t2, response(weights))

            rhs = -(constraint + self._constraint_jvp(t1, t2, base - old))
            used_zero_correction = (
                float(numpy.max(numpy.abs(rhs), initial=0.0)) <= rhs_floor
            )
            if used_zero_correction:
                multipliers = numpy.zeros(constraint.size)
                correction = numpy.zeros_like(base)
            else:
                diagonal = self._constraint_gram_diagonal(t1, t2, metric)
                multipliers, correction = matrix_free_multipliers(
                    self, gram_matvec, response, rhs, diagonal,
                    initial=self.constraint_multipliers,
                )
            self.constraint_rank = None

        if self.constraint_diis_scheme == "tangent":
            # See MultiConstrUCCSD.update_amps (uccsd_multiconstr.py) for
            # why this must be J . residual_selected (matching `rhs` above
            # minus its -C term), not response.T/response_transpose, which
            # would double-apply the metric already baked into
            # residual_selected = base-old, and why r_par adds response(eta_LS)
            # rather than subtracting it (new = base + response(multipliers),
            # so at a fixed point residual_selected = -response(eta*) -- a
            # PLUS is what makes r_par vanish there; the used_zero_correction
            # branch below is this same "+" with eta_LS implicitly 0).
            residual_selected = base - old
            if used_zero_correction:
                self._constraint_tangent_residual = residual_selected
            elif self._dense_gram:
                rhs_ls = -(jacobian @ residual_selected)
                eta_ls = tangent_least_squares_dense(gram, rhs_ls)
                self._constraint_tangent_residual = (
                    residual_selected + response @ eta_ls
                )
            else:
                rhs_ls = -self._constraint_jvp(t1, t2, residual_selected)
                eta_ls = tangent_least_squares_matrix_free(
                    self, gram_matvec, diagonal, rhs_ls, initial=multipliers
                )
                self._constraint_tangent_residual = (
                    residual_selected + response(eta_ls)
                )

        new = base + correction
        linearized = constraint + self._constraint_jvp(t1, t2, new - old)
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
            if self.constraint_diis_scheme == "tangent":
                if self._constraint_old_vector is None:
                    error = numpy.zeros_like(vector)
                else:
                    error = self._constraint_tangent_residual
                    if error is None:
                        error = vector - self._constraint_old_vector
                self.constraint_diis_error = None
            else:
                if self._constraint_old_vector is None:
                    amplitude_error = numpy.zeros_like(vector)
                else:
                    amplitude_error = vector - self._constraint_old_vector
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
                        maximum, self.constraint_tol,
                    )
                self.converged = False
            logger.note(
                self,
                "Constraint max|C(t)| = %.12g  rank = %s of %d  "
                "out-of-range rhs = %s",
                maximum,
                self.constraint_gram_rank
                if self.constraint_gram_rank is not None else "n/a",
                self._constraint_size,
                ("%.3e" % self.constraint_consistency_error
                 if self.constraint_consistency_error is not None else "n/a"),
            )
        dump_constraint_timing(self)
        return gccsd.GCCSD._finalize(self)

    def check_constraint_jacobian(self, t1=None, t2=None, step=1e-6, ndir=3,
                                  seed=1, tolerance=1e-6):
        """Central differences of ``constraint`` against the analytic JVP."""
        if t1 is None or t2 is None:
            if self.t1 is not None and self.t2 is not None:
                t1, t2 = self.t1, self.t2
            else:
                t1, t2 = self.get_init_guess()
        vector = self.amplitudes_to_vector(t1, t2)
        self._constraint_value(t1, t2)
        rng = numpy.random.default_rng(seed)
        maximum = 0.0
        for _ in range(ndir):
            direction = rng.standard_normal(vector.size)
            direction /= numpy.linalg.norm(direction)
            plus = self.vector_to_amplitudes(vector + step * direction)
            minus = self.vector_to_amplitudes(vector - step * direction)
            difference = (self._constraint_value(*plus)
                          - self._constraint_value(*minus)) / (2.0 * step)
            analytic = self._constraint_jvp(t1, t2, direction)
            scale = numpy.maximum(1.0, numpy.maximum(
                numpy.abs(difference), numpy.abs(analytic)
            ))
            maximum = max(maximum, float(
                numpy.max(numpy.abs(difference - analytic) / scale)
            ))
        if maximum > tolerance:
            raise ValueError(
                "Constraint Jacobian finite-difference error %.6e exceeds %.6e"
                % (maximum, tolerance)
            )
        return maximum

    def solve_lambda(self, t1=None, t2=None, l1=None, l2=None, eris=None):
        raise NotImplementedError(
            "the constrained Lagrangian gains a J^T lambda term, so the "
            "ordinary GCCSD lambda equations do not apply; only the constrained "
            "UCCSD solver derives them (ccc.MultiConstrUCCSD.solve_lambda)"
        )

    def make_rdm1(self, *args, **kwargs):
        raise NotImplementedError(
            "response densities need the constrained lambda equations"
        )

    def make_rdm2(self, *args, **kwargs):
        raise NotImplementedError(
            "response densities need the constrained lambda equations"
        )

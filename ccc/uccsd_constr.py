import numpy

from pyscf import lib
from pyscf.cc import uccsd
from pyscf.lib import logger


def _amplitude_denominators(cc, t1, t2, eris):
    nocca, nvira = t1[0].shape
    noccb, nvirb = t1[1].shape
    eia = lib.direct_sum(
        "i-a->ia",
        eris.mo_energy[0][:nocca],
        eris.mo_energy[0][nocca:] + cc.level_shift,
    )
    eIA = lib.direct_sum(
        "i-a->ia",
        eris.mo_energy[1][:noccb],
        eris.mo_energy[1][noccb:] + cc.level_shift,
    )
    d2aa = lib.direct_sum("ia+jb->ijab", eia, eia)
    d2ab = lib.direct_sum("ia+jb->ijab", eia, eIA)
    d2bb = lib.direct_sum("ia+jb->ijab", eIA, eIA)
    return cc.amplitudes_to_vector((eia, eIA), (d2aa, d2ab, d2bb))


def _checked_denominators(cc, t1, t2, eris):
    denominators = _amplitude_denominators(cc, t1, t2, eris)
    scale = max(1.0, float(numpy.max(numpy.abs(denominators))))
    threshold = numpy.finfo(denominators.dtype).eps * scale * 10.0
    if numpy.any(numpy.abs(denominators) <= threshold):
        smallest = float(numpy.min(numpy.abs(denominators)))
        raise RuntimeError("Near-zero CC amplitude denominator (min |D| = %.6e)" % smallest)
    return denominators


class ConstrUCCSD(uccsd.UCCSD):
    """UCCSD with one user-defined real scalar amplitude constraint.

    ``constraint(t1, t2)`` returns the scalar C(t).  The corresponding
    ``constraint_gradient(t1, t2)`` returns unrestricted amplitude-shaped
    tuples ``((g1a, g1b), (g2aa, g2ab, g2bb))``.  Gradients are interpreted in
    PySCF's packed independent-amplitude coordinates.
    """

    def __init__(self, mf, constraint, constraint_gradient,
                 constraint_tol=1e-10, constraint_diis_weight=1.0,
                 frozen=None, mo_coeff=None, mo_occ=None):
        uccsd.UCCSD.__init__(self, mf, frozen=frozen, mo_coeff=mo_coeff, mo_occ=mo_occ)
        if not callable(constraint):
            raise TypeError("constraint must be callable")
        if not callable(constraint_gradient):
            raise TypeError("constraint_gradient must be callable")
        if constraint_tol <= 0:
            raise ValueError("constraint_tol must be positive")
        if constraint_diis_weight <= 0:
            raise ValueError("constraint_diis_weight must be positive")

        keys = {
            "constraint",
            "constraint_gradient",
            "constraint_tol",
            "constraint_diis_weight",
            "constraint_value",
            "constraint_multiplier",
            "constraint_gradient_norm",
            "constraint_multiplier_denominator",
            "constraint_linearized_residual",
            "constraint_diis_error",
        }
        self._keys = set(self.__dict__).union(keys)
        self.constraint = constraint
        self.constraint_gradient = constraint_gradient
        self.constraint_tol = float(constraint_tol)
        self.constraint_diis_weight = float(constraint_diis_weight)
        self.constraint_value = None
        self.constraint_multiplier = None
        self.constraint_gradient_norm = None
        self.constraint_multiplier_denominator = None
        self.constraint_linearized_residual = None
        self.constraint_diis_error = None
        self._constraint_old_vector = None

    def dump_flags(self, verbose=None):
        uccsd.UCCSD.dump_flags(self, verbose=verbose)
        log = logger.new_logger(self, verbose)
        log.info("Nonlinear amplitude constraint tolerance = %.3g", self.constraint_tol)
        log.info("Constraint DIIS weight = %.3g", self.constraint_diis_weight)
        return self

    def _constraint_value(self, t1, t2):
        value = numpy.asarray(self.constraint(t1, t2))
        if value.shape != ():
            raise ValueError("constraint must return one scalar, got shape %s" % (value.shape,))
        if numpy.iscomplexobj(value) and abs(value.imag) > 1e-12:
            raise ValueError("constraint must be real")
        value = float(value.real)
        if not numpy.isfinite(value):
            raise ValueError("constraint returned a non-finite value")
        return value

    def _constraint_gradient(self, t1, t2):
        try:
            g1, g2 = self.constraint_gradient(t1, t2)
            g1 = tuple(g1)
            g2 = tuple(g2)
        except Exception as err:
            raise ValueError("constraint_gradient must return (g1, g2)") from err
        if len(g1) != 2 or len(g2) != 3:
            raise ValueError("constraint_gradient must return two T1 and three T2 blocks")

        expected = t1 + t2
        supplied = tuple(g1) + tuple(g2)
        checked = []
        for n, (gradient, amplitude) in enumerate(zip(supplied, expected)):
            gradient = numpy.asarray(gradient)
            if gradient.shape != amplitude.shape:
                raise ValueError(
                    "constraint gradient block %d has shape %s; expected %s"
                    % (n, gradient.shape, amplitude.shape)
                )
            if numpy.iscomplexobj(gradient) and numpy.any(numpy.abs(gradient.imag) > 1e-12):
                raise ValueError("constraint gradient must be real")
            if not numpy.all(numpy.isfinite(gradient)):
                raise ValueError("constraint gradient contains non-finite values")
            checked.append(numpy.asarray(gradient.real, dtype=amplitude.dtype))

        g1 = tuple(checked[:2])
        g2 = tuple(checked[2:])
        vector = self.amplitudes_to_vector(g1, g2)
        norm = float(numpy.linalg.norm(vector))
        if norm == 0.0 or not numpy.isfinite(norm):
            raise RuntimeError("Constraint gradient is zero or numerically singular")
        self.constraint_gradient_norm = norm
        return g1, g2, vector

    def constraint_residual(self, t1=None, t2=None):
        if t1 is None:
            t1 = self.t1
        if t2 is None:
            t2 = self.t2
        return self._constraint_value(t1, t2)

    def update_amps(self, t1, t2, eris):
        t1base, t2base = uccsd.update_amps(self, t1, t2, eris)
        old = self.amplitudes_to_vector(t1, t2)
        base = self.amplitudes_to_vector(t1base, t2base)
        constraint = self._constraint_value(t1, t2)
        _, _, gradient = self._constraint_gradient(t1, t2)
        denominators = _checked_denominators(self, t1, t2, eris)
        response = -gradient / denominators
        multiplier_denominator = float(numpy.dot(gradient, response))
        denominator_scale = numpy.linalg.norm(gradient) * numpy.linalg.norm(response)
        if (not numpy.isfinite(multiplier_denominator) or
                abs(multiplier_denominator) <= numpy.finfo(float).eps * denominator_scale * 10.0):
            raise RuntimeError("Constraint multiplier equation is singular")

        numerator = constraint + numpy.dot(gradient, base - old)
        multiplier = -float(numerator) / multiplier_denominator
        new = base + multiplier * response
        linearized = constraint + numpy.dot(gradient, new - old)

        self.constraint_multiplier = multiplier
        self.constraint_multiplier_denominator = multiplier_denominator
        self.constraint_linearized_residual = float(linearized)
        self._constraint_old_vector = old.copy()
        t1new, t2new = self.vector_to_amplitudes(new)
        self.constraint_value = self._constraint_value(t1new, t2new)
        return t1new, t2new

    def run_diis(self, t1, t2, istep, normt, de, adiis):
        self.constraint_value = self._constraint_value(t1, t2)
        if (adiis and istep >= self.diis_start_cycle and
                abs(de) < self.diis_start_energy_diff):
            vector = self.amplitudes_to_vector(t1, t2)
            if self._constraint_old_vector is None:
                amplitude_error = numpy.zeros_like(vector)
            else:
                amplitude_error = vector - self._constraint_old_vector
            _, _, gradient = self._constraint_gradient(t1, t2)
            scaled_constraint = (
                self.constraint_diis_weight * self.constraint_value /
                self.constraint_gradient_norm
            )
            error = numpy.append(amplitude_error, scaled_constraint)
            vector = adiis.update(vector, xerr=error)
            t1, t2 = self.vector_to_amplitudes(vector)
            self.constraint_value = self._constraint_value(t1, t2)
            self.constraint_diis_error = float(scaled_constraint)
            logger.debug1(self, "Constrained DIIS for step %d", istep)
        return t1, t2

    def _finalize(self):
        if self.t1 is not None and self.t2 is not None:
            self.constraint_value = self._constraint_value(self.t1, self.t2)
            self._constraint_gradient(self.t1, self.t2)
            if abs(self.constraint_value) > self.constraint_tol:
                if self.converged:
                    logger.warn(
                        self,
                        "CC equations converged but |C(t)| = %.6e exceeds %.6e",
                        abs(self.constraint_value),
                        self.constraint_tol,
                    )
                self.converged = False
            logger.note(
                self,
                "Constraint C(t) = %.12g  multiplier = %s",
                self.constraint_value,
                self.constraint_multiplier,
            )
        return uccsd.UCCSD._finalize(self)

    def check_constraint_gradient(self, t1=None, t2=None, step=1e-6,
                                  ndir=3, seed=1, tolerance=1e-6):
        if t1 is None or t2 is None:
            if self.t1 is not None and self.t2 is not None:
                t1, t2 = self.t1, self.t2
            else:
                t1, t2 = self.get_init_guess()
        vector = self.amplitudes_to_vector(t1, t2)
        _, _, gradient = self._constraint_gradient(t1, t2)
        rng = numpy.random.default_rng(seed)
        errors = []
        for _ in range(ndir):
            direction = rng.standard_normal(vector.size)
            direction /= numpy.linalg.norm(direction)
            t1p, t2p = self.vector_to_amplitudes(vector + step * direction)
            t1m, t2m = self.vector_to_amplitudes(vector - step * direction)
            finite_difference = (
                self._constraint_value(t1p, t2p) - self._constraint_value(t1m, t2m)
            ) / (2.0 * step)
            analytic = float(numpy.dot(gradient, direction))
            scale = max(1.0, abs(finite_difference), abs(analytic))
            errors.append(abs(finite_difference - analytic) / scale)
        maximum = max(errors, default=0.0)
        if maximum > tolerance:
            raise ValueError(
                "Constraint gradient finite-difference error %.6e exceeds %.6e"
                % (maximum, tolerance)
            )
        return maximum

    def solve_lambda(self, *args, **kwargs):
        raise NotImplementedError(
            "Ordinary UCCSD lambda equations are not valid for ConstrUCCSD; "
            "a constraint-consistent left state requires second derivatives of C(t)."
        )

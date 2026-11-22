import itertools
import math

import numpy

from pyscf import lib
from pyscf.cc import uccsdt
from pyscf.lib import logger

from .active_spin_constraint import DominantSectorSpinConstraint
from .uccsd_multiconstr import _constraint_diis_error, damped_multipliers
from .uccsdt_multiconstr import (
    _canonical_group,
    _checked_coordinate_denominators,
    _combination_rank,
    _permutation_sign,
)


BLOCKS = (
    "t1a", "t1b", "t2aa", "t2ab", "t2bb",
    "t3aaa", "t3aab", "t3abb", "t3bbb",
)


def _residuals_simultaneous(cc, tamps, eris):
    """Build every triangular UCCSDT Jacobi increment at the same T."""
    t1, t2, t3 = tamps
    imds = uccsdt._IMDS()

    uccsdt.update_t1_fock_eris_uhf(cc, imds, t1, eris)
    uccsdt.intermediates_t1t2_uhf(cc, imds, t2)
    r1, r2 = uccsdt.compute_r1r2_uhf(cc, imds, t2)
    uccsdt.r1r2_add_t3_tri_uhf_(cc, imds, r1, r2, t3)
    uccsdt.antisymmetrize_r2_uhf_(r2)
    uccsdt.r1r2_divide_e_uhf_(cc, r1, r2, eris.mo_energy[:])

    # A constrained fixed point may have nonzero selected T1/T2 residuals.
    # T3 must therefore be evaluated at the same old T2, not an interim update.
    uccsdt.intermediates_t3_uhf(cc, imds, t2)
    uccsdt.intermediates_t3_add_t3_tri_uhf(cc, imds, t3)
    imds.t1_fock, imds.t1_eris = None, None
    r3 = uccsdt.compute_r3_tri_uhf(cc, imds, t2, t3)
    imds = None
    uccsdt.r3_tri_divide_e_uhf_(cc, r3, eris.mo_energy[:])
    return [r1, r2, r3]


class LowMemorySpinConstrUCCSDT(uccsdt.UCCSDT):
    """Triangular-storage UCCSDT with dominant-sector spin constraints."""

    do_diis_max_t = True

    def __init__(self, mf, spin_constraint, constraint_tol=1e-10,
                 constraint_diis_weight=1.0, constraint_max_step=0.0,
                 frozen=None, mo_coeff=None, mo_occ=None,
                 conv_check=True):
        if not isinstance(spin_constraint, DominantSectorSpinConstraint):
            raise TypeError(
                "spin_constraint must be a DominantSectorSpinConstraint"
            )
        if spin_constraint.max_cluster_rank != 3:
            raise ValueError("UCCSDT spin constraints require max_cluster_rank=3")
        if not spin_constraint.nconstraint:
            raise ValueError("spin_constraint has no nontrivial constraints")
        if constraint_tol <= 0:
            raise ValueError("constraint_tol must be positive")
        if constraint_diis_weight <= 0:
            raise ValueError("constraint_diis_weight must be positive")
        if constraint_max_step < 0:
            raise ValueError("constraint_max_step must be nonnegative")

        uccsdt.UCCSDT.__init__(
            self, mf, frozen=frozen, mo_coeff=mo_coeff, mo_occ=mo_occ
        )
        self.spin_constraint = spin_constraint
        self.constraint_coordinates = spin_constraint.coordinates
        self.constraint_tol = float(constraint_tol)
        self.constraint_diis_weight = float(constraint_diis_weight)
        self.constraint_max_step = float(constraint_max_step)
        self.conv_check = bool(conv_check)
        self.constraint_value = None
        self.constraint_multipliers = None
        self.constraint_rank = None
        self.constraint_damping = 0.0
        self.constraint_gram_rank = None
        self.constraint_gram_condition = None
        self._reported_gram_rank = None
        self.constraint_linearized_residual = None
        self.constraint_diis_error = None
        self._constraint_update_vector = None
        self._constraint_amplitude_size = None
        self._constraint_iteration = 0
        self._coordinate_shapes = None
        self._coordinate_locations = None
        self._keys = set(self._keys).union({
            "conv_check",
            "spin_constraint", "constraint_coordinates", "constraint_tol",
            "constraint_diis_weight", "constraint_max_step",
            "constraint_value", "constraint_multipliers", "constraint_rank",
            "constraint_damping", "constraint_gram_rank",
            "constraint_gram_condition", "constraint_linearized_residual",
            "constraint_diis_error",
        })

    def dump_flags(self, verbose=None):
        uccsdt.UCCSDT.dump_flags(self, verbose=verbose)
        log = logger.new_logger(self, verbose)
        log.info("Dominant-sector constraints = %d",
                 self.spin_constraint.nconstraint)
        log.info("Constraint amplitude coordinates = %d",
                 len(self.constraint_coordinates))
        log.info("Constraint tolerance = %.3g", self.constraint_tol)
        log.info("Constraint DIIS weight = %.3g",
                 self.constraint_diis_weight)
        log.info("Constraint trust radius max|dT| = %s",
                 "%.3g" % self.constraint_max_step
                 if self.constraint_max_step > 0 else "disabled")
        if not self.do_diis_max_t:
            raise ValueError(
                "LowMemorySpinConstrUCCSDT requires do_diis_max_t=True"
            )
        if self.einsum_backend == "pytblis":
            try:
                import pytblis  # noqa: F401
            except ImportError:
                raise ImportError(
                    "einsum_backend='pytblis' requested, but pytblis is not "
                    "installed; PySCF would silently fall back to NumPy"
                )
        return self

    def memory_estimate_log(self):
        uccsdt.memory_estimate_log_uccsdt(self)
        log = logger.new_logger(self)
        nbytes = self.vector_size() * numpy.dtype(float).itemsize
        if self.do_diis_max_t:
            disk = 2 * self.diis_space * nbytes
            transient = 4 * nbytes
            log.info("Constrained DIIS packed vector    %8s",
                     uccsdt.format_size(nbytes))
            if self.incore_complete:
                log.info("Constrained DIIS in-core history %8s",
                         uccsdt.format_size(disk))
            else:
                log.info("Constrained DIIS disk history    %8s",
                         uccsdt.format_size(disk))
            log.info("Constrained DIIS transient bound %8s",
                     uccsdt.format_size(transient))
        return self

    @staticmethod
    def _set_antisymmetric(array, index, value, groups):
        assignments = [(list(index), 1.0)]
        for positions in groups:
            expanded = []
            for permutation in itertools.permutations(range(len(positions))):
                sign = _permutation_sign(permutation)
                for target, old_sign in assignments:
                    updated = target.copy()
                    for destination, source in zip(positions, permutation):
                        updated[destination] = index[positions[source]]
                    expanded.append((updated, old_sign * sign))
            assignments = expanded
        for target, sign in assignments:
            array[tuple(target)] = sign * value

    def _coordinate_location(self, block, index, t1):
        nocca, nvira = t1[0].shape
        noccb, nvirb = t1[1].shape
        index = tuple(int(x) for x in index)
        sign = 1.0

        if block == "t1a":
            location = (0, 0, index)
        elif block == "t1b":
            location = (0, 1, index)
        elif block == "t2aa":
            location = (1, 0, index)
        elif block == "t2ab":
            i, j, a, b = index
            location = (1, 1, (i, a, j, b))
        elif block == "t2bb":
            location = (1, 2, index)
        elif block in ("t3aaa", "t3bbb"):
            nocc, nvir, iblock = (
                (nocca, nvira, 0) if block == "t3aaa"
                else (noccb, nvirb, 3)
            )
            occupied, so = _canonical_group(
                index[:3], nocc, block, "occupied"
            )
            virtual, sv = _canonical_group(
                index[3:], nvir, block, "virtual"
            )
            location = (
                2,
                iblock,
                (_combination_rank(occupied, nocc),
                 _combination_rank(virtual, nvir)),
            )
            sign = so * sv
        elif block == "t3aab":
            i, j, k, a, b, c = index
            occupied, so = _canonical_group(
                (i, j), nocca, block, "occupied"
            )
            virtual, sv = _canonical_group(
                (a, b), nvira, block, "virtual"
            )
            location = (
                2,
                1,
                (_combination_rank(occupied, nocca),
                 _combination_rank(virtual, nvira), k, c),
            )
            sign = so * sv
        elif block == "t3abb":
            i, j, k, a, b, c = index
            occupied, so = _canonical_group(
                (j, k), noccb, block, "occupied"
            )
            virtual, sv = _canonical_group(
                (b, c), nvirb, block, "virtual"
            )
            location = (
                2,
                2,
                (_combination_rank(occupied, noccb),
                 _combination_rank(virtual, nvirb), i, a),
            )
            sign = so * sv
        else:
            raise ValueError("Unknown amplitude block %r" % block)

        try:
            numpy.ravel_multi_index(location[2], self._location_shape(
                location[0], location[1], t1
            ))
        except ValueError as error:
            raise IndexError(
                "%s coordinate %r is out of range" % (block, index)
            ) from error
        return location + (sign,)

    def _location_shape(self, rank, block, t1):
        nocca, nvira = t1[0].shape
        noccb, nvirb = t1[1].shape
        if rank == 0:
            return ((nocca, nvira), (noccb, nvirb))[block]
        if rank == 1:
            return (
                (nocca, nocca, nvira, nvira),
                (nocca, nvira, noccb, nvirb),
                (noccb, noccb, nvirb, nvirb),
            )[block]
        return (
            (math.comb(nocca, 3), math.comb(nvira, 3)),
            (math.comb(nocca, 2), math.comb(nvira, 2), noccb, nvirb),
            (math.comb(noccb, 2), math.comb(nvirb, 2), nocca, nvira),
            (math.comb(noccb, 3), math.comb(nvirb, 3)),
        )[block]

    def _ensure_coordinates(self, tamps):
        shapes = tuple(block.shape for rank in tamps for block in rank)
        if shapes == self._coordinate_shapes:
            return
        locations = tuple(
            self._coordinate_location(block, index, tamps[0])
            for block, index in self.constraint_coordinates
        )
        independent = tuple((rank, block, index) for rank, block, index, _ in locations)
        if len(set(independent)) != len(independent):
            raise ValueError("constraint coordinates contain duplicates")
        self._coordinate_shapes = shapes
        self._coordinate_locations = locations

    def _coordinate_values(self, tamps):
        self._ensure_coordinates(tamps)
        return numpy.asarray([
            sign * tamps[rank][block][index]
            for rank, block, index, sign in self._coordinate_locations
        ])

    def _set_coordinate_values(self, tamps, values):
        self._ensure_coordinates(tamps)
        groups = {
            "t2aa": ((0, 1), (2, 3)),
            "t2bb": ((0, 1), (2, 3)),
        }
        for coordinate, location, value in zip(
                self.constraint_coordinates, self._coordinate_locations, values):
            name, public_index = coordinate
            rank, block, native_index, sign = location
            array = tamps[rank][block]
            if name in groups:
                self._set_antisymmetric(
                    array, public_index, value, groups[name]
                )
            else:
                array[native_index] = sign * value

    def _constraint_jacobian(self, values):
        constraint = numpy.asarray(
            self.spin_constraint.constraint_from_values(values), dtype=float
        )
        jacobian = numpy.asarray(
            self.spin_constraint.jacobian_from_values(values), dtype=float
        )
        expected = (constraint.size, len(self.constraint_coordinates))
        if constraint.shape != (self.spin_constraint.nconstraint,):
            raise ValueError("spin constraint returned an invalid shape")
        if jacobian.shape != expected:
            raise ValueError(
                "spin Jacobian has shape %s; expected %s"
                % (jacobian.shape, expected)
            )
        if not numpy.all(numpy.isfinite(constraint)):
            raise ValueError("spin constraint contains non-finite values")
        if not numpy.all(numpy.isfinite(jacobian)):
            raise ValueError("spin Jacobian contains non-finite values")
        rank = int(numpy.linalg.matrix_rank(jacobian))
        if rank != constraint.size:
            raise RuntimeError(
                "Constraint Jacobian is rank deficient (rank %d of %d)"
                % (rank, constraint.size)
            )
        self.constraint_rank = rank
        return constraint, jacobian

    def _pack_with_constraint_tail(self, tamps, ntail):
        if self.unique_tamps_map is None:
            self.unique_tamps_map = self.build_unique_tamps_map()
        amplitude_size = self.vector_size()
        out = numpy.empty(amplitude_size + ntail, dtype=tamps[0][0].dtype)
        position = 0
        for irank, rank in enumerate(tamps):
            for iblock, block in enumerate(rank):
                selected = block[self.unique_tamps_map[irank][iblock]].ravel()
                out[position:position + selected.size] = selected
                position += selected.size
        if position != amplitude_size:
            raise RuntimeError("packed UCCSDT amplitude size mismatch")
        return out, amplitude_size

    def constraint_residual(self, tamps=None):
        if tamps is None:
            if any(rank is None for rank in self.tamps):
                raise ValueError("UCCSDT amplitudes are not initialized")
            tamps = self.tamps
        return self.spin_constraint.constraint_from_values(
            self._coordinate_values(tamps)
        )

    def sector_spin_square(self, tamps=None):
        if tamps is None:
            tamps = self.tamps
        return self.spin_constraint.sector_spin_square_from_values(
            self._coordinate_values(tamps)
        )

    def update_amps_(self, tamps, eris):
        old = self._coordinate_values(tamps)
        constraint, jacobian = self._constraint_jacobian(old)
        residuals = _residuals_simultaneous(self, tamps, eris)
        base = old + self._coordinate_values(residuals)

        denominators = _checked_coordinate_denominators(self, eris)
        response = -jacobian.T / denominators[:, None]
        gram = jacobian @ response
        rhs = -(constraint + jacobian @ (base - old))
        multipliers, correction = damped_multipliers(
            self, gram, response, rhs, constraint.size
        )
        new = base + correction
        linearized = constraint + jacobian @ (new - old)
        self._set_coordinate_values(residuals, new - old)

        norms = []
        for rank, residual_rank in zip(tamps, residuals):
            for block, residual in zip(rank, residual_rank):
                norms.append(numpy.linalg.norm(residual))
                block += residual

        if (self.diis and self.do_diis_max_t
                and self._constraint_iteration >= self.diis_start_cycle):
            error, amplitude_size = self._pack_with_constraint_tail(
                residuals, constraint.size
            )
            self._constraint_update_vector = error
            self._constraint_amplitude_size = amplitude_size
        else:
            self._constraint_update_vector = None
            self._constraint_amplitude_size = None
        self._constraint_iteration += 1

        self.constraint_multipliers = multipliers
        self.constraint_linearized_residual = linearized
        values = self._coordinate_values(tamps)
        self.constraint_value = self.spin_constraint.constraint_from_values(values)
        return norms

    def run_diis(self, tamps, istep, normt, de, adiis):
        values = self._coordinate_values(tamps)
        self.constraint_value = self.spin_constraint.constraint_from_values(values)
        if (adiis and istep >= self.diis_start_cycle
                and abs(de) < self.diis_start_energy_diff):
            error = self._constraint_update_vector
            amplitude_size = self._constraint_amplitude_size
            if error is None or amplitude_size is None:
                raise RuntimeError("Missing constrained UCCSDT DIIS error")
            _, jacobian = self._constraint_jacobian(values)
            row_norms = numpy.linalg.norm(jacobian, axis=1)
            scaled = _constraint_diis_error(
                self, self.constraint_value, row_norms
            )
            error[amplitude_size:] = scaled
            vector = self.amplitudes_to_vector(tamps)
            self._constraint_update_vector = None
            self._constraint_amplitude_size = None
            vector = adiis.update(vector, xerr=error)
            del error
            tamps = self.vector_to_amplitudes(vector)
            self.constraint_value = self.constraint_residual(tamps)
            self.constraint_diis_error = scaled
            logger.debug1(self, "Spin-constrained DIIS for step %d", istep)
        else:
            self._constraint_update_vector = None
            self._constraint_amplitude_size = None
        return tamps

    def ccsdt(self, tamps=None, eris=None):
        self._constraint_iteration = 0
        self._constraint_update_vector = None
        self._constraint_amplitude_size = None
        self.constraint_multipliers = None
        return uccsdt.UCCSDT.ccsdt(self, tamps=tamps, eris=eris)

    def _finalize(self):
        if not any(rank is None for rank in self.tamps):
            values = self._coordinate_values(self.tamps)
            constraint_value, _ = self._constraint_jacobian(values)
            if (self.conv_check and self.converged
                    and float(numpy.max(numpy.abs(constraint_value))) > self.constraint_tol):
                # One extra Newton/eta correction pass, no DIIS after it --
                # the main loop's convergence test uses the pre-DIIS residual
                # norm, so it can exit leaving the true (post-DIIS) tamps with
                # an unresolved constraint violation that nothing else
                # corrects.  Bounded to exactly one pass (never a retry loop):
                # if it doesn't resolve the constraint, or moves the energy/
                # amplitudes far more than a small cleanup step should, flag
                # not converged rather than silently accepting the result.
                eris = self.ao2mo(self.mo_coeff)
                e_before = self.e_corr
                res_norm = self.update_amps_(self.tamps, eris)
                normt_extra = float(numpy.linalg.norm(res_norm))
                self.e_corr = self.energy(self.tamps, eris)
                values_after = self._coordinate_values(self.tamps)
                constraint_after, _ = self._constraint_jacobian(values_after)
                maximum_after = float(numpy.max(numpy.abs(constraint_after)))
                de_extra = abs(self.e_corr - e_before)
                if (maximum_after > self.constraint_tol
                        or de_extra > 10 * self.conv_tol
                        or normt_extra > 10 * self.conv_tol_normt):
                    logger.warn(
                        self,
                        "Post-loop constraint resync did not settle cleanly "
                        "(max|C(t)| = %.6e, dE = %.6e, norm(d tamps) = %.6e) "
                        "-- flagging not converged",
                        maximum_after, de_extra, normt_extra,
                    )
                    self.converged = False
                values = self._coordinate_values(self.tamps)
            self.constraint_value, _ = self._constraint_jacobian(values)
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
            logger.note(
                self,
                "Constraint max|C(t)| = %.12g  rank = %d",
                maximum,
                self.constraint_rank,
            )
        return uccsdt.UCCSDT._finalize(self)

    def check_constraint_jacobian(self, tamps=None, step=1e-6,
                                  ndir=3, seed=1, tolerance=1e-6):
        if tamps is None:
            tamps = self.tamps if self.tamps[0] is not None else self.init_amps()[1]
        values = self._coordinate_values(tamps)
        _, jacobian = self._constraint_jacobian(values)
        rng = numpy.random.default_rng(seed)
        maximum = 0.0
        for _ in range(ndir):
            direction = rng.standard_normal(values.size)
            direction /= numpy.linalg.norm(direction)
            plus = self.spin_constraint.constraint_from_values(
                values + step * direction
            ).copy()
            minus = self.spin_constraint.constraint_from_values(
                values - step * direction
            ).copy()
            finite_difference = (plus - minus) / (2.0 * step)
            analytic = jacobian @ direction
            scale = numpy.maximum(1.0, numpy.maximum(
                numpy.abs(finite_difference), numpy.abs(analytic)
            ))
            maximum = max(maximum, float(numpy.max(
                numpy.abs(finite_difference - analytic) / scale
            )))
        if maximum > tolerance:
            raise ValueError(
                "Constraint Jacobian finite-difference error %.6e exceeds %.6e"
                % (maximum, tolerance)
            )
        return maximum

    def solve_lambda(self, *args, **kwargs):
        raise NotImplementedError(
            "Constraint-consistent UCCSDT lambda equations are not implemented"
        )

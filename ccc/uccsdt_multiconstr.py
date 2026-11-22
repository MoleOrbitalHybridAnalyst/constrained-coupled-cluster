import math
from itertools import permutations

import numpy

from pyscf import lib
from pyscf.cc import uccsdt as uccsdt_lowm
from pyscf.cc import uccsdt_highm
from pyscf.lib import logger

from .uccsd_multiconstr import (
    _constraint_diis_error,
    configure_constraint_timing,
    configure_diis_scheme,
    configure_multiplier_solver,
    damped_multipliers,
    dump_constraint_timing,
    log_constraint_timing,
    matrix_free_multipliers,
    nonlinear_multipliers,
    nonlinear_multipliers_matrix_free,
    tangent_least_squares_dense,
    tangent_least_squares_matrix_free,
    timed_constraint,
)


BLOCKS = (
    "t1a", "t1b", "t2aa", "t2ab", "t2bb",
    "t3aaa", "t3aab", "t3abb", "t3bbb",
)


def _public_amplitudes(tamps):
    """Return occupied-first, unrestricted amplitude views."""
    t1, t2, t3 = tamps
    t2_public = (t2[0], t2[1].transpose(0, 2, 1, 3), t2[2])
    t3_public = (
        t3[0],
        t3[1].transpose(0, 1, 4, 2, 3, 5),
        t3[2].transpose(4, 0, 1, 5, 2, 3),
        t3[3],
    )
    return tuple(t1), t2_public, t3_public


def _public_amplitudes_tri(tamps):
    """Occupied-first T1/T2 views; T3 is left in its triangular packing.

    A rank-2 constraint never reads T3, and the only other consumer of these
    blocks is `_ensure_coordinates`, which uses their shapes purely as a
    cache-invalidation key.  The full-tensor transposes in
    `_public_amplitudes` are meaningless on i<=j<=k storage, so don't invent
    a public T3 that no caller here is allowed to use.
    """
    t1, t2, t3 = tamps
    t2_public = (t2[0], t2[1].transpose(0, 2, 1, 3), t2[2])
    return tuple(t1), t2_public, tuple(t3)


def _permutation_sign(values):
    if len(set(values)) != len(values):
        raise ValueError("same-spin amplitude has repeated indices")
    inversions = sum(
        values[i] > values[j]
        for i in range(len(values))
        for j in range(i + 1, len(values))
    )
    return -1.0 if inversions % 2 else 1.0


def _combination_rank(values, size):
    """Lexicographic rank used by itertools.combinations and PySCF."""
    values = tuple(sorted(int(x) for x in values))
    rank = 0
    previous = -1
    order = len(values)
    for position, value in enumerate(values):
        if value < 0 or value >= size:
            raise IndexError("amplitude index %d out of range for size %d"
                             % (value, size))
        remaining = order - position - 1
        for candidate in range(previous + 1, value):
            rank += math.comb(size - candidate - 1, remaining)
        previous = value
    return rank


def _canonical_group(values, size, block, label):
    values = tuple(int(x) for x in values)
    for value in values:
        if value < 0 or value >= size:
            raise IndexError(
                "%s %s index %d out of range" % (block, label, value)
            )
    try:
        sign = _permutation_sign(values)
    except ValueError as error:
        raise ValueError(
            "%s has an identically zero same-spin component" % block
        ) from error
    return tuple(sorted(values)), sign


def _checked_denominators(denominators):
    scale = max(1.0, float(numpy.max(numpy.abs(denominators))))
    threshold = numpy.finfo(denominators.dtype).eps * scale * 10.0
    if numpy.any(numpy.abs(denominators) <= threshold):
        smallest = float(numpy.min(numpy.abs(denominators)))
        raise RuntimeError(
            "Near-zero UCCSDT amplitude denominator (min |D| = %.6e)"
            % smallest
        )
    return denominators


def _checked_coordinate_denominators(cc, eris):
    nocca, noccb = eris.nocc
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

    # A compact coordinate set knows its own layout and can hand back the
    # whole array in a few array operations.  The generic walk below is a
    # Python loop over every coordinate, which costs ~17 s per CC cycle at the
    # five million coordinates a UCISD constraint spans on a 97-orbital model.
    compact = getattr(
        cc.constraint_coordinates, "amplitude_denominators", None
    )
    if compact is not None:
        denominators = numpy.asarray(compact(eia, eIA), dtype=float)
        expected = (len(cc.constraint_coordinates),)
        if denominators.shape != expected:
            raise ValueError(
                "amplitude_denominators returned shape %s; expected %s"
                % (denominators.shape, expected)
            )
        return _checked_denominators(denominators)

    denominators = []
    for block, index in cc.constraint_coordinates:
        if block == "t1a":
            i, a = index
            value = eia[i, a]
        elif block == "t1b":
            i, a = index
            value = eIA[i, a]
        elif block == "t2aa":
            i, j, a, b = index
            value = eia[i, a] + eia[j, b]
        elif block == "t2ab":
            i, j, a, b = index
            value = eia[i, a] + eIA[j, b]
        elif block == "t2bb":
            i, j, a, b = index
            value = eIA[i, a] + eIA[j, b]
        elif block == "t3aaa":
            i, j, k, a, b, c = index
            value = eia[i, a] + eia[j, b] + eia[k, c]
        elif block == "t3aab":
            i, j, k, a, b, c = index
            value = eia[i, a] + eia[j, b] + eIA[k, c]
        elif block == "t3abb":
            i, j, k, a, b, c = index
            value = eia[i, a] + eIA[j, b] + eIA[k, c]
        elif block == "t3bbb":
            i, j, k, a, b, c = index
            value = eIA[i, a] + eIA[j, b] + eIA[k, c]
        else:
            raise ValueError("Unknown amplitude block %r" % block)
        denominators.append(value)

    return _checked_denominators(numpy.asarray(denominators))


def _update_amps_simultaneous_tri_(cc, tamps, eris):
    """Triangular-T3 counterpart of `_update_amps_simultaneous_`.

    Written out rather than imported from uccsdt_lowm_spinconstr, which
    imports from this module -- the reverse import would be circular.
    """
    t1, t2, t3 = tamps
    imds = uccsdt_lowm._IMDS()

    uccsdt_lowm.update_t1_fock_eris_uhf(cc, imds, t1, eris)
    uccsdt_lowm.intermediates_t1t2_uhf(cc, imds, t2)
    r1, r2 = uccsdt_lowm.compute_r1r2_uhf(cc, imds, t2)
    uccsdt_lowm.r1r2_add_t3_tri_uhf_(cc, imds, r1, r2, t3)
    uccsdt_lowm.antisymmetrize_r2_uhf_(r2)
    uccsdt_lowm.r1r2_divide_e_uhf_(cc, r1, r2, eris.mo_energy[:])

    # As in the full-storage version: a constrained fixed point can have
    # nonzero selected residuals, so T3 is evaluated at the same old T2.
    uccsdt_lowm.intermediates_t3_uhf(cc, imds, t2)
    uccsdt_lowm.intermediates_t3_add_t3_tri_uhf(cc, imds, t3)
    imds.t1_fock, imds.t1_eris = None, None
    r3 = uccsdt_lowm.compute_r3_tri_uhf(cc, imds, t2, t3)
    imds = None
    uccsdt_lowm.r3_tri_divide_e_uhf_(cc, r3, eris.mo_energy[:])

    for rank_blocks, residual_blocks in zip(tamps, (r1, r2, r3)):
        for block, residual in zip(rank_blocks, residual_blocks):
            block += residual
    return r1, r2, r3


def _update_amps_simultaneous_(cc, tamps, eris):
    """Build every Jacobi increment at the same input amplitudes."""
    if cc._t3_triangular:
        return _update_amps_simultaneous_tri_(cc, tamps, eris)
    t1, t2, t3 = tamps
    imds = uccsdt_highm._IMDS()

    uccsdt_highm.update_t1_fock_eris_uhf(cc, imds, t1, eris)
    uccsdt_highm.intermediates_t1t2_uhf(cc, imds, t2)
    r1, r2 = uccsdt_highm.compute_r1r2_uhf(cc, imds, t2)
    uccsdt_highm.r1r2_add_t3_uhf_(cc, imds, r1, r2, t3)
    uccsdt_highm.antisymmetrize_r2_uhf_(r2)
    uccsdt_highm.r1r2_divide_e_uhf_(cc, r1, r2, eris.mo_energy[:])

    # PySCF updates T1/T2 here.  A constrained fixed point can have nonzero
    # selected residuals, so T3 must instead be evaluated at the same old T2.
    uccsdt_highm.intermediates_t3_uhf(cc, imds, t2)
    uccsdt_highm.intermediates_t3_add_t3_uhf(cc, imds, t3)
    imds.t1_fock, imds.t1_eris = None, None
    r3 = uccsdt_highm.compute_r3_uhf(cc, imds, t2, t3)
    uccsdt_highm.antisymmetrize_r3_uhf_(r3)
    uccsdt_highm.r3_divide_e_uhf_(cc, r3, eris.mo_energy[:])

    for rank_blocks, residual_blocks in zip(tamps, (r1, r2, r3)):
        for block, residual in zip(rank_blocks, residual_blocks):
            block += residual
    return r1, r2, r3


class _ConstrainedUCCSDT:
    """UCCSDT with compact nonlinear amplitude constraints, storage-agnostic.

    Constraint callbacks receive occupied-first tensors.  In particular, the
    public mixed-spin layouts are ``t2ab[i,J,a,B]``, ``t3aab[i,j,K,a,b,C]``,
    and ``t3abb[i,J,K,a,B,C]`` even though PySCF stores these blocks in
    spin-grouped order internally.  Under triangular T3 storage only the
    T1/T2 conventions apply -- see `_public_amplitudes_tri`.

    The concrete classes at the bottom of this file bind the three attributes
    below; everything else here is common to both storages.  For a rank-2
    constraint the coordinate layer is storage-independent outright: the
    packed vector has the same layout in the T1/T2 region either way, so
    `packed_indices_uccsdt` and `_coordinate_flat` need no special casing.
    """

    #: PySCF class this solver is built on, bound by the subclasses below.
    _base_class = None
    #: PySCF module supplying that class's residual builders.
    _uccsdt_module = None
    #: Whether T3 is stored i<=j<=k only.
    _t3_triangular = False

    def _public_amplitudes(self, tamps):
        if self._t3_triangular:
            return _public_amplitudes_tri(tamps)
        return _public_amplitudes(tamps)

    def __init__(self, mf, constraint, constraint_jacobian,
                 constraint_coordinates, constraint_tol=1e-10,
                 constraint_diis_weight=1.0, constraint_max_step=0.0,
                 frozen=None, mo_coeff=None, mo_occ=None,
                 constraint_multiplier_solver="linear",
                 constraint_multiplier_max_cycle=20,
                 constraint_multiplier_ftol=1e-12,
                 constraint_jvp=None, constraint_vjp=None,
                 constraint_jacobian_row_norms=None,
                 constraint_weighted_gram_diagonal=None,
                 constraint_krylov_tol=1e-10,
                 constraint_krylov_max_cycle=None,
                 conv_check=True,
                 constraint_diis_scheme="legacy"):
        self._base_class.__init__(
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

        if (self._t3_triangular and getattr(
                constraint_coordinates, "_ccc_amplitude_rank", 2) == 3):
            raise NotImplementedError(
                "rank-3 constraint coordinates include T3, whose indexing "
                "differs between full and triangular storage; use "
                "MultiConstrUCCSDT for a rank-3 constraint engine"
            )

        self.constraint = constraint
        self.constraint_jacobian = constraint_jacobian
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
        if (self._matrix_free_constraints
                and not getattr(constraint_coordinates,
                                "_ccc_compact_coordinates", False)
                and any(block.startswith("t3")
                        for block, _ in self.constraint_coordinates)):
            raise NotImplementedError(
                "matrix-free UCCSDT currently supports CISD constraints on "
                "T1/T2 coordinates only"
            )

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
        self.conv_check = bool(conv_check)
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
        self._constraint_update_vector = None
        self._constraint_iteration = 0
        self._coordinate_shapes = None
        self._coordinate_flat = None
        self._coordinate_sign = None
        self._constraint_size = None
        self._keys = set(self._keys).union({
            "conv_check", "constraint_diis_scheme", "_constraint_tangent_residual",
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
            "constraint_jvp", "constraint_vjp",
            "constraint_jacobian_row_norms",
            "constraint_weighted_gram_diagonal",
            "constraint_krylov_tol", "constraint_krylov_max_cycle",
            "constraint_krylov_info", "constraint_krylov_iterations",
            "constraint_krylov_residual",
        })

    @classmethod
    def from_constraint_engine(cls, mf, engine, **kwargs):
        """Construct UCCSDT from a compact matrix-free constraint engine."""
        rank = getattr(engine.coordinates, "_ccc_amplitude_rank", 2)
        if rank == 2:
            def constraint(t1, t2, t3):
                return engine.constraint(t1, t2)

            def jacobian(t1, t2, t3):
                return engine.jacobian(t1, t2)

            kwargs.setdefault(
                "constraint_jvp",
                lambda t1, t2, t3, vector: engine.jvp(t1, t2, vector),
            )
            kwargs.setdefault(
                "constraint_vjp",
                lambda t1, t2, t3, weights: engine.vjp(t1, t2, weights),
            )
            kwargs.setdefault(
                "constraint_jacobian_row_norms",
                lambda t1, t2, t3: engine.jacobian_row_norms(t1, t2),
            )
            kwargs.setdefault(
                "constraint_weighted_gram_diagonal",
                lambda t1, t2, t3, metric: engine.weighted_gram_diagonal(
                    t1, t2, metric
                ),
            )
        elif rank == 3:
            constraint = engine.constraint
            jacobian = engine.jacobian
            kwargs.setdefault("constraint_jvp", engine.jvp)
            kwargs.setdefault("constraint_vjp", engine.vjp)
            kwargs.setdefault(
                "constraint_jacobian_row_norms", engine.jacobian_row_norms
            )
            kwargs.setdefault(
                "constraint_weighted_gram_diagonal",
                engine.weighted_gram_diagonal,
            )
        else:
            raise ValueError(
                "constraint engine amplitude rank must be 2 or 3, got %r"
                % rank
            )
        return cls(mf, constraint, jacobian, engine.coordinates, **kwargs)

    def dump_flags(self, verbose=None):
        self._base_class.dump_flags(self, verbose=verbose)
        log = logger.new_logger(self, verbose)
        log.info("Vector constraint tolerance = %.3g", self.constraint_tol)
        log.info("Constraint DIIS weight = %.3g",
                 self.constraint_diis_weight)
        log.info("Constraint amplitude coordinates = %d",
                 len(self.constraint_coordinates))
        log.info("T3 storage = %s",
                 "triangular (i<=j<=k)" if self._t3_triangular else "full")
        log.info("Constraint amplitude convention = occupied-first %s",
                 "T1/T2 tensors" if self._t3_triangular
                 else "full tensors")
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

    def public_amplitudes(self, tamps=None):
        """Return occupied-first amplitude views for callbacks/diagnostics."""
        if tamps is None:
            if any(block is None for block in self.tamps):
                raise ValueError("UCCSDT amplitudes are not initialized")
            tamps = self.tamps
        return self._public_amplitudes(tamps)

    @staticmethod
    def _block_sizes(t1):
        nocca, nvira = t1[0].shape
        noccb, nvirb = t1[1].shape
        return {
            "t1a": nocca * nvira,
            "t1b": noccb * nvirb,
            "t2aa": math.comb(nocca, 2) * math.comb(nvira, 2),
            "t2ab": nocca * nvira * noccb * nvirb,
            "t2bb": math.comb(noccb, 2) * math.comb(nvirb, 2),
            "t3aaa": math.comb(nocca, 3) * math.comb(nvira, 3),
            "t3aab": (math.comb(nocca, 2) * math.comb(nvira, 2)
                       * noccb * nvirb),
            "t3abb": (math.comb(noccb, 2) * math.comb(nvirb, 2)
                       * nocca * nvira),
            "t3bbb": math.comb(noccb, 3) * math.comb(nvirb, 3),
        }

    @classmethod
    def _block_offsets(cls, t1):
        sizes = cls._block_sizes(t1)
        offsets = {}
        position = 0
        for block in BLOCKS:
            offsets[block] = position
            position += sizes[block]
        return sizes, offsets

    def _coordinate_index(self, block, index, t1):
        nocca, nvira = t1[0].shape
        noccb, nvirb = t1[1].shape
        sizes, offsets = self._block_offsets(t1)
        index = tuple(int(x) for x in index)
        sign = 1.0

        if block == "t1a":
            if len(index) != 2:
                raise ValueError("t1a index must have length 2")
            local = numpy.ravel_multi_index(index, (nocca, nvira))
        elif block == "t1b":
            if len(index) != 2:
                raise ValueError("t1b index must have length 2")
            local = numpy.ravel_multi_index(index, (noccb, nvirb))
        elif block in ("t2aa", "t2bb"):
            if len(index) != 4:
                raise ValueError("%s index must have length 4" % block)
            nocc, nvir = ((nocca, nvira) if block == "t2aa"
                          else (noccb, nvirb))
            occupied, so = _canonical_group(index[:2], nocc, block, "occupied")
            virtual, sv = _canonical_group(index[2:], nvir, block, "virtual")
            local = (_combination_rank(occupied, nocc) * math.comb(nvir, 2)
                     + _combination_rank(virtual, nvir))
            sign = so * sv
        elif block == "t2ab":
            if len(index) != 4:
                raise ValueError("t2ab index must have length 4")
            i, j, a, b = index
            local = numpy.ravel_multi_index(
                (i, a, j, b), (nocca, nvira, noccb, nvirb)
            )
        elif block in ("t3aaa", "t3bbb"):
            if len(index) != 6:
                raise ValueError("%s index must have length 6" % block)
            nocc, nvir = ((nocca, nvira) if block == "t3aaa"
                          else (noccb, nvirb))
            occupied, so = _canonical_group(index[:3], nocc, block, "occupied")
            virtual, sv = _canonical_group(index[3:], nvir, block, "virtual")
            local = (_combination_rank(occupied, nocc) * math.comb(nvir, 3)
                     + _combination_rank(virtual, nvir))
            sign = so * sv
        elif block == "t3aab":
            if len(index) != 6:
                raise ValueError("t3aab index must have length 6")
            i, j, k, a, b, c = index
            occupied, so = _canonical_group((i, j), nocca, block, "occupied")
            virtual, sv = _canonical_group((a, b), nvira, block, "virtual")
            pair = (_combination_rank(occupied, nocca)
                    * math.comb(nvira, 2)
                    + _combination_rank(virtual, nvira))
            local = numpy.ravel_multi_index(
                (pair, k, c),
                (math.comb(nocca, 2) * math.comb(nvira, 2), noccb, nvirb),
            )
            sign = so * sv
        elif block == "t3abb":
            if len(index) != 6:
                raise ValueError("t3abb index must have length 6")
            i, j, k, a, b, c = index
            occupied, so = _canonical_group((j, k), noccb, block, "occupied")
            virtual, sv = _canonical_group((b, c), nvirb, block, "virtual")
            pair = (_combination_rank(occupied, noccb)
                    * math.comb(nvirb, 2)
                    + _combination_rank(virtual, nvirb))
            local = numpy.ravel_multi_index(
                (pair, i, a),
                (math.comb(noccb, 2) * math.comb(nvirb, 2), nocca, nvira),
            )
            sign = so * sv
        else:
            raise ValueError("Unknown amplitude block %r" % block)

        if local < 0 or local >= sizes[block]:
            raise IndexError("%s coordinate %r is out of range" % (block, index))
        return offsets[block] + int(local), sign

    def _ensure_coordinates(self, tamps):
        t1, t2, t3 = self._public_amplitudes(tamps)
        shapes = tuple(
            block.shape for block in tuple(t1) + tuple(t2) + tuple(t3)
        )
        if self._coordinate_shapes == shapes:
            return
        compact = getattr(
            self.constraint_coordinates, "packed_indices_uccsdt", None
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

    def _constraint_value(self, tamps):
        value = numpy.asarray(timed_constraint(
            self, "value", self.constraint, *self._public_amplitudes(tamps)
        ))
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

    def _constraint_jacobian(self, tamps, require_full_rank=True):
        self._ensure_coordinates(tamps)
        value = self._constraint_value(tamps)
        jacobian = numpy.asarray(timed_constraint(
            self, "jacobian", self.constraint_jacobian,
            *self._public_amplitudes(tamps)
        ))
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

    def _constraint_jvp(self, tamps, vector, packed=False):
        """Apply J in public coordinates, or packed coordinates if requested."""
        self._ensure_coordinates(tamps)
        vector = numpy.asarray(vector, dtype=float)
        if vector.shape != (len(self.constraint_coordinates),):
            raise ValueError("constraint JVP vector has an invalid shape")
        if packed:
            vector = self._coordinate_sign * vector
        if self._matrix_free_constraints:
            result = timed_constraint(
                self, "jvp", self.constraint_jvp,
                *self._public_amplitudes(tamps), vector,
            )
        else:
            _, packed_jacobian = self._constraint_jacobian(
                tamps, require_full_rank=False
            )
            jacobian = packed_jacobian * self._coordinate_sign
            result = jacobian @ vector
        result = numpy.asarray(result, dtype=float)
        if result.shape != (self._constraint_size,):
            raise ValueError("constraint JVP returned an invalid shape")
        return result

    def _constraint_vjp(self, tamps, weights, packed=False):
        """Apply J.T in public coordinates, or packed coordinates if requested."""
        self._ensure_coordinates(tamps)
        weights = numpy.asarray(weights, dtype=float)
        if weights.shape != (self._constraint_size,):
            raise ValueError("constraint VJP weights have an invalid shape")
        if self._matrix_free_constraints:
            result = timed_constraint(
                self, "vjp", self.constraint_vjp,
                *self._public_amplitudes(tamps), weights,
            )
            result = numpy.asarray(result, dtype=float)
        else:
            _, packed_jacobian = self._constraint_jacobian(
                tamps, require_full_rank=False
            )
            jacobian = packed_jacobian * self._coordinate_sign
            result = jacobian.T @ weights
        if result.shape != (len(self.constraint_coordinates),):
            raise ValueError("constraint VJP returned an invalid shape")
        return self._coordinate_sign * result if packed else result

    def _constraint_row_norms(self, tamps, packed=False):
        if self._matrix_free_constraints:
            norms = timed_constraint(
                self, "row_norms", self.constraint_jacobian_row_norms,
                *self._public_amplitudes(tamps),
            )
            norms = numpy.asarray(norms, dtype=float)
        else:
            _, jacobian = self._constraint_jacobian(
                tamps, require_full_rank=False
            )
            if not packed:
                jacobian = jacobian * self._coordinate_sign
            norms = numpy.linalg.norm(jacobian, axis=1)
        if norms.shape != (self._constraint_size,):
            raise ValueError("constraint row norms have an invalid shape")
        return norms

    def _constraint_gram_diagonal(self, tamps, metric):
        if self._matrix_free_constraints:
            diagonal = timed_constraint(
                self, "gram_diagonal",
                self.constraint_weighted_gram_diagonal,
                *self._public_amplitudes(tamps), metric,
            )
            diagonal = numpy.asarray(diagonal, dtype=float)
        else:
            _, packed_jacobian = self._constraint_jacobian(
                tamps, require_full_rank=False
            )
            jacobian = packed_jacobian * self._coordinate_sign
            diagonal = numpy.einsum("ri,i,ri->r", jacobian, metric, jacobian)
        if diagonal.shape != (self._constraint_size,):
            raise ValueError("constraint Gram diagonal has an invalid shape")
        return diagonal

    def _coordinate_blocks(self, tamps):
        t1, t2, t3 = self._public_amplitudes(tamps)
        return dict(zip(BLOCKS, tuple(t1) + tuple(t2) + tuple(t3)))

    def _coordinate_values(self, tamps):
        compact = getattr(
            self.constraint_coordinates, "amplitude_values", None
        )
        if compact is not None:
            t1, t2, t3 = self._public_amplitudes(tamps)
            if getattr(self.constraint_coordinates,
                       "_ccc_amplitude_rank", 2) == 3:
                return compact(t1, t2, t3)
            return compact(t1, t2)
        blocks = self._coordinate_blocks(tamps)
        return numpy.asarray([
            blocks[block][index]
            for block, index in self.constraint_coordinates
        ])

    @staticmethod
    def _set_antisymmetric_coordinate(array, index, value, groups):
        assignments = [(list(index), 1.0)]
        for positions in groups:
            expanded = []
            for permutation in permutations(range(len(positions))):
                sign = _permutation_sign(permutation)
                for target, old_sign in assignments:
                    updated = target.copy()
                    for destination, source in zip(positions, permutation):
                        updated[destination] = index[positions[source]]
                    expanded.append((updated, old_sign * sign))
            assignments = expanded
        for target, sign in assignments:
            array[tuple(target)] = sign * value

    def _set_coordinate_values(self, tamps, values):
        compact = getattr(self.constraint_coordinates, "set_values", None)
        if compact is not None:
            t1, t2, t3 = self._public_amplitudes(tamps)
            if getattr(self.constraint_coordinates,
                       "_ccc_amplitude_rank", 2) == 3:
                compact(t1, t2, t3, values)
            else:
                compact(t1, t2, values)
            return
        blocks = self._coordinate_blocks(tamps)
        antisymmetric_groups = {
            "t2aa": ((0, 1), (2, 3)),
            "t2bb": ((0, 1), (2, 3)),
            "t3aaa": ((0, 1, 2), (3, 4, 5)),
            "t3aab": ((0, 1), (3, 4)),
            "t3abb": ((1, 2), (4, 5)),
            "t3bbb": ((0, 1, 2), (3, 4, 5)),
        }
        for (block, index), value in zip(
                self.constraint_coordinates, values):
            array = blocks[block]
            groups = antisymmetric_groups.get(block)
            if groups is None:
                array[index] = value
            else:
                self._set_antisymmetric_coordinate(
                    array, index, value, groups
                )

    def constraint_residual(self, t1=None, t2=None, t3=None):
        if t1 is None and t2 is None and t3 is None:
            if any(block is None for block in self.tamps):
                raise ValueError("UCCSDT amplitudes are not initialized")
            tamps = self.tamps
        elif t1 is None or t2 is None or t3 is None:
            raise ValueError("t1, t2, and t3 must be supplied together")
        else:
            # Explicit arguments use the public occupied-first convention.
            native_t2 = (t2[0], t2[1].transpose(0, 2, 1, 3), t2[2])
            native_t3 = (
                t3[0],
                t3[1].transpose(0, 1, 3, 4, 2, 5),
                t3[2].transpose(1, 2, 4, 5, 0, 3),
                t3[3],
            )
            tamps = [t1, native_t2, native_t3]
        return self._constraint_value(tamps)

    def update_amps_(self, tamps, eris):
        self._ensure_coordinates(tamps)
        old = self._coordinate_values(tamps)
        if self._matrix_free_constraints:
            constraint = self._constraint_value(tamps)
            # The CISD and CISDT constraint Jacobians depend only on T1/T2;
            # preserve those because the UCCSDT Jacobi kernel updates in place.
            old_tamps = [
                tuple(block.copy() for block in tamps[0]),
                tuple(block.copy() for block in tamps[1]),
                tamps[2],
            ]
        else:
            constraint, packed_jacobian = self._constraint_jacobian(tamps)
            jacobian = packed_jacobian * self._coordinate_sign
        residuals = _update_amps_simultaneous_(self, tamps, eris)
        base = self._coordinate_values(tamps)

        denominators = _checked_coordinate_denominators(self, eris)
        metric = -1.0 / denominators
        if self._matrix_free_constraints:
            response = lambda weights: metric * self._constraint_vjp(
                old_tamps, weights
            )
            response_transpose = lambda vector: self._constraint_jvp(
                old_tamps, metric * vector
            )
            gram_matvec = lambda weights: self._constraint_jvp(
                old_tamps, response(weights)
            )
            rhs = -(constraint + self._constraint_jvp(
                old_tamps, base - old
            ))
            diagonal = self._constraint_gram_diagonal(old_tamps, metric)
            multipliers, correction = matrix_free_multipliers(
                self, gram_matvec, response, rhs, diagonal,
                initial=self.constraint_multipliers,
            )
            self.constraint_rank = None
            self.constraint_gram_rank = None
            self.constraint_gram_condition = None
        else:
            response_matrix = jacobian.T * metric[:, None]
            gram = jacobian @ response_matrix
            rhs = -(constraint + jacobian @ (base - old))
            multipliers, correction = damped_multipliers(
                self, gram, response_matrix, rhs, constraint.size
            )
        if self.constraint_diis_scheme == "tangent":
            # See uccsd_multiconstr.py's MultiConstrUCCSD.update_amps for
            # why this must be J . residual_selected (matching `rhs` above
            # minus its -C term), not response_transpose/response_matrix.T,
            # which would double-apply the metric already baked into
            # residual_selected = base-old, and why r_par adds
            # response(eta_LS) rather than subtracting it (new = base +
            # response(multipliers), so at a fixed point residual_selected =
            # -response(eta*) -- a PLUS is what makes r_par vanish there).
            residual_selected = base - old
            if self._matrix_free_constraints:
                rhs_ls = -self._constraint_jvp(old_tamps, residual_selected)
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
                    self._set_coordinate_values(tamps, values)

                    def trial_jvp(vector):
                        self._set_coordinate_values(tamps, values)
                        return self._constraint_jvp(tamps, vector)

                    def trial_vjp(weights):
                        self._set_coordinate_values(tamps, values)
                        return self._constraint_vjp(tamps, weights)

                    return (
                        self._constraint_value(tamps),
                        trial_jvp,
                        trial_vjp,
                    )

                try:
                    multipliers, correction = nonlinear_multipliers_matrix_free(
                        self, multipliers, response, response_transpose,
                        base, evaluate_products,
                    )
                finally:
                    self._set_coordinate_values(tamps, base)
            else:
                def evaluate(values):
                    self._set_coordinate_values(tamps, values)
                    trial_constraint, trial_packed = self._constraint_jacobian(
                        tamps, require_full_rank=False
                    )
                    trial_jacobian = trial_packed * self._coordinate_sign
                    return trial_constraint, trial_jacobian

                try:
                    multipliers, correction = nonlinear_multipliers(
                        self, multipliers, response_matrix, base, evaluate
                    )
                finally:
                    self._set_coordinate_values(tamps, base)
        new = base + correction
        if self._matrix_free_constraints:
            linearized = constraint + self._constraint_jvp(
                old_tamps, new - old
            )
        else:
            linearized = constraint + jacobian @ (new - old)

        # Only the selected independent amplitudes change in the multiplier
        # solve.  Expanding a packed UCCSDT vector here would materialize
        # hundreds of GB merely to assign these few coordinates.
        self._set_coordinate_values(tamps, new)
        self._set_coordinate_values(residuals, new - old)

        if (
            self.diis
            and self.do_diis_max_t
            and self._constraint_iteration >= self.diis_start_cycle
        ):
            self._constraint_update_vector = self.amplitudes_to_vector(
                residuals
            )
        else:
            self._constraint_update_vector = None
        self._constraint_iteration += 1

        self.constraint_multipliers = multipliers
        self.constraint_linearized_residual = linearized
        self.constraint_value = self._constraint_value(tamps)
        # The iteration counter has already advanced above, so step back to keep
        # the reported cycle number aligned with the constrained UCCSD path.
        log_constraint_timing(self, self._constraint_iteration - 1)
        return [
            numpy.linalg.norm(block)
            for rank_blocks in residuals
            for block in rank_blocks
        ]

    def run_diis(self, tamps, istep, normt, de, adiis):
        self.constraint_value = self._constraint_value(tamps)
        if (adiis and istep >= self.diis_start_cycle
                and abs(de) < self.diis_start_energy_diff):
            vector = self.amplitudes_to_vector(tamps)
            amplitude_error = self._constraint_update_vector
            if amplitude_error is None:
                raise RuntimeError(
                    "Missing constrained UCCSDT update vector for DIIS"
                )
            if self.constraint_diis_scheme == "tangent":
                error = amplitude_error.copy()
                if self._constraint_tangent_residual is not None:
                    error[self._coordinate_flat] = (
                        self._coordinate_sign * self._constraint_tangent_residual
                    )
                self.constraint_diis_error = None
            else:
                row_norms = self._constraint_row_norms(tamps, packed=True)
                scaled = _constraint_diis_error(
                    self, self.constraint_value, row_norms
                )
                error = numpy.append(amplitude_error, scaled)
                self.constraint_diis_error = scaled
            self._constraint_update_vector = None
            vector = adiis.update(vector, xerr=error)
            tamps = self.vector_to_amplitudes(vector)
            self.constraint_value = self._constraint_value(tamps)
            logger.debug1(self, "Vector-constrained DIIS for step %d", istep)
        else:
            self._constraint_update_vector = None
        return tamps

    def ccsdt(self, tamps=None, eris=None):
        self._constraint_iteration = 0
        self._constraint_update_vector = None
        self.constraint_multipliers = None
        configure_constraint_timing(self)
        return self._base_class.ccsdt(
            self, tamps=tamps, eris=eris
        )

    def _finalize(self):
        if not any(block is None for block in self.tamps):
            # `converged` is about to fold in the constraint tolerance (and
            # possibly the post-loop resync pass below); keep the amplitude-
            # equation verdict separate, since that's what a caller uses to
            # decide whether to continue with the nonlinear multiplier
            # solver.  Captured here, before either of those can touch it.
            self.amplitudes_converged = self.converged
            if (self.conv_check and self.converged
                    and float(numpy.max(numpy.abs(
                        self._constraint_value(self.tamps)))) > self.constraint_tol):
                # One extra Newton/eta correction pass, no DIIS after it --
                # the main loop's convergence test uses the pre-DIIS residual
                # norm, so it can exit leaving the true (post-DIIS) T with an
                # unresolved constraint violation that nothing else corrects.
                # Bounded to exactly one pass (never a retry loop): if it
                # doesn't resolve the constraint, or moves the energy/
                # amplitudes far more than a small cleanup step should, flag
                # not converged rather than silently accepting the result.
                eris = self.ao2mo(self.mo_coeff)
                e_before = self.e_corr
                res_norm = self.update_amps_(self.tamps, eris)
                normt_extra = float(numpy.linalg.norm(res_norm))
                self.e_corr = self.energy(self.tamps, eris)
                maximum_after = float(numpy.max(numpy.abs(
                    self._constraint_value(self.tamps))))
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
            self.constraint_value = self._constraint_value(self.tamps)
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
        return self._base_class._finalize(self)

    def check_constraint_jacobian(self, tamps=None, step=1e-6,
                                  ndir=3, seed=1, tolerance=1e-6):
        if tamps is None:
            if not any(block is None for block in self.tamps):
                tamps = self.tamps
            else:
                tamps = self.init_amps()[1]
        self._ensure_coordinates(tamps)
        if self.unique_tamps_map is None:
            self.unique_tamps_map = self.build_unique_tamps_map()
        vector = self.amplitudes_to_vector(tamps)
        self._constraint_value(tamps)
        rng = numpy.random.default_rng(seed)
        maximum = 0.0
        for _ in range(ndir):
            direction = rng.standard_normal(len(self._coordinate_flat))
            direction /= numpy.linalg.norm(direction)
            plus = vector.copy()
            minus = vector.copy()
            plus[self._coordinate_flat] += step * direction
            minus[self._coordinate_flat] -= step * direction
            finite_difference = (
                self._constraint_value(self.vector_to_amplitudes(plus))
                - self._constraint_value(self.vector_to_amplitudes(minus))
            ) / (2.0 * step)
            analytic = self._constraint_jvp(tamps, direction, packed=True)
            scale = numpy.maximum(1.0, numpy.maximum(
                numpy.abs(finite_difference), numpy.abs(analytic)
            ))
            maximum = max(
                maximum,
                float(numpy.max(
                    numpy.abs(finite_difference - analytic) / scale
                )),
            )
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


class MultiConstrUCCSDT(_ConstrainedUCCSDT, uccsdt_highm.UCCSDT):
    """Constrained UCCSDT on PySCF's full-tensor T3 storage."""

    _base_class = uccsdt_highm.UCCSDT
    _uccsdt_module = uccsdt_highm
    _t3_triangular = False


class LowMemoryMultiConstrUCCSDT(_ConstrainedUCCSDT, uccsdt_lowm.UCCSDT):
    """Constrained UCCSDT on PySCF's triangular (i<=j<=k) T3 storage.

    Identical constraint algebra to `MultiConstrUCCSDT`, including the
    matrix-free path taken whenever the engine supplies jvp/vjp.  What it
    buys is T3: on a 97-orbital, 115-electron model that is 46 GiB here
    against 346 GiB stored in full, which is the difference between a
    five-million-coordinate UCISD constraint being runnable and not.

    Rank-3 constraint engines are rejected -- their coordinates include T3,
    whose indexing genuinely differs between the two storages.
    """

    _base_class = uccsdt_lowm.UCCSDT
    _uccsdt_module = uccsdt_lowm
    _t3_triangular = True

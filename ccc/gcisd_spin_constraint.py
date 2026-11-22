"""CISD-closed spin constraints for a general (spinor) reference.

A GHF determinant is not an eigenstate of Sz, so GCCSD on top of one has
nothing holding the spin structure in place: the amplitudes are free to rotate
the reference's spin arrangement away, and in practice they do -- N2 at 3.0 A on
a canted two-quartet reference runs to <Sz> = -2.1, <S^2> = 9.2 without
converging.  Constraining the correlated state to satisfy

    (Sz - Ms) |Psi> = 0        and        S+ |Psi> = 0

is what makes the calculation mean anything.  Both operators are one-body, so
projecting either equation on a determinant of rank r reaches preimages of rank
r-1, r and r+1; CISD closure -- every preimage parameterised by the CCSD ansatz
-- therefore holds exactly for outputs of rank <= 1, giving

    nconstraint = 2 * (1 + nocc * nvir)

equations, the reference and the singles for each operator.  Unlike the
collinear engines, *both* operators contribute: ``ClosedCISDSpinConstraint`` and
``UCISDSpinConstraint`` need S+ rows only because a UHF-like reference conserves
Sz already, whereas here Sz is a dense matrix in the spinor MO basis.

The rows have a closed tensor form.  With c_0 = 1, c_i^a = t1[i,a] and
c_ij^ab = t2[i,j,a,b] + t1[i,a] t1[j,b] - t1[i,b] t1[j,a], and no rank-three
coefficient needed (a one-body operator cannot connect rank 1 to rank 3),

    <0|O|Psi>     = tr(O_oo) + sum_ia O_ov[i,a] t1[i,a]
    <i^a|O|Psi>   = O_vo[a,i] + tr(O_oo) t1[i,a]
                    + sum_b O_vv[a,b] t1[i,b] - sum_j O_oo[j,i] t1[j,a]
                    + sum_jb O_ov[j,b] c_ij^ab

with ``- Ms c_D`` added for the Sz operator.  The most expensive contraction is
``O_ov . t2``, one CCSD-sized O(nocc^2 nvir^2) einsum, so a constraint
evaluation costs a small fraction of an amplitude iteration and no CI vector is
ever built -- the determinant-space prototype this replaces needs
C(nmo, nocc) coefficients and stops being usable around 20 spinors.

Coordinates are the *entire* packed GCCSD amplitude vector, since dense spin
operators touch every amplitude; derivatives are returned in PySCF's packed
convention, where the independent t2 parameter is ``t2[i, j, a, b]`` with i > j
and a > b, so a t2 gradient is antisymmetrised before packing.

Amplitudes and orbitals must be real.  Complex spinors are rejected rather than
silently truncated: the closed forms above are real-linear, and a complex GHF
would need the conjugate rows as separate equations.

One caveat worth stating, because it looks like a bug and is not: the converged
state comes out with <S^2> close to the target but <Sz> around 0.6 rather than
Ms.  CISD closure controls the spin structure only inside the rank <= 2 block,
which pins S^2 well and Ms loosely.
"""
from collections.abc import Sequence
from math import comb
from time import perf_counter

import numpy

from pyscf.cc import gccsd


def spin_matrices(mo_coeff, overlap):
    """Sz and S+ in the MO spinor basis of a general determinant.

    ``mo_coeff`` is the (2 nao, nmo) GHF coefficient matrix and ``overlap`` the
    ordinary (nao, nao) AO overlap.  The matrices must be built from the
    coefficients like this and never written down from the spinor ordering: a
    guess such as "Sz = diag(+1/2, -1/2)" is only true in the AO spinor basis,
    and using it in the MO basis silently reports <S^2> = 12 for a state whose
    <S^2> is zero.
    """
    mo_coeff = numpy.asarray(mo_coeff)
    overlap = numpy.asarray(overlap)
    nao = overlap.shape[0]
    if mo_coeff.shape[0] != 2 * nao:
        raise ValueError(
            "mo_coeff has %d rows; a GHF spinor basis over %d AOs needs %d"
            % (mo_coeff.shape[0], nao, 2 * nao)
        )
    first, second = mo_coeff[:nao], mo_coeff[nao:]
    sz = 0.5 * (first.conj().T @ overlap @ first
                - second.conj().T @ overlap @ second)
    sp = first.conj().T @ overlap @ second
    return sz, sp


class _GCCSDCoordinates(Sequence):
    """The packed GCCSD amplitude vector, presented as a coordinate list.

    Dense spin operators involve every amplitude, so unlike the collinear
    engines there is no subspace to book-keep: the coordinate set is the whole
    vector and the packed-index map is the identity.
    """

    _ccc_compact_coordinates = True

    def __init__(self, nmo, nocc):
        self.nmo = int(nmo)
        self.nocc = int(nocc)
        self.nvir = self.nmo - self.nocc
        if not 0 < self.nocc < self.nmo:
            raise ValueError("nocc must lie strictly between 0 and nmo")
        self.nt1 = self.nocc * self.nvir
        self.nt2 = comb(self.nocc, 2) * comb(self.nvir, 2)
        self._occ_pairs = numpy.tril_indices(self.nocc, k=-1)
        self._vir_pairs = numpy.tril_indices(self.nvir, k=-1)

    def __len__(self):
        return self.nt1 + self.nt2

    def __getitem__(self, position):
        if isinstance(position, slice):
            return tuple(self[index] for index in range(*position.indices(len(self))))
        position = int(position)
        if position < 0:
            position += len(self)
        if not 0 <= position < len(self):
            raise IndexError("GCCSD coordinate index out of range")
        if position < self.nt1:
            return "t1", divmod(position, self.nvir)
        position -= self.nt1
        occ, vir = divmod(position, self._vir_pairs[0].size)
        return "t2", (int(self._occ_pairs[0][occ]), int(self._occ_pairs[1][occ]),
                      int(self._vir_pairs[0][vir]), int(self._vir_pairs[1][vir]))

    def amplitude_values(self, t1, t2):
        return gccsd.amplitudes_to_vector(numpy.asarray(t1), numpy.asarray(t2))

    def set_values(self, t1, t2, values):
        values = numpy.asarray(values)
        if values.shape != (len(self),):
            raise ValueError("GCCSD coordinate values have the wrong shape")
        new1, new2 = gccsd.vector_to_amplitudes(values, self.nmo, self.nocc)
        t1[:] = new1
        t2[:] = new2


class _OneBodyBlocks:
    """Occupied/virtual blocks of one MO-basis operator, with its target shift.

    ``shift`` is the eigenvalue the constrained state must carry (Ms for Sz,
    zero for S+).  It multiplies the *identity in Fock space*, not a one-body
    matrix, so it enters as ``- shift c_D`` on every row; equivalently it could
    be folded into the matrix as ``- (shift / nelec) * I`` because the CC state
    is an exact eigenstate of the particle number.  Keeping it separate says so
    out loud.
    """

    def __init__(self, name, matrix, nocc, shift):
        matrix = numpy.asarray(matrix)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("%s operator must be a square matrix" % name)
        if numpy.iscomplexobj(matrix):
            largest = float(numpy.max(numpy.abs(matrix.imag), initial=0.0))
            raise ValueError(
                "%s operator is complex (max |imag| = %.3e); the CISD-closed "
                "GCCSD constraints are real-only, so a complex GHF needs the "
                "conjugate rows as separate equations" % (name, largest)
            )
        matrix = numpy.asarray(matrix, dtype=float)
        if not numpy.all(numpy.isfinite(matrix)):
            raise ValueError("%s operator contains non-finite elements" % name)
        self.name = name
        self.oo = matrix[:nocc, :nocc]
        self.ov = numpy.ascontiguousarray(matrix[:nocc, nocc:])
        self.vo = matrix[nocc:, :nocc]
        self.vv = numpy.ascontiguousarray(matrix[nocc:, nocc:])
        self.shift = float(shift)
        # tr(O_oo) - shift: the reference row's constant and, separately, the
        # coefficient of t1[i,a] in every singles row.
        self.diagonal = float(numpy.trace(self.oo)) - self.shift


class GCISDSpinConstraint:
    """CISD-closed {Sz - Ms, S+} equations for a general spinor reference.

    ``sz`` and ``sp`` are the MO-basis spin matrices of the correlated spinors,
    ordered occupied first -- see :func:`spin_matrices`.  The interface matches
    the collinear engines (``constraint``, ``jvp``, ``vjp``, ``jacobian``,
    ``jacobian_row_norms``, ``weighted_gram_diagonal``, ``hessian_vector``,
    ``coordinates``), so :class:`~ccc.gccsd_multiconstr.MultiConstrGCCSD` can
    be built with ``from_constraint_engine``.
    """

    def __init__(self, nmo, nocc, sz, sp, ms):
        start = perf_counter()
        self.nmo = int(nmo)
        self.nocc = int(nocc)
        self.nvir = self.nmo - self.nocc
        self.ms = float(ms)
        if not numpy.isfinite(self.ms):
            raise ValueError("ms must be finite")
        self.coordinates = _GCCSDCoordinates(self.nmo, self.nocc)
        for name, matrix in (("Sz", sz), ("S+", sp)):
            if numpy.asarray(matrix).shape != (self.nmo, self.nmo):
                raise ValueError(
                    "%s operator has shape %s; expected (%d, %d)"
                    % (name, numpy.asarray(matrix).shape, self.nmo, self.nmo)
                )
        self.operators = (
            _OneBodyBlocks("Sz", sz, self.nocc, self.ms),
            _OneBodyBlocks("S+", sp, self.nocc, 0.0),
        )
        self.rows_per_operator = 1 + self.nocc * self.nvir
        self.nconstraint = len(self.operators) * self.rows_per_operator
        self._gradient_cache_t1 = None
        self._gradient_cache = None
        self.build_time = perf_counter() - start

    @classmethod
    def from_cc(cls, cc, ms, frozen_coupling_tol=1e-8):
        """Build the engine from a GCCSD object and its GHF reference.

        Frozen orbitals are allowed only when neither spin operator carries an
        electron between the frozen and correlated spaces.  For a canted
        reference that coupling is generally nonzero -- Sz and S+ mix spinors
        that a collinear calculation would keep apart -- so this gate fires
        rather than silently approximating; correlate everything, or raise the
        tolerance deliberately and accept the violation.
        """
        mo_coeff = numpy.asarray(cc.mo_coeff)
        overlap = numpy.asarray(cc._scf.get_ovlp())
        nao = overlap.shape[0] // 2
        blocks = (overlap[:nao, :nao], overlap[nao:, nao:])
        off_diagonal = max(
            float(numpy.max(numpy.abs(overlap[:nao, nao:]), initial=0.0)),
            float(numpy.max(numpy.abs(overlap[nao:, :nao]), initial=0.0)),
        )
        if (off_diagonal > 1e-12
                or numpy.max(numpy.abs(blocks[0] - blocks[1])) > 1e-12):
            raise ValueError(
                "the reference overlap is not block diagonal with equal alpha "
                "and beta blocks, so the AO spinor basis is not the expected "
                "alpha-then-beta one"
            )
        if (numpy.iscomplexobj(mo_coeff)
                and numpy.max(numpy.abs(mo_coeff.imag)) > frozen_coupling_tol):
            raise ValueError(
                "complex GHF orbitals are not supported (max |imag| = %.3e); "
                "the CISD-closed GCCSD constraints are real-only"
                % float(numpy.max(numpy.abs(mo_coeff.imag)))
            )
        sz, sp = spin_matrices(numpy.asarray(mo_coeff.real, dtype=float),
                               numpy.asarray(blocks[0].real, dtype=float))

        mask = numpy.asarray(cc.get_frozen_mask(), dtype=bool)
        if mask.size != mo_coeff.shape[1]:
            raise ValueError("the frozen mask does not match the MO dimension")
        active = numpy.flatnonzero(mask)
        frozen = numpy.flatnonzero(~mask)
        coupling = 0.0
        for matrix in (sz, sp):
            for rows, columns in ((frozen, active), (active, frozen)):
                if rows.size and columns.size:
                    block = matrix[numpy.ix_(rows, columns)]
                    coupling = max(coupling, float(numpy.max(numpy.abs(block))))
        if coupling > frozen_coupling_tol:
            raise ValueError(
                "the spin operators couple the frozen and correlated spaces "
                "(maximum element %.3e exceeds %.3e); correlate all spinors, "
                "or raise frozen_coupling_tol to accept the violation"
                % (coupling, frozen_coupling_tol)
            )

        occupation = numpy.asarray(cc.mo_occ)[active]
        nocc = int(cc.nocc)
        if (numpy.max(numpy.abs(occupation[:nocc] - 1.0), initial=0.0) > 1e-10
                or numpy.max(numpy.abs(occupation[nocc:]), initial=0.0) > 1e-10):
            raise ValueError(
                "correlated spinors are not ordered as occupied then virtual"
            )
        engine = cls(active.size, nocc,
                     sz[numpy.ix_(active, active)],
                     sp[numpy.ix_(active, active)], ms)
        engine.frozen_coupling = coupling
        return engine

    # ------------------------------------------------------------- bookkeeping

    def amplitude_values(self, t1, t2):
        return self.coordinates.amplitude_values(t1, t2)

    def _check_amplitudes(self, t1, t2):
        t1 = numpy.asarray(t1)
        t2 = numpy.asarray(t2)
        if t1.shape != (self.nocc, self.nvir):
            raise ValueError(
                "t1 has shape %s; expected (%d, %d)"
                % (t1.shape, self.nocc, self.nvir)
            )
        if t2.shape != (self.nocc, self.nocc, self.nvir, self.nvir):
            raise ValueError("t2 has shape %s; expected %s" % (
                t2.shape,
                (self.nocc, self.nocc, self.nvir, self.nvir),
            ))
        if numpy.iscomplexobj(t1) or numpy.iscomplexobj(t2):
            raise ValueError("complex amplitudes are not supported")
        return t1, t2

    def _split_packed_vector(self, vector):
        vector = numpy.asarray(vector, dtype=float)
        if vector.shape != (len(self.coordinates),):
            raise ValueError("constraint coordinate vector has the wrong shape")
        t1 = vector[:self.coordinates.nt1].reshape(self.nocc, self.nvir)
        t2 = vector[self.coordinates.nt1:].reshape(
            self.coordinates._occ_pairs[0].size,
            self.coordinates._vir_pairs[0].size,
        )
        return t1, t2

    def _split_weights(self, weights):
        weights = numpy.asarray(weights, dtype=float)
        if weights.shape != (self.nconstraint,):
            raise ValueError("constraint weights have the wrong shape")
        block = weights.reshape(len(self.operators), self.rows_per_operator)
        return [(float(row[0]), row[1:].reshape(self.nocc, self.nvir))
                for row in block]

    def _gradient_factors(self, operator, t1):
        """The three T-dependent factors shared by every derivative.

        ``scalar`` multiplies x1[i,a], ``vv`` couples other virtuals of the same
        occupied spinor, ``oo`` other occupied spinors of the same virtual; the
        remaining piece of the row gradient is the rank-one ``t1 <O_ov, x1>``.
        """
        scalar = operator.diagonal + float(
            numpy.einsum("ia,ia->", operator.ov, t1, optimize=True)
        )
        vv = operator.vv - numpy.einsum(
            "ja,jb->ab", t1, operator.ov, optimize=True
        )
        oo = operator.oo + numpy.einsum(
            "jb,ib->ji", operator.ov, t1, optimize=True
        )
        return scalar, vv, oo

    def _cached_gradient_factors(self, t1):
        cached = self._gradient_cache_t1
        if (cached is None or cached.shape != t1.shape
                or not numpy.array_equal(cached, t1)):
            self._gradient_cache_t1 = numpy.array(t1, copy=True)
            self._gradient_cache = tuple(
                self._gradient_factors(operator, t1)
                for operator in self.operators
            )
        return self._gradient_cache

    def _packed_doubles_contraction(self, packed, ov):
        """Return ``sum_jb ov[j,b] x2[i,j,a,b]`` without unpacking x2."""
        oi, oj = self.coordinates._occ_pairs
        va, vb = self.coordinates._vir_pairs
        result = numpy.zeros((self.nocc, self.nvir), dtype=packed.dtype)
        for pair, (i, j) in enumerate(zip(oi, oj)):
            row = packed[pair]
            from_j = (
                numpy.bincount(
                    va, weights=row * ov[j, vb], minlength=self.nvir
                )
                - numpy.bincount(
                    vb, weights=row * ov[j, va], minlength=self.nvir
                )
            )
            from_i = (
                numpy.bincount(
                    va, weights=row * ov[i, vb], minlength=self.nvir
                )
                - numpy.bincount(
                    vb, weights=row * ov[i, va], minlength=self.nvir
                )
            )
            result[i] += from_j
            result[j] -= from_i
        return result

    def _add_packed_doubles_gradient(self, target, single, ov):
        """Accumulate a VJP directly in the packed ``(i>j,a>b)`` block."""
        oi, oj = self.coordinates._occ_pairs
        va, vb = self.coordinates._vir_pairs
        for pair, (i, j) in enumerate(zip(oi, oj)):
            row = target[pair]
            row += single[i, va] * ov[j, vb]
            row -= single[j, va] * ov[i, vb]
            row -= single[i, vb] * ov[j, va]
            row += single[j, vb] * ov[i, va]

    def _packed_doubles_gram_diagonal(self, metric, ov_square):
        """Doubles part of ``diag(J diag(metric) J.T)`` in packed form."""
        oi, oj = self.coordinates._occ_pairs
        va, vb = self.coordinates._vir_pairs
        result = numpy.zeros((self.nocc, self.nvir), dtype=metric.dtype)
        for pair, (i, j) in enumerate(zip(oi, oj)):
            row = metric[pair]
            result[i] += numpy.bincount(
                va, weights=row * ov_square[j, vb], minlength=self.nvir
            )
            result[i] += numpy.bincount(
                vb, weights=row * ov_square[j, va], minlength=self.nvir
            )
            result[j] += numpy.bincount(
                va, weights=row * ov_square[i, vb], minlength=self.nvir
            )
            result[j] += numpy.bincount(
                vb, weights=row * ov_square[i, va], minlength=self.nvir
            )
        return result

    # -------------------------------------------------------------- the rows

    def constraint(self, t1, t2):
        t1, t2 = self._check_amplitudes(t1, t2)
        rows = []
        for operator in self.operators:
            reference = operator.diagonal + float(
                numpy.einsum("ia,ia->", operator.ov, t1, optimize=True)
            )
            single = operator.vo.T + operator.diagonal * t1
            single += numpy.einsum("ab,ib->ia", operator.vv, t1, optimize=True)
            single -= numpy.einsum("ji,ja->ia", operator.oo, t1, optimize=True)
            single += numpy.einsum("jb,ijab->ia", operator.ov, t2, optimize=True)
            single += float(numpy.einsum(
                "jb,jb->", operator.ov, t1, optimize=True)) * t1
            single -= numpy.einsum(
                "jb,ib,ja->ia", operator.ov, t1, t1, optimize=True
            )
            rows.append(numpy.append(reference, single.ravel()))
        return numpy.concatenate(rows)

    def jvp(self, t1, t2, vector):
        t1, t2 = self._check_amplitudes(t1, t2)
        x1, x2 = self._split_packed_vector(vector)
        rows = []
        for operator, (scalar, vv, oo) in zip(
                self.operators, self._cached_gradient_factors(t1)):
            overlap = float(numpy.einsum("ia,ia->", operator.ov, x1,
                                         optimize=True))
            single = scalar * x1 + overlap * t1
            single += numpy.einsum("ab,ib->ia", vv, x1, optimize=True)
            single -= numpy.einsum("ji,ja->ia", oo, x1, optimize=True)
            single += self._packed_doubles_contraction(x2, operator.ov)
            rows.append(numpy.append(overlap, single.ravel()))
        return numpy.concatenate(rows)

    def vjp(self, t1, t2, weights):
        t1, t2 = self._check_amplitudes(t1, t2)
        gradient = numpy.zeros(len(self.coordinates), dtype=t1.dtype)
        grad1 = gradient[:self.coordinates.nt1].reshape(t1.shape)
        grad2 = gradient[self.coordinates.nt1:].reshape(
            self.coordinates._occ_pairs[0].size,
            self.coordinates._vir_pairs[0].size,
        )
        factors = self._cached_gradient_factors(t1)
        for operator, (reference, single), (scalar, vv, oo) in zip(
                self.operators, self._split_weights(weights), factors):
            grad1 += (reference + float(numpy.einsum(
                "ia,ia->", single, t1, optimize=True))) * operator.ov
            grad1 += scalar * single
            grad1 += numpy.einsum("ia,ab->ib", single, vv, optimize=True)
            grad1 -= numpy.einsum("ia,ji->ja", single, oo, optimize=True)
            self._add_packed_doubles_gradient(grad2, single, operator.ov)
        return gradient

    def jacobian(self, t1, t2):
        size = self.nconstraint * len(self.coordinates)
        if size * 8 > 2 * 1024 ** 3:
            raise MemoryError(
                "dense GCISD Jacobian would require %.1f GiB; use jvp/vjp"
                % (size * 8 / 1024 ** 3)
            )
        jacobian = numpy.empty((self.nconstraint, len(self.coordinates)))
        basis = numpy.zeros(self.nconstraint)
        for row in range(self.nconstraint):
            basis[row] = 1.0
            jacobian[row] = self.vjp(t1, t2, basis)
            basis[row] = 0.0
        return jacobian

    def weighted_gram_diagonal(self, t1, t2, metric):
        """diag(J diag(metric) J^T), in closed form.

        The rows share every t1 coordinate, so unlike the collinear engines this
        cannot be a sum of independent block norms; the cross terms between the
        rank-one ``t1 <O_ov, x1>`` piece and the rest are kept explicitly.
        ``metric`` must be nonnegative -- it is ``-1/D`` in the solver, and the
        t2 part needs the coordinate weight replicated over the four
        antisymmetric images, which is only meaningful for a magnitude.
        """
        t1, t2 = self._check_amplitudes(t1, t2)
        metric = numpy.asarray(metric, dtype=float)
        if metric.shape != (len(self.coordinates),):
            raise ValueError("constraint metric has the wrong shape")
        if numpy.any(metric < 0):
            raise ValueError("weighted_gram_diagonal needs a nonnegative metric")
        metric1, metric2 = self._split_packed_vector(metric)
        diagonal = []
        for operator, (scalar, vv, oo) in zip(
                self.operators, self._cached_gradient_factors(t1)):
            ov = operator.ov
            weighted = metric1 * ov
            reference = float(numpy.einsum("ia,ia->", weighted, ov,
                                           optimize=True))
            # row (i,a) gradient = t1[i,a] * O_ov + E, with E carrying the
            # scalar, vv and oo pieces on the (i,:) and (:,a) cross
            corner = (scalar + numpy.diag(vv)[None, :]
                      - numpy.diag(oo)[:, None])
            cross = (scalar * weighted
                     + numpy.einsum("ib,ab->ia", weighted, vv, optimize=True)
                     - numpy.einsum("ja,ji->ia", weighted, oo, optimize=True))
            square = metric1 * (corner ** 2
                                - numpy.diag(vv)[None, :] ** 2
                                - numpy.diag(oo)[:, None] ** 2)
            square += numpy.einsum("ib,ab->ia", metric1, vv ** 2, optimize=True)
            square += numpy.einsum("ja,ji->ia", metric1, oo ** 2, optimize=True)
            single = t1 ** 2 * reference + 2.0 * t1 * cross + square
            single += self._packed_doubles_gram_diagonal(
                metric2, ov ** 2
            )
            diagonal.append(numpy.append(reference, single.ravel()))
        # The metric is nonnegative, so the true diagonal is too; the explicit
        # cancellations above return the identically-zero rows of a collinear
        # reference as rounding noise of either sign.
        return numpy.maximum(numpy.concatenate(diagonal), 0.0)

    def jacobian_row_norms(self, t1, t2):
        t1, t2 = self._check_amplitudes(t1, t2)
        diagonal = []
        for operator, (scalar, vv, oo) in zip(
                self.operators, self._cached_gradient_factors(t1)):
            ov = operator.ov
            reference = float(numpy.einsum("ia,ia->", ov, ov, optimize=True))
            vv_diagonal = numpy.diag(vv)
            oo_diagonal = numpy.diag(oo)
            corner = (scalar + vv_diagonal[None, :]
                      - oo_diagonal[:, None])
            cross = (scalar * ov
                     + numpy.einsum("ib,ab->ia", ov, vv, optimize=True)
                     - numpy.einsum("ja,ji->ia", ov, oo, optimize=True))
            square = (corner ** 2 - vv_diagonal[None, :] ** 2
                      - oo_diagonal[:, None] ** 2)
            square += numpy.sum(vv ** 2, axis=1)[None, :]
            square += numpy.sum(oo ** 2, axis=0)[:, None]
            single = t1 ** 2 * reference + 2.0 * t1 * cross + square

            ov_square = ov ** 2
            single += (numpy.sum(ov_square)
                       - numpy.sum(ov_square, axis=1)[:, None]
                       - numpy.sum(ov_square, axis=0)[None, :]
                       + ov_square)
            diagonal.append(numpy.append(reference, single.ravel()))
        # Sums of squares in exact arithmetic; the cancellations above return
        # the identically-zero rows of a collinear reference as rounding noise
        # of either sign, and sqrt of a negative would poison DIIS with NaN.
        return numpy.sqrt(numpy.maximum(numpy.concatenate(diagonal), 0.0))

    def hessian_vector(self, t1, t2, weights, vector):
        """sum_r weights[r] * (d^2 C_r / dt dt) . vector.

        Only the bilinear ``t1 t1`` piece of the doubles coefficient is
        quadratic, so the result lives entirely in the t1 block.
        """
        t1, t2 = self._check_amplitudes(t1, t2)
        x1 = self._split_packed_vector(vector)[0]
        out1 = numpy.zeros_like(t1)
        for operator, (_, single) in zip(self.operators,
                                         self._split_weights(weights)):
            ov = operator.ov
            out1 += single * float(numpy.einsum("ia,ia->", ov, x1,
                                                optimize=True))
            out1 += ov * float(numpy.einsum("ia,ia->", single, x1,
                                             optimize=True))
            out1 -= numpy.einsum("ka,jc,ja->kc", single, ov, x1, optimize=True)
            out1 -= numpy.einsum("ic,kb,ib->kc", single, ov, x1, optimize=True)
        return numpy.concatenate(
            (out1.ravel(), numpy.zeros(self.coordinates.nt2))
        )

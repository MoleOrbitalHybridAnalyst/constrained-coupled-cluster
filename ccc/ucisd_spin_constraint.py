"""Highest-weight CISD spin constraints for unmatched alpha/beta orbitals.

``ClosedCISDSpinConstraint`` needs the alpha and beta MOs to be a signed
permutation of one another, which makes S+ a sum of ``nmo`` diagonal terms.
With genuine UHF or UKS orbitals the two spatial sets differ and

    S+ = sum_pq <phi_p^alpha | phi_q^beta> a+_{p,alpha} a_{q,beta}

runs over every orbital pair.  S+ is still a one-body operator, so lowering an
output determinant shifts its excitation rank by

    rank(preimage) = rank(output) + 1[p in ref_alpha] - 1[q in ref_beta]

which stays in {-1, 0, +1}; CISD closure therefore remains meaningful and is
still decided by one comparison per output.

Retention here is *coefficient agnostic*: every pair (p, q) counts as present
no matter how small its overlap element is, so the number of constraints is a
function of ``(nmo, nalpha, nbeta)`` alone.  It is identical at every geometry,
there is no tolerance to tune, and no row count can jump in the middle of a
scan.  The overlap elements themselves enter the compiled equations exactly --
only the choice of which equations to keep ignores them.

The price is constraint count.  A diagonal overlap forbids the +1 shift, so the
matched engine keeps every output of rank <= 2, while assuming a dense overlap
allows +1 almost everywhere and keeps only rank <= 1: for N2/STO-3G that is 21
equations instead of 357.  The two classes therefore coexist -- use
``ClosedCISDSpinConstraint`` when the alpha and beta orbital sets really do
coincide, and this one when they do not.

The retained rank-one equations have a closed tensor form.  This engine uses
that form directly instead of building determinant metadata or sparse
polynomial dictionaries; it retains the same ``constraint``/``jacobian``/
``jvp``/``vjp``/``hessian_vector`` interface as the matched engine.
"""
from time import perf_counter
from math import comb
from collections.abc import Sequence

import numpy

from .cisd_spin_constraint import ClosedCISDSpinConstraint, _apply_sequence


class _UCISDCoordinates(Sequence):
    """Lazy t1a/t1b/t2ab coordinate sequence in the engine's tensor order."""

    _ccc_compact_coordinates = True

    def __init__(self, nmo, nocc):
        self.nmo = int(nmo)
        self.nocc = tuple(int(value) for value in nocc)
        self.nocca, self.noccb = self.nocc
        self.nvira = self.nmo - self.nocca
        self.nvirb = self.nmo - self.noccb
        self.nt1a = self.nocca * self.nvira
        self.nt1b = self.noccb * self.nvirb
        self.nt2ab = self.nt1a * self.nt1b

    def __len__(self):
        return self.nt1a + self.nt1b + self.nt2ab

    def __getitem__(self, position):
        if isinstance(position, slice):
            return tuple(self[index] for index in range(*position.indices(len(self))))
        position = int(position)
        if position < 0:
            position += len(self)
        if not 0 <= position < len(self):
            raise IndexError("UCISD coordinate index out of range")
        if position < self.nt1a:
            return "t1a", divmod(position, self.nvira)
        position -= self.nt1a
        if position < self.nt1b:
            return "t1b", divmod(position, self.nvirb)
        position -= self.nt1b
        index = numpy.unravel_index(
            position, (self.nocca, self.noccb, self.nvira, self.nvirb)
        )
        return "t2ab", tuple(int(value) for value in index)

    def amplitude_values(self, t1, t2):
        return numpy.concatenate((
            numpy.asarray(t1[0]).ravel(),
            numpy.asarray(t1[1]).ravel(),
            numpy.asarray(t2[1]).ravel(),
        ))

    def set_values(self, t1, t2, values):
        values = numpy.asarray(values)
        if values.shape != (len(self),):
            raise ValueError("UCISD coordinate values have the wrong shape")
        stopa = self.nt1a
        stopb = stopa + self.nt1b
        t1[0][:] = values[:stopa].reshape(t1[0].shape)
        t1[1][:] = values[stopa:stopb].reshape(t1[1].shape)
        t2[1][:] = values[stopb:].reshape(t2[1].shape)

    def amplitude_denominators(self, eia, eIA):
        """Orbital-energy denominators for every coordinate, in order.

        The generic path walks this sequence one coordinate at a time, which
        is ~17 s per CC cycle once the t2ab block runs to millions of
        entries.  The layout is known here, so build it in three array ops.
        """
        eia = numpy.asarray(eia)
        eIA = numpy.asarray(eIA)
        if (eia.shape != (self.nocca, self.nvira)
                or eIA.shape != (self.noccb, self.nvirb)):
            raise ValueError(
                "denominator blocks %s, %s do not match the UCISD "
                "coordinates (%d,%d), (%d,%d)"
                % (eia.shape, eIA.shape, self.nocca, self.nvira,
                   self.noccb, self.nvirb)
            )
        return numpy.concatenate((
            eia.ravel(),
            eIA.ravel(),
            (eia[:, None, :, None] + eIA[None, :, None, :]).ravel(),
        ))

    def packed_indices_uccsd(self, t1):
        nocca, nvira = t1[0].shape
        noccb, nvirb = t1[1].shape
        if (nocca, noccb, nvira, nvirb) != (
                self.nocca, self.noccb, self.nvira, self.nvirb):
            raise ValueError("UCISD coordinates do not match UCCSD amplitudes")
        sizea = (self.nt1a + comb(nocca, 2) * comb(nvira, 2))
        sizeb = (self.nt1b + comb(noccb, 2) * comb(nvirb, 2))
        flat = numpy.concatenate((
            numpy.arange(self.nt1a, dtype=numpy.int64),
            sizea + numpy.arange(self.nt1b, dtype=numpy.int64),
            sizea + sizeb + numpy.arange(self.nt2ab, dtype=numpy.int64),
        ))
        return flat, numpy.ones(len(self))

    def packed_indices_uccsdt(self, t1):
        nocca, nvira = t1[0].shape
        noccb, nvirb = t1[1].shape
        if (nocca, noccb, nvira, nvirb) != (
                self.nocca, self.noccb, self.nvira, self.nvirb):
            raise ValueError("UCISD coordinates do not match UCCSDT amplitudes")
        t2aa = comb(nocca, 2) * comb(nvira, 2)
        t2ab_offset = self.nt1a + self.nt1b + t2aa
        packed_t2ab = numpy.arange(self.nt2ab, dtype=numpy.int64).reshape(
            nocca, nvira, noccb, nvirb
        ).transpose(0, 2, 1, 3).ravel()
        flat = numpy.concatenate((
            numpy.arange(self.nt1a, dtype=numpy.int64),
            self.nt1a + numpy.arange(self.nt1b, dtype=numpy.int64),
            t2ab_offset + packed_t2ab,
        ))
        return flat, numpy.ones(len(self))


class UCISDSpinConstraint(ClosedCISDSpinConstraint):
    """CISD-closed S+ equations for arbitrary alpha and beta spatial orbitals.

    ``overlap`` is the alpha-beta MO overlap of the correlated orbitals,
    ``Ca^T S_AO Cb``, with each spin ordered occupied first.
    """

    def __init__(self, nmo, nocc, overlap, frozen_coupling=0.0):
        # super().__init__ is deliberately not called: its permutation and
        # phase setup describes a diagonal S+ that does not apply here.  The
        # inherited determinant, amplitude, and evaluation machinery needs
        # exactly the attributes established below.
        self.nmo = int(nmo)
        self.nocc = tuple(int(x) for x in nocc)
        if len(self.nocc) != 2:
            raise ValueError("nocc must contain alpha and beta counts")
        if not (0 <= self.nocc[1] <= self.nocc[0] <= self.nmo):
            raise ValueError("highest-weight constraints require nalpha >= nbeta")

        overlap = numpy.asarray(overlap)
        if overlap.shape != (self.nmo, self.nmo):
            raise ValueError("overlap has the wrong shape")
        if (numpy.iscomplexobj(overlap)
                and numpy.max(numpy.abs(overlap.imag)) > 1e-10):
            raise ValueError("complex alpha/beta orbital overlaps are not supported")
        self.overlap = numpy.asarray(overlap.real, dtype=float)
        if not numpy.all(numpy.isfinite(self.overlap)):
            raise ValueError("overlap contains non-finite elements")
        self.frozen_coupling = float(frozen_coupling)

        self._spin_mask = (1 << self.nmo) - 1
        nocca, noccb = self.nocc
        self._reference = ((1 << nocca) - 1
                           | (((1 << noccb) - 1) << self.nmo))
        self._direct_cache = {}
        self._quadratic_cache = {}
        start = perf_counter()
        self._compile_constraints()
        self.build_time = perf_counter() - start

    @classmethod
    def from_cc(cls, cc, frozen_coupling_tol=1e-8):
        """Build the engine from a UCCSD object with arbitrary alpha/beta MOs.

        Frozen orbitals are allowed only when S+ cannot carry an electron
        between the frozen and correlated spaces; otherwise the equations would
        reach determinants the CC ansatz never parameterises and closure would
        be void.  That is a correctness gate which raises -- it never selects
        equations, so the constraint count stays geometry independent.
        """
        coeffa, coeffb = (numpy.asarray(x) for x in cc.mo_coeff)
        if coeffa.shape != coeffb.shape:
            raise ValueError("alpha and beta MO coefficient shapes differ")
        ao_overlap = numpy.asarray(cc._scf.get_ovlp())
        cross = coeffa.conj().T @ ao_overlap @ coeffb
        if (numpy.iscomplexobj(cross)
                and numpy.max(numpy.abs(cross.imag)) > frozen_coupling_tol):
            raise ValueError("complex alpha/beta orbital overlaps are not supported")
        cross = numpy.asarray(cross.real, dtype=float)

        maska, maskb = (numpy.asarray(x, dtype=bool)
                        for x in cc.get_frozen_mask())
        if maska.size != coeffa.shape[1] or maskb.size != coeffb.shape[1]:
            raise ValueError("frozen masks do not match the MO dimensions")
        active_alpha = numpy.flatnonzero(maska)
        active_beta = numpy.flatnonzero(maskb)
        if active_alpha.size != active_beta.size:
            raise ValueError("alpha and beta correlated orbital counts differ")

        frozen_alpha = numpy.flatnonzero(~maska)
        frozen_beta = numpy.flatnonzero(~maskb)
        coupling = 0.0
        for rows, columns in ((frozen_alpha, active_beta),
                              (active_alpha, frozen_beta)):
            if rows.size and columns.size:
                block = cross[numpy.ix_(rows, columns)]
                coupling = max(coupling, float(numpy.max(numpy.abs(block))))
        if coupling > frozen_coupling_tol:
            raise ValueError(
                "S+ couples the frozen and correlated spaces "
                "(maximum overlap %.3e exceeds %.3e); freeze matching "
                "alpha and beta subspaces instead"
                % (coupling, frozen_coupling_tol)
            )

        occa, occb = (numpy.asarray(x) for x in cc.mo_occ)
        nocca, noccb = (int(x) for x in cc.nocc)
        for occupation, nocc, spin in (
                (occa[active_alpha], nocca, "alpha"),
                (occb[active_beta], noccb, "beta")):
            if (numpy.max(numpy.abs(occupation[:nocc] - 1.0), initial=0.0) > 1e-10
                    or numpy.max(numpy.abs(occupation[nocc:]), initial=0.0)
                    > 1e-10):
                raise ValueError(
                    "%s active MOs are not ordered as occupied then virtual"
                    % spin
                )

        return cls(
            active_alpha.size,
            (nocca, noccb),
            cross[numpy.ix_(active_alpha, active_beta)],
            frozen_coupling=coupling,
        )

    # ---------------------------------------------------------------- S+ terms

    def _raise_at_pair(self, state, alpha, beta):
        """One term of S+: annihilate beta orbital `beta`, create alpha `alpha`."""
        sequence = (
            ("annihilate", self.nmo + beta),
            ("create", alpha),
        )
        output, sign = _apply_sequence(state, sequence)
        if output is None:
            return None, 0.0
        return output, sign * self.overlap[alpha, beta]

    def _occupation(self, state):
        """Per-spin occupation label, since the two spins share no labelling."""
        alpha, beta = self._state_parts(state)
        return "%s|%s" % tuple(
            "".join("1" if bits & (1 << i) else "0" for i in range(self.nmo))
            for bits in (alpha, beta)
        )

    def _basis_phase(self, state):
        """No common orbital labelling exists, so no phase convention to undo."""
        return 1.0

    # ------------------------------------------------------------------ closure

    def _is_retained(self, output):
        """Whether every preimage of `output` lies in the SD space.

        A preimage exists for any alpha-occupied p and beta-empty q, and its
        rank is ``rank(output) + 1[p in ref_alpha] - 1[q in ref_beta]``.  The
        two indices are independent, so the worst preimage is found by
        maximising over p and minimising over q separately -- no pair loop and,
        deliberately, no reference to the size of any overlap element.
        """
        alpha, beta = self._state_parts(output)
        reference_alpha, reference_beta = self._state_parts(self._reference)
        empty_beta = ~beta & self._spin_mask
        if not alpha or not empty_beta:
            return False
        raise_from_occupied = 1 if alpha & reference_alpha else 0
        lower_into_empty = 0 if empty_beta & ~reference_beta else 1
        shift = raise_from_occupied - lower_into_empty
        return self._excitation_rank(output) + shift <= 2

    def _candidate_outputs(self):
        """Retained outputs and the closed-form count of all candidates.

        Dense unmatched alpha/beta coupling makes a rank-r output have a
        rank-(r+1) preimage whenever one exists.  CISD closure therefore keeps
        only the rank-one ``(0, 1)`` shell.  The other five shells are counted
        combinatorially for diagnostics but never materialized.
        """
        raised = dict(delta_alpha=1, delta_beta=-1)
        shells = ((0, 1), (1, 1), (0, 2), (2, 1), (1, 2), (0, 3))
        nocca, noccb = self.nocc
        nvira, nvirb = self.nmo - nocca, self.nmo - noccb

        def choose(n, k):
            return comb(n, k) if 0 <= k <= n else 0

        def shell_size(holes_a, holes_b):
            particles_a = holes_a + 1
            particles_b = holes_b - 1
            return (choose(nocca, holes_a) * choose(nvira, particles_a)
                    * choose(noccb, holes_b) * choose(nvirb, particles_b))

        ncandidate = sum(shell_size(*shell) for shell in shells)
        retained = self._determinant_shell(0, 1, **raised)
        return retained, ncandidate

    def _compile_constraints(self):
        """Build the closed-form rank-one UCISD equations.

        For output ``S+ |0>`` indexed by beta hole J and alpha particle a,
        every retained CISD preimage belongs to one of four tensor blocks:
        reference, T1a, T1b, or T2ab.  Constructing determinant strings and
        Python term dictionaries for these known Cartesian products is pure
        overhead, especially for large UHF calculations.
        """
        nocca, noccb = self.nocc
        nvira, nvirb = self.nmo - nocca, self.nmo - noccb
        self._soo = self.overlap[:nocca, :noccb]
        self._sov = self.overlap[:nocca, noccb:]
        self._svo = self.overlap[nocca:, :noccb]
        self._svv = self.overlap[nocca:, noccb:]
        self._row_phases = 1.0 - 2.0 * (numpy.arange(noccb) % 2)

        self.coordinates = _UCISDCoordinates(self.nmo, self.nocc)
        self.nconstraint = noccb * nvira
        shells = ((0, 1), (1, 1), (0, 2), (2, 1), (1, 2), (0, 3))

        def choose(n, k):
            return comb(n, k) if 0 <= k <= n else 0

        def shell_size(holes_a, holes_b):
            return (choose(nocca, holes_a) * choose(nvira, holes_a + 1)
                    * choose(noccb, holes_b) * choose(nvirb, holes_b - 1))

        self.ncandidate = sum(shell_size(*shell) for shell in shells)
        self.nomitted = self.ncandidate - self.nconstraint
        self.ndeterminant = self._sd_space_size()
        self.constant = self._format_rows(self._svo.T)

    def _format_rows(self, values):
        values = self._row_phases[:, None] * numpy.asarray(values)
        return values[::-1].ravel()

    def _unformat_weights(self, weights):
        weights = numpy.asarray(weights, dtype=float)
        if weights.shape != (self.nconstraint,):
            raise ValueError("constraint weights have the wrong shape")
        return self._row_phases[:, None] * weights.reshape(
            self.nocc[1], self.nmo - self.nocc[0]
        )[::-1]

    def _split_vector(self, vector):
        vector = numpy.asarray(vector, dtype=float)
        if vector.shape != (len(self.coordinates),):
            raise ValueError("constraint coordinate vector has the wrong shape")
        stopa = self.coordinates.nt1a
        stopb = stopa + self.coordinates.nt1b
        nocca, noccb = self.nocc
        nvira, nvirb = self.nmo - nocca, self.nmo - noccb
        return (
            vector[:stopa].reshape(nocca, nvira),
            vector[stopa:stopb].reshape(noccb, nvirb),
            vector[stopb:].reshape(nocca, noccb, nvira, nvirb),
        )

    def amplitude_values(self, t1, t2):
        return self.coordinates.amplitude_values(t1, t2)

    def _gradient_factors(self, t1):
        t1a, t1b = (numpy.asarray(block) for block in t1)
        left = self._soo + numpy.einsum(
            "iB,JB->iJ", self._sov, t1b, optimize=True
        )
        right = self._svv - numpy.einsum(
            "ia,iB->aB", t1a, self._sov, optimize=True
        )
        return left, right

    def constraint(self, t1, t2):
        t1a, t1b = (numpy.asarray(block) for block in t1)
        t2ab = numpy.asarray(t2[1])
        values = self._svo.T.copy()
        values -= numpy.einsum(
            "iJ,ia->Ja", self._soo, t1a, optimize=True
        )
        values += numpy.einsum(
            "aB,JB->Ja", self._svv, t1b, optimize=True
        )
        values -= numpy.einsum(
            "iB,iJaB->Ja", self._sov, t2ab, optimize=True
        )
        values -= numpy.einsum(
            "iB,ia,JB->Ja", self._sov, t1a, t1b, optimize=True
        )
        return self._format_rows(values)

    def jvp(self, t1, t2, vector):
        x1a, x1b, x2ab = self._split_vector(vector)
        left, right = self._gradient_factors(t1)
        values = -numpy.einsum(
            "iJ,ia->Ja", left, x1a, optimize=True
        )
        values += numpy.einsum(
            "aB,JB->Ja", right, x1b, optimize=True
        )
        values -= numpy.einsum(
            "iB,iJaB->Ja", self._sov, x2ab, optimize=True
        )
        return self._format_rows(values)

    def vjp(self, t1, t2, weights):
        """Contract the constraint Jacobian's transpose with `weights`.

        Written straight into the output vector rather than built as three
        arrays and concatenated.  The t2ab block is an outer product -- no
        index is summed in "Ja,iB->iJaB" -- so it is the full coordinate
        vector in size with one multiply per element, and materializing it
        twice made this the most memory-bound routine in the constrained
        solver: profiling put ~50 s of a 560 s run in the concatenate alone.
        Signs are folded into `weights`, which is small, instead of negating
        the result afterwards.
        """
        weights = self._unformat_weights(weights)
        left, right = self._gradient_factors(t1)
        coords = self.coordinates
        stopa = coords.nt1a
        stopb = stopa + coords.nt1b
        out = numpy.empty(stopb + coords.nt2ab)
        negated = -weights
        # Deliberately not the naive form, which reads more clearly:
        #
        #     grad1a = -numpy.einsum("Ja,iJ->ia", weights, left)
        #     grad1b =  numpy.einsum("Ja,aB->JB", weights, right)
        #     grad2ab = -numpy.einsum("Ja,iB->iJaB", weights, self._sov)
        #     return numpy.concatenate((grad1a.ravel(), grad1b.ravel(),
        #                               grad2ab.ravel()))
        #
        # That writes the t2ab block once in einsum and copies it again in
        # concatenate.  The block is the whole coordinate vector in size --
        # 19 MB in cc-pvtz-dk, 119 MB in cc-pv5z-dk -- and neither the
        # broadcast nor the copy threads, so the second pass is pure cost:
        # timed on its own the concatenate is 26% of this routine at TZ and
        # 42% at 5Z.  Writing into views of `out` removes it.  Likewise the
        # sign is folded into `weights`, which is small, rather than negating
        # a full-length result afterwards.
        #
        # Keep any future edit in this shape: the three `out=` targets must
        # stay non-overlapping views of one contiguous buffer, in the
        # coordinate order t1a, t1b, t2ab.
        numpy.einsum(
            "Ja,iJ->ia", negated, left, optimize=True,
            out=out[:stopa].reshape(coords.nocca, coords.nvira),
        )
        numpy.einsum(
            "Ja,aB->JB", weights, right, optimize=True,
            out=out[stopa:stopb].reshape(coords.noccb, coords.nvirb),
        )
        numpy.einsum(
            "Ja,iB->iJaB", negated, self._sov, optimize=True,
            out=out[stopb:].reshape(coords.nocca, coords.noccb,
                                    coords.nvira, coords.nvirb),
        )
        return out

    def jacobian(self, t1, t2):
        size = self.nconstraint * len(self.coordinates)
        if size * 8 > 2 * 1024 ** 3:
            raise MemoryError(
                "dense UCISD Jacobian would require %.1f GiB; use jvp/vjp"
                % (size * 8 / 1024 ** 3)
            )
        jacobian = numpy.empty((self.nconstraint, len(self.coordinates)))
        basis = numpy.zeros(self.nconstraint)
        for row in range(self.nconstraint):
            basis[row] = 1.0
            jacobian[row] = self.vjp(t1, t2, basis)
            basis[row] = 0.0
        return jacobian

    def jacobian_row_norms(self, t1, t2):
        left, right = self._gradient_factors(t1)
        squares = (numpy.sum(left ** 2, axis=0)[:, None]
                   + numpy.sum(right ** 2, axis=1)[None, :]
                   + numpy.sum(self._sov ** 2))
        return squares[::-1].ravel() ** 0.5

    def weighted_gram_diagonal(self, t1, t2, metric):
        metric1a, metric1b, metric2ab = self._split_vector(metric)
        left, right = self._gradient_factors(t1)
        diagonal = numpy.einsum(
            "iJ,ia->Ja", left ** 2, metric1a, optimize=True
        )
        diagonal += numpy.einsum(
            "aB,JB->Ja", right ** 2, metric1b, optimize=True
        )
        diagonal += numpy.einsum(
            "iB,iJaB->Ja", self._sov ** 2, metric2ab, optimize=True
        )
        return diagonal[::-1].ravel()

    def hessian_vector(self, t1, t2, weights, vector):
        weights = self._unformat_weights(weights)
        x1a, x1b, x2ab = self._split_vector(vector)
        out1a = -numpy.einsum(
            "Ja,iB,JB->ia", weights, self._sov, x1b, optimize=True
        )
        out1b = -numpy.einsum(
            "Ja,iB,ia->JB", weights, self._sov, x1a, optimize=True
        )
        return numpy.concatenate((out1a.ravel(), out1b.ravel(),
                                  numpy.zeros_like(x2ab).ravel()))

    def _complete_preimages(self, output):
        output_alpha, output_beta = self._state_parts(output)
        preimages = []
        for alpha in range(self.nmo):
            alpha_bit = 1 << alpha
            if not output_alpha & alpha_bit:
                continue
            for beta in range(self.nmo):
                beta_bit = 1 << beta
                if output_beta & beta_bit:
                    continue
                state = ((output_alpha ^ alpha_bit)
                         | ((output_beta | beta_bit) << self.nmo))
                check, coefficient = self._raise_at_pair(state, alpha, beta)
                if check != output:
                    raise RuntimeError("failed to invert a spin-raising operation")
                preimages.append((state, coefficient))
        return preimages

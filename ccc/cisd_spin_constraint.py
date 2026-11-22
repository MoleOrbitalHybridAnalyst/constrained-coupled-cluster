from dataclasses import dataclass
from itertools import combinations
from math import comb
from time import perf_counter

import numpy
from scipy.optimize import linear_sum_assignment


BLOCKS = ("t1a", "t1b", "t2aa", "t2ab", "t2bb")
_BLOCK_ORDER = {name: i for i, name in enumerate(BLOCKS)}


@dataclass(frozen=True)
class SpinPreimage:
    occupation: str
    coefficient: float
    excitation_rank: int


@dataclass(frozen=True)
class SpinEquation:
    output: str
    preimages: tuple


def _coordinate_key(coordinate):
    block, index = coordinate
    return (_BLOCK_ORDER[block],) + tuple(index)


def _annihilate(state, orbital):
    bit = 1 << orbital
    if not state & bit:
        return None, 0.0
    sign = -1.0 if (state & (bit - 1)).bit_count() % 2 else 1.0
    return state ^ bit, sign


def _create(state, orbital):
    bit = 1 << orbital
    if state & bit:
        return None, 0.0
    sign = -1.0 if (state & (bit - 1)).bit_count() % 2 else 1.0
    return state | bit, sign


def _apply_sequence(state, sequence):
    sign = 1.0
    for action, orbital in sequence:
        if action == "annihilate":
            state, phase = _annihilate(state, orbital)
        else:
            state, phase = _create(state, orbital)
        if state is None:
            return None, 0.0
        sign *= phase
    return state, sign


def _consolidated_arrays(terms, width):
    kept = [(key, value) for key, value in terms.items()
            if abs(value) > 1e-14]
    kept.sort(key=lambda item: item[0])
    if not kept:
        return tuple(numpy.zeros(0, dtype=numpy.int64)
                     for _ in range(width)), numpy.zeros(0)
    indices = tuple(numpy.asarray([item[0][i] for item in kept],
                                  dtype=numpy.int64)
                    for i in range(width))
    values = numpy.asarray([item[1] for item in kept], dtype=float)
    return indices, values


class ClosedCISDSpinConstraint:
    """Highest-weight FCI spin equations that close within the CISD space.

    Orbital permutations map PySCF's occupied-first alpha and beta MO orders
    to a common spatial-orbital labeling.  The generated equations retain an
    output of S+ only when every one of its FCI preimages is a reference,
    single, or double determinant.
    """

    def __init__(self, nmo, nocc, alpha_permutation=None,
                 beta_permutation=None, beta_phases=None,
                 orbital_match_error=0.0):
        self.nmo = int(nmo)
        self.nocc = tuple(int(x) for x in nocc)
        if len(self.nocc) != 2:
            raise ValueError("nocc must contain alpha and beta counts")
        if not (0 <= self.nocc[1] <= self.nocc[0] <= self.nmo):
            raise ValueError("highest-weight constraints require nalpha >= nbeta")

        expected = numpy.arange(self.nmo)
        if alpha_permutation is None:
            alpha_permutation = expected
        if beta_permutation is None:
            beta_permutation = expected
        self.alpha_permutation = numpy.asarray(alpha_permutation, dtype=int)
        self.beta_permutation = numpy.asarray(beta_permutation, dtype=int)
        if (not numpy.array_equal(numpy.sort(self.alpha_permutation), expected)
                or not numpy.array_equal(
                    numpy.sort(self.beta_permutation), expected)):
            raise ValueError("alpha and beta orbital mappings must be permutations")
        self.common_to_alpha = numpy.argsort(self.alpha_permutation)
        self.common_to_beta = numpy.argsort(self.beta_permutation)

        if beta_phases is None:
            beta_phases = numpy.ones(self.nmo)
        beta_phases = numpy.asarray(beta_phases)
        if beta_phases.shape != (self.nmo,):
            raise ValueError("beta_phases has the wrong shape")
        if (numpy.iscomplexobj(beta_phases)
                and numpy.max(numpy.abs(beta_phases.imag)) > 1e-10):
            raise ValueError("complex alpha/beta orbital phases are not supported")
        self.beta_phases = numpy.asarray(beta_phases.real, dtype=float)
        if numpy.max(numpy.abs(numpy.abs(self.beta_phases) - 1.0)) > 1e-8:
            raise ValueError("matched alpha/beta orbital phases must have magnitude one")
        self.orbital_match_error = float(orbital_match_error)

        self._spin_mask = (1 << self.nmo) - 1
        nocca, noccb = self.nocc
        self._reference = ((1 << nocca) - 1
                           | (((1 << noccb) - 1) << self.nmo))
        self._shift = self._shift_values()
        self._any_shift = any(self._shift)
        self._direct_cache = {}
        self._quadratic_cache = {}
        # Compiling the constraints is the expensive part of setting up a
        # constrained CC run, and it grows quickly with the orbital count, so the
        # cost is recorded for the caller to report.  Kept as a plain attribute
        # because this module deliberately has no pyscf dependency and therefore
        # no logger of its own.
        start = perf_counter()
        self._compile_constraints()
        self.build_time = perf_counter() - start
        # Only the compilation stage needs these; a long-lived engine should not
        # keep a determinant-keyed table alive for the rest of the calculation.
        self._direct_cache = {}
        self._quadratic_cache = {}

        self._cache_values = None
        self._cache_residual = None
        self._cache_derivative_values = None

    @classmethod
    def from_cc(cls, cc, orbital_match_tol=1e-8):
        coeffa, coeffb = (numpy.asarray(x) for x in cc.mo_coeff)
        if coeffa.shape != coeffb.shape:
            raise ValueError("alpha and beta MO coefficient shapes differ")
        overlap = numpy.asarray(cc._scf.get_ovlp())
        cross = coeffa.conj().T @ overlap @ coeffb
        rows, columns = linear_sum_assignment(-numpy.abs(cross))
        beta_to_alpha = numpy.empty(cross.shape[1], dtype=int)
        beta_to_alpha[columns] = rows
        matched = cross[beta_to_alpha, numpy.arange(cross.shape[1])]
        off = cross.copy()
        off[beta_to_alpha, numpy.arange(cross.shape[1])] = 0
        error = max(
            float(numpy.max(numpy.abs(numpy.abs(matched) - 1.0))),
            float(numpy.max(numpy.abs(off))) if off.size else 0.0,
        )
        if error > orbital_match_tol:
            raise ValueError(
                "alpha and beta MOs are not a signed permutation "
                "(maximum mismatch %.3e exceeds %.3e)"
                % (error, orbital_match_tol)
            )
        if (numpy.iscomplexobj(matched)
                and numpy.max(numpy.abs(matched.imag)) > orbital_match_tol):
            raise ValueError("complex alpha/beta orbital phases are not supported")

        maska, maskb = (numpy.asarray(x, dtype=bool)
                        for x in cc.get_frozen_mask())
        occa, occb = (numpy.asarray(x) for x in cc.mo_occ)
        if maska.size != coeffa.shape[1] or maskb.size != coeffb.shape[1]:
            raise ValueError("frozen masks do not match the MO dimensions")
        for beta, alpha in enumerate(beta_to_alpha):
            if maska[alpha] != maskb[beta]:
                raise ValueError(
                    "alpha and beta frozen orbitals are not spatially matched"
                )
            if not maskb[beta]:
                alpha_occ = bool(occa[alpha] > 0.5)
                beta_occ = bool(occb[beta] > 0.5)
                if alpha_occ != beta_occ:
                    raise ValueError("a frozen spatial orbital is open shell")

        active_alpha = numpy.flatnonzero(maska)
        active_beta = numpy.flatnonzero(maskb)
        if active_alpha.size != active_beta.size:
            raise ValueError("alpha and beta correlated orbital counts differ")
        alpha_common = {int(full): i
                        for i, full in enumerate(active_alpha)}
        beta_permutation = numpy.asarray([
            alpha_common[int(beta_to_alpha[full])] for full in active_beta
        ], dtype=int)
        beta_phases = numpy.asarray(matched[active_beta].real, dtype=float)

        active_occa = occa[active_alpha]
        active_occb = occb[active_beta]
        nocca, noccb = (int(x) for x in cc.nocc)
        for occupation, nocc, spin in (
                (active_occa, nocca, "alpha"),
                (active_occb, noccb, "beta")):
            if (numpy.max(numpy.abs(occupation[:nocc] - 1.0), initial=0.0)
                    > 1e-10
                    or numpy.max(numpy.abs(occupation[nocc:]), initial=0.0)
                    > 1e-10):
                raise ValueError(
                    "%s active MOs are not ordered as occupied then virtual"
                    % spin
                )

        return cls(
            active_alpha.size,
            (nocca, noccb),
            alpha_permutation=numpy.arange(active_alpha.size),
            beta_permutation=beta_permutation,
            beta_phases=beta_phases,
            orbital_match_error=error,
        )

    def _state_parts(self, state):
        return state & self._spin_mask, state >> self.nmo

    def _occupation(self, state):
        alpha, beta = self._state_parts(state)
        alpha_common = {
            int(self.alpha_permutation[i])
            for i in range(self.nmo) if alpha & (1 << i)
        }
        beta_common = {
            int(self.beta_permutation[i])
            for i in range(self.nmo) if beta & (1 << i)
        }
        return "".join(
            "2" if i in alpha_common and i in beta_common
            else "a" if i in alpha_common
            else "b" if i in beta_common
            else "0"
            for i in range(self.nmo)
        )

    def _basis_phase(self, state):
        alpha, beta = self._state_parts(state)
        alpha_sequence = [
            int(self.alpha_permutation[i])
            for i in range(self.nmo) if alpha & (1 << i)
        ]
        beta_local = [i for i in range(self.nmo) if beta & (1 << i)]
        beta_sequence = [int(self.beta_permutation[i]) for i in beta_local]

        inversions = 0
        for sequence in (alpha_sequence, beta_sequence):
            inversions += sum(
                sequence[i] > sequence[j]
                for i in range(len(sequence))
                for j in range(i + 1, len(sequence))
            )
        phase = -1.0 if inversions % 2 else 1.0
        for orbital in beta_local:
            phase *= self.beta_phases[orbital]
        return phase

    def _excitation_rank(self, state):
        alpha, beta = self._state_parts(state)
        reference_alpha, reference_beta = self._state_parts(self._reference)
        return ((reference_alpha & ~alpha).bit_count()
                + (reference_beta & ~beta).bit_count())

    def _raise_at_common(self, state, common):
        alpha = int(self.common_to_alpha[common])
        beta = int(self.common_to_beta[common])
        sequence = (
            ("annihilate", self.nmo + beta),
            ("create", alpha),
        )
        output, sign = _apply_sequence(state, sequence)
        if output is None:
            return None, 0.0
        return output, sign * self.beta_phases[beta]

    def _shift_values(self):
        """s(p) = [p alpha-occupied in the reference] - [p beta-occupied].

        Lowering an output at common orbital p removes an alpha electron there
        and adds a beta one, so it opens an alpha hole only when p is
        alpha-occupied in the reference and fills a beta hole only when p is
        beta-occupied.  Excitation rank counts holes, hence

            rank(preimage at p) = rank(output) + s(p)

        which is what lets SD closure be tested arithmetically instead of by
        constructing every preimage.  s(p) is 0 on doubly-occupied and empty
        orbitals, +1 on the singly-occupied ones, and -1 only where the beta
        orbital set is not contained in the alpha one.
        """
        refa, refb = self._state_parts(self._reference)
        return tuple(
            int(bool(refa >> int(self.common_to_alpha[p]) & 1))
            - int(bool(refb >> int(self.common_to_beta[p]) & 1))
            for p in range(self.nmo)
        )

    def _valid_shifts(self, output):
        """s(p) at the common orbitals where `output` has a preimage at all."""
        alpha, beta = self._state_parts(output)
        return [
            self._shift[p] for p in range(self.nmo)
            if (alpha >> int(self.common_to_alpha[p]) & 1)
            and not (beta >> int(self.common_to_beta[p]) & 1)
        ]

    def _is_retained(self, output):
        """Whether every preimage of `output` lies in the SD space.

        SD membership is exactly rank <= 2, because holes and particles balance
        within a fixed electron-number sector, so with the shift identity this
        is one comparison against the worst preimage position -- no preimage need
        be built to reject an output.

        When no orbital carries a nonzero shift, which happens exactly when the
        alpha and beta reference sets coincide, every generated candidate passes
        and the scan over positions is skipped: that case retains nearly
        everything, so the scan would be pure overhead.
        """
        rank = self._excitation_rank(output)
        if not self._any_shift:
            return rank <= 2
        shifts = self._valid_shifts(output)
        if not shifts:
            return False
        return rank + max(shifts) <= 2

    def _determinant_shell(self, nhole_alpha, nhole_beta,
                           delta_alpha=0, delta_beta=0):
        """Determinants with given hole counts and electron-count offsets.

        Electron conservation fixes the particle counts at ``nhole + delta`` per
        spin, so ``delta = (0, 0)`` walks the SD space and ``(+1, -1)`` walks the
        spin-raised output sector.  Total excitation rank is the sum of the hole
        counts.
        """
        spins = []
        for nocc, nhole, delta in zip(self.nocc,
                                      (nhole_alpha, nhole_beta),
                                      (delta_alpha, delta_beta)):
            nparticle = nhole + delta
            if nhole < 0 or nparticle < 0:
                return ()
            states = []
            for holes in combinations(range(nocc), nhole):
                base = (1 << nocc) - 1
                for orbital in holes:
                    base ^= 1 << orbital
                for particles in combinations(range(nocc, self.nmo), nparticle):
                    state = base
                    for orbital in particles:
                        state |= 1 << orbital
                    states.append(state)
            spins.append(states)
        return tuple(alpha | (beta << self.nmo)
                     for alpha in spins[0] for beta in spins[1])

    def _sd_space_size(self):
        """Size of the reference-plus-SD determinant space."""
        nocca, noccb = self.nocc
        nvira, nvirb = self.nmo - nocca, self.nmo - noccb
        return (1
                + nocca * nvira + noccb * nvirb
                + comb(nocca, 2) * comb(nvira, 2)
                + nocca * noccb * nvira * nvirb
                + comb(noccb, 2) * comb(nvirb, 2))

    def _sd_determinants(self):
        """Reference, singles, and doubles, as determinant bit patterns."""
        for nhole_alpha, nhole_beta in ((0, 0), (1, 0), (0, 1),
                                        (2, 0), (1, 1), (0, 2)):
            for state in self._determinant_shell(nhole_alpha, nhole_beta):
                yield state

    def _candidate_outputs(self):
        """Reachable outputs to test, and the total candidate count.

        When the beta reference orbitals sit inside the alpha ones every s(p) is
        non-negative, so no candidate can exceed rank two and the three
        (alpha hole, beta hole) shells reaching it can be listed directly --
        the SD space is never built.  Every state in those shells really is
        reachable: the beta hole itself is a preimage position with s = 0 for the
        (0,1) and (0,2) shells, and for (1,1) either the beta hole or one of the
        two alpha particles serves.

        With two or more unpaired electrons the rank-two shells are provably all
        discarded, so only the rank-one shell is generated and the rank-two count
        is taken from the shell sizes.  Reasoning: for a rank-two output, keeping
        it needs no unpaired orbital to be a preimage position.  Splitting the
        rank as (alpha, beta) = (1,1) leaves a second unpaired orbital available
        once there are two of them, and (0,2) would need two beta electrons
        outside the beta reference set, forcing beta rank three.  At one unpaired
        electron the second case closes exactly, which is why doublet and
        closed-shell references do keep rank-two equations.

        A non-nested reference admits s(p) = -1 and hence candidates above rank
        two, so it falls back to scanning the SD space.
        """
        nocca, noccb = self.nocc
        nvira, nvirb = self.nmo - nocca, self.nmo - noccb

        if all(shift >= 0 for shift in self._shift):
            raised = dict(delta_alpha=1, delta_beta=-1)
            rank_one = self._determinant_shell(0, 1, **raised)
            if nocca - noccb >= 2:
                # Sizes of the two discarded rank-two shells, (1,1) and (0,2).
                nrank_two = (nocca * comb(nvira, 2) * noccb
                             + nvira * comb(noccb, 2) * nvirb)
                return rank_one, len(rank_one) + nrank_two
            outputs = (rank_one
                       + self._determinant_shell(1, 1, **raised)
                       + self._determinant_shell(0, 2, **raised))
            return outputs, len(outputs)

        outputs = set()
        for state in self._sd_determinants():
            for common in range(self.nmo):
                output, coefficient = self._raise_at_common(state, common)
                if output is not None and coefficient != 0.0:
                    outputs.add(output)
        return outputs, len(outputs)

    def _hole_particle_pattern(self, state):
        """Holes and particles of `state` relative to the reference, per spin."""
        nocca, noccb = self.nocc
        alpha, beta = self._state_parts(state)
        return (
            [i for i in range(nocca) if not alpha >> i & 1],
            [i for i in range(noccb) if not beta >> i & 1],
            [i - nocca for i in range(nocca, self.nmo) if alpha >> i & 1],
            [i - noccb for i in range(noccb, self.nmo) if beta >> i & 1],
        )

    def _single_sequence(self, coordinate):
        """Operator sequence for one single-excitation coordinate."""
        nocca, noccb = self.nocc
        block, (i, a) = coordinate
        if block == "t1a":
            return (("annihilate", i), ("create", nocca + a))
        return (("annihilate", self.nmo + i),
                ("create", self.nmo + noccb + a))

    def _excitation(self, holes_a, holes_b, particles_a, particles_b):
        """Canonical coordinate and operator sequence for a hole/particle pattern.

        The index order matches the canonical CC convention: same-spin blocks
        carry the larger occupied and virtual index first.
        """
        nocca, noccb = self.nocc
        pattern = (len(holes_a), len(holes_b))
        if pattern == (1, 0):
            i, a = holes_a[0], particles_a[0]
            return ("t1a", (i, a)), self._single_sequence(("t1a", (i, a)))
        if pattern == (0, 1):
            i, a = holes_b[0], particles_b[0]
            return ("t1b", (i, a)), self._single_sequence(("t1b", (i, a)))
        if pattern == (2, 0):
            j, i = holes_a
            b, a = particles_a
            return (("t2aa", (i, j, a, b)),
                    (("annihilate", i), ("annihilate", j),
                     ("create", nocca + b), ("create", nocca + a)))
        if pattern == (1, 1):
            i, j = holes_a[0], holes_b[0]
            a, b = particles_a[0], particles_b[0]
            return (("t2ab", (i, j, a, b)),
                    (("annihilate", i), ("annihilate", self.nmo + j),
                     ("create", self.nmo + noccb + b), ("create", nocca + a)))
        if pattern == (0, 2):
            j, i = holes_b
            b, a = particles_b
            return (("t2bb", (i, j, a, b)),
                    (("annihilate", self.nmo + i), ("annihilate", self.nmo + j),
                     ("create", self.nmo + noccb + b),
                     ("create", self.nmo + noccb + a)))
        raise RuntimeError("determinant is not a single or double excitation")

    def _direct_amplitude(self, state):
        """The canonical CC amplitude whose excitation generates `state`.

        Replaces the precomputed determinant-to-amplitude table: the holes and
        particles name the amplitude, and the sign is regenerated by replaying
        the same operator sequence the table was built from, so the phase
        convention is inherited rather than restated.

        Memoized because a closed-shell reference retains almost every equation
        and so revisits the same preimages many times, whereas a high-spin one
        touches only a few thousand determinants -- caching on demand keeps the
        former from recomputing without making the latter tabulate the whole
        SD space.
        """
        cached = self._direct_cache.get(state)
        if cached is not None:
            return cached
        coordinate, sequence = self._excitation(
            *self._hole_particle_pattern(state)
        )
        result, sign = _apply_sequence(self._reference, sequence)
        if result != state:
            raise RuntimeError(
                "a canonical CC excitation does not reproduce its determinant"
            )
        self._direct_cache[state] = (coordinate, sign)
        return coordinate, sign

    def _quadratic_terms(self, state):
        """T1-squared coefficients contributing to `state`.

        A double determinant factorizes into two single excitations one way for
        mixed spin and two ways for same spin; singles and the reference admit
        none, because a repeated orbital annihilates the intermediate.  Both
        operator orderings are replayed and weighted one half each, exactly as
        the tabulated version accumulated them, so no commutation argument is
        assumed.  Memoized for the same reason as `_direct_amplitude`.
        """
        cached = self._quadratic_cache.get(state)
        if cached is not None:
            return cached
        holes_a, holes_b, particles_a, particles_b = \
            self._hole_particle_pattern(state)
        pattern = (len(holes_a), len(holes_b))
        if pattern == (2, 0):
            j, i = holes_a
            b, a = particles_a
            factorizations = ((("t1a", (i, a)), ("t1a", (j, b))),
                              (("t1a", (i, b)), ("t1a", (j, a))))
        elif pattern == (0, 2):
            j, i = holes_b
            b, a = particles_b
            factorizations = ((("t1b", (i, a)), ("t1b", (j, b))),
                              (("t1b", (i, b)), ("t1b", (j, a))))
        elif pattern == (1, 1):
            factorizations = ((("t1a", (holes_a[0], particles_a[0])),
                               ("t1b", (holes_b[0], particles_b[0]))),)
        else:
            self._quadratic_cache[state] = {}
            return {}

        terms = {}
        for first, second in factorizations:
            for left, right in ((first, second), (second, first)):
                intermediate, sign_right = _apply_sequence(
                    self._reference, self._single_sequence(right)
                )
                if intermediate is None:
                    continue
                result, sign_left = _apply_sequence(
                    intermediate, self._single_sequence(left)
                )
                if result != state:
                    continue
                pair = tuple(sorted((left, right), key=_coordinate_key))
                terms[pair] = terms.get(pair, 0.0) + 0.5 * sign_left * sign_right
        terms = {pair: value for pair, value in terms.items()
                 if abs(value) > 1e-14}
        self._quadratic_cache[state] = terms
        return terms

    def _build_sd_space(self):
        """Materialize the reference-plus-SD tables the compiler once precomputed.

        Compilation no longer needs these, because it decodes only the few
        determinants it actually touches, but building the whole space from those
        same decoders is a useful check on them and is what the test suite
        compares against the exact cluster exponential.
        """
        direct = {}
        quadratic = {}
        determinants = set()
        for state in self._sd_determinants():
            determinants.add(state)
            if state == self._reference:
                continue
            direct[state] = self._direct_amplitude(state)
            terms = self._quadratic_terms(state)
            if terms:
                quadratic[state] = terms
        self._direct_cache = {}
        self._quadratic_cache = {}
        return {
            "direct": direct,
            "quadratic": quadratic,
            "determinants": determinants,
        }

    def _complete_preimages(self, output):
        output_alpha, output_beta = self._state_parts(output)
        preimages = []
        for common in range(self.nmo):
            alpha = int(self.common_to_alpha[common])
            beta = int(self.common_to_beta[common])
            alpha_bit = 1 << alpha
            beta_bit = 1 << beta
            if output_alpha & alpha_bit and not output_beta & beta_bit:
                input_alpha = output_alpha ^ alpha_bit
                input_beta = output_beta | beta_bit
                state = input_alpha | (input_beta << self.nmo)
                check, coefficient = self._raise_at_common(state, common)
                if check != output:
                    raise RuntimeError("failed to invert a spin-raising operation")
                preimages.append((state, coefficient))
        return preimages

    def _compile_constraints(self):
        candidates, ncandidate = self._candidate_outputs()

        retained = []
        for output in sorted(candidates):
            if self._is_retained(output):
                retained.append((output, self._complete_preimages(output)))
        omitted = ncandidate - len(retained)

        coordinate_set = set()
        constant = numpy.zeros(len(retained))
        linear_raw = {}
        quadratic_raw = {}
        equations = []
        for row, (output, preimages) in enumerate(retained):
            metadata = []
            output_phase = self._basis_phase(output)
            for state, spin_coefficient in preimages:
                rank = self._excitation_rank(state)
                metadata.append(SpinPreimage(
                    self._occupation(state),
                    float(spin_coefficient
                          / (output_phase * self._basis_phase(state))),
                    rank,
                ))
                if state == self._reference:
                    constant[row] += spin_coefficient
                else:
                    coordinate, direct_coefficient = self._direct_amplitude(state)
                    coordinate_set.add(coordinate)
                    key = (row, coordinate)
                    linear_raw[key] = (linear_raw.get(key, 0.0)
                                       + spin_coefficient * direct_coefficient)
                for pair, coefficient in self._quadratic_terms(state).items():
                    coordinate_set.update(pair)
                    key = (row,) + pair
                    quadratic_raw[key] = (
                        quadratic_raw.get(key, 0.0)
                        + spin_coefficient * coefficient
                    )
            equations.append(SpinEquation(
                self._occupation(output), tuple(metadata)
            ))

        self.coordinates = tuple(sorted(coordinate_set, key=_coordinate_key))
        coordinate_index = {
            coordinate: i for i, coordinate in enumerate(self.coordinates)
        }
        linear = {
            (row, coordinate_index[coordinate]): value
            for (row, coordinate), value in linear_raw.items()
        }
        quadratic_compact = {
            (row, coordinate_index[first], coordinate_index[second]): value
            for (row, first, second), value in quadratic_raw.items()
        }
        linear_indices, self.linear_values = _consolidated_arrays(linear, 2)
        self.linear_rows, self.linear_columns = linear_indices
        quadratic_indices, self.quadratic_values = _consolidated_arrays(
            quadratic_compact, 3
        )
        (self.quadratic_rows, self.quadratic_first,
         self.quadratic_second) = quadratic_indices
        self.constant = constant
        self.equations = tuple(equations)
        self.nconstraint = len(retained)
        self.nomitted = omitted
        self.ncandidate = ncandidate
        self.ndeterminant = self._sd_space_size()
        self._gather = self._build_gather()
        self._compile_derivative_structure()

    def _compile_derivative_structure(self):
        """Compile the fixed sparse pattern of the constraint Jacobian."""
        rows = numpy.concatenate((
            self.linear_rows,
            self.quadratic_rows,
            self.quadratic_rows,
        ))
        columns = numpy.concatenate((
            self.linear_columns,
            self.quadratic_first,
            self.quadratic_second,
        ))
        order = numpy.lexsort((columns, rows))
        sorted_rows = rows[order]
        sorted_columns = columns[order]
        first = numpy.ones(rows.size, dtype=bool)
        if rows.size > 1:
            first[1:] = ((sorted_rows[1:] != sorted_rows[:-1])
                         | (sorted_columns[1:] != sorted_columns[:-1]))
        self.derivative_rows = sorted_rows[first]
        self.derivative_columns = sorted_columns[first]

        sorted_slots = numpy.cumsum(first, dtype=numpy.int64) - 1
        slots = numpy.empty(rows.size, dtype=numpy.int64)
        slots[order] = sorted_slots
        nlinear = self.linear_values.size
        nquadratic = self.quadratic_values.size
        self._linear_derivative_slots = slots[:nlinear]
        self._quadratic_first_slots = slots[
            nlinear:nlinear + nquadratic
        ]
        self._quadratic_second_slots = slots[nlinear + nquadratic:]
        self._linear_derivative_values = numpy.zeros(
            self.derivative_rows.size
        )
        if self.linear_values.size:
            numpy.add.at(
                self._linear_derivative_values,
                self._linear_derivative_slots,
                self.linear_values,
            )

    @staticmethod
    def _blocks(t1, t2):
        return dict(zip(BLOCKS, t1 + t2))

    def _build_gather(self):
        """Advanced-index gather for `coordinates`, grouped by amplitude block.

        Reading the constrained amplitudes is otherwise a Python loop over every
        coordinate, repeated on each evaluation.  The indices depend only on
        `coordinates`, which is fixed once the constraints are compiled, so they
        are built here rather than rediscovered per call.
        """
        gather = []
        for name in BLOCKS:
            positions = [n for n, (block, _) in enumerate(self.coordinates)
                         if block == name]
            if not positions:
                continue
            naxis = len(self.coordinates[positions[0]][1])
            index = tuple(
                numpy.asarray([self.coordinates[n][1][axis] for n in positions],
                              dtype=numpy.int64)
                for axis in range(naxis)
            )
            gather.append(
                (name, numpy.asarray(positions, dtype=numpy.int64), index)
            )
        return tuple(gather)

    def amplitude_values(self, t1, t2):
        if not self._gather:
            return numpy.zeros(0)
        blocks = self._blocks(t1, t2)
        gathered = [(positions, numpy.asarray(blocks[name])[index])
                    for name, positions, index in self._gather]
        values = numpy.empty(
            len(self.coordinates),
            dtype=numpy.result_type(*[chunk for _, chunk in gathered]),
        )
        for positions, chunk in gathered:
            values[positions] = chunk
        return values

    def _scatter(self, rows, weights):
        """Row-wise sum of `weights`, which is what `numpy.add.at` would give.

        `bincount` is buffered where `add.at` is not, so it is markedly faster
        for the same repeated-index accumulation.
        """
        return numpy.bincount(rows, weights=weights,
                              minlength=self.nconstraint)

    def _amplitude_values(self, t1, t2):
        """Gather and validate amplitudes, invalidating value-dependent caches."""
        values = self.amplitude_values(t1, t2)
        if numpy.iscomplexobj(values):
            raise ValueError("complex amplitude values are not supported")
        if (self._cache_values is None
                or not numpy.array_equal(values, self._cache_values)):
            self._cache_values = values.copy()
            self._cache_residual = None
            self._cache_derivative_values = None
        return values

    def _residual(self, values):
        if self._cache_residual is None:
            residual = self.constant.copy()
            if self.linear_values.size:
                residual += self._scatter(
                    self.linear_rows,
                    self.linear_values * values[self.linear_columns],
                )
            if self.quadratic_values.size:
                residual += self._scatter(
                    self.quadratic_rows,
                    self.quadratic_values
                    * values[self.quadratic_first]
                    * values[self.quadratic_second],
                )
            self._cache_residual = residual
        return self._cache_residual

    def _derivative_values(self, values):
        """Nonzero Jacobian entries on the compiled sparse pattern."""
        if self._cache_derivative_values is None:
            derivatives = self._linear_derivative_values.copy()
            if self.quadratic_values.size:
                numpy.add.at(
                    derivatives,
                    self._quadratic_first_slots,
                    self.quadratic_values * values[self.quadratic_second],
                )
                numpy.add.at(
                    derivatives,
                    self._quadratic_second_slots,
                    self.quadratic_values * values[self.quadratic_first],
                )
            self._cache_derivative_values = derivatives
        return self._cache_derivative_values

    def constraint(self, t1, t2):
        values = self._amplitude_values(t1, t2)
        return self._residual(values)

    def jacobian(self, t1, t2):
        values = self._amplitude_values(t1, t2)
        derivatives = self._derivative_values(values)
        jacobian = numpy.zeros((self.nconstraint, len(self.coordinates)))
        jacobian[self.derivative_rows, self.derivative_columns] = derivatives
        return jacobian

    def jvp(self, t1, t2, vector):
        """Apply the constraint Jacobian to a coordinate-space vector."""
        vector = numpy.asarray(vector, dtype=float)
        if vector.shape != (len(self.coordinates),):
            raise ValueError("JVP vector has the wrong shape")
        values = self._amplitude_values(t1, t2)
        derivatives = self._derivative_values(values)
        return self._scatter(
            self.derivative_rows,
            derivatives * vector[self.derivative_columns],
        )

    def vjp(self, t1, t2, weights):
        """Apply the transpose constraint Jacobian to row weights."""
        weights = numpy.asarray(weights, dtype=float)
        if weights.shape != (self.nconstraint,):
            raise ValueError("VJP weights have the wrong shape")
        values = self._amplitude_values(t1, t2)
        derivatives = self._derivative_values(values)
        return numpy.bincount(
            self.derivative_columns,
            weights=derivatives * weights[self.derivative_rows],
            minlength=len(self.coordinates),
        )

    def jacobian_row_norms(self, t1, t2):
        """Euclidean norms of the exact Jacobian rows without densifying it."""
        values = self._amplitude_values(t1, t2)
        derivatives = self._derivative_values(values)
        squares = self._scatter(self.derivative_rows, derivatives ** 2)
        return numpy.sqrt(squares)

    def weighted_gram_diagonal(self, t1, t2, metric):
        """Diagonal of ``J diag(metric) J.T`` without forming either matrix."""
        metric = numpy.asarray(metric, dtype=float)
        if metric.shape != (len(self.coordinates),):
            raise ValueError("Gram metric has the wrong shape")
        values = self._amplitude_values(t1, t2)
        derivatives = self._derivative_values(values)
        return self._scatter(
            self.derivative_rows,
            derivatives ** 2 * metric[self.derivative_columns],
        )

    def hessian_vector(self, t1, t2, weights, vector):
        """sum_r weights[r] * sum_q (d2 C_r / dx_p dx_q) vector[q] over p.

        The constraints are quadratic in the amplitudes -- only the
        disconnected T1*T1 terms contribute beyond linear -- so the Hessian
        is amplitude independent.  The amplitudes are accepted anyway for
        interface uniformity with `constraint` and `jacobian` and ignored.
        The two scatters mirror the quadratic derivative scatters in
        `_evaluate`, accumulated over coordinates instead of constraints.
        """
        weights = numpy.asarray(weights, dtype=float)
        vector = numpy.asarray(vector, dtype=float)
        out = numpy.zeros(len(self.coordinates))
        if self.quadratic_values.size:
            weighted = weights[self.quadratic_rows] * self.quadratic_values
            out += numpy.bincount(
                self.quadratic_first,
                weights=weighted * vector[self.quadratic_second],
                minlength=len(self.coordinates),
            )
            out += numpy.bincount(
                self.quadratic_second,
                weights=weighted * vector[self.quadratic_first],
                minlength=len(self.coordinates),
            )
        return out

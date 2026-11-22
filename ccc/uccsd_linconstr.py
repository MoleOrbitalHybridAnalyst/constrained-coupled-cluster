import numpy

from pyscf.cc import uccsd
from pyscf.lib import logger

from .uccsd_constr import _checked_denominators


T1_BLOCKS = ("t1a", "t1b")
T2_BLOCKS = ("t2aa", "t2ab", "t2bb")
BLOCKS = T1_BLOCKS + T2_BLOCKS


def _amp_blocks(t1, t2):
    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    return {
        "t1a": t1a,
        "t1b": t1b,
        "t2aa": t2aa,
        "t2ab": t2ab,
        "t2bb": t2bb,
    }


def _iter_specs(spec):
    if isinstance(spec, list):
        return spec
    return (spec,)


def _pair_index(i, j):
    return i * (i - 1) // 2 + j


def _same_spin_t2_key(i, j, a, b, nocc, nvir, block):
    for value, size, label in ((i, nocc, "occupied"), (j, nocc, "occupied"),
                               (a, nvir, "virtual"), (b, nvir, "virtual")):
        if value < 0 or value >= size:
            raise IndexError("%s %s index %d out of range" % (block, label, value))
    if i == j or a == b:
        raise ValueError("%s diagonal amplitude %r is identically zero" % (block, (i, j, a, b)))

    sign = 1.0
    if i < j:
        i, j = j, i
        sign *= -1.0
    if a < b:
        a, b = b, a
        sign *= -1.0

    return (_pair_index(i, j), _pair_index(a, b), sign)


class LinConstrUCCSD(uccsd.UCCSD):
    """UCCSD with user-specified linear constraints on unrestricted amplitudes.

    ``constraints`` is a list of dictionaries.  Each dictionary has optional
    ``label`` and ``rhs`` fields and a required ``terms`` field.  Each term is
    ``(block, index, coefficient)`` with ``block`` in ``t1a``, ``t1b``,
    ``t2aa``, ``t2ab``, or ``t2bb``.  The constraint is

        sum_mu coefficient_mu * t_mu = rhs

    ``initial_amplitudes`` is a dictionary mapping a block name to
    ``(index, value)`` entries.  These values are inserted into the initial
    guess before the denominator-weighted constraint correction, but they are
    not fixed unless the constraints require them.

    The correction solves the constrained residual equations
    ``R = C.T @ multipliers`` rather than Euclidean-projecting the amplitudes.
    """

    def __init__(self, mf, constraints=None, initial_amplitudes=None,
                 frozen=None, mo_coeff=None, mo_occ=None):
        uccsd.UCCSD.__init__(self, mf, frozen=frozen, mo_coeff=mo_coeff, mo_occ=mo_occ)
        keys = set((
            "constraints",
            "initial_amplitudes",
            "constraint_labels",
            "constraint_rhs",
            "constraint_rank",
            "max_constraint_residual",
            "constraint_multipliers",
        ))
        self._keys = set(self.__dict__.keys()).union(keys)
        self.constraints = list(constraints or ())
        self.initial_amplitudes = dict(initial_amplitudes or {})
        self.constraint_labels = []
        self.constraint_rhs = None
        self.constraint_rank = None
        self.max_constraint_residual = None
        self.constraint_multipliers = None
        self._constraint_shapes = None
        self._constraint_rows = None
        self._constraint_gram = None
        self._constraint_size = None

    def dump_flags(self, verbose=None):
        uccsd.UCCSD.dump_flags(self, verbose=verbose)
        log = logger.new_logger(self, verbose)
        log.info("Linear amplitude constraints: %d", len(self.constraints))
        if self.initial_amplitudes:
            log.info("Initial constrained-amplitude blocks: %s", sorted(self.initial_amplitudes))
        else:
            log.info("Initial constrained-amplitude blocks: none")
        return self

    def _vector_size_and_offsets(self, t1, t2):
        nocca, nvira = t1[0].shape
        noccb, nvirb = t1[1].shape
        sizea = nocca * nvira + nocca * (nocca - 1) // 2 * nvira * (nvira - 1) // 2
        sizeb = noccb * nvirb + noccb * (noccb - 1) // 2 * nvirb * (nvirb - 1) // 2
        sizeab = nocca * noccb * nvira * nvirb
        return sizea + sizeb + sizeab, sizea, sizeb

    def _term_flat_index(self, block, idx, t1, t2):
        idx = tuple(idx)
        nocca, nvira = t1[0].shape
        noccb, nvirb = t1[1].shape
        _, sizea, sizeb = self._vector_size_and_offsets(t1, t2)

        if block == "t1a":
            if len(idx) != 2:
                raise ValueError("t1a index must have length 2")
            i, a = idx
            return numpy.ravel_multi_index((i, a), (nocca, nvira)), 1.0
        if block == "t1b":
            if len(idx) != 2:
                raise ValueError("t1b index must have length 2")
            i, a = idx
            return sizea + numpy.ravel_multi_index((i, a), (noccb, nvirb)), 1.0
        if block == "t2aa":
            if len(idx) != 4:
                raise ValueError("t2aa index must have length 4")
            i, j, a, b = idx
            p, q, sign = _same_spin_t2_key(i, j, a, b, nocca, nvira, block)
            nvir_pairs = nvira * (nvira - 1) // 2
            return nocca * nvira + p * nvir_pairs + q, sign
        if block == "t2bb":
            if len(idx) != 4:
                raise ValueError("t2bb index must have length 4")
            i, j, a, b = idx
            p, q, sign = _same_spin_t2_key(i, j, a, b, noccb, nvirb, block)
            nvir_pairs = nvirb * (nvirb - 1) // 2
            return sizea + noccb * nvirb + p * nvir_pairs + q, sign
        if block == "t2ab":
            if len(idx) != 4:
                raise ValueError("t2ab index must have length 4")
            i, j, a, b = idx
            return sizea + sizeb + numpy.ravel_multi_index(
                (i, j, a, b), (nocca, noccb, nvira, nvirb)
            ), 1.0
        raise ValueError("Unknown amplitude block %r" % (block,))

    def _ensure_constraints(self, t1, t2):
        shapes = tuple(x.shape for x in t1 + t2)
        if self._constraint_shapes == shapes:
            return

        unknown_initial = sorted(set(self.initial_amplitudes) - set(BLOCKS))
        if unknown_initial:
            raise ValueError("Unknown initial amplitude block(s): %s" % unknown_initial)

        size, _, _ = self._vector_size_and_offsets(t1, t2)
        rows = []
        rhs = []
        labels = []
        for n, constraint in enumerate(self.constraints):
            if not isinstance(constraint, dict):
                raise ValueError("Constraint %d must be a dictionary" % n)
            terms = constraint.get("terms")
            if terms is None:
                raise ValueError("Constraint %d is missing terms" % n)

            row = {}
            for term in terms:
                try:
                    block, idx, coeff = term
                except Exception as err:
                    raise ValueError("Invalid term in constraint %d: %r" % (n, term)) from err
                flat, sign = self._term_flat_index(block, idx, t1, t2)
                row[flat] = row.get(flat, 0.0) + float(coeff) * sign

            row = {k: v for k, v in row.items() if abs(v) > 0.0}
            if not row:
                raise ValueError("Constraint %d has no nonzero terms" % n)
            rows.append(row)
            rhs.append(float(constraint.get("rhs", 0.0)))
            labels.append(str(constraint.get("label", "constraint_%d" % n)))

        self._constraint_shapes = shapes
        self._constraint_rows = rows
        self.constraint_rhs = numpy.asarray(rhs)
        self.constraint_labels = labels
        self._constraint_gram = None
        self._constraint_size = size
        self.constraint_rank = None

    def _apply_initial_amplitudes(self, t1, t2):
        if not self.initial_amplitudes:
            return t1, t2

        blocks = _amp_blocks(t1, t2)
        out = {name: numpy.asarray(block).copy() for name, block in blocks.items()}
        unknown = sorted(set(self.initial_amplitudes) - set(BLOCKS))
        if unknown:
            raise ValueError("Unknown initial amplitude block(s): %s" % unknown)

        for name, specs in self.initial_amplitudes.items():
            for spec in _iter_specs(specs):
                try:
                    idx, value = spec
                    out[name][idx] = value
                except Exception as err:
                    raise ValueError(
                        "Invalid initial amplitude specification for %s: %r" % (name, spec)
                    ) from err

        return (out["t1a"], out["t1b"]), (out["t2aa"], out["t2ab"], out["t2bb"])

    def _constraint_residual_vector(self, vector):
        if not self._constraint_rows:
            return numpy.zeros(0)
        out = numpy.zeros(len(self._constraint_rows))
        for n, row in enumerate(self._constraint_rows):
            out[n] = sum(coeff * vector[flat] for flat, coeff in row.items())
        return out - self.constraint_rhs

    def constraint_residuals(self, t1=None, t2=None):
        if t1 is None:
            t1 = self.t1
        if t2 is None:
            t2 = self.t2
        self._ensure_constraints(t1, t2)
        vec = self.amplitudes_to_vector(t1, t2)
        return self._constraint_residual_vector(vec)

    def _correct_amplitudes(self, t1, t2, eris):
        self._ensure_constraints(t1, t2)
        if not self._constraint_rows:
            self.max_constraint_residual = 0.0
            self.constraint_rank = 0
            self.constraint_multipliers = numpy.zeros(0)
            return t1, t2

        vec = self.amplitudes_to_vector(t1, t2).copy()
        residual = self._constraint_residual_vector(vec)
        self.max_constraint_residual = float(numpy.max(numpy.abs(residual)))
        if self.max_constraint_residual == 0.0:
            self.constraint_multipliers = numpy.zeros(len(self._constraint_rows))
            return t1, t2

        denominators = _checked_denominators(self, t1, t2, eris)
        responses = [
            {flat: -coefficient / denominators[flat] for flat, coefficient in row.items()}
            for row in self._constraint_rows
        ]
        nconstraint = len(self._constraint_rows)
        gram = numpy.zeros((nconstraint, nconstraint))
        for i, row in enumerate(self._constraint_rows):
            for j, response in enumerate(responses):
                gram[i, j] = sum(coefficient * response.get(flat, 0.0)
                                 for flat, coefficient in row.items())

        rank = int(numpy.linalg.matrix_rank(gram))
        self._constraint_gram = gram
        self.constraint_rank = rank
        if rank != nconstraint:
            raise RuntimeError(
                "Linear constraint multiplier matrix is singular (rank %d of %d)"
                % (rank, nconstraint)
            )
        multipliers = numpy.linalg.solve(gram, -residual)
        self.constraint_multipliers = multipliers
        for multiplier, response in zip(multipliers, responses):
            if multiplier == 0.0:
                continue
            for flat, value in response.items():
                vec[flat] += multiplier * value

        t1out, t2out = self.vector_to_amplitudes(vec)
        residual = self._constraint_residual_vector(vec)
        self.max_constraint_residual = float(numpy.max(numpy.abs(residual)))
        return t1out, t2out

    def get_init_guess(self, eris=None):
        if eris is None:
            eris = self.ao2mo(self.mo_coeff)
        t1, t2 = uccsd.UCCSD.get_init_guess(self, eris=eris)
        t1, t2 = self._apply_initial_amplitudes(t1, t2)
        return self._correct_amplitudes(t1, t2, eris)

    def ccsd(self, t1=None, t2=None, eris=None, mbpt2=False):
        if t1 is not None or t2 is not None:
            if eris is None:
                eris = self.ao2mo(self.mo_coeff)
            if t1 is None or t2 is None:
                t1_guess, t2_guess = uccsd.UCCSD.get_init_guess(self, eris=eris)
                if t1 is None:
                    t1 = t1_guess
                if t2 is None:
                    t2 = t2_guess
            t1, t2 = self._correct_amplitudes(t1, t2, eris)
        return uccsd.UCCSD.ccsd(self, t1=t1, t2=t2, eris=eris, mbpt2=mbpt2)

    def update_amps(self, t1, t2, eris):
        t1new, t2new = uccsd.update_amps(self, t1, t2, eris)
        return self._correct_amplitudes(t1new, t2new, eris)

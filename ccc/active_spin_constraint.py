import itertools

import numpy
from scipy import sparse
from pyscf.fci import addons, cistring


BLOCKS = (
    "t1a", "t1b", "t2aa", "t2ab", "t2bb",
    "t3aaa", "t3aab", "t3abb", "t3bbb",
)


def _address(occupied, norb, nelec):
    string = sum(1 << int(i) for i in occupied)
    return cistring.str2addr(norb, nelec, string)


def _reference_permutation(occupied, norb):
    occupied = tuple(int(i) for i in occupied)
    occupied_set = set(occupied)
    return numpy.asarray(
        occupied + tuple(i for i in range(norb) if i not in occupied_set),
        dtype=int,
    )


def occupations_from_string(occupation):
    alpha = tuple(i for i, char in enumerate(occupation) if char in ("a", "2"))
    beta = tuple(i for i, char in enumerate(occupation) if char in ("b", "2"))
    return alpha, beta


class DominantSectorSpinConstraint:
    """Exact highest-weight constraint in one spatial-occupation sector.

    Alpha and beta active orbitals may have different orderings.  Their
    permutations map reordered orbital positions back to a common original
    spatial-orbital label.  ``max_cluster_rank=3`` adds occupied-first
    ``t3aaa``, ``t3aab``, ``t3abb``, and ``t3bbb`` amplitudes to the exact
    finite cluster exponential.
    """

    def __init__(self, norb, nelec, occupation, ncore=0,
                 alpha_permutation=None, beta_permutation=None,
                 max_cluster_rank=2):
        self.norb = int(norb)
        self.nelec = tuple(int(x) for x in nelec)
        self.ncore = int(ncore)
        self.max_cluster_rank = int(max_cluster_rank)
        if self.max_cluster_rank not in (2, 3):
            raise ValueError("max_cluster_rank must be 2 or 3")
        self.occupation = str(occupation)
        if len(self.occupation) != self.norb:
            raise ValueError("occupation length does not match norb")

        alpha, beta = occupations_from_string(self.occupation)
        if (len(alpha), len(beta)) != self.nelec:
            raise ValueError("occupation electron count does not match nelec")
        if alpha_permutation is None:
            alpha_permutation = _reference_permutation(alpha, self.norb)
        if beta_permutation is None:
            beta_permutation = _reference_permutation(beta, self.norb)
        self.alpha_permutation = numpy.asarray(alpha_permutation, dtype=int)
        self.beta_permutation = numpy.asarray(beta_permutation, dtype=int)
        expected = numpy.arange(self.norb)
        if (not numpy.array_equal(numpy.sort(self.alpha_permutation), expected)
                or not numpy.array_equal(numpy.sort(self.beta_permutation), expected)):
            raise ValueError("orbital permutations are invalid")
        self.alpha_inverse = numpy.argsort(self.alpha_permutation)
        self.beta_inverse = numpy.argsort(self.beta_permutation)

        self.paired = tuple(i for i, char in enumerate(self.occupation) if char == "2")
        self.empty = tuple(i for i, char in enumerate(self.occupation) if char == "0")
        self.singly = tuple(
            i for i, char in enumerate(self.occupation) if char in ("a", "b")
        )
        self.nbeta_singly = self.nelec[1] - len(self.paired)
        if self.nbeta_singly < 0:
            raise ValueError("invalid paired occupation")

        self.coordinates, self._operators = self._make_cluster_operators()
        self._input_sector = self._make_sector(self.nbeta_singly, self.nelec)
        if self.nbeta_singly:
            output_nelec = (self.nelec[0] + 1, self.nelec[1] - 1)
            self._output_sector = self._make_sector(
                self.nbeta_singly - 1, output_nelec
            )
        else:
            self._output_sector = ()
        self.nconstraint = len(self._output_sector)
        self._maximum_excitation = (
            min(self.nelec[0], self.norb - self.nelec[0])
            + min(self.nelec[1], self.norb - self.nelec[1])
        )
        self._cache_values = None
        self._cache_wavefunction = None
        self._cache_residual = None
        self._cache_jacobian = None

        # Determinant-space maps, built on first use.  Every matrix element of
        # a cluster operator is a signed lookup, <D| tau |Psi> = sign * Psi[D'],
        # because tau^dagger sends a determinant to a single determinant.  The
        # maps are pure combinatorics and therefore amplitude independent.
        self._shape = None
        self._cluster = None
        self._cluster_column = None
        self._cluster_sign = None
        self._sector_flat = None
        self._source = None
        self._source_sign = None
        self._left = None

    def _make_cluster_operators(self):
        na, nb = self.nelec
        nva, nvb = self.norb - na, self.norb - nb
        coordinates = []
        operators = []
        for i in range(na):
            for a in range(nva):
                coordinates.append(("t1a", (self.ncore + i, a)))
                operators.append(("a", i, na + a))
        for i in range(nb):
            for a in range(nvb):
                coordinates.append(("t1b", (self.ncore + i, a)))
                operators.append(("b", i, nb + a))
        for i in range(na):
            for j in range(i):
                for a in range(nva):
                    for b in range(a):
                        coordinates.append((
                            "t2aa", (self.ncore + i, self.ncore + j, a, b)
                        ))
                        operators.append(("aa", i, j, na + a, na + b))
        for i in range(na):
            for j in range(nb):
                for a in range(nva):
                    for b in range(nvb):
                        coordinates.append((
                            "t2ab", (self.ncore + i, self.ncore + j, a, b)
                        ))
                        operators.append(("ab", i, j, na + a, nb + b))
        for i in range(nb):
            for j in range(i):
                for a in range(nvb):
                    for b in range(a):
                        coordinates.append((
                            "t2bb", (self.ncore + i, self.ncore + j, a, b)
                        ))
                        operators.append(("bb", i, j, nb + a, nb + b))
        if self.max_cluster_rank == 3:
            for i in range(na):
                for j in range(i):
                    for k in range(j):
                        for a in range(nva):
                            for b in range(a):
                                for c in range(b):
                                    coordinates.append((
                                        "t3aaa",
                                        (self.ncore + i, self.ncore + j,
                                         self.ncore + k, a, b, c),
                                    ))
                                    operators.append((
                                        "aaa", i, j, k,
                                        na + a, na + b, na + c,
                                    ))
            for i in range(na):
                for j in range(i):
                    for k in range(nb):
                        for a in range(nva):
                            for b in range(a):
                                for c in range(nvb):
                                    coordinates.append((
                                        "t3aab",
                                        (self.ncore + i, self.ncore + j,
                                         self.ncore + k, a, b, c),
                                    ))
                                    operators.append((
                                        "aab", i, j, k,
                                        na + a, na + b, nb + c,
                                    ))
            for i in range(na):
                for j in range(nb):
                    for k in range(j):
                        for a in range(nva):
                            for b in range(nvb):
                                for c in range(b):
                                    coordinates.append((
                                        "t3abb",
                                        (self.ncore + i, self.ncore + j,
                                         self.ncore + k, a, b, c),
                                    ))
                                    operators.append((
                                        "abb", i, j, k,
                                        na + a, nb + b, nb + c,
                                    ))
            for i in range(nb):
                for j in range(i):
                    for k in range(j):
                        for a in range(nvb):
                            for b in range(a):
                                for c in range(b):
                                    coordinates.append((
                                        "t3bbb",
                                        (self.ncore + i, self.ncore + j,
                                         self.ncore + k, a, b, c),
                                    ))
                                    operators.append((
                                        "bbb", i, j, k,
                                        nb + a, nb + b, nb + c,
                                    ))
        return tuple(coordinates), tuple(operators)

    def _make_sector(self, nbeta_singly, nelec):
        out = []
        paired = set(self.paired)
        singly = set(self.singly)
        for beta_singly in itertools.combinations(self.singly, nbeta_singly):
            beta_original = paired | set(beta_singly)
            alpha_original = paired | (singly - set(beta_singly))
            alpha_local = self.alpha_inverse[list(alpha_original)]
            beta_local = self.beta_inverse[list(beta_original)]
            out.append((
                _address(alpha_local, self.norb, nelec[0]),
                _address(beta_local, self.norb, nelec[1]),
            ))
        return tuple(out)

    @staticmethod
    def _blocks(t1, t2, t3=None):
        amplitudes = tuple(t1) + tuple(t2)
        if t3 is not None:
            amplitudes += tuple(t3)
        return dict(zip(BLOCKS, amplitudes))

    def amplitude_values(self, t1, t2, t3=None):
        if self.max_cluster_rank == 3 and t3 is None:
            raise ValueError("T3 amplitudes are required for max_cluster_rank=3")
        blocks = self._blocks(t1, t2, t3)
        return numpy.asarray([
            blocks[block][index] for block, index in self.coordinates
        ])

    def _apply_operator(self, operator, ci):
        kind = operator[0]
        na, nb = self.nelec
        if kind == "a":
            _, i, a = operator
            tmp = addons.des_a(ci, self.norb, (na, nb), i)
            return addons.cre_a(tmp, self.norb, (na - 1, nb), a)
        if kind == "b":
            _, i, a = operator
            tmp = addons.des_b(ci, self.norb, (na, nb), i)
            return addons.cre_b(tmp, self.norb, (na, nb - 1), a)
        if kind == "aa":
            _, i, j, a, b = operator
            tmp = addons.des_a(ci, self.norb, (na, nb), i)
            tmp = addons.des_a(tmp, self.norb, (na - 1, nb), j)
            tmp = addons.cre_a(tmp, self.norb, (na - 2, nb), b)
            return addons.cre_a(tmp, self.norb, (na - 1, nb), a)
        if kind == "bb":
            _, i, j, a, b = operator
            tmp = addons.des_b(ci, self.norb, (na, nb), i)
            tmp = addons.des_b(tmp, self.norb, (na, nb - 1), j)
            tmp = addons.cre_b(tmp, self.norb, (na, nb - 2), b)
            return addons.cre_b(tmp, self.norb, (na, nb - 1), a)
        if kind == "ab":
            _, i, j, a, b = operator
            tmp = addons.des_a(ci, self.norb, (na, nb), i)
            tmp = addons.des_b(tmp, self.norb, (na - 1, nb), j)
            tmp = addons.cre_b(tmp, self.norb, (na - 1, nb - 1), b)
            return addons.cre_a(tmp, self.norb, (na - 1, nb), a)
        if kind == "aaa":
            _, i, j, k, a, b, c = operator
            tmp = addons.des_a(ci, self.norb, (na, nb), i)
            tmp = addons.des_a(tmp, self.norb, (na - 1, nb), j)
            tmp = addons.des_a(tmp, self.norb, (na - 2, nb), k)
            tmp = addons.cre_a(tmp, self.norb, (na - 3, nb), c)
            tmp = addons.cre_a(tmp, self.norb, (na - 2, nb), b)
            return addons.cre_a(tmp, self.norb, (na - 1, nb), a)
        if kind == "aab":
            _, i, j, k, a, b, c = operator
            tmp = addons.des_a(ci, self.norb, (na, nb), i)
            tmp = addons.des_a(tmp, self.norb, (na - 1, nb), j)
            tmp = addons.des_b(tmp, self.norb, (na - 2, nb), k)
            tmp = addons.cre_b(tmp, self.norb, (na - 2, nb - 1), c)
            tmp = addons.cre_a(tmp, self.norb, (na - 2, nb), b)
            return addons.cre_a(tmp, self.norb, (na - 1, nb), a)
        if kind == "abb":
            _, i, j, k, a, b, c = operator
            tmp = addons.des_a(ci, self.norb, (na, nb), i)
            tmp = addons.des_b(tmp, self.norb, (na - 1, nb), j)
            tmp = addons.des_b(tmp, self.norb, (na - 1, nb - 1), k)
            tmp = addons.cre_b(tmp, self.norb, (na - 1, nb - 2), c)
            tmp = addons.cre_b(tmp, self.norb, (na - 1, nb - 1), b)
            return addons.cre_a(tmp, self.norb, (na - 1, nb), a)
        if kind == "bbb":
            _, i, j, k, a, b, c = operator
            tmp = addons.des_b(ci, self.norb, (na, nb), i)
            tmp = addons.des_b(tmp, self.norb, (na, nb - 1), j)
            tmp = addons.des_b(tmp, self.norb, (na, nb - 2), k)
            tmp = addons.cre_b(tmp, self.norb, (na, nb - 3), c)
            tmp = addons.cre_b(tmp, self.norb, (na, nb - 2), b)
            return addons.cre_b(tmp, self.norb, (na, nb - 1), a)
        raise RuntimeError("unknown excitation operator")

    def _ensure_cluster_matrix(self):
        """Compile sum_mu t_mu tau_mu into a sparse matrix of fixed pattern.

        Each ``tau_mu`` sends a determinant to a single signed determinant, so
        the sum is a sparse matrix whose sparsity is amplitude independent; only
        its data changes between evaluations.  The (destination, source) pair
        determines ``mu`` uniquely, so there are no duplicate entries.

        The map is read off ``_apply_operator`` itself by feeding in a vector
        whose entry is its own flat address plus one: the image of ``tau_mu`` is
        then +-(source address + 1).  Deriving it this way inherits the phase
        convention of the operator applications instead of restating it.
        """
        if self._cluster is not None:
            return
        na, nb = self.nelec
        self._shape = (
            cistring.num_strings(self.norb, na),
            cistring.num_strings(self.norb, nb),
        )
        ndet = self._shape[0] * self._shape[1]
        marker = (numpy.arange(ndet, dtype=float) + 1.0).reshape(self._shape)

        # Seeded with typed empties so an active space without virtual orbitals,
        # which has no excitation operators at all, still compiles.
        rows = [numpy.zeros(0, dtype=numpy.int32)]
        columns = [numpy.zeros(0, dtype=numpy.int32)]
        orbitals = [numpy.zeros(0, dtype=numpy.int32)]
        signs = [numpy.zeros(0)]
        for mu, operator in enumerate(self._operators):
            image = self._apply_operator(operator, marker).ravel()
            nonzero = numpy.flatnonzero(image)
            value = image[nonzero]
            source = numpy.abs(value)
            if not numpy.array_equal(source, numpy.round(source)):
                raise RuntimeError(
                    "excitation operator %r is not a determinant map" % (operator,)
                )
            rows.append(nonzero.astype(numpy.int32))
            columns.append((source - 1.0).astype(numpy.int32))
            orbitals.append(numpy.full(nonzero.size, mu, dtype=numpy.int32))
            signs.append(numpy.sign(value))

        rows = numpy.concatenate(rows)
        columns = numpy.concatenate(columns)
        order = numpy.argsort(rows, kind="stable")
        rows = rows[order]
        self._cluster_column = numpy.concatenate(orbitals)[order]
        self._cluster_sign = numpy.concatenate(signs)[order]
        indptr = numpy.zeros(ndet + 1, dtype=numpy.int64)
        numpy.add.at(indptr, rows.astype(numpy.int64) + 1, 1)
        self._cluster = sparse.csr_matrix(
            (numpy.zeros(rows.size), columns[order], numpy.cumsum(indptr)),
            shape=(ndet, ndet),
        )

    def _ensure_residual_maps(self):
        """Compile the sector residual and its amplitude derivatives.

        ``_left`` is the matrix of ``_sector_residual`` restricted to the input
        sector, and ``(_source, _source_sign)`` give the signed lookup for
        ``<D| tau_mu |Psi>`` on sector determinants ``D``.  Both are read out of
        the cluster matrix, which is indexed by destination determinant, so no
        second scan over the operators is needed.
        """
        if self._left is not None or not self.nconstraint:
            return
        self._ensure_cluster_matrix()
        nsb = self._shape[1]
        self._sector_flat = numpy.asarray(
            [ia * nsb + ib for ia, ib in self._input_sector], dtype=numpy.int64
        )

        ncoordinate = len(self._operators)
        nsector = len(self._sector_flat)
        self._source = numpy.zeros((nsector, ncoordinate), dtype=numpy.int64)
        self._source_sign = numpy.zeros((nsector, ncoordinate))
        indptr = self._cluster.indptr
        for row, flat in enumerate(self._sector_flat):
            entries = slice(indptr[flat], indptr[flat + 1])
            columns = self._cluster_column[entries]
            self._source[row, columns] = self._cluster.indices[entries]
            self._source_sign[row, columns] = self._cluster_sign[entries]

        self._left = numpy.zeros((self.nconstraint, nsector))
        for column, (ia, ib) in enumerate(self._input_sector):
            unit = numpy.zeros(self._shape)
            unit[ia, ib] = 1.0
            self._left[:, column] = self._sector_residual(unit)

    def _build_wavefunction(self, values):
        self._ensure_cluster_matrix()
        values = numpy.asarray(values)
        self._cluster.data[:] = self._cluster_sign * values[self._cluster_column]
        wavefunction = numpy.zeros(self._shape[0] * self._shape[1])
        wavefunction[0] = 1.0
        term = wavefunction.copy()
        for order in range(1, self._maximum_excitation + 1):
            term = self._cluster.dot(term) / order
            wavefunction += term
            if not numpy.any(term):
                break
        return wavefunction.reshape(self._shape)

    def _project_sector(self, ci):
        out = numpy.zeros_like(ci)
        for ia, ib in self._input_sector:
            out[ia, ib] = ci[ia, ib]
        return out

    def _apply_spin_raise(self, ci):
        na, nb = self.nelec
        shape = (
            cistring.num_strings(self.norb, na + 1),
            cistring.num_strings(self.norb, nb - 1),
        )
        out = numpy.zeros(shape)
        for original in self.singly:
            beta = int(self.beta_inverse[original])
            alpha = int(self.alpha_inverse[original])
            tmp = addons.des_b(ci, self.norb, (na, nb), beta)
            out += addons.cre_a(tmp, self.norb, (na, nb - 1), alpha)
        return out

    def _sector_residual(self, ci):
        if not self.nconstraint:
            return numpy.zeros(0)
        raised = self._apply_spin_raise(self._project_sector(ci))
        return numpy.asarray([raised[index] for index in self._output_sector])

    def _evaluate_values(self, values):
        values = numpy.asarray(values)
        expected = (len(self.coordinates),)
        if values.shape != expected:
            raise ValueError(
                "amplitude values have shape %s; expected %s"
                % (values.shape, expected)
            )
        if numpy.iscomplexobj(values) and numpy.any(
                numpy.abs(values.imag) > 1e-12):
            raise ValueError("complex amplitude values are not supported")
        values = numpy.asarray(values.real, dtype=float)
        if not numpy.all(numpy.isfinite(values)):
            raise ValueError("amplitude values contain non-finite entries")
        if (self._cache_values is not None
                and numpy.array_equal(values, self._cache_values)):
            return
        wavefunction = self._build_wavefunction(values)
        if self.nconstraint:
            self._ensure_residual_maps()
            flat = wavefunction.ravel()
            residual = self._left @ flat[self._sector_flat]
        else:
            residual = numpy.zeros(0)
        self._cache_values = values.copy()
        self._cache_wavefunction = wavefunction
        self._cache_residual = residual
        self._cache_jacobian = None

    def _coordinate_vector(self, values, name):
        values = numpy.asarray(values)
        expected = (len(self.coordinates),)
        if values.shape != expected:
            raise ValueError("%s has shape %s; expected %s"
                             % (name, values.shape, expected))
        if numpy.iscomplexobj(values) and numpy.any(
                numpy.abs(values.imag) > 1e-12):
            raise ValueError("%s must be real" % name)
        values = numpy.asarray(values.real, dtype=float)
        if not numpy.all(numpy.isfinite(values)):
            raise ValueError("%s contains non-finite entries" % name)
        return values

    def _constraint_weights(self, weights):
        weights = numpy.asarray(weights)
        expected = (self.nconstraint,)
        if weights.shape != expected:
            raise ValueError("constraint weights have shape %s; expected %s"
                             % (weights.shape, expected))
        if numpy.iscomplexobj(weights) and numpy.any(
                numpy.abs(weights.imag) > 1e-12):
            raise ValueError("constraint weights must be real")
        weights = numpy.asarray(weights.real, dtype=float)
        if not numpy.all(numpy.isfinite(weights)):
            raise ValueError("constraint weights contain non-finite entries")
        return weights

    def _sector_derivatives(self, wavefunction=None):
        """Return D[s,mu] = <s|tau_mu exp(T)|0> in the input sector."""
        if not self.nconstraint:
            return numpy.empty((0, len(self.coordinates)))
        if wavefunction is None:
            wavefunction = self._cache_wavefunction.ravel()
        return self._source_sign * wavefunction[self._source]

    def _evaluate(self, t1, t2, t3=None):
        self._evaluate_values(self.amplitude_values(t1, t2, t3))

    def constraint_from_values(self, values):
        """Evaluate the spin constraints from ``coordinates``-ordered values."""
        self._evaluate_values(values)
        return self._cache_residual

    def jacobian_from_values(self, values):
        """Evaluate the coordinate Jacobian without full amplitude tensors."""
        self._evaluate_values(values)
        if self._cache_jacobian is None:
            if self.nconstraint:
                self._cache_jacobian = (
                    self._left @ self._sector_derivatives()
                )
            else:
                self._cache_jacobian = numpy.empty(
                    (0, len(self.coordinates))
                )
        return self._cache_jacobian

    def wavefunction_from_values(self, values):
        """Build the active-sector wavefunction from selected amplitudes."""
        self._evaluate_values(values)
        return self._cache_wavefunction

    def sector_spin_square_from_values(self, values):
        """Return the dominant-sector spin square from selected amplitudes."""
        self._evaluate_values(values)
        sector = self._project_sector(self._cache_wavefunction)
        norm = float(numpy.vdot(sector, sector).real)
        if norm == 0:
            return numpy.nan
        ms = 0.5 * (self.nelec[0] - self.nelec[1])
        contamination = float(numpy.vdot(
            self._cache_residual, self._cache_residual
        ).real / norm)
        return ms * (ms + 1) + contamination

    def constraint(self, t1, t2, t3=None):
        self._evaluate(t1, t2, t3)
        return self._cache_residual

    def jacobian(self, t1, t2, t3=None):
        return self.jacobian_from_values(
            self.amplitude_values(t1, t2, t3)
        )

    def jvp(self, t1, t2, vector):
        """Apply the active-sector constraint Jacobian without forming it."""
        self._evaluate(t1, t2)
        vector = self._coordinate_vector(vector, "JVP vector")
        if not self.nconstraint:
            return numpy.zeros(0)
        return self._left @ (self._sector_derivatives() @ vector)

    def vjp(self, t1, t2, weights):
        """Apply the transpose active-sector Jacobian without forming it."""
        self._evaluate(t1, t2)
        weights = self._constraint_weights(weights)
        if not self.nconstraint:
            return numpy.zeros(len(self.coordinates))
        return self._sector_derivatives().T @ (self._left.T @ weights)

    def jacobian_row_norms(self, t1, t2):
        """Return exact Jacobian row norms without retaining a dense Jacobian."""
        self._evaluate(t1, t2)
        if not self.nconstraint:
            return numpy.zeros(0)
        derivatives = self._sector_derivatives()
        return numpy.asarray([
            numpy.linalg.norm(derivatives.T @ row) for row in self._left
        ])

    def weighted_gram_diagonal(self, t1, t2, metric):
        """Return diag(J diag(metric) J.T) without retaining J."""
        self._evaluate(t1, t2)
        metric = self._coordinate_vector(metric, "Gram metric")
        if not self.nconstraint:
            return numpy.zeros(0)
        derivatives = self._sector_derivatives()
        diagonal = numpy.empty(self.nconstraint)
        for i, row in enumerate(self._left):
            gradient = derivatives.T @ row
            diagonal[i] = numpy.dot(metric, gradient ** 2)
        return diagonal

    def hessian_vector(self, t1, t2, weights, vector):
        """Apply the weighted constraint Hessian to a coordinate vector."""
        self._evaluate(t1, t2)
        weights = self._constraint_weights(weights)
        vector = self._coordinate_vector(vector, "Hessian vector")
        if not self.nconstraint:
            return numpy.zeros(len(self.coordinates))

        # Occupied-to-virtual excitation operators commute, hence the
        # directional derivative of exp(T)|0> is V exp(T)|0>, where
        # V = sum_mu vector[mu] tau_mu.  Use a separate CSR data array so the
        # cluster matrix cached at the current amplitudes remains untouched.
        data = self._cluster_sign * vector[self._cluster_column]
        directional_cluster = sparse.csr_matrix(
            (data, self._cluster.indices, self._cluster.indptr),
            shape=self._cluster.shape,
        )
        directional_wavefunction = directional_cluster.dot(
            self._cache_wavefunction.ravel()
        )
        derivatives = self._sector_derivatives(directional_wavefunction)
        return derivatives.T @ (self._left.T @ weights)

    def wavefunction(self, t1, t2, t3=None):
        self._evaluate(t1, t2, t3)
        return self._cache_wavefunction

    def sector_spin_square(self, t1, t2, t3=None):
        return self.sector_spin_square_from_values(
            self.amplitude_values(t1, t2, t3)
        )

    def check_jacobian(self, t1, t2, t3=None, step=1e-6, ndir=3,
                       seed=1, tolerance=1e-7):
        values = self.amplitude_values(t1, t2, t3)
        analytic = self.jacobian(t1, t2, t3).copy()
        blocks = self._blocks(t1, t2, t3)
        rng = numpy.random.default_rng(seed)
        maximum = 0.0
        for _ in range(ndir):
            direction = rng.standard_normal(values.size)
            direction /= numpy.linalg.norm(direction)
            for sign, target in ((1.0, "plus"), (-1.0, "minus")):
                for value, delta, (block, index) in zip(
                        values, direction, self.coordinates):
                    blocks[block][index] = value + sign * step * delta
                result = self.constraint(t1, t2, t3).copy()
                if target == "plus":
                    plus = result
                else:
                    minus = result
            for value, (block, index) in zip(values, self.coordinates):
                blocks[block][index] = value
            finite_difference = (plus - minus) / (2 * step)
            expected = analytic @ direction
            scale = numpy.maximum(1.0, numpy.maximum(
                numpy.abs(finite_difference), numpy.abs(expected)
            ))
            maximum = max(maximum, float(numpy.max(
                numpy.abs(finite_difference - expected) / scale
            )))
        self._cache_values = None
        if maximum > tolerance:
            raise ValueError(
                "Dominant-sector Jacobian error %.6e exceeds %.6e"
                % (maximum, tolerance)
            )
        return maximum

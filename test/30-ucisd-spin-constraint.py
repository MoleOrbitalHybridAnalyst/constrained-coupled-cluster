"""UCISDSpinConstraint: S+ equations for unmatched alpha/beta orbitals.

The matched engine needs the two spin orbital sets to agree up to a signed
permutation.  This one takes the dense alpha-beta MO overlap instead, and
decides retention without ever looking at the size of an overlap element, so the
equation count depends only on (nmo, nalpha, nbeta).

What is checked here, in order: the count is coefficient agnostic and matches
the closed form; a reference whose alpha and beta occupied subspaces coincide is
annihilated by S+ even when the overlap is dense; the compiled equations
reproduce S+ applied forward over the whole SD space; the omitted equations
really are the ones that reach outside that space; the retained set is a subset
of the matched engine's with identical coefficients; the analytic Jacobian and
Hessian-vector product match finite differences; and a constrained UCCSD run on
genuine UHF orbitals reaches the FCI singlet.
"""
import numpy
from pyscf import fci, gto, scf
from pyscf.cc import uccsd

from ccc import MultiConstrUCCSD
from ccc.cisd_spin_constraint import (ClosedCISDSpinConstraint,
                                        _apply_sequence)
from ccc.ucisd_spin_constraint import UCISDSpinConstraint

rng = numpy.random.default_rng(30)


def configure(cc):
    # DIIS is left at its defaults on purpose.  The H2 case below carries three
    # amplitudes in total, and neither a subspace wider than the problem nor a
    # delayed DIIS start converges the constrained step there, although both
    # reach the same energy to 1e-11.
    cc.verbose = 0
    cc.max_cycle = 200
    cc.conv_tol = 1e-11
    cc.conv_tol_normt = 1e-8
    return cc


def orthogonal(size):
    matrix, _ = numpy.linalg.qr(rng.standard_normal((size, size)))
    return matrix


def random_amplitudes(nmo, nocc, scale=0.05):
    nocca, noccb = nocc
    nvira, nvirb = nmo - nocca, nmo - noccb
    t1 = [scale * rng.standard_normal((nocca, nvira)),
          scale * rng.standard_normal((noccb, nvirb))]
    t2aa = scale * rng.standard_normal((nocca, nocca, nvira, nvira))
    t2aa = t2aa - t2aa.transpose(1, 0, 2, 3)
    t2aa = t2aa - t2aa.transpose(0, 1, 3, 2)
    t2bb = scale * rng.standard_normal((noccb, noccb, nvirb, nvirb))
    t2bb = t2bb - t2bb.transpose(1, 0, 2, 3)
    t2bb = t2bb - t2bb.transpose(0, 1, 3, 2)
    t2ab = scale * rng.standard_normal((nocca, noccb, nvira, nvirb))
    return t1, [t2aa, t2ab, t2bb]


def brute_candidate_outputs(engine):
    """Old six-shell UCISD candidate construction, kept as test ground truth."""
    raised = dict(delta_alpha=1, delta_beta=-1)
    outputs = ()
    for holes_a, holes_b in ((0, 1), (1, 1), (0, 2),
                             (2, 1), (1, 2), (0, 3)):
        outputs += engine._determinant_shell(
            holes_a, holes_b, **raised
        )
    return outputs


# The equation count must not depend on any overlap element: an identity overlap
# and a dense random one describe very different operators but the same set of
# structurally possible preimages, so they must retain the same equations.  The
# closed form is the rank-one shell of the raised sector, nvirt_alpha * nocc_beta.
for nmo, nocc in ((4, (3, 2)), (6, (3, 3)), (10, (7, 7))):
    counts = set()
    for overlap in (numpy.eye(nmo), rng.standard_normal((nmo, nmo)),
                    0.01 * rng.standard_normal((nmo, nmo))):
        counts.add(UCISDSpinConstraint(nmo, nocc, overlap).nconstraint)
    assert len(counts) == 1
    assert counts.pop() == (nmo - nocc[0]) * nocc[1]

# A reference whose alpha and beta occupied spaces span the same subspace is a
# closed-shell singlet, so S+ annihilates it and every equation must have a
# vanishing constant term -- a check that the dense S+ carries the right signs,
# and one that a diagonal overlap alone would not exercise.
NMO, NOCC = 6, (3, 3)
block_overlap = numpy.zeros((NMO, NMO))
block_overlap[:NOCC[0], :NOCC[1]] = orthogonal(NOCC[0])
block_overlap[NOCC[0]:, NOCC[1]:] = orthogonal(NMO - NOCC[0])
block_engine = UCISDSpinConstraint(NMO, NOCC, block_overlap)
assert numpy.max(numpy.abs(block_engine.constant)) < 1e-12
assert numpy.max(numpy.abs(block_overlap - numpy.diag(numpy.diag(block_overlap)))) > 0.1

# Ground truth: apply S+ forward over the whole SD space and compare with the
# compiled equations.  The amplitude-to-determinant map (_build_sd_space) is
# already validated against the exact cluster exponential in test 19, so the new
# ingredient under test is the S+ representation and the retention rule.
dense_overlap = rng.standard_normal((NMO, NMO))
engine = UCISDSpinConstraint(NMO, NOCC, dense_overlap)
t1, t2 = random_amplitudes(NMO, NOCC)
values = dict(zip(("t1a", "t1b", "t2aa", "t2ab", "t2bb"), t1 + t2))


def sd_coefficients(space):
    """Coefficient of every SD determinant in (1 + T1 + T2 + T1^2/2)|0>."""
    coefficients = {engine._reference: 1.0}
    for state in space["determinants"]:
        if state == engine._reference:
            continue
        block, index = space["direct"][state][0]
        total = space["direct"][state][1] * values[block][index]
        for pair, coefficient in space["quadratic"].get(state, {}).items():
            (first_block, first), (second_block, second) = pair
            total += (coefficient * values[first_block][first]
                      * values[second_block][second])
        coefficients[state] = total
    return coefficients


def raise_forward(state, alpha, beta):
    """One S+ term, written out independently of the engine's own kernel."""
    output, sign = _apply_sequence(
        state, (("annihilate", NMO + beta), ("create", alpha)))
    if output is None:
        return None, 0.0
    return output, sign * dense_overlap[alpha, beta]


space = engine._build_sd_space()
coefficients = sd_coefficients(space)
forward = {}
for state, coefficient in coefficients.items():
    for alpha in range(NMO):
        for beta in range(NMO):
            output, weight = raise_forward(state, alpha, beta)
            if output is not None and weight != 0.0:
                forward[output] = forward.get(output, 0.0) + weight * coefficient

compiled = engine.constraint(t1, t2)
outputs = sorted(
    output for output in brute_candidate_outputs(engine)
    if engine._is_retained(output)
)
assert len(outputs) == engine.nconstraint == compiled.size
assert engine.ncandidate == len(brute_candidate_outputs(engine))
assert engine.nomitted == engine.ncandidate - engine.nconstraint
forward_error = max(
    abs(forward.get(output, 0.0) - compiled[row])
    for row, output in enumerate(outputs)
)
assert forward_error < 1e-12

# The other direction: an omitted output must genuinely reach outside the SD
# space, otherwise it was discarded for nothing.  Rank-three determinants are the
# only ones S+ can lift into the omitted rows.
omitted = [output for output in brute_candidate_outputs(engine)
           if not engine._is_retained(output)]
rank_three = [state for state in engine._determinant_shell(2, 1)
              + engine._determinant_shell(1, 2) + engine._determinant_shell(3, 0)
              + engine._determinant_shell(0, 3)]
reachable_from_rank_three = set()
for state in rank_three:
    for alpha in range(NMO):
        for beta in range(NMO):
            output, weight = raise_forward(state, alpha, beta)
            if output is not None and weight != 0.0:
                reachable_from_rank_three.add(output)
assert omitted, "this reference must omit something for the check to mean anything"
assert all(output in reachable_from_rank_three for output in omitted)

# Against the matched engine: with an identity overlap the two describe the same
# operator, so every equation the UCISD engine keeps must be one the matched
# engine also keeps, with the same compiled coefficients.  The UCISD engine
# keeps far fewer, which is the price of not consulting the overlap.
matched = ClosedCISDSpinConstraint(NMO, NOCC)
identity_engine = UCISDSpinConstraint(NMO, NOCC, numpy.eye(NMO))
matched_value = matched.constraint(t1, t2)
identity_value = identity_engine.constraint(t1, t2)
matched_outputs = sorted(
    output for output in matched._candidate_outputs()[0]
    if matched._is_retained(output)
)
identity_outputs = sorted(
    output for output in brute_candidate_outputs(identity_engine)
    if identity_engine._is_retained(output)
)
assert set(identity_outputs) < set(matched_outputs)
matched_row = {output: row for row, output in enumerate(matched_outputs)}
subset_error = max(
    abs(identity_value[row] - matched_value[matched_row[output]])
    for row, output in enumerate(identity_outputs)
)
assert subset_error < 1e-12

# Analytic Jacobian against central differences, in the engine's own coordinates.
analytic = engine.jacobian(t1, t2).copy()
test_vector = rng.standard_normal(len(engine.coordinates))
test_weights = rng.standard_normal(engine.nconstraint)
test_metric = rng.random(len(engine.coordinates))
assert numpy.max(numpy.abs(
    engine.jvp(t1, t2, test_vector) - analytic @ test_vector
)) < 1e-13
assert numpy.max(numpy.abs(
    engine.vjp(t1, t2, test_weights) - analytic.T @ test_weights
)) < 1e-13
assert numpy.max(numpy.abs(
    engine.jacobian_row_norms(t1, t2)
    - numpy.linalg.norm(analytic, axis=1)
)) < 1e-13
assert numpy.max(numpy.abs(
    engine.weighted_gram_diagonal(t1, t2, test_metric)
    - numpy.einsum("ri,i,ri->r", analytic, test_metric, analytic)
)) < 1e-13
blocks = dict(zip(("t1a", "t1b", "t2aa", "t2ab", "t2bb"), t1 + t2))
step = 1e-6
jacobian_error = 0.0
for _ in range(4):
    direction = rng.standard_normal(len(engine.coordinates))
    direction /= numpy.linalg.norm(direction)
    saved = [blocks[block][index] for block, index in engine.coordinates]
    shifted = []
    for sign in (1.0, -1.0):
        for value, delta, (block, index) in zip(saved, direction,
                                                engine.coordinates):
            blocks[block][index] = value + sign * step * delta
        shifted.append(engine.constraint(t1, t2).copy())
    for value, (block, index) in zip(saved, engine.coordinates):
        blocks[block][index] = value
    finite = (shifted[0] - shifted[1]) / (2.0 * step)
    jacobian_error = max(jacobian_error,
                         float(numpy.max(numpy.abs(finite - analytic @ direction))))
assert jacobian_error < 1e-8

# Same derivative checks with unequal alpha and beta occupancies.  Above,
# NOCC = (3, 3) in six orbitals makes nocca, noccb, nvira and nvirb all equal,
# so every array the engine builds is square in each pair of axes and a
# transposed or swapped index would still produce correctly shaped output.
# (3, 2) makes the four extents 3, 2, 3, 4 -- all distinct -- so the layout of
# the t2ab block is actually pinned down.  jvp and vjp are independent code
# paths, and `jacobian` is assembled from vjp one row at a time, so comparing
# jvp against it cross-checks the two; the finite differences below are what
# tie them to the constraint function itself.
UNEVEN_NMO, UNEVEN_NOCC = 6, (3, 2)
uneven_engine = UCISDSpinConstraint(
    UNEVEN_NMO, UNEVEN_NOCC, rng.standard_normal((UNEVEN_NMO, UNEVEN_NMO)))
uneven_t1, uneven_t2 = random_amplitudes(UNEVEN_NMO, UNEVEN_NOCC)
assert len(set(uneven_engine.coordinates.nocc
               + (uneven_engine.coordinates.nvira,
                  uneven_engine.coordinates.nvirb))) == 3, \
    "this case exists to make the alpha and beta extents differ"

uneven_analytic = uneven_engine.jacobian(uneven_t1, uneven_t2).copy()
uneven_vector = rng.standard_normal(len(uneven_engine.coordinates))
uneven_weights = rng.standard_normal(uneven_engine.nconstraint)
assert numpy.max(numpy.abs(
    uneven_engine.jvp(uneven_t1, uneven_t2, uneven_vector)
    - uneven_analytic @ uneven_vector
)) < 1e-13
assert numpy.max(numpy.abs(
    uneven_engine.vjp(uneven_t1, uneven_t2, uneven_weights)
    - uneven_analytic.T @ uneven_weights
)) < 1e-13

uneven_blocks = dict(zip(("t1a", "t1b", "t2aa", "t2ab", "t2bb"),
                         uneven_t1 + uneven_t2))
uneven_error = 0.0
for _ in range(4):
    direction = rng.standard_normal(len(uneven_engine.coordinates))
    direction /= numpy.linalg.norm(direction)
    saved = [uneven_blocks[block][index]
             for block, index in uneven_engine.coordinates]
    shifted = []
    for sign in (1.0, -1.0):
        for value, delta, (block, index) in zip(
                saved, direction, uneven_engine.coordinates):
            uneven_blocks[block][index] = value + sign * step * delta
        shifted.append(uneven_engine.constraint(uneven_t1, uneven_t2).copy())
    for value, (block, index) in zip(saved, uneven_engine.coordinates):
        uneven_blocks[block][index] = value
    finite = (shifted[0] - shifted[1]) / (2.0 * step)
    uneven_error = max(uneven_error, float(numpy.max(numpy.abs(
        finite - uneven_analytic @ direction))))
assert uneven_error < 1e-8
print("unequal occupancy (3,2): jvp/vjp consistent, "
      "finite-difference error = %.2e" % uneven_error)


# Hessian-vector product against finite differences of the Jacobian.
weights = rng.standard_normal(engine.nconstraint)
d1, d2 = random_amplitudes(NMO, NOCC, scale=1.0)
direction = engine.amplitude_values(d1, d2)
analytic_hvp = engine.hessian_vector(t1, t2, weights, direction)
plus = [t1[0] + step * d1[0], t1[1] + step * d1[1]]
minus = [t1[0] - step * d1[0], t1[1] - step * d1[1]]
plus2 = [t2[i] + step * d2[i] for i in range(3)]
minus2 = [t2[i] - step * d2[i] for i in range(3)]
finite_hvp = (weights @ engine.jacobian(plus, plus2)
              - weights @ engine.jacobian(minus, minus2)) / (2.0 * step)
assert numpy.max(numpy.abs(finite_hvp - analytic_hvp)) < 1e-8

# End to end on genuine UHF orbitals: stretched H2 breaks spin symmetry, so the
# alpha and beta MOs are different spatial functions rather than a permutation of
# a shared set -- exactly the case the matched engine rejects.  The whole FCI
# space is reference plus singles and doubles here, so the constrained solution
# must be the exact singlet.
mol = gto.M(atom="H 0 0 0; H 0 0 2.2", basis="sto-3g", spin=0, verbose=0)
uhf = scf.UHF(mol)
uhf.kernel()
mo, _, stable, _ = uhf.stability(return_status=True)
while not stable:
    uhf.kernel(dm0=uhf.make_rdm1(mo, uhf.mo_occ))
    mo, _, stable, _ = uhf.stability(return_status=True)
assert uhf.spin_square()[0] > 0.5, "the UHF reference must be broken-symmetry"

probe = uccsd.UCCSD(uhf)
cross = (uhf.mo_coeff[0].T @ uhf.get_ovlp() @ uhf.mo_coeff[1])
assert numpy.min(numpy.abs(cross)) > 1e-3, "alpha and beta MOs must be unmatched"
try:
    ClosedCISDSpinConstraint.from_cc(probe)
except ValueError as error:
    assert "signed permutation" in str(error)
else:
    raise AssertionError("the matched engine must reject unmatched orbitals")

h2_engine = UCISDSpinConstraint.from_cc(probe)
assert h2_engine.nconstraint > 0
assert h2_engine.frozen_coupling == 0.0

h2cc = configure(MultiConstrUCCSD(
    uhf, h2_engine.constraint, h2_engine.jacobian, h2_engine.coordinates,
    constraint_tol=1e-10))
h2cc.kernel()
assert h2cc.converged
assert numpy.max(numpy.abs(h2cc.constraint_residual())) < 1e-9
assert h2cc.check_constraint_jacobian(ndir=3) < 1e-8

energies = fci.FCI(scf.RHF(mol).run(verbose=0)).kernel(nroots=2)[0]
assert abs(h2cc.e_tot - energies[0]) < 1e-8

plain = configure(uccsd.UCCSD(uhf))
plain.kernel()

print("counts (nmo, nocc) -> nconstraint: (4,(3,2)) 2, (6,(3,3)) 9, (10,(7,7)) 21")
print("matched engine keeps %d of the same reference's equations, UCISD keeps %d"
      % (matched.nconstraint, identity_engine.nconstraint))
print("forward S+ vs compiled equations = %.3e" % forward_error)
print("subset coefficient agreement = %.3e" % subset_error)
print("constraint Jacobian max error = %.3e" % jacobian_error)
print("H2 UHF <S^2> = %.6f, constrained CC = %.12f, FCI singlet = %.12f"
      % (uhf.spin_square()[0], h2cc.e_tot, energies[0]))
print("unconstrained UCCSD on the same reference = %.12f" % plain.e_tot)
print("row-norm range = %.3e .. %.3e"
      % (engine.jacobian_row_norms(t1, t2).min(),
         engine.jacobian_row_norms(t1, t2).max()))

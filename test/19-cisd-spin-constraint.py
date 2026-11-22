import numpy
from pyscf.cc import uccsd
from pyscf.fci import cistring

from ccc.active_spin_constraint import DominantSectorSpinConstraint
from ccc.cisd_spin_constraint import ClosedCISDSpinConstraint


def equation_terms(equation):
    return {
        preimage.occupation: preimage.coefficient
        for preimage in equation.preimages
    }


# The complete S+ equations for |2baa> contain four output determinants, but
# only two have all of their FCI preimages at reference, single, or double rank.
alpha_permutation = numpy.asarray([0, 2, 3, 1])
beta_permutation = numpy.arange(4)
example = ClosedCISDSpinConstraint(
    4,
    (3, 2),
    alpha_permutation=alpha_permutation,
    beta_permutation=beta_permutation,
)
assert example.ndeterminant == 21
assert example.ncandidate == 4
assert example.nconstraint == 2
assert example.nomitted == 2
equations = {equation.output: equation_terms(equation)
             for equation in example.equations}
assert equations == {
    "2aaa": {"2baa": -1.0, "2aba": 1.0, "2aab": -1.0},
    "a2aa": {"b2aa": -1.0, "a2ba": 1.0, "a2ab": -1.0},
}


# Metadata uses a canonical common-orbital determinant convention and is thus
# unchanged when one of the stored beta MOs has the opposite phase.
phase_example = ClosedCISDSpinConstraint(
    4,
    (3, 2),
    alpha_permutation=alpha_permutation,
    beta_permutation=beta_permutation,
    beta_phases=numpy.asarray([-1.0, 1.0, 1.0, 1.0]),
)
phase_equations = {
    equation.output: equation_terms(equation)
    for equation in phase_example.equations
}
assert phase_equations == equations


# Compare the sparse polynomial representation against the exact finite CC
# exponential in a seven-orbital FCI array, including same-spin amplitudes.
nmo = 7
nocc = (4, 3)
engine = ClosedCISDSpinConstraint(nmo, nocc)
active = DominantSectorSpinConstraint(
    nmo, nocc, "222a000",
    alpha_permutation=numpy.arange(nmo),
    beta_permutation=numpy.arange(nmo),
)
rng = numpy.random.default_rng(19)
zero_t1 = (numpy.zeros((4, 3)), numpy.zeros((3, 4)))
zero_t2 = (
    numpy.zeros((4, 4, 3, 3)),
    numpy.zeros((4, 3, 3, 4)),
    numpy.zeros((3, 3, 4, 4)),
)
vector = rng.normal(scale=0.04, size=uccsd.amplitudes_to_vector(
    zero_t1, zero_t2
).size)
t1, t2 = uccsd.vector_to_amplitudes(vector, (nmo, nmo), nocc)
exact = active.wavefunction(t1, t2)
blocks = dict(zip(
    ("t1a", "t1b", "t2aa", "t2ab", "t2bb"), t1 + t2
))
workspace = engine._build_sd_space()
maximum = 0.0
mask = (1 << nmo) - 1
for state in workspace["determinants"]:
    alpha = state & mask
    beta = state >> nmo
    ia = cistring.str2addr(nmo, nocc[0], alpha)
    ib = cistring.str2addr(nmo, nocc[1], beta)
    coefficient = 1.0 if state == engine._reference else 0.0
    if state != engine._reference:
        coordinate, sign = workspace["direct"][state]
        coefficient += sign * blocks[coordinate[0]][coordinate[1]]
    for (first, second), value in workspace["quadratic"].get(
            state, {}).items():
        coefficient += (
            value
            * blocks[first[0]][first[1]]
            * blocks[second[0]][second[1]]
        )
    maximum = max(maximum, abs(coefficient - exact[ia, ib]))
assert maximum < 1e-12


# Check the compiled analytic Jacobian in its compact amplitude coordinates.
values = engine.amplitude_values(t1, t2)
analytic = engine.jacobian(t1, t2).copy()
jacobian_error = 0.0
for _ in range(4):
    direction = rng.normal(size=values.size)
    direction /= numpy.linalg.norm(direction)
    results = []
    for sign in (1.0, -1.0):
        for value, delta, (block, index) in zip(
                values, direction, engine.coordinates):
            blocks[block][index] = value + sign * 1e-6 * delta
        results.append(engine.constraint(t1, t2).copy())
    for value, (block, index) in zip(values, engine.coordinates):
        blocks[block][index] = value
    finite_difference = (results[0] - results[1]) / 2e-6
    jacobian_error = max(
        jacobian_error,
        float(numpy.max(numpy.abs(
            finite_difference - analytic @ direction
        ))),
    )
assert jacobian_error < 1e-8

print("|2baa> retained, omitted constraints =", example.nconstraint,
      example.nomitted)
print("CISD coefficient max error = %.3e" % maximum)
print("constraint Jacobian max error = %.3e" % jacobian_error)

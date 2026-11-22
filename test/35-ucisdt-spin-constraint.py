"""UCISDT spin constraints: explicit wavefunction and matrix-free checks."""

from itertools import combinations

import numpy
from pyscf import fci, gto, scf
from pyscf.cc import uccsdt_highm
from scipy.linalg import expm

from ccc.ucisd_spin_constraint import UCISDSpinConstraint
from ccc.ucisdt_spin_constraint import UCISDTSpinConstraint
from ccc.uccsdt_multiconstr import MultiConstrUCCSDT, _public_amplitudes


rng = numpy.random.default_rng(35)


def apply_annihilation(state, orbital):
    if not state & (1 << orbital):
        return None, 0.0
    sign = -1.0 if (state & ((1 << orbital) - 1)).bit_count() % 2 else 1.0
    return state ^ (1 << orbital), sign


def apply_creation(state, orbital):
    if state & (1 << orbital):
        return None, 0.0
    sign = -1.0 if (state & ((1 << orbital) - 1)).bit_count() % 2 else 1.0
    return state | (1 << orbital), sign


def apply_sequence(state, annihilate, create):
    sign = 1.0
    for orbital in annihilate:
        state, phase = apply_annihilation(state, orbital)
        if state is None:
            return None, 0.0
        sign *= phase
    for orbital in create:
        state, phase = apply_creation(state, orbital)
        if state is None:
            return None, 0.0
        sign *= phase
    return state, sign


def determinant_states(nmo, nelec):
    alpha = [sum(1 << i for i in occupied)
             for occupied in combinations(range(nmo), nelec[0])]
    beta = [sum(1 << i for i in occupied)
            for occupied in combinations(range(nmo), nelec[1])]
    return tuple(a | (b << nmo) for a in alpha for b in beta)


def excitation_operator(nmo, nelec, coordinate):
    nocca, noccb = nelec
    block, index = coordinate
    if block == "t1a":
        i, a = index
        return (i,), (nocca + a,)
    if block == "t1b":
        i, a = index
        return (nmo + i,), (nmo + noccb + a,)
    if block == "t2aa":
        i, j, a, b = index
        return (i, j), (nocca + b, nocca + a)
    if block == "t2ab":
        i, j, a, b = index
        return (i, nmo + j), (nmo + noccb + b, nocca + a)
    if block == "t2bb":
        i, j, a, b = index
        return (nmo + i, nmo + j), (nmo + noccb + b, nmo + noccb + a)
    if block == "t3aab":
        i, j, k, a, b, c = index
        return ((i, j, nmo + k),
                (nmo + noccb + c, nocca + b, nocca + a))
    if block == "t3abb":
        i, j, k, a, b, c = index
        return ((i, nmo + j, nmo + k),
                (nmo + noccb + c, nmo + noccb + b, nocca + a))
    raise ValueError(block)


def explicit_wavefunction(engine, values):
    states = determinant_states(engine.nmo, engine.nocc)
    address = {state: i for i, state in enumerate(states)}
    cluster = numpy.zeros((len(states), len(states)))
    for coordinate, value in zip(engine.coordinates, values):
        annihilate, create = excitation_operator(
            engine.nmo, engine.nocc, coordinate
        )
        for column, state in enumerate(states):
            output, sign = apply_sequence(state, annihilate, create)
            if output in address:
                cluster[address[output], column] += sign * value
    reference = ((1 << engine.nocc[0]) - 1
                 | (((1 << engine.nocc[1]) - 1) << engine.nmo))
    vector = numpy.zeros(len(states))
    vector[address[reference]] = 1.0
    return states, expm(cluster) @ vector


def explicit_raised_rows(engine, states, wavefunction):
    raised = {}
    for state, coefficient in zip(states, wavefunction):
        for alpha in range(engine.nmo):
            for beta in range(engine.nmo):
                output, sign = apply_sequence(
                    state, (engine.nmo + beta,), (alpha,)
                )
                if output is not None:
                    raised[output] = raised.get(output, 0.0) + (
                        coefficient * sign * engine.overlap[alpha, beta]
                    )

    nocca, noccb = engine.nocc
    nvira, nvirb = engine.nmo - nocca, engine.nmo - noccb
    one = numpy.zeros((noccb, nvira))
    for j in range(noccb):
        for a in range(nvira):
            alpha = (1 << nocca) - 1 | (1 << (nocca + a))
            beta = ((1 << noccb) - 1) ^ (1 << j)
            one[j, a] = raised.get(alpha | (beta << engine.nmo), 0.0)

    aa = numpy.zeros((nocca, noccb, nvira, nvira))
    for i in range(nocca):
        for j in range(noccb):
            for a, b in combinations(range(nvira), 2):
                alpha = ((1 << nocca) - 1) ^ (1 << i)
                alpha |= (1 << (nocca + a)) | (1 << (nocca + b))
                beta = ((1 << noccb) - 1) ^ (1 << j)
                aa[i, j, a, b] = raised.get(
                    alpha | (beta << engine.nmo), 0.0
                )

    bb = numpy.zeros((noccb, noccb, nvira, nvirb))
    for j, k in combinations(range(noccb), 2):
        for a in range(nvira):
            for b in range(nvirb):
                alpha = (1 << nocca) - 1 | (1 << (nocca + a))
                beta = ((1 << noccb) - 1) ^ (1 << j) ^ (1 << k)
                beta |= 1 << (noccb + b)
                bb[j, k, a, b] = raised.get(
                    alpha | (beta << engine.nmo), 0.0
                )
    return engine._pack_rows(one, aa, bb)


# Counts and the old rank-one equations are independent of overlap values.
for nmo, nocc in ((5, (3, 2)), (6, (3, 3)), (7, (4, 2))):
    overlap = rng.normal(size=(nmo, nmo))
    engine = UCISDTSpinConstraint(nmo, nocc, overlap)
    nocca, noccb = nocc
    nvira, nvirb = nmo - nocca, nmo - noccb
    expected = (noccb * nvira
                + nocca * noccb * (nvira * (nvira - 1) // 2)
                + nvira * nvirb * (noccb * (noccb - 1) // 2))
    assert engine.nconstraint == expected


# Independent exp(T) and S+ application.  Absolute values remove only the
# arbitrary row phase associated with representing a raised determinant.
NMO, NOCC = 5, (3, 2)
overlap = rng.normal(size=(NMO, NMO))
engine = UCISDTSpinConstraint(NMO, NOCC, overlap)
values = 0.04 * rng.normal(size=len(engine.coordinates))
amplitudes = engine.coordinates.expand_vector(values)
states, wavefunction = explicit_wavefunction(engine, values)
explicit = explicit_raised_rows(engine, states, wavefunction)
compiled = engine.constraint(*amplitudes)
explicit_error = float(numpy.max(numpy.abs(
    numpy.abs(explicit) - numpy.abs(compiled)
)))
assert explicit_error < 1e-11

ucisd = UCISDSpinConstraint(NMO, NOCC, overlap)
subset_error = float(numpy.max(numpy.abs(
    compiled[:ucisd.nconstraint] - ucisd.constraint(
        amplitudes[0], amplitudes[1]
    )
)))
assert subset_error < 1e-12


# Analytic JVP/VJP and exact small-system Gram diagnostics.
direction = rng.normal(size=len(engine.coordinates))
weights = rng.normal(size=engine.nconstraint)
jvp = engine.jvp(*amplitudes, direction)
vjp = engine.vjp(*amplitudes, weights)
adjoint_error = abs(weights @ jvp - vjp @ direction)
assert adjoint_error < 1e-10

step = 1e-6
plus = engine.coordinates.expand_vector(values + step * direction)
minus = engine.coordinates.expand_vector(values - step * direction)
finite = (engine.constraint(*plus) - engine.constraint(*minus)) / (2 * step)
finite_error = float(numpy.max(numpy.abs(finite - jvp)))
assert finite_error < 1e-8

jacobian = engine.jacobian(*amplitudes)
assert numpy.max(numpy.abs(jacobian @ direction - jvp)) < 1e-11
metric = rng.random(len(engine.coordinates))
diagonal = engine.weighted_gram_diagonal(*amplitudes, metric)
assert numpy.max(numpy.abs(
    diagonal - numpy.einsum("ri,i,ri->r", jacobian, metric, jacobian)
)) < 1e-11
assert numpy.max(numpy.abs(
    engine.jacobian_row_norms(*amplitudes)
    - numpy.linalg.norm(jacobian, axis=1)
)) < 1e-11


# PySCF's packed mixed-spin T3 convention and the rank-three engine adapter.
mol = gto.M(
    atom=";".join("H 0 0 %.1f" % atom for atom in range(6)),
    basis="sto-3g", spin=0, verbose=0,
)
mf = scf.UHF(mol).run(verbose=0)
packed_engine = UCISDTSpinConstraint.from_cc(uccsdt_highm.UCCSDT(mf))
cc = MultiConstrUCCSDT.from_constraint_engine(
    mf,
    packed_engine,
    constraint_jacobian_row_norms=(
        packed_engine.matrix_free_jacobian_row_norms
    ),
    constraint_weighted_gram_diagonal=(
        packed_engine.matrix_free_weighted_gram_diagonal
    ),
)
cc.diis = False
eris = cc.ao2mo(cc.mo_coeff)
_, tamps = cc.init_amps(eris)
cc.unique_tamps_map = cc.build_unique_tamps_map()
cc._ensure_coordinates(tamps)
public = _public_amplitudes(tamps)
random_packed_values = 0.02 * rng.normal(size=len(packed_engine.coordinates))
packed_engine.coordinates.set_values(*public, random_packed_values)
packed = cc.amplitudes_to_vector(tamps)
packed_error = float(numpy.max(numpy.abs(
    random_packed_values
    - cc._coordinate_sign * packed[cc._coordinate_flat]
)))
assert packed_error < 1e-13

# One production update must stay entirely matrix-free.
_, tamps = cc.init_amps(eris)
cc.max_cycle = 1
cc.constraint_max_step = 0.05

def reject_dense_jacobian(*args):
    raise AssertionError("matrix-free update called the dense Jacobian")

cc.constraint_jacobian = reject_dense_jacobian
cc.update_amps_(tamps, eris)
assert cc.constraint_ncall["jacobian"] == 0
assert cc.constraint_ncall["jvp"] > 0
assert cc.constraint_ncall["vjp"] > 0


# Three electrons make full UCCSDT exact.  A broken-symmetry UHF H3 reference
# exercises genuinely unmatched orbitals, and the constrained solution must
# recover the spin-doublet FCI ground state.
h3 = gto.M(
    atom="H 0 0 0; H 0 0 1.5; H 0 0 3.0",
    basis="6-31g", spin=1, verbose=0,
)
h3_uhf = scf.UHF(h3).run(verbose=0)
assert h3_uhf.spin_square()[0] > 0.8
h3_engine = UCISDTSpinConstraint.from_cc(uccsdt_highm.UCCSDT(h3_uhf))
h3_cc = MultiConstrUCCSDT.from_constraint_engine(
    h3_uhf, h3_engine, constraint_tol=1e-8, constraint_max_step=0.05
)
h3_cc.verbose = 0
h3_cc.max_cycle = 150
h3_cc.conv_tol = 1e-10
h3_cc.conv_tol_normt = 1e-8
h3_cc.diis_space = 20
h3_cc.diis_start_cycle = 3
h3_cc.kernel()
h3_fci = fci.FCI(scf.ROHF(h3).run(verbose=0))
h3_fci.spin = 1
h3_energy, h3_ci = h3_fci.kernel()
h3_s2 = h3_fci.spin_square(h3_ci, h3.nao, h3.nelec)[0]
assert h3_cc.converged
assert numpy.max(numpy.abs(h3_cc.constraint_value)) < 1e-8
assert abs(h3_cc.e_tot - h3_energy) < 2e-9
assert abs(h3_s2 - 0.75) < 1e-10

print("explicit S+ exp(T) error       = %.3e" % explicit_error)
print("UCISD subset error            = %.3e" % subset_error)
print("JVP finite-difference error   = %.3e" % finite_error)
print("JVP/VJP adjoint error         = %.3e" % adjoint_error)
print("packed UCCSDT mapping error   = %.3e" % packed_error)
print("matrix-free UCCSDT update used %d JVP and %d VJP calls"
      % (cc.constraint_ncall["jvp"], cc.constraint_ncall["vjp"]))
print("H3 constrained UCCSDT - FCI  = %.3e" % (h3_cc.e_tot - h3_energy))

import numpy
import pyscf
from pyscf.cc import uccsd, uccsdt_highm

from ccc import MultiConstrUCCSD, MultiConstrUCCSDT
from ccc.active_spin_constraint import DominantSectorSpinConstraint
from ccc.uccsdt_multiconstr import _public_amplitudes


def h_chain(natom, basis="sto-3g", spin=0):
    mol = pyscf.M(
        atom=";".join("H 0 0 %.2f" % (1.2 * i) for i in range(natom)),
        basis=basis,
        spin=spin,
        verbose=0,
    )
    return mol.UHF().run(verbose=0)


def configure(cc):
    cc.verbose = 4
    cc.max_cycle = 150
    cc.conv_tol = 1e-10
    cc.conv_tol_normt = 1e-8
    cc.diis_space = 30
    cc.diis_start_cycle = 5
    return cc


# PySCF UCCSDT stores t2ab as (i,a,J,B); callbacks must see the UCCSD
# convention (i,J,a,B).  T1 and the same-spin T2 blocks already agree.
mf3 = h_chain(3, basis="6-31g", spin=1)
ccsd = uccsd.UCCSD(mf3)
ccsdt = uccsdt_highm.UCCSDT(mf3)
_, t1sd, t2sd = ccsd.init_amps(ccsd.ao2mo(ccsd.mo_coeff))
_, tamps = ccsdt.init_amps(ccsdt.ao2mo(ccsdt.mo_coeff))
t1t, t2t, _ = _public_amplitudes(tamps)
for left, right in zip(t1sd, t1t):
    assert numpy.allclose(left, right, atol=1e-13, rtol=0)
for left, right in zip(t2sd, t2t):
    assert numpy.allclose(left, right, atol=1e-13, rtol=0)


# Every public block and a noncanonical same-spin permutation must map to the
# correct independent component in PySCF's packed high-memory UCCSDT vector.
mf6 = h_chain(6)
probe = MultiConstrUCCSDT(
    mf6,
    lambda t1, t2, t3: numpy.asarray([t1[0][0, 0]]),
    lambda t1, t2, t3: numpy.ones((1, 1)),
    (("t1a", (0, 0)),),
)
_, probe_tamps = probe.init_amps()
probe.unique_tamps_map = probe.build_unique_tamps_map()
zero = numpy.zeros_like(probe.amplitudes_to_vector(probe_tamps))
coordinates = (
    ("t1a", (1, 2)),
    ("t1b", (2, 1)),
    ("t2aa", (0, 2, 2, 1)),
    ("t2ab", (1, 2, 0, 1)),
    ("t2bb", (2, 1, 0, 2)),
    ("t3aaa", (2, 0, 1, 1, 2, 0)),
    ("t3aab", (0, 2, 1, 2, 1, 0)),
    ("t3abb", (1, 2, 0, 0, 2, 1)),
    ("t3bbb", (1, 2, 0, 2, 0, 1)),
)
for block, index in coordinates:
    flat, sign = probe._coordinate_index(
        block, index, _public_amplitudes(probe_tamps)[0]
    )
    vector = zero.copy()
    vector[flat] = 1.0
    public = _public_amplitudes(probe.vector_to_amplitudes(vector))
    blocks = dict(zip(
        ("t1a", "t1b", "t2aa", "t2ab", "t2bb",
         "t3aaa", "t3aab", "t3abb", "t3bbb"),
        tuple(public[0]) + tuple(public[1]) + tuple(public[2]),
    ))
    assert abs(blocks[block][index] - sign) < 1e-13

    # Sparse coordinate assignment must exactly reproduce PySCF's packed
    # expansion, including every same-spin antisymmetry partner.
    value = 0.37
    expected_vector = zero.copy()
    expected_vector[flat] = sign * value
    expected = probe.vector_to_amplitudes(expected_vector)
    actual = probe.vector_to_amplitudes(zero.copy())
    probe.constraint_coordinates = ((block, index),)
    probe._set_coordinate_values(actual, (value,))
    for expected_rank, actual_rank in zip(expected, actual):
        for expected_block, actual_block in zip(expected_rank, actual_rank):
            assert numpy.array_equal(expected_block, actual_block)


# Adding a zero T3 must leave the old dominant-sector construction unchanged;
# nonzero triples must enter both the constraint and its analytic Jacobian.
rng = numpy.random.default_rng(7)
t1 = (rng.normal(scale=0.05, size=(3, 2)),
      rng.normal(scale=0.05, size=(2, 3)))
t2 = (rng.normal(scale=0.05, size=(3, 3, 2, 2)),
      rng.normal(scale=0.05, size=(3, 2, 2, 3)),
      rng.normal(scale=0.05, size=(2, 2, 3, 3)))
t3 = (rng.normal(scale=0.05, size=(3, 3, 3, 2, 2, 2)),
      rng.normal(scale=0.05, size=(3, 3, 2, 2, 2, 3)),
      rng.normal(scale=0.05, size=(3, 2, 2, 2, 3, 3)),
      rng.normal(scale=0.05, size=(2, 2, 2, 3, 3, 3)))
rank2 = DominantSectorSpinConstraint(5, (3, 2), "ababa")
rank3 = DominantSectorSpinConstraint(
    5, (3, 2), "ababa", max_cluster_rank=3
)
zero_t3 = tuple(numpy.zeros_like(block) for block in t3)
assert numpy.allclose(
    rank2.wavefunction(t1, t2), rank3.wavefunction(t1, t2, zero_t3),
    atol=1e-13, rtol=0,
)
assert numpy.max(numpy.abs(
    rank3.constraint(t1, t2, t3) - rank2.constraint(t1, t2)
)) > 1e-5
assert rank3.check_jacobian(t1, t2, t3, ndir=5) < 1e-8


# Full integration: start the difficult five-electron doublet from an exact
# dominant-sector CSF, converge constrained CCSD, then continue to CCSDT.
mf5 = h_chain(5, spin=1)
spin_sd = DominantSectorSpinConstraint(5, (3, 2), "ababa")
constrained_sd = configure(MultiConstrUCCSD(
    mf5,
    spin_sd.constraint,
    spin_sd.jacobian,
    spin_sd.coordinates,
    constraint_tol=1e-8,
))
eris_sd = constrained_sd.ao2mo(constrained_sd.mo_coeff)
t1_sd, t2_sd = constrained_sd.get_init_guess(eris_sd)
t1_sd = tuple(numpy.zeros_like(block) for block in t1_sd)
t2_sd = tuple(numpy.zeros_like(block) for block in t2_sd)
spin_exchange = (
    ((0, 0, 0, 0), 0.0),
    ((0, 1, 1, 0), 1.0),
    ((1, 0, 0, 1), 0.5),
    ((1, 1, 1, 1), 0.0),
    ((2, 0, 0, 2), 0.5),
    ((2, 1, 1, 2), 0.0),
)
for index, value in spin_exchange:
    t2_sd[1][index] = value
assert numpy.max(numpy.abs(spin_sd.constraint(t1_sd, t2_sd))) < 1e-13
constrained_sd.kernel(t1=t1_sd, t2=t2_sd, eris=eris_sd)
assert constrained_sd.converged
assert numpy.max(numpy.abs(constrained_sd.constraint_residual())) < 1e-8

spin = DominantSectorSpinConstraint(
    5, (3, 2), "ababa", max_cluster_rank=3
)
constrained = configure(MultiConstrUCCSDT(
    mf5,
    spin.constraint,
    spin.jacobian,
    spin.coordinates,
    constraint_tol=1e-8,
))
eris = constrained.ao2mo(constrained.mo_coeff)
_, initial = constrained.init_amps(eris)
initial[0] = tuple(block.copy() for block in constrained_sd.t1)
initial[1] = (
    constrained_sd.t2[0].copy(),
    constrained_sd.t2[1].transpose(0, 2, 1, 3).copy(),
    constrained_sd.t2[2].copy(),
)
constrained.unique_tamps_map = constrained.build_unique_tamps_map()
assert constrained.check_constraint_jacobian(initial, ndir=3) < 1e-8
assert numpy.max(numpy.abs(
    constrained.constraint_residual(*_public_amplitudes(initial))
)) < 1e-8
constrained.kernel(tamps=initial, eris=eris)
assert constrained.converged
assert numpy.max(numpy.abs(constrained.constraint_residual())) < 1e-8
assert abs(spin.sector_spin_square(
    *constrained.public_amplitudes()
) - 0.75) < 1e-12
assert max(
    numpy.max(numpy.abs(block))
    for block in constrained.public_amplitudes()[2]
) > 1e-4


# With two electrons all triples are structurally empty, so constrained
# UCCSDT must reduce exactly to constrained UCCSD.
mf2 = h_chain(2)
rhs = -0.05
coordinate = (("t2ab", (0, 0, 0, 0)),)


def constraint_sd(t1, t2):
    return numpy.asarray([t2[1][0, 0, 0, 0] - rhs])


def jacobian_sd(t1, t2):
    return numpy.ones((1, 1))


def constraint_sdt(t1, t2, t3):
    return constraint_sd(t1, t2)


def jacobian_sdt(t1, t2, t3):
    return jacobian_sd(t1, t2)


reduced_sd = configure(MultiConstrUCCSD(
    mf2, constraint_sd, jacobian_sd, coordinate, constraint_tol=1e-9
)).run()
reduced_sdt = configure(MultiConstrUCCSDT(
    mf2, constraint_sdt, jacobian_sdt, coordinate, constraint_tol=1e-9
)).run()
assert reduced_sd.converged and reduced_sdt.converged
assert abs(reduced_sd.e_tot - reduced_sdt.e_tot) < 1e-12

print("T1/T2 convention check passed")
print("rank-3 dominant-sector Jacobian passed")
print("E(constrained UCCSDT) = %.15f" % constrained.e_tot)
print("two-electron UCCSDT-UCCSD = %.3e" % (
    reduced_sdt.e_tot - reduced_sd.e_tot
))

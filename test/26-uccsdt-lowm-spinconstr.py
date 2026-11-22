import os
import tempfile

import numpy
import pyscf
from pyscf import lib

from ccc import (
    LowMemorySpinConstrUCCSDT,
    MultiConstrUCCSD,
    MultiConstrUCCSDT,
)
from ccc.active_spin_constraint import DominantSectorSpinConstraint


TRI_MAP_NAMES = (
    "t2c_map_6f_oa", "t2c_mask_6f_oa",
    "t2c_map_2f_oa", "t2c_mask_2f_oa",
    "t2c_map_6f_va", "t2c_mask_6f_va",
    "t2c_map_2f_va", "t2c_mask_2f_va",
    "t2c_map_6f_ob", "t2c_mask_6f_ob",
    "t2c_map_2f_ob", "t2c_mask_2f_ob",
    "t2c_map_6f_vb", "t2c_mask_6f_vb",
    "t2c_map_2f_vb", "t2c_mask_2f_vb",
)


def h_chain(natom, spin):
    mol = pyscf.M(
        atom=";".join("H 0 0 %.2f" % (1.2 * i) for i in range(natom)),
        basis="sto-3g",
        spin=spin,
        verbose=0,
    )
    return mol.UHF().run(verbose=0)


def configure(cc):
    cc.verbose = 0
    cc.max_cycle = 150
    cc.conv_tol = 1e-10
    cc.conv_tol_normt = 1e-8
    cc.diis_space = 30
    cc.diis_start_cycle = 5
    return cc


def setup_triangular_maps(cc):
    for name, value in zip(TRI_MAP_NAMES, cc.setup_tri2block()):
        setattr(cc, name, value)


def public_amplitudes(tamps, t3_full):
    return (
        tuple(tamps[0]),
        (tamps[1][0], tamps[1][1].transpose(0, 2, 1, 3), tamps[1][2]),
        (
            t3_full[0],
            t3_full[1].transpose(0, 1, 4, 2, 3, 5),
            t3_full[2].transpose(4, 0, 1, 5, 2, 3),
            t3_full[3],
        ),
    )


# Every public amplitude coordinate, including all four T3 spin blocks, must
# map exactly to PySCF's native triangular representation.
mf7 = h_chain(7, spin=1)
engine7 = DominantSectorSpinConstraint(
    7, (4, 3), "abababa", max_cluster_rank=3
)
probe = LowMemorySpinConstrUCCSDT(mf7, engine7)
eris7 = probe.ao2mo(probe.mo_coeff)
_, tamps7 = probe.init_amps(eris7)
probe.unique_tamps_map = probe.build_unique_tamps_map()
setup_triangular_maps(probe)
rng = numpy.random.default_rng(17)
values = rng.normal(scale=0.03, size=len(engine7.coordinates))
probe._set_coordinate_values(tamps7, values)
assert numpy.allclose(probe._coordinate_values(tamps7), values, atol=1e-14)

t3_full = probe.tamps_tri2full(tamps7[2])
public = public_amplitudes(tamps7, t3_full)
tensor_values = engine7.amplitude_values(*public)
assert numpy.allclose(tensor_values, values, atol=1e-14)
assert numpy.allclose(
    engine7.constraint(*public),
    engine7.constraint_from_values(values),
    atol=1e-13,
)
assert numpy.allclose(
    engine7.jacobian(*public),
    engine7.jacobian_from_values(values),
    atol=1e-13,
)
assert probe.check_constraint_jacobian(tamps7, ndir=3) < 1e-8

with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as handle:
    checkpoint = handle.name
try:
    probe.chkfile = checkpoint
    probe.e_corr = 0.0
    probe.dump_chk(tamps=tamps7)
    restored = lib.chkfile.load(checkpoint, "uccsdt")["tamps"]
    for original_rank, restored_rank in zip(tamps7, restored):
        for original, saved in zip(original_rank, restored_rank):
            assert original.shape == saved.shape
            assert numpy.array_equal(original, saved)
finally:
    os.remove(checkpoint)


# The triangular and full-storage constrained updates must be identical before
# DIIS, including the simultaneous T1/T2/T3 residual convention.
mf5 = h_chain(5, spin=1)
engine5 = DominantSectorSpinConstraint(
    5, (3, 2), "ababa", max_cluster_rank=3
)
low = LowMemorySpinConstrUCCSDT(mf5, engine5)
high = MultiConstrUCCSDT(
    mf5,
    engine5.constraint,
    engine5.jacobian,
    engine5.coordinates,
)
low.diis = high.diis = False
eris_low = low.ao2mo(low.mo_coeff)
eris_high = high.ao2mo(high.mo_coeff)
_, low_tamps = low.init_amps(eris_low)
low.unique_tamps_map = low.build_unique_tamps_map()
setup_triangular_maps(low)
high_tamps = [
    tuple(block.copy() for block in low_tamps[0]),
    tuple(block.copy() for block in low_tamps[1]),
    low.tamps_tri2full(low_tamps[2]),
]
low.update_amps_(low_tamps, eris_low)
high.update_amps_(high_tamps, eris_high)
expanded_low = [low_tamps[0], low_tamps[1], low.tamps_tri2full(low_tamps[2])]
for low_rank, high_rank in zip(expanded_low, high_tamps):
    for low_block, high_block in zip(low_rank, high_rank):
        assert numpy.allclose(low_block, high_block, atol=1e-12, rtol=0)


# Start from the same exact dominant-sector CCSD state used by the high-memory
# tests and verify full-T3 DIIS converges without ever expanding global T3.
engine_sd = DominantSectorSpinConstraint(5, (3, 2), "ababa")
ccsd = configure(MultiConstrUCCSD(
    mf5,
    engine_sd.constraint,
    engine_sd.jacobian,
    engine_sd.coordinates,
    constraint_tol=1e-8,
))
eris_sd = ccsd.ao2mo(ccsd.mo_coeff)
t1_sd, t2_sd = ccsd.get_init_guess(eris_sd)
t1_sd = tuple(numpy.zeros_like(block) for block in t1_sd)
t2_sd = tuple(numpy.zeros_like(block) for block in t2_sd)
for index, value in (
    ((0, 0, 0, 0), 0.0),
    ((0, 1, 1, 0), 1.0),
    ((1, 0, 0, 1), 0.5),
    ((1, 1, 1, 1), 0.0),
    ((2, 0, 0, 2), 0.5),
    ((2, 1, 1, 2), 0.0),
):
    t2_sd[1][index] = value
ccsd.kernel(t1=t1_sd, t2=t2_sd, eris=eris_sd)
assert ccsd.converged

constrained = configure(LowMemorySpinConstrUCCSDT(
    mf5, engine5, constraint_tol=1e-8
))
eris = constrained.ao2mo(constrained.mo_coeff)
_, initial = constrained.init_amps(eris)
initial[0] = tuple(block.copy() for block in ccsd.t1)
initial[1] = (
    ccsd.t2[0].copy(),
    ccsd.t2[1].transpose(0, 2, 1, 3).copy(),
    ccsd.t2[2].copy(),
)
constrained.tamps_tri2full = lambda *args: (_ for _ in ()).throw(
    AssertionError("low-memory iteration expanded T3")
)
constrained.kernel(tamps=initial, eris=eris)
assert constrained.converged
assert constrained.do_diis_max_t
assert numpy.max(numpy.abs(constrained.constraint_residual())) < 1e-8
assert abs(constrained.sector_spin_square() - 0.75) < 1e-12

print("triangular coordinate count =", len(engine7.coordinates))
print("low-memory constrained UCCSDT energy = %.15f" % constrained.e_tot)
print("final max|C| = %.3e" % numpy.max(
    numpy.abs(constrained.constraint_residual())
))

"""Matrix-free UCISD constraints on triangular-T3 UCCSDT.

`LowMemoryMultiConstrUCCSDT` stores T3 as i<=j<=k only.  For a rank-2
constraint that storage is invisible: the packed vector has the same layout in
the T1/T2 region as the full-tensor solver, so both must reach the same
constrained solution from the same reference.
"""
import numpy

import pyscf
from pyscf.cc import uccsd, uccsdt_highm

from ccc import LowMemoryMultiConstrUCCSDT, MultiConstrUCCSDT
from ccc.ucisd_spin_constraint import UCISDSpinConstraint
from ccc.ucisdt_spin_constraint import UCISDTSpinConstraint
from ccc.uccsdt_multiconstr import _checked_coordinate_denominators


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
    cc.max_cycle = 300
    cc.conv_tol = 1e-11
    cc.conv_tol_normt = 1e-8
    cc.diis_space = 20
    cc.diis_start_cycle = 5
    cc.mo_coeff = numpy.asarray(cc.mo_coeff)
    cc.mo_occ = numpy.asarray(cc.mo_occ)
    return cc


def solve(cls, mf, engine):
    cc = configure(cls.from_constraint_engine(
        mf, engine, constraint_tol=1e-9,
        constraint_multiplier_solver="linear",
        constraint_diis_scheme="tangent",
    ))
    jacobian_error = cc.check_constraint_jacobian(ndir=2)
    cc.kernel()
    return cc, jacobian_error


mf = h_chain(6, spin=2)
engine = UCISDSpinConstraint.from_cc(uccsd.UCCSD(mf))
assert engine.nconstraint > 0

full, full_error = solve(MultiConstrUCCSDT, mf, engine)
low, low_error = solve(LowMemoryMultiConstrUCCSDT, mf, engine)

# Both solvers must take the matrix-free path -- the whole point of routing
# UCISD through this class is never forming the Jacobian.
assert full._matrix_free_constraints
assert low._matrix_free_constraints

# T3 really is stored differently, or this test proves nothing.
assert low.tamps[2][1].ndim == 4, low.tamps[2][1].shape
assert full.tamps[2][1].ndim == 6, full.tamps[2][1].shape

assert full.converged and low.converged
assert abs(full.e_tot - low.e_tot) < 1e-9, (full.e_tot, low.e_tot)
assert full_error < 1e-6 and low_error < 1e-6, (full_error, low_error)
for cc in (full, low):
    assert numpy.max(numpy.abs(cc.constraint_residual())) < 1e-8

# The vectorized denominator fast path must agree with the generic
# coordinate-by-coordinate walk exactly, not just closely.
eris = low.ao2mo(low.mo_coeff)
fast = _checked_coordinate_denominators(low, eris)


class _NoFastPath:
    """The same coordinates, minus the layout shortcut."""

    def __init__(self, coordinates):
        self._coordinates = coordinates

    def __len__(self):
        return len(self._coordinates)

    def __iter__(self):
        return iter(self._coordinates)


saved = low.constraint_coordinates
low.constraint_coordinates = _NoFastPath(saved)
slow = _checked_coordinate_denominators(low, eris)
low.constraint_coordinates = saved
assert numpy.array_equal(fast, slow), numpy.max(numpy.abs(fast - slow))

# A rank-3 engine puts T3 in the coordinate set, whose indexing does differ
# between the two storages; triangular storage must refuse it rather than
# silently index the wrong amplitudes.
h3 = h_chain(3, spin=1)
rank3 = UCISDTSpinConstraint.from_cc(uccsdt_highm.UCCSDT(h3))
assert getattr(rank3.coordinates, "_ccc_amplitude_rank", 2) == 3
try:
    LowMemoryMultiConstrUCCSDT.from_constraint_engine(h3, rank3)
except NotImplementedError as error:
    assert "triangular" in str(error), error
else:
    raise AssertionError("rank-3 engine was accepted under triangular storage")

# ... while the full-storage solver still takes it.
MultiConstrUCCSDT.from_constraint_engine(h3, rank3)

print("E(full)       = %.12f" % full.e_tot)
print("E(triangular) = %.12f" % low.e_tot)
print("difference    = %+.3e Eh" % (low.e_tot - full.e_tot))
print("all checks passed")

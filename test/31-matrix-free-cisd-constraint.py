"""Dense and matrix-free CISD constraint derivatives must be interchangeable."""
import numpy
from pyscf import gto, scf
from pyscf.cc import uccsd

from ccc import MultiConstrUCCSD, MultiConstrUCCSDT
from ccc.cisd_spin_constraint import ClosedCISDSpinConstraint
from ccc.ucisd_spin_constraint import UCISDSpinConstraint


def dense_uccsd(mf, engine, **kwargs):
    # Lambdas deliberately hide the bound-method owner and force the legacy path.
    return MultiConstrUCCSD(
        mf,
        lambda t1, t2: engine.constraint(t1, t2),
        lambda t1, t2: engine.jacobian(t1, t2),
        engine.coordinates,
        **kwargs,
    )


def compare_uccsd_step(mf, engine):
    dense = dense_uccsd(mf, engine)
    sparse = MultiConstrUCCSD.from_constraint_engine(mf, engine)
    assert sparse.constraint_krylov_max_cycle is None
    eris = dense.ao2mo(dense.mo_coeff)
    t1, t2 = dense.get_init_guess(eris)
    dense_result = dense.update_amps(
        tuple(x.copy() for x in t1), tuple(x.copy() for x in t2), eris
    )
    sparse_result = sparse.update_amps(
        tuple(x.copy() for x in t1), tuple(x.copy() for x in t2), eris
    )
    error = max(
        numpy.max(numpy.abs(left - right))
        for left, right in zip(dense_result[0] + dense_result[1],
                               sparse_result[0] + sparse_result[1])
    )
    assert error < 1e-9
    assert sparse.constraint_ncall["jacobian"] == 0
    assert sparse.constraint_ncall["jvp"] > 0
    assert sparse.constraint_ncall["vjp"] > 0
    assert sparse.constraint_krylov_max_cycle == max(
        100, 2 * engine.nconstraint
    )
    return error


def compare_trusted_step(mf, engine, limit=1e-4):
    dense = dense_uccsd(mf, engine, constraint_max_step=limit)
    sparse = MultiConstrUCCSD.from_constraint_engine(
        mf, engine, constraint_max_step=limit
    )
    eris = dense.ao2mo(dense.mo_coeff)
    t1, t2 = dense.get_init_guess(eris)
    dense_result = dense.update_amps(
        tuple(x.copy() for x in t1), tuple(x.copy() for x in t2), eris
    )
    sparse_result = sparse.update_amps(
        tuple(x.copy() for x in t1), tuple(x.copy() for x in t2), eris
    )
    dense_vector = dense.amplitudes_to_vector(*dense_result)
    sparse_vector = sparse.amplitudes_to_vector(*sparse_result)
    error = float(numpy.max(numpy.abs(dense_vector - sparse_vector)))
    assert error < 2e-7
    return error


def check_nonlinear_step(mf, engine):
    sparse = MultiConstrUCCSD.from_constraint_engine(
        mf, engine, constraint_multiplier_solver="nonlinear"
    )
    eris = sparse.ao2mo(sparse.mo_coeff)
    t1, t2 = sparse.get_init_guess(eris)
    initial = sparse.constraint_residual(t1, t2)
    result = sparse.update_amps(
        tuple(x.copy() for x in t1), tuple(x.copy() for x in t2), eris
    )
    final = sparse.constraint_residual(*result)
    assert sparse.constraint_multiplier_success
    assert numpy.linalg.norm(final) < numpy.linalg.norm(initial) + 1e-12
    assert sparse.constraint_ncall["jacobian"] == 0
    return float(numpy.linalg.norm(final))


# Common spin orbitals exercise ClosedCISDSpinConstraint.
mol4 = gto.M(
    atom=";".join("H 0 0 %.1f" % (1.2 * atom) for atom in range(4)),
    basis="sto-3g", spin=0, verbose=0,
)
rhf4 = scf.RHF(mol4).run(verbose=0)
mf4 = scf.UHF(mol4)
mf4.mo_coeff = (rhf4.mo_coeff.copy(), rhf4.mo_coeff.copy())
mf4.mo_occ = (rhf4.mo_occ / 2, rhf4.mo_occ / 2)
dm4 = mf4.make_rdm1()
fock4 = mf4.get_fock(dm=dm4)
mf4.mo_energy = tuple(numpy.diag(c.T @ f @ c)
                      for c, f in zip(mf4.mo_coeff, fock4))
mf4.e_tot = mf4.energy_tot(dm=dm4)
mf4.converged = True
closed = ClosedCISDSpinConstraint.from_cc(uccsd.UCCSD(mf4))
closed_error = compare_uccsd_step(mf4, closed)
trusted_error = compare_trusted_step(mf4, closed)
nonlinear_norm = check_nonlinear_step(mf4, closed)


# Broken-symmetry H2 exercises dense alpha-beta orbital overlaps.
mol2 = gto.M(atom="H 0 0 0; H 0 0 2.2", basis="sto-3g", spin=0, verbose=0)
mf2 = scf.UHF(mol2).run(verbose=0)
unmatched = UCISDSpinConstraint.from_cc(uccsd.UCCSD(mf2))
unmatched_error = compare_uccsd_step(mf2, unmatched)


# UCCSDT uses a different packed layout.  Its engine constructor must give the
# same full amplitudes as the dense callback path after one update.
def constraint_sdt(t1, t2, t3):
    return closed.constraint(t1, t2)


def jacobian_sdt(t1, t2, t3):
    return closed.jacobian(t1, t2)


dense_sdt = MultiConstrUCCSDT(
    mf4, constraint_sdt, jacobian_sdt, closed.coordinates
)
sparse_sdt = MultiConstrUCCSDT.from_constraint_engine(mf4, closed)
assert sparse_sdt.constraint_krylov_max_cycle is None
dense_sdt.diis = sparse_sdt.diis = False
eris_sdt = dense_sdt.ao2mo(dense_sdt.mo_coeff)
_, initial = dense_sdt.init_amps(eris_sdt)
dense_tamps = [tuple(x.copy() for x in rank) for rank in initial]
sparse_tamps = [tuple(x.copy() for x in rank) for rank in initial]
dense_sdt.update_amps_(dense_tamps, eris_sdt)
sparse_sdt.update_amps_(sparse_tamps, eris_sdt)
sdt_error = max(
    numpy.max(numpy.abs(left - right))
    for dense_rank, sparse_rank in zip(dense_tamps, sparse_tamps)
    for left, right in zip(dense_rank, sparse_rank)
)
assert sdt_error < 1e-9
assert sparse_sdt.constraint_ncall["jacobian"] == 0
assert sparse_sdt.constraint_ncall["jvp"] > 0
assert sparse_sdt.constraint_ncall["vjp"] > 0
assert sparse_sdt.constraint_krylov_max_cycle == max(
    100, 2 * closed.nconstraint
)


def compare_ucisd_uccsdt_step(mf, engine):
    def constraint(t1, t2, t3):
        return engine.constraint(t1, t2)

    def jacobian(t1, t2, t3):
        return engine.jacobian(t1, t2)

    dense = MultiConstrUCCSDT(
        mf, constraint, jacobian, engine.coordinates
    )
    sparse = MultiConstrUCCSDT.from_constraint_engine(mf, engine)
    dense.diis = sparse.diis = False
    eris = dense.ao2mo(dense.mo_coeff)
    _, initial = dense.init_amps(eris)
    dense_tamps = [tuple(x.copy() for x in rank) for rank in initial]
    sparse_tamps = [tuple(x.copy() for x in rank) for rank in initial]
    dense.update_amps_(dense_tamps, eris)
    sparse.update_amps_(sparse_tamps, eris)
    error = max(
        numpy.max(numpy.abs(left - right))
        for dense_rank, sparse_rank in zip(dense_tamps, sparse_tamps)
        for left, right in zip(dense_rank, sparse_rank)
    )
    assert error < 1e-9
    return error


ucisd_sdt_error = compare_ucisd_uccsdt_step(mf2, unmatched)

print("ClosedCISD dense/matrix-free step error = %.3e" % closed_error)
print("UCISD dense/matrix-free step error      = %.3e" % unmatched_error)
print("UCCSDT dense/matrix-free step error     = %.3e" % sdt_error)
print("UCISD UCCSDT packed step error         = %.3e" % ucisd_sdt_error)
print("trust-region dense/matrix-free error    = %.3e" % trusted_error)
print("matrix-free nonlinear constraint norm   = %.3e" % nonlinear_norm)

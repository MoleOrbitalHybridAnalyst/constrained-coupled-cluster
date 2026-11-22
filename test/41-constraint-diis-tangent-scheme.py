"""constraint_diis_scheme="tangent" must never change the amplitude update
itself (a telescoping identity: the existing Newton correction is computed
identically in both schemes; only what run_diis treats as its error vector
changes), must produce a DIIS error vector with no appended constraint
segment, must still converge to a constraint-satisfying answer with DIIS
engaged, and must reject option combinations that have no defined meaning."""
import numpy
from pyscf import gto, scf
from pyscf.cc import uccsd
from scipy.linalg import expm

from ccc import MultiConstrGCCSD, MultiConstrUCCSD, MultiConstrUCCSDT
from ccc.cisd_spin_constraint import ClosedCISDSpinConstraint
from ccc.gcisd_spin_constraint import GCISDSpinConstraint, spin_matrices


def copied(t1, t2):
    return tuple(x.copy() for x in t1), tuple(x.copy() for x in t2)


def copied_tamps(tamps):
    return [tuple(block.copy() for block in rank) for rank in tamps]


def common_orbital_h4():
    # Common alpha/beta spin orbitals so the matrix-free CISD engine below
    # (part c) is usable; the hand-written algebraic constraints in (a),(b),(d)
    # do not need it but reuse the same reference for simplicity.
    mol = gto.M(
        atom=";".join("H 0 0 %.1f" % (1.2 * atom) for atom in range(4)),
        basis="sto-3g", spin=0, verbose=0,
    )
    rhf = scf.RHF(mol).run(verbose=0)
    mf = scf.UHF(mol)
    mf.mo_coeff = (rhf.mo_coeff.copy(), rhf.mo_coeff.copy())
    mf.mo_occ = (rhf.mo_occ / 2, rhf.mo_occ / 2)
    dm = mf.make_rdm1()
    fock = mf.get_fock(dm=dm)
    mf.mo_energy = tuple(numpy.diag(c.T @ f @ c)
                         for c, f in zip(mf.mo_coeff, fock))
    mf.e_tot = mf.energy_tot(dm=dm)
    mf.converged = True
    return mf


mf4 = common_orbital_h4()

coordinates = (
    ("t1a", (0, 0)),
    ("t1b", (0, 0)),
    ("t2ab", (0, 0, 0, 0)),
)


def constraint(t1, t2):
    x = t1[0][0, 0]
    y = t1[1][0, 0]
    z = t2[1][0, 0, 0, 0]
    return numpy.asarray([
        x + y + z + 5 * x**2 - 0.03,
        x - y + 3 * y**2 - 0.01,
    ])


def jacobian(t1, t2):
    x = t1[0][0, 0]
    y = t1[1][0, 0]
    return numpy.asarray([
        [1 + 10 * x, 1, 1],
        [1, -1 + 6 * y, 0],
    ])


sdt_coordinates = (
    ("t1a", (0, 0)),
    ("t1b", (0, 0)),
    ("t3aab", (0, 1, 0, 0, 1, 0)),
)


def constraint_sdt(t1, t2, t3):
    x = t1[0][0, 0]
    y = t1[1][0, 0]
    z = t3[1][0, 1, 0, 0, 1, 0]
    return numpy.asarray([
        x + y + z + 5 * x**2 - 0.03,
        x - y + 3 * y**2 - 0.01,
    ])


def jacobian_sdt(t1, t2, t3):
    x = t1[0][0, 0]
    y = t1[1][0, 0]
    return numpy.asarray([
        [1 + 10 * x, 1, 1],
        [1, -1 + 6 * y, 0],
    ])


# ---------------------------------------------- (a) equivalence with diis=False
def check_equivalence_ccsd():
    legacy = MultiConstrUCCSD(mf4, constraint, jacobian, coordinates,
                                constraint_diis_scheme="legacy")
    tangent = MultiConstrUCCSD(mf4, constraint, jacobian, coordinates,
                                 constraint_diis_scheme="tangent")
    legacy.diis = tangent.diis = False
    eris = legacy.ao2mo(legacy.mo_coeff)
    t1, t2 = legacy.get_init_guess(eris)
    legacy_t1, legacy_t2 = copied(t1, t2)
    tangent_t1, tangent_t2 = copied(t1, t2)
    worst = 0.0
    for _ in range(6):
        legacy_t1, legacy_t2 = legacy.update_amps(legacy_t1, legacy_t2, eris)
        tangent_t1, tangent_t2 = tangent.update_amps(tangent_t1, tangent_t2, eris)
        for left, right in zip(legacy_t1 + legacy_t2, tangent_t1 + tangent_t2):
            worst = max(worst, float(numpy.max(numpy.abs(left - right))))
    return worst


def check_equivalence_ccsdt():
    legacy = MultiConstrUCCSDT(mf4, constraint_sdt, jacobian_sdt, sdt_coordinates,
                                 constraint_diis_scheme="legacy")
    tangent = MultiConstrUCCSDT(mf4, constraint_sdt, jacobian_sdt, sdt_coordinates,
                                  constraint_diis_scheme="tangent")
    legacy.diis = tangent.diis = False
    eris = legacy.ao2mo(legacy.mo_coeff)
    _, initial = legacy.init_amps(eris)
    legacy_tamps = copied_tamps(initial)
    tangent_tamps = copied_tamps(initial)
    worst = 0.0
    for _ in range(4):
        legacy.update_amps_(legacy_tamps, eris)
        tangent.update_amps_(tangent_tamps, eris)
        for legacy_rank, tangent_rank in zip(legacy_tamps, tangent_tamps):
            for left, right in zip(legacy_rank, tangent_rank):
                worst = max(worst, float(numpy.max(numpy.abs(left - right))))
    return worst


def check_equivalence_gccsd(dense_limit):
    mol_h4 = gto.M(atom="H 0 0 0; H 0 0 1.4; H 0 0 2.8; H 0 0 4.2",
                   basis="sto-3g", spin=0, unit="Bohr", verbose=0)
    mf_ghf, _ = canted_reference(mol_h4, 2, scale=0.25, seed=1)
    nmo, nocc = mf_ghf.mo_coeff.shape[1], mol_h4.nelectron
    sz, sp = spin_matrices(mf_ghf.mo_coeff, mol_h4.intor_symmetric("int1e_ovlp"))
    engine = GCISDSpinConstraint(nmo, nocc, sz, sp, 1.0)
    legacy = MultiConstrGCCSD.from_constraint_engine(
        mf_ghf, engine, constraint_diis_scheme="legacy",
        constraint_dense_limit=dense_limit,
    )
    tangent = MultiConstrGCCSD.from_constraint_engine(
        mf_ghf, engine, constraint_diis_scheme="tangent",
        constraint_dense_limit=dense_limit,
    )
    eris = legacy.ao2mo(legacy.mo_coeff)
    t1, t2 = legacy.get_init_guess(eris)
    legacy_t1, legacy_t2 = legacy.update_amps(t1.copy(), t2.copy(), eris)
    tangent_t1, tangent_t2 = tangent.update_amps(t1.copy(), t2.copy(), eris)
    return max(
        float(numpy.max(numpy.abs(legacy_t1 - tangent_t1))),
        float(numpy.max(numpy.abs(legacy_t2 - tangent_t2))),
    )


def collinear_spinors(mf):
    coeffa, coeffb = (numpy.asarray(x) for x in mf.mo_coeff)
    nao = coeffa.shape[0]
    nocca, noccb = (int(x) for x in mf.mol.nelec)
    columns, spins = [], []
    for coeff, offset, nocc in ((coeffa, 0, nocca), (coeffb, nao, noccb)):
        for orbital in range(nocc):
            vector = numpy.zeros(2 * nao)
            vector[offset:offset + nao] = coeff[:, orbital]
            columns.append(vector)
            spins.append(0.5 if offset == 0 else -0.5)
    for coeff, offset, nocc in ((coeffa, 0, nocca), (coeffb, nao, noccb)):
        for orbital in range(nocc, coeff.shape[1]):
            vector = numpy.zeros(2 * nao)
            vector[offset:offset + nao] = coeff[:, orbital]
            columns.append(vector)
            spins.append(0.5 if offset == 0 else -0.5)
    return numpy.asarray(columns).T, numpy.asarray(spins)


def general_reference(mol, coeff):
    mf = scf.GHF(mol)
    nelectron = mol.nelectron
    mf.mo_coeff = coeff
    mf.mo_occ = numpy.r_[numpy.ones(nelectron),
                         numpy.zeros(coeff.shape[1] - nelectron)]
    density = mf.make_rdm1()
    mf.mo_energy = numpy.einsum(
        "pi,pq,qi->i", coeff.conj(), mf.get_fock(dm=density), coeff
    ).real
    mf.e_tot = float(mf.energy_tot(dm=density).real)
    mf.converged = True
    return mf


def canted_reference(mol, spin, scale=0.25, seed=0):
    unrestricted = scf.UHF(mol.copy().set(spin=spin))
    unrestricted.conv_tol = 1e-12
    unrestricted.kernel()
    assert unrestricted.converged
    collinear, spins = collinear_spinors(unrestricted)
    nmo = collinear.shape[1]
    nocc = mol.nelectron
    kappa = numpy.zeros((nmo, nmo))
    rng = numpy.random.default_rng(seed)
    kappa[:nocc, nocc:] = scale * rng.standard_normal((nocc, nmo - nocc))
    return general_reference(mol, collinear @ expm(kappa - kappa.T)), spins


ccsd_equivalence_error = check_equivalence_ccsd()
ccsdt_equivalence_error = check_equivalence_ccsdt()
gccsd_dense_equivalence_error = check_equivalence_gccsd(dense_limit=512 * 1024**2)
gccsd_matrix_free_equivalence_error = check_equivalence_gccsd(dense_limit=0.0)
assert ccsd_equivalence_error < 1e-13, ccsd_equivalence_error
assert ccsdt_equivalence_error < 1e-13, ccsdt_equivalence_error
assert gccsd_dense_equivalence_error < 1e-9, gccsd_dense_equivalence_error
assert gccsd_matrix_free_equivalence_error < 1e-7, gccsd_matrix_free_equivalence_error
print("legacy vs tangent, diis=False (telescoping identity):")
print("  UCCSD  max error = %.2e" % ccsd_equivalence_error)
print("  UCCSDT max error = %.2e" % ccsdt_equivalence_error)
print("  GCCSD  dense max error       = %.2e" % gccsd_dense_equivalence_error)
print("  GCCSD  matrix-free max error = %.2e" % gccsd_matrix_free_equivalence_error)


# --------------------------------------------------- (b) DIIS error vector shape
class _RecordingDIIS:
    def update(self, vector, xerr=None):
        self.captured_shape = xerr.shape
        return vector


def check_error_shape_ccsd():
    legacy = MultiConstrUCCSD(mf4, constraint, jacobian, coordinates,
                                constraint_diis_scheme="legacy")
    tangent = MultiConstrUCCSD(mf4, constraint, jacobian, coordinates,
                                 constraint_diis_scheme="tangent")
    legacy.diis_start_cycle = tangent.diis_start_cycle = 0
    eris = legacy.ao2mo(legacy.mo_coeff)
    t1, t2 = legacy.get_init_guess(eris)
    legacy_t1, legacy_t2 = legacy.update_amps(*copied(t1, t2), eris)
    tangent_t1, tangent_t2 = tangent.update_amps(*copied(t1, t2), eris)
    legacy_diis = _RecordingDIIS()
    tangent_diis = _RecordingDIIS()
    legacy.run_diis(legacy_t1, legacy_t2, 5, 1e-3, 1e-3, legacy_diis)
    tangent.run_diis(tangent_t1, tangent_t2, 5, 1e-3, 1e-3, tangent_diis)
    vector_size = legacy.amplitudes_to_vector(legacy_t1, legacy_t2).size
    nconstraint = constraint(legacy_t1, legacy_t2).size
    assert legacy_diis.captured_shape == (vector_size + nconstraint,), (
        legacy_diis.captured_shape, vector_size, nconstraint
    )
    assert tangent_diis.captured_shape == (vector_size,), (
        tangent_diis.captured_shape, vector_size
    )
    return legacy_diis.captured_shape, tangent_diis.captured_shape


def check_error_shape_ccsdt():
    legacy = MultiConstrUCCSDT(mf4, constraint_sdt, jacobian_sdt, sdt_coordinates,
                                 constraint_diis_scheme="legacy")
    tangent = MultiConstrUCCSDT(mf4, constraint_sdt, jacobian_sdt, sdt_coordinates,
                                  constraint_diis_scheme="tangent")
    legacy.diis_start_cycle = tangent.diis_start_cycle = 0
    # Normally built once inside .ccsdt()'s kernel setup; update_amps_ is
    # called directly here (bypassing .ccsdt()), and packing the DIIS-bound
    # residual vector needs the map built already (unlike unpacking, packing
    # has no lazy-build fallback).
    legacy.unique_tamps_map = legacy.build_unique_tamps_map()
    tangent.unique_tamps_map = tangent.build_unique_tamps_map()
    eris = legacy.ao2mo(legacy.mo_coeff)
    _, initial = legacy.init_amps(eris)
    legacy_tamps = copied_tamps(initial)
    tangent_tamps = copied_tamps(initial)
    legacy.update_amps_(legacy_tamps, eris)
    tangent.update_amps_(tangent_tamps, eris)
    legacy_diis = _RecordingDIIS()
    tangent_diis = _RecordingDIIS()
    legacy.run_diis(legacy_tamps, 5, 1e-3, 1e-3, legacy_diis)
    tangent.run_diis(tangent_tamps, 5, 1e-3, 1e-3, tangent_diis)
    vector_size = legacy.amplitudes_to_vector(legacy_tamps).size
    nconstraint = constraint_sdt(*legacy.public_amplitudes(legacy_tamps)).size
    assert legacy_diis.captured_shape == (vector_size + nconstraint,), (
        legacy_diis.captured_shape, vector_size, nconstraint
    )
    assert tangent_diis.captured_shape == (vector_size,), (
        tangent_diis.captured_shape, vector_size
    )
    return legacy_diis.captured_shape, tangent_diis.captured_shape


ccsd_shapes = check_error_shape_ccsd()
ccsdt_shapes = check_error_shape_ccsdt()
print("DIIS xerr shapes (legacy, tangent):")
print("  UCCSD  %s" % (ccsd_shapes,))
print("  UCCSDT %s" % (ccsdt_shapes,))


# --------------------------------------------- (c) convergence with DIIS engaged
def configured(cc):
    cc.verbose = 0
    cc.max_cycle = 60
    cc.conv_tol = 1e-10
    cc.conv_tol_normt = 1e-8
    cc.diis_space = 8
    cc.diis_start_cycle = 2
    return cc


engine4 = ClosedCISDSpinConstraint.from_cc(uccsd.UCCSD(mf4))
legacy_converged = configured(MultiConstrUCCSD.from_constraint_engine(
    mf4, engine4, constraint_diis_scheme="legacy"
)).run()
tangent_converged = configured(MultiConstrUCCSD.from_constraint_engine(
    mf4, engine4, constraint_diis_scheme="tangent"
)).run()
assert legacy_converged.converged and tangent_converged.converged
assert numpy.max(numpy.abs(
    legacy_converged.constraint_residual())) < legacy_converged.constraint_tol
assert numpy.max(numpy.abs(
    tangent_converged.constraint_residual())) < tangent_converged.constraint_tol
energy_agreement = abs(legacy_converged.e_tot - tangent_converged.e_tot)
assert energy_agreement < 1e-8, energy_agreement
# The mathematical property that must hold at any converged constrained
# fixed point: r_par = residual_selected + response(eta_LS) vanishes there
# (residual_selected = -response(eta*) once the amplitude equations are
# satisfied by *some* multiplier).  A wrong metric power or sign in eta_LS's
# construction leaves r_par at an O(residual_selected) plateau instead --
# exactly the class of bug a tiny-system energy-agreement check alone does
# not discriminate against (it passed on both bad formulas tried during
# development; only a real ~1e6-coordinate production case exposed the
# stall this catches directly).
tangent_residual_norm = float(numpy.max(numpy.abs(
    tangent_converged._constraint_tangent_residual)))
assert tangent_residual_norm < 1e-6, tangent_residual_norm
print("convergence with DIIS engaged: legacy E = %.10f, tangent E = %.10f, "
      "|dE| = %.2e, max|r_par| at convergence = %.2e"
      % (legacy_converged.e_tot, tangent_converged.e_tot,
         energy_agreement, tangent_residual_norm))


# --------------------------------------------------------- (d) incompatibility
for bad in ("nonlinear",):
    try:
        MultiConstrUCCSD(
            mf4, constraint, jacobian, coordinates,
            constraint_multiplier_solver=bad, constraint_diis_scheme="tangent",
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "constraint_diis_scheme='tangent' with a nonlinear multiplier "
            "solver was accepted (UCCSD)"
        )

    try:
        MultiConstrUCCSDT(
            mf4, constraint_sdt, jacobian_sdt, sdt_coordinates,
            constraint_multiplier_solver=bad, constraint_diis_scheme="tangent",
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "constraint_diis_scheme='tangent' with a nonlinear multiplier "
            "solver was accepted (UCCSDT)"
        )

for bad_scheme in ("bogus", "", None):
    try:
        MultiConstrUCCSD(
            mf4, constraint, jacobian, coordinates,
            constraint_diis_scheme=bad_scheme,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("invalid constraint_diis_scheme was accepted (UCCSD)")

    try:
        MultiConstrUCCSDT(
            mf4, constraint_sdt, jacobian_sdt, sdt_coordinates,
            constraint_diis_scheme=bad_scheme,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("invalid constraint_diis_scheme was accepted (UCCSDT)")

mol_g = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
mf_g = scf.GHF(mol_g).run(verbose=0)


def dummy_constraint(t1, t2):
    return numpy.zeros(1)


def dummy_jacobian(t1, t2):
    return numpy.zeros((1, t1.size + t2.size))


def dummy_jvp(t1, t2, vector):
    return numpy.zeros(1)


def dummy_vjp(t1, t2, weights):
    return numpy.zeros(t1.size + t2.size)


def dummy_row_norms(t1, t2):
    return numpy.ones(1)


def dummy_gram_diagonal(t1, t2, metric):
    return numpy.ones(1)


for bad_scheme in ("bogus", "", None):
    try:
        MultiConstrGCCSD(
            mf_g, dummy_constraint, dummy_jacobian, list(range(1)),
            constraint_jvp=dummy_jvp, constraint_vjp=dummy_vjp,
            constraint_jacobian_row_norms=dummy_row_norms,
            constraint_weighted_gram_diagonal=dummy_gram_diagonal,
            constraint_diis_scheme=bad_scheme,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("invalid constraint_diis_scheme was accepted (GCCSD)")

print("incompatibility guards: rejected as expected")
print("all constraint DIIS tangent-scheme checks passed")

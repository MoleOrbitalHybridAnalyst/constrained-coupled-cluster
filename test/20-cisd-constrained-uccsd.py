import numpy
import pyscf
from pyscf import mcscf, scf
from pyscf.cc import uccsd

from ccc.cluster_decomposition import cas_amplitudes
from ccc import MultiConstrUCCSD
from ccc.cisd_spin_constraint import ClosedCISDSpinConstraint


def spin_constraint(mf, frozen=None):
    """Generate the FCI-closed SD spin equations for a reference.

    `from_cc` reads the correlated orbital spaces off a probe UCCSD object, so
    the probe must carry the same `frozen` specification as the constrained
    calculation it configures.
    """
    probe = uccsd.UCCSD(mf, frozen=frozen)
    return ClosedCISDSpinConstraint.from_cc(probe)


def constrained_uccsd(mf, frozen=None):
    """Wire the spin equations into MultiConstrUCCSD.

    Same wiring the n2 and hubbard study drivers use; there is no library class
    that pairs the two, since it would be nothing but these few lines.
    """
    engine = spin_constraint(mf, frozen=frozen)
    cc = MultiConstrUCCSD(
        mf,
        constraint=engine.constraint,
        constraint_jacobian=engine.jacobian,
        constraint_coordinates=engine.coordinates,
        frozen=frozen,
    )
    return cc, engine


def configure(cc):
    cc.verbose = 0
    cc.max_cycle = 200
    cc.conv_tol = 1e-10
    cc.conv_tol_normt = 1e-8
    cc.diis_space = 30
    cc.diis_start_cycle = 5
    return cc


def set_reference(mol, mo_coeff, mo_occ):
    mf = scf.UHF(mol)
    mf.mo_coeff = tuple(numpy.asarray(x).copy() for x in mo_coeff)
    mf.mo_occ = tuple(numpy.asarray(x, dtype=float) for x in mo_occ)
    density = mf.make_rdm1(mf.mo_coeff, mf.mo_occ)
    fock = mf.get_fock(dm=density)
    mf.mo_energy = tuple(
        numpy.einsum("pi,pq,qi->i", coeff, matrix, coeff)
        for coeff, matrix in zip(mf.mo_coeff, fock)
    )
    mf.e_tot = mf.energy_tot(dm=density)
    mf.converged = True
    return mf


# Closed-shell common orbitals reduce to ordinary UCCSD.
mol = pyscf.M(
    atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", spin=0, verbose=0
)
mf = mol.UHF().run(verbose=0)
regular = configure(uccsd.UCCSD(mf)).run()
constrained, closed_engine = constrained_uccsd(mf)
constrained = configure(constrained).run()
assert regular.converged and constrained.converged
assert abs(regular.e_tot - constrained.e_tot) < 1e-10
assert numpy.max(numpy.abs(constrained.constraint_residual())) < 1e-10

# Bound CISD engines are detected automatically and use sparse products.
assert constrained.constraint_ncall["value"] > 0
assert constrained.constraint_ncall["jvp"] > 0
assert constrained.constraint_ncall["vjp"] > 0
assert constrained.constraint_ncall["jacobian"] == 0
assert min(constrained.constraint_time.values()) >= 0.0
assert closed_engine.build_time > 0.0


# Localized open-shell H2 has a quasi-restricted |ab> reference with nonzero
# overlap with the exact singlet.  UCCSD is FCI for two electrons, so this is
# a direct numerical test of the generated nonlinear constraint.
stretched = pyscf.M(
    atom="H 0 0 0; H 0 0 2.0", basis="sto-3g", spin=0, verbose=0
)
rhf = stretched.RHF().run(verbose=0)
rotation = numpy.asarray([[1.0, 1.0], [1.0, -1.0]]) / numpy.sqrt(2.0)
localized = rhf.mo_coeff @ rotation
open_shell = set_reference(
    stretched,
    (localized, localized[:, ::-1]),
    (numpy.asarray([1.0, 0.0]), numpy.asarray([1.0, 0.0])),
)
cas = mcscf.UCASCI(open_shell, 2, (1, 1))
cas.fcisolver.nroots = 2
energies, active_energies, ci = cas.kernel()
cas.ci = ci[0]
cas.e_tot = energies[0]
cas.e_cas = active_energies[0]
t1, t2 = cas_amplitudes(cas)

singlet = configure(constrained_uccsd(open_shell)[0])
eris = singlet.ao2mo(singlet.mo_coeff)
perturbed_t2 = tuple(block.copy() for block in t2)
perturbed_t2[1][0, 0, 0, 0] += 1e-3
initial_residual = float(numpy.max(numpy.abs(
    singlet.constraint_residual(t1, perturbed_t2)
)))
assert initial_residual > 1e-5
singlet.kernel(t1=t1, t2=perturbed_t2, eris=eris)
assert singlet.converged
assert abs(singlet.e_tot - energies[0]) < 1e-8
assert numpy.max(numpy.abs(singlet.constraint_residual())) < 1e-9


# Matched, doubly occupied frozen orbitals are accepted.  Asymmetric or
# open-shell freezing and genuinely different alpha/beta orbitals are not.
# These are all rejected by from_cc, which reads the orbital spaces.
lih = pyscf.M(
    atom="Li 0 0 0; H 0 0 1.6", basis="sto-3g", spin=0, verbose=0
)
lih_rhf = lih.RHF().run(verbose=0)
lih_mf = set_reference(
    lih,
    (lih_rhf.mo_coeff, lih_rhf.mo_coeff),
    (lih_rhf.mo_occ / 2, lih_rhf.mo_occ / 2),
)
frozen_cc, frozen_engine = constrained_uccsd(lih_mf, frozen=1)
assert frozen_engine.nconstraint > 0

try:
    spin_constraint(lih_mf, frozen=([0], []))
except ValueError as error:
    assert "frozen" in str(error)
else:
    raise AssertionError("asymmetric frozen orbitals were accepted")

try:
    spin_constraint(open_shell, frozen=([0], [1]))
except ValueError as error:
    assert "open shell" in str(error)
else:
    raise AssertionError("an open-shell frozen orbital was accepted")

angle = 0.1
noncommon_rotation = numpy.asarray([
    [numpy.cos(angle), -numpy.sin(angle)],
    [numpy.sin(angle), numpy.cos(angle)],
])
noncommon = set_reference(
    mol,
    (mf.mo_coeff[0], mf.mo_coeff[1] @ noncommon_rotation),
    mf.mo_occ,
)
try:
    spin_constraint(noncommon)
except ValueError as error:
    assert "signed permutation" in str(error)
else:
    raise AssertionError("non-common alpha/beta orbitals were accepted")

print("E(closed-shell UCCSD) = %.15f" % regular.e_tot)
print("E(open-shell constrained UCCSD) = %.15f" % singlet.e_tot)
print("E(open-shell FCI) = %.15f" % energies[0])
print("initial, final max|C| = %.3e %.3e" % (
    initial_residual,
    numpy.max(numpy.abs(singlet.constraint_residual())),
))
# Milliseconds here because these systems are tiny; the library reports seconds,
# matching pyscf's timer convention.
print("constraint build = %.2f ms | C(t) %d calls %.2f ms | "
      "Jacobian %d calls %.2f ms"
      % (closed_engine.build_time * 1e3,
         constrained.constraint_ncall["value"],
         constrained.constraint_time["value"] * 1e3,
         constrained.constraint_ncall["jacobian"],
         constrained.constraint_time["jacobian"] * 1e3))

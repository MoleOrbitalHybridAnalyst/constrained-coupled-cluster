"""True lambda equations for CISD-constrained UCCSD.

Checks, in order: the engine's analytic Hessian-vector product against
finite differences of its Jacobian; the guard that solve_lambda demands the
Hessian-vector product; exactness on stretched H2 (where the constrained
solution is the FCI singlet and the response density must match the FCI
density); and the real thing -- on a 4-site Hubbard chain with genuinely
active constraints, the response density reproduces dE/dh from finite
differences of the re-converged constrained energy.
"""

import numpy
import pyscf
from pyscf import ao2mo, gto, scf
from pyscf.fci import direct_uhf

from ccc.cluster_decomposition import cas_amplitudes
from ccc import MultiConstrUCCSD
from ccc.cisd_spin_constraint import ClosedCISDSpinConstraint


def antisymmetrize(t2):
    t2 = t2 - t2.transpose(1, 0, 2, 3)
    return t2 - t2.transpose(0, 1, 3, 2)


# --- 1. Hessian-vector product against finite differences of the Jacobian.
# The direction is supplied as full antisymmetric tensors; gathering it at
# the engine coordinates gives the coordinate vector the product contracts.
engine = ClosedCISDSpinConstraint(4, (2, 2))
rng = numpy.random.default_rng(11)
t1 = tuple(0.1 * rng.standard_normal((2, 2)) for _ in range(2))
t2 = (antisymmetrize(0.1 * rng.standard_normal((2, 2, 2, 2))),
      0.1 * rng.standard_normal((2, 2, 2, 2)),
      antisymmetrize(0.1 * rng.standard_normal((2, 2, 2, 2))))
d1 = tuple(rng.standard_normal((2, 2)) for _ in range(2))
d2 = (antisymmetrize(rng.standard_normal((2, 2, 2, 2))),
      rng.standard_normal((2, 2, 2, 2)),
      antisymmetrize(rng.standard_normal((2, 2, 2, 2))))
weights = rng.standard_normal(engine.nconstraint)
direction = engine.amplitude_values(d1, d2)

step = 1e-6
plus = numpy.array(engine.jacobian(
    tuple(a + step * b for a, b in zip(t1, d1)),
    tuple(a + step * b for a, b in zip(t2, d2)),
))
minus = numpy.array(engine.jacobian(
    tuple(a - step * b for a, b in zip(t1, d1)),
    tuple(a - step * b for a, b in zip(t2, d2)),
))
finite_difference = weights @ (plus - minus) / (2.0 * step)
analytic = engine.hessian_vector(t1, t2, weights, direction)
assert numpy.max(numpy.abs(finite_difference - analytic)) < 1e-8
print("Hessian-vector product matches the differentiated Jacobian")


# --- helpers shared by the molecular and Hubbard cases -------------------

def configure(cc):
    cc.verbose = 0
    cc.max_cycle = 500
    cc.conv_tol = 1e-13
    cc.conv_tol_normt = 1e-10
    cc.diis_space = 20
    return cc


def rdm_energy(cc, dm1, dm2):
    """Total energy from the response densities and the MO integrals."""
    mo_a, mo_b = cc.mo_coeff
    hcore = cc._scf.get_hcore()
    eri = cc._scf._eri if cc._scf._eri is not None else cc.mol
    h1a = mo_a.T @ hcore @ mo_a
    h1b = mo_b.T @ hcore @ mo_b
    nmo = mo_a.shape[1]
    eri_aa = ao2mo.restore(1, ao2mo.full(eri, mo_a), nmo)
    eri_ab = ao2mo.restore(
        1, ao2mo.general(eri, (mo_a, mo_a, mo_b, mo_b)), nmo)
    eri_bb = ao2mo.restore(1, ao2mo.full(eri, mo_b), nmo)
    dm1a, dm1b = dm1
    dm2aa, dm2ab, dm2bb = dm2
    energy = numpy.einsum("pq,qp", h1a, dm1a)
    energy += numpy.einsum("pq,qp", h1b, dm1b)
    energy += 0.5 * numpy.einsum("pqrs,pqrs", eri_aa, dm2aa)
    energy += numpy.einsum("pqrs,pqrs", eri_ab, dm2ab)
    energy += 0.5 * numpy.einsum("pqrs,pqrs", eri_bb, dm2bb)
    return float(energy + cc._scf.energy_nuc())


def gradient_from_dm1(cc, dm1, perturbation):
    """dE/d(epsilon) for h1 -> h1 + epsilon*V from the response density."""
    mo_a, mo_b = cc.mo_coeff
    va = mo_a.T @ perturbation @ mo_a
    vb = mo_b.T @ perturbation @ mo_b
    return float(numpy.einsum("pq,qp", va, dm1[0])
                 + numpy.einsum("pq,qp", vb, dm1[1]))


# --- 2 & 3. Stretched H2: guard, then exactness against FCI --------------
# Same localized |ab> reference as test/20.  UCCSD is FCI for two electrons,
# so the constrained solution is the exact singlet, and the true-lambda
# response density must reproduce the FCI density.

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


stretched = pyscf.M(atom="H 0 0 0; H 0 0 2.0", basis="sto-3g", spin=0,
                    verbose=0)
rhf = stretched.RHF().run(verbose=0)
rotation = numpy.asarray([[1.0, 1.0], [1.0, -1.0]]) / numpy.sqrt(2.0)
localized = rhf.mo_coeff @ rotation
open_shell = set_reference(
    stretched,
    (localized, localized[:, ::-1]),
    (numpy.asarray([1.0, 0.0]), numpy.asarray([1.0, 0.0])),
)
h2_engine = ClosedCISDSpinConstraint.from_cc(
    pyscf.cc.uccsd.UCCSD(open_shell))

without_hvp = configure(MultiConstrUCCSD(
    open_shell,
    lambda t1, t2: h2_engine.constraint(t1, t2),
    lambda t1, t2: h2_engine.jacobian(t1, t2),
    h2_engine.coordinates,
))
try:
    without_hvp.solve_lambda()
except NotImplementedError:
    pass
else:
    raise AssertionError(
        "solve_lambda must demand the constraint Hessian-vector product")
print("solve_lambda raises without the Hessian-vector product")

h2cc = configure(MultiConstrUCCSD(
    open_shell, h2_engine.constraint, h2_engine.jacobian,
    h2_engine.coordinates, constraint_max_step=0.1,
    constraint_hessian_vector=h2_engine.hessian_vector,
))
cas = pyscf.mcscf.UCASCI(open_shell, 2, (1, 1))
cas.fcisolver.nroots = 2
energies, active_energies, ci = cas.kernel()
cas.ci, cas.e_tot, cas.e_cas = ci[0], energies[0], active_energies[0]
t1, t2 = cas_amplitudes(cas)
perturbed_t2 = tuple(block.copy() for block in t2)
perturbed_t2[1][0, 0, 0, 0] += 1e-3
h2cc.kernel(t1=t1, t2=perturbed_t2, eris=h2cc.ao2mo(h2cc.mo_coeff))
assert h2cc.converged
assert abs(h2cc.e_tot - energies[0]) < 1e-8

h2cc.solve_lambda()
assert h2cc.converged_lambda
assert h2cc.constraint_lambda_orthogonality < 1e-10
dm1 = h2cc.make_rdm1()
dm2 = h2cc.make_rdm2()
assert abs(numpy.trace(dm1[0]) + numpy.trace(dm1[1]) - 2.0) < 1e-10
# The spin-summed (charge) density matches FCI: for spin-free perturbations
# the constrained solution follows the exact singlet, whose cluster
# amplitudes satisfy every S+ condition on their own.  The individual spin
# components legitimately differ -- they answer spin-dependent
# perturbations, under which the constrained response keeps enforcing the
# spin conditions while the exact state breaks them (kappa is nonzero for
# the same reason).  Each spin block lives in its own MO basis here, so the
# sum is formed in the common AO basis.
fci_dm1 = direct_uhf.make_rdm1s(cas.ci, 2, (1, 1))
mo_a, mo_b = h2cc.mo_coeff
charge_cc = mo_a @ dm1[0] @ mo_a.T + mo_b @ dm1[1] @ mo_b.T
charge_fci = mo_a @ fci_dm1[0] @ mo_a.T + mo_b @ fci_dm1[1] @ mo_b.T
assert numpy.max(numpy.abs(charge_cc - charge_fci)) < 1e-9
assert abs(rdm_energy(h2cc, dm1, dm2) - h2cc.e_tot) < 1e-9
print("H2: charge response density matches FCI, E(RDM) matches E(CC)")


# --- 4. Active constraints: dE/dh against finite differences -------------
# Half-filled 4-site Hubbard chain with a Neel |abab> reference.  The
# constraint multipliers are genuinely nonzero here, so this exercises the
# Hessian term and the projection, not just the ordinary lambda equations.

NSITE, U = 4, 4.0


def hubbard_reference(h1):
    mol = gto.Mole()
    mol.nao = NSITE
    mol.nelectron = NSITE
    mol.spin = 0
    mol.verbose = 0
    mol._built = True
    mol.incore_anyway = True
    mol.energy_nuc = lambda *args: 0.0
    mol.intor = lambda name, *args, **kwargs: (
        numpy.eye(NSITE) if name == "int1e_ovlp" else None
    )
    eri = numpy.zeros((NSITE,) * 4)
    for site in range(NSITE):
        eri[site, site, site, site] = U
    # Neel |abab>: alpha on even sites, beta on odd, occupied first
    alpha_perm = numpy.asarray([0, 2, 1, 3])
    beta_perm = numpy.asarray([1, 3, 0, 2])
    site_orbitals = numpy.eye(NSITE)
    mf = scf.UHF(mol)
    mf.get_hcore = lambda *args: h1
    mf.get_ovlp = lambda *args: numpy.eye(NSITE)
    mf.energy_nuc = lambda *args: 0.0
    mf._eri = ao2mo.restore(8, eri, NSITE)
    mf.mo_coeff = (site_orbitals[:, alpha_perm], site_orbitals[:, beta_perm])
    mf.mo_occ = (numpy.asarray([1.0, 1.0, 0.0, 0.0]),) * 2
    density = mf.make_rdm1(mf.mo_coeff, mf.mo_occ)
    fock = mf.get_fock(dm=density)
    mf.mo_energy = tuple(
        numpy.einsum("pi,pq,qi->i", coeff, matrix, coeff)
        for coeff, matrix in zip(mf.mo_coeff, fock)
    )
    mf.e_tot = mf.energy_tot(dm=density)
    mf.converged = True
    return mf, alpha_perm, beta_perm


def hopping():
    h1 = numpy.zeros((NSITE, NSITE))
    for site in range(NSITE - 1):
        h1[site, site + 1] = h1[site + 1, site] = -1.0
    return h1


def converged_hubbard(h1, seed_t1=None, seed_t2=None, matrix_free=False):
    mf, alpha_perm, beta_perm = hubbard_reference(h1)
    engine = ClosedCISDSpinConstraint(
        NSITE, (2, 2), alpha_permutation=alpha_perm,
        beta_permutation=beta_perm,
    )
    if matrix_free:
        cc = configure(MultiConstrUCCSD.from_constraint_engine(
            mf, engine, constraint_max_step=0.1,
        ))
    else:
        cc = configure(MultiConstrUCCSD(
            mf, engine.constraint, engine.jacobian, engine.coordinates,
            constraint_max_step=0.1,
            constraint_hessian_vector=engine.hessian_vector,
        ))
    cc.kernel(t1=seed_t1, t2=seed_t2, eris=cc.ao2mo(cc.mo_coeff))
    assert cc.converged
    return cc


hub = converged_hubbard(hopping())
zeta = numpy.max(numpy.abs(hub.constraint_multipliers))
assert zeta > 1e-4, "constraints must be active for this test to mean anything"

hub.solve_lambda()
assert hub.converged_lambda
assert hub.constraint_lambda_orthogonality < 1e-10
dm1 = hub.make_rdm1()
dm2 = hub.make_rdm2()
assert abs(numpy.trace(dm1[0]) + numpy.trace(dm1[1]) - NSITE) < 1e-10
assert abs(rdm_energy(hub, dm1, dm2) - hub.e_tot) < 1e-9

rng = numpy.random.default_rng(29)
perturbation = rng.standard_normal((NSITE, NSITE))
perturbation = 0.5 * (perturbation + perturbation.T)
epsilon = 1e-5
above = converged_hubbard(hopping() + epsilon * perturbation,
                          seed_t1=hub.t1, seed_t2=hub.t2)
below = converged_hubbard(hopping() - epsilon * perturbation,
                          seed_t1=hub.t1, seed_t2=hub.t2)
finite_difference = (above.e_tot - below.e_tot) / (2.0 * epsilon)
adjoint = gradient_from_dm1(hub, dm1, perturbation)
print("Hubbard: max|zeta| = %.4f  kappa = %s" % (
    zeta, numpy.array2string(hub.constraint_kappa, precision=4)))
print("dE/dh: finite difference = %.12f  adjoint = %.12f  error = %.2e"
      % (finite_difference, adjoint, abs(finite_difference - adjoint)))
assert abs(finite_difference - adjoint) < 5e-6

# The engine-backed constructor must give the same true lambda through JVP/VJP
# products alone.  Seed from the dense solution so this adds one backend check,
# not a second test of the amplitude convergence basin.
matrix_free = converged_hubbard(
    hopping(), seed_t1=hub.t1, seed_t2=hub.t2, matrix_free=True,
)
assert matrix_free._matrix_free_constraints
assert matrix_free.constraint_ncall["jacobian"] == 0
matrix_free.solve_lambda()
assert matrix_free.converged_lambda
assert matrix_free.constraint_ncall["jacobian"] == 0
assert matrix_free.constraint_lambda_orthogonality < 1e-10
lambda_error = max(
    numpy.max(numpy.abs(sparse - dense))
    for sparse, dense in zip(matrix_free.l1 + matrix_free.l2,
                             hub.l1 + hub.l2)
)
assert lambda_error < 1e-9
matrix_free_dm1 = matrix_free.make_rdm1()
matrix_free_dm2 = matrix_free.make_rdm2()
matrix_free_rdm_error = abs(
    rdm_energy(matrix_free, matrix_free_dm1, matrix_free_dm2)
    - matrix_free.e_tot
)
matrix_free_gradient = gradient_from_dm1(
    matrix_free, matrix_free_dm1, perturbation
)
assert matrix_free_rdm_error < 1e-9
assert abs(matrix_free_gradient - finite_difference) < 5e-6
print("matrix-free lambda: dense error = %.2e  orthogonality = %.2e  "
      "RDM energy error = %.2e"
      % (lambda_error, matrix_free.constraint_lambda_orthogonality,
         matrix_free_rdm_error))

# negative controls, printed for the record: the lambda=t shortcut and a
# lambda solve with the Hessian term switched off both miss the gradient
lambda_t = gradient_from_dm1(
    hub, hub.make_rdm1(hub.t1, hub.t2, hub.t1, hub.t2), perturbation)
no_hessian = converged_hubbard(hopping())
no_hessian.constraint_hessian_vector = (
    lambda t1, t2, weights, vector: numpy.zeros_like(vector))
no_hessian.solve_lambda()
no_hessian_gradient = gradient_from_dm1(
    no_hessian, no_hessian.make_rdm1(), perturbation)
print("negative controls: lambda=t error = %.2e, "
      "Hessian-term-dropped error = %.2e"
      % (abs(lambda_t - finite_difference),
         abs(no_hessian_gradient - finite_difference)))

print("All true-lambda checks passed")

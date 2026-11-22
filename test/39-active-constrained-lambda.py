"""True lambda equations for a dominant-sector spin constraint."""

import numpy
from pyscf import ao2mo, gto, scf

from ccc import MultiConstrUCCSD
from ccc.active_spin_constraint import DominantSectorSpinConstraint


NSITE, U = 4, 4.0


def configure(cc):
    cc.verbose = 0
    cc.max_cycle = 500
    cc.conv_tol = 1e-13
    cc.conv_tol_normt = 1e-10
    cc.diis_space = 20
    return cc


def hopping():
    h1 = numpy.zeros((NSITE, NSITE))
    for site in range(NSITE - 1):
        h1[site, site + 1] = h1[site + 1, site] = -1.0
    return h1


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

    alpha_permutation = numpy.asarray([0, 2, 1, 3])
    beta_permutation = numpy.asarray([1, 3, 0, 2])
    mf = scf.UHF(mol)
    mf.get_hcore = lambda *args: h1
    mf.get_ovlp = lambda *args: numpy.eye(NSITE)
    mf.energy_nuc = lambda *args: 0.0
    mf._eri = ao2mo.restore(8, eri, NSITE)
    mf.mo_coeff = (
        numpy.eye(NSITE)[:, alpha_permutation],
        numpy.eye(NSITE)[:, beta_permutation],
    )
    mf.mo_occ = (numpy.asarray([1.0, 1.0, 0.0, 0.0]),) * 2
    density = mf.make_rdm1(mf.mo_coeff, mf.mo_occ)
    fock = mf.get_fock(dm=density)
    mf.mo_energy = tuple(
        numpy.einsum("pi,pq,qi->i", coeff, matrix, coeff)
        for coeff, matrix in zip(mf.mo_coeff, fock)
    )
    mf.e_tot = mf.energy_tot(dm=density)
    mf.converged = True
    return mf, alpha_permutation, beta_permutation


def constrained_hubbard(h1, seed=None, matrix_free=True):
    mf, alpha_permutation, beta_permutation = hubbard_reference(h1)
    engine = DominantSectorSpinConstraint(
        NSITE, (2, 2), "abab",
        alpha_permutation=alpha_permutation,
        beta_permutation=beta_permutation,
    )
    if matrix_free:
        discovered = MultiConstrUCCSD(
            mf, engine.constraint, engine.jacobian, engine.coordinates,
            constraint_max_step=0.1,
        )
        assert discovered._matrix_free_constraints
        assert discovered.constraint_hessian_vector.__self__ is engine
        cc = MultiConstrUCCSD.from_constraint_engine(
            mf, engine, constraint_max_step=0.1,
        )
        assert cc._matrix_free_constraints
        assert cc.constraint_hessian_vector.__self__ is engine
    else:
        # Plain wrappers prevent bound-engine derivative discovery and exercise
        # the independent dense-Jacobian path.
        cc = MultiConstrUCCSD(
            mf,
            lambda t1, t2: engine.constraint(t1, t2),
            lambda t1, t2: engine.jacobian(t1, t2),
            engine.coordinates,
            constraint_max_step=0.1,
            constraint_hessian_vector=engine.hessian_vector,
        )
    cc = configure(cc)
    eris = cc.ao2mo(cc.mo_coeff)
    t1, t2 = (None, None) if seed is None else seed
    cc.kernel(t1=t1, t2=t2, eris=eris)
    assert cc.converged
    return cc


def rdm_energy(cc, dm1, dm2):
    mo_a, mo_b = cc.mo_coeff
    hcore = cc._scf.get_hcore()
    eri = cc._scf._eri
    h1a = mo_a.T @ hcore @ mo_a
    h1b = mo_b.T @ hcore @ mo_b
    eri_aa = ao2mo.restore(1, ao2mo.full(eri, mo_a), NSITE)
    eri_ab = ao2mo.restore(
        1, ao2mo.general(eri, (mo_a, mo_a, mo_b, mo_b)), NSITE
    )
    eri_bb = ao2mo.restore(1, ao2mo.full(eri, mo_b), NSITE)
    dm1a, dm1b = dm1
    dm2aa, dm2ab, dm2bb = dm2
    energy = numpy.einsum("pq,qp", h1a, dm1a)
    energy += numpy.einsum("pq,qp", h1b, dm1b)
    energy += 0.5 * numpy.einsum("pqrs,pqrs", eri_aa, dm2aa)
    energy += numpy.einsum("pqrs,pqrs", eri_ab, dm2ab)
    energy += 0.5 * numpy.einsum("pqrs,pqrs", eri_bb, dm2bb)
    return float(energy)


def one_body_response(cc, dm1, perturbation):
    mo_a, mo_b = cc.mo_coeff
    va = mo_a.T @ perturbation @ mo_a
    vb = mo_b.T @ perturbation @ mo_b
    return float(numpy.einsum("pq,qp", va, dm1[0])
                 + numpy.einsum("pq,qp", vb, dm1[1]))


dense = constrained_hubbard(hopping(), matrix_free=False)
matrix_free = constrained_hubbard(
    hopping(), seed=(dense.t1, dense.t2), matrix_free=True
)
assert matrix_free.constraint_ncall["jacobian"] == 0
assert numpy.max(numpy.abs(matrix_free.constraint_multipliers)) > 1e-4
assert abs(matrix_free.e_tot - dense.e_tot) < 1e-10

dense.solve_lambda()
matrix_free.solve_lambda()
assert dense.converged_lambda and matrix_free.converged_lambda
assert matrix_free.constraint_ncall["jacobian"] == 0
assert matrix_free.constraint_lambda_orthogonality < 1e-10
lambda_error = max(
    numpy.max(numpy.abs(x - y))
    for x, y in zip(matrix_free.l1 + matrix_free.l2, dense.l1 + dense.l2)
)
assert lambda_error < 1e-9

dm1 = matrix_free.make_rdm1()
dm2 = matrix_free.make_rdm2()
rdm_error = abs(rdm_energy(matrix_free, dm1, dm2) - matrix_free.e_tot)
assert rdm_error < 1e-9

rng = numpy.random.default_rng(39)
perturbation = rng.standard_normal((NSITE, NSITE))
perturbation = 0.5 * (perturbation + perturbation.T)
epsilon = 1e-5
seed = (matrix_free.t1, matrix_free.t2)
above = constrained_hubbard(
    hopping() + epsilon * perturbation, seed=seed, matrix_free=True
)
below = constrained_hubbard(
    hopping() - epsilon * perturbation, seed=seed, matrix_free=True
)
finite_difference = (above.e_tot - below.e_tot) / (2 * epsilon)
response = one_body_response(matrix_free, dm1, perturbation)
response_error = abs(response - finite_difference)
assert response_error < 5e-6

print("dense/matrix-free lambda error = %.3e" % lambda_error)
print("constraint orthogonality = %.3e" % (
    matrix_free.constraint_lambda_orthogonality
))
print("RDM energy error = %.3e" % rdm_error)
print("response finite-difference error = %.3e" % response_error)

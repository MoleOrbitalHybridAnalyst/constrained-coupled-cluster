import numpy
import pyscf
from pyscf import mcscf
from pyscf.fci import cistring

from ccc.cluster_decomposition import cas_amplitudes
from ccc.active_spin_constraint import (
    DominantSectorSpinConstraint,
    _reference_permutation,
    occupations_from_string,
)


def h_chain(natom, spin):
    mol = pyscf.M(
        atom=";".join("H 0 0 %.1f" % (1.4 * i) for i in range(natom)),
        basis="sto-3g",
        spin=spin,
        verbose=0,
    )
    return mol.UHF().run(verbose=0)


# The exact exponential must invert the existing CI -> T1/T2 conversion through
# the full unrestricted CISD sector, including same-spin phases.
mf7 = h_chain(7, 1)
mc = mcscf.UCASCI(mf7, 7, (4, 3))
rng = numpy.random.default_rng(4)
ci = rng.normal(scale=0.03, size=(35, 35))
ci[0, 0] = 1.0
mc.ci = ci
mc.e_tot = mc.e_cas = 0.0
t1, t2 = cas_amplitudes(mc)
engine = DominantSectorSpinConstraint(
    7, (4, 3), "222a000", alpha_permutation=numpy.arange(7),
    beta_permutation=numpy.arange(7),
)
wavefunction = engine.wavefunction(t1, t2)
reference_alpha = set(range(4))
reference_beta = set(range(3))
maximum = 0.0
for ia in range(35):
    alpha_string = cistring.addr2str(7, 4, ia)
    alpha = {i for i in range(7) if alpha_string & (1 << i)}
    alpha_rank = len(reference_alpha - alpha)
    for ib in range(35):
        beta_string = cistring.addr2str(7, 3, ib)
        beta = {i for i in range(7) if beta_string & (1 << i)}
        if alpha_rank + len(reference_beta - beta) <= 2:
            maximum = max(maximum, abs(wavefunction[ia, ib] - ci[ia, ib]))
assert maximum < 1e-12


def random_amplitudes(engine, seed):
    rng = numpy.random.default_rng(seed)
    na, nb = engine.nelec
    nva, nvb = engine.norb - na, engine.norb - nb
    t1 = (rng.normal(scale=0.05, size=(na, nva)),
          rng.normal(scale=0.05, size=(nb, nvb)))
    t2 = (rng.normal(scale=0.05, size=(na, na, nva, nva)),
          rng.normal(scale=0.05, size=(na, nb, nva, nvb)),
          rng.normal(scale=0.05, size=(nb, nb, nvb, nvb)))
    return t1, t2


def check_matrix_free_products(engine, t1, t2, seed):
    rng = numpy.random.default_rng(seed)
    jacobian = engine.jacobian(t1, t2).copy()
    vector = rng.normal(size=len(engine.coordinates))
    weights = rng.normal(size=engine.nconstraint)
    metric = rng.random(len(engine.coordinates))

    assert numpy.allclose(engine.jvp(t1, t2, vector), jacobian @ vector)
    assert numpy.allclose(engine.vjp(t1, t2, weights), jacobian.T @ weights)
    assert numpy.allclose(
        engine.jacobian_row_norms(t1, t2),
        numpy.linalg.norm(jacobian, axis=1),
    )
    assert numpy.allclose(
        engine.weighted_gram_diagonal(t1, t2, metric),
        numpy.einsum("ri,i,ri->r", jacobian, metric, jacobian),
    )

    values = engine.amplitude_values(t1, t2)
    blocks = engine._blocks(t1, t2)
    step = 1e-6
    shifted = []
    for sign in (1.0, -1.0):
        for value, delta, (block, index) in zip(
                values, vector, engine.coordinates):
            blocks[block][index] = value + sign * step * delta
        shifted.append(weights @ engine.jacobian(t1, t2).copy())
    for value, (block, index) in zip(values, engine.coordinates):
        blocks[block][index] = value
    finite_difference = (shifted[0] - shifted[1]) / (2 * step)
    hessian_error = numpy.max(numpy.abs(
        engine.hessian_vector(t1, t2, weights, vector) - finite_difference
    ))

    engine._cache_values = None
    engine.constraint(t1, t2)
    engine.jvp(t1, t2, vector)
    engine.vjp(t1, t2, weights)
    engine.jacobian_row_norms(t1, t2)
    engine.weighted_gram_diagonal(t1, t2, metric)
    engine.hessian_vector(t1, t2, weights, vector)
    assert engine._cache_jacobian is None
    return hessian_error


# A five-spin quartet gives one equation despite different alpha/beta orbital
# permutations.
occupation = "aaaba"
alpha, beta = occupations_from_string(occupation)
quartet = DominantSectorSpinConstraint(
    5, (4, 1), occupation,
    alpha_permutation=_reference_permutation(alpha, 5),
    beta_permutation=_reference_permutation(beta, 5),
)
t1q, t2q = random_amplitudes(quartet, 9)
assert quartet.nconstraint == 1
quartet_error = quartet.check_jacobian(t1q, t2q, ndir=5)
quartet_hessian_error = check_matrix_free_products(quartet, t1q, t2q, 109)


# A five-spin doublet gives five equations.  Its exact coefficients contain
# disconnected terms through quadruple excitation rank.
occupation = "ababa"
alpha, beta = occupations_from_string(occupation)
doublet = DominantSectorSpinConstraint(
    5, (3, 2), occupation,
    alpha_permutation=_reference_permutation(alpha, 5),
    beta_permutation=_reference_permutation(beta, 5),
)
t1d, t2d = random_amplitudes(doublet, 12)
assert doublet.nconstraint == 5
doublet_error = doublet.check_jacobian(t1d, t2d, ndir=5)
doublet_hessian_error = check_matrix_free_products(doublet, t1d, t2d, 112)
assert max(quartet_hessian_error, doublet_hessian_error) < 1e-8

print("CISD coefficient max error = %.3e" % maximum)
print("quartet Jacobian max error = %.3e" % quartet_error)
print("doublet Jacobian max error = %.3e" % doublet_error)
print("matrix-free Hessian max error = %.3e" % max(
    quartet_hessian_error, doublet_hessian_error
))

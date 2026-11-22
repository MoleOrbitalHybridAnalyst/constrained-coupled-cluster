"""exp(T)|0> built from the decomposed T1/T2 must reproduce the CI vector
through doubles, and the embedding must place the active block correctly."""
import numpy
import pyscf
from pyscf import mcscf
from pyscf.fci import cistring

from ccc.active_spin_constraint import DominantSectorSpinConstraint
from ccc.cluster_decomposition import cas_amplitudes, cluster_decompose


def excitation_rank(string, nocc):
    return nocc - bin(int(string) & ((1 << nocc) - 1)).count("1")


mol = pyscf.M(atom=";".join("H 0 0 %.1f" % (1.4 * i) for i in range(9)),
              basis="sto-3g", spin=3, verbose=0)
mf = mol.UHF().run()
for ncas, nelecas in ((5, (3, 2)), (6, (4, 3)), (9, (6, 3))):
    mc = mcscf.UCASCI(mf, ncas, nelecas)
    mc.kernel()
    rng = numpy.random.default_rng(1)
    ci = numpy.asarray(mc.ci) + rng.normal(scale=0.05, size=numpy.shape(mc.ci))
    na, nb = nelecas

    t1, t2 = cluster_decompose(ci, ncas, nelecas)
    occupation = "2" * nb + "a" * (na - nb) + "0" * (ncas - na)
    engine = DominantSectorSpinConstraint(
        ncas, nelecas, occupation, alpha_permutation=numpy.arange(ncas),
        beta_permutation=numpy.arange(ncas))
    wavefunction = engine.wavefunction(t1, t2) * ci[0, 0]
    error = 0.0
    for ia, sa in enumerate(cistring.make_strings(range(ncas), na)):
        for ib, sb in enumerate(cistring.make_strings(range(ncas), nb)):
            if excitation_rank(sa, na) + excitation_rank(sb, nb) <= 2:
                error = max(error, abs(wavefunction[ia, ib] - ci[ia, ib]))
    assert error < 1e-12, error

    full_t1, full_t2 = cas_amplitudes(mc, ci=ci)
    nmo = mol.nao
    nocc = (mc.ncore[0] + na, mc.ncore[1] + nb)
    assert full_t1[0].shape == (nocc[0], nmo - nocc[0])
    assert full_t2[1].shape == (nocc[0], nocc[1], nmo - nocc[0], nmo - nocc[1])
    oa, ob = slice(mc.ncore[0], nocc[0]), slice(mc.ncore[1], nocc[1])
    va, vb = slice(0, ncas - na), slice(0, ncas - nb)
    assert numpy.array_equal(full_t1[0][oa, va], t1[0])
    assert numpy.array_equal(full_t2[1][oa, ob, va, vb], t2[1])
    assert numpy.array_equal(full_t2[2][ob, ob, vb, vb], t2[2])
    for full, active in zip(full_t1 + full_t2, t1 + t2):
        assert abs(numpy.linalg.norm(full) - numpy.linalg.norm(active)) < 1e-14
    print("CAS(%d, %s) ncore %s: max error through doubles %.1e"
          % (ncas, nelecas, mc.ncore, error))
print("cluster decomposition passed")

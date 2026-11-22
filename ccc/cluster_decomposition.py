"""Cluster decomposition of an unrestricted CI vector into T1 and T2.

For a CI vector normalized so that the reference coefficient is one,
``|Psi> = exp(T)|0>`` gives, through doubles,

    C1 = T1,        C2 = T2 + T1^2 / 2,

so ``T1 = C1`` and ``T2 = C2 - T1^2 / 2``.  The CI coefficients are read off
the FCI vector with PySCF's UCISD conventions, which carry the determinant
phases and return ``c2aa``/``c2bb`` as fully antisymmetric tensors.
"""
import numpy
from pyscf.ci import ucisd


def cluster_decompose(ci, norb, nelec, tol=1e-12):
    """T1 and T2 of an FCI vector in its own orbital space.

    Args:
        ci: FCI vector of shape ``(n_alpha_strings, n_beta_strings)``.  The
            reference determinant occupies the lowest ``nelec[0]`` alpha and
            ``nelec[1]`` beta orbitals.
        norb: number of spatial orbitals.
        nelec: ``(n_alpha, n_beta)``.
        tol: smallest acceptable magnitude of the reference coefficient.

    Returns:
        ``(t1a, t1b), (t2aa, t2ab, t2bb)`` in PySCF UCCSD ordering.
    """
    ci = numpy.asarray(ci)
    nelec = tuple(int(n) for n in nelec)
    cisdvec = ucisd.from_fcivec(ci, norb, nelec)
    nocc = nelec
    nmo = (norb, norb)
    c0, (c1a, c1b), (c2aa, c2ab, c2bb) = ucisd.cisdvec_to_amplitudes(
        cisdvec, nmo, nocc)
    if abs(c0) < tol:
        raise ValueError(
            "reference coefficient %.3e is too small to decompose" % abs(c0))

    t1a = c1a / c0
    t1b = c1b / c0
    t2ab = c2ab / c0 - numpy.einsum("ia,jb->ijab", t1a, t1b)
    t2aa = c2aa / c0 - _antisymmetric_square(t1a)
    t2bb = c2bb / c0 - _antisymmetric_square(t1b)
    return (t1a, t1b), (t2aa, t2ab, t2bb)


def _antisymmetric_square(t1):
    """Same-spin ``T1^2 / 2`` block: ``t_i^a t_j^b - t_i^b t_j^a``."""
    direct = numpy.einsum("ia,jb->ijab", t1, t1)
    return direct - direct.transpose(0, 1, 3, 2)


def embed_amplitudes(t1, t2, ncore, nmo):
    """Place active-space amplitudes into full-space UCCSD arrays.

    The active occupied orbitals follow ``ncore`` core orbitals and the active
    virtual orbitals are the lowest virtuals, as in a CASCI orbital ordering.
    Amplitudes outside the active space are zero.
    """
    ncore = _pair(ncore)
    nmo = _pair(nmo)
    nocc_act = tuple(block.shape[0] for block in t1)
    nvir_act = tuple(block.shape[1] for block in t1)
    nocc = tuple(ncore[s] + nocc_act[s] for s in (0, 1))
    nvir = tuple(nmo[s] - nocc[s] for s in (0, 1))
    if any(nvir[s] < nvir_act[s] for s in (0, 1)):
        raise ValueError("active space does not fit in %s orbitals" % (nmo,))
    occ = tuple(slice(ncore[s], nocc[s]) for s in (0, 1))
    vir = tuple(slice(0, nvir_act[s]) for s in (0, 1))

    full_t1 = tuple(numpy.zeros((nocc[s], nvir[s]), dtype=t1[s].dtype)
                    for s in (0, 1))
    for s in (0, 1):
        full_t1[s][occ[s], vir[s]] = t1[s]

    full_t2 = []
    for block, (p, q) in zip(t2, ((0, 0), (0, 1), (1, 1))):
        full = numpy.zeros((nocc[p], nocc[q], nvir[p], nvir[q]),
                           dtype=block.dtype)
        full[occ[p], occ[q], vir[p], vir[q]] = block
        full_t2.append(full)
    return full_t1, tuple(full_t2)


def cas_amplitudes(mc, ci=None):
    """Full-space T1 and T2 of an unrestricted CASCI wavefunction.

    Args:
        mc: a ``pyscf.mcscf.UCASCI`` object (``ncore``, ``ncas``, ``nelecas``,
            ``mo_coeff`` and, unless ``ci`` is given, ``ci`` are read from it).
        ci: optional CI vector overriding ``mc.ci``.

    Returns:
        ``(t1a, t1b), (t2aa, t2ab, t2bb)`` shaped for a UCCSD calculation in
        all of ``mc``'s orbitals, zero outside the active space.
    """
    ci = mc.ci if ci is None else ci
    t1, t2 = cluster_decompose(ci, mc.ncas, mc.nelecas)
    nmo = tuple(numpy.asarray(c).shape[1] for c in mc.mo_coeff)
    return embed_amplitudes(t1, t2, mc.ncore, nmo)


def _pair(value):
    if numpy.ndim(value) == 0:
        return (int(value), int(value))
    return tuple(int(v) for v in value)

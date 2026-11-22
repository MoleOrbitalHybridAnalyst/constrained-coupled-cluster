"""
Reference Lambda-UCCSD(T) energy kernel, adapted from the algebra in
PySCF's pyscf.cc.uccsd_t_slow.

This is intentionally a slow, full-six-index reference implementation.
The Lambda amplitudes are assumed to have been solved externally.

Interface
---------
kernel(mcc, eris, t1=None, t2=None, l1=None, l2=None)

t1 = (t1a, t1b)
t2 = (t2aa, t2ab, t2bb)
l1 = (l1a, l1b)
l2 = (l2aa, l2ab, l2bb)

The right triples moment is built from T2.
The left triples moment is built from Lambda1/Lambda2.

For real amplitudes this is simply the replacement of the historical
left cap (T1,T2) by (Lambda1,Lambda2).

For complex amplitudes, the internal conjugation of Lambda amplitudes
is chosen so that the final .conj() used by the original PySCF slow
formula produces a left moment linear in Lambda rather than Lambda*.
This should still be benchmarked independently before production use
with complex orbitals.
"""

import numpy
from pyscf import lib


def kernel(mcc, eris, t1=None, t2=None, l1=None, l2=None):
    if t1 is None or t2 is None:
        t1, t2 = mcc.t1, mcc.t2
    if l1 is None or l2 is None:
        raise ValueError("Lambda-UCCSD(T) requires externally solved l1 and l2")

    def p6(t):
        return (t + t.transpose(1,2,0,4,5,3) +
                t.transpose(2,0,1,5,3,4) +
                t.transpose(0,2,1,3,5,4) +
                t.transpose(2,1,0,5,4,3) +
                t.transpose(1,0,2,4,3,5))

    def r6(w):
        return (w + w.transpose(2,0,1,3,4,5) +
                w.transpose(1,2,0,3,4,5) -
                w.transpose(2,1,0,3,4,5) -
                w.transpose(0,2,1,3,4,5) -
                w.transpose(1,0,2,3,4,5))

    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    l1a, l1b = l1
    l2aa, l2ab, l2bb = l2

    # The original PySCF slow kernel builds a "ket-oriented" object and
    # conjugates the entire left factor in the final contraction.
    # Build that object from Lambda* so the final conjugation yields a
    # left moment linear in Lambda.
    q1a, q1b = l1a.conj(), l1b.conj()
    q2aa, q2ab, q2bb = l2aa.conj(), l2ab.conj(), l2bb.conj()

    nocca, noccb = t2ab.shape[:2]
    mo_ea, mo_eb = eris.mo_energy
    eia = mo_ea[:nocca,None] - mo_ea[nocca:]
    eIA = mo_eb[:noccb,None] - mo_eb[noccb:]
    fvo = eris.focka[nocca:,:nocca]
    fVO = eris.fockb[noccb:,:noccb]

    # ------------------------------------------------------------------
    # aaa
    # ------------------------------------------------------------------
    d3 = lib.direct_sum('ia+jb+kc->ijkabc', eia, eia, eia)

    # Right triples numerator: W(T2)
    wr = numpy.einsum('ijae,kceb->ijkabc',
                      t2aa, numpy.asarray(eris.get_ovvv()).conj())
    wr -= numpy.einsum('mkbc,iajm->ijkabc',
                       t2aa, numpy.asarray(eris.ovoo).conj())
    rr = r6(wr)

    # Left triples numerator: W(Lambda2) + V(Lambda1,Lambda2)
    wl = numpy.einsum('ijae,kceb->ijkabc',
                      q2aa, numpy.asarray(eris.get_ovvv()).conj())
    wl -= numpy.einsum('mkbc,iajm->ijkabc',
                       q2aa, numpy.asarray(eris.ovoo).conj())
    vl = numpy.einsum('jbkc,ia->ijkabc',
                      numpy.asarray(eris.ovov).conj(), q1a)
    vl += numpy.einsum('jkbc,ai->ijkabc', q2aa, fvo) * .5

    lwd = p6(wl + vl) / d3
    et = numpy.einsum('ijkabc,ijkabc', lwd.conj(), rr)

    # ------------------------------------------------------------------
    # bbb
    # ------------------------------------------------------------------
    d3 = lib.direct_sum('ia+jb+kc->ijkabc', eIA, eIA, eIA)

    wr = numpy.einsum('ijae,kceb->ijkabc',
                      t2bb, numpy.asarray(eris.get_OVVV()).conj())
    wr -= numpy.einsum('imab,kcjm->ijkabc',
                       t2bb, numpy.asarray(eris.OVOO).conj())
    rr = r6(wr)

    wl = numpy.einsum('ijae,kceb->ijkabc',
                      q2bb, numpy.asarray(eris.get_OVVV()).conj())
    wl -= numpy.einsum('imab,kcjm->ijkabc',
                       q2bb, numpy.asarray(eris.OVOO).conj())
    vl = numpy.einsum('jbkc,ia->ijkabc',
                      numpy.asarray(eris.OVOV).conj(), q1b)
    vl += numpy.einsum('jkbc,ai->ijkabc', q2bb, fVO) * .5

    lwd = p6(wl + vl) / d3
    et += numpy.einsum('ijkabc,ijkabc', lwd.conj(), rr)

    # ------------------------------------------------------------------
    # baa: beta-alpha-alpha
    # ------------------------------------------------------------------
    # Right W(T2)
    wr  = numpy.einsum('jIeA,kceb->IjkAbc',
                       t2ab, numpy.asarray(eris.get_ovvv()).conj()) * 2
    wr += numpy.einsum('jIbE,kcEA->IjkAbc',
                       t2ab, numpy.asarray(eris.get_ovVV()).conj()) * 2
    wr += numpy.einsum('jkbe,IAec->IjkAbc',
                       t2aa, numpy.asarray(eris.get_OVvv()).conj())
    wr -= numpy.einsum('mIbA,kcjm->IjkAbc',
                       t2ab, numpy.asarray(eris.ovoo).conj()) * 2
    wr -= numpy.einsum('jMbA,kcIM->IjkAbc',
                       t2ab, numpy.asarray(eris.ovOO).conj()) * 2
    wr -= numpy.einsum('jmbc,IAkm->IjkAbc',
                       t2aa, numpy.asarray(eris.OVoo).conj())

    rr = wr - wr.transpose(0,2,1,3,4,5)
    rr = rr + rr.transpose(0,2,1,3,5,4)

    # Left W(Lambda2)
    wl  = numpy.einsum('jIeA,kceb->IjkAbc',
                       q2ab, numpy.asarray(eris.get_ovvv()).conj()) * 2
    wl += numpy.einsum('jIbE,kcEA->IjkAbc',
                       q2ab, numpy.asarray(eris.get_ovVV()).conj()) * 2
    wl += numpy.einsum('jkbe,IAec->IjkAbc',
                       q2aa, numpy.asarray(eris.get_OVvv()).conj())
    wl -= numpy.einsum('mIbA,kcjm->IjkAbc',
                       q2ab, numpy.asarray(eris.ovoo).conj()) * 2
    wl -= numpy.einsum('jMbA,kcIM->IjkAbc',
                       q2ab, numpy.asarray(eris.ovOO).conj()) * 2
    wl -= numpy.einsum('jmbc,IAkm->IjkAbc',
                       q2aa, numpy.asarray(eris.OVoo).conj())

    # Left V(Lambda1,Lambda2)
    vl  = numpy.einsum('jbkc,IA->IjkAbc',
                       numpy.asarray(eris.ovov).conj(), q1b)
    vl += numpy.einsum('kcIA,jb->IjkAbc',
                       numpy.asarray(eris.ovOV).conj(), q1a)
    vl += numpy.einsum('kcIA,jb->IjkAbc',
                       numpy.asarray(eris.ovOV).conj(), q1a)
    vl += numpy.einsum('jkbc,AI->IjkAbc', q2aa, fVO) * .5
    vl += numpy.einsum('kIcA,bj->IjkAbc', q2ab, fvo) * 2
    wl += vl

    d3 = lib.direct_sum('ia+jb+kc->ijkabc', eIA, eia, eia)
    rr /= d3
    et += numpy.einsum('ijkabc,ijkabc', wl.conj(), rr)

    # ------------------------------------------------------------------
    # bba: alpha-beta-beta
    # ------------------------------------------------------------------
    # Right W(T2)
    wr  = numpy.einsum('ijae,kceb->ijkabc',
                       t2ab, numpy.asarray(eris.get_OVVV()).conj()) * 2
    wr += numpy.einsum('ijeb,kcea->ijkabc',
                       t2ab, numpy.asarray(eris.get_OVvv()).conj()) * 2
    wr += numpy.einsum('jkbe,iaec->ijkabc',
                       t2bb, numpy.asarray(eris.get_ovVV()).conj())
    wr -= numpy.einsum('imab,kcjm->ijkabc',
                       t2ab, numpy.asarray(eris.OVOO).conj()) * 2
    wr -= numpy.einsum('mjab,kcim->ijkabc',
                       t2ab, numpy.asarray(eris.OVoo).conj()) * 2
    wr -= numpy.einsum('jmbc,iakm->ijkabc',
                       t2bb, numpy.asarray(eris.ovOO).conj())

    rr = wr - wr.transpose(0,2,1,3,4,5)
    rr = rr + rr.transpose(0,2,1,3,5,4)

    # Left W(Lambda2)
    wl  = numpy.einsum('ijae,kceb->ijkabc',
                       q2ab, numpy.asarray(eris.get_OVVV()).conj()) * 2
    wl += numpy.einsum('ijeb,kcea->ijkabc',
                       q2ab, numpy.asarray(eris.get_OVvv()).conj()) * 2
    wl += numpy.einsum('jkbe,iaec->ijkabc',
                       q2bb, numpy.asarray(eris.get_ovVV()).conj())
    wl -= numpy.einsum('imab,kcjm->ijkabc',
                       q2ab, numpy.asarray(eris.OVOO).conj()) * 2
    wl -= numpy.einsum('mjab,kcim->ijkabc',
                       q2ab, numpy.asarray(eris.OVoo).conj()) * 2
    wl -= numpy.einsum('jmbc,iakm->ijkabc',
                       q2bb, numpy.asarray(eris.ovOO).conj())

    # Left V(Lambda1,Lambda2)
    vl  = numpy.einsum('jbkc,ia->ijkabc',
                       numpy.asarray(eris.OVOV).conj(), q1a)
    vl += numpy.einsum('iakc,jb->ijkabc',
                       numpy.asarray(eris.ovOV).conj(), q1b)
    vl += numpy.einsum('iakc,jb->ijkabc',
                       numpy.asarray(eris.ovOV).conj(), q1b)
    vl += numpy.einsum('JKBC,ai->iJKaBC', q2bb, fvo) * .5
    vl += numpy.einsum('iKaC,BJ->iJKaBC', q2ab, fVO) * 2
    wl += vl

    d3 = lib.direct_sum('ia+jb+kc->ijkabc', eia, eIA, eIA)
    rr /= d3
    et += numpy.einsum('ijkabc,ijkabc', wl.conj(), rr)

    et *= .25
    return et


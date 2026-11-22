"""Memory-efficient real Lambda-UCCSD(T) energy correction.

The blocked driver follows :mod:`pyscf.cc.uccsd_t`.  The right triples
moment is built from T2, while the left triples moment is built from the
externally supplied Lambda1/Lambda2 amplitudes.
"""

import ctypes

import numpy
from pyscf import lib
from pyscf.cc import uccsd_t
from pyscf.lib import logger

from .lib import load_library


libcc = load_library("libcc")
libcc.CCuccsd_lambdat_aaa.restype = None
libcc.CCuccsd_lambdat_baa.restype = None


def _validate_amplitudes(mycc, eris, t1, t2, l1, l2):
    try:
        t1 = tuple(t1)
        t2 = tuple(t2)
        l1 = tuple(l1)
        l2 = tuple(l2)
    except TypeError as err:
        raise ValueError("T and Lambda amplitudes must be block tuples") from err
    if len(t1) != 2 or len(l1) != 2 or len(t2) != 3 or len(l2) != 3:
        raise ValueError("unrestricted amplitudes require two singles and three doubles blocks")

    nocca, noccb = (int(x) for x in mycc.nocc)
    nmoa, nmob = eris.focka.shape[0], eris.fockb.shape[0]
    nvira, nvirb = nmoa - nocca, nmob - noccb
    expected = (
        (nocca, nvira), (noccb, nvirb),
        (nocca, nocca, nvira, nvira),
        (nocca, noccb, nvira, nvirb),
        (noccb, noccb, nvirb, nvirb),
    )
    blocks = t1 + t2
    left = l1 + l2
    for name, block, shape in zip(
            ("t1a", "t1b", "t2aa", "t2ab", "t2bb"), blocks, expected):
        if numpy.shape(block) != shape:
            raise ValueError("%s has shape %s; expected %s" %
                             (name, numpy.shape(block), shape))
    for name, block, shape in zip(
            ("l1a", "l1b", "l2aa", "l2ab", "l2bb"), left, expected):
        if numpy.shape(block) != shape:
            raise ValueError("%s has shape %s; expected %s" %
                             (name, numpy.shape(block), shape))

    dtype_inputs = blocks + left + (
        eris.focka, eris.fockb, eris.ovov, eris.OVOV, eris.ovOV,
        eris.ovoo, eris.OVOO, eris.ovOO, eris.OVoo,
    )
    dtype = numpy.result_type(*dtype_inputs)
    if numpy.issubdtype(dtype, numpy.complexfloating):
        raise NotImplementedError(
            "optimized Lambda-UCCSD(T) currently supports real data only")
    if dtype != numpy.dtype(numpy.float64):
        raise TypeError("optimized Lambda-UCCSD(T) requires float64 data")
    if not numpy.asarray(t2[1]).flags.c_contiguous:
        raise ValueError("t2ab must be C-contiguous")
    if not numpy.asarray(l2[1]).flags.c_contiguous:
        raise ValueError("l2ab must be C-contiguous")
    if t2[1] is not l2[1] and numpy.shares_memory(t2[1], l2[1]):
        raise ValueError(
            "t2ab and l2ab may be identical or disjoint, but not overlapping views")
    return t1, t2, l1, l2


class _MixedTranspose:
    """Transpose an ab doubles block while reusing writable input storage."""

    def __init__(self, block):
        self.original = numpy.asarray(block)
        self.nocca, self.noccb, self.nvira, self.nvirb = self.original.shape
        self.inplace = bool(
            self.original.flags.c_contiguous and self.original.flags.writeable)
        if self.inplace:
            matrix = self.original.copy().reshape(
                self.nocca * self.noccb, self.nvira * self.nvirb)
            transposed = lib.transpose(matrix, out=self.original)
            self.view = transposed.reshape(
                self.nvira, self.nvirb, self.nocca, self.noccb)
        else:
            self.view = self.original.transpose(2, 3, 0, 1).copy()
        self.orientation = "ab"

    def to_ba(self):
        if self.orientation != "ab":
            return self.view
        data = self.view.transpose(1, 0, 3, 2).copy()
        self.view = numpy.ndarray(
            (self.nvirb, self.nvira, self.noccb, self.nocca),
            buffer=self.view, dtype=self.view.dtype)
        self.view[:] = data
        self.orientation = "ba"
        return self.view

    def restore(self):
        if not self.inplace:
            return
        if self.orientation == "ba":
            abt = self.view.transpose(1, 0, 3, 2).copy()
        else:
            abt = self.view.copy()
        matrix = abt.reshape(
            self.nvira * self.nvirb, self.nocca * self.noccb)
        lib.transpose(matrix, out=self.original)
        self.orientation = "restored"


def kernel(mycc, eris=None, t1=None, t2=None, l1=None, l2=None,
           verbose=logger.NOTE):
    """Return the real Lambda-UCCSD(T) triples energy correction."""
    cpu1 = cpu0 = (logger.process_clock(), logger.perf_counter())
    log = logger.new_logger(mycc, verbose)
    if t1 is None:
        t1 = mycc.t1
    if t2 is None:
        t2 = mycc.t2
    if t1 is None or t2 is None:
        raise ValueError("T1 and T2 amplitudes are required")
    if l1 is None or l2 is None:
        raise ValueError("Lambda-UCCSD(T) requires explicit l1 and l2")
    if eris is None:
        eris = mycc.ao2mo(mycc.mo_coeff)
    t1, t2, l1, l2 = _validate_amplitudes(mycc, eris, t1, t2, l1, l2)

    t1a, t1b = t1
    t2aa, t2ab, t2bb = t2
    l1a, l1b = l1
    l2aa, l2ab, l2bb = l2
    nocca, noccb = (int(x) for x in mycc.nocc)
    nmoa, nmob = eris.focka.shape[0], eris.fockb.shape[0]
    nvira, nvirb = nmoa - nocca, nmob - noccb

    ftmp = None if mycc.incore_complete else lib.H5TmpFile()
    t1aT = t1a.T.copy()
    t1bT = t1b.T.copy()
    l1aT = l1a.T.copy()
    l1bT = l1b.T.copy()
    t2aaT = t2aa.transpose(2, 3, 0, 1).copy()
    t2bbT = t2bb.transpose(2, 3, 0, 1).copy()
    l2aaT = l2aa.transpose(2, 3, 0, 1).copy()
    l2bbT = l2bb.transpose(2, 3, 0, 1).copy()

    eris_vooo = numpy.asarray(eris.ovoo).transpose(1, 3, 0, 2).copy()
    eris_VOOO = numpy.asarray(eris.OVOO).transpose(1, 3, 0, 2).copy()
    eris_vOoO = numpy.asarray(eris.ovOO).transpose(1, 3, 0, 2).copy()
    eris_VoOo = numpy.asarray(eris.OVoo).transpose(1, 3, 0, 2).copy()
    eris_vvop, eris_VVOP, eris_vVoP, eris_VvOp = \
        uccsd_t._sort_eri(mycc, eris, ftmp, log)
    cpu1 = log.timer_debug1("Lambda-UCCSD(T) sort_eri", *cpu1)

    et_sum = numpy.zeros(1, dtype=numpy.float64)
    mem_now = lib.current_memory()[0]
    max_memory = max(0, mycc.max_memory - mem_now)

    bufsize = max(8, int((max_memory * .5e6 / 8
                          - nocca**3 * 3 * lib.num_threads())
                         * .4 / max(1, nocca * nmoa)))
    orbsym = numpy.zeros(nocca, dtype=int)
    contract = _gen_contract_aaa(
        t1aT, t2aaT, l1aT, l2aaT, eris_vooo, eris.focka,
        eris.mo_energy[0], orbsym, log)
    with lib.call_in_background(contract, sync=not mycc.async_io) as ctr:
        for a0, a1 in reversed(list(lib.prange_tril(0, nvira, bufsize))):
            row_a = numpy.asarray(eris_vvop[a0:a1, :a1], order="C")
            col_a = row_a if a0 == 0 else numpy.asarray(
                eris_vvop[:a0, a0:a1], order="C")
            ctr(et_sum, a0, a1, a0, a1, (row_a, col_a, row_a, col_a))
            for b0, b1 in lib.prange_tril(0, a0, bufsize / 8):
                row_b = numpy.asarray(eris_vvop[b0:b1, :b1], order="C")
                col_b = row_b if b0 == 0 else numpy.asarray(
                    eris_vvop[:b0, b0:b1], order="C")
                ctr(et_sum, a0, a1, b0, b1,
                    (row_a, col_a, row_b, col_b))
    cpu1 = log.timer_debug1("contract_lambda_aaa", *cpu1)

    bufsize = max(8, int((max_memory * .5e6 / 8
                          - noccb**3 * 3 * lib.num_threads())
                         * .4 / max(1, noccb * nmob)))
    orbsym = numpy.zeros(noccb, dtype=int)
    contract = _gen_contract_aaa(
        t1bT, t2bbT, l1bT, l2bbT, eris_VOOO, eris.fockb,
        eris.mo_energy[1], orbsym, log)
    with lib.call_in_background(contract, sync=not mycc.async_io) as ctr:
        for a0, a1 in reversed(list(lib.prange_tril(0, nvirb, bufsize))):
            row_a = numpy.asarray(eris_VVOP[a0:a1, :a1], order="C")
            col_a = row_a if a0 == 0 else numpy.asarray(
                eris_VVOP[:a0, a0:a1], order="C")
            ctr(et_sum, a0, a1, a0, a1, (row_a, col_a, row_a, col_a))
            for b0, b1 in lib.prange_tril(0, a0, bufsize / 8):
                row_b = numpy.asarray(eris_VVOP[b0:b1, :b1], order="C")
                col_b = row_b if b0 == 0 else numpy.asarray(
                    eris_VVOP[:b0, b0:b1], order="C")
                ctr(et_sum, a0, a1, b0, b1,
                    (row_a, col_a, row_b, col_b))
    cpu1 = log.timer_debug1("contract_lambda_bbb", *cpu1)

    # A mixed-spin triple requires at least one virtual orbital of each spin.
    # PySCF's unrestricted (T) driver divides its mixed-spin block size by the
    # opposite-spin virtual count, so fully occupied spin spaces need to return
    # after the already-completed aaa and bbb contractions.
    if nocca * noccb == 0 or nvira * nvirb == 0:
        correction = float(et_sum[0] * .25)
        log.timer("Lambda-UCCSD(T)", *cpu0)
        log.note("Lambda-UCCSD(T) correction = %.15g", correction)
        return correction

    shared_mixed = t2ab is l2ab
    twork = lwork = None
    try:
        twork = _MixedTranspose(t2ab)
        lwork = twork if shared_mixed else _MixedTranspose(l2ab)
        t2abT, l2abT = twork.view, lwork.view
        bufsize = int(max(12, (max_memory * .5e6 / 8
                               - noccb * nocca**2 * 5) * .7
                              / max(1, nocca * nmob)))
        ts_r = (t1aT, t1bT, t2aaT, t2abT)
        ts_l = (l1aT, l1bT, l2aaT, l2abT)
        fock = (eris.focka, eris.fockb)
        vooo = (eris_vooo, eris_vOoO, eris_VoOo)
        contract = _gen_contract_baa(
            ts_r, ts_l, vooo, fock, eris.mo_energy, log)
        with lib.call_in_background(contract, sync=not mycc.async_io) as ctr:
            for a0, a1 in lib.prange(0, nvirb, int(bufsize / nvira + 1)):
                row_a = numpy.asarray(eris_VvOp[a0:a1, :], order="C")
                col_a = numpy.asarray(eris_vVoP[:, a0:a1], order="C")
                for b0, b1 in lib.prange_tril(0, nvira, bufsize / 12):
                    row_b = numpy.asarray(eris_vvop[b0:b1, :b1], order="C")
                    col_b = numpy.asarray(eris_vvop[:b0, b0:b1], order="C")
                    ctr(et_sum, a0, a1, b0, b1,
                        (row_a, col_a, row_b, col_b))
        cpu1 = log.timer_debug1("contract_lambda_baa", *cpu1)

        t2baT = twork.to_ba()
        l2baT = lwork.to_ba()
        ts_r = (t1bT, t1aT, t2bbT, t2baT)
        ts_l = (l1bT, l1aT, l2bbT, l2baT)
        fock = (eris.fockb, eris.focka)
        mo_energy = (eris.mo_energy[1], eris.mo_energy[0])
        vooo = (eris_VOOO, eris_VoOo, eris_vOoO)
        contract = _gen_contract_baa(ts_r, ts_l, vooo, fock, mo_energy, log)
        for a0, a1 in lib.prange(0, nvira, int(bufsize / nvirb + 1)):
            with lib.call_in_background(
                    contract, sync=not mycc.async_io) as ctr:
                row_a = numpy.asarray(eris_vVoP[a0:a1, :], order="C")
                col_a = numpy.asarray(eris_VvOp[:, a0:a1], order="C")
                for b0, b1 in lib.prange_tril(0, nvirb, bufsize / 12):
                    row_b = numpy.asarray(eris_VVOP[b0:b1, :b1], order="C")
                    col_b = numpy.asarray(eris_VVOP[:b0, b0:b1], order="C")
                    ctr(et_sum, a0, a1, b0, b1,
                        (row_a, col_a, row_b, col_b))
        log.timer_debug1("contract_lambda_abb", *cpu1)
    finally:
        if lwork is not None and lwork is not twork:
            lwork.restore()
        if twork is not None:
            twork.restore()

    correction = float(et_sum[0] * .25)
    log.timer("Lambda-UCCSD(T)", *cpu0)
    log.note("Lambda-UCCSD(T) correction = %.15g", correction)
    return correction


def _gen_contract_aaa(t1T, t2T, l1T, l2T, vooo, fock, mo_energy,
                      orbsym, log):
    nvir, nocc = t1T.shape
    mo_energy = numpy.asarray(mo_energy, dtype=numpy.float64, order="C")
    fvo = numpy.asarray(fock[nocc:, :nocc], order="C").copy()
    cpu2 = [logger.process_clock(), logger.perf_counter()]
    # The current UCCSD(T) driver disables orbital symmetry with zero labels.
    orbsym = numpy.asarray(orbsym, dtype=numpy.int32, order="C")

    def contract(et_sum, a0, a1, b0, b1, cache):
        row_a, col_a, row_b, col_b = cache
        libcc.CCuccsd_lambdat_aaa(
            et_sum.ctypes.data_as(ctypes.c_void_p),
            mo_energy.ctypes.data_as(ctypes.c_void_p),
            t1T.ctypes.data_as(ctypes.c_void_p),
            t2T.ctypes.data_as(ctypes.c_void_p),
            l1T.ctypes.data_as(ctypes.c_void_p),
            l2T.ctypes.data_as(ctypes.c_void_p),
            vooo.ctypes.data_as(ctypes.c_void_p),
            fvo.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nocc), ctypes.c_int(nvir),
            ctypes.c_int(a0), ctypes.c_int(a1),
            ctypes.c_int(b0), ctypes.c_int(b1),
            row_a.ctypes.data_as(ctypes.c_void_p),
            col_a.ctypes.data_as(ctypes.c_void_p),
            row_b.ctypes.data_as(ctypes.c_void_p),
            col_b.ctypes.data_as(ctypes.c_void_p))
        cpu2[:] = log.timer_debug1(
            "contract %d:%d,%d:%d" % (a0, a1, b0, b1), *cpu2)
    return contract


def _gen_contract_baa(ts_r, ts_l, vooo, fock, mo_energy, log):
    t1aT, t1bT, t2aaT, t2abT = ts_r
    l1aT, l1bT, l2aaT, l2abT = ts_l
    focka, fockb = fock
    vooo, vOoO, VoOo = vooo
    nvira, nocca = t1aT.shape
    nvirb, noccb = t1bT.shape
    mo_ea = numpy.asarray(mo_energy[0], dtype=numpy.float64, order="C")
    mo_eb = numpy.asarray(mo_energy[1], dtype=numpy.float64, order="C")
    fvo = numpy.asarray(focka[nocca:, :nocca], order="C").copy()
    fVO = numpy.asarray(fockb[noccb:, :noccb], order="C").copy()
    cpu2 = [logger.process_clock(), logger.perf_counter()]

    def contract(et_sum, a0, a1, b0, b1, cache):
        row_a, col_a, row_b, col_b = cache
        libcc.CCuccsd_lambdat_baa(
            et_sum.ctypes.data_as(ctypes.c_void_p),
            mo_ea.ctypes.data_as(ctypes.c_void_p),
            mo_eb.ctypes.data_as(ctypes.c_void_p),
            t1aT.ctypes.data_as(ctypes.c_void_p),
            t1bT.ctypes.data_as(ctypes.c_void_p),
            t2aaT.ctypes.data_as(ctypes.c_void_p),
            t2abT.ctypes.data_as(ctypes.c_void_p),
            l1aT.ctypes.data_as(ctypes.c_void_p),
            l1bT.ctypes.data_as(ctypes.c_void_p),
            l2aaT.ctypes.data_as(ctypes.c_void_p),
            l2abT.ctypes.data_as(ctypes.c_void_p),
            vooo.ctypes.data_as(ctypes.c_void_p),
            vOoO.ctypes.data_as(ctypes.c_void_p),
            VoOo.ctypes.data_as(ctypes.c_void_p),
            fvo.ctypes.data_as(ctypes.c_void_p),
            fVO.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int(nocca), ctypes.c_int(noccb),
            ctypes.c_int(nvira), ctypes.c_int(nvirb),
            ctypes.c_int(a0), ctypes.c_int(a1),
            ctypes.c_int(b0), ctypes.c_int(b1),
            row_a.ctypes.data_as(ctypes.c_void_p),
            col_a.ctypes.data_as(ctypes.c_void_p),
            row_b.ctypes.data_as(ctypes.c_void_p),
            col_b.ctypes.data_as(ctypes.c_void_p))
        cpu2[:] = log.timer_debug1(
            "contract %d:%d,%d:%d" % (a0, a1, b0, b1), *cpu2)
    return contract


uccsd_lambdat = kernel

"""Matrix-free CISDT-closed spin constraints for unmatched UHF orbitals.

For different alpha and beta spatial orbitals,

    S+ = sum_pq S[p,q] a+_[p,alpha] a_[q,beta]

is dense.  A rank-r output determinant can therefore have a rank-(r+1)
preimage.  The output shells of rank one and two are consequently a
coefficient-independent subset whose complete preimages lie in CISDT.  This
module evaluates those equations directly as tensor contractions; it never
builds determinant tables, a Jacobian, or a Gram matrix in production use.
"""

from collections.abc import Sequence
from itertools import combinations
from math import comb
from time import perf_counter

import numpy

from .ucisd_spin_constraint import UCISDSpinConstraint


_BLOCKS = ("t1a", "t1b", "t2aa", "t2ab", "t2bb", "t3aab", "t3abb")


def _pairs(size):
    values = tuple(combinations(range(size), 2))
    if not values:
        return numpy.zeros(0, dtype=numpy.int64), numpy.zeros(0, dtype=numpy.int64)
    return tuple(numpy.asarray(x, dtype=numpy.int64) for x in zip(*values))


def _fill_pair_tensor(values, first, second, shape, axes):
    """Expand independent pair-pair values into a full antisymmetric tensor."""
    out = numpy.zeros(shape, dtype=values.dtype)
    i, j = first
    a, b = second
    if not i.size or not a.size:
        return out
    left = i[:, None]
    right = j[:, None]
    va = a[None, :]
    vb = b[None, :]
    canonical = values.reshape(i.size, a.size)
    index = [slice(None)] * len(shape)
    for oi, oj, sa, sb, sign in (
            (left, right, va, vb, 1.0),
            (right, left, va, vb, -1.0),
            (left, right, vb, va, -1.0),
            (right, left, vb, va, 1.0)):
        index[axes[0]] = oi
        index[axes[1]] = oj
        index[axes[2]] = sa
        index[axes[3]] = sb
        out[tuple(index)] = sign * canonical
    return out


def _pair_values(array, first, second, axes):
    i, j = first
    a, b = second
    if not i.size or not a.size:
        return numpy.zeros(i.size * a.size, dtype=array.dtype)
    index = [slice(None)] * array.ndim
    index[axes[0]] = i[:, None]
    index[axes[1]] = j[:, None]
    index[axes[2]] = a[None, :]
    index[axes[3]] = b[None, :]
    return numpy.asarray(array[tuple(index)]).ravel()


def _pair_cotangent(array, first, second, axes):
    """Collapse a full-tensor cotangent onto independent antisymmetric values."""
    i, j = first
    a, b = second
    if not i.size or not a.size:
        return numpy.zeros(i.size * a.size, dtype=array.dtype)
    result = numpy.zeros((i.size, a.size), dtype=array.dtype)
    index = [slice(None)] * array.ndim
    for oi, oj, sa, sb, sign in (
            (i[:, None], j[:, None], a[None, :], b[None, :], 1.0),
            (j[:, None], i[:, None], a[None, :], b[None, :], -1.0),
            (i[:, None], j[:, None], b[None, :], a[None, :], -1.0),
            (j[:, None], i[:, None], b[None, :], a[None, :], 1.0)):
        index[axes[0]] = oi
        index[axes[1]] = oj
        index[axes[2]] = sa
        index[axes[3]] = sb
        result += sign * array[tuple(index)]
    return result.ravel()


class _UCISDTCoordinates(Sequence):
    """Lazy independent T1/T2/T3 coordinates in occupied-first convention."""

    _ccc_compact_coordinates = True
    _ccc_amplitude_rank = 3

    def __init__(self, nmo, nocc):
        self.nmo = int(nmo)
        self.nocc = tuple(int(x) for x in nocc)
        self.nocca, self.noccb = self.nocc
        self.nvira = self.nmo - self.nocca
        self.nvirb = self.nmo - self.noccb
        self.opa = _pairs(self.nocca)
        self.opb = _pairs(self.noccb)
        self.vpa = _pairs(self.nvira)
        self.vpb = _pairs(self.nvirb)
        self.sizes = {
            "t1a": self.nocca * self.nvira,
            "t1b": self.noccb * self.nvirb,
            "t2aa": self.opa[0].size * self.vpa[0].size,
            "t2ab": self.nocca * self.noccb * self.nvira * self.nvirb,
            "t2bb": self.opb[0].size * self.vpb[0].size,
            "t3aab": (self.opa[0].size * self.vpa[0].size
                        * self.noccb * self.nvirb),
            "t3abb": (self.opb[0].size * self.vpb[0].size
                        * self.nocca * self.nvira),
        }
        self.offsets = {}
        position = 0
        for name in _BLOCKS:
            self.offsets[name] = position
            position += self.sizes[name]
        self.size = position

    def __len__(self):
        return self.size

    def _locate(self, position):
        position = int(position)
        if position < 0:
            position += self.size
        if not 0 <= position < self.size:
            raise IndexError("UCISDT coordinate index out of range")
        for name in _BLOCKS:
            offset = self.offsets[name]
            if position < offset + self.sizes[name]:
                return name, position - offset
        raise IndexError("UCISDT coordinate index out of range")

    def __getitem__(self, position):
        if isinstance(position, slice):
            return tuple(self[index] for index in range(*position.indices(self.size)))
        name, local = self._locate(position)
        if name == "t1a":
            return name, tuple(int(x) for x in divmod(local, self.nvira))
        if name == "t1b":
            return name, tuple(int(x) for x in divmod(local, self.nvirb))
        if name == "t2aa":
            oi, vi = divmod(local, self.vpa[0].size)
            return name, (int(self.opa[0][oi]), int(self.opa[1][oi]),
                          int(self.vpa[0][vi]), int(self.vpa[1][vi]))
        if name == "t2ab":
            index = numpy.unravel_index(
                local, (self.nocca, self.noccb, self.nvira, self.nvirb)
            )
            return name, tuple(int(x) for x in index)
        if name == "t2bb":
            oi, vi = divmod(local, self.vpb[0].size)
            return name, (int(self.opb[0][oi]), int(self.opb[1][oi]),
                          int(self.vpb[0][vi]), int(self.vpb[1][vi]))
        if name == "t3aab":
            pair, k, c = numpy.unravel_index(
                local,
                (self.opa[0].size * self.vpa[0].size,
                 self.noccb, self.nvirb),
            )
            oi, vi = divmod(pair, self.vpa[0].size)
            return name, (int(self.opa[0][oi]), int(self.opa[1][oi]), int(k),
                          int(self.vpa[0][vi]), int(self.vpa[1][vi]), int(c))
        pair, i, a = numpy.unravel_index(
            local,
            (self.opb[0].size * self.vpb[0].size,
             self.nocca, self.nvira),
        )
        oi, vi = divmod(pair, self.vpb[0].size)
        return name, (int(i), int(self.opb[0][oi]), int(self.opb[1][oi]), int(a),
                      int(self.vpb[0][vi]), int(self.vpb[1][vi]))

    def _chunks(self, t1, t2, t3):
        return (
            numpy.asarray(t1[0]).ravel(),
            numpy.asarray(t1[1]).ravel(),
            _pair_values(t2[0], self.opa, self.vpa, (0, 1, 2, 3)),
            numpy.asarray(t2[1]).ravel(),
            _pair_values(t2[2], self.opb, self.vpb, (0, 1, 2, 3)),
            self._mixed_aab_values(t3[1]),
            self._mixed_abb_values(t3[2]),
        )

    def _mixed_aab_values(self, array):
        if not self.opa[0].size or not self.vpa[0].size:
            return numpy.zeros(0, dtype=array.dtype)
        return numpy.asarray(array[
            self.opa[0][:, None, None, None],
            self.opa[1][:, None, None, None],
            numpy.arange(self.noccb)[None, None, :, None],
            self.vpa[0][None, :, None, None],
            self.vpa[1][None, :, None, None],
            numpy.arange(self.nvirb)[None, None, None, :],
        ]).transpose(0, 1, 2, 3).reshape(
            self.opa[0].size * self.vpa[0].size,
            self.noccb, self.nvirb,
        ).ravel()

    def _mixed_abb_values(self, array):
        if not self.opb[0].size or not self.vpb[0].size:
            return numpy.zeros(0, dtype=array.dtype)
        return numpy.asarray(array[
            numpy.arange(self.nocca)[None, None, :, None],
            self.opb[0][:, None, None, None],
            self.opb[1][:, None, None, None],
            numpy.arange(self.nvira)[None, None, None, :],
            self.vpb[0][None, :, None, None],
            self.vpb[1][None, :, None, None],
        ]).reshape(
            self.opb[0].size * self.vpb[0].size,
            self.nocca, self.nvira,
        ).ravel()

    def amplitude_values(self, t1, t2, t3):
        return numpy.concatenate(self._chunks(t1, t2, t3))

    def expand_vector(self, vector):
        vector = numpy.asarray(vector, dtype=float)
        if vector.shape != (self.size,):
            raise ValueError("UCISDT coordinate vector has the wrong shape")

        def chunk(name):
            start = self.offsets[name]
            return vector[start:start + self.sizes[name]]

        t1 = (
            chunk("t1a").reshape(self.nocca, self.nvira),
            chunk("t1b").reshape(self.noccb, self.nvirb),
        )
        t2aa = _fill_pair_tensor(
            chunk("t2aa"), self.opa, self.vpa,
            (self.nocca, self.nocca, self.nvira, self.nvira), (0, 1, 2, 3)
        )
        t2ab = chunk("t2ab").reshape(
            self.nocca, self.noccb, self.nvira, self.nvirb
        )
        t2bb = _fill_pair_tensor(
            chunk("t2bb"), self.opb, self.vpb,
            (self.noccb, self.noccb, self.nvirb, self.nvirb), (0, 1, 2, 3)
        )
        t3aab = numpy.zeros(
            (self.nocca, self.nocca, self.noccb,
             self.nvira, self.nvira, self.nvirb)
        )
        if self.sizes["t3aab"]:
            values = chunk("t3aab").reshape(
                self.opa[0].size, self.vpa[0].size,
                self.noccb, self.nvirb,
            ).transpose(0, 2, 1, 3)
            for oi, oj, va, vb, sign in (
                    (self.opa[0], self.opa[1], self.vpa[0], self.vpa[1], 1.0),
                    (self.opa[1], self.opa[0], self.vpa[0], self.vpa[1], -1.0),
                    (self.opa[0], self.opa[1], self.vpa[1], self.vpa[0], -1.0),
                    (self.opa[1], self.opa[0], self.vpa[1], self.vpa[0], 1.0)):
                t3aab[oi[:, None, None, None], oj[:, None, None, None],
                       numpy.arange(self.noccb)[None, :, None, None],
                       va[None, None, :, None], vb[None, None, :, None],
                       numpy.arange(self.nvirb)[None, None, None, :]] = sign * values
        t3abb = numpy.zeros(
            (self.nocca, self.noccb, self.noccb,
             self.nvira, self.nvirb, self.nvirb)
        )
        if self.sizes["t3abb"]:
            values = chunk("t3abb").reshape(
                self.opb[0].size, self.vpb[0].size,
                self.nocca, self.nvira,
            ).transpose(2, 0, 3, 1)
            for oj, ok, vb, vc, sign in (
                    (self.opb[0], self.opb[1], self.vpb[0], self.vpb[1], 1.0),
                    (self.opb[1], self.opb[0], self.vpb[0], self.vpb[1], -1.0),
                    (self.opb[0], self.opb[1], self.vpb[1], self.vpb[0], -1.0),
                    (self.opb[1], self.opb[0], self.vpb[1], self.vpb[0], 1.0)):
                t3abb[numpy.arange(self.nocca)[:, None, None, None],
                       oj[None, :, None, None], ok[None, :, None, None],
                       numpy.arange(self.nvira)[None, None, :, None],
                       vb[None, None, None, :], vc[None, None, None, :]] = sign * values
        return t1, (t2aa, t2ab, t2bb), (
            numpy.zeros((self.nocca,) * 3 + (self.nvira,) * 3),
            t3aab, t3abb,
            numpy.zeros((self.noccb,) * 3 + (self.nvirb,) * 3),
        )

    def pack_cotangent(self, t1, t2, t3):
        chunks = [numpy.asarray(t1[0]).ravel(), numpy.asarray(t1[1]).ravel()]
        chunks.append(_pair_cotangent(t2[0], self.opa, self.vpa,
                                       (0, 1, 2, 3)))
        chunks.append(numpy.asarray(t2[1]).ravel())
        chunks.append(_pair_cotangent(t2[2], self.opb, self.vpb,
                                       (0, 1, 2, 3)))
        if self.sizes["t3aab"]:
            result = numpy.zeros((self.opa[0].size, self.vpa[0].size,
                                  self.noccb, self.nvirb))
            for oi, oj, va, vb, sign in (
                    (self.opa[0], self.opa[1], self.vpa[0], self.vpa[1], 1.0),
                    (self.opa[1], self.opa[0], self.vpa[0], self.vpa[1], -1.0),
                    (self.opa[0], self.opa[1], self.vpa[1], self.vpa[0], -1.0),
                    (self.opa[1], self.opa[0], self.vpa[1], self.vpa[0], 1.0)):
                result += sign * t3[1][
                    oi[:, None, None, None], oj[:, None, None, None],
                    numpy.arange(self.noccb)[None, None, :, None],
                    va[None, :, None, None], vb[None, :, None, None],
                    numpy.arange(self.nvirb)[None, None, None, :],
                ]
            chunks.append(result.ravel())
        else:
            chunks.append(numpy.zeros(0))
        # The generic helper preserves the untouched alpha axes; gather the
        # independent beta pairs explicitly for their packed ordering.
        if self.sizes["t3abb"]:
            result = numpy.zeros((self.opb[0].size, self.vpb[0].size,
                                  self.nocca, self.nvira))
            for oj, ok, vb, vc, sign in (
                    (self.opb[0], self.opb[1], self.vpb[0], self.vpb[1], 1.0),
                    (self.opb[1], self.opb[0], self.vpb[0], self.vpb[1], -1.0),
                    (self.opb[0], self.opb[1], self.vpb[1], self.vpb[0], -1.0),
                    (self.opb[1], self.opb[0], self.vpb[1], self.vpb[0], 1.0)):
                result += sign * t3[2][
                    numpy.arange(self.nocca)[None, None, :, None],
                    oj[:, None, None, None], ok[:, None, None, None],
                    numpy.arange(self.nvira)[None, None, None, :],
                    vb[None, :, None, None], vc[None, :, None, None],
                ]
            chunks.append(result.ravel())
        else:
            chunks.append(numpy.zeros(0))
        return numpy.concatenate(chunks)

    def set_values(self, t1, t2, t3, values):
        new1, new2, new3 = self.expand_vector(values)
        for target, source in zip(t1, new1):
            target[:] = source
        for target, source in zip(t2, new2):
            target[:] = source
        # Pure-spin triples are not coordinates and must not be changed.
        t3[1][:] = new3[1]
        t3[2][:] = new3[2]

    def packed_indices_uccsdt(self, t1):
        nocca, nvira = t1[0].shape
        noccb, nvirb = t1[1].shape
        if (nocca, noccb, nvira, nvirb) != (
                self.nocca, self.noccb, self.nvira, self.nvirb):
            raise ValueError("UCISDT coordinates do not match UCCSDT amplitudes")
        sizes = {
            "t1a": nocca * nvira,
            "t1b": noccb * nvirb,
            "t2aa": comb(nocca, 2) * comb(nvira, 2),
            "t2ab": nocca * nvira * noccb * nvirb,
            "t2bb": comb(noccb, 2) * comb(nvirb, 2),
            "t3aaa": comb(nocca, 3) * comb(nvira, 3),
            "t3aab": comb(nocca, 2) * comb(nvira, 2) * noccb * nvirb,
            "t3abb": comb(noccb, 2) * comb(nvirb, 2) * nocca * nvira,
            "t3bbb": comb(noccb, 3) * comb(nvirb, 3),
        }
        offsets = {}
        position = 0
        for name in ("t1a", "t1b", "t2aa", "t2ab", "t2bb",
                     "t3aaa", "t3aab", "t3abb", "t3bbb"):
            offsets[name] = position
            position += sizes[name]
        t2ab = numpy.arange(sizes["t2ab"], dtype=numpy.int64).reshape(
            nocca, nvira, noccb, nvirb
        ).transpose(0, 2, 1, 3).ravel()
        flat = numpy.concatenate((
            offsets["t1a"] + numpy.arange(sizes["t1a"]),
            offsets["t1b"] + numpy.arange(sizes["t1b"]),
            offsets["t2aa"] + numpy.arange(sizes["t2aa"]),
            offsets["t2ab"] + t2ab,
            offsets["t2bb"] + numpy.arange(sizes["t2bb"]),
            offsets["t3aab"] + numpy.arange(sizes["t3aab"]),
            offsets["t3abb"] + numpy.arange(sizes["t3abb"]),
        )).astype(numpy.int64, copy=False)
        return flat, numpy.ones(flat.size)


def _add_product_tangent(target, subscripts, values, tangents, scale=1.0):
    for position, tangent in enumerate(tangents):
        trial = list(values)
        trial[position] = tangent
        target += scale * numpy.einsum(subscripts, *trial, optimize=True)


def _add_product_adjoint(targets, subscripts, gradient, values, scale=1.0):
    inputs, output = subscripts.split("->")
    labels = inputs.split(",")
    for position, target in enumerate(targets):
        others = [value for i, value in enumerate(values) if i != position]
        other_labels = [label for i, label in enumerate(labels) if i != position]
        expression = ",".join([output] + other_labels) + "->" + labels[position]
        target += scale * numpy.einsum(
            expression, gradient, *others, optimize=True
        )


def _coefficient_values(t1, t2, t3):
    a, b = t1
    aa, ab, bb = t2
    caa = aa + numpy.einsum("ia,jb->ijab", a, a, optimize=True)
    caa -= numpy.einsum("ib,ja->ijab", a, a, optimize=True)
    cab = ab + numpy.einsum("ia,JB->iJaB", a, b, optimize=True)
    cbb = bb + numpy.einsum("JB,KC->JKBC", b, b, optimize=True)
    cbb -= numpy.einsum("JC,KB->JKBC", b, b, optimize=True)

    caab = numpy.array(t3[1], copy=True)
    terms = (
        (1.0, "ijab,KC->ijKabC", (aa, b)),
        (1.0, "iKaC,jb->ijKabC", (ab, a)),
        (-1.0, "jKaC,ib->ijKabC", (ab, a)),
        (-1.0, "iKbC,ja->ijKabC", (ab, a)),
        (1.0, "jKbC,ia->ijKabC", (ab, a)),
        (1.0, "ia,jb,KC->ijKabC", (a, a, b)),
        (-1.0, "ib,ja,KC->ijKabC", (a, a, b)),
    )
    for scale, subscripts, values in terms:
        caab += scale * numpy.einsum(subscripts, *values, optimize=True)

    cabb = numpy.array(t3[2], copy=True)
    terms = (
        (1.0, "ia,JKBC->iJKaBC", (a, bb)),
        (1.0, "iJaB,KC->iJKaBC", (ab, b)),
        (-1.0, "iKaB,JC->iJKaBC", (ab, b)),
        (-1.0, "iJaC,KB->iJKaBC", (ab, b)),
        (1.0, "iKaC,JB->iJKaBC", (ab, b)),
        (1.0, "ia,JB,KC->iJKaBC", (a, b, b)),
        (-1.0, "ia,JC,KB->iJKaBC", (a, b, b)),
    )
    for scale, subscripts, values in terms:
        cabb += scale * numpy.einsum(subscripts, *values, optimize=True)
    return a, b, caa, cab, cbb, caab, cabb


def _coefficient_tangents(t1, t2, t3, tangent):
    a, b = t1
    aa, ab, bb = t2
    xa, xb = tangent[0]
    xaa, xab, xbb = tangent[1]
    xcaa = numpy.array(xaa, copy=True)
    _add_product_tangent(xcaa, "ia,jb->ijab", (a, a), (xa, xa))
    _add_product_tangent(xcaa, "ib,ja->ijab", (a, a), (xa, xa), -1.0)
    xcab = numpy.array(xab, copy=True)
    _add_product_tangent(xcab, "ia,JB->iJaB", (a, b), (xa, xb))
    xcbb = numpy.array(xbb, copy=True)
    _add_product_tangent(xcbb, "JB,KC->JKBC", (b, b), (xb, xb))
    _add_product_tangent(xcbb, "JC,KB->JKBC", (b, b), (xb, xb), -1.0)

    xcaab = numpy.array(tangent[2][1], copy=True)
    terms = (
        (1.0, "ijab,KC->ijKabC", (aa, b), (xaa, xb)),
        (1.0, "iKaC,jb->ijKabC", (ab, a), (xab, xa)),
        (-1.0, "jKaC,ib->ijKabC", (ab, a), (xab, xa)),
        (-1.0, "iKbC,ja->ijKabC", (ab, a), (xab, xa)),
        (1.0, "jKbC,ia->ijKabC", (ab, a), (xab, xa)),
        (1.0, "ia,jb,KC->ijKabC", (a, a, b), (xa, xa, xb)),
        (-1.0, "ib,ja,KC->ijKabC", (a, a, b), (xa, xa, xb)),
    )
    for scale, subscripts, values, tangents in terms:
        _add_product_tangent(xcaab, subscripts, values, tangents, scale)

    xcabb = numpy.array(tangent[2][2], copy=True)
    terms = (
        (1.0, "ia,JKBC->iJKaBC", (a, bb), (xa, xbb)),
        (1.0, "iJaB,KC->iJKaBC", (ab, b), (xab, xb)),
        (-1.0, "iKaB,JC->iJKaBC", (ab, b), (xab, xb)),
        (-1.0, "iJaC,KB->iJKaBC", (ab, b), (xab, xb)),
        (1.0, "iKaC,JB->iJKaBC", (ab, b), (xab, xb)),
        (1.0, "ia,JB,KC->iJKaBC", (a, b, b), (xa, xb, xb)),
        (-1.0, "ia,JC,KB->iJKaBC", (a, b, b), (xa, xb, xb)),
    )
    for scale, subscripts, values, tangents in terms:
        _add_product_tangent(xcabb, subscripts, values, tangents, scale)
    return xa, xb, xcaa, xcab, xcbb, xcaab, xcabb


def _raised_values(coefficients, blocks, include_constant=True):
    a, b, aa, ab, bb, aab, abb = coefficients
    soo, sov, svo, svv = blocks
    one = numpy.array(svo.T if include_constant else numpy.zeros_like(svo.T))
    one -= numpy.einsum("iJ,ia->Ja", soo, a, optimize=True)
    one += numpy.einsum("aB,JB->Ja", svv, b, optimize=True)
    one -= numpy.einsum("iB,iJaB->Ja", sov, ab, optimize=True)

    raa = numpy.einsum("aJ,ib->iJab", svo, a, optimize=True)
    raa -= numpy.einsum("bJ,ia->iJab", svo, a, optimize=True)
    raa -= numpy.einsum("jJ,jiab->iJab", soo, aa, optimize=True)
    raa += numpy.einsum("aC,iJbC->iJab", svv, ab, optimize=True)
    raa -= numpy.einsum("bC,iJaC->iJab", svv, ab, optimize=True)
    raa -= numpy.einsum("jC,jiJabC->iJab", sov, aab, optimize=True)

    rbb = numpy.einsum("aJ,KB->JKaB", svo, b, optimize=True)
    rbb -= numpy.einsum("aK,JB->JKaB", svo, b, optimize=True)
    rbb -= numpy.einsum("iJ,iKaB->JKaB", soo, ab, optimize=True)
    rbb += numpy.einsum("iK,iJaB->JKaB", soo, ab, optimize=True)
    rbb += numpy.einsum("aC,JKCB->JKaB", svv, bb, optimize=True)
    rbb -= numpy.einsum("iC,iJKaCB->JKaB", sov, abb, optimize=True)
    return one, raa, rbb


def _raised_adjoint(weights, blocks, shapes):
    one, waa, wbb = weights
    soo, sov, svo, svv = blocks
    ga = numpy.zeros(shapes[0])
    gb = numpy.zeros(shapes[1])
    gaa = numpy.zeros(shapes[2])
    gab = numpy.zeros(shapes[3])
    gbb = numpy.zeros(shapes[4])
    gaab = numpy.zeros(shapes[5])
    gabb = numpy.zeros(shapes[6])

    ga -= numpy.einsum("Ja,iJ->ia", one, soo, optimize=True)
    gb += numpy.einsum("Ja,aB->JB", one, svv, optimize=True)
    gab -= numpy.einsum("Ja,iB->iJaB", one, sov, optimize=True)

    ga += numpy.einsum("iJab,aJ->ib", waa, svo, optimize=True)
    ga -= numpy.einsum("iJab,bJ->ia", waa, svo, optimize=True)
    gaa -= numpy.einsum("iJab,jJ->jiab", waa, soo, optimize=True)
    gab += numpy.einsum("iJab,aC->iJbC", waa, svv, optimize=True)
    gab -= numpy.einsum("iJab,bC->iJaC", waa, svv, optimize=True)
    gaab -= numpy.einsum("iJab,jC->jiJabC", waa, sov, optimize=True)

    gb += numpy.einsum("JKaB,aJ->KB", wbb, svo, optimize=True)
    gb -= numpy.einsum("JKaB,aK->JB", wbb, svo, optimize=True)
    gab -= numpy.einsum("JKaB,iJ->iKaB", wbb, soo, optimize=True)
    gab += numpy.einsum("JKaB,iK->iJaB", wbb, soo, optimize=True)
    gbb += numpy.einsum("JKaB,aC->JKCB", wbb, svv, optimize=True)
    gabb -= numpy.einsum("JKaB,iC->iJKaCB", wbb, sov, optimize=True)
    return ga, gb, gaa, gab, gbb, gaab, gabb


def _coefficient_adjoint(t1, t2, coefficient_gradients):
    a, b = t1
    aa, ab, bb = t2
    gca, gcb, gcaa, gcab, gcbb, gaab, gabb = coefficient_gradients
    ga = numpy.array(gca, copy=True)
    gb = numpy.array(gcb, copy=True)
    gaa = numpy.array(gcaa, copy=True)
    gab = numpy.array(gcab, copy=True)
    gbb = numpy.array(gcbb, copy=True)
    gtaab = numpy.array(gaab, copy=True)
    gtabb = numpy.array(gabb, copy=True)

    terms = (
        (1.0, "ijab,KC->ijKabC", gaab, (aa, b), (gaa, gb)),
        (1.0, "iKaC,jb->ijKabC", gaab, (ab, a), (gab, ga)),
        (-1.0, "jKaC,ib->ijKabC", gaab, (ab, a), (gab, ga)),
        (-1.0, "iKbC,ja->ijKabC", gaab, (ab, a), (gab, ga)),
        (1.0, "jKbC,ia->ijKabC", gaab, (ab, a), (gab, ga)),
        (1.0, "ia,jb,KC->ijKabC", gaab, (a, a, b), (ga, ga, gb)),
        (-1.0, "ib,ja,KC->ijKabC", gaab, (a, a, b), (ga, ga, gb)),
        (1.0, "ia,JKBC->iJKaBC", gabb, (a, bb), (ga, gbb)),
        (1.0, "iJaB,KC->iJKaBC", gabb, (ab, b), (gab, gb)),
        (-1.0, "iKaB,JC->iJKaBC", gabb, (ab, b), (gab, gb)),
        (-1.0, "iJaC,KB->iJKaBC", gabb, (ab, b), (gab, gb)),
        (1.0, "iKaC,JB->iJKaBC", gabb, (ab, b), (gab, gb)),
        (1.0, "ia,JB,KC->iJKaBC", gabb, (a, b, b), (ga, gb, gb)),
        (-1.0, "ia,JC,KB->iJKaBC", gabb, (a, b, b), (ga, gb, gb)),
    )
    for scale, subscripts, gradient, values, targets in terms:
        _add_product_adjoint(targets, subscripts, gradient, values, scale)

    # Only the incoming C2 adjoints pass through the disconnected C2 terms.
    # The T2 gradients accumulated above from C3 are already cluster gradients.
    _add_product_adjoint((ga, ga), "ia,jb->ijab", gcaa, (a, a))
    _add_product_adjoint((ga, ga), "ib,ja->ijab", gcaa, (a, a), -1.0)
    _add_product_adjoint((ga, gb), "ia,JB->iJaB", gcab, (a, b))
    _add_product_adjoint((gb, gb), "JB,KC->JKBC", gcbb, (b, b))
    _add_product_adjoint((gb, gb), "JC,KB->JKBC", gcbb, (b, b), -1.0)
    return (ga, gb), (gaa, gab, gbb), (gtaab, gtabb)


class UCISDTSpinConstraint(UCISDSpinConstraint):
    """CISDT-closed highest-weight equations for arbitrary UHF orbitals."""

    amplitude_rank = 3
    derivative_rank = 2

    def _is_retained(self, output):
        """The geometry-independent rank-one and rank-two closed subset."""
        return self._excitation_rank(output) <= 2

    def _candidate_outputs(self):
        raised = dict(delta_alpha=1, delta_beta=-1)
        retained = ()
        for rank in (1, 2):
            for holes_a in range(rank):
                retained += self._determinant_shell(
                    holes_a, rank - holes_a, **raised
                )
        return retained, self.ncandidate

    def _compile_constraints(self):
        nocca, noccb = self.nocc
        nvira, nvirb = self.nmo - nocca, self.nmo - noccb
        self._soo = self.overlap[:nocca, :noccb]
        self._sov = self.overlap[:nocca, noccb:]
        self._svo = self.overlap[nocca:, :noccb]
        self._svv = self.overlap[nocca:, noccb:]
        self._row_phases = 1.0 - 2.0 * (numpy.arange(noccb) % 2)
        self.coordinates = _UCISDTCoordinates(self.nmo, self.nocc)
        self._vapairs = _pairs(nvira)
        self._obpairs = _pairs(noccb)
        self._nrank1 = noccb * nvira
        self._nrank2aa = nocca * noccb * comb(nvira, 2)
        self._nrank2bb = nvira * comb(noccb, 2) * nvirb
        self.nconstraint = self._nrank1 + self._nrank2aa + self._nrank2bb
        def choose(n, k):
            return comb(n, k) if 0 <= k <= n else 0

        def output_shell(holes_a, holes_b):
            return (choose(nocca, holes_a) * choose(nvira, holes_a + 1)
                    * choose(noccb, holes_b) * choose(nvirb, holes_b - 1))

        self.ncandidate = sum(
            output_shell(holes_a, rank - holes_a)
            for rank in range(1, 5) for holes_a in range(rank)
        )
        self.nomitted = self.ncandidate - self.nconstraint
        self.ndeterminant = sum(
            choose(nocca, holes_a) * choose(nvira, holes_a)
            * choose(noccb, holes_b) * choose(nvirb, holes_b)
            for holes_a in range(4) for holes_b in range(4 - holes_a)
        )
        self.constant = self._pack_rows(
            self._svo.T,
            numpy.zeros((nocca, noccb, nvira, nvira)),
            numpy.zeros((noccb, noccb, nvira, nvirb)),
        )
        self.build_time = getattr(self, "build_time", 0.0)

    def __init__(self, nmo, nocc, overlap, frozen_coupling=0.0):
        start = perf_counter()
        super().__init__(nmo, nocc, overlap, frozen_coupling=frozen_coupling)
        self.build_time = perf_counter() - start

    def _blocks(self):
        return self._soo, self._sov, self._svo, self._svv

    def _pack_rows(self, one, raa, rbb):
        first = self._format_rows(one)
        va, vb = self._vapairs
        second = (raa[:, :, va, vb].ravel() if va.size
                  else numpy.zeros(0, dtype=numpy.asarray(raa).dtype))
        oj, ok = self._obpairs
        third = (rbb[oj, ok, :, :].ravel() if oj.size
                 else numpy.zeros(0, dtype=numpy.asarray(rbb).dtype))
        return numpy.concatenate((first, second, third))

    def _unpack_weights(self, weights):
        weights = numpy.asarray(weights, dtype=float)
        if weights.shape != (self.nconstraint,):
            raise ValueError("constraint weights have the wrong shape")
        stop1 = self._nrank1
        stop2 = stop1 + self._nrank2aa
        one = self._row_phases[:, None] * weights[:stop1].reshape(
            self.nocc[1], self.nmo - self.nocc[0]
        )[::-1]
        nocca, noccb = self.nocc
        nvira, nvirb = self.nmo - nocca, self.nmo - noccb
        raa = numpy.zeros((nocca, noccb, nvira, nvira))
        va, vb = self._vapairs
        if va.size:
            raa[:, :, va, vb] = weights[stop1:stop2].reshape(
                nocca, noccb, va.size
            )
        rbb = numpy.zeros((noccb, noccb, nvira, nvirb))
        oj, ok = self._obpairs
        if oj.size:
            rbb[oj, ok, :, :] = weights[stop2:].reshape(
                oj.size, nvira, nvirb
            )
        return one, raa, rbb

    def _check_amplitudes(self, t1, t2, t3):
        nocca, noccb = self.nocc
        nvira, nvirb = self.nmo - nocca, self.nmo - noccb
        expected = (
            (nocca, nvira), (noccb, nvirb),
            (nocca, nocca, nvira, nvira),
            (nocca, noccb, nvira, nvirb),
            (noccb, noccb, nvirb, nvirb),
            (nocca, nocca, noccb, nvira, nvira, nvirb),
            (nocca, noccb, noccb, nvira, nvirb, nvirb),
        )
        blocks = tuple(t1) + tuple(t2) + (t3[1], t3[2])
        for name, value, shape in zip(_BLOCKS, blocks, expected):
            if numpy.asarray(value).shape != shape:
                raise ValueError("%s has shape %s; expected %s"
                                 % (name, numpy.asarray(value).shape, shape))
            if numpy.iscomplexobj(value):
                raise ValueError("complex amplitudes are not supported")

    def amplitude_values(self, t1, t2, t3):
        return self.coordinates.amplitude_values(t1, t2, t3)

    def constraint(self, t1, t2, t3):
        self._check_amplitudes(t1, t2, t3)
        coefficients = _coefficient_values(t1, t2, t3)
        return self._pack_rows(*_raised_values(coefficients, self._blocks()))

    def jvp(self, t1, t2, t3, vector):
        self._check_amplitudes(t1, t2, t3)
        tangent = self.coordinates.expand_vector(vector)
        coefficients = _coefficient_tangents(t1, t2, t3, tangent)
        outputs = _raised_values(
            coefficients, self._blocks(), include_constant=False
        )
        return self._pack_rows(*outputs)

    def vjp(self, t1, t2, t3, weights):
        self._check_amplitudes(t1, t2, t3)
        shapes = tuple(block.shape for block in tuple(t1) + tuple(t2)
                       + (t3[1], t3[2]))
        coefficient_gradients = _raised_adjoint(
            self._unpack_weights(weights), self._blocks(), shapes
        )
        g1, g2, mixed3 = _coefficient_adjoint(
            t1, t2, coefficient_gradients
        )
        g3 = (numpy.zeros_like(t3[0]), mixed3[0], mixed3[1],
              numpy.zeros_like(t3[3]))
        return self.coordinates.pack_cotangent(g1, g2, g3)

    def jacobian(self, t1, t2, t3):
        size = self.nconstraint * len(self.coordinates)
        if size * 8 > 2 * 1024 ** 3:
            raise MemoryError(
                "dense UCISDT Jacobian would require %.1f GiB; use jvp/vjp"
                % (size * 8 / 1024 ** 3)
            )
        result = numpy.empty((self.nconstraint, len(self.coordinates)))
        basis = numpy.zeros(self.nconstraint)
        for row in range(self.nconstraint):
            basis[row] = 1.0
            result[row] = self.vjp(t1, t2, t3, basis)
            basis[row] = 0.0
        return result

    def weighted_gram_diagonal(self, t1, t2, t3, metric):
        """Exact diagonal for small problems; guarded against dense-scale use.

        The production multiplier solve only needs a positive preconditioner.
        Until the closed-form diagonal contractions are implemented, use the
        exact dense diagnostic where it is affordable and an overlap-based
        direct-cluster diagonal otherwise.
        """
        metric = numpy.asarray(metric, dtype=float)
        if metric.shape != (len(self.coordinates),):
            raise ValueError("Gram metric has the wrong shape")
        size = self.nconstraint * len(self.coordinates)
        if size * 8 <= 256 * 1024 ** 2:
            jacobian = self.jacobian(t1, t2, t3)
            return numpy.einsum("ri,i,ri->r", jacobian, metric, jacobian)
        return self._direct_gram_diagonal(metric)

    def _direct_gram_diagonal(self, metric):
        """Positive diagonal from the direct C1/C2/C3 cluster derivatives."""
        x1, x2, x3 = self.coordinates.expand_vector(metric)
        x1 = tuple(numpy.abs(block) for block in x1)
        x2 = tuple(numpy.abs(block) for block in x2)
        x3 = tuple(numpy.abs(block) for block in x3)
        m1a, m1b = x1
        m2aa, m2ab, m2bb = x2
        m3aab, m3abb = x3[1], x3[2]
        one = numpy.einsum("iJ,ia->Ja", self._soo ** 2, m1a)
        one += numpy.einsum("aB,JB->Ja", self._svv ** 2, m1b)
        one += numpy.einsum("iB,iJaB->Ja", self._sov ** 2, m2ab)
        aa = numpy.einsum("aJ,ib->iJab", self._svo ** 2, m1a)
        aa += numpy.einsum("bJ,ia->iJab", self._svo ** 2, m1a)
        aa += numpy.einsum("jJ,jiab->iJab", self._soo ** 2, m2aa)
        aa += numpy.einsum("aC,iJbC->iJab", self._svv ** 2, m2ab)
        aa += numpy.einsum("bC,iJaC->iJab", self._svv ** 2, m2ab)
        aa += numpy.einsum("jC,jiJabC->iJab", self._sov ** 2, m3aab)
        bb = numpy.einsum("aJ,KB->JKaB", self._svo ** 2, m1b)
        bb += numpy.einsum("aK,JB->JKaB", self._svo ** 2, m1b)
        bb += numpy.einsum("iJ,iKaB->JKaB", self._soo ** 2, m2ab)
        bb += numpy.einsum("iK,iJaB->JKaB", self._soo ** 2, m2ab)
        bb += numpy.einsum("aC,JKCB->JKaB", self._svv ** 2, m2bb)
        bb += numpy.einsum("iC,iJKaCB->JKaB", self._sov ** 2, m3abb)
        diagonal = numpy.abs(self._pack_rows(one, aa, bb))
        floor = numpy.finfo(float).eps * max(
            float(numpy.max(diagonal, initial=0.0)), 1.0
        )
        return numpy.maximum(diagonal, floor)

    def matrix_free_weighted_gram_diagonal(self, t1, t2, t3, metric):
        """Tensor-only Krylov preconditioner without a dense Jacobian.

        This keeps the direct cluster derivatives and deliberately omits the
        amplitude-dependent disconnected contributions.  It is therefore a
        positive preconditioner, not the exact Gram diagonal; the constraint,
        JVP, and VJP used by the multiplier equations remain exact.
        """
        self._check_amplitudes(t1, t2, t3)
        metric = numpy.asarray(metric, dtype=float)
        if metric.shape != (len(self.coordinates),):
            raise ValueError("Gram metric has the wrong shape")
        return self._direct_gram_diagonal(metric)

    def matrix_free_jacobian_row_norms(self, t1, t2, t3):
        """Row scaling paired with the tensor-only Gram preconditioner."""
        return numpy.sqrt(self.matrix_free_weighted_gram_diagonal(
            t1, t2, t3, numpy.ones(len(self.coordinates))
        ))

    def jacobian_row_norms(self, t1, t2, t3):
        return numpy.sqrt(self.weighted_gram_diagonal(
            t1, t2, t3, numpy.ones(len(self.coordinates))
        ))


__all__ = ["UCISDTSpinConstraint"]

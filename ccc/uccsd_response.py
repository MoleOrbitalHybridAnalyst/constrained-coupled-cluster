"""Unconstrained CCSD residual and exact packed-coordinate derivative actions."""
import numpy as np
from pyscf.cc import uccsd, uccsd_lambda
from .uccsd_constr import _checked_denominators
from ._uccsd_tangent import update_jvp


class UCCSDResponse:
    """One linearization; orbital and integral data are held fixed.

    The transpose includes the Lambda2/R1 correction away from a CC root.
    Lambda's affine energy-gradient term is subtracted before returning J^T v.
    """
    def __init__(self, cc, t1, t2, eris, transpose=True):
        if any(np.iscomplexobj(a) for a in tuple(t1) + tuple(t2)):
            raise NotImplementedError('Newton CCSD currently supports real amplitudes')
        self.cc, self.t1, self.t2, self.eris = cc, t1, t2, eris
        self.denominators = _checked_denominators(cc, t1, t2, eris)
        self.x = cc.amplitudes_to_vector(t1, t2)
        a, b = uccsd.update_amps(cc, t1, t2, eris)
        self.residual = self.denominators * (cc.amplitudes_to_vector(a, b) - self.x)
        self.residual1 = cc.vector_to_amplitudes(self.residual)[0]
        self.imds = uccsd_lambda.make_intermediates(cc, t1, t2, eris) if transpose else None
        self.rho0 = self._rho(np.zeros_like(self.x)) if transpose else None

    def jvp(self, vector):
        v1, v2 = self.cc.vector_to_amplitudes(vector)
        a, b = update_jvp(self.cc, self.t1, self.t2, v1, v2, self.eris)
        return self.denominators * (self.cc.amplitudes_to_vector(a, b) - vector)

    def _rho(self, vector):
        cc = self.cc
        a, b = cc.vector_to_amplitudes(vector)
        na, nb = uccsd_lambda.update_lambda(cc, self.t1, self.t2, a, b, self.eris, self.imds)
        rho = self.denominators * (cc.amplitudes_to_vector(na, nb) - vector)
        r1a, r1b = self.residual1
        ca = (np.einsum('ijab,jb->ia', b[0], r1a)
              + np.einsum('iJaB,JB->ia', b[1], r1b))
        cb = (np.einsum('IJAB,JB->IA', b[2], r1b)
              + np.einsum('jIbA,jb->IA', b[1], r1a))
        rho -= cc.amplitudes_to_vector((ca, cb), tuple(np.zeros_like(x) for x in b))
        return rho

    def vjp(self, vector):
        if self.imds is None:
            raise RuntimeError('transpose intermediates were not requested')
        return self._rho(vector) - self.rho0

    def close(self):
        # PySCF's response intermediates can own an HDF5 temporary file.
        if self.imds is not None:
            handle = getattr(self.imds, 'ftmp', None)
            if handle is not None:
                handle.close()
            self.imds = None

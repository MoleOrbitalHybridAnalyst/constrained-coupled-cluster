"""Independent finite-difference and adjoint checks away from CC roots."""
import numpy as np
from pyscf import gto, scf
from pyscf.cc import uccsd
from ccc.uccsd_response import UCCSDResponse

rng = np.random.default_rng(17)
for spin in [0, 2]:
    mol = gto.M(atom='H 0 0 0; H 0 0 1.2; H 0 0 2.4; H 0 0 3.6',
                basis='sto-3g', spin=spin, verbose=0)
    mf = scf.UHF(mol).run(conv_tol=1e-12)
    cc = uccsd.UCCSD(mf)
    cc.level_shift = .013
    eris = cc.ao2mo()
    _, a, b = cc.init_amps(eris)
    x = cc.amplitudes_to_vector(a, b)
    x += .015*rng.normal(size=x.size)
    a, b = cc.vector_to_amplitudes(x)
    saved = [z.copy() for z in a+b]
    response = UCCSDResponse(cc, a, b, eris)
    v = rng.normal(size=len(x));v /= np.linalg.norm(v)
    w = rng.normal(size=len(x));w /= np.linalg.norm(w)
    h = 2e-5
    plus = UCCSDResponse(cc, *cc.vector_to_amplitudes(x+h*v), eris, transpose=False)
    minus = UCCSDResponse(cc, *cc.vector_to_amplitudes(x-h*v), eris, transpose=False)
    numeric = (plus.residual-minus.residual)/(2*h)
    analytic = response.jvp(v)
    error = np.max(abs(numeric-analytic))
    adjoint = abs(w @ analytic-response.vjp(w) @ v)
    cc.level_shift = 0
    unshifted = UCCSDResponse(cc,a,b,eris,transpose=False)
    shift_error = np.max(abs(unshifted.residual-response.residual))
    cc.level_shift = .013
    print(spin, 'forward error',error,'adjoint error',adjoint,'shift error',shift_error)
    assert error < 1e-8
    assert adjoint < 1e-9
    assert shift_error < 1e-10
    for before,after in zip(saved,a+b):np.testing.assert_array_equal(before,after)
    response.close()
print('CCSD response action checks passed')

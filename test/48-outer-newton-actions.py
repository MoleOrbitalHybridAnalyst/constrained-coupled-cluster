"""Coupled adjoints, matrix-free damped steps, and integration/failure checks."""
from dataclasses import asdict
import numpy as np
from scipy.sparse.linalg import lsmr
from pyscf import gto, scf
from pyscf.cc import uccsd
from ccc import MultiConstrUCCSD
from ccc.ucisd_spin_constraint import UCISDSpinConstraint
from ccc.constraint_newton import (ConstraintActions, coupled_operator,
                                     NewtonOptions, _residual)
from ccc.uccsd_response import UCCSDResponse

rng=np.random.default_rng(62)
mol=gto.M(atom='H 0 0 0; H 0 0 1.4; H 0 0 2.8; H 0 0 4.2',basis='sto-3g',verbose=0)
mf=scf.UHF(mol).run()
plain=uccsd.UCCSD(mf).run()
engine=UCISDSpinConstraint.from_cc(plain)
cc=MultiConstrUCCSD(mf,engine.constraint,engine.jacobian,engine.coordinates,
                    constraint_tol=1e-9,constraint_max_step=.1)
eris=cc.ao2mo()
x=cc.amplitudes_to_vector(plain.t1,plain.t2)+.01*rng.normal(size=plain.vector_size())
cc.t1,cc.t2=cc.vector_to_amplitudes(x)
c=ConstraintActions(cc,cc.t1,cc.t2)
r=UCCSDResponse(cc,cc.t1,cc.t2,eris)
eta=rng.normal(size=len(c.value))
A=coupled_operator(r,c,eta)
v=rng.normal(size=A.shape[1]);w=rng.normal(size=A.shape[0])
assert abs(w@A.matvec(v)-A.rmatvec(w)@v)<1e-8
v/=np.linalg.norm(v)
h=2e-5
numeric=(_residual(cc,x+h*v[:len(x)],eta+h*v[len(x):],eris)
         -_residual(cc,x-h*v[:len(x)],eta-h*v[len(x):],eris))/(2*h)
np.testing.assert_allclose(A.matvec(v),numeric,atol=2e-8,rtol=2e-8)
matrix=np.column_stack([A.matvec(e) for e in np.eye(A.shape[1])])
b=rng.normal(size=A.shape[0]);damp=.03
expected=np.linalg.lstsq(np.vstack((matrix,damp*np.eye(len(b)))),np.r_[b,np.zeros(len(b))],rcond=None)[0]
actual=lsmr(A,b,damp=damp,atol=1e-13,btol=1e-13,maxiter=1000)[0]
np.testing.assert_allclose(actual,expected,atol=1e-7,rtol=1e-7)
r.close()
# Explicit failure at a bounded outer budget; final stored values are accepted.
cc.constraint_multipliers=eta
failed=cc.polish_newton(eris,{'max_cycle':1,'krylov_max_cycle':1,'residual_tol':1e-14,'constraint_tol':1e-14})
assert not failed.success and not cc.converged
assert np.all(np.isfinite(cc.amplitudes_to_vector(cc.t1,cc.t2)))
# Missing derivatives are rejected, not approximated silently.
cc.constraint_hessian_vector=None
try:cc.polish_newton(eris)
except NotImplementedError:pass
else:raise AssertionError('missing Hessian was not rejected')
for invalid in [{'max_cycle':0},{'krylov_tol':float('nan')},{'unknown_option':1}]:
 try:NewtonOptions.parse(invalid,cc)
 except (ValueError,TypeError):pass
 else:raise AssertionError(invalid)
# Existing kernel remains the default; automatic polishing is explicitly enabled.
cc=MultiConstrUCCSD(mf,engine.constraint,engine.jacobian,engine.coordinates,
                    constraint_tol=1e-9,constraint_max_step=.1)
assert cc.newton_polish is False
cc.newton_polish=True
cc.newton_options={'backend':'dense'}
cc.conv_tol=1e-10
cc.conv_tol_normt=1e-7
cc.kernel(t1=plain.t1,t2=plain.t2,eris=eris)
assert cc.newton_report.success and cc.converged
print('Coupled derivative, linear solve, failure, and opt-in integration checks passed')

"""Numerical checks for the blocked real Lambda-UCCSD(T) kernel."""

import numpy
from pyscf import gto, lib, scf
from pyscf.cc import uccsd, uccsd_t

import ccc
from ccc import uccsd_lambdat_slow


mol = gto.M(
    atom="O 0 0 0; H 0 0 0.97",
    basis="6-31g",
    spin=1,
    verbose=0,
)
mf = scf.UHF(mol).run(conv_tol=1e-11)
cc = uccsd.UCCSD(mf).run(conv_tol=1e-10)
eris = cc.ao2mo()
l1, l2 = cc.solve_lambda(eris=eris)
assert cc.converged and cc.converged_lambda

inputs = cc.t1 + cc.t2 + l1 + l2
before = tuple(block.copy() for block in inputs)

fast = ccc.uccsd_lambdat(
    cc, eris=eris, t1=cc.t1, t2=cc.t2, l1=l1, l2=l2, verbose=0)
slow = uccsd_lambdat_slow.kernel(cc, eris, cc.t1, cc.t2, l1, l2)
assert abs(fast - slow) < 1e-11

# Lambda=T must reduce exactly to the historical UCCSD(T) contraction.  This
# also exercises the case where the mixed-spin right and left buffers alias.
lambda_equals_t = ccc.uccsd_lambdat(
    cc, eris=eris, t1=cc.t1, t2=cc.t2,
    l1=cc.t1, l2=cc.t2, verbose=0)
standard = uccsd_t.kernel(cc, eris, cc.t1, cc.t2, verbose=0)
assert abs(lambda_equals_t - standard) < 1e-11

# A deliberately different left state catches kernels that accidentally use T
# for both sides while retaining the proper same-spin antisymmetry.
scaled_l1 = tuple((0.8 + 0.1 * spin) * block for spin, block in enumerate(l1))
scaled_l2 = tuple((0.7 + 0.1 * spin) * block for spin, block in enumerate(l2))
fast_scaled = ccc.uccsd_lambdat(
    cc, eris=eris, t1=cc.t1, t2=cc.t2,
    l1=scaled_l1, l2=scaled_l2, verbose=0)
slow_scaled = uccsd_lambdat_slow.kernel(
    cc, eris, cc.t1, cc.t2, scaled_l1, scaled_l2)
assert abs(fast_scaled - slow_scaled) < 1e-11
assert abs(fast_scaled - standard) > 1e-8

# Read-only caller data takes the copy-based mixed-spin path.
readonly_l2 = tuple(block.copy() for block in l2)
readonly_l2[1].flags.writeable = False
fast_readonly = ccc.uccsd_lambdat(
    cc, eris=eris, t1=cc.t1, t2=cc.t2,
    l1=l1, l2=readonly_l2, verbose=0)
assert abs(fast_readonly - slow) < 1e-11

old_threads = lib.num_threads()
try:
    lib.num_threads(2)
    threaded = ccc.uccsd_lambdat(
        cc, eris=eris, t1=cc.t1, t2=cc.t2, l1=l1, l2=l2, verbose=0)
finally:
    lib.num_threads(old_threads)
assert abs(threaded - fast) < 1e-11

for old, new in zip(before, inputs):
    assert numpy.array_equal(old, new)

try:
    ccc.uccsd_lambdat(cc, eris=eris, l1=l1, verbose=0)
except ValueError:
    pass
else:
    raise AssertionError("missing l2 was accepted")

complex_l1 = (l1[0].astype(complex), l1[1])
try:
    ccc.uccsd_lambdat(
        cc, eris=eris, l1=complex_l1, l2=l2, verbose=0)
except NotImplementedError:
    pass
else:
    raise AssertionError("complex Lambda amplitudes were accepted")

# The same-spin-only early return must handle an empty beta occupied space.
h_atom = gto.M(atom="H 0 0 0", basis="6-31g", spin=1, verbose=0)
h_mf = scf.UHF(h_atom).run()
h_cc = uccsd.UCCSD(h_mf).run()
h_eris = h_cc.ao2mo()
h_l1, h_l2 = h_cc.solve_lambda(eris=h_eris)
assert ccc.uccsd_lambdat(
    h_cc, eris=h_eris, l1=h_l1, l2=h_l2, verbose=0) == 0.0

print("OH optimized - slow Lambda-(T) = %.3e" % (fast - slow))
print("OH Lambda=T - UCCSD(T) = %.3e" % (lambda_equals_t - standard))

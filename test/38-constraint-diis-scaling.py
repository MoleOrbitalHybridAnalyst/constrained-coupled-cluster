"""Constraint DIIS must not amplify numerically inactive Jacobian rows."""
import sys
from types import SimpleNamespace

import numpy

from ccc.uccsd_multiconstr import _constraint_diis_error


cc = SimpleNamespace(
    constraint_tol=1e-6,
    constraint_diis_weight=2.0,
    verbose=0,
    stdout=sys.stdout,
)

root_epsilon = numpy.sqrt(numpy.finfo(float).eps)
constraint = numpy.asarray([0.4, 1e-15, 5e-7, -3e-7])
row_norms = numpy.asarray([2.0, 0.0, root_epsilon, 1e-6])
scaled = _constraint_diis_error(cc, constraint, row_norms)
expected = numpy.asarray([0.4, 0.0, 0.0, -0.6])
error = float(numpy.max(numpy.abs(scaled - expected)))
print("active scaling and inactive rows: max error %.2e" % error)
assert scaled.shape == constraint.shape
assert error < 1e-14

try:
    _constraint_diis_error(
        cc, numpy.asarray([1.1e-6]), numpy.asarray([root_epsilon])
    )
except RuntimeError as caught:
    assert "numerically inactive constraint row is violated" in str(caught)
else:
    raise AssertionError("a violated inactive constraint row was accepted")
print("violated inactive row: rejected")

for values, norms in (
        (numpy.asarray([0.0, numpy.nan]), numpy.ones(2)),
        (numpy.zeros(2), numpy.asarray([1.0, -1.0])),
        (numpy.zeros(2), numpy.ones(3))):
    try:
        _constraint_diis_error(cc, values, norms)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid constraint DIIS data were accepted")
print("invalid values, norms, and shapes: rejected")

print("all constraint DIIS scaling gates passed")

# ccc: constrained coupled cluster

Conventional single-reference coupled cluster (PySCF UCCSD/UCCSDT/GCCSD) with
user-supplied constraints `C(t) = 0` imposed through Lagrange multipliers,
`R(t) = G(t)^T eta`, plus spin-symmetry constraints, constrained Lambda
equations, a Lambda-based (T) correction, and a joint Newton solver.

## Layout

| Module | Contents |
| --- | --- |
| `uccsd_multiconstr` | `MultiConstrUCCSD`: constrained UCCSD, multiplier solvers, DIIS schemes, constrained Lambda |
| `uccsdt_multiconstr` | `MultiConstrUCCSDT`, `LowMemoryMultiConstrUCCSDT` |
| `uccsdt_lowm_spinconstr` | `LowMemorySpinConstrUCCSDT` |
| `gccsd_multiconstr` | constrained GCCSD |
| `uccsd_constr`, `uccsd_linconstr` | earlier single-constraint and linear-constraint UCCSD |
| `ucisd_spin_constraint`, `ucisdt_spin_constraint`, `cisd_spin_constraint`, `gcisd_spin_constraint`, `active_spin_constraint` | spin-constraint engines (CISD, CISDT, CSF/active-space) |
| `constraint_newton`, `uccsd_response`, `_uccsd_tangent` | joint Newton polishing and CCSD Jacobian actions |
| `cluster_decomposition` | T1/T2 from an unrestricted CI or CASCI vector (`cluster_decompose`, `cas_amplitudes`) |
| `uccsd_lambdat`, `uccsd_lambdat_slow` | Lambda-UCCSD(T) (C kernel in `ccc/lib`) and a reference implementation |

## Install

Requires PySCF, NumPy and SciPy.  Build the C kernel once, then put the
repository root on `PYTHONPATH`:

```bash
cd ccc/lib && mkdir -p build && cd build && cmake .. && make
export PYTHONPATH=/path/to/ccc:$PYTHONPATH
```

## Tests

Each file in `test/` is a standalone script: `python test/30-ucisd-spin-constraint.py`.

## Outer Newton polishing for constrained CCSD

`MultiConstrUCCSD` retains its existing multiplier-only Newton and
Gauss--Newton corrections inside each CC iteration.  An optional outer solver
can subsequently solve the amplitude and constraint equations together:

```python
report = cc.polish_newton(eris=eris, options={
    "residual_tol": 1e-9,
    "constraint_tol": 1e-10,
})
```

The joint residual is `(R - G.T @ eta, C)`.  `report.success` is true only
when both maximum norms pass their requested tolerances.  The default
`matrix_free` backend uses CCSD Jacobian-vector products, constraint JVP/VJP
and Hessian-vector products, row and column scaling, damped LSMR, and nonlinear
step acceptance.  It never forms the coupled Jacobian or a constraint Gram
matrix.  Constraints used with it must provide the existing matrix-free
derivative callbacks.  This large-system backend is experimental; callers
should inspect `report.success` and the recorded joint residuals.

The `dense` backend explicitly assembles the coupled Jacobian and applies a
rank-revealing SVD cutoff.  It is limited to at most 1024 coupled unknowns and
exists as a small-system reference and diagnostic implementation.

Automatic post-iteration polishing is opt-in:

```python
cc.newton_polish = True
cc.newton_options = {"residual_tol": 1e-9}
cc.kernel()
```

The original CC iteration remains the default.  If polishing changes the
amplitudes, any stored Lambda solution is invalidated and must be recomputed.

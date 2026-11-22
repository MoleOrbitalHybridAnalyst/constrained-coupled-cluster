"""Warm-starting the Lagrange-multiplier Krylov solve must not change the answer."""
import numpy
from pyscf import gto, scf
from pyscf.cc import uccsd

from ccc import MultiConstrUCCSD
from ccc.cisd_spin_constraint import ClosedCISDSpinConstraint
from ccc.uccsd_multiconstr import matrix_free_multipliers


def common_orbital_h4():
    # Common alpha/beta spin orbitals (RHF-derived) so ClosedCISDSpinConstraint's
    # exact-signed-permutation check passes -- a plain broken-symmetry UHF's
    # alpha/beta MOs miss that check by rounding noise well above its tolerance.
    mol = gto.M(
        atom=";".join("H 0 0 %.1f" % (1.2 * atom) for atom in range(4)),
        basis="sto-3g", spin=0, verbose=0,
    )
    rhf = scf.RHF(mol).run(verbose=0)
    mf = scf.UHF(mol)
    mf.mo_coeff = (rhf.mo_coeff.copy(), rhf.mo_coeff.copy())
    mf.mo_occ = (rhf.mo_occ / 2, rhf.mo_occ / 2)
    dm = mf.make_rdm1()
    fock = mf.get_fock(dm=dm)
    mf.mo_energy = tuple(numpy.diag(c.T @ f @ c)
                         for c, f in zip(mf.mo_coeff, fock))
    mf.e_tot = mf.energy_tot(dm=dm)
    mf.converged = True
    return mf


# ---------------------------------------------------------------- (a) synthetic
class _Namespace:
    pass


rng = numpy.random.default_rng(0)
n = 6
A = rng.standard_normal((n, n))
gram = A @ A.T + numpy.eye(n) * 0.5
diagonal = numpy.diag(gram).copy()
rhs = rng.standard_normal(n)


def gram_matvec(weights):
    return gram @ weights


def response(weights):
    return weights.copy()


cc = _Namespace()
cc.constraint_krylov_tol = 1e-13
cc.constraint_max_step = 0.0
cc.constraint_krylov_max_cycle = 2000
cc.verbose = 0

cold_multipliers, cold_correction = matrix_free_multipliers(
    cc, gram_matvec, response, rhs, diagonal, initial=None
)
cold_iterations = cc.constraint_krylov_iterations

warm_multipliers, warm_correction = matrix_free_multipliers(
    cc, gram_matvec, response, rhs, diagonal, initial=cold_multipliers
)
warm_iterations = cc.constraint_krylov_iterations

assert warm_iterations <= cold_iterations, (warm_iterations, cold_iterations)
assert numpy.max(numpy.abs(warm_multipliers - cold_multipliers)) < 1e-9

# A NaN or wrong-shape initial guess must fall back to the cold start silently
# (a warm start is a performance hint only, never a correctness input).
_, _ = matrix_free_multipliers(
    cc, gram_matvec, response, rhs, diagonal, initial=numpy.full(n, numpy.nan)
)
assert cc.constraint_krylov_iterations == cold_iterations

_, _ = matrix_free_multipliers(
    cc, gram_matvec, response, rhs, diagonal, initial=numpy.zeros(n + 1)
)
assert cc.constraint_krylov_iterations == cold_iterations

print("synthetic warm start: cold=%d iterations, warm=%d iterations"
      % (cold_iterations, warm_iterations))
print("bad initial guess: silently falls back to cold start")


# ------------------------------------------------------------ (b) integration
def track_krylov_iterations(cc, force_cold):
    iterations = []
    bound = type(cc).update_amps.__get__(cc)

    def wrapped(t1, t2, eris):
        if force_cold:
            cc.constraint_multipliers = None
        result = bound(t1, t2, eris)
        iterations.append(cc.constraint_krylov_iterations)
        return result

    cc.update_amps = wrapped
    return iterations


def configured(cc):
    cc.verbose = 0
    cc.max_cycle = 60
    cc.conv_tol = 1e-10
    cc.conv_tol_normt = 1e-8
    cc.diis_space = 8
    cc.diis_start_cycle = 2
    return cc


mf4 = common_orbital_h4()
engine = ClosedCISDSpinConstraint.from_cc(uccsd.UCCSD(mf4))

warm_started = configured(MultiConstrUCCSD.from_constraint_engine(mf4, engine))
cold_every_cycle = configured(MultiConstrUCCSD.from_constraint_engine(mf4, engine))

warm_iterations_per_cycle = track_krylov_iterations(warm_started, force_cold=False)
cold_iterations_per_cycle = track_krylov_iterations(cold_every_cycle, force_cold=True)

warm_started.run()
cold_every_cycle.run()

assert warm_started.converged and cold_every_cycle.converged
assert numpy.max(numpy.abs(warm_started.constraint_residual())) < warm_started.constraint_tol
assert numpy.max(numpy.abs(cold_every_cycle.constraint_residual())) < cold_every_cycle.constraint_tol
assert abs(warm_started.e_tot - cold_every_cycle.e_tot) < 1e-9

warm_total = sum(warm_iterations_per_cycle)
cold_total = sum(cold_iterations_per_cycle)
assert warm_total <= cold_total, (warm_total, cold_total)

print("integration warm start: warm total = %d, forced-cold total = %d"
      % (warm_total, cold_total))
print("all constraint multiplier warm-start checks passed")

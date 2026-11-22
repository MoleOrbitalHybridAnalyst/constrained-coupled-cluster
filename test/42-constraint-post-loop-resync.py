"""conv_check must run exactly one extra Newton/eta correction pass (no DIIS
after it) when the CC loop exits with amplitudes converged but the constraint
still violated, and must flag not converged if that one pass either fails to
satisfy the constraint or moves the energy/amplitudes far more than a small
cleanup step should."""
import numpy
from pyscf import gto, scf

from ccc import MultiConstrUCCSD, MultiConstrUCCSDT


def copied(t1, t2):
    return tuple(x.copy() for x in t1), tuple(x.copy() for x in t2)


def copied_tamps(tamps):
    return [tuple(block.copy() for block in rank) for rank in tamps]


def common_orbital_h4():
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


mf4 = common_orbital_h4()

coordinates = (
    ("t1a", (0, 0)),
    ("t1b", (0, 0)),
    ("t2ab", (0, 0, 0, 0)),
)


def constraint(t1, t2):
    x = t1[0][0, 0]
    y = t1[1][0, 0]
    z = t2[1][0, 0, 0, 0]
    return numpy.asarray([
        x + y + z + 5 * x**2 - 0.03,
        x - y + 3 * y**2 - 0.01,
    ])


def jacobian(t1, t2):
    x = t1[0][0, 0]
    y = t1[1][0, 0]
    return numpy.asarray([
        [1 + 10 * x, 1, 1],
        [1, -1 + 6 * y, 0],
    ])


def build(conv_check):
    cc = MultiConstrUCCSD(mf4, constraint, jacobian, coordinates,
                            conv_check=conv_check)
    cc.diis = False
    # Never runs .ccsd()/.kernel(), so e_hf (normally set inside the kernel)
    # must be supplied by hand for _finalize's e_tot property to work.
    cc.e_hf = mf4.e_tot
    eris = cc.ao2mo(cc.mo_coeff)
    return cc, eris


def count_calls(cc):
    calls = [0]
    original = cc.update_amps

    def counting(t1, t2, eris):
        calls[0] += 1
        return original(t1, t2, eris)

    cc.update_amps = counting
    return calls


def prepare():
    """Two real Newton/eta cycles from a fresh init guess: cycle 1 leaves the
    (nonlinear) constraint mildly violated, cycle 2 -- exactly what the Phase 3
    post-loop pass performs -- brings it much closer to satisfied.  Reused as a
    realistic, deterministic stand-in for "the main loop's convergence test
    used the pre-DIIS residual and exited leaving cycle 1's t1,t2 unresolved"."""
    cc, eris = build(conv_check=True)
    t1_0, t2_0 = cc.get_init_guess(eris)
    t1_a, t2_a = cc.update_amps(*copied(t1_0, t2_0), eris)
    e_a = cc.energy(t1_a, t2_a, eris)
    violation_a = float(numpy.max(numpy.abs(cc._constraint_value(t1_a, t2_a))))
    t1_b, t2_b = cc.update_amps(*copied(t1_a, t2_a), eris)
    e_b = cc.energy(t1_b, t2_b, eris)
    violation_b = float(numpy.max(numpy.abs(cc._constraint_value(t1_b, t2_b))))
    de = abs(e_b - e_a)
    normt = float(numpy.linalg.norm(
        cc.amplitudes_to_vector(t1_b, t2_b) - cc.amplitudes_to_vector(t1_a, t2_a)
    ))
    return dict(t1_a=t1_a, t2_a=t2_a, e_a=e_a, t1_b=t1_b, t2_b=t2_b,
                violation_a=violation_a, violation_b=violation_b,
                de=de, normt=normt)


info = prepare()
assert info["violation_b"] < info["violation_a"], info
print("prepared two real Newton/eta cycles: max|C| cycle1 = %.3e, "
      "cycle2 = %.3e, dE(cycle2-cycle1) = %.3e, |d(t1,t2)| = %.3e"
      % (info["violation_a"], info["violation_b"], info["de"], info["normt"]))


def seed(cc):
    cc.t1, cc.t2 = copied(info["t1_a"], info["t2_a"])
    cc.converged = True
    cc.e_corr = info["e_a"]


# ---------------------------------------------------- (a) conv_check disabled
cc_a, _ = build(conv_check=False)
seed(cc_a)
cc_a.constraint_tol = 0.5 * info["violation_a"]
cc_a.conv_tol = 1.0
cc_a.conv_tol_normt = 1.0
calls_a = count_calls(cc_a)
cc_a._finalize()
assert calls_a[0] == 0, "conv_check=False must never run an extra pass"
assert cc_a.converged is False
print("(a) conv_check=False: unchanged behavior, %d extra passes, converged=%s"
      % (calls_a[0], cc_a.converged))


# ------------------------------------------------- (b) one pass resolves it
cc_b, _ = build(conv_check=True)
seed(cc_b)
cc_b.constraint_tol = 0.5 * (info["violation_a"] + info["violation_b"])
cc_b.conv_tol = max(info["de"], 1e-12)
cc_b.conv_tol_normt = max(info["normt"], 1e-12)
calls_b = count_calls(cc_b)
cc_b._finalize()
assert calls_b[0] == 1, "exactly one extra pass is allowed"
assert cc_b.converged is True
assert float(numpy.max(numpy.abs(
    cc_b._constraint_value(cc_b.t1, cc_b.t2)))) < cc_b.constraint_tol
assert float(numpy.max(numpy.abs(cc_b.t1[0] - info["t1_b"][0]))) < 1e-9
print("(b) one pass resolves the violation: %d extra pass, converged=%s"
      % (calls_b[0], cc_b.converged))


# -------------------------------------------- (c) still violated after one pass
cc_c, _ = build(conv_check=True)
seed(cc_c)
cc_c.constraint_tol = 0.5 * info["violation_b"]
cc_c.conv_tol = max(info["de"], 1e-12)
cc_c.conv_tol_normt = max(info["normt"], 1e-12)
calls_c = count_calls(cc_c)
cc_c._finalize()
assert calls_c[0] == 1, "still exactly one extra pass, never a retry loop"
assert cc_c.converged is False
print("(c) still violated after one pass: %d extra pass, converged=%s"
      % (calls_c[0], cc_c.converged))


# --------------------------------------------- (d) overshoot guards, energy
cc_d, _ = build(conv_check=True)
seed(cc_d)
cc_d.constraint_tol = 0.5 * (info["violation_a"] + info["violation_b"])
cc_d.conv_tol = info["de"] / 20.0
cc_d.conv_tol_normt = 1.0
calls_d = count_calls(cc_d)
cc_d._finalize()
assert calls_d[0] == 1
assert cc_d.converged is False, (
    "the energy overshoot guard must trip even though the constraint itself "
    "ended up satisfied"
)
print("(d) energy overshoot guard: %d extra pass, converged=%s"
      % (calls_d[0], cc_d.converged))


# ------------------------------------------- (e) overshoot guards, amplitudes
cc_e, _ = build(conv_check=True)
seed(cc_e)
cc_e.constraint_tol = 0.5 * (info["violation_a"] + info["violation_b"])
cc_e.conv_tol = 1.0
cc_e.conv_tol_normt = info["normt"] / 20.0
calls_e = count_calls(cc_e)
cc_e._finalize()
assert calls_e[0] == 1
assert cc_e.converged is False, (
    "the amplitude overshoot guard must trip even though the constraint "
    "itself ended up satisfied"
)
print("(e) amplitude overshoot guard: %d extra pass, converged=%s"
      % (calls_e[0], cc_e.converged))


# ------------------------------------------------------- (f) MultiConstrUCCSDT
sdt_coordinates = (
    ("t1a", (0, 0)),
    ("t1b", (0, 0)),
    ("t3aab", (0, 1, 0, 0, 1, 0)),
)


def constraint_sdt(t1, t2, t3):
    x = t1[0][0, 0]
    y = t1[1][0, 0]
    z = t3[1][0, 1, 0, 0, 1, 0]
    return numpy.asarray([
        x + y + z + 5 * x**2 - 0.03,
        x - y + 3 * y**2 - 0.01,
    ])


def jacobian_sdt(t1, t2, t3):
    x = t1[0][0, 0]
    y = t1[1][0, 0]
    return numpy.asarray([
        [1 + 10 * x, 1, 1],
        [1, -1 + 6 * y, 0],
    ])


sdt = MultiConstrUCCSDT(mf4, constraint_sdt, jacobian_sdt, sdt_coordinates,
                         conv_check=True)
sdt.diis = False
sdt.e_hf = mf4.e_tot
eris_sdt = sdt.ao2mo(sdt.mo_coeff)
_, initial = sdt.init_amps(eris_sdt)
tamps_a = copied_tamps(initial)
sdt.update_amps_(tamps_a, eris_sdt)
e_a_sdt = sdt.energy(tamps_a, eris_sdt)
violation_a_sdt = float(numpy.max(numpy.abs(sdt._constraint_value(tamps_a))))
tamps_b = copied_tamps(tamps_a)
sdt.update_amps_(tamps_b, eris_sdt)
violation_b_sdt = float(numpy.max(numpy.abs(sdt._constraint_value(tamps_b))))
assert violation_b_sdt < violation_a_sdt, (violation_a_sdt, violation_b_sdt)

sdt.tamps = copied_tamps(tamps_a)
sdt.converged = True
sdt.e_corr = e_a_sdt
# Normally tagged onto e_corr by the real kernel loop for _finalize's same-
# spin/opposite-spin log line; harmless placeholders since update_amps_ is
# called directly here rather than through that loop.
sdt.e_corr_ss = 0.0
sdt.e_corr_os = 0.0
sdt.constraint_tol = 0.5 * (violation_a_sdt + violation_b_sdt)
sdt.conv_tol = 1.0
sdt.conv_tol_normt = 1.0
sdt_calls = [0]
original_sdt_update = sdt.update_amps_


def counting_sdt_update(tamps, eris):
    sdt_calls[0] += 1
    return original_sdt_update(tamps, eris)


sdt.update_amps_ = counting_sdt_update
sdt._finalize()
assert sdt_calls[0] == 1
assert sdt.converged is True
assert float(numpy.max(numpy.abs(sdt._constraint_value(sdt.tamps)))) < sdt.constraint_tol
print("(f) MultiConstrUCCSDT one pass resolves the violation: %d extra pass, "
      "converged=%s" % (sdt_calls[0], sdt.converged))

print("all post-loop constraint resync checks passed")

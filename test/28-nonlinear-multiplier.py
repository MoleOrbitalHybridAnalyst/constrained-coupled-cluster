import numpy
import pyscf
from pyscf.cc import uccsd
from scipy.optimize import minimize_scalar

from ccc import MultiConstrUCCSD, MultiConstrUCCSDT


def h_chain(natom, spin=0):
    mol = pyscf.M(
        atom=";".join("H 0 0 %.2f" % (1.2 * i) for i in range(natom)),
        basis="sto-3g",
        spin=spin,
        verbose=0,
    )
    return mol.UHF().run(verbose=0)


def copied(t1, t2):
    return tuple(x.copy() for x in t1), tuple(x.copy() for x in t2)


def copied_tamps(tamps):
    return [tuple(block.copy() for block in rank) for rank in tamps]


def configured(cc):
    cc.verbose = 0
    cc.max_cycle = 200
    cc.conv_tol = 1e-10
    cc.conv_tol_normt = 1e-8
    cc.diis_space = 40
    cc.diis_start_cycle = 10
    return cc


# A two-row nonlinear constraint for which changing only the length of the
# linear multiplier step cannot satisfy both equations.
mf4 = h_chain(4)
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


default = MultiConstrUCCSD(mf4, constraint, jacobian, coordinates)
linear = MultiConstrUCCSD(
    mf4, constraint, jacobian, coordinates,
    constraint_multiplier_solver="linear",
)
nonlinear = MultiConstrUCCSD(
    mf4, constraint, jacobian, coordinates,
    constraint_multiplier_solver="nonlinear",
)
eris = default.ao2mo(default.mo_coeff)
t1, t2 = default.get_init_guess(eris)
base1, base2 = uccsd.update_amps(default, t1, t2, eris)
base = default.amplitudes_to_vector(base1, base2)

default1, default2 = default.update_amps(*copied(t1, t2), eris)
linear1, linear2 = linear.update_amps(*copied(t1, t2), eris)
nonlinear1, nonlinear2 = nonlinear.update_amps(*copied(t1, t2), eris)
for left, right in zip(default1 + default2, linear1 + linear2):
    assert numpy.array_equal(left, right)
assert numpy.array_equal(default.constraint_multipliers,
                         linear.constraint_multipliers)

linear_norm = numpy.linalg.norm(constraint(linear1, linear2))
nonlinear_norm = numpy.linalg.norm(constraint(nonlinear1, nonlinear2))
assert nonlinear.constraint_multiplier_success
assert nonlinear.constraint_multiplier_nfev > 1
assert nonlinear_norm < 1e-5 * linear_norm

# Compare against an optimized scalar multiple of the linear correction.
linear_vector = linear.amplitudes_to_vector(linear1, linear2)
direction = linear_vector - base


def scalar_objective(alpha):
    trial1, trial2 = linear.vector_to_amplitudes(base + alpha * direction)
    value = constraint(trial1, trial2)
    return 0.5 * numpy.dot(value, value)


scalar = minimize_scalar(
    scalar_objective, bounds=(-5, 5), method="bounded",
    options={"xatol": 1e-14},
)
assert nonlinear.constraint_multiplier_objective < 1e-6 * scalar.fun

# Exercise the same full-vector nonlinear solve with an actual T3 coordinate.
sdt4_coordinates = (
    ("t1a", (0, 0)),
    ("t1b", (0, 0)),
    ("t3aab", (0, 1, 0, 0, 1, 0)),
)


def constraint_sdt4(t1, t2, t3):
    x = t1[0][0, 0]
    y = t1[1][0, 0]
    z = t3[1][0, 1, 0, 0, 1, 0]
    return numpy.asarray([
        x + y + z + 5 * x**2 - 0.03,
        x - y + 3 * y**2 - 0.01,
    ])


def jacobian_sdt4(t1, t2, t3):
    x = t1[0][0, 0]
    y = t1[1][0, 0]
    return numpy.asarray([
        [1 + 10 * x, 1, 1],
        [1, -1 + 6 * y, 0],
    ])


linear_sdt4 = MultiConstrUCCSDT(
    mf4, constraint_sdt4, jacobian_sdt4, sdt4_coordinates,
    constraint_multiplier_solver="linear",
)
nonlinear_sdt4 = MultiConstrUCCSDT(
    mf4, constraint_sdt4, jacobian_sdt4, sdt4_coordinates,
    constraint_multiplier_solver="nonlinear",
)
linear_sdt4.diis = nonlinear_sdt4.diis = False
eris_sdt4 = linear_sdt4.ao2mo(linear_sdt4.mo_coeff)
_, initial_sdt4 = linear_sdt4.init_amps(eris_sdt4)
linear_tamps4 = copied_tamps(initial_sdt4)
nonlinear_tamps4 = copied_tamps(initial_sdt4)
linear_sdt4.update_amps_(linear_tamps4, eris_sdt4)
nonlinear_sdt4.update_amps_(nonlinear_tamps4, eris_sdt4)
linear_public4 = linear_sdt4.public_amplitudes(linear_tamps4)
nonlinear_public4 = nonlinear_sdt4.public_amplitudes(nonlinear_tamps4)
linear_sdt4_norm = numpy.linalg.norm(constraint_sdt4(*linear_public4))
nonlinear_sdt4_norm = numpy.linalg.norm(constraint_sdt4(*nonlinear_public4))
assert nonlinear_sdt4.constraint_multiplier_success
assert nonlinear_sdt4.constraint_multiplier_nfev > 1
assert nonlinear_sdt4_norm < 1e-4 * linear_sdt4_norm, (
    linear_sdt4_norm, nonlinear_sdt4_norm
)

# The nonlinear optimizer must honor the existing max-amplitude trust region.
limit = 2e-3
trusted = MultiConstrUCCSD(
    mf4, constraint, jacobian, coordinates,
    constraint_max_step=limit,
    constraint_multiplier_solver="nonlinear",
)
trusted1, trusted2 = trusted.update_amps(*copied(t1, t2), eris)
trusted_vector = trusted.amplitudes_to_vector(trusted1, trusted2)
assert numpy.max(numpy.abs(
    trusted_vector[trusted._coordinate_flat]
    - base[trusted._coordinate_flat]
)) <= limit + 1e-12

# A failed optimizer still retains a finite candidate no worse than the linear
# multiplier evaluated with the exact nonlinear constraint.
limited = MultiConstrUCCSD(
    mf4, constraint, jacobian, coordinates,
    constraint_multiplier_solver="nonlinear",
    constraint_multiplier_max_cycle=1,
)
limited1, limited2 = limited.update_amps(*copied(t1, t2), eris)
assert not limited.constraint_multiplier_success
assert numpy.linalg.norm(constraint(limited1, limited2)) <= linear_norm + 1e-12

# Trial scattering must preserve all same-spin antisymmetry partners.
same_coordinate = (("t2aa", (1, 0, 1, 0)),)


def same_constraint(t1, t2):
    x = t2[0][1, 0, 1, 0]
    return numpy.asarray([x + 2 * x**2 - 0.01])


def same_jacobian(t1, t2):
    x = t2[0][1, 0, 1, 0]
    return numpy.asarray([[1 + 4 * x]])


same = MultiConstrUCCSD(
    mf4, same_constraint, same_jacobian, same_coordinate,
    constraint_multiplier_solver="nonlinear",
)
same1, same2 = same.update_amps(*copied(t1, t2), eris)
i, j, a, b = same_coordinate[0][1]
value = same2[0][i, j, a, b]
assert same2[0][j, i, a, b] == -value
assert same2[0][i, j, b, a] == -value
assert same2[0][j, i, b, a] == value

# With two electrons, triples are empty.  Nonlinear UCCSDT must reproduce
# nonlinear UCCSD without packing a full T3 vector during trial evaluations.
mf2 = h_chain(2)
coordinate2 = (("t2ab", (0, 0, 0, 0)),)
rhs = -0.05


def constraint_sd(t1, t2):
    z = t2[1][0, 0, 0, 0]
    return numpy.asarray([z + 4 * z**2 - rhs])


def jacobian_sd(t1, t2):
    z = t2[1][0, 0, 0, 0]
    return numpy.asarray([[1 + 8 * z]])


def constraint_sdt(t1, t2, t3):
    return constraint_sd(t1, t2)


def jacobian_sdt(t1, t2, t3):
    return jacobian_sd(t1, t2)


default_sdt = MultiConstrUCCSDT(
    mf2, constraint_sdt, jacobian_sdt, coordinate2
)
linear_sdt = MultiConstrUCCSDT(
    mf2, constraint_sdt, jacobian_sdt, coordinate2,
    constraint_multiplier_solver="linear",
)
default_sdt.diis = linear_sdt.diis = False
eris_default_sdt = default_sdt.ao2mo(default_sdt.mo_coeff)
_, initial_default_sdt = default_sdt.init_amps(eris_default_sdt)
initial_linear_sdt = copied_tamps(initial_default_sdt)
default_sdt.update_amps_(initial_default_sdt, eris_default_sdt)
linear_sdt.update_amps_(initial_linear_sdt, eris_default_sdt)
for default_rank, linear_rank in zip(initial_default_sdt, initial_linear_sdt):
    for default_block, linear_block in zip(default_rank, linear_rank):
        assert numpy.array_equal(default_block, linear_block)
assert numpy.array_equal(default_sdt.constraint_multipliers,
                         linear_sdt.constraint_multipliers)

step_sd = MultiConstrUCCSD(
    mf2, constraint_sd, jacobian_sd, coordinate2,
    constraint_multiplier_solver="nonlinear",
)
step_sdt = MultiConstrUCCSDT(
    mf2, constraint_sdt, jacobian_sdt, coordinate2,
    constraint_multiplier_solver="nonlinear",
)
step_sdt.diis = False
eris_sd = step_sd.ao2mo(step_sd.mo_coeff)
initial1, initial2 = step_sd.get_init_guess(eris_sd)
result1, result2 = step_sd.update_amps(*copied(initial1, initial2), eris_sd)
eris_sdt = step_sdt.ao2mo(step_sdt.mo_coeff)
_, initial_sdt = step_sdt.init_amps(eris_sdt)
step_sdt.amplitudes_to_vector = lambda *args: (_ for _ in ()).throw(
    AssertionError("nonlinear UCCSDT trial packed the full amplitudes")
)
step_sdt.update_amps_(initial_sdt, eris_sdt)
public1, public2, public3 = step_sdt.public_amplitudes(initial_sdt)
for left, right in zip(result1 + result2, public1 + public2):
    assert numpy.allclose(left, right, atol=1e-13, rtol=0)
assert abs(constraint_sdt(public1, public2, public3)[0]) < 1e-7

converged_sd = configured(MultiConstrUCCSD(
    mf2, constraint_sd, jacobian_sd, coordinate2,
    constraint_tol=1e-9,
    constraint_multiplier_solver="nonlinear",
)).run()
converged_sdt = configured(MultiConstrUCCSDT(
    mf2, constraint_sdt, jacobian_sdt, coordinate2,
    constraint_tol=1e-9,
    constraint_multiplier_solver="nonlinear",
)).run()
assert converged_sd.converged and converged_sdt.converged
assert abs(converged_sd.e_tot - converged_sdt.e_tot) < 1e-12
assert numpy.max(numpy.abs(converged_sdt.constraint_residual())) < 1e-9

converged = configured(MultiConstrUCCSD(
    mf4, constraint, jacobian, coordinates,
    constraint_tol=1e-8,
    constraint_multiplier_solver="nonlinear",
)).run()
assert converged.converged
assert numpy.max(numpy.abs(converged.constraint_residual())) < 1e-8

for bad in ("newton", "", None):
    try:
        MultiConstrUCCSD(
            mf2, constraint_sd, jacobian_sd, coordinate2,
            constraint_multiplier_solver=bad,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("invalid multiplier solver was accepted")

print("linear one-step regression passed")
print("linear exact constraint norm    = %.3e" % linear_norm)
print("nonlinear exact constraint norm = %.3e" % nonlinear_norm)
print("nonlinear UCCSDT constraint norm = %.3e" % nonlinear_sdt4_norm)
print("best scalar objective           = %.3e" % scalar.fun)
print("full-vector objective           = %.3e"
      % nonlinear.constraint_multiplier_objective)
print("nonlinear UCCSDT-UCCSD          = %.3e"
      % (converged_sdt.e_tot - converged_sd.e_tot))

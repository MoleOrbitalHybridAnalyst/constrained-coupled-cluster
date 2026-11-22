import numpy

from ccc.active_spin_constraint import DominantSectorSpinConstraint


def random_amplitudes(engine, seed):
    """Random values laid out in the engine's own coordinate blocks."""
    rng = numpy.random.default_rng(seed)
    na, nb = engine.nelec
    nva, nvb = engine.norb - na, engine.norb - nb
    shapes = (
        (na, nva), (nb, nvb),
        (na, na, nva, nva), (na, nb, nva, nvb), (nb, nb, nvb, nvb),
        (na, na, na, nva, nva, nva), (na, na, nb, nva, nva, nvb),
        (na, nb, nb, nva, nvb, nvb), (nb, nb, nb, nvb, nvb, nvb),
    )
    blocks = [numpy.zeros(shape) for shape in shapes]
    named = dict(zip(
        ("t1a", "t1b", "t2aa", "t2ab", "t2bb",
         "t3aaa", "t3aab", "t3abb", "t3bbb"),
        blocks,
    ))
    for block, index in engine.coordinates:
        named[block][index] = rng.normal(scale=0.05)
    t3 = tuple(blocks[5:]) if engine.max_cluster_rank == 3 else None
    return tuple(blocks[:2]), tuple(blocks[2:5]), t3


for rank in (2, 3):
    engine = DominantSectorSpinConstraint(
        5, (3, 2), "ababa", max_cluster_rank=rank
    )
    t1, t2, t3 = random_amplitudes(engine, 7 + rank)
    values = engine.amplitude_values(t1, t2, t3)

    # The compiled cluster matrix must reproduce a term-by-term application of
    # sum_mu t_mu tau_mu on an arbitrary vector, not just on the reference.
    engine._ensure_cluster_matrix()
    rng = numpy.random.default_rng(rank)
    vector = rng.normal(size=engine._shape)
    brute = numpy.zeros(engine._shape)
    for value, operator in zip(values, engine._operators):
        brute += value * engine._apply_operator(operator, vector)
    engine._cluster.data[:] = (
        engine._cluster_sign * values[engine._cluster_column]
    )
    fused = engine._cluster.dot(vector.ravel()).reshape(engine._shape)
    cluster_error = float(numpy.max(numpy.abs(fused - brute)))
    assert cluster_error < 1e-14, cluster_error

    # The compiled residual and Jacobian must reproduce the same quantities
    # evaluated one excitation operator at a time.
    wavefunction = engine._build_wavefunction(values)
    engine._ensure_residual_maps()
    flat = wavefunction.ravel()
    residual_error = float(numpy.max(numpy.abs(
        engine._left @ flat[engine._sector_flat]
        - engine._sector_residual(wavefunction)
    )))
    assert residual_error < 1e-14, residual_error

    brute_jacobian = numpy.empty((engine.nconstraint, len(engine.coordinates)))
    for column, operator in enumerate(engine._operators):
        brute_jacobian[:, column] = engine._sector_residual(
            engine._apply_operator(operator, wavefunction)
        )
    fused_jacobian = engine._left @ (
        engine._source_sign * flat[engine._source]
    )
    jacobian_error = float(numpy.max(numpy.abs(fused_jacobian - brute_jacobian)))
    assert jacobian_error < 1e-14, jacobian_error

    # ... and the analytic Jacobian must still match finite differences.
    finite_difference = engine.check_jacobian(t1, t2, t3, ndir=5)

    print("rank %d: cluster %.3e  residual %.3e  jacobian %.3e  findiff %.3e"
          % (rank, cluster_error, residual_error, jacobian_error,
             finite_difference))

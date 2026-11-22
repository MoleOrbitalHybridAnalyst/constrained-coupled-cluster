"""Constrained coupled cluster (cCC).

Conventional single-reference CC amplitude equations solved subject to
user-supplied constraints C(t) = 0, with Lagrange multipliers eta:
R(t) = G(t)^T eta.  Spin-symmetry constraints live in the
``*_spin_constraint`` modules.
"""
from .uccsd_linconstr import LinConstrUCCSD
from .uccsd_constr import ConstrUCCSD
from .uccsd_multiconstr import MultiConstrUCCSD
from .uccsdt_multiconstr import (MultiConstrUCCSDT,
                                 LowMemoryMultiConstrUCCSDT)
from .uccsdt_lowm_spinconstr import LowMemorySpinConstrUCCSDT
from .gccsd_multiconstr import MultiConstrGCCSD
from .uccsd_lambdat import uccsd_lambdat
from .cluster_decomposition import cluster_decompose, cas_amplitudes

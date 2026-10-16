# cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
"""Fast Metropolis sweeps for a 2D Ising lattice, with periodic boundaries."""
import numpy as np
cimport numpy as cnp
from libc.math cimport exp

# A sweep visits every site once, in a freshly randomized order.
# spin and random arrays are prepared in Python to keep random generation reproducible.
def sweep(cnp.ndarray[cnp.int8_t, ndim=2] spins,
          double temperature, double field,
          cnp.ndarray[cnp.int32_t, ndim=1] order,
          cnp.ndarray[cnp.float64_t, ndim=1] uniforms):
    cdef int L = spins.shape[0]
    cdef int n = L * L
    cdef int k, i, j, up, down, left, right, neighbors, old
    cdef double delta
    for k in range(n):
        i = order[k] // L
        j = order[k] % L
        up = (i - 1 + L) % L
        down = (i + 1) % L
        left = (j - 1 + L) % L
        right = (j + 1) % L
        old = spins[i, j]
        neighbors = spins[up, j] + spins[down, j] + spins[i, left] + spins[i, right]
        # E = -J sum_<ij> s_i s_j - H sum_i s_i; J = k_B = 1
        delta = 2.0 * old * (neighbors + field)
        if delta <= 0.0 or uniforms[k] < exp(-delta / temperature):
            spins[i, j] = -old

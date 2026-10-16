# cython: boundscheck=False, wraparound=False

from libc.math cimport exp

cpdef long[:,:] loop_c(long[:,:] s, long L, double T,
                       double[:,:] zeta, double H=0.0):

    cdef int i, j, ip1, im1, jp1, jm1
    cdef long st, neighbors
    cdef double E_flip

    for i in range(L):

        ip1 = (i + 1) % L
        im1 = (i - 1 + L) % L

        for j in range(L):

            jp1 = (j + 1) % L
            jm1 = (j - 1 + L) % L

            st = s[i,j]

            neighbors = (
                s[ip1,j] + s[im1,j]
                + s[i,jp1] + s[i,jm1]
            )

            E_flip = 2.0 * st * (neighbors + H)

            if E_flip <= 0:
                s[i,j] = -1*st

            else:
                if zeta[i,j] < exp(-E_flip/T):
                    s[i,j] = -1*st

    return s
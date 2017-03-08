cimport numpy as np
import numpy as np

np.import_array()

cdef extern from "altsim.h":
    void C_pressure_solve(double * pressue, double * pressure_buffer, 
                          double * divergence, unsigned char * boundary, 
                          int Nx, int Ny, double dx, double dy)
    
def pressure_solve(np.ndarray[double, mode="c", ndim=2] p, 
                   np.ndarray[double, mode="c", ndim=2] div, 
                   b, notb, dx):

    cdef int Nx = np.shape(p)[0]
    cdef int Ny = np.shape(p)[1]
    cdef double deltax = dx[0]
    cdef double deltay = dx[1]
           
    cdef np.ndarray[np.double_t, mode="c", ndim=2] pressure_buffer = np.array(p, order='C')
    cdef np.ndarray[np.uint8_t, mode="c", ndim=2] boundary = np.array(b.view(np.uint8), order='C')
    
    C_pressure_solve(&p[0, 0], &pressure_buffer[0, 0], &div[0, 0], 
                     &boundary[0, 0], Nx, Ny, deltax, deltay)
    
    return p


cdef extern from "altsim.h":
    void C_advect_velocity(double * v, const double * const v0, 
                           const unsigned char * bound, int * advect_indexes,
                           double * advect_lerps, const int Nx, const int Ny,
                           const double dx, const double dy, const double dt)
    
def advect_velocity(np.ndarray[double, mode="c", ndim=3] v, 
                    np.ndarray[double, mode="c", ndim=3] v0, 
                    b, indexArray, dx, dt,
                    np.ndarray[int, mode="c", ndim=3] advect_indexes,
                    np.ndarray[double, mode="c", ndim=3] advect_lerps):
    
    cdef int Nx = np.shape(b)[0]
    cdef int Ny = np.shape(b)[1]
    cdef double deltax = dx[0]
    cdef double deltay = dx[1]    
    
    cdef np.ndarray[np.uint8_t, mode="c", ndim=2] boundary = np.array(b.view(np.uint8), order='C')
    
    C_advect_velocity(&v[0, 0, 0], &v0[0, 0, 0], &boundary[0, 0], 
                      &advect_indexes[0, 0, 0], &advect_lerps[0, 0, 0], Nx, Ny,
                      deltax, deltay, dt);



cdef extern from "altsim.h":
    void C_apply_advection(double * x, const double * const x0, 
                           const unsigned char * bound,
                           int * advect_indexes, double * advect_lerps, 
                           const int Nx, const int Ny)
    
def apply_advection(np.ndarray[double, mode="c", ndim=2] x, 
                   np.ndarray[double, mode="c", ndim=2] x0, b,
                   np.ndarray[int, mode="c", ndim=3] advect_indexes,
                   np.ndarray[double, mode="c", ndim=3] advect_lerps):
    
    cdef int Nx = np.shape(x)[0]
    cdef int Ny = np.shape(x)[1]
    
    cdef np.ndarray[np.uint8_t, mode="c", ndim=2] boundary = np.array(b.view(np.uint8), order='C')
    
    C_apply_advection(&x[0, 0], &x0[0, 0], &boundary[0, 0], 
                      &advect_indexes[0, 0, 0], &advect_lerps[0, 0, 0], Nx, Ny);
    
    return x
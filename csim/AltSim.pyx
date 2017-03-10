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


cdef extern from "altsim.h":
    void C_divergence(double * div, const double * const v, 
                           const unsigned char * bound,
                           const int Nx, const int Ny,
                           const double dx, const double dy)
    

def divergence(np.ndarray[double, mode="c", ndim=2] div, 
               np.ndarray[double, mode="c", ndim=3] v, b, dx):
    
    cdef int Nx = np.shape(div)[0]
    cdef int Ny = np.shape(div)[1]
    
    cdef double deltax = dx[0]
    cdef double deltay = dx[1]    
    
    cdef np.ndarray[np.uint8_t, mode="c", ndim=2] boundary = np.array(b.view(np.uint8), order='C')
        
    
    C_divergence(&div[0, 0], &v[0, 0, 0], &boundary[0, 0], Nx, Ny, deltax, deltay)

    return div


cdef extern from "altsim.h":
    void C_sub_gradient(double * v, const double * const v0, 
                           const double * const p,
                           const int Nx, const int Ny,
                           const double dx, const double dy)

def sub_gradient(np.ndarray[double, mode="c", ndim=3] v, 
                 np.ndarray[double, mode="c", ndim=3] v0, 
                 np.ndarray[double, mode="c", ndim=2] p, dx):  
      
    cdef int Nx = np.shape(p)[0]
    cdef int Ny = np.shape(p)[1]
        
    cdef double deltax = dx[0]
    cdef double deltay = dx[1]   
    
    C_sub_gradient(&v[0,0,0], &v0[0,0,0], &p[0,0], Nx, Ny, deltax, deltay)
    
    return v

cdef extern from "altsim.h":
    void C_enforce_slip(double * v, 
                        const unsigned char * bound,
                        const int Nx, const int Ny)

def enforce_slip(np.ndarray[double, mode="c", ndim=3] v, b):
    
    cdef int Nx = np.shape(b)[0]
    cdef int Ny = np.shape(b)[1]
    
    cdef np.ndarray[np.uint8_t, mode="c", ndim=2] boundary = np.array(b.view(np.uint8), order='C')
    
    C_enforce_slip(&v[0,0,0], &boundary[0,0], Nx, Ny)
    
    return v

cdef extern from "altsim.h":
    void C_step(double * v, double * vtmp, double * vtmp2, double * p,
                double * div, double * density_arrays, 
                const int num_density_arrays, const unsigned char * bound,
                int * advect_indexes, double * advect_lerps, const int Nx,
                const int Ny, const double dx, const double dy,const double dt0)

def cstep(np.ndarray[double, mode="c", ndim=3] v,
         np.ndarray[double, mode="c", ndim=3] vtmp,
         np.ndarray[double, mode="c", ndim=3] vtmp2,
         np.ndarray[double, mode="c", ndim=2] p,
         np.ndarray[double, mode="c", ndim=2] div,
         density_arrays,
         bound,
         np.ndarray[int, mode="c", ndim=3] advect_indexes,
         np.ndarray[double, mode="c", ndim=3] advect_lerps, 
         dx, dt):
    
    cdef int Nx = np.shape(p)[0]
    cdef int Ny = np.shape(p)[1]
        
    cdef double deltax = dx[0]
    cdef double deltay = dx[1]   
    
    cdef np.ndarray[np.uint8_t, mode="c", ndim=2] boundary = np.array(bound.view(np.uint8), order='C')
    cdef np.ndarray[double, mode="c", ndim=3] den_arrays
    
    if len(density_arrays) > 0:        
        den_arrays = np.array(density_arrays, order='C')
        C_step(&v[0,0,0], &vtmp[0,0,0], &vtmp2[0,0,0], &p[0,0], &div[0,0], 
               &den_arrays[0,0,0], len(density_arrays), &boundary[0,0], &advect_indexes[0,0,0],
               &advect_lerps[0,0,0], Nx, Ny, deltax, deltay, dt)
        density_arrays[:] = den_arrays
    else:
        C_step(&v[0,0,0], &vtmp[0,0,0], &vtmp2[0,0,0], &p[0,0], &div[0,0], 
               &vtmp2[0,0,0], 0, &boundary[0,0], &advect_indexes[0,0,0],
               &advect_lerps[0,0,0], Nx, Ny, deltax, deltay, dt)
    
            
from numba import jit
import numpy as np

from SimBase import SimBase

from csim.AltSim import pressure_solve, advect_velocity, apply_advection
  
          
@jit
def divergence(div, v, notb, dx):
    div[-1,:] = 0
    div[:-1, :] = v[0, 1:, :] * notb[1:, :] / (2 * dx[0])
    div[1:, :] -= v[0, :-1, :] * notb[:-1, :] / (2 * dx[0])
    div[:, :-1] += v[1, :, 1:] * notb[:, 1:] / (2 * dx[1])
    div[:, 1:] -= v[1, :, :-1] * notb[:, :-1] / (2 * dx[1])  
    #div[self._b] = 0
    return div
      
  
@jit
def sub_gradient(v, v0, p, dx):        
    v[0, 1:-1, :] = v0[0, 1:-1, :] - 1 / (2 * dx[0]) * (p[2:, :] - p[:-2, :])
    v[1, :, 1:-1] = v0[1, :, 1:-1] - 1 / (2 * dx[1]) * (p[:, 2:] - p[:, :-2])
    return v
        
  
@jit
def enforce_slip(v, notb, b):
    v[:, b] = 0
    right_edge = np.logical_and(notb[:-1,:], b[1:,:])
    v[1, :-1,:][right_edge] = v[1, 1:,:][right_edge]
    left_edge = np.logical_and(notb[1:,:], b[:-1,:])
    v[1, 1:,:][left_edge] = v[1, :-1,:][left_edge]
    top_edge = np.logical_and(notb[:,:-1], b[:,1:])
    v[0, :,:-1][top_edge] = v[0, :,:-1][top_edge]
    bottom_edge = np.logical_and(notb[:,1:], b[:,:-1])
    v[0, :, 1:][bottom_edge] = v[0, :, 1:][bottom_edge]
    return v

class Sim(SimBase):
    
    def __init__(self, shape, diffusion, viscosity):    
        super().__init__(shape)
        
        self._div = np.zeros(shape)
        self._p = np.zeros(shape)
        self._vtmp = np.ascontiguousarray(np.zeros_like(self._v, order='C'))
        self._vtmp2 = np.zeros_like(self._v, order='C')
        self._dtmp = np.zeros(shape)
        
        xs = np.arange(0.0, shape[0], 1)
        ys = np.arange(0.0, shape[1], 1)
        x, y = np.meshgrid(xs, ys)
        self._indexArray = np.array([x.T, y.T])  
        self._xi = np.zeros_like(self._v, dtype=np.int32)
        self._s = np.zeros_like(self._v)
        
         
    @jit
    def step(self, dt, density_arrays):
        
        dt /= 2
        for _ in range(2):
            advect_velocity(self._vtmp, self._v, 
                self._b, self._indexArray, self._dx, dt, self._xi, self._s)
            
            '''
            self._vtmp2[:], _, _ = advect_velocity(self._vtmp2, self._v, 
                self._b, self._indexArray, self._dx, dt)
            self._vtmp[:], _, _ = advect_velocity(self._vtmp, self._vtmp2, 
                self._b, self._indexArray, self._dx, -dt)
            self._vtmp2 = 1.5 * self._v - 0.5 * self._vtmp
            self._vtmp[:], self._xi, self._s = advect_velocity(self._vtmp, self._vtmp2, 
                self._b, self._indexArray, self._dx, dt)
            '''
                    
            self._div[:] = divergence(self._div, self._vtmp, self._notb, self._dx)
            self._p[:] = pressure_solve(self._p, self._div, self._b, self._notb, self._dx)
            self._v[:] = sub_gradient(self._v, self._vtmp, self._p, self._dx)
            self._v[:] = enforce_slip(self._v, self._notb, self._b)
    
            for d in density_arrays:
                d[:] = apply_advection(self._dtmp, d, self._b, self._xi, self._s)
            

    def get_pressure_as_rgb(self):        
        width, height = np.shape(self._p)
        rgb = np.zeros((3, width, height))
        pmax = max(np.max(self._p), -np.min(self._p))
        
        if pmax > 0:
            rgb[2, self._p > 0] = self._p[self._p > 0] / pmax
            rgb[0, self._p < 0] = self._p[self._p < 0] / pmax
        
        return rgb

        
        
        
        
        
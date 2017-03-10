from numba import jit
import numpy as np

from SimBase import SimBase

@jit
def advect_velocity(v, v0, b, indexArray, dx, dt):
    shape = np.shape(v0[0])
    x = indexArray - dt * v0 / dx[:,np.newaxis,np.newaxis]
    x = np.array([np.clip(x[0], 0, shape[0] - 1.01),
                  np.clip(x[1], 0, shape[1] - 1.01)])
    indexes = np.array(x, dtype=int)
    advectInter = x - indexes               
    
    xi = indexes[0]
    yi = indexes[1]  
    s = advectInter[0]
    t = advectInter[1]        
    
    v[:] = (1 - s) * ((1 - t) * v0[:, xi, yi] 
        + t * v0[:, xi, yi + 1]) + s * ((1 - t) * v0[:, xi + 1, yi] 
        + t * v0[:, xi + 1, yi + 1])
    v[:, b] = v0[:, b] 
    return v, indexes, advectInter

@jit
def apply_advection(x, x0, xi, s):
    xi, yi = xi       
    s, t = s
    
    x[:] = (1 - s) * ((1 - t) * x0[xi, yi] 
        + t * x0[xi, yi + 1]) + s * ((1 - t) * x0[xi + 1, yi] 
        + t * x0[xi + 1, yi + 1])
    return x
        
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
def pressure_solve(p, div, b, notb, dx):
    p[notb] = 0
    
    bound = 0.0 + b[0:-2,1:-1] + b[2:,1:-1] + b[1:-1,0:-2] + b[1:-1,2:]
    
    for i in range(50):
        p[1:-1,1:-1] = 1 / 4 * (p[1:-1,1:-1] * bound
            + p[0:-2,1:-1] * notb[0:-2,1:-1] 
            + p[2:,1:-1] * notb[2:,1:-1] 
            + p[1:-1,0:-2] * notb[1:-1,0:-2] 
            + p[1:-1,2:] * notb[1:-1,2:]                
            - dx[0] * dx[1] * div[1:-1,1:-1])
        
    return p
         
@jit
def sub_gradient(v, v0, p, dx):        
    v[0, 1:-1, :] = v0[0, 1:-1, :] - 1 / (2 * dx[0]) * (p[2:, :] - p[:-2, :])
    v[1, :, 1:-1] = v0[1, :, 1:-1] - 1 / (2 * dx[1]) * (p[:, 2:] - p[:, :-2])
    return v
        
@jit 
def enforce_slip(v, notb, b):
    v[:, b] = 0
    right_edge = np.logical_and(notb[:-1,:], b[1:,:])
    v[0, :-1,:][right_edge] = v[0, 1:,:][right_edge]
    left_edge = np.logical_and(notb[1:,:], b[:-1,:])
    v[0, 1:,:][left_edge] = v[0, :-1,:][left_edge]
    top_edge = np.logical_and(notb[:,:-1], b[:,1:])
    v[1, :,:-1][top_edge] = v[1, :,:-1][top_edge]
    bottom_edge = np.logical_and(notb[:,1:], b[:,:-1])
    v[1, :, 1:][bottom_edge] = v[1, :, 1:][bottom_edge]
    return v

class Sim(SimBase):
    
    def __init__(self, shape, diffusion, viscosity):    
        super().__init__(shape)
        
        self._div = np.zeros(shape)
        self._p = np.zeros(shape)
        self._vtmp = np.zeros((2, *shape))
        self._vtmp2 = np.zeros((2, *shape))
        self._dtmp = np.zeros(shape)
        
        xs = np.arange(0.0, shape[0], 1)
        ys = np.arange(0.0, shape[1], 1)
        x, y = np.meshgrid(xs, ys)
        self._indexArray = np.array([x.T, y.T])  
        self._xi = np.zeros_like(self._v, dtype=int)
        self._s = np.zeros_like(self._v)
        
    @jit    
    def step(self, dt, density_arrays):
        
        self._vtmp[:], self._xi, self._s = advect_velocity(self._vtmp, self._v, 
            self._b, self._indexArray, self._dx, dt)
        
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
            d[:] = apply_advection(self._dtmp, d, self._xi, self._s)
            d[self._b] = 0
            
    @jit
    def get_pressure_as_rgb(self):        
        width, height = np.shape(self._p)
        rgb = np.zeros((3, width, height))
        pmax = max(np.max(self._p), -np.min(self._p))
        
        if pmax > 0:
            rgb[2, self._p > 0] = self._p[self._p > 0] / pmax
            rgb[0, self._p < 0] = self._p[self._p < 0] / pmax
        
        return rgb

        
        
        
        
        
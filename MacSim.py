from numba import jit
import numpy as np

from SimBase import SimBase

class Sim(SimBase):
    
    def __init__(self, shape, diffusion, viscosity):    
        super().__init__(shape)
        
        self._tmp = np.zeros((4, *shape))   
        
        xs = np.arange(0.0, shape[0], 1)
        ys = np.arange(0.0, shape[1], 1)
        x, y = np.meshgrid(xs, ys)
        self._indexArray = np.array([x.T, y.T])        
        
    @jit
    def _advect_velocity(self, v, v0, indexes, dt):
        shape = np.shape(v0[0])
        x = self._indexArray - dt * v0 / self._dx[:,np.newaxis,np.newaxis]
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
        v[:, self._b] = v0[:, self._b] 
        
    @jit
    def _advect_forward_velocity(self, v, v0, indexes, dt):
        shape = np.shape(v0[0])
        x = self._indexArray + dt * v0 / self._dx[:,np.newaxis,np.newaxis]
        x = np.array([np.clip(x[0], 0, shape[0] - 1.01),
                      np.clip(x[1], 0, shape[1] - 1.01)])
        indexes = np.array(x, dtype=int)
        advectInter = x - indexes               
        
        xi = indexes[0]
        yi = indexes[1]  
        s = advectInter[0]
        t = advectInter[1]  
        
        v[:] = 0
        v[:, xi, yi] += (1 - s) * (1 - t) * v0
        v[:, xi, yi + 1] += (1 - s) * t * v0
        v[:, xi + 1, yi] += s * (1 - t) * v0
        v[:, xi + 1, yi + 1] += s * t * v0
        v[:, self._b] = v0[:, self._b] 
                
    @jit
    def _updateadvect(self, dt):
        shape = np.shape(self._v[0])
        x = self._indexArray - dt * self._v / self._dx[:,np.newaxis,np.newaxis]
        x = np.array([np.clip(x[0], 0, shape[0] - 1.01),
                      np.clip(x[1], 0, shape[1] - 1.01)])
        advectIndex = np.array(x, dtype=int)
        advectInter = x - advectIndex                
        self._xi = advectIndex[:, self._notb]
        self._s = advectInter[:, self._notb]
        
    @jit
    def _advect(self, x, x0):
        xi = self._xi[0]
        yi = self._xi[1]        
        s = self._s[0]
        t = self._s[1]
        
        x[self._notb] = (1 - s) * ((1 - t) * x0[xi, yi] 
            + t * x0[xi, yi + 1]) + s * ((1 - t) * x0[xi + 1, yi] 
            + t * x0[xi + 1, yi + 1])
        x[self._b] = x0[self._b]
        
    @jit
    def _divergence(self, div, x0):
        div[-1,:] = 0
        div[:-1, :] = x0[0, 1:, :] * self._notb[1:, :] / (2 * self._dx[0])
        div[1:, :] -= x0[0, :-1, :] * self._notb[:-1, :] / (2 * self._dx[0])
        div[:, :-1] += x0[1, :, 1:] * self._notb[:, 1:] / (2 * self._dx[1])
        div[:, 1:] -= x0[1, :, :-1] * self._notb[:, :-1] / (2 * self._dx[1])        
        div[self._b] = 0
        
    @jit
    def _pressure_solve(self, p, div):
        p[:] = 0
        
        bound = 0.0 + self._b[0:-2,1:-1] + self._b[2:,1:-1] + self._b[1:-1,0:-2] + self._b[1:-1,2:]
        
        for i in range(50):
            p[1:-1,1:-1] = 1 / 4 * (p[1:-1,1:-1] * bound
                + p[0:-2,1:-1] * self._notb[0:-2,1:-1] 
                + p[2:,1:-1] * self._notb[2:,1:-1] 
                + p[1:-1,0:-2] * self._notb[1:-1,0:-2] 
                + p[1:-1,2:] * self._notb[1:-1,2:]                
                - self._dx[0] * self._dx[1] * div[1:-1,1:-1])
            p[self._b] = 0
         
    @jit
    def _sub_gradient(self, v, v0, p):        
        v[0, 1:-1, :] = v0[0, 1:-1, :] - 1 / (2 * self._dx[0]) * (p[2:, :] - p[:-2, :])
        v[1, :, 1:-1] = v0[1, :, 1:-1] - 1 / (2 * self._dx[1]) * (p[:, 2:] - p[:, :-2])

    @jit    
    def step(self, dt, density_arrays):        
        indexes = np.zeros_like(self._v, dtype=int)
        indexes2 = np.zeros_like(self._v, dtype=int)
        self._advect_velocity(self._tmp[:2], self._v, indexes, dt)
        self._advect_forward_velocity(self._tmp[2:4], self._tmp[:2], indexes2, dt)
        self._tmp[:2] = self._tmp[:2] + 0.5 * (self._v - self._tmp[2:4])
        
        xi = indexes[0]
        yi = indexes[1] 
        
        ui = np.array([self._v[0, xi, yi], self._v[0, xi+1, yi], self._v[0, xi, yi+1], self._v[0, xi+1, yi+1]])
        vi = np.array([self._v[1, xi, yi], self._v[1, xi+1, yi], self._v[1, xi, yi+1], self._v[1, xi+1, yi+1]])

        umin = np.min(ui, axis=0)        
        umax = np.max(ui, axis=0)
        vmin = np.min(vi, axis=0)
        vmax = np.max(vi, axis=0)
        
        self._tmp[0] = np.clip(self._tmp[0], umin, umax)
        self._tmp[1] = np.clip(self._tmp[1], vmin, vmax)
        
        self._divergence(self._v[0], self._tmp[:2])
        self._pressure_solve(self._v[1], self._v[0])
        self._sub_gradient(self._v, self._tmp, self._v[1])
        self._v[:, self._b] = 0

        self._updateadvect(dt)
        for d in density_arrays:
            self._advect(self._tmp[0], d)
            self._tmp[0, self._b] = 0
            d[:] = self._tmp[0]

        
        
        
        
        
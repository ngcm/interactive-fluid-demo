from numba import jit
import numpy as np

from SimBase import SimBase

from csim.AltSim import cstep

class Sim(SimBase):
    
    def __init__(self, shape, diffusion, viscosity):    
        super().__init__(shape)
        
        self._div = np.zeros(shape)
        self._p = np.zeros(shape)
        self._vtmp = np.zeros_like(self._v, order='C')
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
        
        cstep(self._v, self._vtmp, self._vtmp2, self._p, self._div, density_arrays, 
              self._b, self._xi, self._s, self._dx, dt)
            
    @jit
    def get_pressure_as_rgb(self):        
        width, height = np.shape(self._p)
        rgb = np.zeros((3, width, height))
        pmax = max(np.max(self._p), -np.min(self._p))
        
        if pmax > 0:
            rgb[2, self._p > 0] = self._p[self._p > 0] / pmax
            rgb[0, self._p < 0] = self._p[self._p < 0] / pmax
        
        return rgb

        
        
        
        
        
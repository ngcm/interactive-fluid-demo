import numpy as np

class Sim:
    
    
    def __init__(self, shape, diffusion, viscosity):    
        self._v = np.zeros((2, *shape))
        self._b = np.zeros(shape, dtype=bool)    
        self._notb = np.ones_like(self._b)
        self._tmp = np.zeros((2, *shape))   
        
        xs = np.arange(0.0, shape[0], 1)
        ys = np.arange(0.0, shape[1], 1)
        x, y = np.meshgrid(xs, ys)
        self._indexArray = np.array([x.T, y.T])        
        
        self._dx = np.array([4/3,1]) / np.array(shape)
        print(self._dx)
        
    def _updateadvect(self, dt):
        shape = np.shape(self._v[0])
        x = self._indexArray - dt * self._v / self._dx[:,np.newaxis,np.newaxis]
        x = np.array([np.clip(x[0], 0, shape[0] - 1.01),
                      np.clip(x[1], 0, shape[1] - 1.01)])
        advectIndex = np.array(x, dtype=int)
        advectInter = x - advectIndex                
        self._xi = advectIndex[:, self._notb]
        self._s = advectInter[:, self._notb]
        
    def _advect(self, x, x0):
        xi = self._xi[0]
        yi = self._xi[1]        
        s = self._s[0]
        t = self._s[1]
        
        x[self._notb] = (1 - s) * ((1 - t) * x0[xi, yi] 
            + t * x0[xi, yi + 1]) + s * ((1 - t) * x0[xi + 1, yi] 
            + t * x0[xi + 1, yi + 1])
        x[self._b] = x0[self._b]
            
    def _divergence(self, div, x0, dt):
        div[-1,:] = 0
        div[:-1, :] = x0[0, 1:, :] * self._notb[1:, :] / (2 * self._dx[0])
        div[1:, :] -= x0[0, :-1, :] * self._notb[:-1, :] / (2 * self._dx[0])
        div[:, :-1] += x0[1, :, 1:] * self._notb[:, 1:] / (2 * self._dx[1])
        div[:, 1:] -= x0[1, :, :-1] * self._notb[:, :-1] / (2 * self._dx[1])        
        div[self._b] = 0
           
    def _pressure_solve(self, p, div):
        p[:] = 0
        
        bound = 0.0 + self._b[0:-2,1:-1] + self._b[2:,1:-1] + self._b[1:-1,0:-2] + self._b[1:-1,2:]
        
        for i in range(10):
            p[1:-1,1:-1] = 1 / 4 * (p[1:-1,1:-1] * bound
                + p[0:-2,1:-1] * self._notb[0:-2,1:-1] 
                + p[2:,1:-1] * self._notb[2:,1:-1] 
                + p[1:-1,0:-2] * self._notb[1:-1,0:-2] 
                + p[1:-1,2:] * self._notb[1:-1,2:]                
                - self._dx[0] * self._dx[1] * div[1:-1,1:-1])
            p[self._b] = 0
            
    def _sub_gradient(self, v, v0, p, dt):        
        v[0, 1:-1, :] = v0[0, 1:-1, :] - 1 / (2 * self._dx[0]) * (p[2:, :] - p[:-2, :])
        v[1, :, 1:-1] = v0[1, :, 1:-1] - 1 / (2 * self._dx[1]) * (p[:, 2:] - p[:, :-2])
        
    def set_boundary(self, cell_ocupation):
        self._b = cell_ocupation
        self._notb = np.logical_not(self._b)
        
    def set_velocity(self, cells_to_set, cell_velocity):
        self._v[cells_to_set] = cell_velocity
        
    def get_velocity(self):
        return self._v
               
    def reset(self):
        self._v[:] = 0
        
    def step(self, dt, density_arrays):
        self._updateadvect(dt)
        self._advect(self._tmp[0], self._v[0])
        self._advect(self._tmp[1], self._v[1])
        self._divergence(self._v[0], self._tmp, dt)
        self._pressure_solve(self._v[1], self._v[0])
        self._sub_gradient(self._v, self._tmp, self._v[1], dt)
        self._v[:, self._b] = 0

        for d in density_arrays:
            self._advect(self._tmp[0], d)
            self._tmp[0, self._b] = 0
            d[:] = self._tmp[0]

        
        
        
        
        
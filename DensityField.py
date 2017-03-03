from numba import jit
import numpy as np

class DensityField:
    def __init__(self, shape):        
        self._shape = shape
        self._d = np.zeros((3, *shape), dtype=np.float32)
        
        self._colours = np.array([np.array([1.0, 0.8, 0.2]), 
                                  np.array([0.5, 1.0, 0.5]), 
                                  np.array([0.6, 0.3, 1.0])])
        self._num_colours = len(self._colours)
        
    @jit
    def update(self, flow_amount, perp_number, perp_amount):
        step = self._shape[1] // (perp_number + 1)
        ys = range(step, self._shape[1], step)
        
        rx = np.s_[:flow_amount]
        
        dy = step * perp_amount // 20
        for i, y in enumerate(ys):
            ry = np.s_[y - dy : y + dy + 1]
            self._d[i % self._num_colours, rx, ry] = 1
            
    @jit       
    def reset(self):
        self._d[:] = 0
            
    @property
    def field(self):
        return self._d
    
    @property
    def colour_field(self):
        return np.dot(self._colours, self._d.reshape(self._num_colours, 
            -1)).reshape(np.shape(self._d))
    
    @property
    def alpha(self):
        return np.clip(np.sqrt(np.sum(self._d, axis=0)), 0, 1)
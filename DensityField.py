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
    def update(self, mode, flow_amount, perp_amount):
        step = self._shape[1] // 10
        ys = range(step, self._shape[1], step)
        
        if mode == 0:            
            rx = np.s_[:flow_amount]
        elif mode == 1:   
            x = self._shape[0] // 2
            rx = np.s_[x - flow_amount // 2 : x + 1 + flow_amount // 2]
            
        dy = perp_amount // 2
        for i, y in enumerate(ys):
            ry = np.s_[y - dy : y + dy + 1]            
            colour = self._colours[i % self._num_colours]
            self._d[0, rx, ry] = colour[0]
            self._d[1, rx, ry] = colour[1]
            self._d[2, rx, ry] = colour[2]
            
    @jit       
    def reset(self):
        self._d[:] = 0
            
    @property
    def field(self):
        return self._d
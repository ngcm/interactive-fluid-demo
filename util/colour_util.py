from numba import jit
import numpy as np

@jit
def _hue_to_g(hue):
    hue = (hue + 0.6) % 1.0
    
    ret = np.zeros_like(hue)
    a = hue < 1 / 6
    b = np.logical_and(1 / 6 <= hue, hue< 0.5)
    c = np.logical_and(0.5 <= hue, hue < 4 / 6)
    # d = 4/6 <= hue
    ret[a] = hue[a] * 6.0
    ret[b] = 1
    ret[c] = (2/3 - hue[c]) * 6.0
    # ret[d] = 0
       
    return ret

@jit
def to_rgb(hue):
    # 0 - 1 to RGB
    return (_hue_to_g(hue + 1/3), _hue_to_g(hue), _hue_to_g(hue - 1/3))



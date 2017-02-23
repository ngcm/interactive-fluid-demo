import numpy as np
import colorsys

def _v(hue):
    hue = hue % 1.0
    
    ret = hue[:]    
    a = hue < 1 / 6
    b = np.logical_and(1 / 6 <= hue, hue< 0.5)
    c = np.logical_and(0.5 <= hue, hue < 4 / 6)
    d = 4/6 <= hue
    ret[a] = hue[a] * 6.0
    ret[b] = 1
    ret[c] = (2/3 - hue[c]) * 6.0
    ret[d] = 0
       
    return ret

def ToRGB(h):
    return (_v(h + 1/3), _v(h), _v(h - 1/3))
'''
def ToRGB(hue):
    # 0-1 to (r,g,b)
    return (np.abs(hue * 6 - 3) - 1, 2 - np.abs(hue * 6 - 2), 2 - np.abs(hue * 6 - 4))
'''
def ToHue(x, y):
    # (x,y) angle to 0-1
    return (1 + np.arctan2(y, x) / np.pi) / 2

def twoDtoRGB(x, y):
    # (x, y) to (r,g,b)
    return colorsys.hsl_to_rgb(ToHue(x, y), 1, 0.5 * np.sqrt(x**2 + y**2))

def velocity_field_to_RGB(v, power=0.5):
    assert np.shape(v)[0] == 2
    angles = np.arctan2(v[0], v[1])
    hues = (1 + angles / np.pi) / 2
    rel_rgb = ToRGB(hues) * (v[0]**2 + v[1]**2) ** power
    return rel_rgb / np.max(rel_rgb)




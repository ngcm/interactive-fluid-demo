import numpy as np

def ToRGB(hue):
    # 0-1 to (r,g,b)
    return (np.abs(hue * 6 - 3) - 1, 
            2 - np.abs(hue * 6 - 2),
            2 - np.abs(hue * 6 - 4))
    
def ToHue(x, y):
    # (x,y) angle to 0-1
    return (1 + np.arctan2(x, y) / np.pi) / 2

def twoDtoRGB(x, y):
    # (x, y) to (r,g,b)
    return ToRGB(ToHue(x, y)) * np.sqrt(x**2 + y**2)

def velocity_field_to_RGB(v):
    assert np.shape(v)[0] == 2
    angles = np.arctan2(v[0], v[1])
    hues = (1 + angles / np.pi) / 2
    rel_rgb = ToRGB(hues) * np.sqrt(v[0]**2 + v[1]**2)
    return rel_rgb / np.max(rel_rgb)
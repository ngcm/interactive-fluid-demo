from numba import jit
import numpy as np

import util.colour_util as colour_util

class SimBase:
    def __init__(self, shape):
        self._shape = shape
        self._v = np.zeros((2, *shape))
        self._b = np.zeros(shape, dtype=bool)
        self._notb = np.logical_not(self._b)
        self._dx = np.array([4/3,1]) / np.array(shape)
        self._hsv_field = np.zeros((3, *shape), dtype=np.uint8)
        self._hsv_field[1] = 255

    @jit
    def set_boundary(self, cell_ocupation):
        self._b = cell_ocupation
        self._notb = np.logical_not(self._b)

    @jit
    def set_velocity(self, cells_to_set, cell_velocity):
        self._v[cells_to_set] = cell_velocity

    @jit
    def get_velocity(self):
        return self._v

    @jit
    def reset(self):
        self._v[:] = 0

    @jit
    def get_velocity_field_as_HSV(self, power=0.5):
        # assert np.shape(self._v)[0] == 2
        self._hsv_field[0] = 180 * (np.arctan2(-self._v[0], -self._v[1]) / (2 * np.pi) + 0.5)
        self._hsv_field[2] = 255 * (self._v[0]**2 + self._v[1]**2) ** power
        return self._hsv_field

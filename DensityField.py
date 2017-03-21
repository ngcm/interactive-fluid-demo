from numba import jit
import numpy as np

# Class which encpsulates the smoke denisty scalar fields (one for each smoke)
# colour
class DensityField:
    def __init__(self, shape):
        self._shape = shape

        # colour matrix, maps N scalar density fields to RGB
        self._colours = np.array([np.array([1.0, 0.8, 0.2]),
                                  np.array([0.5, 1.0, 0.5]),
                                  np.array([0.6, 0.3, 1.0])]) * 0.5
        self._num_colours = len(self._colours)

        # density scalar fields
        self._d = np.zeros((self._num_colours, *shape), order='C')

        # temp arrays
        self._tmp = np.zeros((3, *shape), order='C')

    @jit
    def add_density(self, flow_amount, flow_direction, perp_number, perp_amount):
        # flow direction determines where smoke is added
        if flow_direction // 2 == 0:
            rx = np.s_[:flow_amount]
        else:
            rx = np.s_[-flow_amount:]

        # decay the smoke where it is input, cleans things up when smoke
        # amount changes
        if flow_direction == 0:
            self._d[:, rx, :] *= 0.9
        elif flow_direction == 1:
            self._d[:, :, rx] *= 0.9
        elif flow_direction == 2:
            self._d[:, rx, :] *= 0.9
        else:
            self._d[:, :, rx] *= 0.9
        self._d[:] *= 0.999 # make sure smoke decays eventually

        # create multiple smoke trails by stepping perpendicular to flow
        # direction
        length = self._shape[1 if flow_direction % 2 == 0 else 0]
        step = length // (perp_number + 1)
        ys = range(step, length, step)
        dy = int(step * perp_amount) // 2
        for i, y in enumerate(ys):
            ry = np.s_[y - dy : y + dy + 1]
            if flow_direction % 2 == 0:
                self._d[i % self._num_colours, rx, ry] = 1
            else:
                self._d[i % self._num_colours, ry, rx] = 1

    @jit
    def reset(self):
        self._d[:] = 0

    @property
    def field(self):
        return self._d

    @jit
    def get_render(self):
        # multiply by colour transform matrix to get RGB
        self._tmp[:] = np.tensordot(self._colours, self._d, axes=([-1],[0]))
        np.sqrt(self._tmp, out=self._tmp)
        return self._tmp

    @jit
    def get_alpha(self):
        # sum and clip smoke density so colours add to white
        np.sum(self._d, axis=0, out=self._tmp[0])
        np.sqrt(self._tmp[0], out=self._tmp[0])
        np.clip(self._tmp[0], 0, 1, out=self._tmp[0])
        return self._tmp[0]

from numba import jit
import numpy as np

class DensityField:
    def __init__(self, shape):
        self._shape = shape

        # colour matrix
        self._colours = np.array([np.array([1.0, 0.8, 0.2]),
                                  np.array([0.5, 1.0, 0.5]),
                                  np.array([0.6, 0.3, 1.0])]) * 0.5
        self._num_colours = len(self._colours)

        # density scalar fields
        self._d = np.zeros((self._num_colours, *shape), order='C')

        # temp arrays
        self._tmp = np.zeros((3, *shape), order='C')

    @jit
    def add_density(self, flow_amount, perp_number, perp_amount):
        step = self._shape[1] // (perp_number + 1)
        ys = range(step, self._shape[1], step)

        rx = np.s_[:flow_amount]

        self._d[:, rx, :] *= 0.9
        self._d[:] *= 0.999

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

    @jit
    def get_render(self):
        self._tmp[:] = np.tensordot(self._colours, self._d, axes=([-1],[0]))
        np.sqrt(self._tmp, out=self._tmp)
        return self._tmp

    @jit
    def get_alpha(self):
        np.sum(self._d, axis=0, out=self._tmp[0])
        np.sqrt(self._tmp[0], out=self._tmp[0])
        np.clip(self._tmp[0], 0, 1, out=self._tmp[0])
        return self._tmp[0]

import numpy as np

#import FluidSim

class StamFluidSim():
    
    _inner = np.s_[1:-1,1:-1]
    
    def __init__(self, shape, diffusion, viscosity):
        
        self._v = np.zeros((2, *shape))
        self._b = np.zeros(shape, dtype=bool)        
        self._tmp = np.zeros((2, *shape))   
        
        print('v=', np.shape(self._v))
        print('b=', np.shape(self._b))
        print('tmp=', np.shape(self._tmp))
        
        self._diff = diffusion
        self._visc = viscosity        
        
    def _set_bnd(self, x, f):        
        x[self._b] = 0
    
        if f==2:
            dfall = np.logical_and(self._b[:,:-1], np.logical_not(self._b[:,1:]))
            x[:,:-1][dfall] += -x[:,1:][dfall]            
            ufall = np.logical_and(np.logical_not(self._b[:,:-1]), self._b[:,1:])
            x[:,1:][ufall] += -x[:,:-1][ufall]
        elif f==1:
            lfall = np.logical_and(np.logical_not(self._b[:-1,:]), self._b[1:,:])
            x[1:,:][lfall] += -x[:-1,:][lfall]              
            rfall = np.logical_and(self._b[:-1,:], np.logical_not(self._b[1:,:]))
            x[:-1,:][rfall] += -x[1:,:][rfall]
        else:
            dfall = np.logical_and(self._b[:,:-1], np.logical_not(self._b[:,1:]))
            x[:,:-1][dfall] = x[:,1:][dfall]        
            ufall = np.logical_and(np.logical_not(self._b[:,:-1]), self._b[:,1:])
            x[:,1:][ufall] = x[:,:-1][ufall]
            lfall = np.logical_and(np.logical_not(self._b[:-1,:]), self._b[1:,:])
            x[1:,:][lfall] = x[:-1,:][lfall]        
            rfall = np.logical_and(self._b[:-1,:], np.logical_not(self._b[1:,:]))
            x[:-1,:][rfall] = x[1:,:][rfall]        
        
    def _lin_solve(self, x, xp, a, c, f):
        x[:] = 0
        for i in range(10):
            #print('lin_solve', i, x[10,10])
            x[StamFluidSim._inner] = 1 / c * (xp[StamFluidSim._inner] 
                + a * (x[0:-2,1:-1] + x[2:,1:-1] + x[1:-1,0:-2] + x[1:-1,2:]))
            self._set_bnd(x, f)
            
    def _diffuse(self, x, xp, diff, dt, f):
        #print('diffuse', x[10,10])
        a = dt * diff * np.prod(np.shape(x))
        self._lin_solve(x, xp, a, 1 + 4 * a, f)
        
    def _advect(self, d, d0, u, v, dt, f):
        shape = np.shape(d)
        
        dt0 = dt * np.sqrt(np.prod(shape))
        
        xs = np.arange(1, shape[0] - 1)
        ys = np.arange(1, shape[1] - 1)
        x, y = np.meshgrid(xs, ys)
        
        x = np.clip(x.T - dt0 * u[StamFluidSim._inner], 0.5, shape[0] - 1.5)
        y = np.clip(y.T - dt0 * v[StamFluidSim._inner], 0.5, shape[1] - 1.5)
        xi = np.array(x, dtype=int)
        yi = np.array(y, dtype=int)
        
        s = x - xi
        t = y - yi
        
        d[StamFluidSim._inner] = (1 - s) * ((1 - t) * d0[xi, yi] + t * d0[xi, yi + 1]) \
            + s * ((1 - t) * d0[xi + 1, yi] + t * d0[xi + 1, yi + 1])
            
        self._set_bnd(d, f)
        
    def _project(self, u, v, div, p):
        shape = np.shape(u)    
        N = np.sqrt(np.prod(shape))
        
        div[StamFluidSim._inner] = (-0.5 / N) * (u[2:, 1:-1] - u[:-2, 1:-1] 
                                    + v[1:-1, 2:] - v[1:-1, :-2])
        p[StamFluidSim._inner] = 0
        
        self._set_bnd(div, 0)
        self._set_bnd(p, 0)
        
        self._lin_solve(p, div, 1, 4, 0)
        
        u[StamFluidSim._inner] -= 0.5 * N * (p[2:, 1:-1] - p[:-2, 1:-1])
        v[StamFluidSim._inner] -= 0.5 * N * (p[1:-1, 2:] - p[1:-1, :-2])
        
        self._set_bnd(u, 1)
        self._set_bnd(v, 2)
                
    def _dens_step(self, dt, x):
        #add_source(x, x0, dt)
        self._diffuse(self._tmp[0], x, self._diff, dt, 0)
        self._advect(x, self._tmp[0], self._v[0], self._v[1], dt, 0)
        
    def _vel_step(self, dt):
        # add_source(u, u0, dt)
        # add_source(v, v0, dt)        
        # print('prediffuse', self._tmp[0,10,10])
        self._diffuse(self._tmp[0], self._v[0], self._visc, dt, 1)
        # print('postdiffuse', self._tmp[0, 10, 10])
        self._diffuse(self._tmp[1], self._v[1], self._visc, dt, 2)
        self._project(self._tmp[0], self._tmp[1], self._v[0], self._v[1])
        # print('postproject', self._tmp[0, 10, 10])
        self._advect(self._v[0], self._tmp[0], self._tmp[0], self._tmp[1], dt, 1)
        self._advect(self._v[1], self._tmp[1], self._tmp[0], self._tmp[1], dt, 2)
        self._project(self._v[0], self._v[1], self._tmp[0], self._tmp[1])
        
    def set_boundary(self, cell_ocupation):
        self._b = cell_ocupation
        
    def set_velocity(self, cells_to_set, cell_velocity):
        self._v[cells_to_set] = cell_velocity
               
    def reset(self):
        self._v[:] = 0
        
    def step(self, dt, density_arrays):        
        self._vel_step(dt)
        for d in density_arrays:
            self._dens_step(dt, d)
            
    def get_velocity(self):
        return self._v
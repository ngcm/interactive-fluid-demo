# Simple 2D Fluid Sim based on [STAM03](www.intpowertechcorp.com/GDC03.pdf)

import numpy as np

inner = np.s_[1:-1,1:-1]

def add_source(x, s, dt):
    x += dt * s

'''
def set_bnd(x, flag, box):

    dfall = np.logical_and(box[:,:-1], np.logical_not(box[:,1:]))
    x[:,:-1][dfall] = -x[:,1:][dfall] if flag == 2 else x[:,1:][dfall]
    ufall = np.logical_and(np.logical_not(box[:,:-1]), box[:,1:])
    x[:,1:][ufall] = -x[:,:-1][ufall] if flag == 2 else x[:,:-1][ufall]
    
    lfall = np.logical_and(np.logical_not(box[:-1,:]), box[1:,:])
    x[1:,:][lfall] = -x[:-1,:][lfall] if flag == 1 else x[:-1,:][lfall] 
    rfall = np.logical_and(box[:-1,:], np.logical_not(box[1:,:]))
    x[:-1,:][rfall] = -x[1:,:][rfall] if flag == 1 else x[1:,:][rfall] 
  
    center = np.logical_and(box[1:-1,1:-1], 
                np.logical_and(box[:-2,1:-1], 
                    np.logical_and(box[2:,1:-1],
                        np.logical_and(box[1:-1,:-2], box[1:-1,2:]))))
    x[inner][center] = 0
'''
  
def set_bnd(x, flag, box):
    x[box] = 0

    if(flag==2):
        dfall = np.logical_and(box[:,:-1], np.logical_not(box[:,1:]))
        x[:,:-1][dfall] += -x[:,1:][dfall]
        
        ufall = np.logical_and(np.logical_not(box[:,:-1]), box[:,1:])
        x[:,1:][ufall] += -x[:,:-1][ufall]
    elif(flag==1):
        lfall = np.logical_and(np.logical_not(box[:-1,:]), box[1:,:])
        x[1:,:][lfall] += -x[:-1,:][lfall]  
        
        rfall = np.logical_and(box[:-1,:], np.logical_not(box[1:,:]))
        x[:-1,:][rfall] += -x[1:,:][rfall]
    else:
        dfall = np.logical_and(box[:,:-1], np.logical_not(box[:,1:]))
        x[:,:-1][dfall] = x[:,1:][dfall]        
        ufall = np.logical_and(np.logical_not(box[:,:-1]), box[:,1:])
        x[:,1:][ufall] = x[:,:-1][ufall]
        lfall = np.logical_and(np.logical_not(box[:-1,:]), box[1:,:])
        x[1:,:][lfall] = x[:-1,:][lfall]        
        rfall = np.logical_and(box[:-1,:], np.logical_not(box[1:,:]))
        x[:-1,:][rfall] = x[1:,:][rfall]

    
    
def lin_solve(x, x0, a, c, flag, box):
    for _ in range(20):
        x[inner] = (x0[inner] + a 
            * (x[0:-2,1:-1] + x[2:,1:-1] + x[1:-1,0:-2] + x[1:-1,2:])) / c
        set_bnd(x, flag, box)

def diffuse(x, x0, diff, dt, flag, box):
    a = dt * diff * np.prod(np.shape(x))
    lin_solve(x, x0, a, 1 + 4 * a, flag, box)
    
def advect(d, d0, u, v, dt, flag, box):
    shape = np.shape(d)
    
    dt0 = dt * np.sqrt(np.prod(shape))
    
    xs = np.arange(1, shape[0] - 1)
    ys = np.arange(1, shape[1] - 1)
    x, y = np.meshgrid(xs, ys)
    
    x = np.clip(x.T - dt0 * u[inner], 0.5, shape[0] - 1.5)
    y = np.clip(y.T - dt0 * v[inner], 0.5, shape[1] - 1.5)
    xi = np.array(x, dtype=int)
    yi = np.array(y, dtype=int)
    
    s = x - xi
    t = y - yi
    
    d[inner] = (1 - s) * ((1 - t) * d0[xi, yi] + t * d0[xi, yi + 1])         + s * ((1 - t) * d0[xi + 1, yi] + t * d0[xi + 1, yi + 1])
        
    set_bnd(d, flag, box)
        
def project(u, v, p, div, box):
    shape = np.shape(u)    
    N = np.sqrt(np.prod(shape))
    
    div[inner] = (-0.5 / N) * (u[2:, 1:-1] - u[:-2, 1:-1] + v[1:-1, 2:] - v[1:-1, :-2])
    p[inner] = 0
    
    set_bnd(div, 0, box)
    set_bnd(p, 0, box)
    
    lin_solve(p, div, 1, 4, 0, box)
    
    u[inner] -= 0.5 * N * (p[2:, 1:-1] - p[:-2, 1:-1])
    v[inner] -= 0.5 * N * (p[1:-1, 2:] - p[1:-1, :-2])
    
    set_bnd(u, 1, box)
    set_bnd(v, 2, box)
    
def dens_step(x, x0, u, v, diff, dt, box):
    add_source(x, x0, dt)
    diffuse(x0, x, diff, dt, 0, box)
    advect(x, x0, u, v, dt, 0, box)
    
def vel_step(u, v, u0, v0, visc, dt, box):
    add_source(u, u0, dt)
    add_source(v, v0, dt)
    diffuse(u0, u, visc, dt, 1, box)
    diffuse(v0, v, visc, dt, 2, box)
    project(u0, v0, u, v, box)
    advect(u, u0, u0, v0, dt, 1, box)
    advect(v, v0, u0, v0, dt, 2, box)
    project(u, v, u0, v0, box)

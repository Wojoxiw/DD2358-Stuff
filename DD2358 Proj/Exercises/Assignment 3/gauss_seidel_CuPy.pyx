import numpy as np
import cupy as cp

def gauss_seidel_CuPy_fn(f):
    f = cp.copy(f)
    newf = f.copy()
    
    sizex = newf.shape()[0]
    sizey = newf.shape()[1]
    
    ## roll then set the edges to 0 again
    newf = 0.25 * ( cp.roll(f, 1, 0) + cp.roll(f, -1, 0) + cp.roll(f, 1, 1) + cp.roll(f,-1, 1) )
    
    newf[0, :] = 0
    newf[sizex, :] = 0
    newf[:, 0] = 0
    newf[:, sizey] = 0
    
    return newf
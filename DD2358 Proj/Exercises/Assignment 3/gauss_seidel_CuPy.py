import numpy as np
import cupy as cp

def gauss_seidel_CuPy_fn(f):
    sizex = np.shape(f)[0]
    sizey = np.shape(f)[1]
    
    f = cp.asarray(f)
    
    ## roll then set the edges to 0 again
    newf = 0.25 * ( cp.roll(f, 1, 0) + cp.roll(f, -1, 0) + cp.roll(f, 1, 1) + cp.roll(f,-1, 1) )
    
    newf[0, :] = 0
    newf[sizex-1, :] = 0
    newf[:, 0] = 0
    newf[:, sizey-1] = 0
    
    return newf
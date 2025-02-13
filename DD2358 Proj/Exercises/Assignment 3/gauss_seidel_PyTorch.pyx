import numpy as np
import torch
from torch import (roll, zeros)

def gauss_seidel_PyTorch_fn(f):
    f = f.cuda()
    newf = f.copy()
    newf = newf.cuda()
    sizex = newf.shape()[0]
    sizey = newf.shape()[1]
    
    #newf[1:sizex-1, 1:sizey-1]+= f[0:sizex-2, 1:sizey-1]  ## left-side
    #newf[1:sizex-1, 1:sizey-1]+= f[2:sizex, 1:sizey-1]  ## right-side
    #newf[1:sizex-1, 1:sizey-1]+= f[1:sizex-1, 0:sizey-1]  ## top-side
    #newf[1:sizex-1, 1:sizey-1]+= f[1:sizex-1, 2:sizey]  ## bottom-side
    
    ## or roll then set the edges to 0 again
    newf = 0.25 * ( np.roll(f, 1, 0) + np.roll(f, -1, 0) + np.roll(f, 1, 1) + np.roll(f,-1, 1) )
    
    newf[0, :] = 0
    newf[sizex, :] = 0
    newf[:, 0] = 0
    newf[:, sizey] = 0
    
    return newf

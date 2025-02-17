import numpy as np
import torch
from torch import (roll, zeros)

def gauss_seidel_PyTorch_fn(f):
    sizex = np.shape(f)[0]
    sizey = np.shape(f)[1]
    
    f = np.random.random_sample((sizex, sizey)) ## need to remake the array, otherwise it somehow sets it as a Tensor, but doesnt let me use it like one
    f2 = torch.from_numpy(f).cuda()
    newf = zeros((sizex, sizey)).cuda()
    
    ## roll then set the edges to 0 again
    newf = 0.25 * ( roll(f2, 1, 0) + roll(f2, -1, 0) + roll(f2, 1, 1) + roll(f2,-1, 1) )
    
    newf[0, :] = 0
    newf[sizex-1, :] = 0
    newf[:, 0] = 0
    newf[:, sizey-1] = 0
    
    return newf
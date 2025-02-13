#cython: boundscheck=False
'''
Created on 11 feb. 2025

@author: al8032pa
'''
import numpy as np
cimport numpy as np
import line_profiler
import atexit
#profile = line_profiler.LineProfiler()
#atexit.register(profile.print_stats)

#@profile
def gauss_seidel_cython_fn(double[:, :] f):
    newf = f.copy()
    cdef unsigned int sizex, sizey, i, j
    sizex = newf.shape[0]
    sizey = newf.shape[1]
    
    #newf[1:sizex-1, 1:sizey-1]+= f[0:sizex-2, 1:sizey-1]  ## left-side
    #newf[1:sizex-1, 1:sizey-1]+= f[2:sizex, 1:sizey-1]  ## right-side
    #newf[1:sizex-1, 1:sizey-1]+= f[1:sizex-1, 0:sizey-1]  ## top-side
    #newf[1:sizex-1, 1:sizey-1]+= f[1:sizex-1, 2:sizey]  ## bottom-side
    
    for i in range(1,sizex-1):
        for j in range(1,sizey-1):
            newf[i,j] = 0.25 * (newf[i,j+1] + newf[i,j-1] + newf[i+1,j] + newf[i-1,j])
    
    return newf

'''
Created on 11 feb. 2025

@author: al8032pa
'''
import numpy as np
from timeit import default_timer as timer
from functools import wraps
import cProfile, pstats
import line_profiler
from matplotlib import pyplot as plt
import line_profiler
import atexit
profile = line_profiler.LineProfiler()
atexit.register(profile.print_stats)

def gauss_seidel_numpy(f):
    newf = f.copy()
    for i in range(1,newf.shape[0]-1):
        for j in range(1,newf.shape[1]-1):
            newf[i,j] = 0.25 * (newf[i,j+1] + newf[i,j-1] +
                                   newf[i+1,j] + newf[i-1,j])
    
    return newf

def gauss_seidel_iterator(x, fun, nIterations): ## plots performance by grid size
    for i in range(nIterations):    
        x = fun(x)
    return x
        
    
        
    
def task1p1():
    print('Task 1.1: Gauss-Seidal solver with numpy')
    nIterations = 1000
    gridSizes = np.arange(10, 300, 10)
    times = np.zeros(np.shape(gridSizes))
    #initialize grid
    for j in range(len(gridSizes)):
        if(j%5 == 1):
            print(f'j = {j}/{np.size(gridSizes)}')
        size = gridSizes[j]
        timeStart = timer()
        x = np.random.random_sample((size, size))
        gauss_seidel_iterator(x, gauss_seidel_numpy, nIterations)
        times[j] = timer() - timeStart
    
    plt.xlabel('Grid Size')
    plt.ylabel('Computation Time [s]')
    plt.yscale('log')
    plt.plot(gridSizes, times)
    plt.grid()
    plt.tight_layout()
    plt.show()
    
def task1p2():
    print('Task 1.2: Profiling the Gauss-Seidal solver')
    nIterations = 1000
    gridSize = 100
    
    
    gauss_seidel_iterator(np.random.random_sample((gridSize, gridSize)), gauss_seidel_numpy, nIterations)
    
if __name__ == '__main__':
    #task1p1()
    task1p2()
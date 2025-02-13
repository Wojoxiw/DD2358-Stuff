'''
Created on 11 feb. 2025

@author: al8032pa
'''
import numpy as np
from timeit import default_timer as timer
from functools import wraps
import cProfile, pstats
from matplotlib import pyplot as plt
import pyximport; pyximport.install(pyimport=True) ## to not have to run the setup file?
from gauss_seidel_cython import gauss_seidel_cython_fn as cythonfn
from gauss_seidel_PyTorch import gauss_seidel_PyTorch_fn as PyTorchfn
from gauss_seidel_CuPy import gauss_seidel_CuPy_fn as CuPyfn
import h5py
import line_profiler
import atexit
#profile = line_profiler.LineProfiler()
#atexit.register(profile.print_stats)

#@profile
def gauss_seidel_numpy(f):
    newf = f.copy()
    for i in range(1,newf.shape[0]-1):
        for j in range(1,newf.shape[1]-1):
            newf[i,j] = 0.25 * (newf[i,j+1] + newf[i,j-1] +
                                   newf[i+1,j] + newf[i-1,j])
    
    return newf

#@profile
def gauss_seidel_iterator(x, fun, nIterations): ## plots performance by grid size
    for i in range(nIterations):    
        x = fun(x)
    return x
        
    
        
    
def task1p1():
    print('Task 1.1: Gauss-Seidal solver with numpy')
    nIterations = 1000
    gridSizes = np.arange(10, 300, 10)
    times = np.zeros(np.shape(gridSizes))
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
    ## after activating @profile
    gauss_seidel_iterator(np.random.random_sample((gridSize, gridSize)), gauss_seidel_numpy, nIterations)
    
def task1p4():
    print('Task 1.4: Gauss-Seidal solver: cython vs original')
    nIterations = 1000
    gridSizes = np.arange(10, 300, 10)
    times = np.zeros(np.shape(gridSizes))
    for j in range(len(gridSizes)):
        if(j%5 == 1):
            print(f'j = {j}/{np.size(gridSizes)}')
        size = gridSizes[j]
        timeStart = timer()
        x = np.random.random_sample((size, size))
        gauss_seidel_iterator(x, gauss_seidel_numpy, nIterations)
        times[j] = timer() - timeStart
    timesCython = np.zeros(np.shape(gridSizes))
    for j in range(len(gridSizes)):
        if(j%5 == 1):
            print(f'Cython, j = {j}/{np.size(gridSizes)}')
        size = gridSizes[j]
        timeStart = timer()
        x = np.random.random_sample((size, size))
        gauss_seidel_iterator(x, cythonfn, nIterations)
        timesCython[j] = timer() - timeStart
    
    plt.xlabel('Grid Size')
    plt.ylabel('Computation Time [s]')
    plt.yscale('log')
    plt.plot(gridSizes, timesCython, label = 'cython')
    plt.plot(gridSizes, times, label = 'simple numpy')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
    
def task1p7():
    print('Task 1.7: Gauss-Seidal solver: cython vs original vs CuPy vs PyTorch')
    nIterations = 1000
    gridSizes = np.arange(10, 300, 10)
    times = np.zeros(np.shape(gridSizes))
    timesCython = np.zeros(np.shape(gridSizes))
    timesPyTorch = np.zeros(np.shape(gridSizes))
    timesCuPy = np.zeros(np.shape(gridSizes))
    for j in range(len(gridSizes)):
        if(j%5 == 1):
            print(f'j = {j}/{np.size(gridSizes)}')
        size = gridSizes[j]
        x = np.random.random_sample((size, size))
        
        timeStart = timer()
        gauss_seidel_iterator(x, gauss_seidel_numpy, nIterations)
        times[j] = timer() - timeStart
        
        timeStart = timer()
        gauss_seidel_iterator(x, cythonfn, nIterations)
        timesCython[j] = timer() - timeStart
        
        timeStart = timer()
        gauss_seidel_iterator(x, PyTorchfn, nIterations)
        timesPyTorch[j] = timer() - timeStart
        
        timeStart = timer()
        gauss_seidel_iterator(x, CuPyfn, nIterations)
        timesCuPy[j] = timer() - timeStart
        
    plt.xlabel('Grid Size')
    plt.ylabel('Computation Time [s]')
    plt.yscale('log')
    plt.plot(gridSizes, timesCython, label = 'cython')
    plt.plot(gridSizes, timesCuPy, label = 'CuPy')
    plt.plot(gridSizes, timesPyTorch, label = 'PyTorch')
    plt.plot(gridSizes, times, label = 'simple numpy')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
    
def task1p8():
    print('Task 1.8: Saving the grids with h5py')
    nIterations = 1000
    size = 300
    x = np.random.random_sample((size, size))
    newx = gauss_seidel_iterator(x, cythonfn, nIterations)
    
    file = h5py.File('grids.hdf5', 'w')
    file.create_dataset('original_grid', data = x)
    file.create_dataset('iterated_grid', data = newx)
    
if __name__ == '__main__':
    #task1p1()
    #task1p2()
    #task1p4()
    task1p7()
    #task1p8()
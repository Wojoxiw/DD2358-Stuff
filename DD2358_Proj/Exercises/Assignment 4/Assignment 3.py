'''
Created on 17 feb. 2025

@author: al8032pa
'''
import numpy as np
from timeit import default_timer as timer
from functools import wraps
import cProfile, pstats
from matplotlib import pyplot as plt
import h5py
      
    
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

    
if __name__ == '__main__':
    task1p1()
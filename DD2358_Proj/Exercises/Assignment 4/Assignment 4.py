'''
Created on 17 feb. 2025

@author: al8032pa
'''
import numpy as np
from timeit import default_timer as timer
from matplotlib import pyplot as plt
import wildfires
import itertools
import multiprocessing
    
def task1p1():
    print('Task 1.1: Parallelization with multiprocessing')
    numProcesses = 42
    with multiprocessing.Pool(processes = numProcesses) as pool:
        treesOnFire = pool.map(wildfires.simulate_wildfire_serial, list(range(numProcesses))) ## needs dummy input, it seems
    
    plt.xlabel('Days')
    plt.ylabel('# of Trees Burning')
    plt.title('Wildfire Spread Over Time')
    
    treesOnFireArray = np.array(list(itertools.zip_longest(*treesOnFire, fillvalue=0))).T
    
    avgTreesOnFire = np.mean(treesOnFireArray, axis=0)
    plt.plot(range(np.shape(treesOnFireArray)[1]), avgTreesOnFire, label = 'Average')
    
    for i in range(len(treesOnFire)):
        days = range(len(treesOnFire[i]))
        if(i == 0):
            plt.plot(days, treesOnFire[i], linewidth = 0.15, label = 'Multiprocess #'+str(i+1))
        else:
            plt.plot(days, treesOnFire[i], linewidth = 0.15)
        
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
    
def task1p2():
    print('Task 1.2: Parallelization with Dask')
    
def task1p3():
    print('Task 1.3: Comparing execution times')
    
def task1p4():
    print('Task 1.4: VTK+Paraview Visualization')
    wildfire_serial.simulate_wildfire_serial_vtkwriting() ## writes the grid to grid.vtk for each day
    
if __name__ == '__main__':
    task1p1()
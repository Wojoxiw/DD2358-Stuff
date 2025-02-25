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
import dask
from dask.distributed import Client, LocalCluster, as_completed
from functools import wraps
from timeit import default_timer as timer
import dask.array as da
from wildfires import GRID_SIZE, DAYS

def wildfires_parallel(numRuns):
    with multiprocessing.Pool(processes = 20) as pool:
        treesOnFire = pool.map(wildfires.simulate_wildfire_serial, list(range(numRuns))) ## needs dummy input, it seems
    return treesOnFire

def task1p1():
    print('Task 1.1: Parallelization with multiprocessing')
    numRuns = 12
    treesOnFire = wildfires_parallel(numRuns)
    treesOnFireArray = np.array(list(itertools.zip_longest(*treesOnFire, fillvalue=0))).T
    avgTreesOnFire = np.mean(treesOnFireArray, axis=0)
    plt.plot(range(np.shape(treesOnFireArray)[1]), avgTreesOnFire, label = 'Average')
    
    for i in range(len(treesOnFire)):
        days = range(len(treesOnFire[i]))
        if(i == 0):
            plt.plot(days, treesOnFire[i], linewidth = 0.15, label = 'Multiprocess #'+str(i+1))
        else:
            plt.plot(days, treesOnFire[i], linewidth = 0.15)
        
    plt.xlabel('Days')
    plt.ylabel('# of Trees Burning')
    plt.title('Wildfire Spread Over Time')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
    
def wildfires_dask(numRuns, client, chunkSize = DAYS, VTKing = False):
    
    #===========================================================================
    # i=0 # this did not work
    # for fires in treesOnFire:
    #     fire = fires.compute()
    #     if(i==0): ## first element set up the array
    #         treesOnFireArray = fire
    #     else:
    #         treesOnFireArray = da.vstack((treesOnFireArray, fire))
    #     i+=1
    #===========================================================================
    
    futures = [client.submit(wildfires.simulate_wildfire_dask, chunkSize, i, VTKing) for i in range(numRuns)] ## seems to require the 'i' input to actually randomize values, otherwise they are all the same
    #treesOnFire = client.gather(futures)
    i=0 # delayed version - this does work
    for future, result in as_completed(futures, with_results=True):
        if(i==0): ## first element set up the array
            treesOnFireArray = result
        else:
            treesOnFireArray = da.vstack((treesOnFireArray, result))
        #print('i =', i, result.compute())
        i+=1
        
    #===========================================================================
    # daskResults = [wildfires.simulate_wildfire_dask() for _ in range(numRuns)]
    # treesOnFire = dask.compute(*daskResults)
    # #dask.visualize(*daskResults) ## creates a very wide image...
    # i=0 # this does work
    # for result in treesOnFire:
    #     if(i==0): ## first element set up the array
    #         treesOnFireArray = result
    #     else:
    #         treesOnFireArray = da.vstack((treesOnFireArray, result))
    #     #print('i =', i, result.compute())
    #     i+=1
    #===========================================================================
    
    return treesOnFireArray
    
def task1p2():
    print('Task 1.2: Parallelization with Dask')
    client = Client()
    print('Client dashboard running at:', client.dashboard_link)
    numRuns = 50
    
    treesOnFireArray = wildfires_dask(numRuns, client)
    
    avgTreesOnFire = da.mean(treesOnFireArray, axis=0)
    plt.plot(range(da.shape(treesOnFireArray)[1]), avgTreesOnFire, label = 'Average')
    
    for i in range(da.shape(treesOnFireArray)[0]):
        treesOnFire = treesOnFireArray[i]
        days = range(len(treesOnFire))
        if(i == 0):
            plt.plot(days, treesOnFire, linewidth = 0.15, label = 'Multiprocess #'+str(i+1))
        else:
            plt.plot(days, treesOnFire, linewidth = 0.15)
        
    plt.xlabel('Days')
    plt.ylabel('# of Trees Burning')
    plt.title('Wildfire Spread Over Time')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
    
def task1p3():
    print('Task 1.3: Comparing execution times')
    numRuns = 200
    
    cluster = LocalCluster()
    client = Client(cluster)
    numWorkers = 1
    cluster.scale(numWorkers)
    print('Client dashboard running at:', client.dashboard_link)
    t1 = timer()
    wildfires_dask(numRuns, client, chunkSize = GRID_SIZE)
    tdask = timer()-t1
    print('Average simulation runtime for dask with one-chunk results ('+str(numRuns)+' runs):',tdask/numRuns,'s, numWorkers = ',str(numWorkers))
    numWorkers = 2
    cluster.scale(numWorkers)
    t1 = timer()
    wildfires_dask(numRuns, client, chunkSize = GRID_SIZE)
    tdask = timer()-t1
    print('Average simulation runtime for dask with one-chunk results ('+str(numRuns)+' runs):',tdask/numRuns,'s, numWorkers = ',str(numWorkers))
    numWorkers = 4
    cluster.scale(numWorkers)
    t1 = timer()
    wildfires_dask(numRuns, client, chunkSize = GRID_SIZE)
    tdask = timer()-t1
    print('Average simulation runtime for dask with one-chunk results ('+str(numRuns)+' runs):',tdask/numRuns,'s, numWorkers = ',str(numWorkers))
    numWorkers = 20
    cluster.scale(numWorkers)
    t1 = timer()
    wildfires_dask(numRuns, client, chunkSize = GRID_SIZE)
    tdask = timer()-t1
    print('Average simulation runtime for dask with one-chunk results ('+str(numRuns)+' runs):',tdask/numRuns,'s, numWorkers = ',str(numWorkers))
    t1 = timer()
    wildfires_dask(numRuns, client, chunkSize = int(GRID_SIZE/2))
    tdask = timer()-t1
    print('Average simulation runtime for dask with two-chunk results ('+str(numRuns)+' runs):',tdask/numRuns,'s, numWorkers = ',str(numWorkers))
    t1 = timer()
    wildfires_dask(numRuns, client, chunkSize = int(GRID_SIZE/8))
    tdask = timer()-t1
    print('Average simulation runtime for dask with eight-chunk results ('+str(numRuns)+' runs):',tdask/numRuns,'s, numWorkers = ',str(numWorkers))
    t1 = timer()
    wildfires_dask(numRuns, client, chunkSize = int(GRID_SIZE/30))
    tdask = timer()-t1
    print('Average simulation runtime for dask with 30-chunk results ('+str(numRuns)+' runs):',tdask/numRuns,'s, numWorkers = ',str(numWorkers))
    
    t1 = timer()
    wildfires_parallel(numRuns)
    tmultiprocess = timer()-t1
    print('Average simulation runtime for multiprocessing pool ('+str(numRuns)+' runs):',tmultiprocess/numRuns,'s')
    
    t1 = timer()
    for i in range(numRuns):
        wildfires.simulate_wildfire_serial()
    tserial = timer()-t1
    print('Average simulation runtime for serial simulations ('+str(numRuns)+' runs):',tserial/numRuns,'s')
    
def task1p4():
    print('Task 1.4: Visualizing the grid')
    client = Client()
    print('Client dashboard running at:', client.dashboard_link)
    treesOnFireArray = wildfires_dask(1, client, VTKing = True)
    
    
if __name__ == '__main__':
    #task1p1()
    #task1p2()
    #task1p3()
    #task1p4()
'''
Created on 4 feb. 2025

@author: al8032pa
'''
import numpy as np
import time
from timeit import default_timer as timer
from functools import wraps
import cProfile, pstats
import line_profiler
import JuliaSet
import DiffusionProcess

def task1p1(): ## task 1.1
    def checktick(timething):
        M = 2000
        timesfound = np.empty((M,))
        for i in range(M):
            t1 = timething() # get timestamp from timer
            t2 = timething() # get timestamp from timer
            while (t2 - t1) < 1e-16: # if zero then we are below clock granularity, retake timing
                t2 = timething() # get timestamp from timer
            timesfound[i] = t2 # record the time stamp
        Delta = np.diff(timesfound) # it should be cast to int only when needed
        minDelta = Delta.min()
        return minDelta
    
    print('Task 1.1: Clock granularities:')
    print(f'Result for timeit: {checktick(timer):.4e}')
    print(f'Result for time: {checktick(time.time):.4e}')
    print(f'Result for time_ns: {checktick(time.time_ns)*1e-9:.4e}')
    
def task1p2():
    print('Task 1.2: Timing the Julia set code functions:')
    JuliaSet.calc_pure_python(1000, 300)
    
def task1p3():
    print('Task 1.3: Profiling the Julia set code:')
    
    #===========================================================================
    # ## with cProfile
    # profiler = cProfile.Profile()
    # #profiler.enable()
    # JuliaSet.calc_pure_python(1000, 300, profiler = profiler)
    # #profiler.disable()
    # stats = pstats.Stats(profiler).sort_stats('tottime')
    # stats.strip_dirs()
    # stats.print_stats()
    # stats.dump_stats('export-data')
    #===========================================================================
     
    # with line_profiler (activated via @profile in JuliaSet.py)
    JuliaSet.calc_pure_python(1000, 300)
    
def task1p4():
    print('Task 1.4: Memory-profiling the Julia set code:')
    JuliaSet.calc_pure_python(1000, 300)
    # with memory_profiler (activated via @profile in JuliaSet.py)
    
def task2p1():
    print('Task 2.1: Profiling the diffusion code:')
    
    ## with cProfile
    profiler = cProfile.Profile()
    profiler.enable()
    DiffusionProcess.run_experiment(100)
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('tottime')
    stats.strip_dirs()
    stats.print_stats()
    stats.dump_stats('export-data2')
     
    #===========================================================================
    # # with line_profiler (activated via @profile in JuliaSet.py)
    # DiffusionProcess.run_experiment(100)
    #===========================================================================
    
def task2p2():
    print('Task 2.2: Memory-profiling the diffusion code:')
    DiffusionProcess.run_experiment(100)
    # with memory_profiler (activated via @profile in DiffusionProcess.py)
    
    
if __name__ == '__main__':
    #task1p1()
    #task1p2()
    #task1p3()
    #task1p4()
    
    #task2p1()
    task2p2()

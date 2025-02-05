'''
Created on 4 feb. 2025

@author: al8032pa
'''
import numpy as np
import time
from timeit import default_timer as timer

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
    

if __name__ == '__main__':
    task1p1()
'''
Created on 4 feb. 2025

@author: al8032pa
'''

import pytest
import numpy as np
from timeit import default_timer as timer
from functools import wraps

def timefn(fn):
    @wraps(fn)
    def measure_time(*args, **kwargs):
        numTries = 10
        times = np.zeros(numTries)
        for i in range(numTries):
            t1 = timer()
            result = fn(*args, **kwargs)
            t2 = timer()
            times[i] = t2-t1
        print(f"@timefn: {fn.__name__} took an average of {np.mean(times):.2e} seconds, with a standard deviation of {np.std(times):.2e} s.")
        return result
    return measure_time

@timefn
def DGEMMarray(A, B, C): ## implementation using loops and arrays
    N = np.shape(A)[0]
    result = np.zeros((N, N), dtype=float)
    
    for i in range(N):
        for j in range(N):
            result[i, j] = C[i, j]
            for k in range(N):
                result[i, j]+= A[i, k]*B[k, j]
                
    return result

@timefn
def DGEMMnumpy(A, B, C): ## numpy implementation
    return C + A@B

@timefn
def DGEMMlist(A, B, C): ### implementation using as many lists as possible, and loops
    N = np.shape(A)[0]
    result = []
    for i in range(N):
        resulti = []
        for j in range(N):
            resultj = C[i, j]
            for k in range(N):
                resultj += A[i, k]*B[k, j]
            resulti.append(resultj)
        result.append(resulti)
    return result

def task2p3():
    print('Task 2.3: Timing the DGEMM functions:')
    Ns = [10, 100, 1000] ## array size
    for N in Ns:
        print('N =',N)
        A = np.random.rand(N, N)
        B = np.random.rand(N, N)
        C = np.random.rand(N, N)
        
        ans = DGEMMarray(A, B, C)
        ans = DGEMMnumpy(A, B, C)
        ans = DGEMMlist(A, B, C)
    
if __name__ == '__main__':
    task2p3()
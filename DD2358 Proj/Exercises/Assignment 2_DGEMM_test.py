'''
Created on 4 feb. 2025

@author: al8032pa
'''

from Assignment_2_DGEMMstuff import DGEMMarray, DGEMMnumpy, DGEMMlist
import pytest
import numpy as np

@pytest.fixture(autouse=True)
def get_DGEMM_test_data():
    N = 10
    return [(np.identity(N), np.identity(N), np.identity(N), np.identity(N)*2)]

def test_DGEMMarray(get_DGEMM_test_data):
    for data in get_DGEMM_test_data:
        A = data[0]
        B = data[1]
        C = data[2]
        assert np.allclose(DGEMMarray(A, B, C), data[3] )
        
def test_DGEMMlist(get_DGEMM_test_data):
    for data in get_DGEMM_test_data:
        A = data[0]
        B = data[1]
        C = data[2]
        assert np.allclose(DGEMMlist(A, B, C), data[3] )
        
def test_DGEMMnumpy(get_DGEMM_test_data):
    for data in get_DGEMM_test_data:
        A = data[0]
        B = data[1]
        C = data[2]
        assert np.allclose(DGEMMnumpy(A, B, C), data[3] )
    
if __name__ == '__main__':
    args_str = "-v -s"
    args = args_str.split(" ")
    retcode = pytest.main(args)
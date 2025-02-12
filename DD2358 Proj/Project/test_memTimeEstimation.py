from Scatt3D import memTimeEstimation
import pytest
import numpy as np

@pytest.fixture(autouse=True)
def get_test_data():
    try1 = [1e5, 1] # [num_cells, num_freqs] - small
    try2 = [1e6, 40] # large
    return [(try1, try2)]

def test_memEst(get_DGEMM_test_data):
    for data in get_DGEMM_test_data:
        estMem, estTime = memTimeEstimation(data[0], data[1])
        assert (estMem > 1) & (estTime > 1)
        assert (estMem < 1e30) & (estTime < 1e30)
    
if __name__ == '__main__':
    args_str = "-v -s"
    args = args_str.split(" ")
    retcode = pytest.main(args)
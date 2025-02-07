'''
Created on 4 feb. 2025

@author: al8032pa
'''

import JuliaSet
import pytest

@pytest.fixture(autouse=True)
def get_JuliaSet_test_data():
    return [(1000,300,33219980)]
        
def test_JuliaSet(get_JuliaSet_test_data):
    for data in get_JuliaSet_test_data:
        assert sum(JuliaSet.calc_pure_python(data[0], data[1])) == data[2]
    
if __name__ == '__main__':
    args_str = "-v -s"
    args = args_str.split(" ")
    retcode = pytest.main(args)
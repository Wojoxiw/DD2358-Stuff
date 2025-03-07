import os
os.environ["OMP_NUM_THREADS"] = "1" # seemingly needed for MPI speedup
from mpi4py import MPI
from scripts import memTimeEstimation
import meshMaker
import scatteringProblem
import pytest
import numpy as np
import sys, petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc

@pytest.fixture(autouse=True)
def get_test_data():
    try1 = [600000, 1] # [num_cells, num_freqs] - small
    try2 = [2000000, 40] # large
    return [try1, try2]

def test_memEst(get_test_data): ## check that memTimeEstimation returns sensible (a.k.a. finite and positive) estimates of time and memory costs
    for data in get_test_data:
        print(data[0], data[1])
        estMem, estTime = memTimeEstimation(data[0], data[1])
        assert (estMem > 1) & (estTime > 1)
        assert (estMem < 1e30) & (estTime < 1e30)

def test_PETScType(get_test_data): ## PETSc check from https://docs.fenicsproject.org/dolfinx/v0.9.0/python/demos/demo_axis.html
    # The time-harmonic Maxwell equation is complex-valued PDE. PETSc
    # must therefore have compiled with complex scalars. This has never been a problem for me.
    if not np.issubdtype(PETSc.ScalarType, np.complexfloating):  # type: ignore
        print("We need PETSc to be using complex scalars.")
        assert False
    else:
        assert True
        
def test_FF_SphereScatt(get_test_data): ## run a spherica domain and object, test the far-field scattering from a sphere vs Mie theoretical result
    comm = MPI.COMM_WORLD
    MPInum = comm.size
    verbosity = 1
    refMesh = meshMaker.MeshData(comm, reference = True, viewGMSH = False, verbosity = verbosity, N_antennas=1, h=1/15, domain_geom='sphere', FF_surface = True) ## this will have around 190000 elements
    prob = scatteringProblem.Scatt3DProblem(comm, refMesh, verbosity = verbosity, MPInum = MPInum)
    
if __name__ == '__main__':
    args_str = "-v -s"
    args = args_str.split(" ")
    retcode = pytest.main(args)
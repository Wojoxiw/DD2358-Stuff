# encoding: utf-8
### Modification of scatt2d to handle 3d geometry
# Stripped down and rewritten for DD2358 course
#
# Adapted from 2D code started by Daniel Sjoberg, 2024-12-13
# Alexandros Pallaris, after that

import os
os.environ["OMP_NUM_THREADS"] = "1" # seemingly needed for MPI speedup
from mpi4py import MPI
import numpy as np
import dolfinx, ufl, basix
import dolfinx.fem.petsc
import gmsh
from matplotlib import pyplot as plt
import pyvista as pv
import pyvistaqt as pvqt
import functools

import sys, petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
import psutil
import scipy

from memory_profiler import memory_usage
from timeit import default_timer as timer
import sys
import meshMaker
import scatteringProblem
import memTimeEstimation



##MAIN STUFF
if __name__ == '__main__':
    # MPI settings
    comm = MPI.COMM_WORLD
    model_rank = 0
    verbosity = 1
    MPInum = comm.size
    
    runName = 'test' # testing
    folder = 'data3D/'
    
    print(f"{comm.rank=} {comm.size=}, {MPI.COMM_SELF.rank=} {MPI.COMM_SELF.size=}")
    if(comm.rank == model_rank):
        print('Expected number of MPI processes:', MPInum)
        print('Scatt3D start:')
    sys.stdout.flush()
    
    
    def profilingMemsTimes(): ## as used to make plots for the report
        
        ## when using MPI for speedup:
        prevRuns = memTimeEstimation.runTimesMems(folder, comm, filename = 'initProfilingMPI.npz')
        
        numRuns = 10 ## run these 10 times to find averages/stds
        hs = [1/10, 1/11, 1/12, 1/13, 1/14, 1/15, 1/16, 1/17, 1/18, 1/19, 1/20] ## run it for different mesh sizes
        for i in range(numRuns):
            for h in hs:
                refMesh = meshMaker.MeshData(comm, folder+runName+'mesh.msh', reference = True, viewGMSH = False, verbosity = verbosity, h=h)
                prob = scatteringProblem.Scatt3DProblem(comm, refMesh, verbosity = verbosity, MPInum = MPInum)
                prevRuns.memTimeAppend(prob)
            
    def testRun():
        prevRuns = memTimeEstimation.runTimesMems(folder, comm)
        #prevRuns.makePlots()
        refMesh = meshMaker.MeshData(comm, folder+runName+'mesh.msh', reference = True, viewGMSH = False, verbosity = verbosity)
        prob = scatteringProblem.Scatt3DProblem(comm, refMesh, verbosity = verbosity, MPInum = MPInum)
        prob.saveEFieldsForAnim()
    
        prevRuns.memTimeAppend(prob)
    
    
    #testRun()
    
    profilingMemsTimes()
    
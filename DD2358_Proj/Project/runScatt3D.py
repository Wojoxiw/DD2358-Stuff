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
from scripts import memTimeEstimation



##MAIN STUFF
if __name__ == '__main__':
    # MPI settings
    comm = MPI.COMM_WORLD
    model_rank = 0
    args = sys.argv[1:] ## can take the number of MPI processes as an argument. If not given, just say 0. Current not used anywhere.
    if(len(args) == 0):
        MPInum = 1
    else:
        MPInum = int(args[0]) ## This is apparently a sequence that needs to be converted to int so I can multiply it...
    
    runName = 'test' # testing
    folder = 'data3D/'
    
    print(f"{MPI.COMM_WORLD.rank=} {MPI.COMM_WORLD.size=}, {MPI.COMM_SELF.rank=} {MPI.COMM_SELF.size=}")
      
    
    if(comm.rank == model_rank):
        print('Expected number of MPI processes:', MPInum)
        print('Scatt3D start:')
    sys.stdout.flush()
        
      
    
    #memTimeEstimation(printPlots = True)
    
    
    def testRun(runName, viewGMSH = False, verbosity = 1, Nf=1):
        refMesh = meshMaker.MeshData(comm, folder+runName+'mesh.msh', reference = True, viewGMSH = viewGMSH, verbosity = verbosity)
        
        estmem, esttime = memTimeEstimation(refMesh.ncells, Nf)
        print(f'Estimated memory requirement for size {refMesh.ncells:.3e}: {estmem:.2f} GB')
        print(f'Estimated computation time for size {refMesh.ncells:.3e}, Nf = {Nf}: {esttime/3600:.2f} hours')
        prob = scatteringProblem.Scatt3DProblem(comm, refMesh, verbosity = verbosity)
        prob.saveEFieldsForAnim()
    
    b = memory_usage((testRun, (runName,), {'viewGMSH' : False}), max_usage = True)
    print(b)
    mem_usage = MPInum * b
    
    print('Max. memory:',mem_usage/1000,'GiB'+f"{MPI.COMM_WORLD.rank=} {MPI.COMM_WORLD.size=}, {MPI.COMM_SELF.rank=} {MPI.COMM_SELF.size=}")
    #if(comm.rank == model_rank):
        #memTimeAppend(size, Nf, mem_usage/1000, totT, reference=False) ## '0' memory cost to ignore this one (or later fill in manually) - not sure how to easily estimate this without slowing the code
        
    
    
    #===========================================================================
    # ## Possible to mpirun from script, but difficult to then cancel processes - seems unsuitable
    # N = 3
    # child=MPI.COMM_SELF.Spawn(sys.executable,args=['Scatt3D.py'],maxprocs=N) ## will run it like mpirun -n N Scatt3D.py, I think
    #===========================================================================
    
    #mem_usage = memory_usage(subprocess.run(command, stdout=subprocess.PIPE), max_usage=True)
    #print(mem_usage)
    #print('Max. memory:',mem_usage/1000,'GiB')
    #memTimeAppend(size, Nf, mem_usage/1000, totT, reference=False) ## '0' memory cost to ignore this one (or later fill in manually) - not sure how to easily estimate this without slowing the code
      
    
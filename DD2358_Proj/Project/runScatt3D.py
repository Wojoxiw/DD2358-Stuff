# encoding: utf-8
### Modification of scatt2d to handle 3d geometry
# Stripped down and rewritten for DD2358 course
#
# Adapted from 2D code started by Daniel Sjoberg, 2024-12-13
# Alexandros Pallaris, after that

import os
#os.environ["OMP_NUM_THREADS"] = "1" # perhaps needed for MPI speedup if using many processes?
#os.environ['MKL_NUM_THREADS'] = '1' # maybe also relevent
#os.environ['NUMEXPR_NUM_THREADS'] = '1' # maybe also relevent
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
import time
import sys
import meshMaker
import scatteringProblem
import memTimeEstimation

#===============================================================================
# ##line profiling
# import line_profiler
# import atexit
# profile = line_profiler.LineProfiler()
# atexit.register(profile.print_stats)
#===============================================================================

#===============================================================================
# ##memory profiling
# from memory_profiler import profile
#===============================================================================


##MAIN STUFF
if __name__ == '__main__':
    # MPI settings
    comm = MPI.COMM_WORLD
    model_rank = 0
    verbosity = 1
    MPInum = comm.size
    
    runName = 'testRun' # testing
    folder = 'data3D/'
    
    print(f"{comm.rank=} {comm.size=}, {MPI.COMM_SELF.rank=} {MPI.COMM_SELF.size=}, {MPI.Get_processor_name()=}")
    if(comm.rank == model_rank):
        print('Expected number of MPI processes:', MPInum)
        print('Scatt3D start:')
        #print(psutil.virtual_memory().total / 1024**3, 'GB of available RAM')
    sys.stdout.flush()
    
    def profilingMemsTimes(): ## as used to make plots for the report
        prevRuns = memTimeEstimation.runTimesMems(folder, comm, filename = '8nodes24MPI1threads2b.npz') ## make sure to change to filename so it doesn't get overwritten - the data is stored here
        numRuns = 1 ## run these 10 times to find averages/stds
        hs = [1/10, 1/11, 1/12, 1/13, 1/14, 1/15, 1/16, 1/17, 1/18, 1/19, 1/20] ## run it for different mesh sizes
        for i in range(numRuns):
            if(comm.rank == model_rank):
                print('############')
                print(f'  RUN {i+1}/{numRuns} ')
                print('############')
            for h in hs:
                refMesh = meshMaker.MeshData(comm, folder+runName+'mesh.msh', reference = True, viewGMSH = False, verbosity = verbosity, h=h)
                prob = scatteringProblem.Scatt3DProblem(comm, refMesh, verbosity = verbosity, MPInum = MPInum)
                prevRuns.memTimeAppend(prob, '8nodes24MPI1threads2b')
    
    def actualProfilerRunning(): # Here I call more things explicitly in order to more easily profile the code in separate methods (profiling activated in the methods themselves also).
        refMesh = meshMaker.MeshData(comm, folder+runName+'mesh.msh', reference = True, viewGMSH = False, verbosity = verbosity, h=1/10) ## this will have around 190000 elements
        prob = scatteringProblem.Scatt3DProblem(comm, refMesh, verbosity = verbosity, MPInum = MPInum)
            
    def testRun():
        prevRuns = memTimeEstimation.runTimesMems(folder, comm)
        #prevRuns.makePlots()
        #prevRuns.makePlots()
        refMesh = meshMaker.MeshData(comm, folder+runName+'mesh.msh', reference = True, viewGMSH = False, verbosity = verbosity, h=1/2, object_geom='None', N_antennas=0)
        prevRuns.memTimeEstimation(refMesh.ncells, doPrint=True)
        #refMesh.plotMeshPartition()
        prob = scatteringProblem.Scatt3DProblem(comm, refMesh, verbosity = verbosity, MPInum = MPInum, name = runName, excitation = 'planewave')
        #prob.saveEFieldsForAnim()
        #prevRuns.memTimeAppend(prob)
        
    def testFarField(): ## run a spherica domain and object, test the far-field scattering for an incident plane-wave from a sphere vs Mie theoretical result
        prevRuns = memTimeEstimation.runTimesMems(folder, comm)
        refMesh = meshMaker.MeshData(comm, reference = True, viewGMSH = False, verbosity = verbosity, N_antennas=0, object_radius = 0.34, domain_radius=1.0, h=1/20, domain_geom='sphere', FF_surface = True)
        prevRuns.memTimeEstimation(refMesh.ncells, doPrint=True)
        freqs = np.linspace(10e9, 12e9, 1)
        prob = scatteringProblem.Scatt3DProblem(comm, refMesh, verbosity = verbosity, MPInum = MPInum, excitation = 'planewave', freqs = freqs, material_epsr=6)
        prob.saveDofsView(prob.refMeshdata)
        prob.saveEFieldsForAnim()
        nvals = int(360/10)
        angles = np.zeros((nvals, 2))
        angles[:, 0] = 90
        angles[:, 1] = np.linspace(0, 360, nvals)
        prob.calcFarField(reference=True, angles = angles, compareToMie = True)
    
    
    #testRun()
    #profilingMemsTimes()
    #actualProfilerRunning()
    testFarField()
    
    otherprevs = []
    #prevRuns = memTimeEstimation.runTimesMems(folder, comm, otherPrevs = otherprevs)
    #prevRuns.makePlotsSTD()
    
    if(comm.rank == model_rank):
        print('runScatt3D complete, exiting...')
        sys.stdout.flush()
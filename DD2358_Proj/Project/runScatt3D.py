# encoding: utf-8
### Modification of scatt2d to handle 3d geometry
# Stripped down and rewritten for DD2358 course
#
# Adapted from 2D code started by Daniel Sjoberg, (https://github.com/dsjoberg-git/rotsymsca, https://github.com/dsjoberg-git/ekas3d) approx. 2024-12-13 
# Alexandros Pallaris, after that

import os
import numpy as np
import dolfinx, ufl, basix
import dolfinx.fem.petsc
#os.environ["OMP_NUM_THREADS"] = "1" # perhaps needed for MPI speedup if using many processes? These do not seem to matter on the cluster
#os.environ['MKL_NUM_THREADS'] = '1' # maybe also relevent
#os.environ['NUMEXPR_NUM_THREADS'] = '1' # maybe also relevent
from mpi4py import MPI
import gmsh
from matplotlib import pyplot as plt
import matplotlib.lines as mlines
import functools
from scipy.constants import c as c0, mu_0 as mu0, epsilon_0 as eps0, pi

import sys, petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
import scipy

import psutil
from memory_profiler import memory_usage
from timeit import default_timer as timer
import time
import sys
import meshMaker
import scatteringProblem
import memTimeEstimation
import postProcessing

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
    verbosity = 2 ## 3 will print everything. 2, most things. 1, just the main process stuff.
    MPInum = comm.size
    
    if(len(sys.argv) == 1): ## assume computing on local computer, not cluster. In jobscript for cluster, give a dummy argument
        filename = 'localCompTimesMems.npz'
    else:
        filename = 'prevRuns.npz'
    
    runName = 'testRun' # testing
    folder = 'data3D/'
    if(verbosity>2):
        print(f"{comm.rank=} {comm.size=}, {MPI.COMM_SELF.rank=} {MPI.COMM_SELF.size=}, {MPI.Get_processor_name()=}")
    if(comm.rank == model_rank):
        print('Expected number of MPI processes:', MPInum)
        print('Scatt3D start:')
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
            
    def testRun(h = 1/2): ## A quick test run to check it works. Default settings make this run in a second
        prevRuns = memTimeEstimation.runTimesMems(folder, comm, filename = filename)
        refMesh = meshMaker.MeshData(comm, folder+runName+'mesh.msh', reference = True, viewGMSH = False, verbosity = verbosity, h=h, object_geom='None', N_antennas=0)
        prevRuns.memTimeEstimation(refMesh.ncells, doPrint=True)
        #refMesh.plotMeshPartition()
        prob = scatteringProblem.Scatt3DProblem(comm, refMesh, verbosity = verbosity, MPInum = MPInum, name = runName, excitation = 'planewave')
        #prob.saveEFieldsForAnim()
        prevRuns.memTimeAppend(prob)
        
    def testRun2(h = 1/15): ## Testing toward postprocessing stuff
        prevRuns = memTimeEstimation.runTimesMems(folder, comm, filename = filename)
        refMesh = meshMaker.MeshData(comm, folder+runName+'mesh.msh', reference = True, viewGMSH = False, verbosity = verbosity, h=h, N_antennas=5)
        dutMesh = meshMaker.MeshData(comm, folder+runName+'mesh.msh', reference = False, viewGMSH = False, verbosity = verbosity, h=h, N_antennas=5)
        prevRuns.memTimeEstimation(refMesh.ncells, doPrint=True)
        #refMesh.plotMeshPartition()
        prob = scatteringProblem.Scatt3DProblem(comm, refMesh, DUTMeshdata=dutMesh, computeBoth=True, verbosity = verbosity, MPInum = MPInum, name = runName, Nf = 6)
        prob.saveEFieldsForAnim()
        prevRuns.memTimeAppend(prob)
        postProcessing.testSVD(prob.dataFolder+prob.name)
        
    def testFarField(h = 1/12): ## run a spherical domain and object, test the far-field scattering for an incident plane-wave from a sphere vs Mie theoretical result
        prevRuns = memTimeEstimation.runTimesMems(folder, comm, filename = filename)
        refMesh = meshMaker.MeshData(comm, reference = True, viewGMSH = False, verbosity = verbosity, N_antennas=0, object_radius = 0.3118824290102722, domain_radius=1.3, PML_thickness=0.35, h=h, domain_geom='sphere', FF_surface = True)
        prevRuns.memTimeEstimation(refMesh.ncells, doPrint=True)
        freqs = np.linspace(10e9, 12e9, 1)
        prob = scatteringProblem.Scatt3DProblem(comm, refMesh, verbosity = verbosity, name=runName, MPInum = MPInum, makeOptVects=False, excitation = 'planewave', freqs = freqs, material_epsr=2, fem_degree=3)
        #prob.saveDofsView(prob.refMeshdata)
        #prob.saveEFieldsForAnim()
        nvals = int(360/4)
        angles = np.zeros((nvals, 2))
        angles[:, 0] = 90
        angles[:, 1] = np.linspace(0, 360, nvals)
        prob.calcFarField(reference=True, angles = angles, compareToMie = True, showPlots=True)
        prevRuns.memTimeAppend(prob)
        
        
    def convergenceTestPlots(): ## Runs with reducing mesh size, for convergence plots. Uses the far-field surface test case
        ks = np.arange(3, 18, 2)
        vals = [] ## vals returned from the calculations
        for k in ks: ## 1/h
            h = 1/k
            refMesh = meshMaker.MeshData(comm, reference = True, viewGMSH = False, verbosity = verbosity, N_antennas=0, object_radius = 0.35, domain_radius=1.3, h=h, domain_geom='sphere', FF_surface = True)
            prob = scatteringProblem.Scatt3DProblem(comm, refMesh, verbosity = verbosity, name=runName, MPInum = MPInum, makeOptVects=False, excitation = 'planewave', material_epsr=3.5, Nf=1)
            vals.append(prob.calcFarField(reference=True, compareToMie = False, showPlots=False, returnConvergenceVals=True)) ## each return is [FF surface area, khat integral, forward scattering mag.**2 at f[0], backward scattering mag.**2 at f[0]]
        vals = np.array(vals, dtype=np.float64)
        
        fig1 = plt.figure()
        ax1 = plt.subplot(1, 1, 1)
        ax1.grid(True)
        ax1.set_title('Convergence of Different Values')
        ax1.set_xlabel(r'Inverse mesh size ($\lambda / h$)')
        
        import miepython ## this shouldn't need to be installed on the cluster (I can't figure out how to) so only import it here
        import miepython.field
        
        m = np.sqrt(prob.material_epsr)
        lamb = c0/prob.fvec[0] ## should be the only frequency
        x = 2*pi*prob.refMeshdata.object_radius/lamb
        mieforward = miepython.i_par(m, x, np.cos(0), norm='qsca')*pi*prob.refMeshdata.object_radius**2
        miebackward = miepython.i_par(m, x, np.cos(pi), norm='qsca')*pi*prob.refMeshdata.object_radius**2
        print('fw', mieforward, vals[:, 2])
        print('bw', miebackward, vals[:, 3])
        real_area = 4*pi*prob.refMeshdata.FF_surface_radius**2
        ax1.plot(ks, np.abs((real_area-vals[:, 0])/real_area), marker='o', linestyle='--', label = r'area - rel. error')
        ax1.plot(ks, np.abs(vals[:, 1]), marker='o', linestyle='--', label = r'khat integral - abs. error')
        ax1.plot(ks, np.abs((mieforward-vals[:, 2])/mieforward), marker='o', linestyle='--', label = r'forward scat. - rel. error')
        ax1.plot(ks, np.abs((miebackward-vals[:, 2])/miebackward), marker='o', linestyle='--', label = r'backward scat. - rel. error')
        
        #=======================================================================
        # first_legend = ax1.legend(framealpha=0.5, loc = 'lower left') ## extra legend stuff in case I want to plot error for many angles
        # ##second legend to distinguish between dashed and regular lines (phi- and theta- pols)
        # handleds = []
        # line_dashed = mlines.Line2D([], [], color='black', linestyle='--', linewidth=1.5, label=r'max. rel. error') ##fake lines to create second legend elements
        # handleds.append(line_dashed)
        # line_solid = mlines.Line2D([], [], color='black', linestyle='solid', linewidth=1.5, label=r'rms rel. error') ##fake lines to create second legend elements
        # handleds.append(line_solid)
        # second_legend = ax1.legend(handles=handleds, loc='upper right', framealpha=0.5) #best upper-right
        # ax1.add_artist(first_legend)
        #=======================================================================
        ax1.set_yscale('log')
        ax1.legend()
        fig1.tight_layout()
        plt.savefig(prob.dataFolder+prob.name+'convergences.png')
        plt.show()
        
    #testRun(h=1/20)
    #profilingMemsTimes()
    #actualProfilerRunning()
    #testRun2(h=1/10)
    runName = 'test-fem3h2-10'
    testFarField(h=1/10)
    #convergenceTestPlots()
    
    #===========================================================================
    # for k in np.arange(10, 35, 4):
    #     runName = 'test-fem3h'+str(k)
    #     testFarField(h=1/k)
    #===========================================================================
    
    #===========================================================================
    # for k in range(15, 40, 2):
    #     runName = 'testRunbiggerdomainfaraway(10hawayFF)FF'+str(k)
    #     testFarField(h=1/k)
    #===========================================================================
    
    otherprevs = [] ## if adding other files here, specify here (i.e. prevRuns.npz.old)
    #prevRuns = memTimeEstimation.runTimesMems(folder, comm, otherPrevs = otherprevs, filename = filename)
    #prevRuns.makePlots()
    #prevRuns.makePlotsSTD()
    
    if(comm.rank == model_rank):
        print('runScatt3D complete, exiting...')
        sys.stdout.flush()
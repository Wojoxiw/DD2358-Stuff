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
#os.environ["OMP_NUM_THREADS"] = "1" # perhaps needed for MPI speedup if using many processes locally? These do not seem to matter on the cluster
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
    
    t1 = timer()
    
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
        
    def testFarField(h = 1/12): ## run a spherical domain and object, test the far-field scattering for an incident plane-wave from a sphere vs Mie theoretical result.
        prevRuns = memTimeEstimation.runTimesMems(folder, comm, filename = filename)
        refMesh = meshMaker.MeshData(comm, reference = True, viewGMSH = False, verbosity = verbosity, N_antennas=0, object_radius = 1/3, domain_radius=0.9, PML_thickness=0.5, h=h, domain_geom='sphere', FF_surface = True)
        prevRuns.memTimeEstimation(refMesh.ncells, doPrint=True)
        freqs = np.linspace(10e9, 12e9, 1)
        prob = scatteringProblem.Scatt3DProblem(comm, refMesh, verbosity = verbosity, name=runName, MPInum = MPInum, makeOptVects=False, excitation = 'planewave', freqs = freqs, material_epsr=2.0)
        #prob.saveDofsView(prob.refMeshdata)
        #prob.saveEFieldsForAnim()
        prob.calcFarField(reference=True, compareToMie = True, showPlots=True)
        prevRuns.memTimeAppend(prob)
        
        
    def convergenceTestPlots(convergence = 'meshsize'): ## Runs with reducing mesh size, for convergence plots. Uses the far-field surface test case. If showPlots, show them - otherwise just save them
        if(convergence == 'meshsize'):
            ks = np.linspace(4, 40, 20)
        elif(convergence == 'pmlR0'):
            ks = np.linspace(2, 15, 20)
            ks = 10**(-ks)
            
        areaVals = [] ## vals returned from the calculations
        FFrmsRelErrs = np.zeros(len(ks)) ## for the farfields
        FFrmsAbsErrs = np.zeros(len(ks))
        FFmaxRelErrs = np.zeros(len(ks))
        khatRmsErrs = np.zeros(len(ks))
        khatMaxErrs = np.zeros(len(ks))
        meshOptions = dict()
        probOptions = dict()
        for i in range(len(ks)):
            if(convergence == 'meshsize'):
                meshOptions = dict(h = 1/ks[i])
            elif(convergence == 'pmlR0'):
                probOptions = dict(PML_R0 = ks[i])
                
            refMesh = meshMaker.MeshData(comm, reference = True, viewGMSH = False, verbosity = verbosity, N_antennas=0, object_radius = 0.35, PML_thickness=1.0, domain_radius=0.9, domain_geom='sphere', FF_surface = True, **meshOptions)
            prob = scatteringProblem.Scatt3DProblem(comm, refMesh, verbosity = verbosity, name=runName, MPInum = MPInum, makeOptVects=False, excitation = 'planewave', material_epsr=3.5, Nf=1, **probOptions)
            newval, khats, farfields, mies = prob.calcFarField(reference=True, compareToMie = False, showPlots=True, returnConvergenceVals=True) ## each return is FF surface area, khat integral at each angle, farfields+mies at each angle
            if(comm.rank == model_rank): ## only needed for main process
                areaVals.append(newval)
                khatRmsErrs[i] = np.sqrt(np.sum(khats**2)/np.size(khats))
                khatMaxErrs[i] = np.max(khats)
                intenss = np.abs(farfields[0,:,0])**2 + np.abs(farfields[0,:,1])**2
                FFrelativeErrors = np.abs( (intenss - mies) / mies )
                FFrmsRelErrs[i] = np.sqrt(np.sum(FFrelativeErrors**2)/np.size(FFrelativeErrors))
                FFrmsAbsErrs[i] = np.sqrt(np.sum(np.abs(intenss - mies)**2)/np.size(intenss))
                FFmaxRelErrs[i] = np.max(FFrelativeErrors)
        if(comm.rank == model_rank): ## only needed for main process
            areaVals = np.array(areaVals)
            
            fig1 = plt.figure()
            ax1 = plt.subplot(1, 1, 1)
            ax1.grid(True)
            ax1.set_title('Convergence of Different Values')
            if(convergence == 'meshsize'):
                ax1.set_xlabel(r'Inverse mesh size ($\lambda / h$)')
            elif(convergence == 'pmlR0'):
                ax1.set_xscale('log')
            
            real_area = 4*pi*prob.refMeshdata.FF_surface_radius**2
            ax1.plot(ks, np.abs((real_area-areaVals)/real_area), marker='o', linestyle='--', label = r'area - rel. error')
            ax1.plot(ks, khatMaxErrs, marker='o', linestyle='--', label = r'khat integral - max. abs. error')
            ax1.plot(ks, khatRmsErrs, marker='o', linestyle='--', label = r'khat integral - RMS error')
            ax1.plot(ks, FFrmsRelErrs, marker='o', linestyle='--', label = r'Farfield cuts RMS rel. error')
            ax1.plot(ks, FFrmsAbsErrs, marker='o', linestyle='--', label = r'Farfield cuts RMS abs. error')
            ax1.plot(ks, FFmaxRelErrs, marker='o', linestyle='--', label = r'Farfield cuts max. rel. error')
            
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
            plt.savefig(prob.dataFolder+prob.name+convergence+'h15PML1.0lambdathickconvergence.png')
            #plt.show()
        
    #testRun(h=1/20)
    #profilingMemsTimes()
    #actualProfilerRunning()
    #testRun2(h=1/10)
    #testFarField(h=1/40)
    #convergenceTestPlots('pmlR0')
    convergenceTestPlots('meshsize')
    
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
        print(f'runScatt3D complete in {timer()-t1:.2f} s, exiting...')
        sys.stdout.flush()
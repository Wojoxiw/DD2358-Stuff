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
        print(f'runScatt3D starting with {MPInum} MPI process(es):')
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
        prevRuns.memTimeEstimation(refMesh.ncells, doPrint=True, MPInum = comm.size)
        #refMesh.plotMeshPartition()
        prob = scatteringProblem.Scatt3DProblem(comm, refMesh, verbosity = verbosity, MPInum = MPInum, name = runName, excitation = 'planewave')
        #prob.saveEFieldsForAnim()
        prevRuns.memTimeAppend(prob)
        
    def testRun2(h = 1/15): ## Testing toward postprocessing stuff
        prevRuns = memTimeEstimation.runTimesMems(folder, comm, filename = filename)
        refMesh = meshMaker.MeshData(comm, folder+runName+'mesh.msh', reference = True, viewGMSH = False, verbosity = verbosity, h=h, N_antennas=5)
        dutMesh = meshMaker.MeshData(comm, folder+runName+'mesh.msh', reference = False, viewGMSH = False, verbosity = verbosity, h=h, N_antennas=5)
        prevRuns.memTimeEstimation(refMesh.ncells, doPrint=True, MPInum = comm.size)
        #refMesh.plotMeshPartition()
        prob = scatteringProblem.Scatt3DProblem(comm, refMesh, DUTMeshdata=dutMesh, computeBoth=True, verbosity = verbosity, MPInum = MPInum, name = runName, Nf = 6)
        prob.saveEFieldsForAnim()
        prevRuns.memTimeAppend(prob)
        postProcessing.testSVD(prob.dataFolder+prob.name)
        
    def testSphereScattering(h = 1/12, fem_degree=1, showPlots=False): ## run a spherical domain and object, test the far-field scattering for an incident plane-wave from a sphere vs Mie theoretical result.
        prevRuns = memTimeEstimation.runTimesMems(folder, comm, filename = filename)
        refMesh = meshMaker.MeshData(comm, reference = True, viewGMSH = False, verbosity = verbosity, N_antennas=0, object_radius = .33, domain_radius=.9, PML_thickness=0.5, h=h, domain_geom='sphere', order=1, object_geom='sphere', FF_surface = True)
        prevRuns.memTimeEstimation(refMesh.ncells, doPrint=True, MPInum = comm.size)
        freqs = np.linspace(10e9, 12e9, 1)
        prob = scatteringProblem.Scatt3DProblem(comm, refMesh, verbosity=verbosity, name=runName, MPInum=MPInum, makeOptVects=False, excitation='planewave', freqs = freqs, material_epsr=2.0, fem_degree=fem_degree)
        #prob.saveDofsView(prob.refMeshdata)
        #prob.saveEFieldsForAnim()
        if(showPlots):
            prob.calcNearField(direction='side')
        prob.calcFarField(reference=True, compareToMie = True, showPlots=showPlots, returnConvergenceVals=False)
        #prevRuns.memTimeAppend(prob)
 
    def convergenceTestPlots(convergence = 'meshsize'): ## Runs with reducing mesh size, for convergence plots. Uses the far-field surface test case. If showPlots, show them - otherwise just save them
        if(convergence == 'meshsize'):
            ks = np.linspace(4, 16, 6)
        elif(convergence == 'pmlR0'): ## result of this is that the value must be below 1e-2, from there further reduction matches the forward-scattering better, the back-scattering less
            ks = np.linspace(0, 25, 10)
            ks = 10**(-ks)
        elif(convergence == 'dxquaddeg'): ## Result of this showed a large increase in time near the end, and an accuracy improvement for increasing from 2 to 3. Not sending any value causes a huge memory cost/error (process gets killed).
            ks = np.arange(1, 20)
            
        areaVals = [] ## vals returned from the calculations
        FFrmsRelErrs = np.zeros(len(ks)) ## for the farfields
        FFrmsveryRelErrs = np.zeros(len(ks))
        FFmaxRelErrs = np.zeros(len(ks))
        FFforwardrelErr = np.zeros(len(ks))
        khatRmsErrs = np.zeros(len(ks))
        khatMaxErrs = np.zeros(len(ks))
        meshOptions = dict()
        probOptions = dict()
        for i in range(len(ks)):
            if(convergence == 'meshsize'):
                meshOptions = dict(h = 1/ks[i])
            elif(convergence == 'pmlR0'):
                probOptions = dict(PML_R0 = ks[i])
            elif(convergence == 'dxquaddeg'):
                probOptions = dict(quaddeg = ks[i])
                
            refMesh = meshMaker.MeshData(comm, reference = True, viewGMSH = False, verbosity = verbosity, N_antennas=0, object_radius = .33, PML_thickness=0.5, domain_radius=0.9, domain_geom='sphere', order=3, FF_surface = True, **meshOptions)
            prob = scatteringProblem.Scatt3DProblem(comm, refMesh, verbosity = verbosity, name=runName, MPInum = MPInum, makeOptVects=False, excitation = 'planewave', material_epsr=2.0, Nf=1, fem_degree=3, **probOptions)
            newval, khats, farfields, mies = prob.calcFarField(reference=True, compareToMie = False, showPlots=False, returnConvergenceVals=True) ## each return is FF surface area, khat integral at each angle, farfields+mies at each angle
            if(comm.rank == model_rank): ## only needed for main process
                areaVals.append(newval)
                khatRmsErrs[i] = np.sqrt(np.sum(khats**2)/np.size(khats))
                khatMaxErrs[i] = np.max(khats)
                intenss = np.abs(farfields[0,:,0])**2 + np.abs(farfields[0,:,1])**2
                FFrelativeErrors = np.abs( (intenss - mies) ) / np.abs( mies )
                FFrmsRelErrs[i] = np.sqrt(np.sum(FFrelativeErrors**2)/np.size(FFrelativeErrors))
                FFvrelativeErrors = np.abs( (intenss - mies) ) / np.max(np.abs(mies)) ## relative to the max. mie intensity, to make it even more relative
                FFrmsveryRelErrs[i] = np.sqrt(np.sum(FFvrelativeErrors**2)/np.size(FFvrelativeErrors)) ## absolute error, scaled by the max. mie value
                FFmaxRelErrs[i] = np.max(FFrelativeErrors)
                FFforwardrelErr[i] = np.abs( (intenss[int(360/4)] - mies[int(360/4)])) / np.abs(mies[int(360/4)])
                if(verbosity>1):
                    print(f'Run {i+1}/{len(ks)} completed')
        if(comm.rank == model_rank): ## only needed for main process
            areaVals = np.array(areaVals)
            
            fig1 = plt.figure()
            ax1 = plt.subplot(1, 1, 1)
            ax1.grid(True)
            ax1.set_title('Convergence of Different Values')
            if(convergence == 'meshsize'):
                ax1.set_xlabel(r'Inverse mesh size ($\lambda / h$)')
            elif(convergence == 'pmlR0'):
                ax1.set_xlabel(r'R0')
                ax1.set_xscale('log')
            elif(convergence == 'dxquaddeg'):
                ax1.set_xlabel(r'dx Quadrature Degree')
            
            real_area = 4*pi*prob.refMeshdata.FF_surface_radius**2
            ax1.plot(ks, np.abs((real_area-areaVals)/real_area), marker='o', linestyle='--', label = r'area - rel. error')
            ax1.plot(ks, khatMaxErrs, marker='o', linestyle='--', label = r'khat integral - max. abs. error')
            ax1.plot(ks, khatRmsErrs, marker='o', linestyle='--', label = r'khat integral - RMS error')
            ax1.plot(ks, FFrmsRelErrs, marker='o', linestyle='--', label = r'Farfield cuts RMS rel. error')
            ax1.plot(ks, FFrmsveryRelErrs, marker='o', linestyle='--', label = r'Farfield cuts RMS. veryrel. error')
            ax1.plot(ks, FFforwardrelErr, marker='o', linestyle='--', label = r'Farfield forward rel. error')
            
            ax1.set_yscale('log')
            ax1.legend()
            fig1.tight_layout()
            plt.savefig(prob.dataFolder+prob.name+convergence+'order3fem3convergence.png')
            plt.show()
            
    def testSolverSettings(h = 1/12): # Varies settings in the ksp solver/preconditioner, plots the time and iterations a computation takes. Uses the sphere-scattering test case
        refMesh = meshMaker.MeshData(comm, reference = True, viewGMSH = False, verbosity = verbosity, N_antennas=0, object_radius = .33, domain_radius=.9, PML_thickness=0.5, h=h, domain_geom='sphere', object_geom='sphere', FF_surface = True)
        omegas = np.arange(0, 14) ## the setting being varied
        num = np.size(omegas)
        ts = np.zeros(num)
        its = np.zeros(num)
        norms = np.zeros(num)
        for i in range(len(omegas)):
            sets = {"ksp_lgmres_augment" : omegas[i]}
            prob = scatteringProblem.Scatt3DProblem(comm, refMesh, verbosity=verbosity, name=runName, MPInum=MPInum, makeOptVects=False, excitation='planewave', material_epsr=2.0, fem_degree=1, solver_settings=sets)
            ts[i] = prob.calcTime
            its[i] = prob.solver_its
            norms[i] = prob.solver_norm
        fig, ax1 = plt.subplots()
        fig.subplots_adjust(right=0.75)
        ax2 = ax1.twinx()
        ax3 = ax1.twinx()
        ax3.spines.right.set_position(("axes", 1.2))
        ax3.set_yscale('log')
        ax1.grid()
        
        l1, = ax1.plot(omegas, its, label = 'Iterations', linewidth = 2, color='tab:red')
        l2, = ax2.plot(omegas, ts, label = 'Time [s]', linewidth = 2, color='tab:blue')
        l3, = ax3.plot(omegas, norms, label = 'norms', linewidth = 2, color = 'orange')
        
        plt.title(f'Solver Time by Setting (fem_degree=1, h={h:.2f})')
        ax1.set_xlabel(r'Setting (ksp_lgmres_augment)')
        ax1.set_ylabel('#')
        ax2.set_ylabel('Time [s]')
        ax3.set_ylabel('log10(Norms)')
        
        ax1.yaxis.label.set_color(l1.get_color())
        ax2.yaxis.label.set_color(l2.get_color())
        ax3.yaxis.label.set_color(l3.get_color())
        tkw = dict(size=4, width=1.5)
        ax1.tick_params(axis='y', colors=l1.get_color(), **tkw)
        ax2.tick_params(axis='y', colors=l2.get_color(), **tkw)
        ax3.tick_params(axis='y', colors=l3.get_color(), **tkw)
        ax1.tick_params(axis='x', **tkw)
        ax1.legend(handles=[l1, l2, l3])
        
        fig.tight_layout()
        plt.savefig(prob.dataFolder+prob.name+'ksp_lgmres_augment_solversettingsplot.png')
        plt.show()
        
    #testRun(h=1/20)
    #profilingMemsTimes()
    #actualProfilerRunning()
    #testRun2(h=1/10)
    #testSphereScattering(h=1/5, fem_degree=1, showPlots=False)
    #convergenceTestPlots('pmlR0')
    convergenceTestPlots('meshsize')
    #convergenceTestPlots('dxquaddeg')
    #testSolverSettings(h=1/15)
    
    #===========================================================================
    # for k in np.arange(10, 35, 4):
    #     runName = 'test-fem3h'+str(k)
    #     testSphereScattering(h=1/k)
    #===========================================================================
    
    #===========================================================================
    # for k in range(15, 40, 2):
    #     runName = 'testRunbiggerdomainfaraway(10hawayFF)FF'+str(k)
    #     testSphereScattering(h=1/k)
    #===========================================================================
    
    #filename = 'prevRuns.npz'
    otherprevs = [] ## if adding other files here, specify here (i.e. prevRuns.npz.old)
    #prevRuns = memTimeEstimation.runTimesMems(folder, comm, otherPrevs = otherprevs, filename = filename)
    #prevRuns.makePlots(MPInum = comm.size)
    #prevRuns.makePlotsSTD()
    
    if(comm.rank == model_rank):
        print(f'runScatt3D complete in {timer()-t1:.2f} s, exiting...')
        sys.stdout.flush()